#!/usr/bin/env python3
"""Translate a line-oriented Kazakh corpus with one NLLB process per GPU."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch


def rank_info() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("LOCAL_RANK", "0")),
    )


def iter_rank_sources(path: Path, rank: int, world_size: int, start: int = 0, limit: int = 0):
    end = start + limit if limit else None
    with path.open(encoding="utf-8") as file:
        for index, line in enumerate(file):
            if index < start:
                continue
            if end is not None and index >= end:
                break
            source = line.rstrip("\r\n").partition("\t")[0].strip()
            if source and index % world_size == rank:
                yield index, source


def truncate_partial_jsonl(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as file:
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(-1, os.SEEK_END)
        if file.read(1) == b"\n":
            return
        position = size
        while position:
            position = max(0, position - 1024 * 1024)
            file.seek(position)
            chunk = file.read(min(size - position, 1024 * 1024))
            last_newline = chunk.rfind(b"\n")
            if last_newline >= 0:
                file.truncate(position + last_newline + 1)
                return
        file.truncate(0)


def load_completed(path: Path) -> set[int]:
    truncate_partial_jsonl(path)
    completed = set()
    if not path.exists():
        return completed
    temporary = path.with_suffix(path.suffix + ".tmp")
    with path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as output:
        for line_number, line in enumerate(source, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from error
            if record.get("status") != "ok":
                continue
            index = int(record["index"])
            if index in completed:
                continue
            completed.add(index)
            output.write(line if line.endswith("\n") else line + "\n")
    os.replace(temporary, path)
    return completed


def iter_length_batches(rows, tokenizer, batch_size: int, bucket_size: int, max_source_tokens: int):
    bucket = []
    for index, source in rows:
        token_count = len(tokenizer(source, add_special_tokens=True, truncation=True, max_length=max_source_tokens)["input_ids"])
        bucket.append((token_count, index, source))
        if len(bucket) >= bucket_size:
            yield from _yield_bucket(bucket, batch_size)
            bucket = []
    if bucket:
        yield from _yield_bucket(bucket, batch_size)


def _yield_bucket(bucket, batch_size: int):
    bucket.sort(key=lambda row: row[0])
    for offset in range(0, len(bucket), batch_size):
        yield [(index, source) for _, index, source in bucket[offset : offset + batch_size]]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
        output.flush()
        os.fsync(output.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for per-rank JSONL shards")
    parser.add_argument("--model", default="facebook/nllb-200-3.3B")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bucket-size", type=int, default=2048)
    parser.add_argument("--max-source-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = rank_info()
    if args.batch_size < 1 or args.bucket_size < args.batch_size or args.max_source_tokens < 1:
        raise SystemExit("batch/bucket/token limits are invalid")
    if not torch.cuda.is_available():
        raise SystemExit("NLLB multi-GPU translation requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, src_lang="kaz_Cyrl", tgt_lang="rus_Cyrl", trust_remote_code=args.trust_remote_code)
    target_id = tokenizer.convert_tokens_to_ids("rus_Cyrl")
    if target_id is None or target_id == tokenizer.unk_token_id:
        raise RuntimeError("Tokenizer does not define the rus_Cyrl target language token")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = args.out_dir / f"rank-{rank:05d}.jsonl"
    metrics_path = args.out_dir / f"rank-{rank:05d}.metrics.json"
    completed = load_completed(shard_path)
    rows = (
        row
        for row in iter_rank_sources(args.input, rank, world_size, args.start, args.limit)
        if row[0] not in completed
    )
    batches = iter_length_batches(rows, tokenizer, args.batch_size, args.bucket_size, args.max_source_tokens)
    started = time.perf_counter()
    processed = 0
    ok = 0
    errors = 0

    def translate_batch(batch):
        sources = [source for _, source in batch]
        encoded = tokenizer(
            sources,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_source_tokens,
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=target_id,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
        translations = tokenizer.batch_decode(generated, skip_special_tokens=True)
        if len(translations) != len(batch) or any(not value.strip() for value in translations):
            raise RuntimeError("NLLB returned a missing or empty translation")
        return [
            {"index": index, "source": source, "translation": translation.strip(), "status": "ok", "model": args.model}
            for (index, source), translation in zip(batch, translations)
        ]

    def translate_resilient(batch):
        try:
            return translate_batch(batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) > 1:
                middle = len(batch) // 2
                return translate_resilient(batch[:middle]) + translate_resilient(batch[middle:])
            index, source = batch[0]
            return [{"index": index, "source": source, "status": "error", "error": "CUDA out of memory"}]
        except Exception as error:
            if len(batch) > 1:
                middle = len(batch) // 2
                return translate_resilient(batch[:middle]) + translate_resilient(batch[middle:])
            index, source = batch[0]
            return [{"index": index, "source": source, "status": "error", "error": str(error)}]

    for batch in batches:
        records = translate_resilient(batch)
        write_jsonl(shard_path, records)
        processed += len(records)
        batch_ok = sum(record["status"] == "ok" for record in records)
        ok += batch_ok
        errors += len(records) - batch_ok
        print(f"rank {rank}/{world_size}: processed={processed:,} ok={ok:,} errors={errors:,} last_index={records[-1]['index']}", flush=True)

    elapsed = time.perf_counter() - started
    metrics = {
        "rank": rank,
        "world_size": world_size,
        "model": args.model,
        "processed_this_run": processed,
        "ok_this_run": ok,
        "errors_this_run": errors,
        "elapsed_sec": elapsed,
        "rows_per_sec": processed / elapsed if elapsed else 0.0,
        "num_beams": args.num_beams,
        "batch_size": args.batch_size,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
