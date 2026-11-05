"""Reproduce P0 report evaluation and rescore existing translation artifacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import time
import unicodedata
from pathlib import Path

import sacrebleu
import torch
from sacrebleu.metrics import BLEU, CHRF, TER

from kk_ru.config import load_config
from kk_ru.data import iter_pairs
from kk_ru.eval import translate
from kk_ru.model import build_model, count_params
from kk_ru.tokenizer import load_tokenizer

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).parent / "evidence"
OUTPUT = ROOT / "reports" / "p0-report"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def score(hypotheses: list[str], references: list[str]) -> dict:
    if len(hypotheses) != len(references):
        raise ValueError(f"Expected {len(references)} hypotheses, found {len(hypotheses)}")
    scorers = {"bleu": BLEU(), "chrf": CHRF(), "chrf++": CHRF(word_order=2), "ter": TER()}
    values = {name: scorer.corpus_score(hypotheses, [references]).score for name, scorer in scorers.items()}
    return {
        "rows": len(hypotheses),
        **values,
        "signatures": {name: str(scorer.get_signature()) for name, scorer in scorers.items()},
        "empty_hypotheses": sum(not text.strip() for text in hypotheses),
        "length_ratio_words": sum(len(t.split()) for t in hypotheses) / sum(len(t.split()) for t in references),
    }


def rescore_archives() -> None:
    pairs = list(iter_pairs(str(ROOT / "data/eval/flores_plus/devtest.kk-ru.tsv")))
    sources, refs = map(list, zip(*pairs))
    results = []
    rejected = []
    files = [
        ("KazRush KK-RU", "reports/kazRush-devtest.hyp.txt", "lines"),
        ("Google Translate", "reports/google-translate/devtest.hyp.txt", "lines"),
        ("Qwen3-0.6B", "reports/qwen3-0.6b/devtest.jsonl", "records"),
        ("gpt-6-luna (single)", "reports/a6api-gpt-6-luna/devtest.jsonl", "records"),
        ("gpt-6-luna (batched)", "reports/a6api-gpt-6-luna-batched/devtest.jsonl", "records"),
        ("qwen/qwen3.8-27b", "reports/openrouter/models/qwen-3.8-27b/records.jsonl", "records"),
        ("google/gemma-4-26b-a4b-it", "reports/openrouter/models/gemma-4-26b-a4b-it/records.jsonl", "records"),
        ("google/gemma-4-31b-it", "reports/openrouter/models/gemma-4-31b-it/records.jsonl", "records"),
        ("Unidentified legacy P0 output", "reports/devtest.hyp.txt", "lines"),
    ]
    for name, relative, kind in files:
        path = ROOT / relative
        if not path.exists():
            rejected.append({"model": name, "path": relative, "reason": "file unavailable"})
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            if kind == "records":
                by_index = {}
                for line in lines:
                    record = json.loads(line)
                    index = int(record["index"])
                    if not 0 <= index < len(pairs) or index in by_index:
                        raise ValueError(f"Invalid or duplicate index: {index}")
                    if record["source"] != sources[index] or record["reference"] != refs[index]:
                        raise ValueError(f"Source/reference mismatch at index {index}")
                    if record.get("status", "ok") != "ok":
                        raise ValueError(f"Unsuccessful record at index {index}")
                    by_index[index] = record["hypothesis"]
                hypotheses = [by_index[i] for i in range(len(pairs))]
                alignment = "source and reference verified for every indexed record"
            else:
                hypotheses = lines
                alignment = "positional alignment assumed; archive has no source identifiers"
            metrics = score(hypotheses, refs)
            normalized = OUTPUT / "archives" / f"{path.parent.name}-{path.stem}.hyp.txt"
            normalized.parent.mkdir(parents=True, exist_ok=True)
            normalized.write_text("\n".join(hypotheses) + "\n", encoding="utf-8")
            result = {"model": name, "path": relative, "sha256": sha256(path), "alignment": alignment, **metrics}
            results.append(result)
            print(name, metrics["bleu"], metrics["chrf++"], flush=True)
        except (ValueError, KeyError) as error:
            rejected.append({"model": name, "path": relative,
                             "reason": f"{type(error).__name__}: {error}"})
    write_json(EVIDENCE / "archived_benchmarks.json", {"results": results, "excluded": rejected})


def audit_corpora() -> None:
    def normalize(text):
        return unicodedata.normalize("NFC", text).strip()

    evaluation = {}
    for split in ("dev", "devtest"):
        path = ROOT / f"data/eval/flores_plus/{split}.kk-ru.tsv"
        evaluation[split] = {(normalize(s), normalize(t)) for s, t in iter_pairs(str(path))}
    source_sets = {split: {s for s, _ in pairs} for split, pairs in evaluation.items()}
    result = {"definition": "Exact unique source/pair matches after NFC normalization and stripping"}
    for kind in ("base", "synthetic"):
        overlaps = {split: {"source": set(), "pair": set()} for split in evaluation}
        files = ([ROOT / "data/filtered/train.kk-ru.tsv"] if kind == "base"
                 else sorted((ROOT / "hf-nllb-kk-ru").glob("rank-*.jsonl.gz")))
        count = 0
        invalid = 0
        file_results = []
        for path in files:
            digest = hashlib.sha256()
            file_count = 0
            opener = gzip.open if kind == "synthetic" else open
            with opener(path, "rb") as file:
                for raw in file:
                    digest.update(raw)
                    count += 1
                    file_count += 1
                    if kind == "base":
                        parts = raw.decode("utf-8").rstrip("\n").split("\t")
                        if len(parts) != 2:
                            invalid += 1
                            continue
                        source, target = parts
                    else:
                        row = json.loads(raw)
                        source, target = row["source"], row["translation"]
                    source, target = normalize(source), normalize(target)
                    for split, sources in source_sets.items():
                        if source in sources:
                            overlaps[split]["source"].add(source)
                        if (source, target) in evaluation[split]:
                            overlaps[split]["pair"].add((source, target))
            file_results.append({"path": str(path.relative_to(ROOT)), "rows": file_count,
                                 "decompressed_sha256": digest.hexdigest()})
            print(kind, path.name, file_count, flush=True)
        result[kind] = {"rows": count, "invalid_columns": invalid, "files": file_results,
                        "overlap": {s: {k: len(v) for k, v in values.items()}
                                    for s, values in overlaps.items()}}
    write_json(EVIDENCE / "corpus_overlap.json", result)
    print(json.dumps(result, indent=2), flush=True)


def evaluate_p0(split: str, batch_size: int, limit: int) -> None:
    cfg = load_config(str(ROOT / "configs/p0.yaml"))
    path = ROOT / "checkpoints/p0/best/model.pt"
    tokenizer = load_tokenizer(cfg.tokenizer.sp_model)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model(cfg.model)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.to(device).eval()
    data_path = ROOT / f"data/eval/flores_plus/{split}.kk-ru.tsv"
    pairs = list(iter_pairs(str(data_path)))
    if limit:
        pairs = pairs[:limit]
    run_path = OUTPUT / f"p0-{split}{'-pilot' if limit else ''}.jsonl"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    if run_path.exists():
        rows = [json.loads(line) for line in run_path.read_text().splitlines()]
        for index, record in enumerate(rows):
            if record["index"] != index or (record["source"], record["reference"]) != pairs[index]:
                raise ValueError("Existing evaluation records do not match this dataset")
    started = time.perf_counter()
    print(f"P0 {split}: {len(pairs)} rows, device={device}, beam={cfg.gen.beam}, batch={batch_size}; resume={len(rows)}", flush=True)
    with run_path.open("a", encoding="utf-8") as file:
        for start in range(len(rows), len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            chunk_start = time.perf_counter()
            hypotheses = translate(model, tokenizer, [p[0] for p in chunk], device,
                                   beam=cfg.gen.beam, max_len=cfg.gen.max_len,
                                   source_max_len=cfg.model.max_len, batch_size=batch_size)
            if device.type == "mps":
                torch.mps.synchronize()
            seconds = time.perf_counter() - chunk_start
            for offset, ((source, reference), hypothesis) in enumerate(zip(chunk, hypotheses)):
                row = {"index": start + offset, "source": source, "reference": reference,
                       "hypothesis": hypothesis, "batch_seconds": seconds}
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
            file.flush()
            print(f"{split} {len(rows)}/{len(pairs)}; batch {seconds:.2f}s; session {time.perf_counter()-started:.1f}s", flush=True)
    hypotheses = [row["hypothesis"] for row in rows]
    metrics = score(hypotheses, [p[1] for p in pairs])
    inference_seconds = sum(rows[i]["batch_seconds"] for i in range(0, len(rows), batch_size))
    record = {"model": "P0", "split": split, "checkpoint": str(path.relative_to(ROOT)),
              "checkpoint_sha256": sha256(path), "dataset_sha256": sha256(data_path),
              "hypotheses_sha256": sha256(run_path), "parameters": count_params(model),
              "device": str(device), "platform": platform.platform(), "beam": cfg.gen.beam,
              "batch_size": batch_size, "max_len": cfg.gen.max_len,
              "length_penalty": 0.6, "dtype": str(next(model.parameters()).dtype),
              "inference_seconds_including_first_batch": inference_seconds,
              "sentences_per_second": len(rows) / inference_seconds,
              "torch_version": torch.__version__, "sacrebleu_version": sacrebleu.__version__,
              **metrics}
    write_json(EVIDENCE / f"p0-{split}{'-pilot' if limit else ''}.json", record)
    print(json.dumps(record, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["archives", "p0", "corpora"])
    parser.add_argument("--split", choices=["dev", "devtest"], default="devtest")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "archives":
        rescore_archives()
    elif args.mode == "p0":
        evaluate_p0(args.split, args.batch_size, args.limit)
    else:
        audit_corpora()


if __name__ == "__main__":
    main()
