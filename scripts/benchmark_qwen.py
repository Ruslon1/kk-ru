#!/usr/bin/env python3
"""Evaluate a Qwen MLX model on the local KK-RU evaluation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
from pathlib import Path

import sacrebleu
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


SYSTEM_PROMPT = (
    "Translate Kazakh to Russian. Return only the Russian translation. "
    "Do not explain or add commentary."
)


def pairs(path: Path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            source, separator, reference = line.rstrip("\n").partition("\t")
            if separator and source and reference:
                yield source, reference


def clean_output(text: str) -> str:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text.replace("<|im_end|>", "").strip()


def request_prompt(tokenizer, source: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": source},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def source_hash(source: str, reference: str) -> str:
    return hashlib.sha256(f"{source}\0{reference}".encode()).hexdigest()


def load_completed(path: Path) -> dict[str, dict]:
    completed = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            completed[row["id"]] = row
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--split", choices=["dev", "devtest"], default="devtest")
    parser.add_argument("--data-root", default="data/eval/flores_plus")
    parser.add_argument("--out", default="reports/qwen3-0.6b")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    data_path = Path(args.data_root) / f"{args.split}.kk-ru.tsv"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / f"{args.split}.jsonl"
    completed = load_completed(records_path)
    rows = list(pairs(data_path))
    if args.limit:
        rows = rows[: args.limit]

    model, tokenizer = load(args.model)
    sampler = make_sampler(temp=0.0)
    started = time.perf_counter()
    with records_path.open("a", encoding="utf-8") as file:
        for index, (source, reference) in enumerate(rows):
            row_id = source_hash(source, reference)
            if row_id in completed:
                continue
            request_started = time.perf_counter()
            prompt = request_prompt(tokenizer, source)
            hypothesis = clean_output(
                generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=args.max_tokens,
                    sampler=sampler,
                    verbose=False,
                )
            )
            record = {
                "index": index,
                "id": row_id,
                "source": source,
                "reference": reference,
                "hypothesis": hypothesis,
                "latency_sec": time.perf_counter() - request_started,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            print(f"{index + 1}/{len(rows)} latency={record['latency_sec']:.2f}s", flush=True)

    ordered = [load_completed(records_path)[source_hash(source, reference)] for source, reference in rows]
    hypotheses = [row["hypothesis"] for row in ordered]
    references = [row["reference"] for row in ordered]
    metrics = {
        "model": args.model,
        "split": args.split,
        "rows": len(ordered),
        "elapsed_sec": time.perf_counter() - started,
        "bleu": sacrebleu.corpus_bleu(hypotheses, [references]).score,
        "chrf": sacrebleu.corpus_chrf(hypotheses, [references]).score,
        "chrf++": sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score,
        "latency_p50_sec": sorted(row["latency_sec"] for row in ordered)[len(ordered) // 2],
    }
    try:
        metrics["spbleu"] = sacrebleu.corpus_bleu(
            hypotheses, [references], tokenize="flores200"
        ).score
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        metrics["metric_errors"] = {"spbleu": str(error)}
    (out_dir / f"{args.split}.metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
