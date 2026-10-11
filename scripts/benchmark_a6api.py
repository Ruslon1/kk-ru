#!/usr/bin/env python3
"""Benchmark a Responses-compatible translation model in resumable batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTIONS = (
    "Translate every Kazakh sentence to Russian. Return only a valid JSON array of Russian strings, "
    "in exactly the same order. Do not add explanations or markdown."
)


def pairs(path: Path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            source, separator, reference = line.rstrip("\n").partition("\t")
            if separator and source and reference:
                yield source, reference


def source_hash(source: str, reference: str) -> str:
    return hashlib.sha256(f"{source}\0{reference}".encode()).hexdigest()


def response_text(payload: dict) -> str:
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return content["text"].strip()
    raise RuntimeError("A6API response has no output_text")


def parse_translations(text: str, expected: int) -> list[str]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    translations = json.loads(cleaned)
    if not isinstance(translations, list) or len(translations) != expected:
        raise RuntimeError(f"Expected JSON array with {expected} translations")
    if not all(isinstance(item, str) and item.strip() for item in translations):
        raise RuntimeError("Translation array contains an empty or non-string item")
    return [item.strip() for item in translations]


def request_batch(model: str, sources: list[str], api_key: str, base_url: str, max_tokens: int, timeout: float, reasoning_effort: str) -> dict:
    payload = {
        "model": model,
        "instructions": INSTRUCTIONS,
        "input": json.dumps(sources, ensure_ascii=False),
        "reasoning": {"effort": reasoning_effort},
        "max_output_tokens": max_tokens,
    }
    req = Request(
        f"{base_url.rstrip('/')}/responses",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read().decode())
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(str(error)) from error
    return {
        "translations": parse_translations(response_text(body), len(sources)),
        "latency_sec": time.perf_counter() - started,
        "usage": body.get("usage") or {},
        "response_id": body.get("id"),
        "model": body.get("model", model),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--base-url", default="https://api.a6api.com/v1")
    parser.add_argument("--api-key-env", default="A6API_API_KEY")
    parser.add_argument("--data", type=Path, default=ROOT / "data/eval/flores_plus/devtest.kk-ru.tsv")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/a6api-gpt-6-luna")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high"), default="none")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env} in the environment; it is never written to reports")
    rows = list(pairs(args.data))
    if args.limit:
        rows = rows[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    records_path = args.out / "devtest.jsonl"
    completed = {}
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                if record.get("status") == "ok" and "hypothesis" in record:
                    completed[record["index"]] = record
            except json.JSONDecodeError:
                continue
    pending = [(i, rows[i][0], rows[i][1]) for i in range(len(rows)) if i not in completed]
    batches = [pending[i : i + args.batch_size] for i in range(0, len(pending), args.batch_size)]
    started = time.perf_counter()
    with records_path.open("a", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(request_batch, args.model, [item[1] for item in batch], api_key, args.base_url, args.max_tokens, args.timeout, args.reasoning_effort): batch
                for batch in batches
            }
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    result = future.result()
                    records = [
                        {"index": item[0], "id": source_hash(item[1], item[2]), "source": item[1], "reference": item[2], "status": "ok", "hypothesis": result["translations"][offset], "latency_sec": result["latency_sec"], "usage": result["usage"], "response_id": result["response_id"], "model": result["model"]}
                        for offset, item in enumerate(batch)
                    ]
                except Exception as error:
                    records = [{"index": item[0], "id": source_hash(item[1], item[2]), "source": item[1], "reference": item[2], "status": "error", "error": str(error)} for item in batch]
                for record in records:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                print(f"processed {len(records)} rows; last id {records[-1]['index']}", flush=True)

    ordered = {}
    for line in records_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("status") == "ok":
            ordered[record["index"]] = record
    ids = sorted(ordered)
    hypotheses = [ordered[index]["hypothesis"] for index in ids]
    references = [rows[index][1] for index in ids]
    latencies = sorted(ordered[index]["latency_sec"] for index in ids)
    metrics = {
        "model": args.model, "base_url": args.base_url, "split": "devtest", "rows": len(rows), "successful": len(ids), "errors": len(rows) - len(ids), "batch_size": args.batch_size, "reasoning_effort": args.reasoning_effort, "elapsed_sec": time.perf_counter() - started,
        "latency_p50_sec": latencies[len(latencies) // 2] if latencies else None, "latency_p95_sec": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else None,
        "bleu": sacrebleu.corpus_bleu(hypotheses, [references]).score if hypotheses else None, "chrf": sacrebleu.corpus_chrf(hypotheses, [references]).score if hypotheses else None, "chrf++": sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score if hypotheses else None, "ter": sacrebleu.corpus_ter(hypotheses, [references]).score if hypotheses else None,
        "input_tokens": sum(ordered[index].get("usage", {}).get("input_tokens", 0) for index in ids), "output_tokens": sum(ordered[index].get("usage", {}).get("output_tokens", 0) for index in ids),
    }
    (args.out / "devtest.metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / "devtest.hyp.txt").write_text("\n".join(ordered[index]["hypothesis"] for index in range(len(rows)) if index in ordered) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
