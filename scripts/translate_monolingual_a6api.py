#!/usr/bin/env python3
"""Translate a large Kazakh monolingual file through A6API in resumable batches."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

INSTRUCTIONS = (
    "Translate every Kazakh sentence to Russian. Return only a valid JSON array of Russian strings, "
    "in exactly the same order. Do not add explanations or markdown."
)


def read_sources(path: Path, start: int, limit: int) -> list[tuple[int, str]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for index, line in enumerate(file):
            if index < start:
                continue
            if limit and index >= start + limit:
                break
            source = line.rstrip("\n").strip()
            if source:
                rows.append((index, source))
    return rows


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
    request = Request(
        f"{base_url.rstrip('/')}/responses",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode())
    return {
        "translations": parse_translations(response_text(body), len(sources)),
        "usage": body.get("usage") or {},
        "response_id": body.get("id"),
        "model": body.get("model", model),
    }


def call_with_retries(args, sources: list[str], api_key: str) -> dict:
    last_error = None
    for attempt in range(args.max_retries + 1):
        started = time.perf_counter()
        try:
            result = request_batch(args.model, sources, api_key, args.base_url, args.max_tokens, args.timeout, args.reasoning_effort)
            result["latency_sec"] = time.perf_counter() - started
            return result
        except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as error:
            last_error = error
            if attempt < args.max_retries:
                time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(str(last_error))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Checkpoint JSONL path")
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--base-url", default="https://api.a6api.com/v1")
    parser.add_argument("--api-key-env", default="A6API_API_KEY")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high"), default="none")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env}; the key is never written to disk")
    if args.batch_size < 1 or args.workers < 1:
        raise SystemExit("--batch-size and --workers must be positive")

    rows = read_sources(args.input, args.start, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.out.exists():
        with args.out.open(encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("status") == "ok":
                    completed.add(record["index"])
    pending = [(index, source) for index, source in rows if index not in completed]
    batches = [pending[pos : pos + args.batch_size] for pos in range(0, len(pending), args.batch_size)]
    print(f"input rows: {len(rows):,}; completed: {len(completed):,}; pending: {len(pending):,}; batches: {len(batches):,}", flush=True)

    with args.out.open("a", encoding="utf-8") as output:
        for group_start in range(0, len(batches), args.workers):
            group = batches[group_start : group_start + args.workers]
            with ThreadPoolExecutor(max_workers=len(group)) as pool:
                futures = {pool.submit(call_with_retries, args, [source for _, source in batch], api_key): batch for batch in group}
                for future in as_completed(futures):
                    batch = futures[future]
                    try:
                        result = future.result()
                        records = [
                            {"index": index, "source": source, "translation": result["translations"][offset], "status": "ok", "usage": result["usage"], "latency_sec": result["latency_sec"], "response_id": result["response_id"], "model": result["model"]}
                            for offset, (index, source) in enumerate(batch)
                        ]
                    except Exception as error:
                        records = [{"index": index, "source": source, "status": "error", "error": str(error)} for index, source in batch]
                    output.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
                    output.flush()
                    ok = sum(record["status"] == "ok" for record in records)
                    print(f"processed {len(records):,} rows ({ok:,} ok); last index {records[-1]['index']}", flush=True)


if __name__ == "__main__":
    main()
