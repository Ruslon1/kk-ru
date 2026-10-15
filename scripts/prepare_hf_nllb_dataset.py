#!/usr/bin/env python3
"""Prepare clean Hugging Face JSONL shards from NLLB translation shards."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("shards"))
    parser.add_argument("--output", type=Path, default=Path("hf-nllb-kk-ru"))
    args = parser.parse_args()

    inputs = sorted(args.input.glob("rank-*.jsonl"))
    if not inputs:
        raise SystemExit(f"no rank-*.jsonl files found in {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    total = written = skipped = 0

    for source_path in inputs:
        output_path = args.output / f"{source_path.stem}.jsonl.gz"
        with source_path.open(encoding="utf-8") as source, gzip.open(
            output_path, "wt", encoding="utf-8"
        ) as output:
            for line_number, line in enumerate(source, 1):
                total += 1
                record = json.loads(line)
                translation = record.get("translation", "").strip()
                source_text = record.get("source", "").strip()
                if record.get("status") != "ok" or not source_text or not translation or source_text == translation:
                    skipped += 1
                    continue
                output.write(json.dumps(
                    {"source": source_text, "translation": translation},
                    ensure_ascii=False,
                ) + "\n")
                written += 1

    manifest = {
        "dataset": "kk-ru-nllb-synthetic",
        "format": "jsonl.gz",
        "files": len(inputs),
        "input_records": total,
        "written_records": written,
        "skipped_records": skipped,
        "features": {"source": "string", "translation": "string"},
        "source_model": "facebook/nllb-200-3.3B",
        "source_language": "kaz_Cyrl",
        "target_language": "rus_Cyrl",
    }
    (args.output / "dataset_info.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
