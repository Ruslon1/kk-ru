#!/usr/bin/env python3
"""Prepare one fixed KK->RU benchmark for comparing translation systems."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def normalize(text: str) -> str:
    return " ".join(text.split()).strip()


def bucket(text: str) -> str:
    words = len(text.split())
    if words <= 8:
        return "short"
    if words <= 30:
        return "medium"
    return "long"


def reservoir_by_bucket(rows, limits: dict[str, int], seed: int):
    rng = random.Random(seed)
    samples: dict[str, list[tuple[str, str]]] = {key: [] for key in limits}
    seen = Counter()
    for row in rows:
        key = bucket(row[0])
        if key not in limits:
            continue
        seen[key] += 1
        sample = samples[key]
        if len(sample) < limits[key]:
            sample.append(row)
        else:
            index = rng.randrange(seen[key])
            if index < limits[key]:
                sample[index] = row
    return samples, seen


def iter_pairs(path: Path):
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) >= 2:
                kk, ru = normalize(parts[0]), normalize(parts[1])
                if kk and ru:
                    yield kk, ru


def iter_mono(path: Path):
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            text = normalize(line)
            if text:
                yield text, ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=ROOT / "data/filtered/train.kk-ru.tsv")
    parser.add_argument("--mono", type=Path, default=ROOT / "data/monolingual/kazakh.sentences.txt")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/benchmark")
    parser.add_argument("--gold-count", type=int, default=5000)
    parser.add_argument("--mono-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    for path in (args.gold, args.mono):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    flores_rows = []
    for split in ("dev", "devtest"):
        path = ROOT / "data/eval/flores_plus" / f"{split}.kk-ru.tsv"
        for kk, ru in iter_pairs(path):
            flores_rows.append((split, kk, ru))

    gold_limits = {key: args.gold_count // 3 for key in ("short", "medium", "long")}
    gold_limits["long"] += args.gold_count - sum(gold_limits.values())
    gold_samples, gold_seen = reservoir_by_bucket(
        ((kk, ru) for kk, ru in iter_pairs(args.gold)), gold_limits, args.seed
    )
    mono_limits = {"short": args.mono_count // 3, "medium": args.mono_count // 3}
    mono_limits["long"] = args.mono_count - sum(mono_limits.values())
    mono_samples, mono_seen = reservoir_by_bucket(
        iter_mono(args.mono), mono_limits, args.seed + 1
    )

    rows: list[dict[str, str]] = []
    seen = set()

    def add(split: str, category: str, kk: str, ru: str) -> None:
        key = normalize(kk).casefold()
        if not key or key in seen:
            return
        seen.add(key)
        rows.append({"id": f"{split}-{len(rows):05d}", "split": split, "category": category, "kk": kk, "ru": ru})

    for split, kk, ru in flores_rows:
        add(f"flores_{split}", "flores", kk, ru)
    for category in ("short", "medium", "long"):
        for kk, ru in gold_samples[category]:
            add("gold", category, kk, ru)
    for category in ("short", "medium", "long"):
        for kk, ru in mono_samples[category]:
            add("monolingual", category, kk, ru)

    output = args.out_dir / "translation_candidates.tsv"
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("id", "split", "category", "kk", "ru"), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "seed": args.seed,
        "gold_input": str(args.gold),
        "mono_input": str(args.mono),
        "output": str(output),
        "rows": len(rows),
        "counts_by_split": dict(Counter(row["split"] for row in rows)),
        "counts_by_category": dict(Counter(row["category"] for row in rows)),
        "gold_available_by_bucket": dict(gold_seen),
        "mono_available_by_bucket": dict(mono_seen),
        "sha256": sha256(output),
        "reference_note": "ru is empty for monolingual rows; use only gold and FLORES rows for reference metrics.",
    }
    (args.out_dir / "translation_candidates.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
