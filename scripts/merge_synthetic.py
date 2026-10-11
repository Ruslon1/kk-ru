#!/usr/bin/env python3
"""Merge translated JSONL with a TSV corpus using SQLite-backed deduplication."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import unicodedata
from pathlib import Path

WS = re.compile(r"\s+")


def normalize(value: str) -> str:
    return WS.sub(" ", unicodedata.normalize("NFC", value)).strip()


def pair_key(kk: str, ru: str) -> str:
    return hashlib.sha256(f"{kk}\0{ru}".encode()).hexdigest()


def add_pair(output, db, kk: str, ru: str, stats: dict) -> None:
    kk, ru = normalize(kk), normalize(ru)
    if not kk or not ru or kk == ru:
        stats["dropped"] += 1
        return
    key = pair_key(kk, ru)
    if db.execute("INSERT OR IGNORE INTO pairs(key) VALUES (?)", (key,)).rowcount == 0:
        stats["duplicates"] += 1
        return
    output.write(f"{kk}\t{ru}\n")
    stats["written"] += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.db.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.db)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS pairs (key TEXT PRIMARY KEY)")
    stats = {"written": 0, "dropped": 0, "duplicates": 0, "translation_errors": 0}
    with args.out.open("w", encoding="utf-8") as output:
        with args.base.open(encoding="utf-8") as base:
            for line in base:
                if "\t" in line:
                    kk, ru = line.rstrip("\n").split("\t", 1)
                    add_pair(output, db, kk, ru, stats)
        with args.translations.open(encoding="utf-8") as translated:
            for line in translated:
                record = json.loads(line)
                if record.get("status") == "ok":
                    add_pair(output, db, record["source"], record["translation"], stats)
                else:
                    stats["translation_errors"] += 1
    db.commit()
    db.close()
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
