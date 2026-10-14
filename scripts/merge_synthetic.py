#!/usr/bin/env python3
"""Atomically merge translated JSONL shards into a deduplicated KK-RU TSV."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import unicodedata
from pathlib import Path

WS = re.compile(r"\s+")


def normalize(value: str) -> str:
    return WS.sub(" ", unicodedata.normalize("NFC", value)).strip()


def pair_key(kk: str, ru: str) -> str:
    return hashlib.sha256(f"{kk}\0{ru}".encode()).hexdigest()


def translation_files(path: Path) -> list[Path]:
    files = sorted(path.glob("rank-*.jsonl")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No translation JSONL files found at {path}")
    return files


def add_pair(output, db, kk: str, ru: str, stats: dict) -> None:
    kk, ru = normalize(kk), normalize(ru)
    if not kk or not ru or kk == ru:
        stats["dropped_invalid_pairs"] += 1
        return
    if db.execute("INSERT OR IGNORE INTO pairs(key) VALUES (?)", (pair_key(kk, ru),)).rowcount == 0:
        stats["duplicates"] += 1
        return
    output.write(f"{kk}\t{ru}\n")
    stats["written"] += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--translations", type=Path, required=True, help="JSONL shard file or directory of rank-*.jsonl files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True, help="Final SQLite dedup index")
    parser.add_argument("--skip-errors", action="store_true", help="Skip failed translation records")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.db.parent.mkdir(parents=True, exist_ok=True)
    shards = translation_files(args.translations)
    out_fd, out_temp_name = tempfile.mkstemp(prefix=f".{args.out.name}.", suffix=".tmp", dir=args.out.parent)
    os.close(out_fd)
    db_fd, db_temp_name = tempfile.mkstemp(prefix=f".{args.db.name}.", suffix=".tmp", dir=args.db.parent)
    os.close(db_fd)
    out_temp, db_temp = Path(out_temp_name), Path(db_temp_name)
    stats = {"base_rows": 0, "translation_rows": 0, "written": 0, "duplicates": 0, "dropped_invalid_pairs": 0, "translation_errors": 0}
    try:
        db = sqlite3.connect(db_temp)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute("CREATE TABLE pairs (key TEXT PRIMARY KEY)")
        db.execute("CREATE TABLE translation_indexes (idx INTEGER PRIMARY KEY)")
        with out_temp.open("w", encoding="utf-8") as output:
            with args.base.open(encoding="utf-8") as base:
                for line in base:
                    if "\t" not in line:
                        continue
                    kk, ru = line.rstrip("\n").split("\t", 1)
                    stats["base_rows"] += 1
                    add_pair(output, db, kk, ru, stats)
            for shard in shards:
                with shard.open(encoding="utf-8") as translated:
                    for line_number, line in enumerate(translated, 1):
                        record = json.loads(line)
                        stats["translation_rows"] += 1
                        if record.get("status") != "ok":
                            stats["translation_errors"] += 1
                            if args.skip_errors:
                                continue
                            raise ValueError(f"translation error in {shard}:{line_number}: {record.get('error', 'unknown error')}")
                        index = int(record["index"])
                        if db.execute("INSERT OR IGNORE INTO translation_indexes(idx) VALUES (?)", (index,)).rowcount == 0:
                            raise ValueError(f"duplicate translated input index {index}")
                        add_pair(output, db, record.get("source", ""), record.get("translation", ""), stats)
                        if stats["translation_rows"] % 100_000 == 0:
                            db.commit()
                            print(f"merged {stats['translation_rows']:,} translated rows", flush=True)
            output.flush()
            os.fsync(output.fileno())
        db.commit()
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        db.close()
        os.replace(out_temp, args.out)
        os.replace(db_temp, args.db)
        stats["output_rows"] = stats["written"]
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    finally:
        out_temp.unlink(missing_ok=True)
        db_temp.unlink(missing_ok=True)
        Path(str(db_temp) + "-wal").unlink(missing_ok=True)
        Path(str(db_temp) + "-shm").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
