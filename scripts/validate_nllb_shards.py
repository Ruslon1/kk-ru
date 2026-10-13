#!/usr/bin/env python3
"""Verify that NLLB JSONL shards cover each non-empty input row exactly once."""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=None, help="Temporary SQLite index; defaults to system temp")
    args = parser.parse_args()
    shard_files = sorted(args.shards.glob("rank-*.jsonl"))
    if not shard_files:
        raise SystemExit(f"No rank-*.jsonl files found in {args.shards}")
    temporary = args.db is None
    if temporary:
        handle = tempfile.NamedTemporaryFile(prefix="nllb-shard-check-", suffix=".sqlite", delete=False)
        db_path = Path(handle.name)
        handle.close()
    else:
        db_path = args.db
        db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path)
    db.execute("DROP TABLE IF EXISTS expected")
    db.execute("DROP TABLE IF EXISTS seen")
    db.execute("CREATE TABLE expected (idx INTEGER PRIMARY KEY, source TEXT NOT NULL)")
    db.execute("CREATE TABLE seen (idx INTEGER PRIMARY KEY)")
    with args.input.open(encoding="utf-8") as source_file:
        for index, line in enumerate(source_file):
            source = line.rstrip("\r\n").partition("\t")[0].strip()
            if source:
                db.execute("INSERT INTO expected(idx, source) VALUES (?, ?)", (index, source))
    expected = db.execute("SELECT COUNT(*) FROM expected").fetchone()[0]
    stats = {"expected_nonempty": expected, "records": 0, "ok": 0, "errors": 0, "empty_translation": 0, "same_as_source": 0, "source_mismatch": 0, "duplicates": 0, "out_of_range": 0}
    try:
        with db:
            for shard in shard_files:
                with shard.open(encoding="utf-8") as file:
                    for line in file:
                        record = json.loads(line)
                        stats["records"] += 1
                        index = int(record["index"])
                        expected_row = db.execute("SELECT source FROM expected WHERE idx=?", (index,)).fetchone()
                        if expected_row is None:
                            stats["out_of_range"] += 1
                            continue
                        if record.get("source", "").strip() != expected_row[0]:
                            stats["source_mismatch"] += 1
                        if db.execute("INSERT OR IGNORE INTO seen(idx) VALUES (?)", (index,)).rowcount == 0:
                            stats["duplicates"] += 1
                        if record.get("status") != "ok":
                            stats["errors"] += 1
                            continue
                        stats["ok"] += 1
                        source = record.get("source", "").strip()
                        translation = record.get("translation", "").strip()
                        if not translation:
                            stats["empty_translation"] += 1
                        if source and source == translation:
                            stats["same_as_source"] += 1
        unique = db.execute("SELECT COUNT(*) FROM seen").fetchone()[0]
        stats["unique_indexes"] = unique
        stats["missing"] = db.execute(
            "SELECT COUNT(*) FROM expected e LEFT JOIN seen s ON e.idx=s.idx WHERE s.idx IS NULL"
        ).fetchone()[0]
        valid = (
            unique == expected
            and stats["records"] == expected
            and stats["ok"] == expected
            and not any(stats[key] for key in ("errors", "empty_translation", "same_as_source", "source_mismatch", "duplicates", "out_of_range", "missing"))
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        if not valid:
            raise SystemExit(1)
    finally:
        db.close()
        if temporary:
            db_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
