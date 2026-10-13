from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "merge_synthetic.py"


class MergeSyntheticTests(unittest.TestCase):
    def run_merge(self, base, translations, output, database):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--base", str(base),
                "--translations", str(translations),
                "--out", str(output),
                "--db", str(database),
            ],
            capture_output=True,
            text=True,
        )

    def test_merge_deduplicates_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.tsv"
            shards = root / "shards"
            shards.mkdir()
            output = root / "merged.tsv"
            database = root / "dedup.sqlite"
            base.write_text("Қазақша\tРусский\n", encoding="utf-8")
            records = [
                {"index": 0, "source": "Қазақша", "translation": "Русский", "status": "ok"},
                {"index": 1, "source": "Сәлем", "translation": "Привет", "status": "ok"},
            ]
            (shards / "rank-00000.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                encoding="utf-8",
            )

            first = self.run_merge(base, shards, output, database)
            content = output.read_text(encoding="utf-8")
            second = self.run_merge(base, shards, output, database)
            final_content = output.read_text(encoding="utf-8")

        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(content, "Қазақша\tРусский\nСәлем\tПривет\n")
        self.assertEqual(final_content, content)
        self.assertIn('"duplicates": 1', first.stdout)

    def test_failed_merge_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, shard, output, database = (root / name for name in ("base.tsv", "bad.jsonl", "out.tsv", "db.sqlite"))
            base.write_text("a\tb\n", encoding="utf-8")
            output.write_text("previous\n", encoding="utf-8")
            shard.write_text(json.dumps({"index": 0, "source": "x", "status": "error", "error": "failed"}) + "\n", encoding="utf-8")
            result = self.run_merge(base, shard, output, database)
            final_content = output.read_text(encoding="utf-8")

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(final_content, "previous\n")


if __name__ == "__main__":
    unittest.main()
