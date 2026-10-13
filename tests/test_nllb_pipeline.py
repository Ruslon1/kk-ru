from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "translate_nllb.py"
SPEC = importlib.util.spec_from_file_location("translate_nllb", SCRIPT)
NLLB = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(NLLB)


class NllbPipelineTests(unittest.TestCase):
    def test_rank_shards_cover_nonempty_input_without_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.txt"
            source.write_text("first\n\nsecond\nthird\nfourth\n", encoding="utf-8")
            shards = [
                list(NLLB.iter_rank_sources(source, rank, 3))
                for rank in range(3)
            ]

        indexes = [index for shard in shards for index, _ in shard]
        self.assertCountEqual(indexes, [0, 2, 3, 4])
        self.assertEqual(len(indexes), len(set(indexes)))

    def test_resume_keeps_successes_and_removes_errors_and_partial_record(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "rank-00000.jsonl"
            shard.write_text(
                json.dumps({"index": 2, "status": "ok"}) + "\n"
                + json.dumps({"index": 3, "status": "error"}) + "\n"
                + '{"index": 4, "status": "ok"',
                encoding="utf-8",
            )
            completed = NLLB.load_completed(shard)
            content = shard.read_text(encoding="utf-8")

        self.assertEqual(completed, {2})
        self.assertTrue(content.endswith("\n"))
        self.assertNotIn('"index": 4', content)
        self.assertNotIn('"index": 3', content)

    def test_resume_compacts_duplicate_successes(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "rank-00000.jsonl"
            row = json.dumps({"index": 2, "status": "ok"}) + "\n"
            shard.write_text(row + row, encoding="utf-8")

            self.assertEqual(NLLB.load_completed(shard), {2})
            self.assertEqual(shard.read_text(encoding="utf-8"), row)

    def test_length_bucketing_keeps_indices_and_groups_similar_lengths(self):
        class Tokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": list(range(len(text)))}

        batches = list(
            NLLB.iter_length_batches(
                [(0, "a"), (1, "aaaa"), (2, "aa"), (3, "aaaaa")],
                Tokenizer(),
                batch_size=2,
                bucket_size=4,
                max_source_tokens=20,
            )
        )
        self.assertEqual([[i for i, _ in batch] for batch in batches], [[0, 2], [1, 3]])

    def test_validator_can_reuse_its_sqlite_index(self):
        script = SCRIPT.with_name("validate_nllb_shards.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            shards = root / "shards"
            shards.mkdir()
            database = root / "validation.sqlite"
            source.write_text("one\ntwo\n", encoding="utf-8")
            (shards / "rank-00000.jsonl").write_text(
                "".join(json.dumps({"index": i, "source": text, "translation": f"ru-{i}", "status": "ok"}) + "\n"
                      for i, text in enumerate(("one", "two"))),
                encoding="utf-8",
            )

            command = [sys.executable, str(script), "--input", str(source), "--shards", str(shards), "--db", str(database)]
            first = subprocess.run(command, capture_output=True, text=True)
            second = subprocess.run(command, capture_output=True, text=True)

        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)

    def test_validator_rejects_source_index_mismatch(self):
        script = SCRIPT.with_name("validate_nllb_shards.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            shards = root / "shards"
            shards.mkdir()
            source.write_text("expected\n", encoding="utf-8")
            (shards / "rank-00000.jsonl").write_text(
                json.dumps({"index": 0, "source": "wrong", "translation": "text", "status": "ok"}) + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(script), "--input", str(source), "--shards", str(shards)],
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn('"source_mismatch": 1', result.stdout)


if __name__ == "__main__":
    unittest.main()
