import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from kk_ru.openrouter import ModelInfo


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "benchmark_openrouter.py"
SPEC = importlib.util.spec_from_file_location("benchmark_openrouter", SCRIPT_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


class BenchmarkHelpersTests(unittest.TestCase):
    def setUp(self):
        self.model = ModelInfo(
            id="qwen/test",
            name="Qwen test",
            prompt_price=0.000001,
            completion_price=0.000002,
            context_length=4096,
        )

    def test_messages_use_one_stable_translation_prompt(self):
        messages = benchmark.messages_for("Сәлем")

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Kazakh", messages[0]["content"])
        self.assertEqual(messages[1], {"role": "user", "content": "Сәлем"})

    def test_estimate_run_cost_scales_with_sources(self):
        one = benchmark.estimate_run_cost(self.model, ["a"], 10, safety_factor=1)
        two = benchmark.estimate_run_cost(self.model, ["a", "b"], 10, safety_factor=1)

        self.assertIsNotNone(one)
        self.assertGreater(two, one)

    def test_load_completed_ignores_stale_or_failed_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.jsonl"
            rows = [
                {
                    "index": 0,
                    "model": "qwen/test",
                    "source_sha256": "source-0",
                    "request_sha256": "request-0",
                    "status": "ok",
                    "hypothesis": "один",
                },
                {
                    "index": 1,
                    "model": "qwen/test",
                    "source_sha256": "wrong",
                    "request_sha256": "request-1",
                    "status": "ok",
                    "hypothesis": "два",
                },
                {
                    "index": 2,
                    "model": "qwen/test",
                    "source_sha256": "source-2",
                    "request_sha256": "request-2",
                    "status": "error",
                },
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            completed = benchmark.load_completed(
                path,
                "qwen/test",
                ["source-0", "source-1", "source-2"],
                ["request-0", "request-1", "request-2"],
            )

        self.assertEqual(sorted(completed), [0])
        self.assertEqual(completed[0]["hypothesis"], "один")

    def test_metrics_reject_misaligned_inputs(self):
        with self.assertRaises(ValueError):
            benchmark.calculate_metrics(["перевод"], ["эталон"], [])


if __name__ == "__main__":
    unittest.main()
