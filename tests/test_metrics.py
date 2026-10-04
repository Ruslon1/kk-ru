import sys
import types
import unittest
from unittest.mock import patch

from kk_ru.metrics import compute_translation_metrics


class Score:
    def __init__(self, value):
        self.score = value


class FakeSacrebleu:
    @staticmethod
    def corpus_bleu(hypotheses, references, **kwargs):
        return Score(10 if kwargs.get("tokenize") == "flores200" else 20)

    @staticmethod
    def corpus_chrf(hypotheses, references, **kwargs):
        return Score(30 if kwargs.get("word_order") == 2 else 40)

    @staticmethod
    def corpus_ter(hypotheses, references):
        return Score(50)


class MetricsTests(unittest.TestCase):
    def test_calculates_lexical_metrics(self):
        with patch.dict(sys.modules, {"sacrebleu": FakeSacrebleu}):
            metrics = compute_translation_metrics(["перевод"], ["эталон"], ["мәтін"])

        self.assertEqual(
            metrics,
            {"bleu": 20, "chrf": 40, "chrf++": 30, "ter": 50, "spbleu": 10},
        )

    def test_records_optional_metric_import_errors(self):
        with patch.dict(sys.modules, {"sacrebleu": FakeSacrebleu, "comet": None}):
            metrics = compute_translation_metrics(
                ["перевод"],
                ["эталон"],
                ["мәтін"],
                comet_model="Unbabel/test",
            )

        self.assertIn("metric_errors", metrics)
        self.assertIn("comet", metrics["metric_errors"])
        self.assertNotIn("comet", metrics)

    def test_rejects_misaligned_rows(self):
        with self.assertRaises(ValueError):
            compute_translation_metrics(["перевод"], [], ["мәтін"])


if __name__ == "__main__":
    unittest.main()
