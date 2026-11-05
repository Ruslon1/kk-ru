import sys
import unittest
from unittest.mock import patch

from kk_ru.metrics import compute_lexical_metrics, compute_translation_metrics
from kk_ru.eval import compute_metrics


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
    def test_real_lexical_metrics_score_identical_translations(self):
        text = ["Это точный перевод достаточно длинного предложения."]

        metrics = compute_lexical_metrics(text, text)

        self.assertAlmostEqual(metrics["bleu"], 100)
        self.assertAlmostEqual(metrics["chrf"], 100)
        self.assertAlmostEqual(metrics["chrf++"], 100)
        self.assertAlmostEqual(metrics["ter"], 0)
        self.assertEqual(compute_metrics(text, text), metrics)

    def test_calculates_lexical_metrics(self):
        with patch.dict(sys.modules, {"sacrebleu": FakeSacrebleu}):
            metrics = compute_translation_metrics(["перевод"], ["эталон"], ["мәтін"])

        self.assertEqual(
            metrics,
            {"bleu": 20, "chrf": 40, "chrf++": 30, "ter": 50, "spbleu": 10},
        )

    def test_p0_evaluation_includes_ter(self):
        with patch.dict(sys.modules, {"sacrebleu": FakeSacrebleu}):
            metrics = compute_metrics(["перевод"], ["эталон"])

        self.assertEqual(metrics["ter"], 50)

    def test_records_optional_metric_import_errors(self):
        with patch.dict(sys.modules, {"sacrebleu": FakeSacrebleu, "comet": None}):
            metrics = compute_translation_metrics(
                ["перевод"],
                ["эталон"],
                ["мәтін"],
                comet_model="Unbabel/test",
                cometkiwi_model="Unbabel/test-kiwi",
                xcomet_model="Unbabel/test-xcomet",
            )

        self.assertIn("metric_errors", metrics)
        for name in ("comet", "cometkiwi", "xcomet"):
            self.assertIn(name, metrics["metric_errors"])
            self.assertNotIn(name, metrics)

    def test_rejects_misaligned_rows(self):
        with self.assertRaises(ValueError):
            compute_translation_metrics(["перевод"], [], ["мәтін"])


if __name__ == "__main__":
    unittest.main()
