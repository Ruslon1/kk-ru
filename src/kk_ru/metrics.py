from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NeuralMetricRuntime:
    cache: dict[str, Any] = field(default_factory=dict)

    def _load_comet(self, model_id: str) -> Any:
        if model_id not in self.cache:
            from comet import download_model, load_from_checkpoint

            checkpoint = download_model(model_id)
            self.cache[model_id] = load_from_checkpoint(checkpoint)
        return self.cache[model_id]


def _mean(value: Any) -> float:
    if hasattr(value, "mean"):
        value = value.mean()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _system_score(result: Any) -> float:
    if isinstance(result, dict):
        return _mean(result["system_score"])
    return _mean(result.system_score)


def _gpu_count() -> int:
    try:
        import torch
    except ImportError:
        return 0
    return int(torch.cuda.is_available())


def _metric_error(errors: dict[str, str], name: str, error: Exception) -> None:
    errors[name] = f"{type(error).__name__}: {error}"


def compute_lexical_metrics(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    import sacrebleu

    return {
        "bleu": sacrebleu.corpus_bleu(hypotheses, [references]).score,
        "chrf": sacrebleu.corpus_chrf(hypotheses, [references]).score,
        "chrf++": sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score,
        "ter": sacrebleu.corpus_ter(hypotheses, [references]).score,
    }


def compute_translation_metrics(
    hypotheses: list[str],
    references: list[str],
    sources: list[str],
    *,
    comet_model: str | None = None,
    cometkiwi_model: str | None = None,
    xcomet_model: str | None = None,
    bertscore_model: str | None = None,
    runtime: NeuralMetricRuntime | None = None,
) -> dict[str, Any]:
    if not (len(hypotheses) == len(references) == len(sources)):
        raise ValueError("hypotheses, references, and sources must have equal lengths")

    try:
        import sacrebleu
    except ImportError as error:
        raise RuntimeError("Install sacrebleu to calculate translation metrics") from error

    metrics: dict[str, Any] = compute_lexical_metrics(hypotheses, references)
    errors: dict[str, str] = {}
    try:
        metrics["spbleu"] = sacrebleu.corpus_bleu(
            hypotheses, [references], tokenize="flores200"
        ).score
    except (OSError, TimeoutError, ValueError) as error:
        _metric_error(errors, "spbleu", error)

    runtime = runtime or NeuralMetricRuntime()
    comet_data = [
        {"src": source, "mt": hypothesis, "ref": reference}
        for source, hypothesis, reference in zip(sources, hypotheses, references)
    ]
    for name, model_id in (
        ("comet", comet_model),
        ("cometkiwi", cometkiwi_model),
        ("xcomet", xcomet_model),
    ):
        if not model_id:
            continue
        data = (
            [{"src": source, "mt": hypothesis} for source, hypothesis in zip(sources, hypotheses)]
            if name == "cometkiwi" else comet_data
        )
        try:
            scorer = runtime._load_comet(model_id)
            result = scorer.predict(data, batch_size=8, gpus=_gpu_count())
            metrics[name] = _system_score(result)
            metrics[f"{name}_model"] = model_id
        except Exception as error:
            _metric_error(errors, name, error)

    if bertscore_model:
        try:
            import bert_score

            _, _, f1 = bert_score.score(
                hypotheses,
                references,
                model_type=bertscore_model,
                lang="ru",
                rescale_with_baseline=True,
                verbose=False,
            )
            metrics["bertscore_f1"] = _mean(f1)
            metrics["bertscore_model"] = bertscore_model
        except Exception as error:
            _metric_error(errors, "bertscore", error)

    if errors:
        metrics["metric_errors"] = errors
    return metrics
