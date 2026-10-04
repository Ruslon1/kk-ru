#!/usr/bin/env python3
"""Compare hosted chat models on the project's Kazakh→Russian eval set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kk_ru.openrouter import (
    ModelInfo,
    OpenRouterClient,
    OpenRouterError,
    estimate_cost,
    usage_cost,
)
from kk_ru.metrics import NeuralMetricRuntime, compute_translation_metrics


DEFAULT_MODELS = [
    "qwen/qwen3.8-27b",
    "qwen/qwen3.8-2.4t-a95b",
    "google/gemma-4-26b-a4b-it",
    "google/gemma-4-31b-it",
    "qwen/qwen3.8-max-0902",
]
DEFAULT_NEURAL_METRICS = {
    "comet_model": "Unbabel/wmt22-comet-da",
    "cometkiwi_model": "Unbabel/wmt22-cometkiwi-da",
    "xcomet_model": "Unbabel/XCOMET-XL",
    "bertscore_model": "xlm-roberta-large",
}
SYSTEM_PROMPT = "You are a professional translator. Translate from Kazakh to Russian. Return only the Russian translation, with no explanation."


def iter_pairs(tsv_path: str):
    with open(tsv_path, encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                yield parts[0], parts[1]


def eval_path_from_config(config_path: Path, split: str) -> str:
    key = "eval_dev" if split == "dev" else "eval_test"
    in_data_section = False
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if line and not line[0].isspace():
            in_data_section = line.strip() == "data:"
        elif in_data_section:
            name, separator, value = line.strip().partition(":")
            if separator and name == key:
                return value.strip().strip("\"'")
    raise ValueError(f"{key} is missing from config {config_path}")


def messages_for(source: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": source},
    ]


def reasoning_for(model: ModelInfo) -> dict[str, Any] | None:
    if "reasoning" not in model.supported_parameters:
        return None
    return {"effort": "none", "exclude": True}


def request_hash(
    model_id: str,
    source: str,
    max_tokens: int,
    reasoning: dict[str, Any] | None,
) -> str:
    request = {
        "model": model_id,
        "messages": messages_for(source),
        "temperature": 0,
        "max_tokens": max_tokens,
        "reasoning": reasoning,
    }
    encoded = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def estimate_run_cost(
    model: ModelInfo,
    sources: list[str],
    max_tokens: int,
    safety_factor: float = 1.25,
) -> float | None:
    if model.prompt_price is None or model.completion_price is None:
        return None
    estimate = sum(
        estimate_cost(model, messages_for(source), max_tokens)
        for source in sources
    )
    return estimate * safety_factor


def calculate_metrics(
    hypotheses: list[str],
    references: list[str],
    sources: list[str],
    comet_model: str | None = None,
    cometkiwi_model: str | None = None,
    xcomet_model: str | None = None,
    bertscore_model: str | None = None,
    runtime: NeuralMetricRuntime | None = None,
) -> dict[str, Any]:
    return compute_translation_metrics(
        hypotheses,
        references,
        sources,
        comet_model=comet_model,
        cometkiwi_model=cometkiwi_model,
        xcomet_model=xcomet_model,
        bertscore_model=bertscore_model,
        runtime=runtime,
    )


def _record_cost(record: dict[str, Any], model: ModelInfo) -> float:
    value = record.get("cost_usd")
    if value is not None:
        return float(value)
    return usage_cost(record.get("usage", {}), model) or 0.0


def load_completed(
    path: Path,
    model: str,
    source_hashes: list[str],
    request_hashes: list[str],
) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            index = int(record["index"])
            if (
                record.get("model") == model
                and 0 <= index < len(source_hashes)
                and record.get("source_sha256") == source_hashes[index]
                and record.get("request_sha256") == request_hashes[index]
                and record.get("status") == "ok"
            ):
                completed[index] = record
    return completed


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_json(path.with_suffix(".json"), rows)
    fields = {"model", "provider", "rows", "cost_usd", "mean_latency_seconds"}
    flattened = []
    for row in rows:
        item = {key: value for key, value in row.items() if key != "metrics"}
        for key, value in (row.get("metrics") or {}).items():
            column = f"metric_{key}"
            item[column] = value if isinstance(value, (str, int, float, bool)) else json.dumps(value, ensure_ascii=False)
            fields.add(column)
        flattened.append(item)
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=sorted(fields))
        writer.writeheader()
        writer.writerows(flattened)


def _metrics_if_available(
    hypotheses: list[str],
    references: list[str],
    sources: list[str],
    args: argparse.Namespace,
    runtime: NeuralMetricRuntime,
) -> dict[str, Any] | None:
    try:
        return calculate_metrics(
            hypotheses,
            references,
            sources,
            comet_model=args.comet_model,
            cometkiwi_model=args.cometkiwi_model,
            xcomet_model=args.xcomet_model,
            bertscore_model=args.bertscore_model,
            runtime=runtime,
        )
    except RuntimeError as error:
        if "Install sacrebleu" not in str(error):
            raise
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p0.yaml")
    parser.add_argument("--split", choices=("dev", "devtest"), default="devtest")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--budget-usd", type=float, default=8.75)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N rows")
    parser.add_argument("--safety-factor", type=float, default=1.25, help="multiplier for cost reservations")
    parser.add_argument("--out", default="reports/openrouter")
    parser.add_argument("--comet-model", default=None, help="optional COMET checkpoint/model identifier")
    parser.add_argument("--cometkiwi-model", default=None, help="optional reference-free COMETKiwi model identifier")
    parser.add_argument("--xcomet-model", default=None, help="optional XCOMET model identifier")
    parser.add_argument("--bertscore-model", default=None, help="optional multilingual BERTScore model identifier")
    parser.add_argument("--all-metrics", action="store_true", help="enable standard neural metrics in addition to lexical metrics")
    parser.add_argument("--dry-run", action="store_true", help="fetch catalog and print cost estimates without translating")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.all_metrics:
        for name, default in DEFAULT_NEURAL_METRICS.items():
            if getattr(args, name) is None:
                setattr(args, name, default)
    if (
        args.budget_usd <= 0
        or args.max_tokens <= 0
        or args.limit < 0
        or args.safety_factor < 1
    ):
        raise SystemExit("budget and max-tokens must be positive, limit non-negative, and safety-factor at least 1")
    if not args.dry_run and not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("Set OPENROUTER_API_KEY before running translations")
    if not args.dry_run:
        try:
            import sacrebleu
        except ImportError as error:
            raise SystemExit("Install sacrebleu before running translations") from error

    config_path = ROOT / args.config
    if not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")
    try:
        data_path = ROOT / eval_path_from_config(config_path, args.split)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    pairs = list(iter_pairs(str(data_path)))
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit(f"evaluation set is empty: {data_path}")
    sources = [source for source, _ in pairs]
    references = [reference for _, reference in pairs]
    source_hashes = [hashlib.sha256(source.encode("utf-8")).hexdigest() for source in sources]

    client = OpenRouterClient()
    catalog = client.list_models()
    missing = [model for model in args.models if model not in catalog]
    if missing:
        raise SystemExit("models not present in current OpenRouter catalog: " + ", ".join(missing))
    estimates = {
        model_id: estimate_run_cost(
            catalog[model_id], sources, args.max_tokens, args.safety_factor
        )
        for model_id in args.models
    }
    unknown = [model for model, estimate in estimates.items() if estimate is None]
    if unknown:
        raise SystemExit("cannot estimate pricing for: " + ", ".join(unknown))
    projected = sum(estimates.values())
    print(json.dumps({"rows": len(sources), "projected_usd_with_25pct_buffer": projected, "estimates": estimates}, indent=2))
    if projected > args.budget_usd:
        raise SystemExit(
            f"projected cost ${projected:.4f} exceeds budget ${args.budget_usd:.2f}; no translations sent"
        )
    if args.dry_run:
        return 0

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(args.config)),
        "split": args.split,
        "dataset": str(data_path.relative_to(ROOT)),
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "source_count": len(sources),
        "models": args.models,
        "system_prompt": SYSTEM_PROMPT,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "budget_usd": args.budget_usd,
        "projected_cost_usd": projected,
        "safety_factor": args.safety_factor,
        "comet_model": args.comet_model,
        "cometkiwi_model": args.cometkiwi_model,
        "xcomet_model": args.xcomet_model,
        "bertscore_model": args.bertscore_model,
        "all_metrics": args.all_metrics,
        "openrouter_base_url": client.base_url,
        "catalog_prices": {
            model_id: {
                "name": catalog[model_id].name,
                "prompt": catalog[model_id].prompt_price,
                "completion": catalog[model_id].completion_price,
                "context_length": catalog[model_id].context_length,
                "supported_parameters": catalog[model_id].supported_parameters,
            }
            for model_id in args.models
        },
    }
    _write_json(manifest_path, manifest)
    total_spent = 0.0
    summary = []
    metric_runtime = NeuralMetricRuntime()

    for model_id in args.models:
        model_info = catalog[model_id]
        log_path = out / f"{model_id.replace('/', '__')}.jsonl"
        reasoning = reasoning_for(model_info)
        request_hashes = [
            request_hash(model_id, source, args.max_tokens, reasoning) for source in sources
        ]
        completed = load_completed(log_path, model_id, source_hashes, request_hashes)
        total_spent += sum(_record_cost(record, model_info) for record in completed.values())
        print(f"{model_id}: resuming {len(completed)}/{len(sources)} rows")
        with log_path.open("a", encoding="utf-8") as log:
            for index, (source, reference) in enumerate(pairs):
                if index in completed:
                    continue
                request_messages = messages_for(source)
                request_estimate = estimate_cost(model_info, request_messages, args.max_tokens)
                reserved_cost = request_estimate * args.safety_factor if request_estimate is not None else None
                if reserved_cost is None or (total_spent + reserved_cost) > args.budget_usd:
                    raise SystemExit(
                        f"budget guard stopped before {model_id} row {index}; "
                        f"spent/reserved ${total_spent:.4f}, next reserved request ${reserved_cost or 0:.4f}"
                    )
                started = time.perf_counter()
                try:
                    result = client.complete(
                        model_id,
                        request_messages,
                        temperature=0,
                        max_tokens=args.max_tokens,
                        reasoning=reasoning,
                    )
                except OpenRouterError as error:
                    failure = {
                        "index": index,
                        "model": model_id,
                        "source_sha256": source_hashes[index],
                        "status": "error",
                        "error": str(error),
                        "http_status": error.status,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    log.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    log.flush()
                    raise SystemExit(f"OpenRouter failed on {model_id} row {index}: {error}") from error
                elapsed = time.perf_counter() - started
                charged = usage_cost(result.usage, model_info)
                if charged is None:
                    charged = request_estimate or 0.0
                total_spent += charged
                record = {
                    "index": index,
                    "model": model_id,
                    "source": source,
                    "reference": reference,
                    "hypothesis": result.text,
                    "source_sha256": source_hashes[index],
                    "request_sha256": request_hashes[index],
                    "status": "ok",
                    "response_id": result.response_id,
                    "returned_model": result.model,
                    "provider": result.provider,
                    "usage": result.usage,
                    "cost_usd": charged,
                    "latency_seconds": elapsed,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                completed[index] = record
                print(f"{model_id} {index + 1}/{len(sources)} cost=${charged:.6f}", flush=True)

        ordered = [completed[index] for index in range(len(sources))]
        hypotheses = [record["hypothesis"] for record in ordered]
        _write_json(out / f"{model_id.replace('/', '__')}.translations.json", ordered)
        metrics = _metrics_if_available(
            hypotheses,
            references,
            sources,
            args,
            metric_runtime,
        )
        row = {
            "model": model_id,
            "provider": ordered[-1].get("provider") if ordered else None,
            "rows": len(ordered),
            "cost_usd": sum(_record_cost(record, model_info) for record in ordered),
            "mean_latency_seconds": sum(record["latency_seconds"] for record in ordered) / len(ordered),
            "metrics": metrics,
        }
        summary.append(row)
        _write_summary(out / "summary", summary)
        print(json.dumps(row, ensure_ascii=False))

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["spent_usd"] = total_spent
    _write_json(manifest_path, manifest)
    _write_summary(out / "summary", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
