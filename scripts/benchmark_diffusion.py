from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from kk_ru.config import load_config
from kk_ru.data import iter_pairs
from kk_ru.diffusion_eval import evaluate_diffusion
from kk_ru.diffusion_model import build_diffusion_model, count_params
from kk_ru.eval import compute_metrics, translate
from kk_ru.model import build_model
from kk_ru.tokenizer import load_tokenizer, validate_vocab
from kk_ru.utils import get_device


def load_state(model, checkpoint: str, device: torch.device) -> None:
    path = Path(checkpoint)
    if path.is_dir():
        path = path / "model.pt"
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare diffusion steps with the AR baseline.")
    parser.add_argument("--config", default="configs/diffusion.yaml")
    parser.add_argument("--checkpoint", required=True, help="diffusion checkpoint")
    parser.add_argument("--baseline-config", default="configs/p0.yaml")
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--split", choices=("dev", "devtest"), default="dev")
    parser.add_argument("--steps", nargs="+", type=int, default=(8, 16, 32))
    parser.add_argument("--out", default="reports/diffusion/benchmark-dev")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    baseline_cfg = load_config(args.baseline_config)
    if cfg.diffusion is None:
        raise ValueError("diffusion config section is required")
    device = get_device()

    diffusion_tokenizer = load_tokenizer(cfg.tokenizer.sp_model, cfg.model.max_len)
    validate_vocab(diffusion_tokenizer, cfg.model.vocab)
    diffusion_model = build_diffusion_model(cfg.model, cfg.diffusion.steps)
    load_state(diffusion_model, args.checkpoint, device)

    baseline_tokenizer = load_tokenizer(
        baseline_cfg.tokenizer.sp_model, baseline_cfg.model.max_len
    )
    validate_vocab(baseline_tokenizer, baseline_cfg.model.vocab)
    baseline_model = build_model(baseline_cfg.model)
    load_state(baseline_model, args.baseline_checkpoint, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    eval_path = {
        "dev": baseline_cfg.data.eval_dev,
        "devtest": baseline_cfg.data.eval_test,
    }[args.split]
    baseline_rows = list(iter_pairs(eval_path))
    baseline_sources = [source for source, _ in baseline_rows]
    baseline_references = [target for _, target in baseline_rows]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    baseline_hypotheses = translate(
        baseline_model,
        baseline_tokenizer,
        baseline_sources,
        device,
        baseline_cfg.gen.beam,
        baseline_cfg.gen.max_len,
        baseline_cfg.model.max_len,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    baseline_elapsed = time.perf_counter() - started
    baseline_metrics = compute_metrics(baseline_hypotheses, baseline_references)
    rows = [{
        "model": "autoregressive",
        "params": count_params(baseline_model),
        "steps": 1,
        "generation_seconds": baseline_elapsed,
        "seconds_per_sentence": baseline_elapsed / max(len(baseline_sources), 1),
        **baseline_metrics,
    }]

    for steps in args.steps:
        metrics, _ = evaluate_diffusion(
            cfg,
            diffusion_tokenizer,
            diffusion_model,
            args.split,
            device,
            steps=steps,
        )
        rows.append({
            "model": "discrete-diffusion",
            "params": count_params(diffusion_model),
            "steps": steps,
            **metrics,
        })
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fields = sorted({key for row in rows for key in row})
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
