#!/usr/bin/env python3
"""Evaluate configured experiment checkpoints and write one comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from kk_ru.config import load_config
from kk_ru.eval import evaluate
from kk_ru.model import build_model, count_params
from kk_ru.tokenizer import load_tokenizer, validate_vocab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=[
        "configs/small.yaml", "configs/p0.yaml", "configs/large.yaml"
    ])
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--split", choices=["dev", "devtest"], default="devtest")
    parser.add_argument("--out", default="reports/benchmark")
    parser.add_argument("--spbleu", action="store_true")
    return parser.parse_args()


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    if args.checkpoints is not None and len(args.checkpoints) != len(args.configs):
        raise SystemExit("--checkpoints must have the same length as --configs")
    checkpoints = args.checkpoints or [
        str(Path(load_config(path).paths.checkpoints) / "best" / "model.pt")
        for path in args.configs
    ]
    device = _device()
    rows = []
    for config_path, checkpoint_path in zip(args.configs, checkpoints):
        cfg = load_config(config_path)
        tokenizer = load_tokenizer(cfg.tokenizer.sp_model, max_len=cfg.model.max_len)
        validate_vocab(tokenizer, cfg.model.vocab)
        model = build_model(cfg.model)
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        metrics = evaluate(
            cfg, tokenizer, model, args.split, device,
            write=True, compute_spbleu=args.spbleu,
        )
        row = {
            "model": Path(config_path).stem,
            "config": config_path,
            "checkpoint": checkpoint_path,
            "params": count_params(model),
            **metrics,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.with_suffix(".json")).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fields = sorted({key for row in rows for key in row})
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
