#!/usr/bin/env python3
"""Validate the local training environment before a long experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from kk_ru.config import load_config
from kk_ru.model import build_model, count_params
from kk_ru.tokenizer import load_tokenizer, validate_vocab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", dest="configs")
    parser.add_argument("--min-gpus", type=int, default=1)
    args = parser.parse_args()
    configs = args.configs or [
        "configs/small.yaml", "configs/p0.yaml", "configs/large.yaml"
    ]

    gpu_count = torch.cuda.device_count()
    if gpu_count < args.min_gpus:
        raise SystemExit(f"expected at least {args.min_gpus} CUDA GPUs, found {gpu_count}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda} gpus={gpu_count}")
    for index in range(gpu_count):
        print(f"gpu[{index}]={torch.cuda.get_device_name(index)}")

    for config_path in configs:
        cfg = load_config(config_path)
        tokenizer_path = Path(cfg.tokenizer.sp_model)
        paths = (tokenizer_path, Path(cfg.data.train_tsv), Path(cfg.data.eval_dev), Path(cfg.data.eval_test))
        for path in paths:
            if not path.exists():
                raise SystemExit(f"missing required path: {path}")
        tokenizer = load_tokenizer(str(tokenizer_path), max_len=cfg.model.max_len)
        validate_vocab(tokenizer, cfg.model.vocab)
        print(f"{config_path}: params={count_params(build_model(cfg.model)):,} vocab={tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
