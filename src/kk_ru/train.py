"""Обучение P0. Спецификация — TASKS.md T7, PLAN.md §8.

Запуск:
    python -m kk_ru.train --config configs/p0.yaml
    accelerate launch -m kk_ru.train --config configs/p0.yaml           # DDP
    accelerate launch -m kk_ru.train --config configs/p0.yaml --overfit # гейт 10k

Контракт (T7):
- HuggingFace Accelerate DDP, bf16 + SDPA;
- AdamW (lr=2e-4, weight_decay=0.01), cosine + warmup 2000, min_lr=2e-5;
- label smoothing 0.1, grad clip 1.0, grad-accum (global batch 256);
- чекпоинты каждые save_every + best по FLORES dev, **resume**;
- в каталог чекпоинта класть копию конфига и логировать полный конфиг на старте (воспроизводимость);
- логирование loss/lr/шаг; set_seed(seed).
- ``--overfit``: 10k пар (фикс. seed), маленький батч, гнать до loss ~0.
"""
from __future__ import annotations

import argparse

from .config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KK→RU model.")
    parser.add_argument("--config", type=str, default="configs/p0.yaml")
    parser.add_argument("--opts", nargs="*", default=None, help="dotted overrides, e.g. train.epochs=5")
    parser.add_argument("--overfit", action="store_true", help="overfit 10k subset (gate)")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint dir to resume")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.opts)
    # TODO(T7): set_seed -> tokenizer -> model -> dataloaders -> Accelerate -> train loop.
    raise NotImplementedError("T7: реализовать цикл обучения")


if __name__ == "__main__":
    main()
