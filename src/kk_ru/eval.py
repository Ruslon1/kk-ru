"""Оценка на FLORES+ (BLEU / chrF / COMET). Спецификация — TASKS.md T8.

Запуск:
    python -m kk_ru.eval --config configs/p0.yaml --checkpoint checkpoints/best --split devtest

Контракт (T8):
- beam 5, max_len 256;
- sacrebleu -> BLEU и chrF (сигнатуры совместимые с kazRush, FLORES+ kk→ru);
- unbabel-comet -> COMET (wmt22-comet-da или та же модель, что у kazRush — уточнить);
- вывод таблицы BLEU/chrF/COMET по dev и devtest.
"""
from __future__ import annotations

import argparse

from .config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KK→RU model on FLORES+.")
    parser.add_argument("--config", type=str, default="configs/p0.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to model checkpoint")
    parser.add_argument("--split", type=str, default="devtest", choices=["dev", "devtest"])
    parser.add_argument("--opts", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.opts)
    # TODO(T8): load checkpoint -> translate split -> sacrebleu BLEU/chrF + COMET.
    raise NotImplementedError("T8: реализовать оценку")


if __name__ == "__main__":
    main()
