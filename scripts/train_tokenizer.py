#!/usr/bin/env python3
"""Обучение SentencePiece 32k (shared KK+RU). Спецификация — TASKS.md T5.

Вход: data/filtered/train.kk-ru.tsv (обе колонки).
Выход: data/tokenizer/kk-ru-sp32k.model + .vocab.
Спецтокены: <pad> <unk> <bos> <eos>. character_coverage вынести в параметр (старт 0.9995).
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SentencePiece tokenizer.")
    p.add_argument("--input", type=str, default="data/filtered/train.kk-ru.tsv")
    p.add_argument("--out", type=str, default="data/tokenizer/kk-ru-sp32k")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--model-type", type=str, default="unigram", choices=["unigram", "bpe"])
    p.add_argument("--character-coverage", type=float, default=0.9995)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError("T5: реализовать обучение SentencePiece")


if __name__ == "__main__":
    main()
