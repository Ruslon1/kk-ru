#!/usr/bin/env python3
"""Фильтр корпуса KK→RU. Спецификация — TASKS.md T4, PLAN.md §7.

Порядок (именно так, как у kazRush):
  1. дедуп (точные пары kk+ru);
  2. чистка (HTML/теги, пробелы/юникод, пустые/короткие, kk == ru);
  3. langid: facebook/fasttext-language-identification (kk->kk, ru->ru);
  4. LaBSE: sentence-transformers/LaBSE, cosine(kk, ru) >= threshold (параметр).

Вход: data/raw/{opus,wmt19_crawl,kazparc}.kk-ru.tsv (+ til при появлении).
Выход: data/filtered/train.kk-ru.tsv. Каждый шаг печатает «было → стало».
FLORES+ (data/eval/**) в train НЕ класть; проверить отсутствие точного пересечения.
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter raw KK-RU bitext.")
    p.add_argument("--in", dest="inputs", nargs="+", required=True)
    p.add_argument("--out", type=str, default="data/filtered/train.kk-ru.tsv")
    p.add_argument("--labse-threshold", type=float, default=0.55)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError("T4: реализовать фильтр корпуса")


if __name__ == "__main__":
    main()
