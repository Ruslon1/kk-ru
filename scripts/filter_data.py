#!/usr/bin/env python3
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
