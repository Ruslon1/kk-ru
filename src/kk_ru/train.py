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

    raise NotImplementedError("T7: реализовать цикл обучения")


if __name__ == "__main__":
    main()
