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

    raise NotImplementedError("evaluation")


if __name__ == "__main__":
    main()
