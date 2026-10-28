"""Interactively translate Kazakh text with a trained KK->RU checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from kk_ru.config import load_config
from kk_ru.eval import translate
from kk_ru.model import build_model
from kk_ru.tokenizer import load_tokenizer, validate_vocab


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive KK->RU translation")
    parser.add_argument("--checkpoint", required=True, help="checkpoint directory or model.pt")
    parser.add_argument("--config", default="configs/p0.yaml")
    parser.add_argument("--beam", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    target_device = device()
    tokenizer = load_tokenizer(cfg.tokenizer.sp_model, max_len=cfg.model.max_len)
    validate_vocab(tokenizer, cfg.model.vocab)

    model = build_model(cfg.model)
    checkpoint = Path(args.checkpoint)
    state_path = checkpoint / "model.pt" if checkpoint.is_dir() else checkpoint
    model.load_state_dict(torch.load(state_path, map_location=target_device, weights_only=True))
    model.to(target_device).eval()

    beam = args.beam or cfg.gen.beam
    max_len = args.max_len or cfg.gen.max_len
    print(f"loaded {state_path} on {target_device}; beam={beam}; max_len={max_len}")
    print("Введите казахский текст. Ctrl-D для выхода.")
    try:
        while True:
            try:
                text = input("kk> ").strip()
            except EOFError:
                print()
                break
            if not text:
                continue
            result = translate(
                model,
                tokenizer,
                [text],
                target_device,
                beam=beam,
                max_len=max_len,
                source_max_len=cfg.model.max_len,
                batch_size=1,
            )[0]
            print(f"ru> {result}")
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
