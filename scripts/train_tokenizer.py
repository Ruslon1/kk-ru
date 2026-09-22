#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a shared KK+RU SentencePiece tokenizer.")
    p.add_argument("--input", type=str, default="data/filtered/train.kk-ru.tsv")
    p.add_argument("--out", type=str, default="data/tokenizer/kk-ru-sp32k")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--model-type", type=str, default="unigram", choices=["unigram", "bpe"])
    p.add_argument("--character-coverage", type=float, default=0.9995)
    p.add_argument("--input-sentence-size", type=int, default=5_000_000)
    p.add_argument("--max-lines", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    corpus = out.with_name(out.name + ".corpus")

    n = 0
    with open(ROOT / args.input, encoding="utf-8") as fin, corpus.open("w", encoding="utf-8") as f:
        for line in fin:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            f.write(parts[0] + "\n")
            f.write(parts[1] + "\n")
            n += 1
            if args.max_lines and n >= args.max_lines:
                break
    print(f"wrote {n:,} pairs (kk+ru interleaved) to {corpus.name}")

    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(out),
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        pad_id=3,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_piece="<pad>",
        normalization_rule_name="nmt_nfkc",
        add_dummy_prefix=True,
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=True,
        seed_sentencepiece_size=1_000_000,
        num_threads=min(os.cpu_count() or 4, 16),
        hard_vocab_limit=True,
    )

    corpus.unlink(missing_ok=True)
    print(f"wrote {out}.model / {out}.vocab")


if __name__ == "__main__":
    main()
