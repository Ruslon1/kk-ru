#!/usr/bin/env python3
"""Evaluate a base Qwen model or a PEFT adapter on KK-RU FLORES data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sacrebleu
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "Translate Kazakh to Russian. Return only the Russian translation. "
    "Do not explain or add commentary."
)


def read_pairs(path: Path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            source, separator, target = line.rstrip("\n").partition("\t")
            if separator and source and target:
                yield source, target


def clean(text: str) -> str:
    return text.rsplit("</think>", 1)[-1].replace("<|im_end|>", "").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--split", choices=["dev", "devtest"], default="devtest")
    parser.add_argument("--data-root", default="data/eval/flores_plus")
    parser.add_argument("--output", default="reports/qwen3-0.6b-hf")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    rows = list(read_pairs(Path(args.data_root) / f"{args.split}.kk-ru.tsv"))
    hypotheses = []
    latencies = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": source},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for source, _ in batch
        ]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        latencies.append(time.perf_counter() - started)
        prompt_length = encoded["input_ids"].shape[1]
        for index, output in enumerate(generated):
            hypotheses.append(
                clean(tokenizer.decode(output[prompt_length:], skip_special_tokens=True))
            )

    references = [target for _, target in rows]
    metrics = {
        "model": args.model,
        "adapter": args.adapter,
        "split": args.split,
        "rows": len(rows),
        "bleu": sacrebleu.corpus_bleu(hypotheses, [references]).score,
        "chrf": sacrebleu.corpus_chrf(hypotheses, [references]).score,
        "chrf++": sacrebleu.corpus_chrf(
            hypotheses, [references], word_order=2
        ).score,
        "batch_latency_p50_sec": sorted(latencies)[len(latencies) // 2],
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{args.split}.metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / f"{args.split}.jsonl").write_text(
        "".join(
            json.dumps(
                {"source": source, "reference": reference, "hypothesis": hypothesis},
                ensure_ascii=False,
            )
            + "\n"
            for (source, reference), hypothesis in zip(rows, hypotheses)
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
