#!/usr/bin/env python3
"""Evaluate the original Qwen3.5-0.8B weights on KK->RU FLORES+ with MLX."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from kk_ru.data import iter_pairs
from kk_ru.eval import compute_metrics

MODEL_ID = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
SYSTEM_PROMPT = (
    "Translate the user's text from Kazakh into Russian. "
    "Output only the Russian translation, without explanations or additional text."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3.5-0.8B"))
    parser.add_argument("--split", choices=["dev", "devtest"], default="devtest")
    parser.add_argument("--out", type=Path, default=Path("reports/qwen3.5-0.8b"))
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, help="Evaluate a prefix for a smoke test only")
    parser.add_argument("--spbleu", action="store_true")
    args = parser.parse_args()
    if args.max_tokens <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--max-tokens and --limit must be positive")

    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx.utils import tree_flatten
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    from sacrebleu.metrics import BLEU, CHRF, TER

    input_path = Path("data/eval/flores_plus") / f"{args.split}.kk-ru.tsv"
    pairs = list(iter_pairs(str(input_path)))
    if args.limit is not None:
        pairs = pairs[:args.limit]
    if not pairs:
        raise ValueError(f"No evaluation pairs in {input_path}")
    args.out.mkdir(parents=True, exist_ok=True)
    records_path = args.out / f"{args.split}.jsonl"
    if records_path.exists():
        raise FileExistsError(f"Choose a new --out directory; {records_path} already exists")

    snapshot_download(MODEL_ID, revision=REVISION, local_dir=args.model)
    model, tokenizer = load(str(args.model))
    parameters = tree_flatten(model.parameters())
    manifest = {
        "model": MODEL_ID,
        "revision": REVISION,
        "model_path": str(args.model.resolve()),
        "weight_sha256": {
            path.name: sha256(path) for path in sorted(args.model.glob("*.safetensors"))
        },
        "parameters_text_model": sum(value.size for _, value in parameters),
        "dtypes": sorted({str(value.dtype) for _, value in parameters}),
        "quantized": False,
        "backend": "mlx",
        "device": str(mx.default_device()),
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("mlx", "mlx-lm", "transformers", "sacrebleu")
        },
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "dataset": str(input_path),
        "dataset_sha256": sha256(input_path),
        "rows": len(pairs),
        "limit": args.limit,
        "system_prompt": SYSTEM_PROMPT,
        "enable_thinking": False,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "batch_size": 1,
        "output_normalization": "collapse whitespace only; raw output saved per record",
    }
    (args.out / f"{args.split}.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sampler = make_sampler(temp=0.0)
    records = []
    started = time.perf_counter()
    with records_path.open("x", encoding="utf-8") as file:
        for index, (source, reference) in enumerate(pairs):
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": source},
                ],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            tick = time.perf_counter()
            segments = []
            for response in stream_generate(
                model, tokenizer, prompt, max_tokens=args.max_tokens, sampler=sampler
            ):
                segments.append(response.text)
            raw = "".join(segments)
            record = {
                "index": index,
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "source": source,
                "reference": reference,
                "hypothesis": " ".join(raw.split()),
                "raw_output": raw,
                "latency_sec": time.perf_counter() - tick,
                "prompt_tokens": response.prompt_tokens,
                "generation_tokens": response.generation_tokens,
                "finish_reason": response.finish_reason,
            }
            records.append(record)
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(pairs):
                print(
                    f"{index + 1}/{len(pairs)} | {time.perf_counter() - started:.1f}s | "
                    f"{record['hypothesis'][:140]}", flush=True,
                )
    elapsed = time.perf_counter() - started
    hypotheses = [record["hypothesis"] for record in records]
    references = [reference for _, reference in pairs]
    sources = [source for source, _ in pairs]
    metrics = compute_metrics(hypotheses, references, sources, compute_spbleu=args.spbleu)
    scorers = {"bleu": BLEU(), "chrf": CHRF(), "chrf++": CHRF(word_order=2), "ter": TER()}
    signatures = {}
    for name, scorer in scorers.items():
        scorer.corpus_score(hypotheses, [references])
        signatures[name] = str(scorer.get_signature())
    metrics.update({
        "model": MODEL_ID,
        "revision": REVISION,
        "split": args.split,
        "dataset_sha256": manifest["dataset_sha256"],
        "rows": len(pairs),
        "elapsed_sec": elapsed,
        "sentences_per_second": len(pairs) / elapsed,
        "latency_p50_sec": statistics.median(record["latency_sec"] for record in records),
        "generated_tokens": sum(record["generation_tokens"] for record in records),
        "truncated_outputs": sum(record["finish_reason"] == "length" for record in records),
        "empty_outputs": sum(not hypothesis for hypothesis in hypotheses),
        "copied_sources": sum(hypothesis == source for hypothesis, source in zip(hypotheses, sources)),
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "signatures": signatures,
    })
    (args.out / f"{args.split}.hyp.txt").write_text("\n".join(hypotheses) + "\n", encoding="utf-8")
    (args.out / f"{args.split}.metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
