from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .config import Config, load_config
from .data import iter_pairs
from .diffusion_model import build_diffusion_model, count_params
from .diffusion_ops import reveal_tokens
from .eval import compute_metrics
from .tokenizer import load_tokenizer, validate_vocab
from .utils import get_device


@torch.inference_mode()
def translate_diffusion(
    model,
    tokenizer,
    sources: list[str],
    device: torch.device,
    max_len: int,
    steps: int,
    batch_size: int = 16,
) -> tuple[list[str], float]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    model.eval()
    hypotheses: list[str] = []
    started = time.perf_counter()
    for start in range(0, len(sources), batch_size):
        chunk = sources[start : start + batch_size]
        encoded = [tokenizer.encode(text, add_bos=True)[:max_len] for text in chunk]
        width = max(map(len, encoded))
        src_ids = torch.full(
            (len(chunk), width), tokenizer.pad_id, dtype=torch.long, device=device
        )
        src_mask = torch.zeros_like(src_ids, dtype=torch.bool)
        for row, ids in enumerate(encoded):
            src_ids[row, : len(ids)] = torch.tensor(ids, device=device)
            src_mask[row, : len(ids)] = True

        memory = model.encode(src_ids, src_mask)
        lengths = (model.predict_length(memory, src_mask).argmax(dim=-1) + 1)
        lengths = lengths.clamp(min=2, max=max_len)
        target_width = int(lengths.max().item())
        positions = torch.arange(target_width, device=device).unsqueeze(0)
        target_mask = positions < lengths.unsqueeze(1)
        noisy = torch.full(
            (len(chunk), target_width), tokenizer.pad_id, dtype=torch.long, device=device
        )
        noisy[target_mask] = model.mask_id
        for timestep in range(steps, 0, -1):
            timestep_ids = torch.full(
                (len(chunk),), timestep, dtype=torch.long, device=device
            )
            logits = model.denoise(
                memory,
                src_mask,
                noisy,
                timestep_ids,
                target_mask,
                total_steps=steps,
            )
            noisy = reveal_tokens(
                noisy, logits, target_mask, model.mask_id, timestep
            )

        for row, length in enumerate(lengths.tolist()):
            ids = noisy[row, :length].tolist()
            if tokenizer.eos_id in ids:
                ids = ids[: ids.index(tokenizer.eos_id) + 1]
            hypotheses.append(tokenizer.decode(ids, skip_special=True).strip())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return hypotheses, time.perf_counter() - started


def evaluate_diffusion(
    cfg: Config,
    tokenizer,
    model,
    split: str,
    device: torch.device,
    steps: int | None = None,
    batch_size: int = 16,
) -> tuple[dict, list[str]]:
    paths = {"dev": cfg.data.eval_dev, "devtest": cfg.data.eval_test}
    if split not in paths:
        raise ValueError(f"unknown evaluation split: {split}")
    rows = list(iter_pairs(paths[split]))
    sources = [source for source, _ in rows]
    references = [target for _, target in rows]
    hypotheses, elapsed = translate_diffusion(
        model,
        tokenizer,
        sources,
        device,
        cfg.model.max_len,
        steps or cfg.diffusion.steps,
        batch_size,
    )
    metrics = compute_metrics(hypotheses, references)
    metrics["generation_seconds"] = elapsed
    metrics["seconds_per_sentence"] = elapsed / max(len(sources), 1)
    metrics["denoising_steps"] = steps or cfg.diffusion.steps
    metrics["parameters"] = count_params(model)
    return metrics, hypotheses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a discrete diffusion translator.")
    parser.add_argument("--config", default="configs/diffusion.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("dev", "devtest"), default="devtest")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if cfg.diffusion is None:
        raise ValueError("diffusion config section is required")
    device = get_device()
    tokenizer = load_tokenizer(cfg.tokenizer.sp_model, max_len=cfg.model.max_len)
    validate_vocab(tokenizer, cfg.model.vocab)
    model = build_diffusion_model(cfg.model, cfg.diffusion.steps).to(device)
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    metrics, hypotheses = evaluate_diffusion(
        cfg, tokenizer, model, args.split, device, args.steps, args.batch_size
    )
    output = Path(cfg.paths.reports) / f"diffusion-{args.split}-t{metrics['denoising_steps']}"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".hyp.txt").write_text("\n".join(hypotheses), encoding="utf-8")
    output.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
