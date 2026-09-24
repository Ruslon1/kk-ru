from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed

from .config import load_config
from .data import build_dataloaders
from .eval import evaluate
from .model import build_model, count_params
from .tokenizer import load_tokenizer

OVERFIT_LIMIT = 10_000
LOG_EVERY = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KK→RU model.")
    parser.add_argument("--config", default="configs/p0.yaml")
    parser.add_argument("--opts", nargs="*", default=None, help="dotted overrides, e.g. train.epochs=5")
    parser.add_argument("--overfit", action="store_true", help="train on a 10k-example subset")
    parser.add_argument("--resume", help="Accelerate checkpoint directory")
    return parser.parse_args()


def build_optimizer(model, cfg):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr,
        betas=(0.9, 0.95),
    )


def build_scheduler(optimizer, cfg, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = min(1.0, (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps))
        return cfg.min_lr / cfg.lr + (1.0 - cfg.min_lr / cfg.lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _checkpoint_path(root: Path, step: int) -> Path:
    return root / f"step-{step}"


def _save_checkpoint(accelerator: Accelerator, model, root: Path, step: int) -> None:
    checkpoint = _checkpoint_path(root, step)
    accelerator.save_state(str(checkpoint))
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.state_dict(), checkpoint / "model.pt")
        (checkpoint / "meta.json").write_text(json.dumps({"step": step}), encoding="utf-8")
    accelerator.wait_for_everyone()


def _save_export(accelerator: Accelerator, model, root: Path, name: str, step: int, metrics: dict) -> None:
    if not accelerator.is_main_process:
        return
    export_dir = root / name
    export_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    if hasattr(unwrapped, "_orig_mod"):
        unwrapped = unwrapped._orig_mod
    torch.save(unwrapped.state_dict(), export_dir / "model.pt")
    (export_dir / "meta.json").write_text(
        json.dumps({"step": step, "metrics": metrics}, indent=2), encoding="utf-8"
    )


def _prune_checkpoints(root: Path, keep: int = 3) -> None:
    checkpoints = sorted(
        (path for path in root.glob("step-*") if path.is_dir()),
        key=lambda path: int(path.name.removeprefix("step-")),
    )
    for checkpoint in checkpoints[:-keep]:
        shutil.rmtree(checkpoint)


def _load_step(checkpoint: str | None) -> int:
    if checkpoint is None:
        return 0
    metadata = Path(checkpoint) / "meta.json"
    if not metadata.exists():
        raise FileNotFoundError(f"checkpoint metadata is missing: {metadata}")
    return int(json.loads(metadata.read_text(encoding="utf-8"))["step"])


def _set_loader_epoch(loader, epoch: int) -> None:
    sampler = getattr(loader, "batch_sampler", None)
    sampler = getattr(sampler, "batch_sampler", sampler)
    sampler = getattr(sampler, "sampler", sampler)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.opts)
    set_seed(cfg.train.seed)
    precision = cfg.train.precision if isinstance(cfg.train.precision, str) else "no"
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.grad_accum,
        mixed_precision=precision,
        log_with="tensorboard",
    )
    accelerator.init_trackers("kk-ru", config={"config": str(args.config), "seed": cfg.train.seed})

    tokenizer = load_tokenizer(cfg.tokenizer.sp_model, max_len=cfg.model.max_len)
    with accelerator.main_process_first():
        loaders = build_dataloaders(
            cfg, tokenizer, train_limit=OVERFIT_LIMIT if args.overfit else 0
        )
    train_loader = loaders["train"]
    model = build_model(cfg.model)
    if cfg.train.compile:
        model = torch.compile(model)
    optimizer = build_optimizer(model, cfg.train)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg.train.grad_accum))
    total_steps = cfg.train.max_steps or steps_per_epoch * cfg.train.epochs
    scheduler = build_scheduler(optimizer, cfg.train, total_steps)
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)

    start_step = _load_step(args.resume)
    if args.resume:
        accelerator.load_state(args.resume)

    accelerator.print(f"model params: {count_params(model):,}")
    accelerator.print(f"train batches/rank: {len(train_loader)} | total steps: {total_steps}")
    optimizer.zero_grad(set_to_none=True)
    global_step = start_step
    model.train()
    save_root = Path(cfg.paths.checkpoints)
    save_root.mkdir(parents=True, exist_ok=True)
    best_spbleu = -float("inf")

    for epoch in range(cfg.train.epochs):
        _set_loader_epoch(train_loader, epoch)
        for batch in train_loader:
            if global_step >= total_steps:
                break
            batch = {
                key: value.to(accelerator.device, non_blocking=True)
                for key, value in batch.items()
            }
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    logits = model(
                        batch["src_ids"],
                        batch["src_mask"],
                        batch["tgt_ids"][:, :-1],
                        tgt_mask=batch["tgt_mask"][:, :-1],
                    )
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        batch["labels"].reshape(-1),
                        ignore_index=tokenizer.pad_id,
                        label_smoothing=cfg.train.label_smoothing,
                    )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.train.clip_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    learning_rate = scheduler.get_last_lr()[0]
                    accelerator.log(
                        {"loss": loss.detach().float().item(), "learning_rate": learning_rate},
                        step=global_step,
                    )
                    if global_step % LOG_EVERY == 0:
                        accelerator.print(
                            f"epoch {epoch} step {global_step}/{total_steps} "
                            f"loss {loss.item():.4f} lr {learning_rate:.2e}"
                        )
                    if cfg.train.save_every and global_step % cfg.train.save_every == 0:
                        _save_checkpoint(accelerator, model, save_root, global_step)
                        if accelerator.is_main_process:
                            _prune_checkpoints(save_root)
                    if cfg.train.eval_every and global_step % cfg.train.eval_every == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process and "dev" in loaders:
                            raw = accelerator.unwrap_model(model)
                            metrics = evaluate(cfg, tokenizer, raw, "dev", accelerator.device)
                            raw.train()
                            accelerator.log({f"dev/{name}": value for name, value in metrics.items()}, step=global_step)
                            _save_export(accelerator, model, save_root, "latest", global_step, metrics)
                            if metrics["spbleu"] > best_spbleu:
                                best_spbleu = metrics["spbleu"]
                                _save_export(accelerator, model, save_root, "best", global_step, metrics)
                            accelerator.print(
                                f"dev @ step {global_step}: bleu {metrics['bleu']:.2f} chrf {metrics['chrf']:.2f}"
                            )
                        accelerator.wait_for_everyone()
        if global_step >= total_steps:
            break

    _save_checkpoint(accelerator, model, save_root, global_step)
    if accelerator.is_main_process:
        accelerator.print(f"done. final checkpoint at {_checkpoint_path(save_root, global_step)}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
