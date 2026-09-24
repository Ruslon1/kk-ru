from __future__ import annotations

import argparse
import math
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KK→RU model.")
    parser.add_argument("--config", type=str, default="configs/p0.yaml")
    parser.add_argument("--opts", nargs="*", default=None, help="dotted overrides, e.g. train.epochs=5")
    parser.add_argument("--overfit", action="store_true", help="overfit 10k subset (gate)")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint dir to resume")
    return parser.parse_args()


def build_optimizer(model, cfg):
    decay, no_decay = [], []
    for p in model.parameters():
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))


def build_scheduler(optimizer, cfg, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        return cfg.min_lr / cfg.lr + (1.0 - cfg.min_lr / cfg.lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.opts)
    set_seed(cfg.train.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.grad_accum,
        mixed_precision=cfg.train.precision,
    )

    tokenizer = load_tokenizer(cfg.tokenizer.sp_model, max_len=cfg.model.max_len)
    train_limit = OVERFIT_LIMIT if args.overfit else 0
    loaders = build_dataloaders(
        cfg, tokenizer, rank=accelerator.process_index, world_size=accelerator.num_processes, train_limit=train_limit
    )
    train_loader = loaders["train"]

    model = build_model(cfg.model)
    if args.resume:
        state = torch.load(Path(args.resume) / "model.pt", map_location="cpu")
        model.load_state_dict(state)

    accelerator.print(f"model params: {count_params(model):,}")
    accelerator.print(f"train batches/rank: {len(train_loader)} | grad_accum: {cfg.train.grad_accum}")

    optimizer = build_optimizer(model, cfg.train)
    model, optimizer = accelerator.prepare(model, optimizer)

    steps_per_epoch = (len(train_loader) + cfg.train.grad_accum - 1) // cfg.train.grad_accum
    total_steps = cfg.train.max_steps or (steps_per_epoch * cfg.train.epochs)
    scheduler = build_scheduler(optimizer, cfg.train, total_steps)
    accelerator.print(f"total optimizer steps: {total_steps}")

    save_dir = Path(cfg.paths.checkpoints)
    save_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    model.train()

    for epoch in range(cfg.train.epochs):
        for batch in train_loader:
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            with accelerator.accumulate(model):
                logits = model(batch["src_ids"], batch["src_mask"], batch["tgt_ids"][:, :-1])
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
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % 50 == 0:
                        accelerator.print(
                            f"epoch {epoch} step {global_step}/{total_steps} "
                            f"loss {loss.item():.4f} lr {scheduler.get_last_lr()[0]:.2e}"
                        )

                    if global_step % cfg.train.save_every == 0:
                        if accelerator.is_main_process:
                            unwrapped = accelerator.unwrap_model(model)
                            torch.save(unwrapped.state_dict(), save_dir / f"step-{global_step}.pt")
                            accelerator.print(f"saved step-{global_step}.pt")

                    if cfg.train.eval_every and global_step % cfg.train.eval_every == 0:
                        if accelerator.is_main_process:
                            raw = accelerator.unwrap_model(model)
                            raw.eval()
                            metrics = evaluate(cfg, tokenizer, raw, "dev", accelerator.device)
                            raw.train()
                            accelerator.print(
                                f"dev @ step {global_step}: "
                                f"bleu {metrics['bleu']:.2f} chrf {metrics['chrf']:.2f}"
                            )
                        accelerator.wait_for_everyone()

                    if global_step >= total_steps:
                        break
        if global_step >= total_steps:
            break

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.state_dict(), save_dir / "model.pt")
        accelerator.print(f"done. final model at {save_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
