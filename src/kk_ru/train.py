from __future__ import annotations

import argparse
import dataclasses
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

from .config import Config, load_config
from .data import build_dataloaders
from .eval import evaluate
from .model import build_model, count_params
from .tokenizer import load_tokenizer, validate_vocab

OVERFIT_LIMIT = 10_000
LOG_EVERY = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KK→RU model.")
    parser.add_argument("--config", default="configs/p0.yaml")
    parser.add_argument("--opts", nargs="*", default=None, help="dotted overrides, e.g. train.epochs=5")
    parser.add_argument("--overfit", action="store_true", help="train on a 10k-example subset")
    parser.add_argument("--resume", help="Accelerate checkpoint directory")
    return parser.parse_args()


def create_accelerator(cfg: Config, config_path: str) -> Accelerator:
    set_seed(cfg.train.seed)
    precision = cfg.train.precision if isinstance(cfg.train.precision, str) else "no"
    project_config = ProjectConfiguration(
        project_dir=cfg.paths.logs,
        logging_dir=cfg.paths.logs,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.grad_accum,
        mixed_precision=precision,
        log_with="tensorboard",
        project_config=project_config,
    )
    accelerator.init_trackers(
        "kk-ru", config={"config": config_path, "seed": cfg.train.seed}
    )
    return accelerator


def create_tokenizer(cfg: Config):
    tokenizer = load_tokenizer(cfg.tokenizer.sp_model, max_len=cfg.model.max_len)
    validate_vocab(tokenizer, cfg.model.vocab)
    return tokenizer


def build_optimizer(model, cfg):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.95),
    )


def build_scheduler(optimizer, cfg, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = min(
            1.0,
            (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps),
        )
        return cfg.min_lr / cfg.lr + (1.0 - cfg.min_lr / cfg.lr) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def prepare_training(accelerator, model, optimizer, train_loader, cfg):
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg.train.grad_accum))
    total_steps = cfg.train.max_steps or steps_per_epoch * cfg.train.epochs
    scheduler = build_scheduler(optimizer, cfg.train, total_steps)
    scheduler = accelerator.prepare(scheduler)
    return model, optimizer, train_loader, scheduler, total_steps


def _checkpoint_path(root: Path, step: int) -> Path:
    return root / f"step-{step}"


def _save_checkpoint(
    accelerator: Accelerator,
    model,
    root: Path,
    step: int,
    epoch: int,
    batch_index: int,
) -> None:
    checkpoint = _checkpoint_path(root, step)
    accelerator.save_state(str(checkpoint))
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.state_dict(), checkpoint / "model.pt")
        (checkpoint / "meta.json").write_text(
            json.dumps({"step": step, "epoch": epoch, "batch_index": batch_index}),
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()


def _save_export(
    accelerator: Accelerator,
    model,
    root: Path,
    name: str,
    step: int,
    metrics: dict,
) -> None:
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


def _load_resume_state(checkpoint: str | None) -> dict[str, int]:
    if checkpoint is None:
        return {"step": 0, "epoch": 0, "batch_index": 0}
    metadata = Path(checkpoint) / "meta.json"
    if not metadata.exists():
        raise FileNotFoundError(f"checkpoint metadata is missing: {metadata}")
    state = json.loads(metadata.read_text(encoding="utf-8"))
    return {
        "step": int(state["step"]),
        "epoch": int(state.get("epoch", 0)),
        "batch_index": int(state.get("batch_index", 0)),
    }


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_run_manifest(cfg: Config, config_path: str) -> None:
    root = Path(cfg.paths.checkpoints)
    root.mkdir(parents=True, exist_ok=True)
    (root / "resolved_config.yaml").write_text(
        yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "config": str(Path(config_path)),
        "git_revision": _git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _set_loader_epoch(loader, epoch: int) -> None:
    sampler = getattr(loader, "batch_sampler", None)
    sampler = getattr(sampler, "batch_sampler", sampler)
    sampler = getattr(sampler, "sampler", sampler)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


class Trainer:
    def __init__(
        self,
        accelerator: Accelerator,
        model,
        optimizer,
        scheduler,
        tokenizer,
        loaders: dict,
        cfg: Config,
        total_steps: int,
    ) -> None:
        self.accelerator = accelerator
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.loaders = loaders
        self.train_loader = loaders["train"]
        self.cfg = cfg
        self.total_steps = total_steps
        self.save_root = Path(cfg.paths.checkpoints)
        self.save_root.mkdir(parents=True, exist_ok=True)
        self.metrics_path = Path(cfg.paths.logs) / "metrics.jsonl"
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.global_step = 0
        self.best_spbleu = -float("inf")
        self._tokens_since_step = 0
        self._step_started = time.perf_counter()

    def train(self, resume: str | None = None) -> None:
        resume_state = _load_resume_state(resume)
        if resume:
            self.accelerator.load_state(resume)
        self.global_step = resume_state["step"]
        self._log_start()
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()

        for epoch in range(resume_state["epoch"], self.cfg.train.epochs):
            _set_loader_epoch(self.train_loader, epoch)
            for batch_index, batch in enumerate(self.train_loader):
                if epoch == resume_state["epoch"] and batch_index < resume_state["batch_index"]:
                    continue
                if self.global_step >= self.total_steps:
                    break
                self._train_batch(batch, epoch, batch_index)
            if self.global_step >= self.total_steps:
                break

        _save_checkpoint(
            self.accelerator,
            self.model,
            self.save_root,
            self.global_step,
            epoch + 1,
            0,
        )
        if self.accelerator.is_main_process:
            self.accelerator.print(
                f"done. final checkpoint at {_checkpoint_path(self.save_root, self.global_step)}"
            )
        self.accelerator.end_training()

    def _log_start(self) -> None:
        self.accelerator.print(f"model params: {count_params(self.model):,}")
        self.accelerator.print(
            f"train batches/rank: {len(self.train_loader)} | total steps: {self.total_steps}"
        )

    def _train_batch(self, batch, epoch: int, batch_index: int) -> None:
        batch = {
            key: value.to(self.accelerator.device, non_blocking=True)
            for key, value in batch.items()
        }
        self._tokens_since_step += batch["labels"].ne(self.tokenizer.pad_id).sum().item()
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                logits = self.model(
                    batch["src_ids"],
                    batch["src_mask"],
                    batch["tgt_ids"][:, :-1],
                    tgt_mask=batch["tgt_mask"][:, :-1],
                )
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    batch["labels"].reshape(-1),
                    ignore_index=self.tokenizer.pad_id,
                    label_smoothing=self.cfg.train.label_smoothing,
                )
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self._optimizer_step(loss, epoch, batch_index)

    def _optimizer_step(self, loss, epoch: int, batch_index: int) -> None:
        grad_norm = self.accelerator.clip_grad_norm_(
            self.model.parameters(), self.cfg.train.clip_grad_norm
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        learning_rate = self.scheduler.get_last_lr()[0]
        elapsed = max(time.perf_counter() - self._step_started, 1e-9)
        token_count = self.accelerator.gather(
            torch.tensor([self._tokens_since_step], device=self.accelerator.device)
        ).sum().item()
        mean_loss = self.accelerator.gather(
            loss.detach().float().reshape(1)
        ).mean().item()
        metrics = {
            "loss": mean_loss,
            "learning_rate": learning_rate,
            "grad_norm": float(grad_norm),
            "step_time": elapsed,
            "tokens_per_second": token_count / elapsed,
        }
        if torch.cuda.is_available():
            metrics["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 2**30
            metrics["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 2**30
        self.accelerator.log(metrics, step=self.global_step)
        if self.accelerator.is_main_process:
            with self.metrics_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps({"step": self.global_step, **metrics}) + "\n")
                file.flush()
        self._tokens_since_step = 0
        self._step_started = time.perf_counter()
        if self.global_step % LOG_EVERY == 0:
            self.accelerator.print(
                f"epoch {epoch} step {self.global_step}/{self.total_steps} "
                f"loss {loss.item():.4f} lr {learning_rate:.2e}"
            )
        if self.cfg.train.save_every and self.global_step % self.cfg.train.save_every == 0:
            self._save_training_checkpoint(epoch, batch_index)
        if self.cfg.train.eval_every and self.global_step % self.cfg.train.eval_every == 0:
            self._evaluate()

    def _save_training_checkpoint(self, epoch: int, batch_index: int) -> None:
        _save_checkpoint(
            self.accelerator,
            self.model,
            self.save_root,
            self.global_step,
            epoch,
            batch_index + 1,
        )
        if self.accelerator.is_main_process:
            _prune_checkpoints(self.save_root)

    def _evaluate(self) -> None:
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process and "dev" in self.loaders:
            raw = self.accelerator.unwrap_model(self.model)
            metrics = evaluate(
                self.cfg, self.tokenizer, raw, "dev", self.accelerator.device
            )
            raw.train()
            self.accelerator.log(
                {f"dev/{name}": value for name, value in metrics.items()},
                step=self.global_step,
            )
            _save_export(
                self.accelerator,
                self.model,
                self.save_root,
                "latest",
                self.global_step,
                metrics,
            )
            selection_score = metrics.get("spbleu", metrics["chrf++"])
            if selection_score > self.best_spbleu:
                self.best_spbleu = selection_score
                _save_export(
                    self.accelerator,
                    self.model,
                    self.save_root,
                    "best",
                    self.global_step,
                    metrics,
                )
            self.accelerator.print(
                f"dev @ step {self.global_step}: "
                f"bleu {metrics['bleu']:.2f} chrf {metrics['chrf']:.2f}"
            )
        self.accelerator.wait_for_everyone()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.opts)
    accelerator = create_accelerator(cfg, args.config)
    if accelerator.is_main_process:
        _write_run_manifest(cfg, args.config)
    accelerator.wait_for_everyone()
    tokenizer = create_tokenizer(cfg)
    with accelerator.main_process_first():
        loaders = build_dataloaders(
            cfg,
            tokenizer,
            train_limit=OVERFIT_LIMIT if args.overfit else 0,
        )

    model = build_model(cfg.model)
    if cfg.train.compile:
        model = torch.compile(model)
    optimizer = build_optimizer(model, cfg.train)
    model, optimizer, loaders["train"], scheduler, total_steps = prepare_training(
        accelerator,
        model,
        optimizer,
        loaders["train"],
        cfg,
    )

    trainer = Trainer(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer,
        loaders=loaders,
        cfg=cfg,
        total_steps=total_steps,
    )
    trainer.train(resume=args.resume)


if __name__ == "__main__":
    main()
