from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F

from .config import Config, load_config
from .data import build_dataloaders
from .diffusion_eval import evaluate_diffusion
from .diffusion_model import build_diffusion_model
from .diffusion_ops import corrupt_targets, masked_token_loss, target_length_labels
from .tokenizer import load_tokenizer, validate_vocab
from .train import (
    OVERFIT_LIMIT,
    Trainer,
    _write_run_manifest,
    build_optimizer,
    create_accelerator,
    prepare_training,
)


class DiffusionTrainer(Trainer):
    def _train_batch(self, batch, epoch: int, batch_index: int) -> None:
        batch = {
            key: value.to(self.accelerator.device, non_blocking=True)
            for key, value in batch.items()
        }
        target_ids = batch["tgt_ids"]
        target_mask = batch["tgt_mask"]
        timestep = torch.randint(
            1,
            self.cfg.diffusion.steps + 1,
            (target_ids.shape[0],),
            device=self.accelerator.device,
        )
        noisy_ids, masked = corrupt_targets(
            target_ids,
            target_mask,
            timestep,
            self.cfg.diffusion.steps,
            self.cfg.model.vocab,
        )
        self._tokens_since_step += int(target_mask.sum().item())
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                logits, length_logits = self.model(
                    batch["src_ids"],
                    batch["src_mask"],
                    noisy_ids,
                    timestep,
                    target_mask,
                )
                token_loss = masked_token_loss(logits, target_ids, masked)
                length_loss = F.cross_entropy(
                    length_logits.float(), target_length_labels(target_mask)
                )
                loss = token_loss + self.cfg.diffusion.length_loss_weight * length_loss
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self._optimizer_step(loss, epoch, batch_index)

    def _evaluate_dev(self, model) -> dict:
        metrics, _ = evaluate_diffusion(
            self.cfg, self.tokenizer, model, "dev", self.accelerator.device
        )
        return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a discrete diffusion translator.")
    parser.add_argument("--config", default="configs/diffusion.yaml")
    parser.add_argument("--opts", nargs="*", default=None)
    parser.add_argument("--overfit", action="store_true")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume")
    resume.add_argument("--resume-new-data")
    return parser.parse_args()


def train(cfg: Config, config_path: str, overfit: bool, resume: str | None, resume_new_data: str | None) -> None:
    if cfg.diffusion is None:
        raise ValueError("diffusion config section is required")
    accelerator = create_accelerator(cfg, config_path)
    if accelerator.is_main_process:
        _write_run_manifest(cfg, config_path)
    accelerator.wait_for_everyone()
    tokenizer = load_tokenizer(cfg.tokenizer.sp_model, max_len=cfg.model.max_len)
    validate_vocab(tokenizer, cfg.model.vocab)
    with accelerator.main_process_first():
        loaders = build_dataloaders(
            cfg,
            tokenizer,
            train_limit=OVERFIT_LIMIT if overfit else 0,
        )
    model = build_diffusion_model(cfg.model, cfg.diffusion.steps)
    if cfg.train.compile:
        model = torch.compile(model)
    optimizer = build_optimizer(model, cfg.train)
    model, optimizer, loaders["train"], scheduler, total_steps = prepare_training(
        accelerator, model, optimizer, loaders["train"], cfg
    )
    trainer = DiffusionTrainer(
        accelerator,
        model,
        optimizer,
        scheduler,
        tokenizer,
        loaders,
        cfg,
        total_steps,
    )
    trainer.train(resume=resume, resume_new_data=resume_new_data)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.opts)
    train(cfg, args.config, args.overfit, args.resume, args.resume_new_data)


if __name__ == "__main__":
    main()
