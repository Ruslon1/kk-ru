from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import sentencepiece as spm
import torch
from accelerate import Accelerator

from kk_ru.config import DiffusionConfig, PathsConfig, load_config
from kk_ru.data import build_dataloaders
from kk_ru.diffusion_model import build_diffusion_model
from kk_ru.diffusion_train import DiffusionTrainer
from kk_ru.model import build_model
from kk_ru.tokenizer import load_tokenizer
from kk_ru.train import Trainer, build_optimizer, build_scheduler


class EvaluationTests(unittest.TestCase):
    def test_dev_evaluation_exports_both_models_and_restores_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "pairs.tsv"
            corpus.write_text("Бұл қазақ мәтіні.\tЭто русский текст.\n", encoding="utf-8")
            spm.SentencePieceTrainer.train(
                input=str(corpus), model_prefix=str(root / "tokenizer"),
                vocab_size=64, hard_vocab_limit=False, pad_id=3, minloglevel=2,
            )
            tokenizer = load_tokenizer(str(root / "tokenizer.model"))
            accelerator = Accelerator(cpu=True)

            for diffusion in (False, True):
                with self.subTest(diffusion=diffusion):
                    cfg = load_config("configs/p0.yaml")
                    cfg.model = replace(
                        cfg.model, vocab=tokenizer.vocab_size, d_model=16,
                        n_heads=2, head_dim=8, n_kv_heads_encoder=2,
                        n_kv_heads_decoder_self=1, n_kv_heads_cross=2,
                        n_encoder_layers=1, n_decoder_layers=1,
                        ffn_hidden=32, max_len=16, dropout=0,
                    )
                    cfg.data = replace(
                        cfg.data, train_tsv=str(corpus), eval_dev=str(corpus),
                        eval_test=str(corpus), num_workers=0,
                    )
                    cfg.train.micro_batch_per_gpu = 1
                    cfg.gen = replace(cfg.gen, max_len=8, beam=2)
                    run = root / ("diffusion" if diffusion else "ar")
                    cfg.paths = PathsConfig(str(run / "checkpoints"), str(run / "reports"), str(run / "logs"))
                    if diffusion:
                        cfg.diffusion = DiffusionConfig(steps=2)
                        model = build_diffusion_model(cfg.model, 2)
                        trainer_type = DiffusionTrainer
                    else:
                        model = build_model(cfg.model)
                        trainer_type = Trainer
                    optimizer = build_optimizer(model, cfg.train)
                    scheduler = build_scheduler(optimizer, cfg.train, 10)
                    trainer = trainer_type(
                        accelerator, model, optimizer, scheduler, tokenizer,
                        build_dataloaders(cfg, tokenizer), cfg, 10,
                    )

                    trainer._evaluate()

                    self.assertTrue(model.training)
                    for export in ("latest", "best"):
                        path = Path(cfg.paths.checkpoints) / export
                        state = torch.load(path / "model.pt", weights_only=True)
                        for name, value in model.state_dict().items():
                            torch.testing.assert_close(state[name], value)
                        metadata = json.loads((path / "meta.json").read_text())
                        self.assertEqual(metadata["step"], 0)
                        self.assertIn("chrf++", metadata["metrics"])
                        if diffusion:
                            self.assertEqual(metadata["metrics"]["denoising_steps"], 2)
