from __future__ import annotations

import unittest

import torch

from kk_ru.config import ModelConfig, load_config
from kk_ru.diffusion_model import AdaRMSNorm, build_diffusion_model
from kk_ru.diffusion_eval import translate_diffusion
from kk_ru.diffusion_ops import (
    corrupt_targets,
    masked_token_loss,
    reveal_tokens,
    target_length_labels,
)


def tiny_config() -> ModelConfig:
    return ModelConfig(
        type="discrete-diffusion",
        vocab=20,
        tie_embeddings=True,
        d_model=32,
        n_heads=4,
        head_dim=8,
        n_kv_heads_encoder=4,
        n_kv_heads_decoder_self=2,
        n_kv_heads_cross=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        ffn="swiglu",
        ffn_hidden=64,
        norm="rmsnorm",
        qk_norm=True,
        pos="rope",
        dropout=0.0,
        max_len=8,
    )


class DiffusionTests(unittest.TestCase):
    def test_base_config_remains_valid_without_diffusion_section(self):
        self.assertIsNone(load_config("configs/p0.yaml").diffusion)

    def test_corruption_masks_valid_tokens_and_fully_masks_terminal_step(self):
        targets = torch.tensor([[1, 4, 2, 3], [1, 5, 6, 2]])
        valid = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
        timestep = torch.tensor([4, 4])
        noisy, selected = corrupt_targets(targets, valid, timestep, 4, 20)

        self.assertTrue(torch.equal(selected, valid))
        self.assertTrue(torch.equal(noisy[valid], torch.full((7,), 20)))
        self.assertEqual(noisy[0, 3].item(), 3)

    def test_corruption_masks_at_least_one_token_per_nonempty_row(self):
        targets = torch.tensor([[1, 2], [1, 3]])
        valid = torch.ones_like(targets, dtype=torch.bool)
        for _ in range(20):
            _, selected = corrupt_targets(targets, valid, torch.ones(2, dtype=torch.long), 16, 20)
            self.assertTrue(torch.all(selected.any(dim=1)))

    def test_masked_loss_only_scores_corrupted_positions(self):
        logits = torch.tensor([[[0.0, 8.0], [8.0, 0.0]]])
        targets = torch.tensor([[1, 1]])
        masked = torch.tensor([[True, False]])

        self.assertLess(masked_token_loss(logits, targets, masked).item(), 0.001)

    def test_lengths_include_all_nonpadding_tokens(self):
        valid = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
        self.assertEqual(target_length_labels(valid).tolist(), [2, 3])

    def test_adarmsnorm_uses_timestep_conditioning(self):
        norm = AdaRMSNorm(4)
        with torch.no_grad():
            norm.modulation.weight.fill_(0.25)
        hidden = torch.ones(2, 3, 4)
        time = torch.stack((torch.zeros(4), torch.ones(4)))

        output = norm(hidden, time)

        self.assertFalse(torch.equal(output[0], output[1]))

    def test_decoder_is_bidirectional_and_outputs_only_vocabulary_logits(self):
        cfg = tiny_config()
        model = build_diffusion_model(cfg, diffusion_steps=4)
        self.assertFalse(model.decoder[0].self_attn.causal)
        src = torch.tensor([[1, 4, 2], [1, 5, 2]])
        src_mask = torch.ones_like(src, dtype=torch.bool)
        noisy = torch.tensor([[20, 4, 20], [20, 5, 20]])
        target_mask = torch.ones_like(noisy, dtype=torch.bool)
        timestep = torch.tensor([2, 3])

        logits, length_logits = model(src, src_mask, noisy, timestep, target_mask)

        self.assertEqual(logits.shape, (2, 3, cfg.vocab))
        self.assertEqual(length_logits.shape, (2, cfg.max_len))

    def test_reverse_step_reveals_high_confidence_then_finishes(self):
        noisy = torch.full((1, 4), 20)
        valid = torch.ones_like(noisy, dtype=torch.bool)
        logits = torch.tensor([[[8.0, 0.0], [0.0, 8.0], [7.0, 0.0], [0.0, 6.0]]])

        once = reveal_tokens(noisy, logits, valid, 20, 2)
        self.assertEqual((once != 20).sum().item(), 2)
        self.assertEqual(once[0, 0].item(), 0)
        complete = reveal_tokens(once, logits, valid, 20, 1)
        self.assertFalse((complete == 20).any())

    def test_translation_runs_all_reverse_steps_and_returns_text(self):
        class TinyTokenizer:
            pad_id = 3
            bos_id = 1
            eos_id = 2

            @staticmethod
            def encode(text, add_bos=False):
                return [1, 4, 2] if add_bos else [4]

            @staticmethod
            def decode(ids, skip_special=True):
                return "ok"

        model = build_diffusion_model(tiny_config(), diffusion_steps=4)
        translations, elapsed = translate_diffusion(
            model,
            TinyTokenizer(),
            ["source one", "source two"],
            torch.device("cpu"),
            max_len=8,
            steps=4,
            batch_size=2,
        )

        self.assertEqual(translations, ["ok", "ok"])
        self.assertGreaterEqual(elapsed, 0)


if __name__ == "__main__":
    unittest.main()
