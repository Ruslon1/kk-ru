"""Контракт модели (красный, пока T3 не реализован)."""

import torch

from kk_ru.config import ModelConfig
from kk_ru.model import build_model, count_params


def test_param_count_in_range():
    model = build_model(ModelConfig())
    n = count_params(model)
    assert 190_000_000 <= n <= 210_000_000, f"count_params={n:,}"


def test_tied_embeddings():
    model = build_model(ModelConfig())
    assert model.embed.weight is model.lm_head.weight


def test_forward_shapes():
    cfg = ModelConfig()
    model = build_model(cfg)
    model.eval()
    B, S, T = 2, 64, 64
    src_ids = torch.randint(0, cfg.vocab, (B, S))
    src_mask = torch.ones(B, S, dtype=torch.bool)
    tgt_ids = torch.randint(0, cfg.vocab, (B, T))
    logits = model(src_ids, src_mask, tgt_ids)
    assert logits.shape == (B, T, cfg.vocab)


def test_decoder_self_attn_is_gqa():
    from kk_ru.model import Attention

    attn = Attention(
        d_model=512,
        n_heads=8,
        n_kv_heads=4,
        causal=True,
        use_rope=True,
        qk_norm=True,
        head_dim=64,
    )
    assert attn.n_kv_heads == 4
