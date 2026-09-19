"""P0 model: Qwen-style encoder–decoder (~200M). Спецификация — PLAN.md §2–§4, TASKS.md T3.

Скелет (12+12, d_model=512, 8 heads, head_dim=64) — как у kazRush; блок — Qwen-style.

- **Encoder** (12 слоёв): bidirectional **MHA** self-attn только (без causal, без cross-attn):
  ``x = x + SelfAttn(RMSNorm(x))`` → ``x = x + SwiGLU(RMSNorm(x))``.
  Self-attn: RoPE + QK-Norm, 8Q/8KV.
- **Decoder** (12 слоёв), три субячейки / три резудиала / три Pre-LN RMSNorm:
  1. ``y = y + MaskedSelfAttn(RMSNorm(y))`` — **GQA 8Q/4KV**, causal, RoPE, QK-Norm;
  2. ``y = y + CrossAttn(RMSNorm(y), encoder_out)`` — **MHA 8Q/8KV**, Q=RU, K/V=KK,
     **RoPE нет**, QK-Norm да;
  3. ``y = y + SwiGLU(RMSNorm(y))``.
- Norm: RMSNorm **Pre-LN** (перед каждой субячейкой, резудиал после).
- Pos: RoPE **только self-attn** (encoder self + decoder masked self). Cross-attn — без RoPE.
- FFN: SwiGLU ``512 → 4096 → 512`` (gate + up → down).
- Эмбеддинги: обучаются с нуля, ``tie_embeddings=True`` (lm_head = input embeddings), shared KK+RU.
- Attention через ``F.scaled_dot_product_attention`` (flash-совместимо).

Ожидание: ``count_params`` ≈ 190–210M (цель ~202M).
"""
from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


class RMSNorm(nn.Module):
    """Pre-LN root-mean-square normalization (Qwen-style), без обучаемого bias по умолчанию."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        raise NotImplementedError("T3: реализовать RMSNorm")


class RotaryEmbedding(nn.Module):
    """RoPE для self-attention. Применяется только к Q/K self-attn, НЕ к cross-attn."""

    def __init__(self, head_dim: int, max_len: int = 256):
        super().__init__()
        raise NotImplementedError("T3: реализовать RoPE")


class Attention(nn.Module):
    """Multi-head / grouped-query attention через ``F.scaled_dot_product_attention``.

    Args:
        d_model: ширина скрытого слоя (512).
        n_heads: число query-голов (8).
        n_kv_heads: число K/V-голов. ``< n_heads`` включает GQA (decoder self: 4).
        causal: наложить causal-маску (только decoder self-attn).
        use_rope: применить RoPE (только self-attn; cross-attn — False).
        qk_norm: применить QK-Norm (все attention).
        head_dim: размерность головы (64).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        causal: bool,
        use_rope: bool,
        qk_norm: bool,
        head_dim: int = 64,
    ):
        super().__init__()
        raise NotImplementedError("T3: реализовать Attention")


class SwiGLU(nn.Module):
    """SwiGLU FFN: d_model -> ffn_hidden (gate|up) -> SiLU(gate)*up -> down -> d_model."""

    def __init__(self, d_model: int, ffn_hidden: int):
        super().__init__()
        raise NotImplementedError("T3: реализовать SwiGLU")


class EncoderBlock(nn.Module):
    """x = x + MHA(RMSNorm(x)) ; x = x + SwiGLU(RMSNorm(x)). Без causal, без cross-attn."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        raise NotImplementedError("T3: реализовать EncoderBlock")


class DecoderBlock(nn.Module):
    """masked-GQA self + MHA cross + SwiGLU, три Pre-LN RMSNorm, три резудиала."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        raise NotImplementedError("T3: реализовать DecoderBlock")


class TransformerEncoderDecoder(nn.Module):
    """token_embed -> encoder -> decoder -> lm_head (tied)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        raise NotImplementedError("T3: реализовать TransformerEncoderDecoder")

    def forward(
        self,
        src_ids: torch.Tensor,          # (B, S) казахский
        src_mask: torch.Tensor,         # (B, S) pad-mask
        tgt_ids: torch.Tensor,          # (B, T) русский (со сдвигом для teacher forcing)
    ) -> torch.Tensor:
        """Возвращает логиты (B, T, vocab)."""
        raise NotImplementedError("T3: реализовать forward")


def build_model(cfg: ModelConfig) -> TransformerEncoderDecoder:
    """Фабрика: собрать P0-модель по конфигу (веса случайные)."""
    raise NotImplementedError("T3: реализовать build_model")


def count_params(model: nn.Module) -> int:
    """Число обучаемых параметров. Ожидание ~202M."""
    raise NotImplementedError("T3: реализовать count_params")
