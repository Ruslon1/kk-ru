from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        raise NotImplementedError("RMSNorm")


class RotaryEmbedding(nn.Module):

    def __init__(self, head_dim: int, max_len: int = 256):
        super().__init__()
        raise NotImplementedError("RotaryEmbedding")


class Attention(nn.Module):

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
        raise NotImplementedError("Attention")


class SwiGLU(nn.Module):

    def __init__(self, d_model: int, ffn_hidden: int):
        super().__init__()
        raise NotImplementedError("SwiGLU")


class EncoderBlock(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        raise NotImplementedError("EncoderBlock")


class DecoderBlock(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        raise NotImplementedError("DecoderBlock")


class TransformerEncoderDecoder(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        raise NotImplementedError("TransformerEncoderDecoder")

    def forward(
        self,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_ids: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("forward")


def build_model(cfg: ModelConfig) -> TransformerEncoderDecoder:
    raise NotImplementedError("build_model")


def count_params(model: nn.Module) -> int:
    raise NotImplementedError("count_params")
