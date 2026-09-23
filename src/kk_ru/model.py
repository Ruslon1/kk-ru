from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig


class RotaryEmbedding(nn.Module):

    def __init__(self, head_dim: int, max_len: int = 256):
        super().__init__()
        self.head_dim = head_dim
        self.max_len = max_len
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.shape[-2]
        cos = self.cos[:seq].to(x.dtype)
        sin = self.sin[:seq].to(x.dtype)
        half = self.head_dim // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        rot1 = x1 * cos[..., :half] - x2 * sin[..., :half]
        rot2 = x1 * sin[..., :half] + x2 * cos[..., :half]
        return torch.cat([rot1, rot2], dim=-1)


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
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.causal = causal
        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.rotary = RotaryEmbedding(head_dim) if use_rope else None
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6) if qk_norm else None
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6) if qk_norm else None

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = x.shape
        q = self.wq(x).view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        src = x if kv is None else kv
        k = self.wk(src).view(bsz, -1, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(src).view(bsz, -1, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.rotary is not None:
            q = self.rotary(q)
            k = self.rotary(k)
        if self.n_kv_heads != self.n_heads:
            reps = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=self.causal, dropout_p=0.0
        )
        out = out.transpose(1, 2).reshape(bsz, q_len, -1)
        return self.wo(out)


class SwiGLU(nn.Module):

    def __init__(self, d_model: int, ffn_hidden: int):
        super().__init__()
        self.gate = nn.Linear(d_model, ffn_hidden, bias=False)
        self.up = nn.Linear(d_model, ffn_hidden, bias=False)
        self.down = nn.Linear(ffn_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class EncoderBlock(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = Attention(
            cfg.d_model,
            cfg.n_heads,
            cfg.n_kv_heads_encoder,
            causal=False,
            use_rope=cfg.pos == "rope",
            qk_norm=cfg.qk_norm,
            head_dim=cfg.head_dim,
        )
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)
        self.norm1 = nn.RMSNorm(cfg.d_model, eps=1e-6)
        self.norm2 = nn.RMSNorm(cfg.d_model, eps=1e-6)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class DecoderBlock(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = Attention(
            cfg.d_model,
            cfg.n_heads,
            cfg.n_kv_heads_decoder_self,
            causal=True,
            use_rope=cfg.pos == "rope",
            qk_norm=cfg.qk_norm,
            head_dim=cfg.head_dim,
        )
        self.cross_attn = Attention(
            cfg.d_model,
            cfg.n_heads,
            cfg.n_kv_heads_cross,
            causal=False,
            use_rope=False,
            qk_norm=False,
            head_dim=cfg.head_dim,
        )
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)
        self.norm1 = nn.RMSNorm(cfg.d_model, eps=1e-6)
        self.norm2 = nn.RMSNorm(cfg.d_model, eps=1e-6)
        self.norm3 = nn.RMSNorm(cfg.d_model, eps=1e-6)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x)))
        x = x + self.dropout(self.cross_attn(self.norm2(x), kv=memory, mask=src_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class KkRuModel(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab, cfg.d_model)
        self.encoder = nn.ModuleList(
            [EncoderBlock(cfg) for _ in range(cfg.n_encoder_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderBlock(cfg) for _ in range(cfg.n_decoder_layers)]
        )
        self.norm = nn.RMSNorm(cfg.d_model, eps=1e-6)
        self.output = (
            None if cfg.tie_embeddings else nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        )

    def forward(
        self,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_ids: torch.Tensor,
    ) -> torch.Tensor:
        h = self.embed_tokens(src_ids)
        for block in self.encoder:
            h = block(h, mask=src_mask)
        memory = h
        h = self.embed_tokens(tgt_ids)
        for block in self.decoder:
            h = block(h, memory, src_mask=src_mask)
        h = self.norm(h)
        if self.output is None:
            return h @ self.embed_tokens.weight.t()
        return self.output(h)


def build_model(cfg: ModelConfig) -> KkRuModel:
    return KkRuModel(cfg)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
