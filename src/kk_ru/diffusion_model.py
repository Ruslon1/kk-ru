from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .model import Attention, EncoderBlock, SwiGLU, count_params


class TimestepEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        half = dimension // 2
        frequencies = torch.exp(
            torch.arange(half, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.projection = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.SiLU(),
            nn.Linear(dimension * 4, dimension),
        )

    def forward(self, timestep: torch.Tensor, total_steps: int) -> torch.Tensor:
        positions = timestep.float().unsqueeze(1) / total_steps
        angles = positions * self.frequencies.unsqueeze(0) * (2 * math.pi)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.dimension:
            embedding = F.pad(embedding, (0, self.dimension - embedding.shape[-1]))
        return self.projection(embedding)


class AdaRMSNorm(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.norm = nn.RMSNorm(dimension, eps=1e-6)
        self.modulation = nn.Linear(dimension, dimension * 2)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(time).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class DiffusionDecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = Attention(
            cfg.d_model,
            cfg.n_heads,
            cfg.n_kv_heads_decoder_self,
            causal=False,
            use_rope=cfg.pos == "rope",
            qk_norm=cfg.qk_norm,
            head_dim=cfg.head_dim,
            max_len=cfg.max_len,
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
        self.norm1 = AdaRMSNorm(cfg.d_model)
        self.norm2 = AdaRMSNorm(cfg.d_model)
        self.norm3 = AdaRMSNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        time: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x, time), mask=tgt_mask))
        x = x + self.dropout(
            self.cross_attn(self.norm2(x, time), kv=memory, mask=src_mask)
        )
        x = x + self.dropout(self.ffn(self.norm3(x, time)))
        return x


class DiscreteDiffusionTranslator(nn.Module):
    def __init__(self, cfg: ModelConfig, diffusion_steps: int = 16):
        super().__init__()
        self.vocab_size = cfg.vocab
        self.max_len = cfg.max_len
        self.diffusion_steps = diffusion_steps
        self.mask_id = cfg.vocab
        self.embed_tokens = nn.Embedding(cfg.vocab + 1, cfg.d_model)
        self.encoder = nn.ModuleList(
            EncoderBlock(cfg) for _ in range(cfg.n_encoder_layers)
        )
        self.time_embedding = TimestepEmbedding(cfg.d_model)
        self.decoder = nn.ModuleList(
            DiffusionDecoderBlock(cfg) for _ in range(cfg.n_decoder_layers)
        )
        self.norm = nn.RMSNorm(cfg.d_model, eps=1e-6)
        self.length_predictor = nn.Linear(cfg.d_model, cfg.max_len)
        self.apply(self._init_weights)
        for block in self.decoder:
            for norm in (block.norm1, block.norm2, block.norm3):
                nn.init.zeros_(norm.modulation.weight)
                nn.init.zeros_(norm.modulation.bias)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, src_ids: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(src_ids)
        for block in self.encoder:
            hidden = block(hidden, mask=src_mask)
        return hidden

    def predict_length(self, memory: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        weights = src_mask.unsqueeze(-1).to(memory.dtype)
        pooled = (memory * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return self.length_predictor(pooled)

    def denoise(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        noisy_ids: torch.Tensor,
        timestep: torch.Tensor,
        tgt_mask: torch.Tensor,
        total_steps: int | None = None,
    ) -> torch.Tensor:
        time = self.time_embedding(timestep, total_steps or self.diffusion_steps)
        hidden = self.embed_tokens(noisy_ids)
        for block in self.decoder:
            hidden = block(hidden, memory, time, src_mask, tgt_mask)
        hidden = self.norm(hidden)
        return F.linear(hidden, self.embed_tokens.weight[: self.vocab_size])

    def forward(
        self,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor,
        noisy_ids: torch.Tensor,
        timestep: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory = self.encode(src_ids, src_mask)
        logits = self.denoise(memory, src_mask, noisy_ids, timestep, tgt_mask)
        length_logits = self.predict_length(memory, src_mask)
        return logits, length_logits


def build_diffusion_model(cfg: ModelConfig, diffusion_steps: int = 16):
    return DiscreteDiffusionTranslator(cfg, diffusion_steps)


__all__ = ["DiscreteDiffusionTranslator", "build_diffusion_model", "count_params"]
