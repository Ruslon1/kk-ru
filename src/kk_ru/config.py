"""Config loading: YAML -> typed dataclasses, with dotted ``key=value`` overrides.

Single source of truth is ``configs/*.yaml``. Overrides for quick sweeps:
    python -m kk_ru.train --config configs/p0.yaml --opts train.epochs=5 model.d_model=768
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    type: str = "encoder-decoder"
    vocab: int = 32000
    tie_embeddings: bool = True
    d_model: int = 512
    n_heads: int = 8
    head_dim: int = 64
    n_kv_heads_encoder: int = 8
    n_kv_heads_decoder_self: int = 4
    n_kv_heads_cross: int = 8
    n_encoder_layers: int = 12
    n_decoder_layers: int = 12
    ffn: str = "swiglu"
    ffn_hidden: int = 4096
    norm: str = "rmsnorm"
    qk_norm: bool = True
    pos: str = "rope"
    dropout: float = 0.1
    max_len: int = 256


@dataclass
class TokenizerConfig:
    sp_model: str = "data/tokenizer/kk-ru-sp32k.model"
    vocab_size: int = 32000


@dataclass
class DataConfig:
    train_tsv: str = "data/filtered/train.kk-ru.tsv"
    eval_dev: str = "data/eval/flores_plus/dev.kk-ru.tsv"
    eval_test: str = "data/eval/flores_plus/devtest.kk-ru.tsv"


@dataclass
class TrainConfig:
    precision: str = "bf16"
    devices: int = 4
    strategy: str = "ddp"
    micro_batch_per_gpu: int = 16
    grad_accum: int = 4
    epochs: int = 3
    optimizer: str = "adamw"
    lr: float = 2.0e-4
    warmup_steps: int = 2000
    schedule: str = "cosine"
    min_lr: float = 2.0e-5
    weight_decay: float = 0.01
    clip_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    save_every: int = 2000
    eval_every: int = 2000
    max_steps: int | None = None
    seed: int = 42


@dataclass
class GenConfig:
    beam: int = 5
    max_len: int = 256


@dataclass
class PathsConfig:
    checkpoints: str = "checkpoints"
    reports: str = "reports"
    logs: str = "runs"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    gen: GenConfig = field(default_factory=GenConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


_SECTIONS = {
    "model": ModelConfig,
    "tokenizer": TokenizerConfig,
    "data": DataConfig,
    "train": TrainConfig,
    "gen": GenConfig,
    "paths": PathsConfig,
}


def _coerce(current: Any, value: str) -> Any:
    """Coerce a CLI string to the type of the field it is overriding."""
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if current is None:
        return value
    return type(current)(value)


def _apply_overrides(cfg: Config, overrides: list[str]) -> None:
    for kv in overrides:
        if "=" not in kv:
            raise ValueError(f"override must be key=value, got {kv!r}")
        key, value = kv.split("=", 1)
        parts = key.split(".")
        obj: Any = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        last = parts[-1]
        if not hasattr(obj, last):
            raise ValueError(f"unknown config key: {key}")
        setattr(obj, last, _coerce(getattr(obj, last), value))


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config into a :class:`Config`, then apply dotted overrides."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")

    cfg = Config()
    for section, section_type in _SECTIONS.items():
        if section not in raw:
            continue
        data = raw[section]
        if not isinstance(data, dict):
            raise ValueError(f"section {section!r} must be a mapping in {path}")
        target = getattr(cfg, section)
        for key, value in data.items():
            if not hasattr(target, key):
                raise ValueError(f"unknown key {section}.{key} in {path}")
            setattr(target, key, value)

    if overrides:
        _apply_overrides(cfg, overrides)
    return cfg
