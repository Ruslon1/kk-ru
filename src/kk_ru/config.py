from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    type: str
    vocab: int
    tie_embeddings: bool
    d_model: int
    n_heads: int
    head_dim: int
    n_kv_heads_encoder: int
    n_kv_heads_decoder_self: int
    n_kv_heads_cross: int
    n_encoder_layers: int
    n_decoder_layers: int
    ffn: str
    ffn_hidden: int
    norm: str
    qk_norm: bool
    pos: str
    dropout: float
    max_len: int


@dataclass
class TokenizerConfig:
    sp_model: str
    vocab_size: int


@dataclass
class DataConfig:
    train_tsv: str
    eval_dev: str
    eval_test: str
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2


@dataclass
class TrainConfig:
    precision: str
    micro_batch_per_gpu: int
    grad_accum: int
    epochs: int
    optimizer: str = "adamw"
    lr: float
    warmup_steps: int
    schedule: str = "cosine"
    min_lr: float
    weight_decay: float
    clip_grad_norm: float
    label_smoothing: float
    save_every: int
    eval_every: int
    max_steps: int | None
    seed: int
    compile: bool = False


@dataclass
class GenConfig:
    beam: int
    max_len: int


@dataclass
class PathsConfig:
    checkpoints: str
    reports: str
    logs: str


@dataclass
class Config:
    model: ModelConfig
    tokenizer: TokenizerConfig
    data: DataConfig
    train: TrainConfig
    gen: GenConfig
    paths: PathsConfig


_SECTIONS = {
    "model": ModelConfig,
    "tokenizer": TokenizerConfig,
    "data": DataConfig,
    "train": TrainConfig,
    "gen": GenConfig,
    "paths": PathsConfig,
}


def _coerce(current: Any, value: str) -> Any:
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


def _build_section(section_type: Any, name: str, data: Any, path: Path) -> Any:
    if not isinstance(data, dict):
        raise ValueError(f"section {name!r} must be a mapping in {path}")
    allowed = {f.name for f in fields(section_type)}
    for key in data:
        if key not in allowed:
            raise ValueError(f"unknown key {name}.{key} in {path}")
    return section_type(**data)


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    for section in raw:
        if section not in _SECTIONS:
            raise ValueError(f"unknown section {section!r} in {path}")
    for section in _SECTIONS:
        if section not in raw:
            raise ValueError(f"missing section {section!r} in {path}")

    cfg = Config(
        **{
            name: _build_section(section_type, name, raw[name], path)
            for name, section_type in _SECTIONS.items()
        }
    )
    if overrides:
        _apply_overrides(cfg, overrides)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    model = cfg.model
    train = cfg.train
    data = cfg.data
    gen = cfg.gen
    if model.d_model != model.n_heads * model.head_dim:
        raise ValueError("model.d_model must equal model.n_heads * model.head_dim")
    for name, heads in (
        ("n_kv_heads_encoder", model.n_kv_heads_encoder),
        ("n_kv_heads_decoder_self", model.n_kv_heads_decoder_self),
        ("n_kv_heads_cross", model.n_kv_heads_cross),
    ):
        if heads <= 0 or model.n_heads % heads:
            raise ValueError(f"model.{name} must be a positive divisor of model.n_heads")
    if model.max_len <= 0 or gen.max_len <= 0:
        raise ValueError("model.max_len and gen.max_len must be positive")
    if train.micro_batch_per_gpu <= 0 or train.grad_accum <= 0:
        raise ValueError("micro_batch_per_gpu and grad_accum must be positive")
    if train.optimizer != "adamw":
        raise ValueError("only train.optimizer=adamw is supported")
    if train.schedule != "cosine":
        raise ValueError("only train.schedule=cosine is supported")
    if train.lr <= 0 or train.min_lr < 0 or train.min_lr > train.lr:
        raise ValueError("train.min_lr must be in [0, train.lr]")
    if not 0 <= train.label_smoothing < 1:
        raise ValueError("train.label_smoothing must be in [0, 1)")
    if data.num_workers < 0 or data.prefetch_factor <= 0:
        raise ValueError("data.num_workers must be non-negative and prefetch_factor positive")
    if gen.beam <= 0:
        raise ValueError("gen.beam must be positive")
