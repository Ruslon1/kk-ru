from __future__ import annotations

from typing import Iterator

from .config import Config


def iter_pairs(tsv_path: str) -> Iterator[tuple[str, str]]:
    raise NotImplementedError("T6: реализовать iter_pairs")


def collate_fn(batch: list[tuple[list[int], list[int]]], pad_id: int) -> dict:
    raise NotImplementedError("T6: реализовать collate_fn")


def build_dataloaders(cfg: Config, tokenizer, rank: int = 0, world_size: int = 1):
    raise NotImplementedError("T6: реализовать build_dataloaders")
