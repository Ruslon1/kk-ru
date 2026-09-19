"""Даталоадер и collate. Спецификация — TASKS.md T6.

Стриминг по ``data/filtered/train.kk-ru.tsv`` (не грузить 12M пар в память).
- shuffle буфером;
- токенизация обеих сторон;
- паддинг до максимума в батче (не до 256 всегда);
- encoder: ``src_ids`` + ``src_mask``;
- decoder: ``tgt_ids`` (со сдвигом вправо) + ``labels`` (``-100`` на pad);
- causal-маска нижнетреугольная (или через ``is_causal=True`` в SDPA).
"""
from __future__ import annotations

from typing import Iterator

from .config import Config


def iter_pairs(tsv_path: str) -> Iterator[tuple[str, str]]:
    """Итерировать (kk, ru) из TSV без загрузки всего файла в память."""
    raise NotImplementedError("T6: реализовать iter_pairs")


def collate_fn(batch: list[tuple[list[int], list[int]]], pad_id: int) -> dict:
    """Собрать батч: src_ids, src_mask, tgt_ids, labels. Паддинг -> labels=-100."""
    raise NotImplementedError("T6: реализовать collate_fn")


def build_dataloaders(cfg: Config, tokenizer, rank: int = 0, world_size: int = 1):
    """Вернуть (train_loader, dev_loader) для Accelerate (использовать accelerate.prepare)."""
    raise NotImplementedError("T6: реализовать build_dataloaders")
