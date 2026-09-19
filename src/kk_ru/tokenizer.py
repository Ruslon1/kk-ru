"""SentencePiece 32k обёртка (shared KK+RU). Спецификация — TASKS.md T5.

Обучение модели — ``scripts/train_tokenizer.py``; здесь только загрузка и API.
Спецтокены: ``<pad> <unk> <bos> <eos>`` (порядок/индексы зафиксировать и задокументировать).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpecialTokens:
    pad: int
    unk: int
    bos: int
    eos: int


class Tokenizer:
    """Обёртка над sentencepiece, умеющая добавлять/убирать BOS/EOS и обрезать до max_len."""

    def __init__(self, sp_model_path: str, max_len: int = 256):
        raise NotImplementedError("T5: реализовать Tokenizer")

    @property
    def pad_id(self) -> int:
        raise NotImplementedError("T5")

    @property
    def bos_id(self) -> int:
        raise NotImplementedError("T5")

    @property
    def eos_id(self) -> int:
        raise NotImplementedError("T5")

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Токенизировать; при необходимости добавить BOS/EOS; обрезать до max_len."""
        raise NotImplementedError("T5")

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        raise NotImplementedError("T5")


def load_tokenizer(sp_model_path: str, max_len: int = 256) -> Tokenizer:
    raise NotImplementedError("T5: реализовать load_tokenizer")
