from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpecialTokens:
    pad: int
    unk: int
    bos: int
    eos: int


class Tokenizer:

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
        raise NotImplementedError("T5")

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        raise NotImplementedError("T5")


def load_tokenizer(sp_model_path: str, max_len: int = 256) -> Tokenizer:
    raise NotImplementedError("T5: реализовать load_tokenizer")
