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
        raise NotImplementedError("Tokenizer")

    @property
    def pad_id(self) -> int:
        raise NotImplementedError("pad_id")

    @property
    def bos_id(self) -> int:
        raise NotImplementedError("bos_id")

    @property
    def eos_id(self) -> int:
        raise NotImplementedError("eos_id")

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        raise NotImplementedError("encode")

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        raise NotImplementedError("decode")


def load_tokenizer(sp_model_path: str, max_len: int = 256) -> Tokenizer:
    raise NotImplementedError("load_tokenizer")
