from __future__ import annotations

from dataclasses import dataclass

import sentencepiece as spm


@dataclass
class SpecialTokens:
    pad: int
    unk: int
    bos: int
    eos: int


class Tokenizer:
    def __init__(self, sp_model_path: str, max_len: int = 256):
        self.sp = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.max_len = max_len

    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @property
    def special(self) -> SpecialTokens:
        return SpecialTokens(self.pad_id, self.unk_id, self.bos_id, self.eos_id)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not text:
            ids = []
        else:
            ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id, *ids]
        if add_eos:
            ids = [*ids, self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        if skip_special:
            ids = [i for i in ids if i not in (self.pad_id, self.unk_id, self.bos_id, self.eos_id)]
        return self.sp.decode(ids)


def load_tokenizer(sp_model_path: str, max_len: int = 256) -> Tokenizer:
    return Tokenizer(sp_model_path, max_len=max_len)
