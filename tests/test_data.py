from pathlib import Path

import pytest

from kk_ru.data import TranslationDataset


class FakeTokenizer:
    bos_id = 1
    eos_id = 2
    pad_id = 0

    def __init__(self, model_path: Path):
        self.sp_model_path = str(model_path)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False):
        ids = [ord(char) for char in text]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids


def test_cache_rebuilds_when_source_changes(tmp_path: Path):
    source = tmp_path / "data.tsv"
    source.write_text("қ\tк\n", encoding="utf-8")
    model = tmp_path / "tokenizer.model"
    model.write_bytes(b"model-v1")
    tokenizer = FakeTokenizer(model)

    first = TranslationDataset(str(source), tokenizer, max_len=16)
    assert len(first) == 1

    source.write_text("қ\tк\nа\tb\n", encoding="utf-8")
    rebuilt = TranslationDataset(str(source), tokenizer, max_len=16)
    assert len(rebuilt) == 2

    source.write_text("қ\tк\nа\tb\nә\tc\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale or incompatible"):
        TranslationDataset(str(source), tokenizer, max_len=16, cache=False)


def test_cache_rebuilds_when_tokenizer_changes(tmp_path: Path):
    source = tmp_path / "data.tsv"
    source.write_text("қ\tк\n", encoding="utf-8")
    model = tmp_path / "tokenizer.model"
    model.write_bytes(b"model-v1")
    tokenizer = FakeTokenizer(model)
    TranslationDataset(str(source), tokenizer, max_len=16)

    model.write_bytes(b"model-v2")
    TranslationDataset(str(source), tokenizer, max_len=16)
