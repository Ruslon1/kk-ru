"""Контракт токенизатора (красный, пока T5 не реализован)."""

import pytest

from kk_ru.tokenizer import load_tokenizer

SP_MODEL = "data/tokenizer/kk-ru-sp32k.model"


@pytest.mark.skipif(not __import__("os").path.exists(SP_MODEL), reason="tokenizer not trained yet")
def test_roundtrip():
    tok = load_tokenizer(SP_MODEL, max_len=256)
    for text in ["Қазақстан Республикасы", "Это тестовое предложение на русском."]:
        ids = tok.encode(text, add_bos=True, add_eos=True)
        assert tok.pad_id >= 0 and tok.bos_id >= 0 and tok.eos_id >= 0
        assert len(ids) <= 256
        assert isinstance(tok.decode(ids), str)
