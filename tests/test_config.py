from pathlib import Path

import pytest

from kk_ru.config import _coerce, load_config
from kk_ru.tokenizer import validate_vocab


def test_null_override_is_none(tmp_path: Path):
    config = Path("configs/p0.yaml").read_text(encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text(config, encoding="utf-8")

    loaded = load_config(path, ["train.max_steps=null"])

    assert loaded.train.max_steps is None


def test_null_override_for_non_nullable_field_fails():
    with pytest.raises(ValueError, match="cannot set non-null"):
        _coerce(0, "null")


def test_nullable_integer_override_can_be_set():
    assert _coerce(None, "100") == 100


def test_config_rejects_vocab_mismatch(tmp_path: Path):
    config = Path("configs/p0.yaml").read_text(encoding="utf-8").replace(
        "vocab_size: 32000", "vocab_size: 16000"
    )
    path = tmp_path / "config.yaml"
    path.write_text(config, encoding="utf-8")

    with pytest.raises(ValueError, match="model.vocab must equal tokenizer.vocab_size"):
        load_config(path)


def test_tokenizer_vocab_validation():
    class Tokenizer:
        vocab_size = 3

    validate_vocab(Tokenizer(), 3)
    with pytest.raises(ValueError, match="has 3 pieces, expected 4"):
        validate_vocab(Tokenizer(), 4)
