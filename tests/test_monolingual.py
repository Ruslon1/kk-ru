from __future__ import annotations

import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "download_monolingual", Path(__file__).parents[1] / "scripts" / "download_monolingual.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_split_sentences_normalizes_whitespace():
    assert list(MODULE.split_sentences("Сәлем!\n  Бұл тест.  ")) == ["Сәлем!", "Бұл тест."]


def test_kazakh_filter_rejects_links_and_short_noise():
    assert MODULE.looks_like_kazakh("Бұл қазақ тіліндегі қалыпты сөйлем.", 20, 100)
    assert not MODULE.looks_like_kazakh("http://example.com", 3, 100)
    assert not MODULE.looks_like_kazakh("abc", 20, 100)


def test_sentence_hash_is_stable():
    assert MODULE.sentence_hash("мәтін") == MODULE.sentence_hash("мәтін")
