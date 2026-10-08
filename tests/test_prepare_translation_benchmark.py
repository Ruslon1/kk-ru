from __future__ import annotations

import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "prepare_translation_benchmark",
    Path(__file__).parents[1] / "scripts" / "prepare_translation_benchmark.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_bucket_boundaries():
    assert MODULE.bucket("a " * 8) == "short"
    assert MODULE.bucket("a " * 9) == "medium"
    assert MODULE.bucket("a " * 31) == "long"


def test_reservoir_is_deterministic():
    rows = [(f"text {i} " + "word " * 8, "ru") for i in range(100)]
    first, _ = MODULE.reservoir_by_bucket(rows, {"medium": 10}, 42)
    second, _ = MODULE.reservoir_by_bucket(rows, {"medium": 10}, 42)
    assert first == second
