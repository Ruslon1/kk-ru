
from pathlib import Path

from kk_ru.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_p0_loads():
    cfg = load_config(ROOT / "configs" / "p0.yaml")
    assert cfg.model.d_model == 512
    assert cfg.model.vocab == 32000
    assert cfg.model.tie_embeddings is True
    assert cfg.model.n_kv_heads_decoder_self == 4
    assert cfg.train.seed == 42


def test_override_coercion():
    cfg = load_config(ROOT / "configs" / "p0.yaml", overrides=["model.d_model=768", "train.epochs=5"])
    assert cfg.model.d_model == 768
    assert cfg.train.epochs == 5
