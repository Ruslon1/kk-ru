import torch

from kk_ru.config import ModelConfig
from kk_ru.data import collate_fn
from kk_ru.model import build_model


def tiny_config() -> ModelConfig:
    return ModelConfig(
        type="encoder-decoder",
        vocab=32,
        tie_embeddings=True,
        d_model=16,
        n_heads=4,
        head_dim=4,
        n_kv_heads_encoder=2,
        n_kv_heads_decoder_self=2,
        n_kv_heads_cross=2,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ffn="swiglu",
        ffn_hidden=32,
        norm="rmsnorm",
        qk_norm=True,
        pos="rope",
        dropout=0.0,
        max_len=16,
    )


def test_cached_decode_matches_full_decode():
    torch.manual_seed(0)
    model = build_model(tiny_config()).eval()
    src_ids = torch.tensor([[1, 5, 6, 0]])
    src_mask = torch.tensor([[True, True, True, False]])
    tgt_ids = torch.tensor([[1, 7, 8, 9]])
    memory = model.encode(src_ids, src_mask)
    full = model.decode(tgt_ids, memory, src_mask)

    cache = None
    pieces = []
    for index in range(tgt_ids.shape[1]):
        logits, cache = model.decode(
            tgt_ids[:, index : index + 1],
            memory,
            src_mask,
            past_key_values=cache,
            use_cache=True,
        )
        pieces.append(logits)

    cached = torch.cat(pieces, dim=1)
    torch.testing.assert_close(cached, full, rtol=1e-5, atol=1e-5)


def test_collate_shifts_labels_and_masks_padding():
    batch = collate_fn([([1, 4], [1, 8, 2]), ([1], [1, 9, 2])], pad_id=0)

    assert batch["src_ids"].tolist() == [[1, 4], [1, 0]]
    assert batch["tgt_ids"].tolist() == [[1, 8, 2], [1, 9, 2]]
    assert batch["labels"].tolist() == [[8, 2], [9, 2]]
    assert batch["src_mask"].tolist() == [[True, True], [True, False]]
    assert batch["tgt_mask"].tolist() == [[True, True, True], [True, True, True]]
