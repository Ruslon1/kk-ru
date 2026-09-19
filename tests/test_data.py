"""Контракт даталоадера (красный, пока T6 не реализован)."""

from kk_ru.data import collate_fn


def test_collate_pads_to_batch_max():
    pad_id = 0
    batch = [([1, 2, 3], [4, 5]), ([6], [7, 8, 9, 10])]
    out = collate_fn(batch, pad_id)
    assert out["src_ids"].shape[1] == 3   # max src len
    assert out["tgt_ids"].shape[1] == 4   # max tgt len
    assert out["labels"].shape == out["tgt_ids"].shape
    assert (out["labels"] != -100).sum().item() == (1 + 3)  # pad только там, где нужно
