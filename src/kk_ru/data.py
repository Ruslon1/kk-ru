from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .config import Config


def iter_pairs(tsv_path: str) -> Iterator[tuple[str, str]]:
    with open(tsv_path, encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                yield parts[0], parts[1]


def _cache_dir(tsv_path: str) -> Path:
    return Path(f"{tsv_path}.tok")


def _cache_is_complete(path: Path) -> bool:
    return (path / "meta.json").exists() and all(
        (path / name).exists()
        for name in ("src.bin", "tgt.bin", "src_offsets.npy", "tgt_offsets.npy")
    )


def _build_mmap_cache(tsv_path: str, tokenizer, max_len: int, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    src_offsets = [0]
    tgt_offsets = [0]
    count = 0
    with (cache_dir / "src.bin").open("wb") as src_file, (cache_dir / "tgt.bin").open("wb") as tgt_file:
        for kk, ru in iter_pairs(tsv_path):
            src_ids = tokenizer.encode(kk, add_bos=True)[:max_len]
            tgt_ids = tokenizer.encode(ru, add_eos=True)[:max_len]
            np.asarray(src_ids, dtype=np.int32).tofile(src_file)
            np.asarray(tgt_ids, dtype=np.int32).tofile(tgt_file)
            src_offsets.append(src_offsets[-1] + len(src_ids))
            tgt_offsets.append(tgt_offsets[-1] + len(tgt_ids))
            count += 1
    np.save(cache_dir / "src_offsets.npy", np.asarray(src_offsets, dtype=np.int64))
    np.save(cache_dir / "tgt_offsets.npy", np.asarray(tgt_offsets, dtype=np.int64))
    (cache_dir / "meta.json").write_text(
        json.dumps({"count": count, "max_len": max_len}), encoding="utf-8"
    )


class TranslationDataset(Dataset):
    """Map-style dataset backed by disk-backed token arrays and offsets."""

    def __init__(self, tsv_path: str, tokenizer, max_len: int, cache: bool = True, limit: int = 0):
        self.max_len = max_len
        self._temporary = limit > 0
        if limit:
            self._load_limited(tsv_path, tokenizer, max_len, limit)
            return

        cache_dir = _cache_dir(tsv_path)
        if cache and not _cache_is_complete(cache_dir):
            _build_mmap_cache(tsv_path, tokenizer, max_len, cache_dir)
        elif not _cache_is_complete(cache_dir):
            raise FileNotFoundError(f"token cache is missing: {cache_dir}")

        meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
        self.src_offsets = np.load(cache_dir / "src_offsets.npy", mmap_mode="r")
        self.tgt_offsets = np.load(cache_dir / "tgt_offsets.npy", mmap_mode="r")
        self.src_tokens = np.memmap(cache_dir / "src.bin", dtype=np.int32, mode="r")
        self.tgt_tokens = np.memmap(cache_dir / "tgt.bin", dtype=np.int32, mode="r")
        self._lengths = np.diff(self.src_offsets) + np.diff(self.tgt_offsets)
        self._count = int(meta["count"])

    def _load_limited(self, tsv_path: str, tokenizer, max_len: int, limit: int) -> None:
        src, tgt = [], []
        for index, (kk, ru) in enumerate(iter_pairs(tsv_path)):
            if index >= limit:
                break
            src.append(tokenizer.encode(kk, add_bos=True)[:max_len])
            tgt.append(tokenizer.encode(ru, add_eos=True)[:max_len])
        self.src_tokens = np.asarray([token for item in src for token in item], dtype=np.int32)
        self.tgt_tokens = np.asarray([token for item in tgt for token in item], dtype=np.int32)
        self.src_offsets = np.zeros(len(src) + 1, dtype=np.int64)
        self.tgt_offsets = np.zeros(len(tgt) + 1, dtype=np.int64)
        np.cumsum([len(item) for item in src], out=self.src_offsets[1:])
        np.cumsum([len(item) for item in tgt], out=self.tgt_offsets[1:])
        self._lengths = np.diff(self.src_offsets) + np.diff(self.tgt_offsets)
        self._count = len(src)

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        src_start, src_end = self.src_offsets[index : index + 2]
        tgt_start, tgt_end = self.tgt_offsets[index : index + 2]
        return (
            self.src_tokens[src_start:src_end].tolist(),
            self.tgt_tokens[tgt_start:tgt_end].tolist(),
        )

    def lengths(self) -> np.ndarray:
        return self._lengths


class LengthBatchSampler(Sampler):
    """Sort by length, group into batches, shuffle the batch order."""

    def __init__(self, lengths, batch_size: int, shuffle: bool = True, seed: int = 42,
                 drop_last: bool = False, rank: int = 0, world_size: int = 1):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        order = sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        batches = [order[start : start + self.batch_size] for start in range(0, len(order), self.batch_size)]
        if self.drop_last:
            batches = [batch for batch in batches if len(batch) == self.batch_size]
        if self.shuffle:
            random.Random(self.seed).shuffle(batches)
        yield from batches[self.rank :: self.world_size]

    def __len__(self) -> int:
        batch_count = len(self.lengths) // self.batch_size
        if not self.drop_last and len(self.lengths) % self.batch_size:
            batch_count += 1
        return max(0, (batch_count - self.rank + self.world_size - 1) // self.world_size)


def collate_fn(batch: list[tuple[list[int], list[int]]], pad_id: int) -> dict:
    src_lens = [len(src) for src, _ in batch]
    tgt_lens = [len(tgt) for _, tgt in batch]
    src_ids = torch.full((len(batch), max(src_lens)), pad_id, dtype=torch.long)
    tgt_ids = torch.full((len(batch), max(tgt_lens)), pad_id, dtype=torch.long)
    src_mask = torch.zeros_like(src_ids, dtype=torch.bool)
    for index, (src, tgt) in enumerate(batch):
        src_ids[index, : len(src)] = torch.tensor(src)
        tgt_ids[index, : len(tgt)] = torch.tensor(tgt)
        src_mask[index, : len(src)] = True
    return {
        "src_ids": src_ids,
        "src_mask": src_mask,
        "tgt_ids": tgt_ids,
        "labels": tgt_ids[:, 1:].clone(),
    }


def _make_loader(dataset, batch_size: int, tokenizer, shuffle: bool, seed: int, rank: int, world_size: int):
    if shuffle:
        sampler = LengthBatchSampler(
            dataset.lengths(), batch_size, shuffle=True, seed=seed,
            drop_last=True, rank=rank, world_size=world_size,
        )
        return torch.utils.data.DataLoader(
            dataset, batch_sampler=sampler,
            collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_id),
            num_workers=0,
        )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_id), num_workers=0,
    )


def build_dataloaders(cfg: Config, tokenizer, rank: int = 0, world_size: int = 1, train_limit: int = 0):
    train = _make_loader(
        TranslationDataset(cfg.data.train_tsv, tokenizer, cfg.model.max_len, limit=train_limit),
        cfg.train.micro_batch_per_gpu, tokenizer, shuffle=True,
        seed=cfg.train.seed, rank=rank, world_size=world_size,
    )
    loaders = {"train": train}
    for name, path in (("dev", cfg.data.eval_dev), ("test", cfg.data.eval_test)):
        if Path(path).exists():
            dataset = TranslationDataset(path, tokenizer, cfg.model.max_len)
            loaders[name] = torch.utils.data.DataLoader(
                dataset, batch_size=cfg.train.micro_batch_per_gpu, shuffle=False,
                collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_id), num_workers=0,
            )
    return loaders
