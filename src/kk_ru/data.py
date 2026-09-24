from __future__ import annotations

import json
import random
import shutil
import time
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
    temporary_dir = cache_dir.with_name(f"{cache_dir.name}.tmp-{time.time_ns()}")
    temporary_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = temporary_dir
    src_offsets = [0]
    tgt_offsets = [0]
    count = 0
    with (cache_dir / "src.bin").open("wb") as src_file, (cache_dir / "tgt.bin").open("wb") as tgt_file:
        for kk, ru in iter_pairs(tsv_path):
            src_ids = tokenizer.encode(kk, add_bos=True)[:max_len]
            tgt_ids = tokenizer.encode(ru, add_eos=True)[:max_len]
            if tgt_ids[-1:] != [tokenizer.eos_id]:
                tgt_ids[-1] = tokenizer.eos_id
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
    final_dir = Path(str(cache_dir).split(".tmp-", 1)[0])
    if final_dir.exists():
        shutil.rmtree(cache_dir)
    else:
        cache_dir.replace(final_dir)


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
        if meta["max_len"] != max_len:
            raise ValueError(f"token cache {cache_dir} was built with max_len={meta['max_len']}, expected {max_len}")
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
            target = tokenizer.encode(ru, add_eos=True)[:max_len]
            if target[-1:] != [tokenizer.eos_id]:
                target[-1] = tokenizer.eos_id
            tgt.append(target)
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
                 drop_last: bool = False):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        order = sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        batches = [order[start : start + self.batch_size] for start in range(0, len(order), self.batch_size)]
        if self.drop_last:
            batches = [batch for batch in batches if len(batch) == self.batch_size]
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        batch_count = len(self.lengths) // self.batch_size
        if not self.drop_last and len(self.lengths) % self.batch_size:
            batch_count += 1
        return batch_count


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


class TranslationCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[tuple[list[int], list[int]]]) -> dict:
        return collate_fn(batch, self.pad_id)


def _make_loader(dataset, batch_size: int, tokenizer, shuffle: bool, seed: int, data_cfg):
    loader_args = {
        "collate_fn": TranslationCollator(tokenizer.pad_id),
        "num_workers": data_cfg.num_workers,
        "pin_memory": data_cfg.pin_memory,
    }
    if data_cfg.num_workers:
        loader_args.update(
            persistent_workers=data_cfg.persistent_workers,
            prefetch_factor=data_cfg.prefetch_factor,
        )
    if shuffle:
        sampler = LengthBatchSampler(
            dataset.lengths(), batch_size, shuffle=True, seed=seed,
            drop_last=True,
        )
        return torch.utils.data.DataLoader(dataset, batch_sampler=sampler, **loader_args)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, **loader_args)


def build_dataloaders(cfg: Config, tokenizer, train_limit: int = 0):
    train = _make_loader(
        TranslationDataset(cfg.data.train_tsv, tokenizer, cfg.model.max_len, limit=train_limit),
        cfg.train.micro_batch_per_gpu, tokenizer, shuffle=True,
        seed=cfg.train.seed, data_cfg=cfg.data,
    )
    loaders = {"train": train}
    for name, path in (("dev", cfg.data.eval_dev), ("test", cfg.data.eval_test)):
        if Path(path).exists():
            dataset = TranslationDataset(path, tokenizer, cfg.model.max_len)
            loaders[name] = _make_loader(
                dataset, cfg.train.micro_batch_per_gpu, tokenizer, shuffle=False,
                seed=cfg.train.seed, data_cfg=cfg.data,
            )
    return loaders
