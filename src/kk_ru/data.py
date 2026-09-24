from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .config import Config


def iter_pairs(tsv_path: str) -> Iterator[tuple[str, str]]:
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            yield parts[0], parts[1]


def _tokenize_stream(tsv_path: str, tokenizer, max_len: int, chunk: int = 200_000):
    src_parts, tgt_parts = [], []
    src_len_parts, tgt_len_parts = [], []
    cur_src, cur_tgt = [], []
    cur_sl, cur_tl = [], []

    def flush():
        nonlocal cur_src, cur_tgt, cur_sl, cur_tl
        if cur_sl:
            src_parts.append(np.asarray(cur_src, dtype=np.int32))
            tgt_parts.append(np.asarray(cur_tgt, dtype=np.int32))
            src_len_parts.append(np.asarray(cur_sl, dtype=np.int32))
            tgt_len_parts.append(np.asarray(cur_tl, dtype=np.int32))
            cur_src, cur_tgt = [], []
            cur_sl, cur_tl = [], []

    for kk, ru in iter_pairs(tsv_path):
        s = tokenizer.encode(kk, add_bos=True)[:max_len]
        t = tokenizer.encode(ru, add_eos=True)[:max_len]
        cur_src.extend(s)
        cur_tgt.extend(t)
        cur_sl.append(len(s))
        cur_tl.append(len(t))
        if len(cur_sl) >= chunk:
            flush()
    flush()

    empty = np.array([], dtype=np.int32)
    src_flat = np.concatenate(src_parts) if src_parts else empty
    tgt_flat = np.concatenate(tgt_parts) if tgt_parts else empty
    src_lens = np.concatenate(src_len_parts) if src_len_parts else empty
    tgt_lens = np.concatenate(tgt_len_parts) if tgt_len_parts else empty
    src_off = np.zeros(len(src_lens) + 1, dtype=np.int64)
    tgt_off = np.zeros(len(tgt_lens) + 1, dtype=np.int64)
    np.cumsum(src_lens, out=src_off[1:])
    np.cumsum(tgt_lens, out=tgt_off[1:])
    return src_flat, src_off, src_lens, tgt_flat, tgt_off, tgt_lens


class TranslationDataset(Dataset):
    """Tokenized bitext stored as flat int32 arrays with offset indexes."""

    def __init__(self, tsv_path: str, tokenizer, max_len: int, cache: bool = True):
        self.max_len = max_len
        cache_path = Path(str(tsv_path) + ".tok.npz")
        if cache and cache_path.exists():
            z = np.load(cache_path)
            self.src_flat = z["src_flat"]
            self.src_off = z["src_off"]
            self.tgt_flat = z["tgt_flat"]
            self.tgt_off = z["tgt_off"]
            self.src_lens = z["src_lens"]
            self.tgt_lens = z["tgt_lens"]
        else:
            self.src_flat, self.src_off, self.src_lens, self.tgt_flat, self.tgt_off, self.tgt_lens = _tokenize_stream(
                tsv_path, tokenizer, max_len
            )
            if cache:
                np.savez_compressed(
                    cache_path,
                    src_flat=self.src_flat,
                    src_off=self.src_off,
                    src_lens=self.src_lens,
                    tgt_flat=self.tgt_flat,
                    tgt_off=self.tgt_off,
                    tgt_lens=self.tgt_lens,
                )

    def __len__(self) -> int:
        return len(self.src_lens)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        s0, s1 = self.src_off[idx], self.src_off[idx + 1]
        t0, t1 = self.tgt_off[idx], self.tgt_off[idx + 1]
        return (
            self.src_flat[s0:s1].tolist(),
            self.tgt_flat[t0:t1].tolist(),
        )

    def lengths(self) -> np.ndarray:
        return self.src_lens + self.tgt_lens


class LengthBatchSampler(Sampler):
    """Sort by length, group into batches, shuffle the batch order."""

    def __init__(
        self,
        lengths,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        n = len(self.lengths)
        order = sorted(range(n), key=lambda i: self.lengths[i])
        start = 0
        batches = []
        while start < n:
            end = min(start + self.batch_size, n)
            if self.drop_last and end - start < self.batch_size:
                break
            batches.append(order[start:end])
            start = end
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(batches)
        batches = batches[self.rank :: self.world_size]
        for b in batches:
            yield b

    def __len__(self) -> int:
        n = len(self.lengths)
        nb = n // self.batch_size
        if not self.drop_last and n % self.batch_size:
            nb += 1
        return (nb - self.rank + self.world_size - 1) // self.world_size


def collate_fn(batch: list[tuple[list[int], list[int]]], pad_id: int) -> dict:
    src_lens = [len(s) for s, _ in batch]
    tgt_lens = [len(t) for _, t in batch]
    src_max = max(src_lens)
    tgt_max = max(tgt_lens)
    bsz = len(batch)

    src_ids = torch.full((bsz, src_max), pad_id, dtype=torch.long)
    tgt_ids = torch.full((bsz, tgt_max), pad_id, dtype=torch.long)
    src_mask = torch.zeros((bsz, src_max), dtype=torch.bool)
    for i, (s, t) in enumerate(batch):
        src_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        tgt_ids[i, : len(t)] = torch.tensor(t, dtype=torch.long)
        src_mask[i, : len(s)] = True

    labels = tgt_ids[:, 1:].clone()
    return {
        "src_ids": src_ids,
        "src_mask": src_mask,
        "tgt_ids": tgt_ids,
        "labels": labels,
    }


def _make_loader(dataset: TranslationDataset, batch_size: int, tokenizer, shuffle: bool, seed: int, rank: int, world_size: int):
    if shuffle:
        sampler = LengthBatchSampler(
            dataset.lengths(), batch_size, shuffle=True, seed=seed, drop_last=True, rank=rank, world_size=world_size
        )
        return torch.utils.data.DataLoader(
            dataset, batch_sampler=sampler, collate_fn=lambda b: collate_fn(b, tokenizer.pad_id), num_workers=0
        )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer.pad_id), num_workers=0
    )


def build_dataloaders(cfg: Config, tokenizer, rank: int = 0, world_size: int = 1):
    from torch.utils.data import DataLoader

    train = _make_loader(
        TranslationDataset(cfg.data.train_tsv, tokenizer, cfg.model.max_len),
        cfg.train.micro_batch_per_gpu,
        tokenizer,
        shuffle=True,
        seed=cfg.train.seed,
        rank=rank,
        world_size=world_size,
    )

    loaders = {"train": train}
    for name, path in (("dev", cfg.data.eval_dev), ("test", cfg.data.eval_test)):
        if Path(path).exists():
            ds = TranslationDataset(path, tokenizer, cfg.model.max_len, cache=False)
            loaders[name] = DataLoader(
                ds, batch_size=cfg.train.micro_batch_per_gpu, shuffle=False,
                collate_fn=lambda b: collate_fn(b, tokenizer.pad_id), num_workers=0,
            )
    return loaders
