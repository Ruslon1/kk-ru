#!/usr/bin/env python3
"""Import the TIL kk-ru corpus from the locally downloaded Google Drive mirror.

The TIL corpus (https://github.com/turkic-interlingua/til-mt) is mirrored on
Google Drive; the link and download layout are in data/README.md. Keep the
kk-ru zips under ``til/`` exactly as downloaded:

    til/
      train/kk-ru-*.zip           # two parts: one holds kk-ru.kk, the other kk-ru.ru
      dev/kk-ru-*.zip             # kk-ru.kk + kk-ru.ru
      test/{bible,ted,x-wmt}/kk-ru-*.zip   # archived, not used (eval is FLORES+)

Every zip contains ``kk-ru/kk-ru.kk`` and ``kk-ru/kk-ru.ru`` (line-aligned
bitext). This script streams the zips (never loads a side into memory) and
writes one gzip-compressed ``data/raw/til*.kk-ru.tsv.gz`` per split (``kk\\tru``).

Usage:
    python scripts/import_til.py              # train -> data/raw/til.kk-ru.tsv.gz
    python scripts/import_til.py --split dev  # dev   -> data/raw/til.dev.kk-ru.tsv.gz
"""
from __future__ import annotations

import argparse
import gzip
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIL = ROOT / "til"
RAW = ROOT / "data" / "raw"

EXPECTED = {"train": 4_400_000}  # ~4.4M; exact count varies with the mirror snapshot


def _find_member(zips: list[Path], suffix: str) -> tuple[Path | None, str | None]:
    """Return (zip_path, member_name) for the first member ending in ``suffix``."""
    for zp in zips:
        with zipfile.ZipFile(zp) as zf:
            for info in zf.infolist():
                if not info.is_dir() and info.filename.endswith(suffix):
                    return zp, info.filename
    return None, None


def _iter_lines(zp: Path, name: str):
    """Yield decoded, newline-stripped lines from one member (streaming)."""
    with zipfile.ZipFile(zp) as zf:
        for raw in zf.open(name):
            yield raw.decode("utf-8", errors="replace").rstrip("\r\n")


def import_split(split: str) -> None:
    out = RAW / ("til.kk-ru.tsv.gz" if split == "train" else f"til.{split}.kk-ru.tsv.gz")
    if out.exists() and out.stat().st_size > 0:
        print(f"skip exists {out.name}")
        return

    zips = sorted((TIL / split).rglob("*.zip"))
    if not zips:
        raise SystemExit(f"no zips under {TIL / split} — see data/README.md for the Drive link")

    kk_zip, kk_name = _find_member(zips, "kk-ru.kk")
    ru_zip, ru_name = _find_member(zips, "kk-ru.ru")
    if not kk_name or not ru_name:
        raise SystemExit(f"kk-ru.kk / kk-ru.ru not found under {TIL / split}")

    RAW.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for kk, ru in zip(_iter_lines(kk_zip, kk_name), _iter_lines(ru_zip, ru_name)):
            f.write(f"{kk}\t{ru}\n")
            n += 1

    note = f" (expected ~{EXPECTED[split]:,})" if split in EXPECTED else ""
    print(f"{out.name}: {n:,} pairs from {kk_name} + {ru_name}{note}")


def main() -> None:
    p = argparse.ArgumentParser(description="Import TIL kk-ru zips into data/raw/*.kk-ru.tsv.gz")
    p.add_argument("--split", choices=["train", "dev"], default="train")
    import_split(p.parse_args().split)


if __name__ == "__main__":
    main()
