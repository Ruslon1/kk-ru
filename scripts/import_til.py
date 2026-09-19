#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIL = ROOT / "til"
RAW = ROOT / "data" / "raw"

EXPECTED = {"train": 4_400_000}


def _find_member(zips: list[Path], suffix: str) -> tuple[Path | None, str | None]:
    for zp in zips:
        with zipfile.ZipFile(zp) as zf:
            for info in zf.infolist():
                if not info.is_dir() and info.filename.endswith(suffix):
                    return zp, info.filename
    return None, None


def _iter_lines(zp: Path, name: str):
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
        raise SystemExit(f"no zips under {TIL / split}")

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
