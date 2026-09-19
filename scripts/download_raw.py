#!/usr/bin/env python3
"""Download raw KK–RU bitext for the kazRush-matched mix (+ FLORES+ eval).

Usage:
  python scripts/download_raw.py --flores
  python scripts/download_raw.py --opus
  python scripts/download_raw.py --wmt19
  python scripts/download_raw.py --til       # TIL: Drive mirror + scripts/import_til.py
  python scripts/download_raw.py --kazparc   # needs HF login + accepted terms
  python scripts/download_raw.py --all-open  # flores + opus + wmt19 (no gated)

Writes TSV files: data/raw/<name>.kk-ru.tsv  (kk \\t ru)
"""

from __future__ import annotations

import argparse
import gzip
import io
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EVAL = ROOT / "data" / "eval" / "flores_plus"

OPUS_MOSES = [
    # latest moses dumps from opus.nlpl.eu API (kk-ru)
    "https://object.pouta.csc.fi/OPUS-GNOME/v1/moses/kk-ru.txt.zip",
    "https://object.pouta.csc.fi/OPUS-KDE4/v2/moses/kk-ru.txt.zip",
    "https://object.pouta.csc.fi/OPUS-MultiCCAligned/v1.1/moses/kk-ru.txt.zip",
    "https://object.pouta.csc.fi/OPUS-NeuLab-TedTalks/v1/moses/kk-ru.txt.zip",
    "https://object.pouta.csc.fi/OPUS-News-Commentary/v16/moses/kk-ru.txt.zip",
    "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/moses/kk-ru.txt.zip",
    "https://object.pouta.csc.fi/OPUS-QED/v2.0a/moses/kk-ru.txt.zip",
    "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/kk-ru.txt.zip",
    "https://object.pouta.csc.fi/OPUS-Tatoeba/v2026-07-08/moses/kk-ru.txt.zip",
    "https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/moses/kk-ru.txt.zip",
    "https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/kk-ru.txt.zip",
    "https://object.pouta.csc.fi/OPUS-XLEnt/v1.2/moses/kk-ru.txt.zip",
    "https://object.pouta.csc.fi/OPUS-wikimedia/v20260327/moses/kk-ru.txt.zip",
]

WMT19_CRAWL = "http://data.statmt.org/wmt19/translation-task/crawl.kk-ru.gz"

# TIL corpus: distributed via the Google Drive mirror (see data/README.md) and
# imported from local zips with scripts/import_til.py. License: CC BY-NC-SA 4.0.

FLORES_REPO = "openlanguagedata/flores_plus"
FLORES_LANGS = ("kaz_Cyrl", "rus_Cyrl")


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip exists {dest.name}")
        return dest
    print(f"GET {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def _write_tsv(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for kk, ru in pairs:
            kk = " ".join(str(kk).split())
            ru = " ".join(str(ru).split())
            if kk and ru:
                f.write(f"{kk}\t{ru}\n")
    print(f"wrote {path} ({len(pairs):,} lines)")


def _opus_zip_name(url: str) -> str:
    parts = url.rstrip("/").split("/")
    # parts[-4]: corpus name (e.g. OPUS-GNOME)
    # parts[-3]: version (e.g. v1)
    # parts[-1]: filename (e.g. kk-ru.txt.zip)
    return f"{parts[-4]}_{parts[-3]}_{parts[-1]}"


def download_opus() -> None:
    staging = RAW / "_opus_zip"
    staging.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[str, str]] = []
    for url in OPUS_MOSES:
        name = _opus_zip_name(url)
        zpath = staging / name
        try:
            _download(url, zpath)
        except Exception as e:
            print(f"WARN skip {url}: {e}", file=sys.stderr)
            continue
        try:
            with zipfile.ZipFile(zpath) as zf:
                names = zf.namelist()
                kk_name = next((n for n in names if n.endswith(".kk")), None)
                ru_name = next((n for n in names if n.endswith(".ru")), None)
                if not kk_name or not ru_name:
                    print(f"WARN no .kk/.ru in {zpath}: {names[:8]}", file=sys.stderr)
                    continue
                kk_lines = zf.read(kk_name).decode("utf-8", errors="replace").splitlines()
                ru_lines = zf.read(ru_name).decode("utf-8", errors="replace").splitlines()
        except zipfile.BadZipFile as e:
            print(f"WARN bad zip {zpath}: {e}", file=sys.stderr)
            continue
        n = min(len(kk_lines), len(ru_lines))
        print(f"  {zpath.name}: {n:,} pairs")
        pairs.extend(zip(kk_lines[:n], ru_lines[:n]))
    _write_tsv(RAW / "opus.kk-ru.tsv", pairs)


def download_wmt19() -> None:
    target = RAW / "wmt19_crawl.kk-ru.tsv"
    if target.exists() and target.stat().st_size > 0:
        print(f"skip exists {target.name}")
        return

    gz = RAW / "_wmt19" / "crawl.kk-ru.gz"
    _download(WMT19_CRAWL, gz)
    pairs: list[tuple[str, str]] = []
    with gzip.open(gz, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    _write_tsv(target, pairs)


def _load_kazparc_split(config: str) -> list[tuple[str, str]]:
    from datasets import load_dataset

    print(f"loading issai/kazparc {config}")
    ds = load_dataset("issai/kazparc", config, split="train")
    pairs: list[tuple[str, str]] = []
    for row in ds:
        kk = (row.get("kk") or "").strip()
        ru = (row.get("ru") or "").strip()
        if kk and ru:
            pairs.append((kk, ru))
    print(f"  {config}: {len(pairs):,} kk-ru pairs / {len(ds):,} total")
    return pairs


def download_kazparc() -> None:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("pip install datasets huggingface_hub") from e

    human = _load_kazparc_split("kazparc_raw")
    _write_tsv(RAW / "kazparc_human.kk-ru.tsv", human)

    sync = _load_kazparc_split("sync_raw")
    _write_tsv(RAW / "kazparc_sync.kk-ru.tsv", sync)

    _write_tsv(RAW / "kazparc.kk-ru.tsv", human + sync)


def download_flores() -> None:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("pip install datasets huggingface_hub") from e

    EVAL.mkdir(parents=True, exist_ok=True)
    for split in ("dev", "devtest"):
        parts = {}
        for lang in FLORES_LANGS:
            print(f"FLORES+ {split} {lang}")
            ds = load_dataset(FLORES_REPO, lang, split=split)
            parts[lang] = [row["text"] for row in ds]
        n = min(len(parts["kaz_Cyrl"]), len(parts["rus_Cyrl"]))
        pairs = list(zip(parts["kaz_Cyrl"][:n], parts["rus_Cyrl"][:n]))
        _write_tsv(EVAL / f"{split}.kk-ru.tsv", pairs)


def download_til() -> None:
    """TIL comes from a local Drive-mirror download, imported by import_til.py."""
    print(
        "TIL corpus: download the kk-ru zips from the Google Drive mirror into til/\n"
        "(link and layout in data/README.md), then import them:\n\n"
        "  python scripts/import_til.py\n"
    )


def til_from(src: Path) -> None:
    """Merge two aligned files (*.kk + *.ru) or a tsv into opus-style tsv."""
    src = src.expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"missing {src}")
    pairs: list[tuple[str, str]] = []
    if src.is_file() and src.suffix in {".tsv", ".gz"}:
        opener = gzip.open if src.suffix == ".gz" else open
        with opener(src, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
    else:
        kks = sorted(src.rglob("*.kk"))
        rus = sorted(src.rglob("*.ru"))
        if not kks or not rus:
            raise SystemExit(f"no *.kk/*.ru under {src}")
        for kkf, ruf in zip(kks, rus):
            kk_lines = kkf.read_text("utf-8", errors="replace").splitlines()
            ru_lines = ruf.read_text("utf-8", errors="replace").splitlines()
            n = min(len(kk_lines), len(ru_lines))
            pairs.extend(zip(kk_lines[:n], ru_lines[:n]))
            print(f"  {kkf.name}: {n:,}")
    _write_tsv(RAW / "til.kk-ru.tsv", pairs)


def main() -> None:
    p = argparse.ArgumentParser(description="Download KK-RU parallel datasets.")
    p.add_argument("--flores", action="store_true", help="Download FLORES+ eval (dev, devtest)")
    p.add_argument("--opus", action="store_true", help="Download OPUS kk-ru moses dumps")
    p.add_argument("--wmt19", action="store_true", help="Download WMT19 crawl")
    p.add_argument("--kazparc", action="store_true", help="Download KazParC (human + sync)")
    p.add_argument("--til", action="store_true", help="Import TIL kk-ru from local Drive zips (see data/README.md)")
    p.add_argument("--til-from", type=Path, help="Directory or file to import TIL bitext from")
    p.add_argument("--all-open", action="store_true", help="Download all open datasets (flores + opus + wmt19)")
    args = p.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)

    ran = False
    if args.all_open or args.flores:
        download_flores()
        ran = True
    if args.all_open or args.opus:
        download_opus()
        ran = True
    if args.all_open or args.wmt19:
        download_wmt19()
        ran = True
    if args.kazparc:
        download_kazparc()
        ran = True
    if args.til:
        download_til()
        ran = True
    if args.til_from:
        til_from(args.til_from)
        ran = True
    if not ran:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
