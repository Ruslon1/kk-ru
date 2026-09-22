#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf-cache"))

SOURCES = [
    "data/raw/opus.kk-ru.tsv",
    "data/raw/wmt19_crawl.kk-ru.tsv",
    "data/raw/kazparc.kk-ru.tsv",
    "data/raw/til.kk-ru.tsv.gz",
]

TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")


def open_text(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def write_text(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8")
    return open(path, "w", encoding="utf-8")


def iter_pairs(paths):
    for path in paths:
        src = Path(path).name
        with open_text(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if "\t" not in line:
                    continue
                kk, ru, *_ = line.split("\t")
                yield kk, ru, src


def normalize(s):
    s = TAG.sub(" ", s)
    s = unicodedata.normalize("NFC", s)
    return WS.sub(" ", s).strip()


def clean(kk, ru, max_len):
    kk = normalize(kk)
    ru = normalize(ru)
    if not kk or not ru:
        return None
    if kk == ru:
        return None
    if len(kk.split()) <= 1 or len(ru.split()) <= 1:
        return None
    if max_len and (len(kk.split()) > max_len or len(ru.split()) > max_len):
        return None
    return kk, ru


def dedup_key(kk, ru):
    return hashlib.md5(f"{kk}\t{ru}".encode("utf-8")).hexdigest()


def base_pass(inputs, out_path, max_len):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    seen = set()
    total = 0
    drop_clean = 0
    drop_dup = 0
    kept = 0
    kept_by_src = {}
    with write_text(tmp) as f:
        for kk, ru, src in iter_pairs(inputs):
            total += 1
            c = clean(kk, ru, max_len)
            if c is None:
                drop_clean += 1
                continue
            kk, ru = c
            key = dedup_key(kk, ru)
            if key in seen:
                drop_dup += 1
                continue
            seen.add(key)
            f.write(f"{kk}\t{ru}\n")
            kept += 1
            kept_by_src[src] = kept_by_src.get(src, 0) + 1
            if kept % 2_000_000 == 0:
                print(f"  kept {kept:,}", file=sys.stderr)
    tmp.replace(out_path)
    print(f"total {total:,} | after-clean {total - drop_clean:,} | after-dedup {kept:,}")
    print(f"dropped: clean {drop_clean:,} | dup {drop_dup:,}")
    for s in sorted(kept_by_src):
        print(f"  {s}: {kept_by_src[s]:,}")
    return out_path


def langid_pass(in_path):
    import fasttext

    model_path = ROOT / "data" / "tokenizer" / "lid.176.bin"
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print("downloading lid.176.bin", file=sys.stderr)
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
            model_path,
        )
    model = fasttext.load_model(str(model_path))
    in_path = Path(in_path)
    tmp = in_path.with_name(in_path.name + ".tmp")
    kept = 0
    dropped = 0
    with open_text(in_path) as fin, write_text(tmp) as fout:
        for line in fin:
            kk, ru = line.rstrip("\n").split("\t")
            kk_lang = model.predict(kk.strip(), k=1)[0][0]
            ru_lang = model.predict(ru.strip(), k=1)[0][0]
            if kk_lang == "__label__kk" and ru_lang == "__label__ru":
                fout.write(f"{kk}\t{ru}\n")
                kept += 1
            else:
                dropped += 1
            if (kept + dropped) % 1_000_000 == 0:
                print(f"  langid processed {kept + dropped:,}", file=sys.stderr)
    tmp.replace(in_path)
    print(f"langid: kept {kept:,} | dropped {dropped:,}")


def labse_pass(in_path, threshold, device="auto"):
    import torch
    from sentence_transformers import SentenceTransformer

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"labse device: {device}", file=sys.stderr)
    model = SentenceTransformer("sentence-transformers/LaBSE", device=device)
    in_path = Path(in_path)
    tmp = in_path.with_name(in_path.name + ".tmp")
    kept = 0
    dropped = 0
    processed = 0
    kk_buf = []
    ru_buf = []

    def flush(fout):
        nonlocal kept, dropped, processed
        if not kk_buf:
            return
        e_kk = model.encode(
            kk_buf, batch_size=128, convert_to_tensor=True, normalize_embeddings=True
        )
        e_ru = model.encode(
            ru_buf, batch_size=128, convert_to_tensor=True, normalize_embeddings=True
        )
        sims = (e_kk * e_ru).sum(dim=1).tolist()
        for kk, ru, s in zip(kk_buf, ru_buf, sims):
            processed += 1
            if s >= threshold:
                fout.write(f"{kk}\t{ru}\n")
                kept += 1
            else:
                dropped += 1
            if processed % 500_000 == 0:
                print(f"  labse processed {processed:,}", file=sys.stderr)
        kk_buf.clear()
        ru_buf.clear()

    with open_text(in_path) as fin, write_text(tmp) as fout:
        for line in fin:
            kk, ru = line.rstrip("\n").split("\t")
            kk_buf.append(kk.strip())
            ru_buf.append(ru.strip())
            if len(kk_buf) >= 128:
                flush(fout)
        flush(fout)
    tmp.replace(in_path)
    print(f"labse: kept {kept:,} | dropped {dropped:,}")


def main():
    p = argparse.ArgumentParser(description="Filter raw KK-RU bitext.")
    p.add_argument("--in", dest="inputs", nargs="+", default=SOURCES)
    p.add_argument("--out", type=str, default="data/filtered/train.kk-ru.tsv")
    p.add_argument("--labse-threshold", type=float, default=0.55)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--skip-base", action="store_true")
    p.add_argument("--skip-langid", action="store_true")
    p.add_argument("--skip-labse", action="store_true")
    p.add_argument("--labse-device", type=str, default="auto", choices=["auto", "cpu", "mps"])
    args = p.parse_args()

    out = args.out
    if args.skip_base:
        if not Path(out).exists():
            raise SystemExit(f"missing {out}")
        print(f"skip base pass, reuse {out}")
    else:
        out = base_pass(args.inputs, args.out, args.max_len)

    if not args.skip_langid:
        try:
            langid_pass(out)
        except ImportError:
            print("WARN: fasttext not installed, skipping langid", file=sys.stderr)

    if not args.skip_labse:
        try:
            labse_pass(out, args.labse_threshold, device=args.labse_device)
        except ImportError:
            print("WARN: sentence-transformers not installed, skipping labse", file=sys.stderr)

    print("wrote", out)


if __name__ == "__main__":
    main()
