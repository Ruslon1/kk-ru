#!/usr/bin/env python3
"""Quick inventory for downloaded KK–RU bitext (no GPU)."""

from __future__ import annotations

import gzip
import hashlib
from collections import Counter
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EVAL = ROOT / "data" / "eval"
OUT = ROOT / "data" / "INVENTORY.md"


def iter_pairs(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            kk, ru, *rest = line.split("\t")
            yield kk, ru


def summarize(path: Path) -> dict:
    n = 0
    empty = 0
    identical = 0
    kk_chars, ru_chars, kk_toks, ru_toks = [], [], [], []
    seen = {}
    dups = 0
    for kk, ru in iter_pairs(path):
        n += 1
        if not kk.strip() or not ru.strip():
            empty += 1
        if kk.strip() == ru.strip():
            identical += 1
        kk_chars.append(len(kk))
        ru_chars.append(len(ru))
        kk_toks.append(len(kk.split()))
        ru_toks.append(len(ru.split()))
        key = (kk.strip(), ru.strip())
        if key in seen:
            dups += 1
        else:
            seen[key] = 1
        if n % 2_000_000 == 0:
            print(f"  {path.name}: {n:,}")

    def pct(xs, p):
        if not xs:
            return 0
        ys = sorted(xs)
        i = min(len(ys) - 1, int(round((p / 100) * (len(ys) - 1))))
        return ys[i]

    return {
        "file": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "n": n,
        "unique": n - dups,
        "exact_dups": dups,
        "empty": empty,
        "identical_kk_ru": identical,
        "kk_char_mean": mean(kk_chars) if kk_chars else 0,
        "ru_char_mean": mean(ru_chars) if ru_chars else 0,
        "kk_tok_mean": mean(kk_toks) if kk_toks else 0,
        "ru_tok_mean": mean(ru_toks) if ru_toks else 0,
        "kk_tok_p50": median(kk_toks) if kk_toks else 0,
        "ru_tok_p50": median(ru_toks) if ru_toks else 0,
        "kk_tok_p95": pct(kk_toks, 95),
        "ru_tok_p95": pct(ru_toks, 95),
        "kk_tok_max": max(kk_toks) if kk_toks else 0,
        "ru_tok_max": max(ru_toks) if ru_toks else 0,
        "hashes": {hashlib.md5(f"{k}\t{v}".encode()).hexdigest() for k, v in seen},
    }


def mb(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MB"


def main() -> None:
    files = [
        RAW / "opus.kk-ru.tsv",
        RAW / "wmt19_crawl.kk-ru.tsv",
        RAW / "kazparc_human.kk-ru.tsv",
        RAW / "kazparc_sync.kk-ru.tsv",
        RAW / "kazparc.kk-ru.tsv",
        RAW / "til.kk-ru.tsv.gz",
        EVAL / "flores_plus" / "dev.kk-ru.tsv",
        EVAL / "flores_plus" / "devtest.kk-ru.tsv",
        RAW / "extra" / "kazlit.kk-ru.tsv",
        RAW / "extra" / "kaznu.kk-ru.tsv",
    ]
    rows = []
    for p in files:
        if not p.exists():
            print("MISSING", p)
            continue
        print("stats", p)
        rows.append(summarize(p))

    # exact overlap vs WMT19 (largest)
    wmt = next((r for r in rows if "wmt19" in r["file"]), None)
    lines = ["# Inventory KK–RU (raw, до фильтра)", ""]
    lines.append("| файл | размер | пар | unique | exact dups | identical kk=ru | kk tok mean/p50/p95 | ru tok mean/p50/p95 |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|")
    for r in rows:
        lines.append(
            f"| `{r['file']}` | {mb(r['bytes'])} | {r['n']:,} | {r['unique']:,} | {r['exact_dups']:,} | {r['identical_kk_ru']:,} | "
            f"{r['kk_tok_mean']:.1f}/{r['kk_tok_p50']:.0f}/{r['kk_tok_p95']} | "
            f"{r['ru_tok_mean']:.1f}/{r['ru_tok_p50']:.0f}/{r['ru_tok_p95']} |"
        )

    if wmt:
        lines += ["", "## Exact overlap with WMT19 crawl", ""]
        lines.append("| файл | ∩ WMT19 | доля файла |")
        lines.append("|---|---:|---:|")
        for r in rows:
            if r is wmt:
                continue
            inter = len(r["hashes"] & wmt["hashes"])
            frac = inter / r["unique"] if r["unique"] else 0
            lines.append(f"| `{r['file']}` | {inter:,} | {frac:.1%} |")

    # opus vs kazparc human
    by = {r["file"]: r for r in rows}
    def inter(a, b):
        if a not in by or b not in by:
            return
        n = len(by[a]["hashes"] & by[b]["hashes"])
        lines.append(f"- `{a}` ∩ `{b}` = **{n:,}**")

    lines += ["", "## Other exact overlaps", ""]
    inter("data/raw/opus.kk-ru.tsv", "data/raw/kazparc_human.kk-ru.tsv")
    inter("data/raw/opus.kk-ru.tsv", "data/raw/kazparc_sync.kk-ru.tsv")
    inter("data/raw/kazparc_human.kk-ru.tsv", "data/raw/kazparc_sync.kk-ru.tsv")
    inter("data/eval/flores_plus/devtest.kk-ru.tsv", "data/raw/wmt19_crawl.kk-ru.tsv")
    inter("data/eval/flores_plus/devtest.kk-ru.tsv", "data/raw/opus.kk-ru.tsv")
    inter("data/eval/flores_plus/devtest.kk-ru.tsv", "data/raw/kazparc.kk-ru.tsv")

    lines += [
        "",
        "## Notes",
        "",
        "- kazRush raw mix: OPUS ~718k + kazparc ~2,150k + WMT19 5,063k + TIL ~4,403k.",
        "- WMT19 crawl: скачался ровно **5,063,666** пар — совпало с карточкой kazRush.",
        "- KazParC: суммарно **2,168,968** пар (human 371,902 + SynC 1,797,066; у kazRush 2,150k после дополнительной чистки).",
        "- OPUS moses: собрано **789,436** пар из 13 подкорпусов (GNOME, KDE4, MultiCCAligned, NeuLab-TedTalks, News-Commentary, OpenSubtitles, QED, TED2020, Tatoeba, Ubuntu, WikiMatrix, XLEnt, wikimedia).",
        "- FLORES+: эталонный eval из `openlanguagedata/flores_plus` (dev: 997, devtest: 1,012).",
        "- Extra (опциональные для M_extra): `kazlit.kk-ru.tsv` (54.5k) и `kaznu.kk-ru.tsv` (209.7k).",
        "- **TIL**: скачан из [Drive-зеркала](https://drive.google.com/drive/folders/1kUp_vpDsNUZvVC6HvwNxGn7ImwnCfM1E) в `til/` (train в 2 zip + dev + test{bible,ted,x-wmt}). Импорт: `python scripts/import_til.py` → `data/raw/til.kk-ru.tsv.gz`. Лицензия CC BY-NC-SA 4.0.",
        "- Это сырой корпус. Следующий этап — фильтрация по пайплайну kazRush (дедуп, чистка мусора, FastText langid, LaBSE, OpusFilter).",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
