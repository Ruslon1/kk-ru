#!/usr/bin/env python3
"""Stream Kazakh monolingual text from Hugging Face and extract clean sentences."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sqlite3
import sys
import unicodedata
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "kz-transformers/multidomain-kazakh-dataset"
DEFAULT_CONFIG = ""
MDBKD_REVISION = "7a1fcdf9830b1c34b44b3038aafb672447f41890"
MDBKD_FILES = (
    "leipzig.csv",
    "cc100-monolingual-crawled-data.csv",
    "oscar.csv",
    "kazakhNews.csv",
    "kazakhBooks.csv",
)
DEFAULT_OUTPUT = ROOT / "data" / "monolingual" / "kazakh.sentences.txt"
WS = re.compile(r"\s+")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])(?:\s+|(?=[«“\"(A-ZА-ЯӘҒҚҢӨҰҮҺІ]))|\n+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    return WS.sub(" ", text).strip()


def split_sentences(text: str) -> Iterator[str]:
    for chunk in SENTENCE_BOUNDARY.split(text):
        sentence = normalize(chunk)
        if sentence:
            yield sentence


def looks_like_kazakh(sentence: str, min_chars: int, max_chars: int) -> bool:
    if not min_chars <= len(sentence) <= max_chars:
        return False
    if sentence.count("http://") + sentence.count("https://"):
        return False
    letters = [char for char in sentence if char.isalpha()]
    if len(letters) < 3:
        return False
    kazakh_letters = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
    cyrillic = sum(char.lower() in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" for char in letters)
    kazakh = sum(char in kazakh_letters for char in letters)
    return kazakh > 0 and cyrillic + kazakh >= max(3, len(letters) // 2)


def sentence_hash(sentence: str) -> bytes:
    return hashlib.sha256(normalize(sentence).casefold().encode("utf-8")).digest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_existing(conn: sqlite3.Connection, paths: Iterable[Path]) -> int:
    count = 0
    for path in paths:
        if not path.exists():
            print(f"WARN missing existing corpus: {path}", file=sys.stderr)
            continue
        with path.open(encoding="utf-8", errors="replace") as file:
            for line in file:
                source = line.rstrip("\n").split("\t", 1)[0]
                for sentence in split_sentences(source):
                    conn.execute("INSERT OR IGNORE INTO seen(value) VALUES (?)", (sentence_hash(sentence),))
                    count += 1
        conn.commit()
    return count


def load_rows(dataset: str, config: str, split: str):
    if dataset == DEFAULT_DATASET:
        for name in MDBKD_FILES:
            url = f"https://huggingface.co/datasets/{dataset}/resolve/{MDBKD_REVISION}/{name}"
            print(f"streaming {name}", file=sys.stderr, flush=True)
            with urllib.request.urlopen(url) as response:
                text = io.TextIOWrapper(response, encoding="utf-8-sig", newline="")
                yield from csv.DictReader(text)
        return
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install datasets first: pip install datasets") from exc
    kwargs = {"path": dataset, "split": split, "streaming": True}
    if config:
        kwargs["name"] = config
    return load_dataset(**kwargs)


def extract(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    db_path = output.with_suffix(output.suffix + ".seen.sqlite3")
    if db_path.exists() and not args.resume:
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS seen (value BLOB PRIMARY KEY)")
    existing = add_existing(conn, [Path(path) for path in args.existing]) if not args.resume else 0
    written = add_existing(conn, [output]) if args.resume and output.exists() else 0
    documents = sentences = 0
    with output.open("a" if args.resume else "w", encoding="utf-8") as file:
        for row in load_rows(args.dataset, args.config, args.split):
            documents += 1
            if documents % 100_000 == 0:
                print(f"scanned {documents:,} documents | written {written:,}", file=sys.stderr)
            if row.get("predicted_language") not in (None, "kaz"):
                continue
            text = row.get(args.text_field, "")
            for sentence in split_sentences(text):
                if not looks_like_kazakh(sentence, args.min_chars, args.max_chars):
                    continue
                sentences += 1
                before = conn.total_changes
                conn.execute("INSERT OR IGNORE INTO seen(value) VALUES (?)", (sentence_hash(sentence),))
                if conn.total_changes == before:
                    continue
                file.write(sentence + "\n")
                written += 1
                if written % args.commit_every == 0:
                    conn.commit()
                    file.flush()
                    print(f"written {written:,} | documents {documents:,}", file=sys.stderr)
                if args.limit and written >= args.limit:
                    break
            if args.limit and written >= args.limit:
                break
    conn.commit()
    conn.close()
    manifest = {
        "dataset": args.dataset,
        "dataset_revision": MDBKD_REVISION if args.dataset == DEFAULT_DATASET else None,
        "source_files": list(MDBKD_FILES) if args.dataset == DEFAULT_DATASET else None,
        "config": args.config,
        "split": args.split,
        "text_field": args.text_field,
        "output": str(output),
        "documents_read": documents,
        "sentences_after_filter": sentences,
        "sentences_written": written,
        "output_sha256": file_hash(output),
        "existing_paths": args.existing,
        "existing_sentences_indexed": existing,
        "license_note": "Verify the dataset license before redistributing or training a commercial model.",
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--existing",
        action="append",
        default=[
            str(ROOT / "data/filtered/train.kk-ru.tsv"),
            str(ROOT / "data/eval/flores_plus/dev.kk-ru.tsv"),
            str(ROOT / "data/eval/flores_plus/devtest.kk-ru.tsv"),
        ],
    )
    parser.add_argument("--limit", type=int, default=4_000_000)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=1_000)
    parser.add_argument("--commit-every", type=int, default=10_000)
    parser.add_argument("--resume", action="store_true")
    extract(parser.parse_args())


if __name__ == "__main__":
    main()
