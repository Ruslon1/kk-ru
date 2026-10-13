#!/usr/bin/env python3
"""Score NLLB translation shards against a KK-RU TSV reference set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sacrebleu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True, help="KK<TAB>RU reference TSV")
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    refs = []
    with args.references.open(encoding="utf-8") as file:
        for line in file:
            kk, separator, ru = line.rstrip("\n").partition("\t")
            if not separator:
                raise ValueError("reference file must contain KK<TAB>RU rows")
            refs.append((kk, ru))
    hypotheses = {}
    for shard in sorted(args.shards.glob("rank-*.jsonl")):
        with shard.open(encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                index = int(record["index"])
                if not 0 <= index < len(refs):
                    raise ValueError(f"shard index {index} is outside the reference set")
                if record.get("source") != refs[index][0]:
                    raise ValueError(f"source mismatch at index {index}")
                if record.get("status") != "ok" or not record.get("translation", "").strip():
                    raise ValueError(f"translation missing or failed at index {index}")
                if index in hypotheses:
                    raise ValueError(f"duplicate hypothesis index {index}")
                hypotheses[index] = record["translation"].strip()
    missing = sorted(set(range(len(refs))) - hypotheses.keys())
    if missing:
        raise ValueError(f"missing {len(missing)} hypotheses; first missing index {missing[0]}")
    ordered = [hypotheses[i] for i in range(len(refs))]
    references = [pair[1] for pair in refs]
    metrics = {
        "rows": len(refs),
        "bleu": sacrebleu.corpus_bleu(ordered, [references]).score,
        "chrf": sacrebleu.corpus_chrf(ordered, [references]).score,
        "chrf++": sacrebleu.corpus_chrf(ordered, [references], word_order=2).score,
        "ter": sacrebleu.corpus_ter(ordered, [references]).score,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
