# NLLB corpus expansion and P0 continuation

Run the commands from the repository root on the 6x RTX 4090 server. The user
provides and extracts the monolingual archive. NLLB-200 3.3B is CC-BY-NC 4.0;
confirm the project is within that license before generating the full corpus.

## 1. Prepare the server

Use the project's supported Python environment and install its dependencies.
Make sure the extracted input has one Kazakh Cyrillic sentence per line and that
the existing tokenizer and P0 checkpoint are present. Keep the original P0
checkpoint intact.

```bash
mkdir -p models data/synthetic/nllb-rank-shards data/merged \
  checkpoints/p0-merged-3ep runs/p0-merged-3ep reports/p0-before reports/p0-after
hf download facebook/nllb-200-3.3B --local-dir models/nllb-200-3.3B
nvidia-smi
```

Find the server checkpoint from the interrupted P0 run and verify it contains
`meta.json`, `model.pt`, and Accelerate state files. Do not use the local
`checkpoints/step-471` without confirming that it is the intended ~1.5-epoch
checkpoint. Set its path explicitly below:

```bash
export P0_CHECKPOINT=/path/to/server/checkpoints/step-N
test -s "$P0_CHECKPOINT/meta.json" && test -s "$P0_CHECKPOINT/model.pt"
cat "$P0_CHECKPOINT/meta.json"
```

## 2. Establish translation quality and throughput

First score NLLB on `dev`, then use a small representative slice of the
monolingual corpus for throughput and output inspection. Sweep beam size and
batch size on this limited run; larger beam counts improve search but reduce
throughput. Keep a clean output directory for each setting.

```bash
torchrun --standalone --nproc_per_node=6 scripts/translate_nllb.py \
  --input data/eval/flores_plus/dev.kk-ru.tsv \
  --out-dir data/synthetic/nllb-dev --model models/nllb-200-3.3B \
  --batch-size 16 --bucket-size 2048 --num-beams 4
python scripts/score_nllb.py \
  --references data/eval/flores_plus/dev.kk-ru.tsv \
  --shards data/synthetic/nllb-dev --out reports/nllb-dev.json

torchrun --standalone --nproc_per_node=6 scripts/translate_nllb.py \
  --input data/monolingual/kazakh.sentences.txt \
  --out-dir data/synthetic/nllb-throughput --model models/nllb-200-3.3B \
  --limit 12000 --batch-size 16 --bucket-size 2048 --num-beams 4
```

Check per-rank `*.metrics.json`, the aggregate wall-clock rows/sec, GPU memory,
and sample translations. Adjust `--batch-size` until all six GPUs stay busy
without OOM. The translator recursively splits failed/OOM batches and records
individual unrecoverable rows as errors; the shard validator rejects them.

## 3. Translate the complete corpus

After choosing settings, run the full job. The rank is derived from the
`torchrun` environment and each input index belongs to exactly one rank. Each
rank appends and fsyncs completed batches to its own shard. Re-run the same
command after interruption; successful rows are skipped and prior error rows
are removed so they can be retried.

```bash
torchrun --standalone --nproc_per_node=6 scripts/translate_nllb.py \
  --input data/monolingual/kazakh.sentences.txt \
  --out-dir data/synthetic/nllb-rank-shards \
  --model models/nllb-200-3.3B \
  --batch-size 16 --bucket-size 2048 --num-beams 4
```

Resume by repeating that command with the same input, model, and output
directory. Do not run two jobs against the same shard directory concurrently.
After the job finishes, validate full coverage before any merge:

```bash
python scripts/validate_nllb_shards.py \
  --input data/monolingual/kazakh.sentences.txt \
  --shards data/synthetic/nllb-rank-shards \
  --db /tmp/nllb-validation.sqlite
```

The validator must exit successfully with every non-empty input represented
once, all translations non-empty and successful, and no missing, duplicate, or
out-of-range indices. Review a random sample for language/script and quality.

## 4. Score NLLB on devtest and merge

Keep devtest for final reporting; use dev for tuning. Translate the devtest TSV
with the same model settings, then save the score report:

```bash
torchrun --standalone --nproc_per_node=6 scripts/translate_nllb.py \
  --input data/eval/flores_plus/devtest.kk-ru.tsv \
  --out-dir data/synthetic/nllb-devtest --model models/nllb-200-3.3B \
  --batch-size 16 --bucket-size 2048 --num-beams 4
python scripts/score_nllb.py \
  --references data/eval/flores_plus/devtest.kk-ru.tsv \
  --shards data/synthetic/nllb-devtest --out reports/nllb-devtest.json
```

Merge only validated full-corpus shards. Output is rebuilt via temporary files,
and exact normalized `(Kazakh, Russian)` pairs are deduplicated against both
the base corpus and other generated pairs. A failed merge leaves the prior
published output intact.

```bash
python scripts/merge_synthetic.py \
  --base data/filtered/train.kk-ru.tsv \
  --translations data/synthetic/nllb-rank-shards \
  --out data/merged/train.kk-ru.tsv \
  --db data/merged/train.kk-ru.sqlite
```

Record `base_rows`, `translation_rows`, `written`, `duplicates`, and
`dropped_invalid_pairs` printed by the merge. Validate the merged TSV has two
columns and keep the generated output and SQLite index for reproducibility.

## 5. Evaluate the P0 checkpoint before training

Evaluate the server's interrupted checkpoint with the existing P0 config and
tokenizer on both splits. Reports go into a new directory and do not overwrite
older evaluations.

```bash
python scripts/benchmark.py --configs configs/p0.yaml \
  --checkpoints "$P0_CHECKPOINT/model.pt" --split dev \
  --out reports/p0-before/dev --report-dir reports/p0-before/dev --spbleu
python scripts/benchmark.py --configs configs/p0.yaml \
  --checkpoints "$P0_CHECKPOINT/model.pt" --split devtest \
  --out reports/p0-before/devtest --report-dir reports/p0-before/devtest --spbleu
```

The benchmark reports BLEU, chrF, chrF++, TER, and spBLEU when the FLORES
tokenizer is available. `--out` stores the comparison table and `--report-dir`
stores hypotheses and per-split metrics. The training JSONL logs token loss.
Evaluation loss is not currently computed by the benchmark, so compare the
same translation metrics and training loss curves; do not present loss as a
devtest metric.

## 6. Train three full merged-data epochs

`--resume-new-data` restores model, optimizer, and other Accelerate state from
the checkpoint, resets the old data cursor, and starts a fresh cosine schedule
for the configured three complete epochs. Keep the original tokenizer and
override checkpoint/log/report paths so this phase cannot prune or overwrite
the old run.

```bash
accelerate launch --multi_gpu --num_processes 6 -m kk_ru.train \
  --config configs/p0.yaml \
  --resume-new-data "$P0_CHECKPOINT" \
  --opts data.train_tsv=data/merged/train.kk-ru.tsv \
    train.epochs=3 train.max_steps=null \
    paths.checkpoints=checkpoints/p0-merged-3ep \
    paths.logs=runs/p0-merged-3ep paths.reports=reports/p0-after
```

For a preemption during this new phase, resume exactly from its latest
`step-N` checkpoint with `--resume` (not `--resume-new-data`) and the same
config/overrides. Metadata stores phase progress so the trainer resumes the
merged-data epoch and batch cursor with its in-progress scheduler:

```bash
accelerate launch --multi_gpu --num_processes 6 -m kk_ru.train \
  --config configs/p0.yaml --resume checkpoints/p0-merged-3ep/step-N \
  --opts data.train_tsv=data/merged/train.kk-ru.tsv \
    train.epochs=3 train.max_steps=null \
    paths.checkpoints=checkpoints/p0-merged-3ep \
    paths.logs=runs/p0-merged-3ep paths.reports=reports/p0-after
```

Do not point `--resume-new-data` at a checkpoint from the new phase; that flag
intentionally restarts the three-epoch phase from the beginning of the merged
corpus. After training, find the final saved checkpoint and evaluate its
`model.pt` on both splits using the baseline commands, changing the checkpoint
path, `--out`, and `--report-dir` to split-specific directories:

```bash
FINAL_CKPT=$(find checkpoints/p0-merged-3ep -maxdepth 1 -type d -name 'step-*' | sort -V | tail -n 1)
python scripts/benchmark.py --configs configs/p0.yaml \
  --checkpoints "$FINAL_CKPT/model.pt" --split dev \
  --out reports/p0-after/dev --report-dir reports/p0-after/dev --spbleu
python scripts/benchmark.py --configs configs/p0.yaml \
  --checkpoints "$FINAL_CKPT/model.pt" --split devtest \
  --out reports/p0-after/devtest --report-dir reports/p0-after/devtest --spbleu
```
