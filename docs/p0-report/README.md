# P0 technical report

English working draft for a first supervisor discussion. P0 only; diffusion
results are excluded.
Author: Ruslan Rakhimov, Kazakhstan-British Technical University (KBTU).
The concise report retains the model, data/training history and a comparison
of all complete local benchmark runs plus published KazRush/NLLB scores.
Detailed inventories and provenance stay in this folder.

## Read and edit

- Main source: main.tex
- Exported PDF snapshot: main.pdf (rebuild after source edits)
- Bibliography: references.bib
- Vector architecture: figures/architecture.tex
- Editable architecture: figures/p0-architecture.drawio
- Raw evidence and provenance snapshots: evidence/
- Generated tables: generated/

Open this directory in the Codex LaTeX editor, or upload it to Overleaf and
select XeLaTeX. The layout uses nips_2017.sty from the supplied Attention Is All You Need
source, with its author credits preserved. The conference notice is replaced
by a technical-report notice. TeX Gyre Termes, Heros and Cursor provide the
Times/Helvetica/Courier typography through the TeX distribution. Bundled
Noto Serif supplies Cyrillic text under the SIL Open Font License. The
architecture palette follows the reference figure, with colors by operation.
Tectonic compilation, from the repository root:

    tectonic docs/p0-report/main.tex

Regenerate numerical tables instead of hand-editing generated/ files.
The comparison table combines archived_benchmarks.json and p0-devtest.json
with published_benchmarks.json, transcribed from the saved KazRush model
card. Published scores and common rescoring are shown in separate panels;
unmeasured metrics are left blank. Failed/incomplete runs are excluded.

## Verified results

Fresh evaluation: checkpoints/p0/best/model.pt, FP32, beam 5, batch 16.

| Split / system | BLEU | chrF | chrF++ | TER |
|---|---:|---:|---:|---:|
| P0 dev, 997 pairs | 20.3053 | 49.4584 | 46.4031 | 69.7250 |
| P0 devtest, 1,012 pairs | 20.1358 | 49.5532 | 46.4807 | 69.8448 |
| KazRush archived devtest | 18.8878 | 48.7007 | 45.4781 | 71.4678 |

P0 was generated on 31 October 2026. KazRush was rescored from an archive,
not regenerated. Alignment is positional; its original inference manifest
is unavailable. The paper discloses this limitation.
All four checkpoint metadata scores exactly match the fresh dev run.

All 29 repository tests pass under pytest. Unittest misses five function-based
tests. Full software checks and benchmark archives are recorded in evidence/
and generated/, rather than printed as report appendices.

The base file has 6,851,060 pairs; six synthetic shards have 3,999,605.
Full scans found no exact source/pair intersections with dev or devtest
after NFC normalization and stripping. This does not exclude near duplicates
or teacher exposure.

## Reproduce

Run from the repository root:

    PYTHONPATH=src .venv/bin/python -m pytest tests -v --junitxml=docs/p0-report/evidence/tests.xml
    PYTHONPATH=src .venv/bin/python docs/p0-report/collect_evidence.py archives
    PYTHONPATH=src .venv/bin/python docs/p0-report/collect_evidence.py corpora
    PYTHONPATH=src .venv/bin/python docs/p0-report/collect_evidence.py p0 --split dev
    PYTHONPATH=src .venv/bin/python docs/p0-report/collect_evidence.py p0 --split devtest
    PYTHONPATH=src .venv/bin/python docs/p0-report/summarize_results.py
    .venv/bin/python docs/p0-report/figures/make_drawio.py
    tectonic docs/p0-report/main.tex

P0 generation writes/resumes indexed JSONL in reports/p0-report/. Move a run
file aside for fresh independent generation. Keep checkpoint, config, batch
size and dataset unchanged during resume. No paid API requests are made.
Large inputs and translation archives stay Git-ignored; compact metrics
and provenance are stored with the report.

Python dependencies are the project training/evaluation stack plus pytest.
Scoring uses SacreBLEU 2.6.0. ONNX Runtime is only for export checks.
The measured MPS throughput is not Android performance.

## Questions and missing server artifacts

1. Supervisor and acknowledgment, if applicable.
2. At the approximate 17-BLEU plateau: checkpoint, split, decoding settings,
   metric implementation and stopping criterion.
3. Per-stage resolved configurations, run manifests, TensorBoard event
   files, metrics JSONL, actual hardware and wall-clock durations.
4. Continuation used base+synthetic data, confirmed by the author; recover
   the final row count, mixture weights, epochs and optimizer updates.
5. Final merged-corpus manifest and filtering/merge statistics.
6. Dataset revisions and KazRush inference/model revision.

The plateau is the author's account. No curves or compute totals are
invented. YAML defaults are labeled as defaults, not historical execution.
Later supply the original events.out.tfevents files together with the matching
config and run manifest. Preserve stage names and global-step offsets.
