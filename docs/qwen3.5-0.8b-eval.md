# Qwen3.5-0.8B KK-to-RU baseline

Evaluation date: 2026-11-05. The original
[`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) checkpoint was
evaluated zero-shot on all 1,012 FLORES+ KK-to-RU devtest pairs. This is the
instruction-tuned release, not the separately named `Qwen3.5-0.8B-Base` model.

## Results

| Model | BLEU ↑ | chrF ↑ | chrF++ ↑ | TER ↓ |
|---|---:|---:|---:|---:|
| P0, archived translations | 20.14 | 49.55 | 46.48 | 69.84 |
| KazRush, archived translations | 18.89 | 48.70 | 45.48 | 71.47 |
| **Qwen3.5-0.8B, this run** | **3.00** | **24.26** | **21.43** | **124.08** |
| Qwen3-0.6B, archived translations | 0.99 | 16.39 | 13.89 | 100.71 |

All scores were calculated with SacreBLEU 2.6.0 against the current devtest TSV.
BLEU uses the default `13a` tokenizer. The archived predictions were rescored;
their models were not rerun. Every archived P0 and Qwen3-0.6B source/reference
pair was checked against the dataset. KazRush has a plain hypothesis file, so
its alignment relies on the existing line order. Systems retain their original
decoding settings; this is a comparison of their saved translations on the
same test set.

## Protocol

- Model revision: `2fc06364715b967f1860aea9cf38778875588b17`.
- Original unquantized weights, loaded through MLX's text-model implementation.
  The runtime reports 752,393,024 text-model parameters and bfloat16/float32 tensors.
- Hardware: Apple M4 Pro, 24 GB unified memory; MLX GPU backend.
- Runtime: MLX 0.32.3, mlx-lm 0.32.0, Transformers 5.17.0.
- Dataset: `data/eval/flores_plus/devtest.kk-ru.tsv`, 1,012 pairs in original order.
- One sentence per request, zero-shot, thinking explicitly disabled.
- Greedy decoding (`temperature=0.0`), at most 512 generated tokens per sentence.
  These settings differ from the model card's sampling preset.
- Only whitespace is normalized for scoring. Raw outputs are retained.
- References are used for scoring only. The prompt and settings were fixed before
  devtest; the preceding smoke test used three dev sentences.

System prompt:

```text
Translate the user's text from Kazakh into Russian. Output only the Russian translation, without explanations or additional text.
```

The user message contains the source sentence alone. The model's own chat
template is applied with `enable_thinking=False` and `add_generation_prompt=True`.

## Runtime and observed failures

- Generation time, including the first sentence but excluding model loading and
  download: **480.68 seconds** (8.01 minutes), **2.11 sentences/second**.
- Median sentence latency: **0.362 seconds**; 58,782 generated tokens.
- Peak MLX allocation: **1.875 GB**; this is not total system memory usage.
- Empty responses: **0 / 1,012**.
- Exact source copies: **71 / 1,012** (7.02%).
- Responses cut at the 512-token limit: **24 / 1,012** (2.37%).
- Responses containing Kazakh-specific Cyrillic letters: **499 / 1,012**.
  This is a character-level diagnostic, not a language classifier: proper names
  can also contain these letters.

Qualitative inspection found untranslated or mixed-language output and factual
substitutions. For example, the first source contains `4 айлық тышқандар`
(four-month-old mice), whereas the hypothesis says `4-летние пациенты`
(four-year-old patients).

With these settings, Qwen3.5-0.8B scores below P0 and KazRush. Alternative
prompts, decoding settings, fine-tuning, and the Weight-Space Programming
weight edit were not tested.

## Reproduction and artifacts

```bash
PYTHONPATH=src .venv/bin/python scripts/eval_qwen.py \
  --out reports/qwen3.5-0.8b-rerun
```

The pinned Hugging Face model is stored in `models/Qwen3.5-0.8B/`.
The original run artifacts are in `reports/qwen3.5-0.8b/`:

- `devtest.jsonl`: all sources, references, raw outputs, normalized hypotheses,
  source hashes, token counts, stop reasons, and latencies.
- `devtest.hyp.txt`: one normalized hypothesis per dataset row.
- `devtest.metrics.json`: full-precision metrics and SacreBLEU signatures.
- `devtest.manifest.json`: model hash, dataset hash, prompt, settings, runtime
  versions, and evaluator Git revision.
- `devtest.validation.json`: additional diagnostics and output hashes.
- `comparison.json`: rescored baseline comparison.
- `devtest.log`: generation progress and final metrics.

Model weights and generated reports are ignored by Git. This summary and the
evaluation script are versioned.

## Verification

The model file SHA-256 matches Hugging Face's published LFS hash. All 1,012
records were checked for source/reference alignment, source hash, raw-output
normalization, token limit, termination metadata, and hypothesis-file agreement.
All four corpus metrics were independently recomputed from the saved predictions.
At evaluation time, all 24 tests discovered by unittest and the three-sentence
real-model smoke test passed. The current `make test` command uses pytest,
which also collects function-based tests.

| Artifact | SHA-256 |
|---|---|
| Original safetensors file | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |
| Devtest TSV | `867dcb0ac62b096821516eff462d18efd1b049a9c17715026f755d7d00e21bcb` |
| Prediction JSONL | `0728319441888620650a2564a186c899eff5207b5f7c80d38cde27a4cb772976` |
| Hypotheses text | `01f5e2e7cf34171033e473a550260788edd8ca95a714ec801933fcc8b1858118` |
