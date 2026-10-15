---
language:
  - kk
  - ru
task_categories:
  - translation
pretty_name: Synthetic Kazakh-Russian Translation Pairs
tags:
  - kazakh
  - russian
  - machine-translation
  - synthetic-data
size_categories:
  - 1M<n<10M
---

# Synthetic Kazakh-Russian Translation Pairs

Synthetic Kazakh-to-Russian sentence pairs generated with
`facebook/nllb-200-3.3B`. The dataset is public and intended for research and
machine-translation experiments. It was generated on 6 NVIDIA RTX 4090 GPUs.

The project and dataset are still under development. The published files contain
only successful, non-empty translations and exclude pairs identical to the
Kazakh source.

## Fields

- `source`: Kazakh sentence in Cyrillic (`kaz_Cyrl`)
- `translation`: Russian translation in Cyrillic (`rus_Cyrl`)

## Loading

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="rank-*.jsonl.gz", split="train")
```

## Intended use

Use this data for research, benchmarking, and training Kazakh-Russian
translation models. Machine-generated translations can contain errors and
should be filtered or reviewed for production use. Check the licenses of the
underlying public source data and NLLB model before commercial use.

## Upload

```bash
huggingface-cli upload YOUR_USERNAME/kk-ru-nllb-synthetic hf-nllb-kk-ru . \
  --repo-type dataset
```
