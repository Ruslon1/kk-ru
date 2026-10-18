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
`facebook/nllb-200-3.3B` on 6 NVIDIA RTX 4090 GPUs. The dataset contains
3,999,605 filtered pairs from 4,000,000 generated records.

The published files contain successful, non-empty translations and exclude
pairs identical to the Kazakh source.

## Fields

- `source`: Kazakh sentence in Cyrillic (`kaz_Cyrl`)
- `translation`: Russian translation in Cyrillic (`rus_Cyrl`)

## Loading

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="rank-*.jsonl.gz", split="train")
```

## Notes

The translations are machine-generated and may contain errors. Check the
licenses of the underlying source data and NLLB model before commercial use.
