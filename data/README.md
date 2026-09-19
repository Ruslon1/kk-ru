# Данные KK→RU

Честное сравнение с kazRush: **те же 4 источника**, потом тот же фильтр.
Дополнительные корпуса — отдельный прогон (`M_extra`), не мешать в M1.

Eval: **только** FLORES+ `dev` (early stop) и `devtest` (финальный test). В train не класть.

## M1 — как kazRush (~12.3M сырых пар)

| Источник | Пар | Скачать |
|---|---:|---|
| [OPUS kk–ru](https://opus.nlpl.eu/opusapi/?source=kk&target=ru&preprocessing=moses&latest=True) | ~0.72M у них; сейчас ~0.79M moses | `scripts/download_raw.py --opus` |
| [issai/kazparc](https://huggingface.co/datasets/issai/kazparc) | 2.15M kk–ru (весь HF gated) | `huggingface-cli login` + accept license, затем `--kazparc` |
| [WMT19 crawl kk–ru](http://data.statmt.org/wmt19/translation-task/crawl.kk-ru.gz) | 5.06M | `--wmt19` |
| [TIL corpus](https://github.com/turkic-interlingua/til-mt) | 4.40M | [Drive-зеркало](https://drive.google.com/drive/folders/1kUp_vpDsNUZvVC6HvwNxGn7ImwnCfM1E) → `til/` → `scripts/import_til.py` |

Фильтр kazRush (в этом порядке):

1. дедуп
2. чистка мусора / тегов / пробелов
3. langid: `facebook/fasttext-language-identification`
4. выкинуть пары с низким LaBSE (`sentence-transformers/LaBSE`)
5. [OpusFilter](https://github.com/Helsinki-NLP/OpusFilter)

## Что ещё есть (не в M1)

| Корпус | Размер kk–ru | Зачем / риск |
|---|---|---|
| OPUS свежее 2024–26 (OpenSubtitles v2024, wikimedia v2026) | +~0.1M к их OPUS | можно тихо добавить в OPUS-скачивалку |
| MultiCCAligned v1.1 | 432k | веб-майнинг, пересекается с crawl |
| WikiMatrix | 33k | ок, уже в OPUS |
| XLEnt | 87k | имена/сущности, шум |
| Tatoeba | 2.4k | чистое, крохи |
| [allenai/nllb](https://huggingface.co/datasets/allenai/nllb) `kaz_Cyrl–rus_Cyrl` | майнинг LASER3; kk–ru в NLLB скорее через CCMatrix | огромный шум, не для честного vs kazRush |
| [stukenov/ekitil-parallel-kkru-v2](https://huggingface.co/datasets/stukenov/ekitil-parallel-kkru-v2) | 5.1M | **это уже WMT19 crawl + KazParC**, gated, не суммировать |
| [BorisovMaksim/kk_ru_csw](https://huggingface.co/datasets/BorisovMaksim/kk_ru_csw) | 619 | code-switching, только на eval-хотелку |
| NTREX / FLORES | ~2k | **только test** |

KazParC paper ([arxiv:2403.19399](https://arxiv.org/abs/2403.19399)): **372k human** kk/en/ru/tr. Цифра kazRush **2.15M** — kk–ru срез HF-датасета (там ещё crawled/другие пары). Берём kk–ru, как они.

## Диск и формат хранения

```
til/                      # сырые zip'ы TIL из Drive-зеркала (архив, не трогаем)
data/raw/*.kk-ru.tsv[.gz] # распакованное сырьё по источникам, kк<TAB>ru
data/filtered/            # результат фильтра (T4), вход обучения
data/eval/flores_plus/    # dev/devtest — в train НЕ класть
```

Правила для больших текстов (стандарт OPUS: `train.raw.tsv.gz`):

- **потоково** читать/писать — не грузить весь файл в память (`scripts/import_til.py` так и делает);
- **gzip** для крупных файлов (`*.tsv.gz`, экономит ~75% диска);
- сырьё и обработанное раздельно, сырьё не переписывать;
- в git ничего не класть (`.gitignore`);
- Parquet — только если данных станет на порядок больше; на ~12M пар потокового TSV достаточно.

Сырьё ~5–12 GB. После фильтра меньше. Качать/держать на машине с 4090, не в git.
