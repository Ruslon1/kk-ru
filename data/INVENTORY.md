# Inventory KK–RU (raw, до фильтра)

| файл | размер | пар | unique | exact dups | identical kk=ru | kk tok mean/p50/p95 | ru tok mean/p50/p95 |
|---|---:|---:|---:|---:|---:|---|---|
| `data/raw/opus.kk-ru.tsv` | 194.4 MB | 788,960 | 744,537 | 44,423 | 17,924 | 9.5/5/33 | 10.5/5/38 |
| `data/raw/wmt19_crawl.kk-ru.tsv` | 2631.8 MB | 5,063,666 | 4,516,956 | 546,710 | 3,322 | 17.4/15/38 | 18.2/16/39 |
| `data/raw/kazparc_human.kk-ru.tsv` | 132.5 MB | 371,902 | 368,201 | 3,701 | 1,638 | 12.5/9/34 | 13.5/10/37 |
| `data/raw/kazparc_sync.kk-ru.tsv` | 735.1 MB | 1,797,066 | 1,787,446 | 9,620 | 90 | 14.8/14/27 | 16.8/16/31 |
| `data/raw/kazparc.kk-ru.tsv` | 867.6 MB | 2,168,968 | 2,155,581 | 13,387 | 1,728 | 14.4/13/28 | 16.3/15/32 |
| `data/raw/til.kk-ru.tsv.gz` | 625.5 MB | 4,413,843 | 4,413,818 | 25 | 3,746 | 17.7/15/38 | 18.5/16/40 |
| `data/eval/flores_plus/dev.kk-ru.tsv` | 0.5 MB | 997 | 997 | 0 | 0 | 16.6/16/27 | 19.2/18/32 |
| `data/eval/flores_plus/devtest.kk-ru.tsv` | 0.5 MB | 1,012 | 1,012 | 0 | 0 | 17.2/16/28 | 19.4/19/32 |
| `data/raw/extra/kazlit.kk-ru.tsv` | 8.0 MB | 22,132 | 22,132 | 0 | 7 | 15.5/13/35 | 16.6/14/38 |
| `data/raw/extra/kaznu.kk-ru.tsv` | 41.8 MB | 82,704 | 82,704 | 0 | 235 | 19.3/17/41 | 20.6/18/44 |

## Exact overlap with WMT19 crawl

| файл | ∩ WMT19 | доля файла |
|---|---:|---:|
| `data/raw/opus.kk-ru.tsv` | 517 | 0.1% |
| `data/raw/kazparc_human.kk-ru.tsv` | 4,376 | 1.2% |
| `data/raw/kazparc_sync.kk-ru.tsv` | 1 | 0.0% |
| `data/raw/kazparc.kk-ru.tsv` | 4,377 | 0.2% |
| `data/raw/til.kk-ru.tsv.gz` | 4,337,902 | 98.3% |
| `data/eval/flores_plus/dev.kk-ru.tsv` | 0 | 0.0% |
| `data/eval/flores_plus/devtest.kk-ru.tsv` | 0 | 0.0% |
| `data/raw/extra/kazlit.kk-ru.tsv` | 0 | 0.0% |
| `data/raw/extra/kaznu.kk-ru.tsv` | 3,805 | 4.6% |

## Other exact overlaps

- `data/raw/opus.kk-ru.tsv` ∩ `data/raw/kazparc_human.kk-ru.tsv` = **7,453**
- `data/raw/opus.kk-ru.tsv` ∩ `data/raw/kazparc_sync.kk-ru.tsv` = **26**
- `data/raw/kazparc_human.kk-ru.tsv` ∩ `data/raw/kazparc_sync.kk-ru.tsv` = **66**
- `data/eval/flores_plus/devtest.kk-ru.tsv` ∩ `data/raw/wmt19_crawl.kk-ru.tsv` = **0**
- `data/eval/flores_plus/devtest.kk-ru.tsv` ∩ `data/raw/opus.kk-ru.tsv` = **0**
- `data/eval/flores_plus/devtest.kk-ru.tsv` ∩ `data/raw/kazparc.kk-ru.tsv` = **0**

## Notes

- kazRush raw mix: OPUS ~718k + kazparc ~2,150k + WMT19 5,063k + TIL ~4,403k.
- WMT19 crawl: скачался ровно **5,063,666** пар — совпало с карточкой kazRush.
- KazParC: суммарно **2,168,968** пар (human 371,902 + SynC 1,797,066; у kazRush 2,150k после дополнительной чистки).
- OPUS moses: собрано **789,436** пар из 13 подкорпусов (GNOME, KDE4, MultiCCAligned, NeuLab-TedTalks, News-Commentary, OpenSubtitles, QED, TED2020, Tatoeba, Ubuntu, WikiMatrix, XLEnt, wikimedia).
- FLORES+: эталонный eval из `openlanguagedata/flores_plus` (dev: 997, devtest: 1,012).
- Extra (опциональные для M_extra): `kazlit.kk-ru.tsv` (54.5k) и `kaznu.kk-ru.tsv` (209.7k).
- **TIL**: скачан из [Drive-зеркала](https://drive.google.com/drive/folders/1kUp_vpDsNUZvVC6HvwNxGn7ImwnCfM1E) в `til/` (train в 2 zip + dev + test{bible,ted,x-wmt}). Импорт: `python scripts/import_til.py` → `data/raw/til.kk-ru.tsv.gz`. Лицензия CC BY-NC-SA 4.0.
- Это сырой корпус. Следующий этап — фильтрация по пайплайну kazRush (дедуп, чистка мусора, FastText langid, LaBSE, OpusFilter).
