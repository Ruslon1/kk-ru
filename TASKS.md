# Задачи на реализацию — для DeepSeek-исполнителя

Это документ главного (архитектора). Исполнитель (DeepSeek) пишет код по задачам ниже.
Каждая задача — самодостаточный блок: его можно вставить в чистый чат DeepSeek, и он должен
выдать рабочий код + отчёт о проверке. Каноническая спецификация — `PLAN.md` (читать обязательно).

---

## 0. Где мы сейчас (факт)

- Архитектура **зафиксирована** в `PLAN.md` §4 и §8. Ничего менять без согласования.
- **Скелет репозитория готов** (T2 выполнен): `src/kk_ru/` пакет, `configs/p0.yaml`,
  `Makefile`, `requirements*.txt`, `pyproject.toml`, `tests/`. Конфиг уже работает,
  `src/kk_ru/config.py` и `src/kk_ru/utils.py` реализованы; остальные модули — заглушки
  с точной сигнатурой под задачи T3–T8.
- Данные **сырые** скачаны почти полностью, фильтрации ещё нет:

| Корпус | Пары | Статус |
|---|---:|---|
| OPUS moses kk–ru | 788 960 | ✅ `data/raw/opus.kk-ru.tsv` |
| WMT19 crawl | 5 063 666 | ✅ `data/raw/wmt19_crawl.kk-ru.tsv` |
| KazParC (human+sync) | 2 168 968 | ✅ `data/raw/kazparc*.kk-ru.tsv` |
| FLORES+ dev / devtest | 997 / 1 012 | ✅ `data/eval/flores_plus/*.tsv` |
| **TIL** | **~4 400 000** | ❌ **НЕ скачан** (GCS bucket 403, billing closed) |
| extra (kazlit / kaznu) | 22k / 82k | ✅ опционально, `data/raw/extra/` |

- Кода модели/обучения **нет** (заглушки). Скачивание/статистика — `scripts/download_raw.py`, `scripts/data_stats.py`.
- Окружение: в `.venv` (Mac, Python 3.14) есть только `datasets`, `huggingface_hub`, `numpy`,
  `pandas`, `sacrebleu`, `tqdm`, `pyyaml`. **Нет** `torch`, `transformers`, `sentencepiece`,
  `accelerate`, `sentence-transformers`, `fasttext`, `unbabel-comet`, `tokenizers`.
- Машина для обучения: **6–8× RTX 4090 24GB** (Linux, CUDA). Локальный Mac — только подготовка данных.

---

## 1. Правила для исполнителя (в каждый промпт)

1. Прочитай `PLAN.md` перед началом. Не меняй архитектуру, конфиг и данные без явного указания.
2. Не изобретай новые корпуса/параметры. P0 = ровно те же 4 источника данных, что у kazRush.
3. Каждый результат — **отчёт**: что сделал, какие файлы создал, какие цифры получил, как проверить.
4. Код воспроизводим: фиксированный `seed`, версии в `requirements.txt`, запуск одной командой.
5. Python **3.10–3.12** на Linux (torch отстаёт на 3.13+; локальный Mac 3.14 — только подготовка данных).
6. Никакого llama.cpp / DeepSpeed / FSDP / Fairseq / LLaMA-Factory. Только PyTorch + Accelerate DDP.
7. Код ложится в `src/kk_ru/` (пакет) и `scripts/` (пайплайн данных). Запуск — через `make` (см. `README.md`).

---

## 2. Порядок (критический путь)

```
T1 окружение ──► T3 модель+count_params ──► T7 train + оверфит (гейт)
   └─► T4 фильтр ──► T5 SentencePiece ──► T6 даталоадер ──┘        └─► T8 eval FLORES+
   (T2 скелет — уже готов)
```

- **T3 (модель)** и **T4 (фильтр)** независимы — можно параллельно.
- **T7-оверфит** — гейт: пока 10k пар не оверфитятся (loss → ~0), большой прогон не начинать.
- Параллельно всем задачам решается **блокер TIL** (см. §4).

---

## 3. Задачи

---

### T1. Окружение и проверка GPU

**Цель:** воспроизводимое окружение для обучения на 4090.

**Сделать:**
1. `requirements.txt` уже есть (список зависимостей). T1 — **закрепить версии**:
   поставить на машине с 4090 (Python 3.10–3.12), добиться совместимых версий, затем
   `pip freeze > requirements.lock.txt` (не трогать `requirements.txt` руками под Mac).
2. Скрипт `scripts/check_env.py` — печатает и проверяет:
   - `torch.__version__`, `torch.cuda.is_available()`, число видимых GPU, имена GPU;
   - `torch.cuda.is_bf16_supported()` (на 4090 должно быть True);
   - SDPA: пробный `F.scaled_dot_product_attention` на случайном батче;
   - `accelerate` видит все GPU (вывод `accelerate env`).

**DoD:** `python scripts/check_env.py` на машине с 4090 выводит `OK` по всем пунктам;
bf16 = True; SDPA работает; Accelerate видит N GPU. Вывод сохранить в `ENV_REPORT.md`.

---

### T2. Скелет репозитория + конфиг + утилиты — ✅ готово

Скелет создан. Исполнителю **ничего делать не надо**, только знать структуру:

```
configs/p0.yaml          # единственный источник правды по конфигу (уже заполнен)
src/kk_ru/
  __init__.py
  config.py              # ✅ реализован: YAML → dataclass + --opts key=value
  utils.py               # ✅ реализован: set_seed / logger / timer / human
  model.py               # заглушка → T3
  tokenizer.py           # заглушка → T5
  data.py                # заглушка → T6
  train.py               # заглушка → T7
  eval.py                # заглушка → T8
scripts/
  download_raw.py        # ✅ (скачивание сырых данных)
  data_stats.py          # ✅ (инвентарь)
  filter_data.py         # заглушка → T4
  train_tokenizer.py     # заглушка → T5
tests/                   # test_config.py зелёный; остальные — контракт T3/T5/T6
```

Запуск: `make install / data / filter / tokenizer / overfit / train / eval / test`.

---

### T3. Модель Qwen-style encoder–decoder + count_params

**Цель:** сердце проекта. Проверить, что архитектура и число параметров сходятся (~202M).

**Спецификация (из PLAN.md §2–§4):**
- Encoder–decoder (НЕ decoder-only). 12 encoder + 12 decoder, `d_model=512`, `8 heads`, `head_dim=64`.
- **Encoder-слой** (bidirectional, БЕЗ cross-attn, БЕЗ causal):
  `x = x + SelfAttn(RMSNorm(x))` (MHA 8Q/8KV, RoPE, QK-Norm) → `x = x + SwiGLU(RMSNorm(x))`.
- **Decoder-слой** (три субячейки, три резудиала, три Pre-LN RMSNorm):
  1. `y = y + MaskedSelfAttn(RMSNorm(y))` — **GQA 8Q/4KV**, causal, RoPE, QK-Norm;
  2. `y = y + CrossAttn(RMSNorm(y), encoder_out)` — **MHA 8Q/8KV**, Q=decoder(RU), K/V=encoder(KK), **RoPE нет**, QK-Norm да;
  3. `y = y + SwiGLU(RMSNorm(y))`.
- Norm: RMSNorm, **Pre-LN** (норма перед каждой субячейкой, резудиал после).
- Pos: RoPE **только на self-attn** (encoder self и decoder masked self). На cross-attn RoPE нет.
- QK-Norm: да (на self и cross).
- FFN: SwiGLU `512 → 4096 → 512` (gate + up → down).
- Эмбеддинги: обучаются с нуля, `tie_embeddings: true` (lm_head = input embeddings), shared KK+RU.
- dropout 0.1, max_len 256.

**Сделать:**
1. Заполнить `src/kk_ru/model.py` (сигнатуры уже есть):
   - `RMSNorm`, `RotaryEmbedding` (RoPE), `Attention` (флаги `causal`/`use_rope`/`n_kv_heads` для GQA),
     `SwiGLU`, `EncoderBlock`, `DecoderBlock`, `TransformerEncoderDecoder`, `build_model`, `count_params`.
   - Attention через `F.scaled_dot_product_attention` (flash-совместимо).
2. Реализовать `tests/test_model.py` (каркас уже есть) и довести до зелёного:
   - прямой проход `(B=2, KK_len=64)` → логиты `(B=2, RU_len=64, vocab)`, loss конечный;
   - causal-маска: logit на позиции `t` не зависит от RU-токенов `>t`;
   - GQA: decoder self-attn `n_kv_heads=4` при `n_heads=8`;
   - RoPE только на self-attn, не на cross-attn;
   - tied embeddings: `model.embed.weight is model.lm_head.weight`.
3. Отчёт: точное число параметров по компонентам (embeddings / encoder / decoder / head) и сумма.
   Ожидание: **~202M** (диапазон 190–210M). Если сильно не сходится — написать, где расхождение,
   НЕ менять конфиг молча.

**DoD:** `count_params` ≈ 190–210M; `tests/test_model.py` зелёный; прямой проход в bf16 не падает.

---

### T4. Фильтр корпуса (дедуп → чистка → langid → LaBSE)

**Цель:** из сырых TSV получить один `data/filtered/train.kk-ru.tsv` по пайплайну kazRush.

**Порядок фильтра (именно такой, PLAN.md §7):**
1. **дедуп** (точные пары kk+ru);
2. **чистка**: убрать HTML/теги, нормализовать пробелы/юникод, выкинуть пустые/короткие (≤1 токена) и `kk == ru`;
3. **langid**: `facebook/fasttext-language-identification` — kk→kk, ru→ru;
4. **LaBSE**: `sentence-transformers/LaBSE`, cosine(kk, ru) ≥ порога (параметр, старт ~0.55; показать гистограмму);
5. (опц.) **OpusFilter** — финальная чистка, не критично для M1.

**Сделать:**
1. Заполнить `scripts/filter_data.py` (аргументы уже есть): `--in`, `--out`, `--labse-threshold`, `--max-len`, `--seed`.
   LaBSE на GPU, остальное на CPU.
2. Каждый шаг печатает «было → стало» (для отчёта).
3. Уметь добавлять `data/raw/til.kk-ru.tsv`, когда он появится.
4. НЕ класть в train `data/eval/**`; проверить отсутствие точного пересечения train ∩ FLORES+.

**DoD:** создан `data/filtered/train.kk-ru.tsv`; отчёт по шагам; гистограмма LaBSE-similarity;
нулевое точное пересечение train ∩ FLORES+ dev/devtest.

---

### T5. SentencePiece 32k + модуль токенизатора

**Цель:** свой токенизатор (shared KK+RU, vocab 32000), эмбеддинги с нуля.

**Сделать:**
1. Заполнить `scripts/train_tokenizer.py` (аргументы уже есть):
   - вход `data/filtered/train.kk-ru.tsv` (обе колонки);
   - `sentencepiece` Unigram (или BPE), `vocab_size=32000`, `character_coverage≈0.9995` (параметр);
   - спецтокены `<pad> <unk> <bos> <eos>` — порядок/индексы зафиксировать и задокументировать;
   - выход `data/tokenizer/kk-ru-sp32k.model` + `.vocab`.
2. Заполнить `src/kk_ru/tokenizer.py`: `encode/decode`, `pad_id/bos_id/eos_id/unk_id`, обрезка до max_len.
3. Реализовать `tests/test_tokenizer.py` (каркас уже есть): round-trip на кириллице (kk и ru),
   спецтокены на месте, длина ≤ max_len.

**DoD:** vocab 32000; round-trip корректен; индексы спецтокенов зафиксированы;
модель грузится из `configs/p0.yaml` (`data/tokenizer/kk-ru-sp32k.model`).

---

### T6. Даталоадер + collate

**Цель:** эффективная подача данных (стриминг, паддинг, causal-маска, shift).

**Сделать:**
1. Заполнить `src/kk_ru/data.py` (сигнатуры уже есть):
   - стриминговый `Dataset` по `data/filtered/train.kk-ru.tsv` (не грузить всё в память);
   - shuffle буфером; `collate_fn`: токенизация, паддинг до max в батче, `attention_mask`;
   - decoder: `labels` = RU со сдвигом вправо, pad → `-100` (исключить из loss);
   - обрезка до `max_len=256`.
2. Реализовать `tests/test_data.py` (каркас уже есть): формы, causal-маска, `-100` на pad.

**DoD:** `tests/test_data.py` зелёный; формы батча корректны; паддинг не попадает в loss.

---

### T7. Цикл обучения (Accelerate DDP) + оверфит-тест

**Цель:** воспроизводимый train, который оверфитит 10k пар — гейт перед большим прогоном.

**Сделать:**
1. Заполнить `src/kk_ru/train.py` (CLI уже есть: `--config`, `--opts`, `--overfit`, `--resume`):
   - HuggingFace **Accelerate** DDP; bf16 + SDPA;
   - AdamW (`lr=2e-4`, `weight_decay=0.01`), cosine + warmup 2000, `min_lr=2e-5`;
   - label smoothing 0.1; grad clip 1.0; grad-accum (`micro_batch_per_gpu × grad_accum × N = 256`);
   - чекпоинты каждые `save_every` + best по FLORES dev, **resume**; в каталог чекпоинта класть
     копию конфига, а полный конфиг логировать на старте (воспроизводимость);
   - логирование loss/lr/шаг; `set_seed(seed)`.
2. Режим `--overfit`: 10k пар (фикс. seed), маленький батч, гнать до loss ~0 —
   доказывает согласованность модель+токенизатор+даталоадер+оптимизатор.
3. Оверфит **обязательно** прогнать, результат в `OVERFIT_REPORT.md` (финальный loss / график).

**DoD:** `python -m kk_ru.train --overfit` доводит loss до ~0; `train.py` стартует с
`accelerate launch -m kk_ru.train`, сохраняет и возобновляет чекпоинт.

---

### T8. Eval FLORES+ (BLEU / chrF / COMET)

**Цель:** метрики как у kazRush (BLEU 18.8 / chrF 48.7 / COMET 86.7) на тех же сплитах.

**Сделать:**
1. Заполнить `src/kk_ru/eval.py` (CLI уже есть: `--checkpoint`, `--split dev|devtest`):
   - грузит чекпоинт, переводит `data/eval/flores_plus/{split}.kk-ru.tsv`;
   - декодирование beam=5, max_len=256;
   - `sacrebleu` → **BLEU** и **chrF** (сигнатуры совместимые с kazRush);
   - `unbabel-comet` (wmt22-comet-da или та же модель, что у kazRush — уточнить) → **COMET**;
   - вывод таблицы BLEU/chrF/COMET.
2. (опц.) `--baseline kazRush`: прогнать `deepvk/kazRush-kk-ru` → воспроизвести 18.8/48.7/86.7
   — калибровка нашего eval.

**DoD:** `python -m kk_ru.eval --config configs/p0.yaml --checkpoint ... --split devtest` выдаёт
BLEU/chrF/COMET; таблица готова для отчёта.

---

## 4. Блокеры / параллельные задачи (не код, но нужны)

### B1. TIL корпус (4.4M пар) — критично для честного сравнения
Сейчас GCS bucket `til-corpus` отдаёт 403. Нужно найти зеркало:
- репозиторий `github.com/turkic-interlingua/til-mt` (README/raw файлы, Drive-ссылки);
- HF-зеркала TIL kk–ru;
- альтернативный дамп.
Когда найдётся — `python scripts/download_raw.py --til-from <path>` → `data/raw/til.kk-ru.tsv`,
потом прогнать через T4-фильтр. Без TIL M1 неполный: сравнивать надо на тех же 12.3M.

### B2. Точная модель COMET у kazRush
Проверить карточку `deepvk/kazRush-kk-ru` — какая COMET-модель и какие sacrebleu-сигнатуры,
чтобы наш eval был сравнимым (иначе «у нас выше» не зачтётся).

---

## 5. После того как код готов (фаза прогонов)

Прогоны = конфиги, не новый код. Каждый — отдельная строка таблицы:

| ID | Прогон | Что |
|---|---|---|
| M1 | Qwen-style 200M, полный корпус | главный результат |
| M2 | T5-блок ~200M, наш пайплайн, 32k vocab | изолировать «Qwen-блок vs T5-блок» |
| M3 | Qwen-style 200M, vocab 8k | изолировать vocab |
| A1 | Qwen-style 100M | efficiency |
| A2 | без QK-Norm | абляция |
| A3 | decoder MHA вместо GQA | абляция GQA |
| A4 | 2 vs 3 эпохи | абляция |

Расписание (PLAN.md §10): **неделя 1** — код (T1–T8 + оверфит); **неделя 2** — M1+M2+M3;
**неделя 3** — A1–A4 + один финальный test + черновик отчёта.

---

## 6. Что делать прямо сейчас (рекомендация главного)

1. **Сегодня:** T1 (окружение на 4090) + T3 (модель + count_params) — де-риск архитектуры,
   можно запускать первым и параллельно.
2. **Сразу после:** T4 (фильтр) → T5 (SentencePiece) → T6 (даталоадер) → T7 (оверфит).
   Оверфит — зелёный свет на M1. (T2 скелет уже готов.)
3. **Параллельно всё время:** B1 (достать TIL) — без него M1 не считается честным.
4. T8 (eval) делать параллельно с T7, чтобы метрики были готовы к первому же чекпоинту.
