# План проекта KK→RU (актуальный)

Препод одобрил. **Приоритет: Qwen-style encoder–decoder ~200M, обучение с нуля.**
Машина: **6–8× RTX 4090 24GB**. Frozen Qwen (~700M) — второй приоритет, не замена.

---

## 1. Что теперь главное

Научная ставка:

> Encoder–decoder с Qwen-блоком (Pre-LN, RMSNorm, RoPE, QK-Norm, SwiGLU) на KK→RU лучше T5-197M kazRush **при том же масштабе (~200M)** и тех же данных.

kazRush — 2024, T5, 12+12, d=512, vocab 8k, **197M**, FLORES+ BLEU 18.8.  
Мы — 2026, тот же скелет 12+12 / d=512, Qwen-блок вместо T5-блока, vocab 32k, **~205M**.

100M не отменяем: это efficiency-абляция («можно ли почти то же меньшим»). Главная таблица — **~200M vs 197M**.

---

## 2. Encoder vs decoder vs cross-attention

Это не decoder-only Qwen. Это **обычный encoder–decoder MT**, как T5/NLLB, только блок внутри — Qwen-style.

```
Казахский                         Русский (уже сгенерированный)
    │                                      │
    ▼                                      ▼
┌─────────────────┐              ┌──────────────────────────┐
│ ENCODER         │              │ DECODER                  │
│ 12 × Qwen-блок  │              │ 12 × Qwen-блок           │
│                 │   K, V       │  1. masked self-attn     │
│ self-attn only  │─────────────►│  2. CROSS-ATTN  ◄──────  │
│ (bidirectional) │  все 12 слоёв│  3. SwiGLU               │
└─────────────────┘              └────────────┬─────────────┘
                                              ▼
                                         следующий RU токен
```

### Encoder — **без** cross-attention

Encoder видит **только казахский**. Ему нечего спрашивать у русского: target во время encoding ещё не существует (и при inference его нет).

Один encoder-слой:

```
x = kazakh_hidden
x = x + SelfAttn(RMSNorm(x))     # MHA (не GQA) + RoPE + QK-Norm
x = x + SwiGLU(RMSNorm(x))
```

Это Qwen-блок, но **двунаправленный** (как BERT/T5 encoder, не causal). RoPE, RMSNorm, QK-Norm, SwiGLU — да. Causal mask — нет.

### Decoder — Qwen-блок **плюс** cross-attention

Decoder генерирует русский. Ему нужно:

1. не заглядывать в будущее русского → **masked self-attn** (как в Qwen);
2. смотреть на казахский → **cross-attn**;
3. FFN → **SwiGLU**.

Один decoder-слой:

```
y = russian_hidden
y = y + MaskedSelfAttn(RMSNorm(y))   # GQA, causal + RoPE + QK-Norm
y = y + CrossAttn(RMSNorm(y),         # ВОТ СЮДА, только decoder
                  encoder_out)        # Q из RU, K/V из KK
y = y + SwiGLU(RMSNorm(y))
```

Cross-attention:

- **Q** — из текущего decoder state (русский)
- **K, V** — из encoder output (казахский)
- RoPE на cross-attn **нет** (разные последовательности, разные длины)
- QK-Norm — да
- ставится **в каждый из 12 decoder-слоёв**, не в 4

«4 слоя xattn» было только для **frozen Qwen grafting** (P2), чтобы не раздувать trainable. Здесь модель с нуля — стандарт Vaswani: cross-attn везде в decoder.

### Куда не вставлять cross-attn

| Место | Cross-attn? | Почему |
|---|---|---|
| Encoder | **нет** | не видит target |
| Decoder self-attn | нет, это другое | русский по русскому |
| Decoder после self-attn | **да, все 12 слоёв** | KK → RU |
| Между encoder и decoder отдельным модулем | нет | это и есть cross-attn внутри decoder |

Encoder и decoder — **оба Qwen-style**. Разница только в маске и в лишнем cross-attn у decoder.

---

## 3. Почему ~200M, а не 100M

| | 100M | **~200M (P0)** |
|---|---|---|
| vs kazRush 197M | «мы меньше, поэтому проиграли / выиграли непонятно чем» | **честный scale-matched** |
| 8×4090 | влезает слишком легко, GPU простаивают | всё ещё 1 карта с запасом, 3–8 ч на прогон |
| История для препода | efficiency | архитектура блока |

100M оставляем как **M_small**: если 200M выиграет, показываем, сколько можно срезать. Если 200M проиграет — 100M почти наверняка тоже.

---

## 4. Зафиксированный конфиг P0 (~205M)

Тот же скелет, что kazRush, другой блок.

```yaml
type: encoder-decoder          # не decoder-only
vocab: 32000                   # shared SentencePiece KK+RU, свои эмбеддинги
tie_embeddings: true
d_model: 512
n_heads: 8                     # query heads везде
head_dim: 64
n_kv_heads_encoder: 8          # MHA
n_kv_heads_decoder_self: 4     # GQA 8Q / 4KV
n_kv_heads_cross: 8            # MHA: encoder K/V полностью
n_encoder_layers: 12           # как kazRush
n_decoder_layers: 12
encoder_attn: full MHA         # bidirectional, RoPE, QK-Norm
decoder_self_attn: GQA         # causal, RoPE, QK-Norm, как Qwen
decoder_cross_attn: full MHA   # каждый слой, Q=RU K/V=KK, без RoPE
ffn: swiglu
ffn_hidden: 4096               # как d_ff kazRush, не 2048
norm: rmsnorm                  # Pre-LN
qk_norm: true
pos: rope                      # только self-attn
dropout: 0.1
max_len: 256
```

Оценка параметров (уточним `count_params`):

| Кусок | Params |
|---|---:|
| embeddings 32k × 512, tied | 16.4M |
| encoder 12L (MHA + SwiGLU 4096) | ~88M |
| decoder 12L (GQA self + MHA **cross** + SwiGLU 4096) | ~98M |
| **Total** | **~202M** |

kazRush: 197M, vocab 8k, gated-GELU, relative bias, Post-LN T5.  
Разница ~5M — vocab 32k vs 8k минус экономия GQA. Для таблицы это один масштаб.

GQA **только** в decoder self-attn (8 query / 4 KV). Encoder и cross-attn — полный MHA.  
На 200M это почти не экономит train FLOPs (~2–3M весов), зато схема как у Qwen и дешевле KV-cache на инференсе. Ablation A3 — decoder MHA вместо GQA.

---

## 5. Приоритеты

| # | Что | Зачем |
|---|---|---|
| **P0** | Qwen-style **~200M** с нуля | главный результат, scale-matched с kazRush |
| **P0** | FLORES+ BLEU / chrF / COMET | каждый прогон |
| **P1** | T5-блок ~200M на *нашем* пайплайне | изолировать «Qwen-блок», не данные |
| **P1** | Qwen-style ~100M | efficiency |
| **P1** | vocab 8k vs 32k на 200M | vocab vs блок |
| **P2** | Frozen `Qwen3-0.6B-Base` + encoder + 4× xattn (~700M / ~107M train) | вторая гипотеза |

---

## 6. 6–8×4090

200M на одной 4090 влезает спокойно (bf16, seq 256). 8 GPU на один DDP — нет.

```
GPU 0–3  → P0 Qwen-style 200M (DDP), global batch 256
GPU 4    → T5-блок 200M (наш пайплайн)
GPU 5    → Qwen-style 200M, vocab 8k
GPU 6    → Qwen-style 100M
GPU 7    → ablation: без QK-Norm / 10+10 / decoder MHA (без GQA)
```

| Сетап | 2–3 эпохи на ~12M пар |
|---|---|
| 1×4090, 200M | ~12–24 ч |
| 2–4×4090 DDP, 200M | **~5–12 ч** |
| 8 независимых прогонов по 1 GPU | те же 12–24 ч, 8 строк в таблице |
| Frozen Qwen 700M | ~12–30 ч на 1–2 GPU |

Global batch **256**, не 1024. 3 эпохи, early stop по FLORES dev. Test — один раз в конце.

---

## 7. Данные

Как у kazRush, без изменений состава:

| Корпус | Сырьё |
|---|---:|
| OPUS ru–kk | 0.72M |
| issai/kazparc | 2.15M |
| WMT19 kk–ru | 5.06M |
| TIL | 4.40M |
| **сырьё** | **~12.3M** |

Фильтр: дедуп → чистка → fasttext langid → LaBSE → opusfilter.  
Eval: только FLORES+ `dev` (early stop) и `devtest` (один раз в конце).

Скачалка: `scripts/download_raw.py`. Список URL и extra-корпуса — `data/README.md`.

**M1 = ровно эти 4 источника.** Свежий OPUS чуть больше их 718k — нормально.  
NLLB-mined / ekitil / code-switch — не мешать в M1 (ekitil уже = WMT19+KazParC). Extra-mix — отдельная строка таблицы.

LaBSE на 8×4090 — часы. Train не начинать на сыром корпусе как «финальный» прогон.

---

## 8. Train P0

```yaml
precision: bf16
flash_attn: true
devices: 2-4
strategy: ddp
micro_batch_per_gpu: 16        # 200M, seq 256; если OOM — 8, если влезает 32 — поднять
grad_accum: 4                  # 4 GPU × 16 × 4 = 256
epochs: 3
optimizer: adamw
lr: 2.0e-4
warmup_steps: 2000
schedule: cosine
min_lr: 2.0e-5
weight_decay: 0.01
clip_grad_norm: 1.0
label_smoothing: 0.1
dropout: 0.1
save_every: 2000
eval_every: 2000
eval: FLORES+ dev
gen: beam 5, max_len 256
```

---

## 9. Сетка

### Must

| ID | Что |
|---|---|
| M0 | kazRush: официальные 18.8 / 48.7 / 86.7 (+ свой прогон, если успеем) |
| **M1** | **Qwen-style 200M, полный корпус** |
| M2 | T5-блок ~200M, наш пайплайн, наш 32k vocab |
| M3 | Qwen-style 200M, vocab 8k как kazRush |

M2 обязателен. Без него скажут: «вы просто другой препроцессинг / другой vocab».

### Should

| ID | Что |
|---|---|
| A1 | Qwen-style 100M (efficiency) |
| A2 | с QK-Norm vs без |
| A3 | decoder GQA (P0) vs full MHA |
| A4 | 2 vs 3 эпохи |

### Nice

| ID | Что |
|---|---|
| N1 | Frozen Qwen3-0.6B-Base + encoder + 4 xattn |
| N2 | NLLB-600M как внешняя цифра (18.0 / 47.3 / 85.6) |

---

## 10. Календарь

**Неделя 1** — код, не метрики  
модель encoder–decoder, DDP, FLORES, чекпоинты, SentencePiece, оверфит 10k, фильтр корпуса.

**Неделя 2** — M1 + M2 + M3 параллельно.

**Неделя 3** — A1–A4, при необходимости N1, один test на FLORES+, черновик отчёта.

---

## 11. Как говорить преподу

> Основная модель — encoder–decoder ~200M: 12+12, d=512, как kazRush по скелету. Блок — Qwen-style (RMSNorm, RoPE, QK-Norm, SwiGLU). Encoder смотрит только на казахский, без cross-attention. Decoder — causal self-attn по русскому и cross-attention на encoder в каждом слое.
>
> Сравнение честное по размеру с kazRush 197M, те же данные, FLORES+. 100M — efficiency. Frozen Qwen 0.6B — второй эксперимент.

---

## 12. Definition of done (P0)

- [ ] `count_params` ≈ 190–210M
- [ ] encoder без cross-attn, decoder с cross-attn во всех 12 слоях
- [ ] SentencePiece 32k на отфильтрованном корпусе
- [ ] DDP 2–4×4090, resume
- [ ] FLORES+ BLEU / chrF / COMET
- [ ] таблица: kazRush | NLLB-600M | наш 200M | наш T5-блок 200M | наш 200M vocab 8k | наш 100M
- [ ] воспроизводимый yaml + seed
