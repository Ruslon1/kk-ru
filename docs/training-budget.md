# Техническая спецификация обучения

## Базовые параметры

| Параметр | Значение |
|---|---:|
| Корпус | `data/filtered/train.kk-ru.tsv` |
| Пар в корпусе | 6 851 060 |
| Токенизатор | `data/tokenizer/kk-ru-sp32k.model` |
| Vocabulary | 32 000 |
| `max_len` | 256 |
| Эпохи | 3 |
| Precision | BF16 autocast |
| Optimizer | AdamW (`betas=(0.9, 0.95)`) |
| Learning rate | `2e-4` → `2e-5`, cosine |
| Warmup | 2 000 optimizer steps |
| `eval_every` / `save_every` | 2 000 steps |

Конфиги: `configs/small.yaml`, `configs/p0.yaml`, `configs/large.yaml`.

## Токены

В token cache: source получает BOS, target получает BOS и EOS; затем обе стороны
обрезаются до 256 токенов. `target labels = target[:, 1:]`.

| Величина | 1 эпоха | 3 эпохи |
|---|---:|---:|
| Пары | 6 851 060 | 20 553 180 посещений |
| Source-токены с BOS | 189 745 655 | 569 236 965 |
| Target-токены с BOS/EOS | 207 790 427 | 623 371 281 |
| Все закодированные токены | **397 536 082** | **1 192 608 246** |
| Target prediction labels | 200 939 367 | **602 818 101** |
| Source + decoder input positions | **390 685 022** | **1 172 055 066** |

Последняя строка не включает padding внутри batch. Dynamic batching использует
`LengthBatchSampler` и сортировку по суммарной длине source+target.

## Модели и batch

`micro_batch_per_gpu` — пары на GPU за один forward/backward.
`grad_accum` — micro-batch на один optimizer step.

| Конфиг | Параметры | Precision | Micro-batch / GPU | `grad_accum` | Effective batch, 8 GPU | Effective batch, 1 GPU |
|---|---:|---|---:|---:|---:|---:|
| `small` | 81 905 024 (~81.9M) | BF16 | 16 | 4 | 512 | 64 |
| `p0` | 202 016 256 (~202.0M) | BF16 | 16 | 4 | 512 | 64 |
| `large` | 578 290 432 (~578.3M) | BF16 | 8 | 8 | 512 | 64 |

### Optimizer steps

| Запуск | Effective batch | Steps / эпоху | Steps / 3 эпохи | Eval при `eval_every=2000` |
|---|---:|---:|---:|---:|
| 8 GPU, YAML | 512 | ~13 381 | **~40 143** | ~20 |
| 1 A100, YAML | 64 | ~107 048 | **~321 144** | ~160 |
| 1 A100, batch 512 | 512 | ~13 381 | **~40 143** | ~20 |

Для batch 512 на одной A100:

```bash
# small/p0
make docker-train MODEL=p0 GPUS=1 OPTS="train.grad_accum=32"

# large
make docker-train MODEL=large GPUS=1 OPTS="train.grad_accum=64"
```

`Accelerate` может дополнить последний DDP shard несколькими примерами.

## VRAM на одну GPU

Базовый расчёт: DDP хранит полную модель на каждой GPU; параметры, градиенты и
два состояния AdamW считаются как FP32, то есть `16 bytes/parameter`.

| Конфиг | Вес FP32 | Вес + градиенты + AdamW | Плановый peak* |
|---|---:|---:|---:|
| `small` | 0.31 GiB | 1.22 GiB | **3–6 GiB** |
| `p0` | 0.75 GiB | 3.01 GiB | **6–10 GiB** |
| `large` | 2.15 GiB | 8.62 GiB | **14–24 GiB** |

\* Плановый peak включает активации при `max_len=256`, attention tensors, DDP
и CUDA runtime. Это оценка, не замер на конкретной машине. При OOM:

```bash
# 8 GPU, effective batch сохраняется
make docker-train MODEL=large GPUS=8 \
  OPTS="train.micro_batch_per_gpu=4 train.grad_accum=16"
```

## Время полного запуска

Оценка для CUDA 12.x, BF16, готового token cache и `eval_every=2000`.
Диапазоны не измерены на этом хосте; перед полным запуском нужна калибровка
на целевых GPU.

| Конфиг | 8 × RTX 5090 | 1 × A100 80GB, batch 512 |
|---|---:|---:|
| `small` | **~0.75–1.5 ч** | **~2–4 ч** |
| `p0` | **~2–4 ч** | **~5–9 ч** |
| `large` | **~6–12 ч** | **~15–28 ч** |

Время включает периодическую оценку на dev (997 предложений). Первый запуск
дополнительно строит token cache.

Калибровка:

```bash
make docker-train MODEL=p0 GPUS=8 \
  OPTS="train.max_steps=100 train.epochs=1 train.save_every=0 train.eval_every=0"
watch -n 1 nvidia-smi
```

## Запуски

```bash
# 8 × RTX 5090
make docker-train MODEL=p0 GPUS=8
make docker-train MODEL=large GPUS=8

# 1 × A100, effective batch 512
make docker-train MODEL=p0 GPUS=1 OPTS="train.grad_accum=32"
make docker-train MODEL=large GPUS=1 OPTS="train.grad_accum=64"
```
