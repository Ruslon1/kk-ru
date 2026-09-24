# KK→RU Neural Machine Translation

Encoder–decoder **~200M** с Qwen-блоком (Pre-LN, RMSNorm, RoPE, QK-Norm, SwiGLU),
обучение с нуля. Цель — побить [`deepvk/kazRush-kk-ru`](https://huggingface.co/deepvk/kazRush-kk-ru)
(197M T5, FLORES+ kk→ru BLEU 18.8 / chrF 48.7 / COMET 86.7) при **том же масштабе и тех же данных**.

## Структура

```
configs/p0.yaml        # единственный источник правды по конфигу
src/kk_ru/             # библиотека: config, model, tokenizer, data, train, eval
scripts/               # пайплайн данных (download / filter)
data/                  # сырые / отфильтрованные данные (вне git)
checkpoints/           # чекпоинты (вне git)
reports/               # метрики, графики (вне git)
```

## Быстрый старт

```bash
# 1. Окружение — на машине с 4090, Python 3.10–3.12 (torch отстаёт на 3.13+)
make install

# 2. Сырые данные (уже в data/raw; при необходимости докачать)
make data

# 3. Фильтр корпуса: дедуп → чистка → langid → LaBSE
make filter

# 4. Оверфит 10k — гейт перед большим прогоном (loss → 0)
make overfit

# 5. Обучение P0 (Accelerate DDP)
make train

# 6. Оценка FLORES+ (BLEU / chrF / chrF++ / spBLEU)
make eval
```

## Конфиг и переопределения

Один YAML — `configs/p0.yaml`. Для свупов править файл не обязательно:

```bash
python -m kk_ru.train --config configs/p0.yaml --opts train.epochs=5 model.d_model=768
accelerate launch -m kk_ru.train --config configs/p0.yaml --opts train.micro_batch_per_gpu=8
```

## Команды

| Что | Команда |
|---|---|
| Установить окружение | `make install` |
| Скачать открытые данные | `make data` |
| Отфильтровать корпус | `make filter` |
| Оверфит-тест | `make overfit` |
| Обучение P0 | `make train` |
| Оценка | `make eval` |

Для продолжения обучения используется Accelerate checkpoint directory:

```bash
accelerate launch -m kk_ru.train \
  --config configs/p0.yaml \
  --resume checkpoints/step-2000
```

Оценка принимает `checkpoints/best/model.pt` или обычный файл весов. COMET запускается
отдельно и требует локальный checkpoint модели:

```bash
python -m kk_ru.eval --config configs/p0.yaml \
  --checkpoint checkpoints/best/model.pt \
  --split devtest \
  --comet-checkpoint /path/to/comet.ckpt
```
