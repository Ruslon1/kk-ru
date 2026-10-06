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
docs/training-budget.md # токены, batch, VRAM и время обучения
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
| Проверить окружение | `make check-env` |
| Собрать Docker | `make docker-build` |
| Запустить Docker-обучение | `make docker-train GPUS=8` |
| Сравнить checkpoints | `make benchmark` |
| Проверить OpenRouter-модели и стоимость без перевода | `make benchmark-openrouter` |
| Запустить тесты | `make test` |

### OpenRouter benchmark

The hosted-model benchmark uses the same FLORES+ `devtest` references for each
model. It fetches current model prices from OpenRouter, estimates the full run
with a 25% safety margin, and refuses to start if the estimate exceeds the
budget. The default candidate list is Qwen3.8 27B, Qwen3.8 2.4T A95B,
Gemma 4 26B A4B, Gemma 4 31B, and Qwen3.8 Max. Confirm catalog availability and
pricing with a dry run before spending credits:

```bash
make benchmark-openrouter

export OPENROUTER_API_KEY="..."
python scripts/benchmark_openrouter.py \
  --config configs/p0.yaml --split devtest --budget-usd 8.75
```

Use `--models` to select candidates, `--limit N` for a small smoke test,
`--out DIR` for a report directory, and `--all-metrics` to request COMET,
COMETKiwi, XCOMET-XL, and multilingual BERTScore in addition to lexical
metrics. Neural metrics are optional; missing packages or model downloads are
recorded in `metric_errors` instead of being silently omitted. Individual flags
such as `--comet-model MODEL` override the standard identifiers. Each response
is flushed to a per-model JSONL file so an interrupted run can resume. The
report directory contains a manifest, raw records, translations, and a summary
with BLEU, chrF, chrF++, TER, FLORES spBLEU, optional neural metrics, cost,
provider, and latency in both JSON and CSV.
The API key is read from the environment and is never written to reports.

## Экспериментальная матрица

В репозитории есть три конфигурации одной Qwen-подобной encoder-decoder архитектуры:

| Конфиг | Параметры | Checkpoints | Reports |
|---|---:|---|---|
| `configs/small.yaml` | ~82M | `checkpoints/small` | `reports/small` |
| `configs/p0.yaml` | ~202M | `checkpoints/p0` | `reports/p0` |
| `configs/large.yaml` | ~578M | `checkpoints/large` | `reports/large` |

У всех конфигов одинаковые данные, tokenizer, seed и evaluation-параметры. Размер
модели меняется через `d_model`, число слоёв и ширину SwiGLU. Для отдельной GPU
конфигурацию можно переопределить без изменения YAML:

```bash
accelerate launch --num_processes 1 -m kk_ru.train \
  --config configs/large.yaml \
  --opts train.micro_batch_per_gpu=2 train.grad_accum=32
```

Каждый запуск сохраняет `resolved_config.yaml` и `run_manifest.json` в своём
каталоге checkpoints. Manifest содержит git revision, версии Python/PyTorch/CUDA,
список GPU и время старта.

## Обучение на Linux с NVIDIA Docker

Первый запуск не гарантирован «в один клик»: на сервере должны быть Linux x86_64,
совместимый NVIDIA driver, Docker с NVIDIA Container Toolkit, восемь доступных GPU
и подготовленные файлы данных. Проверь сервер до переноса файлов:

```bash
nvidia-smi
docker --version
docker run --rm --gpus all --entrypoint nvidia-smi \
  nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
```

### Подготовить код и данные

Склонируй репозиторий на сервер и перейди в его каталог:

```bash
git clone <REPOSITORY_URL> kk-ru
cd kk-ru
```

Для обучения не нужно переносить `data/raw`: достаточно готового корпуса,
токенизатора и eval-наборов.
На машине, где эти файлы уже есть, собери архив:

```bash
tar -czf kk-ru-training-data.tar.gz -C data \
  filtered/train.kk-ru.tsv \
  tokenizer/kk-ru-sp32k.model \
  eval/flores_plus/dev.kk-ru.tsv \
  eval/flores_plus/devtest.kk-ru.tsv
scp kk-ru-training-data.tar.gz USER@SERVER:~/kk-ru/
```

На сервере, из корня клона:

```bash
mkdir -p data
tar -xzf kk-ru-training-data.tar.gz -C data
df -h .
```

Убедись, что существуют `data/filtered/train.kk-ru.tsv`,
`data/tokenizer/kk-ru-sp32k.model` и оба файла в `data/eval/flores_plus/`.
Нужны свободные гигабайты для token cache: при первом обучении он автоматически
создаётся рядом с TSV в `data/filtered/train.kk-ru.tsv.tok`.

### Собрать и проверить контейнер

Сборка скачивает CUDA-образ и несколько гигабайт Python/CUDA-пакетов. При медленном
интернете собери `kk-ru:cuda128` один раз и дождись окончания команды:

```bash
make docker-build
```

Альтернатива для переноса уже собранного amd64-образа с другой машины: там выполни
`docker save kk-ru:cuda128 | gzip -1 > kk-ru-cuda128.tar.gz`, перенеси архив на сервер,
затем загрузи его:

```bash
gzip -dc kk-ru-cuda128.tar.gz | docker load
```

Проверь, что Docker видит все восемь GPU и необходимые файлы для P0:

```bash
make docker-check MODEL=p0 GPUS=8
```

Если проверка завершилась ошибкой, не запускай длинное обучение: исправь указанную
проблему с GPU, драйвером, контейнером или файлами. Первый запуск обучения также
потратит время на создание token cache; этот этап выполняется до появления строк
обучения в логе.

### Короткий запуск и обучение

Сначала выполни короткую проверку на одной GPU и маленькой модели. Она делает 10
optimizer steps и пишет изолированные результаты в `checkpoints/smoke` и `runs/smoke`:

```bash
make docker-overfit
```

Основной эксперимент P0 на восьми одинаковых GPU:

```bash
make docker-train MODEL=p0 GPUS=8
```

Доступны `MODEL=small`, `MODEL=p0` и `MODEL=large`. По умолчанию это три разных
конфига; для первого полного эксперимента используй P0. Аргумент `GPUS` должен
совпадать с числом GPU, которые будут участвовать в запуске. Если на сервере есть
дополнительная A100, начни с восьми одинаковых 5090; смешение разных GPU делает
скорость обучения менее предсказуемой.

Снизить micro-batch при нехватке памяти или переопределить другие параметры можно
без правки YAML. Например, вдвое меньший micro-batch с удвоенным grad accumulation
сохраняет прежний effective batch:

```bash
make docker-train MODEL=p0 GPUS=8 \
  OPTS="train.micro_batch_per_gpu=8 train.grad_accum=8"
```

### Смотреть логи и чекпоинты

Запуск идёт в текущем терминале. В нём печатаются параметры модели, число шагов,
loss и метрики при оценке. Для загрузки GPU открой второе SSH-окно:

```bash
watch -n 1 nvidia-smi
```

TensorBoard события пишутся в `runs/<model>/kk-ru`, а построчный резервный лог
optimizer steps — в `runs/<model>/metrics.jsonl`. В JSONL сохраняются loss,
learning rate, gradient norm, время шага, throughput токенов и VRAM текущего
процесса. Открой ещё один терминал на сервере:

```bash
docker run --rm -it -p 6006:6006 \
  -v "$PWD/runs:/workspace/kk-ru/runs" kk-ru:cuda128 \
  tensorboard --logdir /workspace/kk-ru/runs --host 0.0.0.0
```

Для удалённого сервера пробрось порт в отдельном локальном терминале:

```bash
ssh -L 6006:localhost:6006 USER@SERVER
```

Затем открой `http://localhost:6006`. Checkpoints и логи сохраняются в примонтированные
каталоги хоста и останутся после завершения контейнера. Веса и состояние обучения
сохраняются каждые `save_every` шагов (сейчас 2000); безопаснее останавливать запуск
после появления очередного `checkpoints/<model>/step-N`. Прерывание до первого
чекпоинта не сохраняет прогресс.

Продолжить с сохранённого шага можно так:

```bash
make docker-train MODEL=p0 GPUS=8 RESUME=checkpoints/p0/step-2000
```

Замени `step-2000` на существующий checkpoint. Возобновление использует сохранённые
веса, optimizer, scheduler и позицию в epoch.

### Оценить результат

Dev-оценка запускается во время обучения каждые `eval_every` шагов и обновляет
`checkpoints/<model>/best/model.pt`. Для финального FLORES+ devtest:

```bash
make docker-eval MODEL=p0
```

После нескольких моделей можно собрать общую таблицу:

```bash
make benchmark
```

Итоги появятся в `reports/benchmark.json` и `reports/benchmark.csv`.

Каждый запуск пишет `resolved_config.yaml` и `run_manifest.json` в каталог модели.
Manifest фиксирует версии Python/PyTorch/CUDA, видимые GPU и время старта.

### LoRA fine-tuning Qwen3-0.6B

Для отдельного эксперимента с decoder-only Qwen используется LoRA. Базовые веса
не меняются: adapter сохраняется в checkpoints/qwen3-0.6b-lora, а TensorBoard
пишет в checkpoints/qwen3-0.6b-lora/runs.

Сначала проверь корпус и собери образ:

    make docker-build
    make docker-check MODEL=p0 GPUS=8

Перед полным прогоном сделай короткий DDP smoke:

    docker run --rm --gpus all --ipc=host --shm-size=16g \
      -v "$PWD/data:/workspace/kk-ru/data" \
      -v "$PWD/checkpoints:/workspace/kk-ru/checkpoints" \
      -v "$PWD/runs:/workspace/kk-ru/runs" \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      kk-ru:cuda128 accelerate launch --num_processes 8 \
      scripts/finetune_qwen.py --limit 1000 --eval-limit 20 --max-steps 5 \
      --batch-size 1 --grad-accum 1 --eval-steps 5 --save-steps 5 \
      --output checkpoints/qwen3-0.6b-lora-smoke

Полный запуск:

    tmux new -s qwen-lora
    make docker-qwen-lora 2>&1 | tee runs/qwen3-0.6b-lora/train.log

Продолжение после checkpoint:

    docker run --rm --gpus all --ipc=host --shm-size=16g \
      -v "$PWD/data:/workspace/kk-ru/data" \
      -v "$PWD/checkpoints:/workspace/kk-ru/checkpoints" \
      -v "$PWD/runs:/workspace/kk-ru/runs" \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      kk-ru:cuda128 accelerate launch --num_processes 8 \
      scripts/finetune_qwen.py --resume checkpoints/qwen3-0.6b-lora/checkpoint-1000 \
      --output checkpoints/qwen3-0.6b-lora

Оценка adapter на FLORES devtest:

    make docker-qwen-eval
