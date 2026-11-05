# KK→RU Neural Machine Translation

Переводчик с казахского на русский. Основная модель P0 — encoder–decoder
на **~202M параметров** с Pre-LN, RMSNorm, RoPE, QK-Norm и SwiGLU, обученный
с нуля. Для сравнения реализована discrete-diffusion модель на **~223M параметров**.

Ориентир — [`deepvk/kazRush-kk-ru`](https://huggingface.co/deepvk/kazRush-kk-ru):
197M T5, опубликованные FLORES+ kk→ru BLEU 18.8 / chrF 48.7 / COMET 86.7.
P0 дополнительно обучали на переводах NLLB, поэтому обучающие данные различаются.
Результаты и условия сравнения описаны в [техническом отчёте](docs/p0-report/README.md).

## Структура

```
configs/               # конфиги small, p0, large и diffusion
src/kk_ru/             # библиотека: config, model, tokenizer, data, train, eval
scripts/               # подготовка данных, бенчмарки и экспорт ONNX
data/                  # сырые / отфильтрованные данные (вне git)
checkpoints/           # чекпоинты (вне git)
reports/               # метрики, графики (вне git)
docs/training-budget.md # токены, batch, VRAM и время обучения
```

## Быстрый старт

```bash
# Контейнер с CUDA и зависимостями
make docker-build

# Проверить GPU, токенизатор и подготовленные корпуса
make docker-check MODEL=p0 GPUS=6

# Короткий smoke test
make docker-overfit

# Обучить P0 на merged-корпусе
make docker-train MODEL=p0 GPUS=6

# Оценить FLORES+
make docker-eval MODEL=p0
```

## Discrete diffusion experiment

The diffusion model shares the corpus and SentencePiece vocabulary with the
baseline. It predicts target length from pooled encoder states, then iteratively
reveals masked target tokens with a bidirectional Transformer decoder and
timestep-conditioned AdaRMSNorm. The mask token is an extra input embedding and
is excluded from the output vocabulary. The default sampler uses 16 denoising
steps.

```bash
# Validate the training path on a 10k-example subset
make diffusion-overfit

# Full run on the same parallel corpus
make diffusion-train GPUS=6

# Compare 8/16/32 denoising steps on dev against the trained AR baseline
make diffusion-benchmark BASELINE_CHECKPOINT=checkpoints/p0-merged-3ep/best/model.pt

# Final held-out evaluation after selecting the step count on dev
make diffusion-eval DIFFUSION_CHECKPOINT=checkpoints/diffusion/best/model.pt
```

The benchmark writes JSON and CSV reports under `reports/diffusion/`. Do not use
devtest to choose the denoising-step count.

## Дополнительный казахский монолингвальный текст

Для синтетического расширения можно стримить предложения из
`kz-transformers/multidomain-kazakh-dataset` (около 25 млн текстов) и убрать
совпадения с текущим параллельным корпусом:

```bash
python scripts/download_monolingual.py \
  --output data/monolingual/kazakh.sentences.txt \
  --limit 4000000 \
  --existing data/filtered/train.kk-ru.tsv
```

Скрипт не скачивает весь датасет, пишет предложения построчно и сохраняет
`kazakh.sentences.txt.manifest.json` с параметрами выгрузки. Для резервного
источника можно указать `--dataset HuggingFaceFW/fineweb-2 --config kaz_Cyrl`.
Лицензию выбранного источника нужно проверить перед публикацией и обучением
коммерческой модели.

Для сравнения переводчиков на фиксированной выборке:

```bash
python scripts/prepare_translation_benchmark.py
```

Команда создаёт `data/benchmark/translation_candidates.tsv`: FLORES dev/devtest,
до 5 000 gold-пар из train и до 2 000 монолингвальных предложений после удаления
повторов. У строк `monolingual` поле `ru` пустое; они подходят для проверки
качества без эталона, но не для reference-метрик.

### Qwen3.5-0.8B baseline on Apple Silicon

Install `mlx-lm` in the project environment, then evaluate the unquantized
`Qwen/Qwen3.5-0.8B` checkpoint on all FLORES+ devtest pairs:

```bash
hf download Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --local-dir models/Qwen3.5-0.8B
PYTHONPATH=src .venv/bin/python scripts/eval_qwen.py
```

The evaluator loads an existing local checkpoint and uses
greedy decoding with thinking disabled and a 512-token output limit. It saves
raw responses, translations, BLEU/chrF/chrF++/TER, the prompt, hashes, runtime
versions, and settings under `reports/qwen3.5-0.8b/`. References are used only
for scoring. Each run needs a new `--out` directory. For a smoke test, use
`--split dev --limit 3 --out reports/qwen3.5-0.8b-smoke`.

To create and evaluate a 4-bit MLX checkpoint:

```bash
.venv/bin/python -m mlx_lm convert \
  --hf-path models/Qwen3.5-0.8B --mlx-path models/Qwen3.5-0.8B-4bit \
  -q --q-bits 4 --q-group-size 64
PYTHONPATH=src .venv/bin/python scripts/eval_qwen.py \
  --model models/Qwen3.5-0.8B-4bit --out reports/qwen3.5-0.8b-4bit
```

The completed baseline and comparison with P0, KazRush, and Qwen3-0.6B are in
[`docs/qwen3.5-0.8b-eval.md`](docs/qwen3.5-0.8b-eval.md).

Подготовка полного серверного цикла NLLB → merge → обучение описана в
[`docs/nllb-p0-pipeline.md`](docs/nllb-p0-pipeline.md). Она включает возобновление
перевода, проверку полноты shards и сборку merged-корпуса.

Короткий запуск на сервере после распаковки корпуса:

```bash
make nllb-download
make nllb-translate
make nllb-validate
make merge-synthetic
make docker-train MODEL=p0 GPUS=6
```

Все основные конфиги и команды Docker используют
`data/merged/train.kk-ru.tsv`; Make проверяет его наличие до запуска. Чтобы
явно выбрать прежний корпус, укажи `TRAIN_TSV=data/filtered/train.kk-ru.tsv`.
Однократный переход со старого filtered checkpoint описан в
[`docs/nllb-p0-pipeline.md`](docs/nllb-p0-pipeline.md).

## Конфиг и переопределения

Основной конфиг P0 — `configs/p0.yaml`. Параметры запуска можно переопределить:

```bash
make docker-train MODEL=p0 GPUS=6 OPTS="train.epochs=5 train.lr=1.0e-4"
make docker-train MODEL=p0 GPUS=6 OPTS="train.micro_batch_per_gpu=8"
```

## Тесты

Установи зависимости проекта и pytest в выбранное Python-окружение:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e '.[test]'
make test
```

Для существующего `.venv`: `make test PYTHON=.venv/bin/python`.

## Экспериментальная матрица

В репозитории есть три конфигурации одной encoder-decoder архитектуры:

| Конфиг | Параметры | Checkpoints | Reports |
|---|---:|---|---|
| `configs/small.yaml` | ~82M | `checkpoints/small` | `reports/small` |
| `configs/p0.yaml` | ~202M | `checkpoints/p0-merged-main` | `reports/p0-merged-main` |
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

Для запуска на шести RTX 4090 (24 GB на GPU) нужны Linux x86_64,
совместимый NVIDIA driver, Docker с NVIDIA Container Toolkit и подготовленные
данные. Проверь сервер до переноса файлов:

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
  merged/train.kk-ru.tsv \
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

Убедись, что существуют `data/merged/train.kk-ru.tsv`,
`data/tokenizer/kk-ru-sp32k.model` и оба файла в `data/eval/flores_plus/`.
Нужны свободные гигабайты для token cache: при первом обучении он автоматически
создаётся рядом с TSV в `data/merged/train.kk-ru.tsv.tok`.

### Собрать и проверить контейнер

Сборка скачивает CUDA-образ и несколько гигабайт Python/CUDA-пакетов:

```bash
make docker-build
```

Альтернатива для переноса уже собранного amd64-образа с другой машины: там выполни
`docker save kk-ru:cuda128 | gzip -1 > kk-ru-cuda128.tar.gz`, перенеси архив на сервер,
затем загрузи его:

```bash
gzip -dc kk-ru-cuda128.tar.gz | docker load
```

Проверь, что Docker видит все шесть GPU и необходимые файлы для P0:

```bash
make docker-check MODEL=p0 GPUS=6
```

Первый запуск обучения строит token cache до появления строк loss в логе.

### Короткий запуск и обучение

Сначала выполни короткую проверку на одной GPU и маленькой модели. Она делает 10
optimizer steps и пишет изолированные результаты в `checkpoints/smoke` и `runs/smoke`:

```bash
make docker-overfit
```

Основной эксперимент P0 на шести RTX 4090:

```bash
make docker-train MODEL=p0 GPUS=6
```

Доступны `MODEL=small`, `MODEL=p0` и `MODEL=large`; по умолчанию выбран P0.
`GPUS` задаёт число процессов обучения. Чтобы выбрать конкретные устройства,
ограничь их видимость для контейнера через Docker `--gpus`.

Снизить micro-batch при нехватке памяти или переопределить другие параметры можно
без правки YAML. Например, вдвое меньший micro-batch с удвоенным grad accumulation
сохраняет прежний effective batch:

```bash
make docker-train MODEL=p0 GPUS=6 \
  OPTS="train.micro_batch_per_gpu=8 train.grad_accum=8"
```

### Смотреть логи и чекпоинты

Запуск идёт в текущем терминале. В нём печатаются параметры модели, число шагов,
loss и метрики при оценке. Для загрузки GPU открой второе SSH-окно:

```bash
watch -n 1 nvidia-smi
```

Каталог логов задан в `paths.logs`: для P0 это `runs/p0-merged-main`.
В нём сохраняются события TensorBoard в `kk-ru/` и построчный лог
optimizer steps в `metrics.jsonl`. В JSONL записываются loss,
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

Затем открой [TensorBoard](http://localhost:6006). Checkpoints и логи сохраняются в примонтированные
каталоги хоста и останутся после завершения контейнера. Веса и состояние обучения
сохраняются каждые `save_every` шагов (сейчас 2000); безопаснее останавливать запуск
после появления очередного `step-N` в `paths.checkpoints`. Прерывание до первого
чекпоинта не сохраняет прогресс.

Продолжить с сохранённого шага можно так:

```bash
make docker-train MODEL=p0 GPUS=6 RESUME=checkpoints/p0-merged-main/step-N
```

Замени `step-N` на существующий checkpoint. Возобновление использует сохранённые
веса, optimizer, scheduler и позицию в epoch.

### Оценить результат

Dev-оценка запускается во время обучения каждые `eval_every` шагов и обновляет
`best/model.pt` в `paths.checkpoints`. Для финального FLORES+ devtest:

```bash
make docker-eval MODEL=p0
```

После нескольких моделей можно собрать общую таблицу:

```bash
make benchmark
```

Итоги появятся в `reports/benchmark.json` и `reports/benchmark.csv`.
