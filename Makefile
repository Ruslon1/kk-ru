GPUS ?= 8
PLATFORM ?= linux/amd64
MODEL ?= p0
CONFIG = configs/$(MODEL).yaml
CHECKPOINT = checkpoints/$(MODEL)
REPORTS = reports/$(MODEL)
RUNS = runs/$(MODEL)

.PHONY: install data filter tokenizer check-env overfit train eval train-small train-p0 train-large eval-small eval-p0 eval-large benchmark docker-build docker-shell docker-train docker-eval

install:
	pip install -r requirements.txt
	pip install -e . --no-deps

data:
	python scripts/download_raw.py --all-open

filter:
	python scripts/filter_data.py

tokenizer:
	python scripts/train_tokenizer.py

check-env:
	python scripts/check_environment.py

overfit:
	accelerate launch -m kk_ru.train --config configs/p0.yaml --overfit

train:
	accelerate launch -m kk_ru.train --config configs/p0.yaml

train-small:
	accelerate launch -m kk_ru.train --config configs/small.yaml

train-p0:
	accelerate launch -m kk_ru.train --config configs/p0.yaml

train-large:
	accelerate launch -m kk_ru.train --config configs/large.yaml

eval:
	python -m kk_ru.eval --config configs/p0.yaml --checkpoint checkpoints/p0/best/model.pt --split devtest

eval-small:
	python -m kk_ru.eval --config configs/small.yaml --checkpoint checkpoints/small/best/model.pt --split devtest

eval-p0:
	python -m kk_ru.eval --config configs/p0.yaml --checkpoint checkpoints/p0/best/model.pt --split devtest

eval-large:
	python -m kk_ru.eval --config configs/large.yaml --checkpoint checkpoints/large/best/model.pt --split devtest

benchmark:
	python scripts/benchmark.py --spbleu

docker-build:
	docker build --platform $(PLATFORM) -t kk-ru:cuda128 .

docker-shell:
	docker run --rm -it --gpus all -v "$(PWD)/data:/workspace/kk-ru/data" -v "$(PWD)/checkpoints:/workspace/kk-ru/checkpoints" -v "$(PWD)/reports:/workspace/kk-ru/reports" kk-ru:cuda128 /bin/bash

docker-train:
	@test -f "$(CONFIG)" || (echo "unknown model: $(MODEL) (expected small, p0, or large)" >&2; exit 2)
	docker run --rm --gpus all --ipc=host --shm-size=16g -v "$(PWD)/data:/workspace/kk-ru/data" -v "$(PWD)/checkpoints:/workspace/kk-ru/checkpoints" -v "$(PWD)/reports:/workspace/kk-ru/reports" -v "$(PWD)/runs:/workspace/kk-ru/runs" kk-ru:cuda128 accelerate launch --num_processes $(GPUS) -m kk_ru.train --config $(CONFIG)

docker-eval:
	@test -f "$(CONFIG)" || (echo "unknown model: $(MODEL) (expected small, p0, or large)" >&2; exit 2)
	docker run --rm --gpus all -v "$(PWD)/data:/workspace/kk-ru/data" -v "$(PWD)/checkpoints:/workspace/kk-ru/checkpoints" -v "$(PWD)/reports:/workspace/kk-ru/reports" kk-ru:cuda128 python3 -m kk_ru.eval --config $(CONFIG) --checkpoint $(CHECKPOINT)/best/model.pt --split devtest
