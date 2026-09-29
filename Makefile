.PHONY: install data filter tokenizer overfit train eval train-small train-p0 train-large eval-small eval-p0 eval-large docker-build docker-shell docker-train docker-eval

install:
	pip install -r requirements.txt
	pip install -e . --no-deps

data:
	python scripts/download_raw.py --all-open

filter:
	python scripts/filter_data.py

tokenizer:
	python scripts/train_tokenizer.py

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

docker-build:
	docker build -t kk-ru:cuda128 .

docker-shell:
	docker run --rm -it --gpus all -v "$(PWD)/data:/workspace/kk-ru/data" -v "$(PWD)/checkpoints:/workspace/kk-ru/checkpoints" -v "$(PWD)/reports:/workspace/kk-ru/reports" kk-ru:cuda128 /bin/bash

docker-train:
	docker run --rm --gpus all --ipc=host --shm-size=16g -v "$(PWD)/data:/workspace/kk-ru/data" -v "$(PWD)/checkpoints:/workspace/kk-ru/checkpoints" -v "$(PWD)/reports:/workspace/kk-ru/reports" kk-ru:cuda128 kk_ru.train --config configs/p0.yaml

docker-eval:
	docker run --rm --gpus all -v "$(PWD)/data:/workspace/kk-ru/data" -v "$(PWD)/checkpoints:/workspace/kk-ru/checkpoints" -v "$(PWD)/reports:/workspace/kk-ru/reports" kk-ru:cuda128 kk_ru.eval --config configs/p0.yaml --checkpoint checkpoints/best/model.pt --split devtest
