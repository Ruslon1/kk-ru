.PHONY: install data filter tokenizer overfit train eval docker-build docker-shell docker-train docker-eval

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

eval:
	python -m kk_ru.eval --config configs/p0.yaml --checkpoint checkpoints/step-471/model.pt --split devtest

docker-build:
	docker build -t kk-ru:cuda128 .

docker-shell:
	docker run --rm -it --gpus all -v "$(PWD)/data:/workspace/kk-ru/data" -v "$(PWD)/checkpoints:/workspace/kk-ru/checkpoints" -v "$(PWD)/reports:/workspace/kk-ru/reports" kk-ru:cuda128 /bin/bash

docker-train:
	docker run --rm --gpus all --ipc=host --shm-size=16g -v "$(PWD)/data:/workspace/kk-ru/data" -v "$(PWD)/checkpoints:/workspace/kk-ru/checkpoints" -v "$(PWD)/reports:/workspace/kk-ru/reports" kk-ru:cuda128 kk_ru.train --config configs/p0.yaml

docker-eval:
	docker run --rm --gpus all -v "$(PWD)/data:/workspace/kk-ru/data" -v "$(PWD)/checkpoints:/workspace/kk-ru/checkpoints" -v "$(PWD)/reports:/workspace/kk-ru/reports" kk-ru:cuda128 kk_ru.eval --config configs/p0.yaml --checkpoint checkpoints/best/model.pt --split devtest
