.PHONY: install data filter tokenizer overfit train eval test lint

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e . --no-deps

data:
	python scripts/download_raw.py --all-open

filter:
	python scripts/filter_data.py

tokenizer:
	python scripts/train_tokenizer.py

# Оверфит — гейт: loss должен упасть к ~0 на 10k пар.
overfit:
	accelerate launch -m kk_ru.train --config configs/p0.yaml --overfit

train:
	accelerate launch -m kk_ru.train --config configs/p0.yaml

eval:
	python -m kk_ru.eval --config configs/p0.yaml --split devtest

test:
	pytest

lint:
	ruff check src tests scripts
