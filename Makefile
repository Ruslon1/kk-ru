.PHONY: install data filter tokenizer overfit train eval

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
	python -m kk_ru.eval --config configs/p0.yaml --checkpoint checkpoints/latest/model.pt --split devtest
