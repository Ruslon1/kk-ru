GPUS ?= 6
PLATFORM ?= linux/amd64
MODEL ?= p0
CONFIG = configs/$(MODEL).yaml
CHECKPOINT ?= $(if $(filter p0,$(MODEL)),checkpoints/p0-merged-main,checkpoints/$(MODEL))
REPORTS = reports/$(MODEL)
RUNS = runs/$(MODEL)
MONOLINGUAL ?= data/monolingual/kazakh.sentences.txt
NLLB_MODEL ?= models/nllb-200-3.3B
NLLB_SHARDS ?= data/synthetic/nllb-rank-shards
NLLB_BATCH ?= 16
NLLB_BEAMS ?= 1
QWEN_OUTPUT ?= checkpoints/qwen3-0.6b-lora
QWEN_BATCH ?= 4
QWEN_GRAD_ACCUM ?= 8
QWEN_ARGS ?=
QWEN_ADAPTER ?= checkpoints/qwen3-0.6b-lora
QWEN_EVAL_OUT ?= reports/qwen3-0.6b-lora
QWEN_EVAL_ARGS ?=
BASELINE_CHECKPOINT ?= checkpoints/p0-merged-3ep/best/model.pt
TRAIN_TSV ?= data/merged/train.kk-ru.tsv
DIFFUSION_CHECKPOINT ?= checkpoints/diffusion/best/model.pt

.PHONY: install data filter tokenizer check-env overfit train eval train-small train-p0 train-large eval-small eval-p0 eval-large benchmark benchmark-openrouter diffusion-train diffusion-overfit diffusion-eval diffusion-benchmark qwen-lora qwen-eval docker-qwen-lora docker-qwen-eval test docker-build docker-shell docker-check docker-overfit docker-train docker-train-merged docker-diffusion-train docker-diffusion-overfit docker-diffusion-eval docker-diffusion-benchmark docker-eval nllb-download nllb-translate nllb-validate merge-synthetic

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
	accelerate launch --multi_gpu --num_processes $(GPUS) -m kk_ru.train --config configs/p0.yaml

train-small:
	accelerate launch --multi_gpu --num_processes $(GPUS) -m kk_ru.train --config configs/small.yaml

train-p0:
	accelerate launch --multi_gpu --num_processes $(GPUS) -m kk_ru.train --config configs/p0.yaml

train-large:
	accelerate launch --multi_gpu --num_processes $(GPUS) -m kk_ru.train --config configs/large.yaml

eval:
	python -m kk_ru.eval --config configs/p0.yaml --checkpoint checkpoints/p0-merged-main/best/model.pt --split devtest

eval-small:
	python -m kk_ru.eval --config configs/small.yaml --checkpoint checkpoints/small/best/model.pt --split devtest

eval-p0:
	python -m kk_ru.eval --config configs/p0.yaml --checkpoint checkpoints/p0-merged-main/best/model.pt --split devtest

eval-large:
	python -m kk_ru.eval --config configs/large.yaml --checkpoint checkpoints/large/best/model.pt --split devtest

benchmark:
	python scripts/benchmark.py --spbleu

diffusion-train:
	$(MAKE) docker-diffusion-train GPUS=$(GPUS)

diffusion-overfit:
	$(MAKE) docker-diffusion-overfit

diffusion-eval:
	$(MAKE) docker-diffusion-eval CHECKPOINT=$(DIFFUSION_CHECKPOINT)

diffusion-benchmark:
	$(MAKE) docker-diffusion-benchmark CHECKPOINT=$(DIFFUSION_CHECKPOINT)

benchmark-openrouter:
	python scripts/benchmark_openrouter.py --config configs/p0.yaml --split devtest --budget-usd 8.75 --dry-run

qwen-lora:
	accelerate launch --multi_gpu --num_processes $(GPUS) scripts/finetune_qwen.py --output checkpoints/qwen3-0.6b-lora

qwen-eval:
	python scripts/evaluate_qwen.py --adapter checkpoints/qwen3-0.6b-lora --output reports/qwen3-0.6b-lora

docker-qwen-lora:
	docker run --rm --gpus all --ipc=host --shm-size=16g -v "$(PWD)/data:/workspace/kk-ru/data" -v "$(PWD)/checkpoints:/workspace/kk-ru/checkpoints" -v "$(PWD)/runs:/workspace/kk-ru/runs" -v "$$HOME/.cache/huggingface:/root/.cache/huggingface" kk-ru:cuda128 accelerate launch --multi_gpu --num_processes $(GPUS) scripts/finetune_qwen.py --output $(QWEN_OUTPUT) --batch-size $(QWEN_BATCH) --grad-accum $(QWEN_GRAD_ACCUM) $(QWEN_ARGS)

docker-qwen-eval:
	docker run --rm --gpus all -v "$(PWD)/data:/workspace/kk-ru/data" -v "$(PWD)/reports:/workspace/kk-ru/reports" -v "$(PWD)/checkpoints:/workspace/kk-ru/checkpoints" -v "$$HOME/.cache/huggingface:/root/.cache/huggingface" kk-ru:cuda128 python3 scripts/evaluate_qwen.py --adapter $(QWEN_ADAPTER) --output $(QWEN_EVAL_OUT) $(QWEN_EVAL_ARGS)

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

docker-build:
	docker build --platform $(PLATFORM) -t kk-ru:cuda128 .

docker-shell:
	docker run --rm -it --gpus all -v "$(PWD)/data:/workspace/kk-ru/data" -v "$(PWD)/checkpoints:/workspace/kk-ru/checkpoints" -v "$(PWD)/reports:/workspace/kk-ru/reports" kk-ru:cuda128 /bin/bash

docker-check:
	docker run --rm --gpus all -v "$(PWD)/data:/workspace/kk-ru/data" kk-ru:cuda128 python3 scripts/check_environment.py --config $(CONFIG) --min-gpus $(GPUS)

docker-overfit:
	docker run --rm --gpus all --ipc=host --shm-size=16g -v "$(PWD)/data:/workspace/kk-ru/data" -v "$(PWD)/checkpoints:/workspace/kk-ru/checkpoints" -v "$(PWD)/reports:/workspace/kk-ru/reports" -v "$(PWD)/runs:/workspace/kk-ru/runs" kk-ru:cuda128 accelerate launch --num_processes 1 -m kk_ru.train --config configs/small.yaml --overfit --opts train.max_steps=10 train.epochs=1 train.micro_batch_per_gpu=1 train.grad_accum=1 train.save_every=10 train.eval_every=0 paths.checkpoints=checkpoints/smoke paths.reports=reports/smoke paths.logs=runs/smoke

docker-train:
	@test -f "$(CONFIG)" || (echo "unknown model: $(MODEL) (expected small, p0, or large)" >&2; exit 2)
	@test -s "$(TRAIN_TSV)" || (echo "training corpus is missing: $(TRAIN_TSV)" >&2; exit 2)
	docker run --rm --gpus all --ipc=host --shm-size=16g -v "$(PWD)/data:/workspace/kk-ru/data" -v "$(PWD)/checkpoints:/workspace/kk-ru/checkpoints" -v "$(PWD)/reports:/workspace/kk-ru/reports" -v "$(PWD)/runs:/workspace/kk-ru/runs" kk-ru:cuda128 accelerate launch --multi_gpu --num_processes $(GPUS) -m kk_ru.train --config $(CONFIG) --opts data.train_tsv=$(TRAIN_TSV) $(if $(strip $(OPTS)),$(OPTS)) $(if $(strip $(RESUME)),--resume $(RESUME))

nllb-download:
	hf download facebook/nllb-200-3.3B --local-dir $(NLLB_MODEL)

nllb-translate:
	PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=error torchrun --standalone --nproc_per_node=$(GPUS) scripts/translate_nllb.py --input $(MONOLINGUAL) --out-dir $(NLLB_SHARDS) --model $(NLLB_MODEL) --batch-size $(NLLB_BATCH) --bucket-size 2048 --num-beams $(NLLB_BEAMS)

nllb-validate:
	python scripts/validate_nllb_shards.py --input $(MONOLINGUAL) --shards $(NLLB_SHARDS) --db data/merged/nllb-validation.sqlite

merge-synthetic:
	python scripts/merge_synthetic.py --base data/filtered/train.kk-ru.tsv --translations $(NLLB_SHARDS) --out data/merged/train.kk-ru.tsv --db data/merged/train.kk-ru.sqlite --skip-errors

docker-train-merged:
	@test -n "$(RESUME)" || (echo "set RESUME=checkpoints/.../step-N to the server P0 checkpoint" >&2; exit 2)
	@test -s "$(RESUME)/meta.json" -a -s "$(RESUME)/model.pt" || (echo "checkpoint is missing meta.json or model.pt: $(RESUME)" >&2; exit 2)
	@test -s data/merged/train.kk-ru.tsv || (echo "merged corpus is missing: data/merged/train.kk-ru.tsv" >&2; exit 2)
	docker run --rm --gpus all --ipc=host --shm-size=16g -v "$(PWD)/data:/workspace/kk-ru/data" -v "$(PWD)/checkpoints:/workspace/kk-ru/checkpoints" -v "$(PWD)/reports:/workspace/kk-ru/reports" -v "$(PWD)/runs:/workspace/kk-ru/runs" kk-ru:cuda128 accelerate launch --multi_gpu --num_processes $(GPUS) -m kk_ru.train --config configs/p0.yaml --resume-new-data $(RESUME) --opts data.train_tsv=data/merged/train.kk-ru.tsv train.epochs=3 train.max_steps=null paths.checkpoints=checkpoints/p0-merged-3ep paths.logs=runs/p0-merged-3ep paths.reports=reports/p0-merged-3ep

docker-diffusion-train:
	@test -s "$(TRAIN_TSV)" || (echo "training corpus is missing: $(TRAIN_TSV)" >&2; exit 2)
	docker run --rm --gpus all --ipc=host --shm-size=16g -v "$(PWD)/data:/workspace/kk-ru/data" -v "$(PWD)/checkpoints:/workspace/kk-ru/checkpoints" -v "$(PWD)/reports:/workspace/kk-ru/reports" -v "$(PWD)/runs:/workspace/kk-ru/runs" kk-ru:cuda128 accelerate launch --multi_gpu --num_processes $(GPUS) -m kk_ru.diffusion_train --config configs/diffusion.yaml --opts data.train_tsv=$(TRAIN_TSV) $(if $(strip $(OPTS)),$(OPTS)) $(if $(strip $(RESUME)),--resume $(RESUME))

docker-diffusion-overfit:
	@test -s "$(TRAIN_TSV)" || (echo "training corpus is missing: $(TRAIN_TSV)" >&2; exit 2)
	docker run --rm --gpus all --ipc=host --shm-size=16g -v "$(PWD)/data:/workspace/kk-ru/data" -v "$(PWD)/checkpoints:/workspace/kk-ru/checkpoints" -v "$(PWD)/reports:/workspace/kk-ru/reports" -v "$(PWD)/runs:/workspace/kk-ru/runs" kk-ru:cuda128 accelerate launch --num_processes 1 -m kk_ru.diffusion_train --config configs/diffusion.yaml --overfit --opts data.train_tsv=$(TRAIN_TSV) train.max_steps=200 train.epochs=1 train.micro_batch_per_gpu=4 train.grad_accum=1 train.save_every=100 train.eval_every=0 paths.checkpoints=checkpoints/diffusion-smoke paths.reports=reports/diffusion-smoke paths.logs=runs/diffusion-smoke

docker-diffusion-eval:
	docker run --rm --gpus all -v "$(PWD)/data:/workspace/kk-ru/data" -v "$(PWD)/checkpoints:/workspace/kk-ru/checkpoints" -v "$(PWD)/reports:/workspace/kk-ru/reports" kk-ru:cuda128 python3 -m kk_ru.diffusion_eval --config configs/diffusion.yaml --checkpoint $(CHECKPOINT) --split devtest

docker-diffusion-benchmark:
	docker run --rm --gpus all -v "$(PWD)/data:/workspace/kk-ru/data" -v "$(PWD)/checkpoints:/workspace/kk-ru/checkpoints" -v "$(PWD)/reports:/workspace/kk-ru/reports" kk-ru:cuda128 python3 scripts/benchmark_diffusion.py --config configs/diffusion.yaml --checkpoint $(CHECKPOINT) --baseline-checkpoint $(BASELINE_CHECKPOINT) --split dev

docker-eval:
	@test -f "$(CONFIG)" || (echo "unknown model: $(MODEL) (expected small, p0, or large)" >&2; exit 2)
	docker run --rm --gpus all -v "$(PWD)/data:/workspace/kk-ru/data" -v "$(PWD)/checkpoints:/workspace/kk-ru/checkpoints" -v "$(PWD)/reports:/workspace/kk-ru/reports" kk-ru:cuda128 python3 -m kk_ru.eval --config $(CONFIG) --checkpoint $(CHECKPOINT)/best/model.pt --split devtest
