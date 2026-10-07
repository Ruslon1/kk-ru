#!/usr/bin/env python3
"""LoRA fine-tuning for Qwen3 on the local KK-RU TSV corpus."""

from __future__ import annotations

import argparse
from array import array
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


SYSTEM_PROMPT = (
    "Translate Kazakh to Russian. Return only the Russian translation. "
    "Do not explain or add commentary."
)


def token_ids(tokenizer, messages, *, add_generation_prompt: bool) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    if hasattr(encoded, "input_ids"):
        return list(encoded.input_ids)
    return list(encoded)


def read_pairs(path: str, limit: int = 0) -> list[tuple[str, str]]:
    pairs = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            source, separator, target = line.rstrip("\n").partition("\t")
            if separator and source and target:
                pairs.append((source, target))
                if limit and len(pairs) >= limit:
                    break
    return pairs


class TranslationDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int, limit: int = 0):
        self.path = str(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.offsets = array("Q")
        with Path(path).open("rb") as file:
            while True:
                offset = file.tell()
                line = file.readline()
                if not line:
                    break
                if b"\t" in line and line.rstrip(b"\n").split(b"\t", 1)[0]:
                    self.offsets.append(offset)
                    if limit and len(self.offsets) >= limit:
                        break
        self._file = None

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        if self._file is None:
            self._file = open(self.path, "rb")
        self._file.seek(self.offsets[index])
        source, separator, target = self._file.readline().decode("utf-8").rstrip("\n").partition("\t")
        if not separator or not source or not target:
            raise ValueError(f"invalid training row at byte offset {self.offsets[index]}")
        prompt_ids = token_ids(
            self.tokenizer,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": source},
            ],
            add_generation_prompt=True,
        )
        full_ids = token_ids(
            self.tokenizer,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": source},
                {"role": "assistant", "content": target},
            ],
            add_generation_prompt=False,
        )[: self.max_length]
        labels = [-100] * min(len(prompt_ids), len(full_ids))
        labels.extend(full_ids[len(labels) :])
        labels = labels[: len(full_ids)]
        if not any(label != -100 for label in labels):
            return None
        return {"input_ids": full_ids, "labels": labels}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __del__(self):
        if getattr(self, "_file", None) is not None:
            self._file.close()


@dataclass
class Collator:
    pad_id: int

    def __call__(self, rows):
        rows = [row for row in rows if row is not None]
        if not rows:
            raise ValueError("batch contains no examples with target tokens")
        length = max(len(row["input_ids"]) for row in rows)
        input_ids = torch.full((len(rows), length), self.pad_id, dtype=torch.long)
        labels = torch.full((len(rows), length), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), length), dtype=torch.long)
        for index, row in enumerate(rows):
            size = len(row["input_ids"])
            input_ids[index, :size] = torch.tensor(row["input_ids"])
            labels[index, :size] = torch.tensor(row["labels"])
            attention_mask[index, :size] = 1
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train", default="data/filtered/train.kk-ru.tsv")
    parser.add_argument("--eval", default="data/eval/flores_plus/dev.kk-ru.tsv")
    parser.add_argument("--output", default="checkpoints/qwen3-0.6b-lora")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    train_dataset = TranslationDataset(
        args.train, tokenizer, args.max_length, args.limit
    )
    eval_dataset = TranslationDataset(
        args.eval, tokenizer, args.max_length, args.eval_limit
    )
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to=["tensorboard"],
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        run_name="qwen3-0.6b-lora",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Collator(tokenizer.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
