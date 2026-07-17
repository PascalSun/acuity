"""LoRA fine-tuning script for TGR (Topology-Guided Reasoning) experiments.

Trains Qwen2.5-Coder-7B (or similar) on Spider/BIRD data with optional
topology reasoning chains. Uses QLoRA (4-bit base + LoRA) for efficiency.

Usage:
    python -m talk2metadata.core.solution.paths.text2sql.finetuning.train \
        --train-file data/tgr_training/tgr_train.jsonl \
        --output-dir models/tgr_qwen7b
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


@dataclass
class TGRTrainingConfig:
    """Training configuration."""

    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    train_file: str = "data/tgr_training/tgr_train.jsonl"
    val_file: str | None = None
    output_dir: str = "models/tgr_qwen7b"

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # Training
    num_epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 2048
    bf16: bool = True
    load_in_4bit: bool = True

    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def tokenize_chat(example: dict, tokenizer, max_length: int) -> dict:
    """Tokenize a chat-format example using the model's chat template."""
    messages = example["messages"]

    # Apply chat template to get full text
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # Tokenize
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    # For causal LM: labels = input_ids (shifted internally by the model)
    tokenized["labels"] = tokenized["input_ids"].copy()

    # Mask the prompt tokens (system + user) so loss is only on assistant response
    # Find where the assistant response starts
    prompt_messages = messages[:-1]  # system + user
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = tokenizer(
        prompt_text, truncation=True, max_length=max_length, padding=False
    )
    prompt_len = len(prompt_tokens["input_ids"])

    # Set prompt tokens to -100 (ignored in loss)
    tokenized["labels"][:prompt_len] = [-100] * prompt_len

    return tokenized


def train(config: TGRTrainingConfig) -> None:
    """Run LoRA fine-tuning."""
    print(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config for QLoRA
    bnb_config = None
    if config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading model: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
    )

    # Prepare for k-bit training
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    print(f"Loading training data: {config.train_file}")
    train_data = load_jsonl(config.train_file)
    train_dataset = Dataset.from_list(train_data)

    val_dataset = None
    if config.val_file and Path(config.val_file).exists():
        print(f"Loading validation data: {config.val_file}")
        val_data = load_jsonl(config.val_file)
        val_dataset = Dataset.from_list(val_data)

    # Tokenize
    def tokenize_fn(example):
        return tokenize_chat(example, tokenizer, config.max_seq_length)

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        tokenize_fn, remove_columns=train_dataset.column_names
    )
    if val_dataset is not None:
        val_dataset = val_dataset.map(
            tokenize_fn, remove_columns=val_dataset.column_names
        )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=config.eval_steps if val_dataset else None,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if config.load_in_4bit else "adamw_torch",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    # Save adapter
    print(f"Saving adapter to {config.output_dir}")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="TGR LoRA Fine-Tuning")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    config = TGRTrainingConfig(
        model_name=args.model,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_len,
        load_in_4bit=not args.no_4bit,
    )

    train(config)


if __name__ == "__main__":
    main()
