"""GRPO (Group Relative Policy Optimization) training for TGR.

Applies RL on top of an SFT-trained model using execution correctness
as the primary reward signal, with a topology consistency bonus.

Follows the Arctic-Text2SQL-R1 approach: generate multiple completions
per prompt, reward based on SQL execution, and update with GRPO.

Usage:
    python -m talk2metadata.core.solution.paths.text2sql.finetuning.train_grpo \
        --sft-adapter models/spider_tgr \
        --train-file data/tgr_training/spider/tgr_train.jsonl \
        --db-dir data/spider/database \
        --tables-json data/spider/tables.json \
        --output-dir models/spider_tgr_grpo
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer


@dataclass
class GRPOTrainingConfig:
    """GRPO training configuration."""

    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    sft_adapter: str | None = None
    train_file: str = "data/tgr_training/spider/tgr_train.jsonl"
    db_dir: str = "data/spider/database"
    tables_json: str = "data/spider/tables.json"
    output_dir: str = "models/spider_tgr_grpo"

    # GRPO
    num_generations: int = 4  # completions per prompt (G in GRPO)
    max_new_tokens: int = 1024
    temperature: float = 0.7

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
    num_epochs: int = 1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6  # lower LR for RL
    warmup_ratio: float = 0.05
    max_completion_length: int = 1024
    bf16: bool = True
    load_in_4bit: bool = True

    # Reward weights
    reward_correct: float = 1.0
    reward_executable: float = 0.3
    reward_syntax_error: float = -0.5
    reward_topology_bonus: float = 0.2

    # Logging
    logging_steps: int = 5
    save_steps: int = 200


# SQL extraction patterns
_SQL_TAG = re.compile(r"<sql>(.*?)</sql>", re.DOTALL | re.IGNORECASE)
_SELECT_STMT = re.compile(r"(SELECT\b.*?)(?:;|$)", re.DOTALL | re.IGNORECASE)
_TOPOLOGY_TAG = re.compile(r"\[TOPOLOGY\].*?archetype=(\w+)", re.IGNORECASE)
_PATTERN_TAG = re.compile(r"\[PATTERN\]\s*(\w+):", re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Extract SQL from model output."""
    m = _SQL_TAG.search(text)
    if m:
        return m.group(1).strip()
    m = _SELECT_STMT.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def extract_chain_info(text: str) -> dict:
    """Extract topology and pattern from reasoning chain."""
    info = {}
    m = _TOPOLOGY_TAG.search(text)
    if m:
        info["archetype"] = m.group(1).lower()
    m = _PATTERN_TAG.search(text)
    if m:
        info["pattern"] = m.group(1)
    return info


def load_prompts(path: str) -> list[dict]:
    """Load JSONL and extract prompts (system + user messages only)."""
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            messages = ex["messages"]
            # Keep only system + user (prompt), store gold SQL for reward
            prompt_messages = [m for m in messages if m["role"] != "assistant"]
            gold_content = next(
                (m["content"] for m in messages if m["role"] == "assistant"), ""
            )
            # Extract gold SQL from the assistant content
            gold_sql = extract_sql(gold_content)
            # Extract db_id from the schema text
            db_id = ""
            for m in messages:
                if m["role"] == "user":
                    # Try to find db_id hint, or we'll need it from elsewhere
                    break

            prompts.append({
                "prompt": prompt_messages,
                "gold_sql": gold_sql,
                "gold_content": gold_content,
            })
    return prompts


class SQLRewardFunction:
    """Reward function based on SQL execution correctness."""

    __name__ = "sql_execution_reward"

    def __init__(
        self,
        db_dir: str,
        tables_json_path: str,
        train_file: str,
        config: GRPOTrainingConfig,
    ):
        self.db_dir = Path(db_dir)
        self.config = config

        # Build gold SQL lookup from training data
        self._gold_sqls: list[str] = []
        self._db_ids: list[str] = []
        self._build_gold_lookup(train_file, tables_json_path)

    def _build_gold_lookup(self, train_file: str, tables_json_path: str):
        """Build lookup of gold SQLs and db_ids from training data."""
        # Load tables.json to get db schemas
        with open(tables_json_path) as f:
            tables = json.load(f)
        self._db_schemas = {db["db_id"]: db for db in tables}

        # Parse training data to extract db_id from schema text
        with open(train_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                messages = ex["messages"]

                # Get gold SQL
                gold_content = next(
                    (m["content"] for m in messages if m["role"] == "assistant"), ""
                )
                gold_sql = extract_sql(gold_content)
                self._gold_sqls.append(gold_sql)

                # Infer db_id from schema text by matching table names
                user_msg = next(
                    (m["content"] for m in messages if m["role"] == "user"), ""
                )
                db_id = self._infer_db_id(user_msg)
                self._db_ids.append(db_id)

    def _infer_db_id(self, schema_text: str) -> str:
        """Infer db_id from schema text by matching table structure."""
        # Extract table names from "## TableName" headers
        table_names = re.findall(r"^## (\w+)", schema_text, re.MULTILINE)
        table_set = set(t.lower() for t in table_names if t not in ("Foreign", "Database"))

        if not table_set:
            return ""

        # Match against known schemas
        best_match = ""
        best_score = 0
        for db_id, db in self._db_schemas.items():
            db_tables = set(t.lower() for t in db.get("table_names_original", []))
            if not db_tables:
                continue
            overlap = len(table_set & db_tables)
            if overlap > best_score:
                best_score = overlap
                best_match = db_id

        return best_match

    def __call__(self, completions: list[str], prompts: list[str], **kwargs) -> list[float]:
        """Compute rewards for a batch of completions.

        Args:
            completions: List of model completions (one per generation).
            prompts: List of corresponding prompts.

        Returns:
            List of reward scores.
        """
        rewards = []

        # Get the indices from kwargs if available, otherwise use prompt matching
        indices = kwargs.get("indices", list(range(len(completions))))

        for i, completion in enumerate(completions):
            idx = indices[i] if i < len(indices) else i
            reward = self._score_one(completion, idx)
            rewards.append(reward)

        return rewards

    def _score_one(self, completion: str, example_idx: int) -> float:
        """Score a single completion."""
        predicted_sql = extract_sql(completion)

        if not predicted_sql or not predicted_sql.upper().startswith("SELECT"):
            return self.config.reward_syntax_error

        # Get gold SQL and db_id
        if example_idx >= len(self._gold_sqls):
            return 0.0

        gold_sql = self._gold_sqls[example_idx]
        db_id = self._db_ids[example_idx]

        if not db_id:
            return 0.0

        # Try to execute
        db_path = self.db_dir / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            return 0.0

        try:
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA journal_mode=WAL")

            # Execute predicted SQL
            try:
                pred_results = conn.execute(predicted_sql).fetchall()
            except Exception:
                conn.close()
                return self.config.reward_syntax_error

            # Execute gold SQL
            try:
                gold_results = conn.execute(gold_sql).fetchall()
            except Exception:
                conn.close()
                return self.config.reward_executable  # pred ran, gold failed (unusual)

            conn.close()

            # Compare results (set comparison for order-insensitive matching)
            pred_set = set(str(r) for r in pred_results)
            gold_set = set(str(r) for r in gold_results)

            if pred_set == gold_set:
                reward = self.config.reward_correct
            else:
                reward = self.config.reward_executable

        except Exception:
            return 0.0

        # Topology consistency bonus
        chain_info = extract_chain_info(completion)
        gold_chain_info = extract_chain_info(
            self._gold_sqls[example_idx]
            if example_idx < len(self._gold_sqls) else ""
        )
        # Check if predicted chain mentions same pattern as gold
        if chain_info.get("pattern") and chain_info.get("pattern") == gold_chain_info.get("pattern"):
            reward += self.config.reward_topology_bonus

        return reward


def build_dataset(train_file: str, tokenizer) -> Dataset:
    """Build dataset with prompt text for GRPO."""
    records = []
    with open(train_file) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            messages = ex["messages"]

            # Build prompt (system + user only)
            prompt_messages = [m for m in messages if m["role"] != "assistant"]

            # Apply chat template for prompt
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )

            records.append({
                "prompt": prompt_text,
                "index": i,
            })

    return Dataset.from_list(records)


def train(config: GRPOTrainingConfig) -> None:
    """Run GRPO training."""
    print(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for generation

    # Quantization config
    bnb_config = None
    if config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # Load model
    print(f"Loading model: {config.model_name}")
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if config.bf16 else torch.float16,
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    # If SFT adapter exists, merge it first
    if config.sft_adapter and Path(config.sft_adapter).exists():
        print(f"Loading SFT adapter: {config.sft_adapter}")
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name, **model_kwargs
        )
        model = PeftModel.from_pretrained(base_model, config.sft_adapter)
        model = model.merge_and_unload()
        print("  Merged SFT adapter into base model")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, **model_kwargs
        )

    # Build dataset
    print(f"Building dataset from {config.train_file}")
    dataset = build_dataset(config.train_file, tokenizer)
    print(f"  {len(dataset)} training prompts")

    # Build reward function
    print("Setting up reward function...")
    reward_fn = SQLRewardFunction(
        db_dir=config.db_dir,
        tables_json_path=config.tables_json,
        train_file=config.train_file,
        config=config,
    )

    # LoRA config for GRPO (new trainable adapter on top of merged model)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
    )

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        report_to="none",
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        gradient_checkpointing=True,
    )

    # GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("Starting GRPO training...")
    print(f"  Reward: correct={config.reward_correct}, "
          f"executable={config.reward_executable}, "
          f"syntax_error={config.reward_syntax_error}, "
          f"topology_bonus={config.reward_topology_bonus}")
    trainer.train()

    # Save
    print(f"Saving to {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    print("GRPO training complete!")


def main():
    parser = argparse.ArgumentParser(description="TGR GRPO Training")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--sft-adapter", default=None,
                        help="Path to SFT adapter to start from")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--db-dir", required=True,
                        help="Path to Spider database directory")
    parser.add_argument("--tables-json", required=True,
                        help="Path to tables.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    config = GRPOTrainingConfig(
        model_name=args.model,
        sft_adapter=args.sft_adapter,
        train_file=args.train_file,
        db_dir=args.db_dir,
        tables_json=args.tables_json,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_generations=args.num_generations,
        lora_r=args.lora_r,
        load_in_4bit=not args.no_4bit,
    )

    train(config)


if __name__ == "__main__":
    main()
