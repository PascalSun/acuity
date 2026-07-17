"""Launch TGR GRPO (RL) training.

Usage:
    # GRPO on top of SFT model:
    uv run python scripts/py/run_tgr_grpo.py \
        --sft-adapter models/spider_tgr \
        --train-file data/tgr_training/spider/tgr_train.jsonl \
        --db-dir data/spider/database \
        --tables-json data/spider/tables.json \
        --output-dir models/spider_tgr_grpo

    # GRPO from scratch (no SFT warmup):
    uv run python scripts/py/run_tgr_grpo.py \
        --train-file data/tgr_training/spider/tgr_train.jsonl \
        --db-dir data/spider/database \
        --tables-json data/spider/tables.json \
        --output-dir models/spider_tgr_grpo_scratch
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from talk2metadata.core.solution.paths.text2sql.finetuning.train_grpo import main

if __name__ == "__main__":
    main()
