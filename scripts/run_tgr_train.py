"""Launch TGR LoRA fine-tuning.

Wrapper script for the training module.

Usage:
    # Train TGR model (with reasoning chains):
    uv run python scripts/py/run_tgr_train.py \
        --train-file data/tgr_training/spider/tgr_train.jsonl \
        --val-file data/tgr_training/spider/tgr_val.jsonl \
        --output-dir models/spider_tgr

    # Train baseline model (SQL only):
    uv run python scripts/py/run_tgr_train.py \
        --train-file data/tgr_training/spider/baseline_train.jsonl \
        --val-file data/tgr_training/spider/baseline_val.jsonl \
        --output-dir models/spider_baseline
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from talk2metadata.core.solution.paths.text2sql.finetuning.train import main

if __name__ == "__main__":
    main()
