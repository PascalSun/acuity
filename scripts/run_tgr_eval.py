"""Run TGR evaluation on Spider-dev or BIRD-dev.

Wrapper script for the evaluation harness.

Usage:
    # Evaluate TGR model:
    uv run python scripts/py/run_tgr_eval.py \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --adapter models/spider_tgr \
        --db-dir data/spider/database \
        --tables-json data/spider/tables.json \
        --dev-file data/spider/dev.json \
        --output data/tgr_eval/spider_tgr_results.json

    # Evaluate baseline model:
    uv run python scripts/py/run_tgr_eval.py \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --adapter models/spider_baseline \
        --db-dir data/spider/database \
        --tables-json data/spider/tables.json \
        --dev-file data/spider/dev.json \
        --output data/tgr_eval/spider_baseline_results.json \
        --no-tgr

    # Quick test (10 examples):
    uv run python scripts/py/run_tgr_eval.py \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --adapter models/spider_tgr \
        --db-dir data/spider/database \
        --tables-json data/spider/tables.json \
        --dev-file data/spider/dev.json \
        --output data/tgr_eval/test_results.json \
        --max-examples 10
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from talk2metadata.core.solution.paths.text2sql.finetuning.eval_harness import main

if __name__ == "__main__":
    main()
