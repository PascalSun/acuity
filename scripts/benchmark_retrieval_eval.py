"""Lightweight RQ3 retrieval eval on Spider/BIRD FlexBench QA pairs.

Thin wrapper around talk2metadata.core.qa.retrieval_eval.

Usage:
    uv run python scripts/py/benchmark_retrieval_eval.py \
        --qa-file data/spider/qa/flexbench/all_qa_pairs.json \
        --db-dir data/spider/data/spider/hf_download/database \
        --output data/spider/retrieval_eval.json

    uv run python scripts/py/benchmark_retrieval_eval.py \
        --qa-file data/bird/qa/flexbench/all_qa_pairs.json \
        --db-dir data/bird/hf_download/train/train_databases \
        --output data/bird/retrieval_eval.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from talk2metadata.core.qa.retrieval_eval import run_eval


def main():
    parser = argparse.ArgumentParser(
        description="RQ3 retrieval eval on Spider/BIRD FlexBench QA pairs"
    )
    parser.add_argument("--qa-file", required=True, type=Path)
    parser.add_argument("--db-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()

    run_eval(args.qa_file, args.db_dir, args.output, args.max_pairs)


if __name__ == "__main__":
    main()
