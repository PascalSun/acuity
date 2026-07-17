"""Run FlexBench QA generation on all Spider databases for RQ1 (BERTScore comparison).

Usage:
    uv run python scripts/py/spider_batch_qa.py \\
        --spider-db-dir /path/to/spider/database \\
        --output-dir data/spider/qa/flexbench \\
        --pairs-per-db 50 \\
        --max-dbs 10  # optional: limit to N DBs for testing

Each Spider DB gets its own subdirectory: {output-dir}/{db_id}/qa_pairs.json

After completion, a combined file is written:
    {output-dir}/all_qa_pairs.json   — all generated questions with db_id
    {output-dir}/summary.json        — per-DB strategy coverage statistics

Prerequisites:
    - data/spider/schema_analysis.json  (from: talk2metadata analysis spider analyze)
    - data/spider/tables.json           (from: talk2metadata analysis spider download)
    - Spider SQLite database files      (from: https://yale-lily.github.io/spider or HF)

This script is a thin wrapper around BenchmarkRunner. For BIRD or baseline
modes, use the CLI instead:
    talk2metadata analysis bird generate-qa --db-dir /path/to/bird/database
    talk2metadata analysis spider generate-qa --mode random_sql --db-dir ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

from talk2metadata.core.qa.benchmark_runner import BenchmarkConfig, BenchmarkRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FlexBench QA generation on all Spider databases (RQ1)"
    )
    parser.add_argument(
        "--spider-db-dir",
        required=True,
        type=Path,
        help="Path to Spider database directory containing {db_id}/{db_id}.sqlite files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/spider/qa/flexbench",
        type=Path,
        help="Output directory (default: data/spider/qa/flexbench)",
    )
    parser.add_argument(
        "--pairs-per-db",
        type=int,
        default=50,
        help="QA pairs to generate per database (default: 50)",
    )
    parser.add_argument(
        "--max-dbs",
        type=int,
        default=None,
        help="Limit to first N databases (for testing)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-generate even if output already exists",
    )
    parser.add_argument(
        "--mode",
        choices=["flexbench", "random_sql", "direct_llm"],
        default="flexbench",
        help="Generation mode (default: flexbench)",
    )

    args = parser.parse_args()

    config = BenchmarkConfig(
        benchmark="spider",
        db_dir=args.spider_db_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        pairs_per_db=args.pairs_per_db,
        max_dbs=args.max_dbs,
        skip_existing=not args.no_skip_existing,
    )
    runner = BenchmarkRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
