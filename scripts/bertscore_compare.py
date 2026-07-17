"""BERTScore comparison: FlexBench-generated questions vs human gold (RQ1).

Supports both Spider and BIRD benchmarks.

Usage:
    # Spider (default):
    uv run python scripts/py/bertscore_compare.py \\
        --generated data/spider/qa/flexbench/all_qa_pairs.json \\
        --gold-hf \\
        --output data/spider/bertscore_results.json

    # BIRD:
    uv run python scripts/py/bertscore_compare.py \\
        --generated data/bird/qa/flexbench/all_qa_pairs.json \\
        --gold-hf --benchmark bird \\
        --output data/bird/bertscore_results.json

    # Local gold file (any benchmark):
    uv run python scripts/py/bertscore_compare.py \\
        --generated data/spider/qa/flexbench/all_qa_pairs.json \\
        --gold data/spider/gold_queries.json \\
        --output data/spider/bertscore_results.json

This script is a thin wrapper around talk2metadata.core.qa.bertscore.
For programmatic use, import from there directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from talk2metadata.core.qa.bertscore import run_bertscore_comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BERTScore comparison: FlexBench generated Qs vs human gold (RQ1)"
    )
    parser.add_argument(
        "--generated",
        required=True,
        type=Path,
        help="Path to all_qa_pairs.json (from benchmark runner)",
    )

    gold_group = parser.add_mutually_exclusive_group(required=True)
    gold_group.add_argument(
        "--gold",
        type=Path,
        help="Path to local gold questions JSON file ({db_id, question} records)",
    )
    gold_group.add_argument(
        "--gold-hf",
        action="store_true",
        help="Download gold questions from HuggingFace (Spider or BIRD)",
    )

    parser.add_argument(
        "--benchmark",
        choices=["spider", "bird"],
        default="spider",
        help="Benchmark to use for HuggingFace gold (default: spider)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for results (auto-resolved from benchmark if omitted)",
    )
    parser.add_argument(
        "--model",
        default="roberta-large",
        help="BERTScore model type (default: roberta-large)",
    )

    args = parser.parse_args()

    output_path = args.output or Path(f"data/{args.benchmark}/bertscore_results.json")

    summary = run_bertscore_comparison(
        generated_path=args.generated,
        gold_path=args.gold,
        use_hf_gold=args.gold_hf,
        benchmark=args.benchmark,
        output_path=output_path,
        model_type=args.model,
    )

    # Print summary to stdout
    print(f"\nBERTScore Results ({args.benchmark}, model: {args.model})")
    print(f"  Overall mean F1:  {summary['overall_mean_f1']:.4f}")
    print(f"  Target F1 (>=0.85): {'PASS' if summary['meets_target'] else 'FAIL'}")
    print(f"  Common DBs:       {summary['total_dbs']}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
