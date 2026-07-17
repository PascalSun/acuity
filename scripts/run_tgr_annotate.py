"""Annotate Spider/BIRD training data with topology-guided reasoning chains.

Reads Spider/BIRD training examples and tables.json, produces JSONL files
for both baseline (SQL only) and TGR (chain + SQL) training formats.

Usage:
    uv run python scripts/py/run_tgr_annotate.py \
        --tables-json data/spider/tables.json \
        --train-json data/spider/train_spider.json \
        --output-dir data/tgr_training/spider

    # For BIRD:
    uv run python scripts/py/run_tgr_annotate.py \
        --tables-json data/bird/tables.json \
        --train-json data/bird/train.json \
        --output-dir data/tgr_training/bird

    # Both combined:
    uv run python scripts/py/run_tgr_annotate.py \
        --tables-json data/spider/tables.json data/bird/tables.json \
        --train-json data/spider/train_spider.json data/bird/train.json \
        --output-dir data/tgr_training/combined
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from talk2metadata.core.qa.tgr_data_builder import TGRDataBuilder


def main():
    parser = argparse.ArgumentParser(
        description="Annotate Spider/BIRD data with TGR reasoning chains"
    )
    parser.add_argument(
        "--tables-json",
        nargs="+",
        required=True,
        help="Path(s) to tables.json file(s)",
    )
    parser.add_argument(
        "--train-json",
        nargs="+",
        required=True,
        help="Path(s) to training data JSON file(s) (Spider/BIRD format)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Fraction of databases for validation split (default: 0.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit examples (for testing)",
    )
    args = parser.parse_args()

    # Load all tables.json files
    all_tables = []
    for path in args.tables_json:
        print(f"Loading tables.json: {path}")
        with open(path) as f:
            tables = json.load(f)
        all_tables.extend(tables)
        print(f"  Loaded {len(tables)} databases")

    # Load all training data
    all_examples = []
    for path in args.train_json:
        print(f"Loading training data: {path}")
        with open(path) as f:
            examples = json.load(f)
        all_examples.extend(examples)
        print(f"  Loaded {len(examples)} examples")

    if args.max_examples:
        all_examples = all_examples[: args.max_examples]
        print(f"Limited to {len(all_examples)} examples")

    # Build annotated training data
    print(f"\nAnnotating {len(all_examples)} examples from {len(all_tables)} databases...")
    builder = TGRDataBuilder(all_tables)
    stats = builder.build(
        examples=all_examples,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"  TGR Annotation Summary")
    print(f"{'='*60}")
    print(f"  Total examples:  {stats.total}")
    print(f"  CEJSQ:           {stats.cejsq} ({stats.cejsq/stats.total*100:.1f}%)")
    print(f"  Non-CEJSQ:       {stats.non_cejsq} ({stats.non_cejsq/stats.total*100:.1f}%)")
    print(f"  Parse errors:    {stats.parse_errors}")

    if stats.pattern_distribution:
        print(f"\n  Pattern distribution (CEJSQ):")
        for pat, count in sorted(
            stats.pattern_distribution.items(), key=lambda x: -x[1]
        ):
            pct = count / stats.cejsq * 100 if stats.cejsq else 0
            print(f"    {pat:10s}: {count:4d} ({pct:.1f}%)")

    if stats.archetype_distribution:
        print(f"\n  Archetype distribution:")
        for arch, count in sorted(stats.archetype_distribution.items()):
            pct = count / stats.total * 100 if stats.total else 0
            print(f"    {arch:12s}: {count:4d} ({pct:.1f}%)")

    if stats.exclusion_reasons:
        print(f"\n  Non-CEJSQ exclusion reasons:")
        for reason, count in sorted(
            stats.exclusion_reasons.items(), key=lambda x: -x[1]
        ):
            print(f"    {reason:20s}: {count:4d}")

    print(f"\n  Output directory: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
