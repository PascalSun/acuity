"""Aggregate FlexBench generation reports into paper-friendly summary tables.

Usage:
    python scripts/py/summarize_generation_reports.py \
        --input data/spider/qa/flexbench \
        --output-dir docs/papers/FlexBench/results/spider
"""

from __future__ import annotations

import argparse
from pathlib import Path

from talk2metadata.core.qa.report_summary import (
    aggregate_generation_reports,
    discover_generation_reports,
    write_summary_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate generation_report.json files into JSON/CSV/Markdown summaries."
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        type=Path,
        help=(
            "Input directory, generation_report.json, or summary.json. "
            "Repeat the flag to combine multiple runs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where summary outputs will be written.",
    )
    parser.add_argument(
        "--label",
        default="generation_report_summary",
        help="Filename prefix for generated outputs.",
    )

    args = parser.parse_args()

    report_paths = discover_generation_reports(args.inputs)
    summary = aggregate_generation_reports(report_paths)
    outputs = write_summary_outputs(summary, args.output_dir, args.label)

    print(f"Reports aggregated: {summary['report_count']}")
    print(
        "Target/realized/shortfall: "
        f"{summary['target_total']}/{summary['realized_total']}/{summary['shortfall_total']}"
    )
    print(f"Overall fulfillment: {summary['overall_fulfillment_rate']:.1%}")
    print(f"Summary JSON: {outputs['summary_json']}")
    print(f"Per-DB CSV: {outputs['per_db_csv']}")
    print(f"Per-strategy CSV: {outputs['per_strategy_csv']}")
    print(f"Markdown tables: {outputs['markdown']}")


if __name__ == "__main__":
    main()
