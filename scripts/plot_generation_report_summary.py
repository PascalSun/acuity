"""Render aggregated quota/shortfall summaries as standalone PDF figures.

Usage:
    PYTHONPATH=src python scripts/py/plot_generation_report_summary.py \
        --summary docs/papers/FlexBench/results/spider_quota/spider_flexbench.json \
        --output-dir docs/papers/FlexBench/figures \
        --basename fig6_quota_shortfall
"""

from __future__ import annotations

import argparse
from pathlib import Path

from talk2metadata.core.qa.report_plots import (
    compile_tex_to_pdf,
    load_generation_summary,
    write_quota_shortfall_figure,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a standalone quota/shortfall figure from an aggregated summary."
    )
    parser.add_argument(
        "--summary",
        required=True,
        type=Path,
        help="Path to aggregated summary JSON produced by summarize_generation_reports.py",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where the TeX/PDF figure will be written.",
    )
    parser.add_argument(
        "--basename",
        default="quota_shortfall_figure",
        help="Basename for the generated .tex and .pdf files.",
    )
    parser.add_argument(
        "--title",
        default="Quota Fulfillment and Shortfall Reasons",
        help="Figure title for the left panel.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Write the standalone TeX source only and skip pdflatex compilation.",
    )

    args = parser.parse_args()

    summary = load_generation_summary(args.summary)
    tex_path = args.output_dir / f"{args.basename}.tex"
    write_quota_shortfall_figure(summary, tex_path, title=args.title)
    print(f"TeX figure: {tex_path}")

    if args.no_compile:
        return

    pdf_path = compile_tex_to_pdf(tex_path)
    print(f"PDF figure: {pdf_path}")


if __name__ == "__main__":
    main()
