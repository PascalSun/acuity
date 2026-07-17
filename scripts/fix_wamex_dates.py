"""
Convert .NET JSON dates (/Date(milliseconds)/) in wamex_reports.csv to YYYY-MM-DD format.

Usage:
    python scripts/py/fix_wamex_dates.py
    python scripts/py/fix_wamex_dates.py --input data/wamex/raw/wamex_reports.csv --output data/wamex/raw/wamex_reports.csv
"""

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "wamex" / "raw" / "wamex_reports.csv"


def parse_dotnet_date(value: str) -> str | None:
    """Convert '/Date(1758556800000)/' to 'YYYY-MM-DD'. Returns None for non-matching values."""
    if not isinstance(value, str):
        return value
    m = re.match(r"^/Date\((-?\d+)\)/$", value.strip())
    if not m:
        return value
    ms = int(m.group(1))
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


def main():
    parser = argparse.ArgumentParser(
        description="Fix .NET JSON dates in wamex_reports.csv"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to input CSV (default: data/wamex/raw/wamex_reports.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to output CSV (default: overwrite input file)",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    output_path: Path = args.output or input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading {input_path} ...")
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)

    if "ReportDate" not in df.columns:
        raise ValueError(
            f"Column 'ReportDate' not found. Available columns: {list(df.columns)}"
        )

    total = len(df)
    dotnet_mask = df["ReportDate"].str.match(r"^/Date\(-?\d+\)/$", na=False)
    to_convert = dotnet_mask.sum()

    print(f"Total rows: {total}")
    print(f"Rows with .NET date format: {to_convert}")

    df.loc[dotnet_mask, "ReportDate"] = df.loc[dotnet_mask, "ReportDate"].apply(
        parse_dotnet_date
    )

    # Show a few samples
    print("\nSample converted dates (first 5):")
    print(df.loc[dotnet_mask, "ReportDate"].head().to_string())

    print(f"\nWriting to {output_path} ...")
    df.to_csv(output_path, index=False)
    print("Done!")


if __name__ == "__main__":
    main()
