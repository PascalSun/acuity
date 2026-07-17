"""Automatic column type inference for type-aware QA generation.

Infers semantic column types (date, enum, boolean, etc.) from pandas dtype,
column name, and data statistics. User-provided overrides in schema JSON
take priority over inference.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import pandas as pd

from talk2metadata.core.schema.types import (
    FILTERABLE_COLUMN_TYPES,
    ColumnType,
    TableMetadata,
)
from talk2metadata.utils.logging import get_logger

# Re-export for convenience
__all__ = [
    "infer_column_type",
    "infer_all_column_types",
    "ColumnType",
    "FILTERABLE_COLUMN_TYPES",
]

logger = get_logger(__name__)

# Boolean-like string values (lowercase)
_BOOLEAN_VALUES = frozenset(
    {"t", "f", "true", "false", "yes", "no", "y", "n", "0", "1"}
)

# Date pattern: YYYY-MM-DD (with optional time)
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}")

# Thresholds
_ENUM_MAX_CARDINALITY = 20  # Max distinct values to consider as enum
_ENUM_MAX_AVG_LENGTH = 50  # Max average string length for enum
_LABEL_MAX_AVG_LENGTH = 50  # Max average string length for label (short strings)
_COMMA_SEP_MIN_RATIO = 0.5  # Min fraction of values containing commas
_COMMA_SEP_MAX_TOKEN_LEN = 60  # Max average token length
_DATE_PARSE_THRESHOLD = 0.9  # Min fraction of values that must parse as dates
_YEAR_MIN = 1800
_YEAR_MAX = 2100


def infer_column_type(
    col_name: str,
    dtype: str,
    series: Optional[pd.Series] = None,
    sample_values: Optional[List[str]] = None,
) -> ColumnType:
    """Infer the semantic type of a single column.

    Args:
        col_name: Column name
        dtype: Pandas dtype string (e.g., "int64", "object", "float64", "bool")
        series: Full pandas Series for the column (preferred for accurate inference)
        sample_values: Sample values from schema (fallback when series unavailable)

    Returns:
        Inferred ColumnType
    """
    col_lower = col_name.lower()

    # Bool dtype
    if dtype == "bool":
        return ColumnType.BOOLEAN

    # Numeric types
    if dtype in ("int64", "float64"):
        return _infer_numeric_type(col_lower, dtype, series)

    # String/object types
    if dtype == "object":
        return _infer_string_type(col_lower, series, sample_values)

    # Unknown dtype fallback
    return ColumnType.TEXT


def _infer_numeric_type(
    col_lower: str, dtype: str, series: Optional[pd.Series]
) -> ColumnType:
    """Infer type for numeric columns."""
    if dtype == "int64" and series is not None:
        non_null = series.dropna()
        if len(non_null) > 0:
            # Year detection: all values in plausible year range + name hint
            min_val, max_val = non_null.min(), non_null.max()
            if _YEAR_MIN <= min_val and max_val <= _YEAR_MAX:
                name_hint = any(kw in col_lower for kw in ("year", "yr", "date"))
                if name_hint:
                    return ColumnType.YEAR
    return ColumnType.NUMERIC


def _infer_string_type(
    col_lower: str,
    series: Optional[pd.Series],
    sample_values: Optional[List[str]],
) -> ColumnType:
    """Infer type for string/object columns."""
    # Get values to analyze
    if series is not None:
        non_null = series.dropna()
        if len(non_null) == 0:
            return ColumnType.TEXT
    elif sample_values:
        non_null = pd.Series([v for v in sample_values if v is not None])
        if len(non_null) == 0:
            return ColumnType.TEXT
    else:
        return ColumnType.TEXT

    # Convert to string for analysis
    str_values = non_null.astype(str)

    # 1. Boolean check — small set of boolean-like values
    unique_lower = set(str_values.str.lower().unique())
    if unique_lower.issubset(_BOOLEAN_VALUES) and len(unique_lower) <= 2:
        return ColumnType.BOOLEAN

    # 2. Date check — most values match YYYY-MM-DD
    if _is_date_like(str_values):
        return ColumnType.DATE

    # 3. Comma-separated check
    if _is_comma_separated(str_values):
        return ColumnType.COMMA_SEPARATED

    # 4. Enum check — low cardinality, short strings
    nunique = str_values.nunique()
    avg_len = str_values.str.len().mean()
    if nunique <= _ENUM_MAX_CARDINALITY and avg_len <= _ENUM_MAX_AVG_LENGTH:
        return ColumnType.ENUM

    # 5. Label check — high-cardinality but SHORT strings (names, titles,
    # cities). These are the most natural filters ("name = 'Alice'") and were
    # previously dropped as TEXT, starving many hub tables of any filterable
    # column (the dominant cause of zero-yield DBs at full scale).
    if avg_len <= _LABEL_MAX_AVG_LENGTH:
        # Exclude multi-line values (free text) from labels
        has_newline = str_values.str.contains("\n", na=False, regex=False)
        if not has_newline.any():
            return ColumnType.LABEL

    # 6. Fallback to text (not filterable)
    return ColumnType.TEXT


def _is_date_like(str_values: pd.Series) -> bool:
    """Check if values match date patterns."""
    matches = str_values.str.match(_DATE_PATTERN, na=False)
    if len(str_values) == 0:
        return False
    return matches.mean() >= _DATE_PARSE_THRESHOLD


def _is_comma_separated(str_values: pd.Series) -> bool:
    """Check if values are comma-separated lists."""
    has_comma = str_values.str.contains(",", na=False)
    comma_ratio = has_comma.mean()
    if comma_ratio < _COMMA_SEP_MIN_RATIO:
        return False
    # Check that individual tokens are short (not natural language with commas)
    sample = str_values.head(50)
    avg_token_len = sample.apply(
        lambda v: pd.Series([len(t.strip()) for t in str(v).split(",")]).mean()
    ).mean()
    return avg_token_len <= _COMMA_SEP_MAX_TOKEN_LEN


def infer_all_column_types(
    table_meta: TableMetadata,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, ColumnType]:
    """Infer semantic types for all columns in a table.

    User-provided overrides in table_meta.column_types take priority.

    Args:
        table_meta: Table metadata with columns, dtypes, and optional overrides
        df: DataFrame for the table (optional, enables more accurate inference)

    Returns:
        Dict mapping lowercase column name -> ColumnType
    """
    result: Dict[str, ColumnType] = {}
    user_overrides = {k.lower(): v for k, v in table_meta.column_types.items()}

    for col_name, dtype in table_meta.columns.items():
        col_lower = col_name.lower()

        # User override takes priority
        if col_lower in user_overrides:
            try:
                result[col_lower] = ColumnType(user_overrides[col_lower])
                continue
            except ValueError:
                logger.warning(
                    f"Invalid column_type override '{user_overrides[col_lower]}' "
                    f"for {table_meta.name}.{col_name}, will auto-infer"
                )

        # Get series from DataFrame if available
        series = None
        if df is not None and col_lower in df.columns:
            series = df[col_lower]

        # Get sample values as fallback
        samples = table_meta.sample_values.get(col_name)

        result[col_lower] = infer_column_type(col_lower, dtype, series, samples)

    return result
