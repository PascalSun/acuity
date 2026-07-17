"""Schema data types and models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ColumnType(str, Enum):
    """Semantic column types for type-aware QA generation.

    Each type drives operator selection, value picking, and question phrasing.
    """

    NUMERIC = "numeric"  # Continuous numbers (revenue, counts)
    YEAR = "year"  # Integer years (1800-2100)
    DATE = "date"  # ISO date strings (YYYY-MM-DD)
    ENUM = "enum"  # Small fixed set of categorical values
    BOOLEAN = "boolean"  # t/f, true/false, yes/no
    COMMA_SEPARATED = "comma_separated"  # Comma-delimited multi-value strings
    LABEL = "label"  # Short high-cardinality strings (names, titles, cities)
    TEXT = "text"  # Free-form long text (not filterable)
    IDENTIFIER = "identifier"  # IDs, codes, internal references (not filterable)


# Column types that can be used in WHERE filters
FILTERABLE_COLUMN_TYPES = {
    ColumnType.NUMERIC,
    ColumnType.YEAR,
    ColumnType.DATE,
    ColumnType.ENUM,
    ColumnType.BOOLEAN,
    ColumnType.COMMA_SEPARATED,
    ColumnType.LABEL,
}


@dataclass
class ForeignKey:
    """Foreign key relationship."""

    child_table: str
    child_column: str
    parent_table: str
    parent_column: str
    coverage: float  # 0.0-1.0 (percentage of child values found in parent)

    def __repr__(self) -> str:
        return (
            f"FK({self.child_table}.{self.child_column} -> "
            f"{self.parent_table}.{self.parent_column}, coverage={self.coverage:.2f})"
        )


@dataclass
class TableMetadata:
    """Metadata for a single table."""

    name: str
    columns: Dict[str, str]  # column_name -> dtype
    primary_key: Optional[str] = None
    row_count: int = 0
    sample_values: Dict[str, List[str]] = field(default_factory=dict)
    description: Optional[str] = None  # Human-readable description of the table
    column_descriptions: Dict[str, str] = field(
        default_factory=dict
    )  # column_name -> description
    self_ref_depth: Optional[int] = None
    """Max traversal depth for self-referential FK on this table during QA chain generation.
    Only meaningful when the table has a self-referential FK (e.g. a hierarchy like Theme).
    Defaults to None (falls back to the module-level _MAX_SELF_REF_DEPTH = 3).
    """
    non_filterable_columns: List[str] = field(default_factory=list)
    """Columns that should NOT be used in WHERE filters during QA generation.

    Use this to mark non-semantic columns (internal IDs, codes, timestamps, etc.)
    that would produce awkward questions. FK columns and primary keys are always
    excluded automatically; this list is for additional columns beyond those.

    Configurable in schema JSON, e.g.:
        "non_filterable_columns": ["created_at", "updated_at", "legacy_code"]
    """
    column_types: Dict[str, str] = field(default_factory=dict)
    """User-specified semantic column types (column_name -> ColumnType value string).

    When provided, these override the auto-inferred types during QA generation.
    Unspecified columns are inferred automatically from dtype and data statistics.

    Configurable in schema JSON, e.g.:
        "column_types": {"ReportDate": "date", "Confidentiality": "enum"}
    """

    def __repr__(self) -> str:
        return f"TableMetadata({self.name}, columns={len(self.columns)}, rows={self.row_count})"
