"""Gold SQL parser for topology-guided reasoning (TGR) annotation.

Wraps SchemaAwareQueryAnalyzer to parse gold SQL into:
  - QueryClassification (is_cejsq, pattern_code, n_joins, n_where_conditions)
  - Ordered list of tables from FROM/JOIN clauses
  - WHERE column details (table, column, operator)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from talk2metadata.analysis.spider.query_analyzer import QueryClassification
from talk2metadata.analysis.spider.schema_aware_query_analyzer import (
    SchemaAwareQueryAnalyzer,
    _FROM_JOIN,
)
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)

# Regex for WHERE conditions: table.column op value  OR  column op value
_WHERE_COND = re.compile(
    r"""
    (\w+)\.(\w+)              # table.column
    \s*(=|!=|<>|>=|<=|>|<|LIKE|IN|NOT\s+IN|BETWEEN|NOT\s+LIKE)  # operator
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Simpler: just column (no table prefix) with operator
_WHERE_COND_NO_TABLE = re.compile(
    r"""
    \b(\w+)\s+(=|!=|<>|>=|<=|>|<|LIKE|IN|NOT\s+IN|BETWEEN|NOT\s+LIKE)\s+
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Extract WHERE body (between WHERE and ORDER BY/GROUP BY/LIMIT/HAVING/end)
_WHERE_BODY = re.compile(
    r"\bWHERE\b(.+?)(?:\bORDER\b|\bLIMIT\b|\bGROUP\b|\bHAVING\b|$)",
    re.IGNORECASE | re.DOTALL,
)

# Extract SELECT columns
_SELECT_COLS = re.compile(
    r"\bSELECT\b\s+(?:DISTINCT\s+)?(.+?)\bFROM\b",
    re.IGNORECASE | re.DOTALL,
)

# Table alias: FROM tbl AS T1 or FROM tbl T1 (where T1 is an alias like T1, t1, etc.)
_TABLE_ALIAS = re.compile(
    r"\b(?:FROM|JOIN)\s+`?\"?(\w+)`?\"?\s+(?:AS\s+)?(\w+)",
    re.IGNORECASE,
)


@dataclass
class WhereCondition:
    """A single WHERE condition extracted from gold SQL."""

    table: str  # original-case table name (or alias resolved)
    column: str
    operator: str  # =, >, <, LIKE, etc.


@dataclass
class ParsedGoldSQL:
    """Full parse result for a gold SQL query."""

    classification: QueryClassification
    join_tables: list[str]  # ordered tables from FROM/JOIN
    where_conditions: list[WhereCondition]
    select_columns: list[str]  # raw column expressions from SELECT


class GoldSQLParser:
    """Parses gold SQL queries for TGR annotation.

    Wraps SchemaAwareQueryAnalyzer for CEJSQ classification and adds
    extraction of join tables, WHERE columns, and SELECT columns.
    """

    def __init__(self, tables_json: list[dict]):
        self._analyzer = SchemaAwareQueryAnalyzer(tables_json)
        self._table_lower = self._analyzer._table_lower

        # Build alias-aware column→table maps per db
        self._db_columns: dict[str, dict[str, list[str]]] = {}
        for db in tables_json:
            db_id = db["db_id"]
            col_map: dict[str, list[str]] = {}
            tables = db["table_names_original"]
            cols = db["column_names_original"]
            for col_idx, (tbl_idx, col_name) in enumerate(cols):
                if tbl_idx < 0:
                    continue
                tbl_name = tables[tbl_idx]
                col_lower = col_name.lower()
                if col_lower not in col_map:
                    col_map[col_lower] = []
                col_map[col_lower].append(tbl_name)
            self._db_columns[db_id] = col_map

    def parse(self, sql: str, db_id: str, question: str = "") -> ParsedGoldSQL:
        """Parse a gold SQL query into classification + structural details."""
        # Get CEJSQ classification with precise pattern code
        example = {"query": sql, "db_id": db_id, "question": question}
        classification = self._analyzer._classify(example)

        # Extract ordered join tables
        join_tables = self._extract_join_tables_ordered(sql, db_id)

        # Extract WHERE conditions
        where_conditions = self._extract_where_conditions(sql, db_id)

        # Extract SELECT columns
        select_columns = self._extract_select_columns(sql)

        return ParsedGoldSQL(
            classification=classification,
            join_tables=join_tables,
            where_conditions=where_conditions,
            select_columns=select_columns,
        )

    def _extract_join_tables_ordered(self, sql: str, db_id: str) -> list[str]:
        """Extract tables in order of appearance in FROM/JOIN clauses."""
        lower_map = self._table_lower.get(db_id, {})
        tables = []
        seen = set()
        for m in _FROM_JOIN.finditer(sql):
            name_lower = m.group(1).lower()
            if name_lower in lower_map and name_lower not in seen:
                tables.append(lower_map[name_lower])
                seen.add(name_lower)
        return tables

    def _build_alias_map(self, sql: str, db_id: str) -> dict[str, str]:
        """Build alias → original table name map from SQL."""
        lower_map = self._table_lower.get(db_id, {})
        alias_map: dict[str, str] = {}
        for m in _TABLE_ALIAS.finditer(sql):
            tbl_raw = m.group(1).lower()
            alias_raw = m.group(2).lower()
            if tbl_raw in lower_map:
                orig = lower_map[tbl_raw]
                alias_map[alias_raw] = orig
                # Also map the table name to itself
                alias_map[tbl_raw] = orig
        # Also add direct table names
        for m in _FROM_JOIN.finditer(sql):
            tbl_raw = m.group(1).lower()
            if tbl_raw in lower_map:
                alias_map[tbl_raw] = lower_map[tbl_raw]
        return alias_map

    def _extract_where_conditions(
        self, sql: str, db_id: str
    ) -> list[WhereCondition]:
        """Extract WHERE conditions with table, column, and operator."""
        where_match = _WHERE_BODY.search(sql)
        if not where_match:
            return []

        where_body = where_match.group(1).strip()
        alias_map = self._build_alias_map(sql, db_id)
        col_map = self._db_columns.get(db_id, {})
        conditions = []

        # Try table.column pattern first
        for m in _WHERE_COND.finditer(where_body):
            tbl_ref = m.group(1).lower()
            col_name = m.group(2)
            operator = m.group(3).strip().upper()

            # Resolve alias to original table name
            resolved_table = alias_map.get(tbl_ref, tbl_ref)
            conditions.append(
                WhereCondition(table=resolved_table, column=col_name, operator=operator)
            )

        if conditions:
            return conditions

        # Fallback: column-only pattern (no table prefix)
        # Split by AND, extract each condition
        parts = re.split(r"\bAND\b", where_body, flags=re.IGNORECASE)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            m = _WHERE_COND_NO_TABLE.search(part)
            if m:
                col_name = m.group(1).lower()
                operator = m.group(2).strip().upper()
                # Try to resolve column to table
                tables_with_col = col_map.get(col_name, [])
                table = tables_with_col[0] if tables_with_col else "unknown"
                conditions.append(
                    WhereCondition(table=table, column=m.group(1), operator=operator)
                )

        return conditions

    def _extract_select_columns(self, sql: str) -> list[str]:
        """Extract column expressions from SELECT clause."""
        m = _SELECT_COLS.search(sql)
        if not m:
            return []
        cols_str = m.group(1).strip()
        if cols_str == "*":
            return ["*"]
        # Split by comma, handling parentheses
        cols = []
        depth = 0
        current = []
        for ch in cols_str:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                cols.append("".join(current).strip())
                current = []
                continue
            current.append(ch)
        if current:
            cols.append("".join(current).strip())
        return cols
