"""Schema-aware query analyzer that uses FK graph to distinguish path vs intersection.

Problem with SpiderQueryAnalyzer._infer_pattern():
  It only knows the number of JOINs, not the *shape* of the join subgraph.
  So 2 JOINs could be:
    - 2p: D1 → D2 → T  (linear chain, management → department + head → management)
    - 2i: D1 → T ← D2  (two tables both FK into the same hub)
  This module resolves the ambiguity using the schema's FK edges.

Algorithm:
  1. Extract tables referenced in SQL FROM/JOIN clauses.
  2. Build the induced FK subgraph for those tables.
  3. Compute in-degree of each table (# distinct child tables pointing to it).
  4. If max in-degree >= 2  →  intersection (ki where k = max in-degree)
     Else                   →  path        (kp where k = n_joins, capped at 3)
"""

from __future__ import annotations

import re
from collections import defaultdict

from talk2metadata.analysis.spider.query_analyzer import (
    QueryClassification,
    SpiderQueryAnalyzer,
)
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)

# Regex: extract table name right after FROM or JOIN keyword
# Handles: FROM tbl, FROM `tbl`, FROM "tbl", JOIN tbl AS T1, etc.
_FROM_JOIN = re.compile(r'\b(?:FROM|JOIN)\s+`?"?(\w+)`?"?', re.IGNORECASE)


class SchemaAwareQueryAnalyzer(SpiderQueryAnalyzer):
    """Extends SpiderQueryAnalyzer with schema FK info for precise path/intersection detection."""

    def __init__(self, schemas: list[dict]):
        """
        Args:
            schemas: List of tables.json-format dicts (db_id, table_names_original,
                     column_names_original, foreign_keys).
        """
        # Per-db: list of (child_table_original, parent_table_original)
        self._fk_edges: dict[str, list[tuple[str, str]]] = {}

        # Per-db: lowercase table name → original table name (for case-insensitive lookup)
        self._table_lower: dict[str, dict[str, str]] = {}

        for db in schemas:
            db_id = db["db_id"]
            tables = db["table_names_original"]
            cols = db["column_names_original"]

            self._table_lower[db_id] = {t.lower(): t for t in tables}

            edges = []
            for child_idx, parent_idx in db.get("foreign_keys", []):
                ct_idx = cols[child_idx][0]
                pt_idx = cols[parent_idx][0]
                if ct_idx < 0 or pt_idx < 0:
                    continue
                ct = tables[ct_idx]
                pt = tables[pt_idx]
                if ct != pt:
                    edges.append((ct, pt))
            self._fk_edges[db_id] = edges

        logger.info(f"SchemaAwareQueryAnalyzer loaded {len(schemas)} DB schemas")

    # ------------------------------------------------------------------
    # Override: pattern inference
    # ------------------------------------------------------------------

    def _infer_pattern(self, c: QueryClassification) -> str:
        n_conds = c.n_where_conditions
        if n_conds <= 2:
            diff = "E"
        elif n_conds <= 5:
            diff = "M"
        else:
            diff = "H"

        pat = self._classify_join_shape(c.query, c.db_id, c.n_joins)
        return f"{pat}{diff}"

    # ------------------------------------------------------------------
    # JOIN shape classification
    # ------------------------------------------------------------------

    def _classify_join_shape(self, sql: str, db_id: str, n_joins: int) -> str:
        """Return precise JOIN pattern code using schema FK graph.

        Returns one of: 0, 1p, 2p, 3p, 2i, 3i, 4i, Xm, or <n>? (unknown).
        """
        if n_joins == 0:
            return "0"
        if n_joins == 1:
            return "1p"  # single join is always a 1-hop path

        tables = self._extract_query_tables(sql, db_id)

        if len(tables) < 2:
            # Table extraction failed (schema mismatch or parse error)
            return f"{min(n_joins, 3)}?"

        # Build induced FK subgraph: in_sources[parent] = {child tables that FK to parent}
        in_sources: dict[str, set[str]] = defaultdict(set)
        for child, parent in self._fk_edges.get(db_id, []):
            if child in tables and parent in tables:
                in_sources[parent].add(child)

        max_in = max((len(v) for v in in_sources.values()), default=0)

        if max_in >= 2:
            # Intersection: k tables FK into same hub
            k = min(max_in, 4)
            if max_in > 4:
                return "Xm"
            return f"{k}i"

        # Path: cap reported depth at 3; deeper goes to Expert tier
        depth = min(n_joins, 3)
        return f"{depth}p"

    def _extract_query_tables(self, sql: str, db_id: str) -> set[str]:
        """Extract original-case table names referenced in SQL FROM/JOIN clauses."""
        lower_map = self._table_lower.get(db_id, {})
        tables: set[str] = set()
        for m in _FROM_JOIN.finditer(sql):
            name_lower = m.group(1).lower()
            if name_lower in lower_map:
                tables.add(lower_map[name_lower])
        return tables

    # ------------------------------------------------------------------
    # Report extra: schema-resolution stats
    # ------------------------------------------------------------------

    def resolution_stats(self, classifications: list[QueryClassification]) -> dict:
        """Count how many CEJSQ patterns were resolved precisely vs unknown."""
        precise = unknown = 0
        for c in classifications:
            if not c.is_cejsq:
                continue
            pat = c.pattern_code.rstrip("EMH")
            if "?" in pat:
                unknown += 1
            else:
                precise += 1
        total = precise + unknown
        return {
            "total_cejsq": total,
            "precise": precise,
            "unknown": unknown,
            "precise_pct": round(precise / total * 100, 1) if total else 0.0,
        }
