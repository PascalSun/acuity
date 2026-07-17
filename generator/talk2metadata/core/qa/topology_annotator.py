"""Topology-guided reasoning (TGR) chain annotator.

Generates deterministic reasoning chains from schema topology and parsed SQL.
Used to annotate Spider/BIRD training data for TGR fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass

from talk2metadata.analysis.spider.analyzer import SpiderAnalyzer
from talk2metadata.analysis.spider.models import DatabaseSchema
from talk2metadata.analysis.spider.query_analyzer import QueryClassification
from talk2metadata.core.qa.sql_parser import WhereCondition
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)

# Human-readable descriptions for join pattern codes
PATTERN_DESCRIPTIONS: dict[str, str] = {
    "0": "direct query, no joins",
    "1p": "1-hop path join (2 tables)",
    "2p": "2-hop chain join (3 tables)",
    "3p": "3-hop chain join (4 tables)",
    "2i": "2-way intersection at hub",
    "3i": "3-way intersection at hub",
    "4i": "4-way intersection at hub",
    "Xm": "mixed join pattern",
}

DIFFICULTY_DESCRIPTIONS: dict[str, str] = {
    "E": "easy (1-2 filters)",
    "M": "medium (3-5 filters)",
    "H": "hard (6+ filters)",
}


@dataclass
class TopologyInfo:
    """Pre-computed topology for a database."""

    db_id: str
    archetype: str  # "flat" | "chain" | "star" | "snowflake"
    d: int  # max path depth
    k: int  # hub in-degree
    hub_table: str | None
    schema_type: str  # "no_fk" | "path_only" | "intersection"
    n_tables: int
    feasible_patterns: list[str]


class TopologyAnnotator:
    """Generates topology-guided reasoning chains for SQL queries.

    Pre-computes schema topology for all databases, then generates
    deterministic reasoning chains for each (question, SQL) pair.
    """

    def __init__(self, tables_json: list[dict]):
        analyzer = SpiderAnalyzer()
        self._db_schemas: dict[str, DatabaseSchema] = {}
        self._topology: dict[str, TopologyInfo] = {}

        for db in tables_json:
            db_id = db["db_id"]
            schema = analyzer.analyze_one(db)
            self._db_schemas[db_id] = schema
            self._topology[db_id] = self._compute_topology(schema)

        logger.info(f"TopologyAnnotator loaded {len(self._topology)} databases")

    def _compute_topology(self, schema: DatabaseSchema) -> TopologyInfo:
        """Map DatabaseSchema to TopologyInfo with archetype."""
        if schema.schema_type == "no_fk":
            archetype = "flat"
        elif schema.schema_type == "path_only":
            archetype = "chain"
        elif schema.max_path_depth <= 1:
            archetype = "star"
        else:
            archetype = "snowflake"

        return TopologyInfo(
            db_id=schema.db_id,
            archetype=archetype,
            d=schema.max_path_depth,
            k=schema.hub_in_degree,
            hub_table=schema.hub_table,
            schema_type=schema.schema_type,
            n_tables=schema.n_tables,
            feasible_patterns=schema.pattern_codes,
        )

    def get_topology(self, db_id: str) -> TopologyInfo:
        """Return cached topology for a database."""
        return self._topology[db_id]

    def build_chain(
        self,
        db_id: str,
        classification: QueryClassification,
        join_tables: list[str],
        where_conditions: list[WhereCondition],
        select_columns: list[str],
    ) -> str:
        """Build full reasoning chain for a CEJSQ query.

        Returns:
            Chain string like:
            <think>
            [TOPOLOGY] ...
            [PATTERN] ...
            [PLAN] ...
            [COLUMNS] ...
            </think>
        """
        topo = self._topology.get(db_id)
        if topo is None:
            return self._build_fallback_chain(db_id, join_tables)

        parts = ["<think>"]

        # [TOPOLOGY] section
        parts.append(
            f"[TOPOLOGY] db={db_id}, archetype={topo.archetype.upper()}, "
            f"hub={topo.hub_table or 'none'}, k={topo.k}, d={topo.d}"
        )

        # [PATTERN] section
        pattern_code = classification.pattern_code
        # Split into pattern + difficulty: "2iM" → ("2i", "M")
        pat, diff = self._split_pattern_code(pattern_code)
        pat_desc = PATTERN_DESCRIPTIONS.get(pat, f"pattern {pat}")
        diff_desc = DIFFICULTY_DESCRIPTIONS.get(diff, f"difficulty {diff}")
        parts.append(f"[PATTERN] {pattern_code}: {pat_desc}, {diff_desc}")

        # [PLAN] section
        plan_lines = self._build_plan(
            join_tables, where_conditions, topo, pat
        )
        parts.append("[PLAN]")
        for line in plan_lines:
            parts.append(f"  {line}")

        # [COLUMNS] section
        if select_columns:
            cols_str = ", ".join(select_columns[:10])  # cap at 10 for brevity
            parts.append(f"[COLUMNS] {cols_str}")

        parts.append("</think>")
        return "\n".join(parts)

    def build_chain_simple(
        self,
        db_id: str,
        classification: QueryClassification,
        join_tables: list[str],
    ) -> str:
        """Build simplified chain for non-CEJSQ queries.

        Includes topology info but skips detailed pattern/plan.
        """
        topo = self._topology.get(db_id)
        if topo is None:
            return self._build_fallback_chain(db_id, join_tables)

        parts = ["<think>"]

        parts.append(
            f"[TOPOLOGY] db={db_id}, archetype={topo.archetype.upper()}, "
            f"hub={topo.hub_table or 'none'}, k={topo.k}, d={topo.d}"
        )

        reason = classification.exclusion_reason or "complex"
        parts.append(f"[QUERY_TYPE] non-CEJSQ ({reason})")

        if join_tables:
            parts.append(f"[TABLES] {' -> '.join(join_tables)}")

        parts.append("</think>")
        return "\n".join(parts)

    def _build_plan(
        self,
        join_tables: list[str],
        where_conditions: list[WhereCondition],
        topo: TopologyInfo,
        pattern: str,
    ) -> list[str]:
        """Build plan lines describing join paths and filters."""
        lines = []

        if not join_tables:
            lines.append(f"Query single table: {topo.hub_table or 'target'}")
        elif pattern.endswith("i"):
            # Intersection pattern: tables join to hub
            hub = topo.hub_table or join_tables[0]
            lines.append(f"Intersect at {hub}:")
            for i, tbl in enumerate(join_tables[1:], 1):
                # Find filters for this table
                tbl_filters = [
                    w for w in where_conditions if w.table == tbl
                ]
                filter_str = ""
                if tbl_filters:
                    filter_str = " (filter: " + ", ".join(
                        f"{w.column} {w.operator} ..." for w in tbl_filters
                    ) + ")"
                lines.append(f"  path{i}: {hub} <- {tbl}{filter_str}")
        elif pattern.endswith("p"):
            # Path pattern: chain of tables
            chain = " -> ".join(join_tables)
            lines.append(f"Chain: {chain}")
        else:
            # Generic fallback
            lines.append(f"Tables: {', '.join(join_tables)}")

        # Add filters on the first table (or hub)
        first_table = join_tables[0] if join_tables else None
        if first_table:
            hub_filters = [
                w for w in where_conditions if w.table == first_table
            ]
            if hub_filters:
                filter_str = ", ".join(
                    f"{w.column} {w.operator} ..." for w in hub_filters
                )
                lines.append(f"  filters on {first_table}: {filter_str}")

        return lines

    def _build_fallback_chain(self, db_id: str, join_tables: list[str]) -> str:
        """Minimal chain when topology is unavailable."""
        parts = ["<think>"]
        parts.append(f"[TOPOLOGY] db={db_id}, archetype=UNKNOWN")
        if join_tables:
            parts.append(f"[TABLES] {' -> '.join(join_tables)}")
        parts.append("</think>")
        return "\n".join(parts)

    @staticmethod
    def _split_pattern_code(code: str) -> tuple[str, str]:
        """Split '2iM' → ('2i', 'M'), '0E' → ('0', 'E')."""
        if not code:
            return ("0", "E")
        diff = code[-1] if code[-1] in ("E", "M", "H") else ""
        pat = code[:-1] if diff else code
        return (pat, diff)
