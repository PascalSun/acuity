"""Analyze Spider database schemas for star-schema prevalence.

Key design choices (documented for paper clarity):
-------------------------------------------------
1. in-degree = DISTINCT SOURCE TABLES pointing to a table, not raw FK column count.
   Rationale: a table pair (A, B) may have multiple FK columns (composite key).
   We count the relationship once per unique source table.

2. Schema types are MUTUALLY EXCLUSIVE:
   - no_fk:        zero FK relationships
   - path_only:    has FK chains but no table has in-degree >= 2
   - intersection: at least one table has in-degree >= 2 (may also have paths)

3. Feasible strategy patterns are INCLUSIVE (cumulative):
   - If a DB supports 3i, it also supports 2i, 1p, and 0.
   - Reported counts overlap intentionally; the paper must state this.

4. hub_out_degree is reported separately to distinguish:
   - Pure endpoint hub (out-degree=0): true star schema center
   - Intermediate hub (out-degree>0): inside a chain AND a convergence point
     (i.e., snowflake schema where dimension tables also have sub-dimensions)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from talk2metadata.analysis.spider.models import (
    DatabaseSchema,
    ForeignKey,
    StarSchemaReport,
)
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


class SpiderAnalyzer:
    """Analyzes Spider database schemas to detect hub-centric patterns."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_all(
        self, raw_schemas: list[dict]
    ) -> tuple[list[DatabaseSchema], StarSchemaReport]:
        schemas = [self._parse_database(db) for db in raw_schemas]
        report = self._build_report(schemas)
        return schemas, report

    def analyze_one(self, raw_db: dict) -> DatabaseSchema:
        return self._parse_database(raw_db)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_database(self, db: dict) -> DatabaseSchema:
        table_names = db["table_names_original"]
        col_names = db["column_names_original"]

        fks = []
        for child_col_idx, parent_col_idx in db.get("foreign_keys", []):
            child_tbl_idx = col_names[child_col_idx][0]
            parent_tbl_idx = col_names[parent_col_idx][0]
            if child_tbl_idx < 0 or parent_tbl_idx < 0:
                continue
            fks.append(
                ForeignKey(
                    child_table=table_names[child_tbl_idx],
                    child_column=col_names[child_col_idx][1],
                    parent_table=table_names[parent_tbl_idx],
                    parent_column=col_names[parent_col_idx][1],
                )
            )

        schema = DatabaseSchema(
            db_id=db["db_id"], tables=list(table_names), foreign_keys=fks
        )
        self._compute_hub(schema)
        self._compute_path_depth(schema)
        self._assign_schema_type(schema)
        self._compute_feasible_patterns(schema)
        return schema

    def _compute_hub(self, schema: DatabaseSchema) -> None:
        """Find hub table using DISTINCT SOURCE TABLE in-degree.

        For each parent table, count how many distinct child tables FK into it.
        This avoids inflating in-degree when two tables share a composite FK
        (multiple FK columns between the same pair count as one relationship).
        """
        # distinct child tables per parent
        in_sources: dict[str, set[str]] = defaultdict(set)
        # distinct parent tables per child (for out-degree)
        out_targets: dict[str, set[str]] = defaultdict(set)

        for fk in schema.foreign_keys:
            if fk.child_table != fk.parent_table:  # skip self-referential
                in_sources[fk.parent_table].add(fk.child_table)
                out_targets[fk.child_table].add(fk.parent_table)

        if not in_sources:
            return

        hub = max(in_sources, key=lambda t: len(in_sources[t]))
        schema.hub_table = hub
        schema.hub_in_degree = len(in_sources[hub])
        schema.hub_out_degree = len(out_targets.get(hub, set()))
        schema.hub_is_pure_endpoint = schema.hub_out_degree == 0

    def _compute_path_depth(self, schema: DatabaseSchema) -> None:
        """Longest directed path in FK DAG (excluding self-references)."""
        # Build adjacency: child -> set of parents
        parents: dict[str, set[str]] = defaultdict(set)
        for fk in schema.foreign_keys:
            if fk.child_table != fk.parent_table:
                parents[fk.child_table].add(fk.parent_table)

        def depth(node: str, visited: set[str]) -> int:
            if node in visited:
                return 0
            visited.add(node)
            ps = parents.get(node, set())
            return 0 if not ps else 1 + max(depth(p, visited) for p in ps)

        schema.max_path_depth = max((depth(t, set()) for t in schema.tables), default=0)

    def _assign_schema_type(self, schema: DatabaseSchema) -> None:
        """Assign mutually exclusive schema type label."""
        if not schema.foreign_keys:
            schema.schema_type = "no_fk"
        elif schema.hub_in_degree >= 2:
            schema.schema_type = "intersection"
        else:
            schema.schema_type = "path_only"

    def _compute_feasible_patterns(self, schema: DatabaseSchema) -> None:
        """Which strategy patterns are feasible?

        Note: these are INCLUSIVE — supporting 3i implies also supporting
        2i, 1p, 0. The paper must state this when reporting counts.
        """
        codes = ["0"]  # direct always feasible

        depth = schema.max_path_depth
        for k in range(1, min(depth, 3) + 1):
            codes.append(f"{k}p")

        hub_deg = schema.hub_in_degree
        for k in range(2, min(hub_deg, 4) + 1):
            codes.append(f"{k}i")

        schema.pattern_codes = codes

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def _build_report(self, schemas: list[DatabaseSchema]) -> StarSchemaReport:
        r = StarSchemaReport(total_databases=len(schemas))

        for s in schemas:
            # Schema type (mutually exclusive)
            if s.schema_type == "no_fk":
                r.type_no_fk += 1
            elif s.schema_type == "path_only":
                r.type_path_only += 1
            else:  # intersection
                r.type_intersection += 1
                r.intersection_db_ids.append(s.db_id)

            # Intersection sub-counts (inclusive)
            if s.hub_in_degree >= 2:
                r.intersection_2plus += 1
            if s.hub_in_degree >= 3:
                r.intersection_3plus += 1
            if s.hub_in_degree >= 4:
                r.intersection_4plus += 1

            # Snowflake: intersection DB that also has a path depth > 1
            if s.schema_type == "intersection" and s.max_path_depth > 1:
                r.snowflake_count += 1

            # Pure endpoint hub
            if s.schema_type == "intersection" and s.hub_is_pure_endpoint:
                r.pure_endpoint_hub += 1

            # Distributions
            r.hub_in_degree_distribution[s.hub_in_degree] = (
                r.hub_in_degree_distribution.get(s.hub_in_degree, 0) + 1
            )
            r.hub_out_degree_distribution[s.hub_out_degree] = (
                r.hub_out_degree_distribution.get(s.hub_out_degree, 0) + 1
            )
            r.path_depth_distribution[s.max_path_depth] = (
                r.path_depth_distribution.get(s.max_path_depth, 0) + 1
            )
            r.table_count_distribution[s.n_tables] = (
                r.table_count_distribution.get(s.n_tables, 0) + 1
            )

            # Feasible patterns (inclusive — see docstring)
            for p in s.pattern_codes:
                r.feasible_patterns[p] = r.feasible_patterns.get(p, 0) + 1

        return r

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save_report(
        self,
        schemas: list[DatabaseSchema],
        report: StarSchemaReport,
        output_dir: str | Path = "data/spider",
    ) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        schemas_data = [
            {
                "db_id": s.db_id,
                "n_tables": s.n_tables,
                "n_foreign_keys": s.n_foreign_keys,
                "hub_table": s.hub_table,
                "hub_in_degree": s.hub_in_degree,
                "hub_out_degree": s.hub_out_degree,
                "hub_is_pure_endpoint": s.hub_is_pure_endpoint,
                "max_path_depth": s.max_path_depth,
                "schema_type": s.schema_type,
                "feasible_patterns": s.pattern_codes,
            }
            for s in schemas
        ]
        with open(out / "schema_analysis.json", "w") as f:
            json.dump(schemas_data, f, indent=2)

        report_data = {
            "_notes": {
                "schema_types": "mutually exclusive: no_fk | path_only | intersection",
                "intersection_subsets": "inclusive: 2plus ⊇ 3plus ⊇ 4plus",
                "feasible_patterns": "inclusive: supporting 3i implies also 2i, 1p, 0",
                "in_degree": "counts DISTINCT source tables, not FK column pairs",
            },
            "total_databases": report.total_databases,
            "schema_types": {
                "no_fk": report.type_no_fk,
                "path_only": report.type_path_only,
                "intersection_2plus": report.type_intersection,
            },
            "intersection_breakdown": {
                "hub_indegree_2plus": report.intersection_2plus,
                "hub_indegree_3plus": report.intersection_3plus,
                "hub_indegree_4plus": report.intersection_4plus,
                "pure_endpoint_hub": report.pure_endpoint_hub,
                "snowflake_also_has_path_depth_gt1": report.snowflake_count,
            },
            "hub_in_degree_distribution": dict(
                sorted(report.hub_in_degree_distribution.items())
            ),
            "hub_out_degree_distribution": dict(
                sorted(report.hub_out_degree_distribution.items())
            ),
            "path_depth_distribution": dict(
                sorted(report.path_depth_distribution.items())
            ),
            "feasible_patterns_inclusive": dict(
                sorted(report.feasible_patterns.items())
            ),
            "intersection_db_ids": sorted(report.intersection_db_ids),
        }
        with open(out / "star_schema_report.json", "w") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Saved analysis to {out}/")

    def print_summary(self, report: StarSchemaReport) -> None:
        total = report.total_databases

        def pct(n):
            return f"{n:3d} ({n/total*100:4.1f}%)"

        print("\n" + "=" * 68)
        print("  Schema Topology Analysis — Framework Applicability & Richness")
        print("=" * 68)
        print(f"  Total databases: {total}")
        print()

        # --- Universal coverage statement ---
        all_supported = total  # every schema with a target table is supported
        print("  Framework applicability (every schema is supported):")
        print(f"    All databases        : {pct(all_supported)}  ← 100% supported")
        print()

        # --- Strategy richness by topology archetype (mutually exclusive) ---
        # Flat: d=0, k=0 → no_fk
        # Chain: d≥1, k≤1 → path_only
        # Star/Snowflake: k≥2 → intersection
        print("  Schema topology archetypes (mutually exclusive, relative to proxy T):")
        print(
            f"    Flat       (d=0,k=0) : {pct(report.type_no_fk)}"
            f"  → supports {{0}} only"
        )
        print(
            f"    Chain      (d≥1,k≤1) : {pct(report.type_path_only)}"
            f"  → supports path patterns (1p..3p)"
        )
        print(
            f"    Star/Snowflake (k≥2) : {pct(report.type_intersection)}"
            f"  → supports intersection patterns (2i..4i)"
        )
        print()

        # --- Intersection richness sub-breakdown ---
        print("  Convergence degree breakdown (INCLUSIVE — k≥3 ⊆ k≥2):")
        print(f"    k >= 2 (2-way intersection) : {pct(report.intersection_2plus)}")
        print(f"    k >= 3 (3-way intersection) : {pct(report.intersection_3plus)}")
        print(f"    k >= 4 (4-way intersection) : {pct(report.intersection_4plus)}")
        print()

        n_int = report.type_intersection or 1
        print("  Among Star/Snowflake schemas:")
        print(
            f"    Pure endpoint hub (k_out=0) : "
            f"{report.pure_endpoint_hub:3d} ({report.pure_endpoint_hub/n_int*100:.1f}%)"
            f"  ← true star center"
        )
        print(
            f"    Snowflake (also d>1)        : "
            f"{report.snowflake_count:3d} ({report.snowflake_count/n_int*100:.1f}%)"
            f"  ← chains beyond hub"
        )
        print()

        # --- Feasible query strategy patterns ---
        print(
            "  Feasible query strategy patterns (INCLUSIVE — supporting 3i ⊇ 2i ⊇ 1p ⊇ 0):"
        )
        print("  NOTE: (d,k) is computed relative to proxy T = max-in-degree table.")
        print("        True richness may be higher if T is chosen differently.\n")
        for pat in ["0", "1p", "2p", "3p", "2i", "3i", "4i"]:
            count = report.feasible_patterns.get(pat, 0)
            bar = "#" * min(count, 35)
            print(f"    {pat:4s}: {count:3d} ({count/total*100:4.1f}%)  {bar}")
        print("=" * 68 + "\n")
