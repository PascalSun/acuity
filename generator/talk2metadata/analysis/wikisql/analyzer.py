"""Analyze WikiSQL benchmark for topology and CEJSQ coverage.

WikiSQL (Zhong et al. 2017) — 80,594 questions over 25,669 Wikipedia tables.
Key characteristics:
  - Every question targets exactly ONE table (no JOINs ever).
  - No FK relationships between tables.
  - Schema topology: 100% Flat (d=0, k=0).
  - SQL is simple: SELECT [agg] col FROM table WHERE cond [AND cond...]

Source: kaxap/pg-wikiSQL-sql-instructions-80k (HuggingFace)
Fields: question, create_table_statement, sql_query, wiki_sql_table_id
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from talk2metadata.analysis.spider.query_analyzer import SpiderQueryAnalyzer
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)

HF_DATASET = "kaxap/pg-wikiSQL-sql-instructions-80k"


@dataclass
class WikiSQLReport:
    total_queries: int = 0
    unique_tables: int = 0

    # Topology (always Flat for WikiSQL)
    type_flat: int = 0  # = unique_tables (all single-table, no FKs)

    # CEJSQ coverage
    cejsq_count: int = 0
    excluded_aggregate: int = 0
    excluded_group_by: int = 0
    excluded_set_op: int = 0
    excluded_subquery: int = 0
    excluded_or_condition: int = 0
    excluded_cross_join: int = 0

    pattern_distribution: dict[str, int] = field(default_factory=dict)

    @property
    def cejsq_pct(self) -> float:
        return (
            self.cejsq_count / self.total_queries * 100 if self.total_queries else 0.0
        )


class WikiSQLAnalyzer:
    """Analyzes WikiSQL benchmark for CEJSQ coverage and topology."""

    def load_queries(self, splits: list[str] | None = None) -> list[dict]:
        """Load WikiSQL queries from HuggingFace.

        Args:
            splits: list of splits to load, e.g. ['train', 'validation', 'test'].
                    Defaults to all three splits.
        Returns:
            List of dicts with keys: query, db_id, question.
        """
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:
            raise RuntimeError("Install 'datasets': uv pip install datasets") from e

        if splits is None:
            splits = ["train", "validation", "test"]

        rows = []
        for split in splits:
            logger.info(f"Loading WikiSQL split '{split}'...")
            ds = load_dataset(HF_DATASET, split=split)
            for row in ds:
                rows.append(
                    {
                        "query": row["sql_query"],
                        "db_id": row["wiki_sql_table_id"],
                        "question": row["question"],
                    }
                )
            logger.info(f"  Loaded {len(ds)} examples from '{split}'")

        logger.info(f"Total WikiSQL queries loaded: {len(rows)}")
        return rows

    def count_unique_tables(self, queries: list[dict]) -> int:
        return len({q["db_id"] for q in queries})

    def analyze(
        self,
        splits: list[str] | None = None,
        output_dir: str | Path | None = None,
        save: bool = True,
    ) -> WikiSQLReport:
        """Run full analysis: load queries, classify CEJSQ, build report."""
        queries = self.load_queries(splits)
        n_tables = self.count_unique_tables(queries)
        logger.info(f"Unique tables: {n_tables}")

        qa = SpiderQueryAnalyzer()
        classifications, qa_report = qa.analyze_all(queries)

        report = WikiSQLReport(
            total_queries=len(queries),
            unique_tables=n_tables,
            type_flat=n_tables,
            cejsq_count=qa_report.cejsq_count,
            excluded_aggregate=qa_report.excluded_aggregate,
            excluded_group_by=qa_report.excluded_group_by,
            excluded_set_op=qa_report.excluded_set_op,
            excluded_subquery=qa_report.excluded_subquery,
            excluded_or_condition=qa_report.excluded_or_condition,
            excluded_cross_join=qa_report.excluded_cross_join,
            pattern_distribution=qa_report.pattern_distribution,
        )

        self.print_summary(report)

        if save and output_dir:
            self._save(report, Path(output_dir))

        return report

    def print_summary(self, report: WikiSQLReport) -> None:
        total = report.total_queries

        def pct(n):
            return f"{n:6d} ({n/total*100:4.1f}%)"

        print("\n" + "=" * 60)
        print("  WikiSQL Analysis — Topology & CEJSQ Coverage")
        print("=" * 60)
        print(f"  Total queries    : {total:,}")
        print(f"  Unique tables    : {report.unique_tables:,}")
        print()
        print("  Schema topology (all tables are single-table, no FKs):")
        print(
            f"    Flat (d=0, k=0): {report.unique_tables:,} (100%)  ← supports {{0}} only"
        )
        print()
        print(f"  CEJSQ (in scope) : {pct(report.cejsq_count)}")
        out = total - report.cejsq_count
        print(f"  Out of scope     : {pct(out)}")
        print()
        print("  Exclusion breakdown:")
        excl = [
            ("Aggregate (COUNT/SUM/...)", report.excluded_aggregate),
            ("GROUP BY", report.excluded_group_by),
            ("Set operations", report.excluded_set_op),
            ("Subqueries", report.excluded_subquery),
            ("OR in WHERE", report.excluded_or_condition),
            ("CROSS JOIN", report.excluded_cross_join),
        ]
        for label, count in excl:
            if count:
                print(f"    {label:35s}: {count:5d}  ({count/total*100:.1f}%)")
        print()
        print("  CEJSQ pattern distribution:")
        for pat, count in sorted(
            report.pattern_distribution.items(), key=lambda x: -x[1]
        ):
            pct2 = count / report.cejsq_count * 100 if report.cejsq_count else 0
            print(f"    {pat:12s}: {count:5d}  ({pct2:.1f}% of CEJSQ)")
        print("=" * 60 + "\n")

    def _save(self, report: WikiSQLReport, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        data = {
            "total_queries": report.total_queries,
            "unique_tables": report.unique_tables,
            "topology": {
                "flat_100pct": True,
                "note": "WikiSQL is single-table only — no FK relationships exist",
            },
            "cejsq_count": report.cejsq_count,
            "cejsq_pct": round(report.cejsq_pct, 1),
            "excluded_aggregate": report.excluded_aggregate,
            "excluded_group_by": report.excluded_group_by,
            "excluded_set_op": report.excluded_set_op,
            "excluded_subquery": report.excluded_subquery,
            "excluded_or_condition": report.excluded_or_condition,
            "excluded_cross_join": report.excluded_cross_join,
            "pattern_distribution": report.pattern_distribution,
        }
        path = out / "query_analysis.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved WikiSQL analysis to {path}")
