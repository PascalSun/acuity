"""Analyze Spider SQL queries to measure CEJSQ coverage.

CEJSQ = Conjunctive Equi-Join Selection Query:
  SELECT [cols] FROM T [JOIN ...] WHERE [AND-connected conditions]

Exclusions (out of scope for our taxonomy):
  - Aggregates in SELECT: COUNT, SUM, AVG, MIN, MAX
  - GROUP BY / HAVING
  - UNION / INTERSECT / EXCEPT
  - Subqueries (nested SELECT)
  - Non-equi-joins (range joins, CROSS JOIN)
  - ORDER BY without WHERE (rankings, not filtering)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)

# Regex patterns
_AGG_PATTERN = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", re.IGNORECASE)
_SUBQUERY_PATTERN = re.compile(r"\(\s*SELECT\b", re.IGNORECASE)
_SET_OP_PATTERN = re.compile(r"\b(UNION|INTERSECT|EXCEPT)\b", re.IGNORECASE)
_GROUP_BY_PATTERN = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)
_HAVING_PATTERN = re.compile(r"\bHAVING\b", re.IGNORECASE)
_JOIN_PATTERN = re.compile(r"\bJOIN\b", re.IGNORECASE)
_CROSS_JOIN_PATTERN = re.compile(r"\bCROSS\s+JOIN\b", re.IGNORECASE)
_WHERE_PATTERN = re.compile(r"\bWHERE\b", re.IGNORECASE)
_OR_IN_WHERE_PATTERN = re.compile(r"\bOR\b", re.IGNORECASE)
_ORDER_BY_PATTERN = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)
_LIMIT_PATTERN = re.compile(r"\bLIMIT\b", re.IGNORECASE)
_DISTINCT_PATTERN = re.compile(r"\bSELECT\s+DISTINCT\b", re.IGNORECASE)


@dataclass
class QueryClassification:
    query: str
    db_id: str
    question: str

    # Exclusion flags
    has_aggregate: bool = False
    has_group_by: bool = False
    has_having: bool = False
    has_set_op: bool = False
    has_subquery: bool = False
    has_cross_join: bool = False
    has_or_in_where: bool = False
    has_order_by_only: bool = False  # ORDER BY without WHERE (ranking queries)

    # Structural flags
    has_join: bool = False
    has_where: bool = False
    n_joins: int = 0
    n_where_conditions: int = 0

    # Classification
    is_cejsq: bool = False
    exclusion_reason: str = ""

    # Strategy pattern (if CEJSQ)
    pattern_code: str = ""


@dataclass
class QueryReport:
    total_queries: int = 0
    cejsq_count: int = 0

    # Breakdown of excluded queries
    excluded_aggregate: int = 0
    excluded_group_by: int = 0
    excluded_set_op: int = 0
    excluded_subquery: int = 0
    excluded_cross_join: int = 0
    excluded_or_condition: int = 0
    excluded_order_only: int = 0

    # Among CEJSQ: strategy pattern distribution
    pattern_distribution: dict[str, int] = field(default_factory=dict)

    # Per-database breakdown
    per_db: dict[str, dict] = field(default_factory=dict)

    @property
    def cejsq_pct(self) -> float:
        return (
            self.cejsq_count / self.total_queries * 100 if self.total_queries else 0.0
        )


class SpiderQueryAnalyzer:
    """Classifies Spider SQL queries as CEJSQ or not, and maps to strategy patterns."""

    def analyze_all(
        self, queries: list[dict]
    ) -> tuple[list[QueryClassification], QueryReport]:
        """Analyze all Spider query examples.

        Args:
            queries: List of Spider examples with 'query', 'db_id', 'question'.

        Returns:
            (list of QueryClassification, aggregate QueryReport)
        """
        classifications = [self._classify(q) for q in queries]
        report = self._build_report(classifications)
        return classifications, report

    def _classify(self, example: dict) -> QueryClassification:
        sql = example.get("query", "").strip()
        db_id = example.get("db_id", "")
        question = example.get("question", "")

        c = QueryClassification(query=sql, db_id=db_id, question=question)

        # Must be a SELECT statement
        if not sql.upper().lstrip().startswith("SELECT"):
            c.exclusion_reason = "not_select"
            return c

        # Check exclusions in priority order
        if _AGG_PATTERN.search(sql):
            c.has_aggregate = True
            c.exclusion_reason = "aggregate"
            return c

        if _GROUP_BY_PATTERN.search(sql):
            c.has_group_by = True
            c.exclusion_reason = "group_by"
            return c

        if _HAVING_PATTERN.search(sql):
            c.has_having = True
            c.exclusion_reason = "having"
            return c

        if _SET_OP_PATTERN.search(sql):
            c.has_set_op = True
            c.exclusion_reason = "set_op"
            return c

        if _SUBQUERY_PATTERN.search(sql):
            c.has_subquery = True
            c.exclusion_reason = "subquery"
            return c

        if _CROSS_JOIN_PATTERN.search(sql):
            c.has_cross_join = True
            c.exclusion_reason = "cross_join"
            return c

        # OR in WHERE makes it non-conjunctive
        where_match = re.search(
            r"\bWHERE\b(.+?)(?:\bORDER\b|\bLIMIT\b|\bGROUP\b|$)",
            sql,
            re.IGNORECASE | re.DOTALL,
        )
        if where_match and _OR_IN_WHERE_PATTERN.search(where_match.group(1)):
            c.has_or_in_where = True
            c.exclusion_reason = "or_condition"
            return c

        # Passed all checks — it's a CEJSQ
        has_where = bool(_WHERE_PATTERN.search(sql))
        c.is_cejsq = True
        c.has_join = bool(_JOIN_PATTERN.search(sql))
        c.has_where = has_where
        c.n_joins = len(re.findall(r"\bJOIN\b", sql, re.IGNORECASE))
        c.n_where_conditions = self._count_conditions(sql)
        c.pattern_code = self._infer_pattern(c)

        return c

    def _count_conditions(self, sql: str) -> int:
        """Count AND-connected conditions in WHERE clause."""
        where_match = re.search(
            r"\bWHERE\b(.+?)(?:\bORDER\b|\bLIMIT\b|\bGROUP\b|$)",
            sql,
            re.IGNORECASE | re.DOTALL,
        )
        if not where_match:
            return 0
        where_body = where_match.group(1).strip()
        # Count AND keywords as separators (+1 for the first condition)
        n_and = len(re.findall(r"\bAND\b", where_body, re.IGNORECASE))
        return n_and + 1 if where_body else 0

    def _infer_pattern(self, c: QueryClassification) -> str:
        """Map query to strategy pattern code (approximate from SQL structure)."""
        n_joins = c.n_joins
        n_conds = c.n_where_conditions

        # Filter complexity suffix
        if n_conds <= 2:
            diff = "E"
        elif n_conds <= 5:
            diff = "M"
        else:
            diff = "H"

        # JOIN pattern (simplified — can't distinguish path vs intersection
        # without schema info, so we just use join count)
        if n_joins == 0:
            pat = "0"
        elif n_joins == 1:
            pat = "1p"
        elif n_joins == 2:
            pat = "2p_or_2i"
        elif n_joins == 3:
            pat = "3p_or_3i"
        else:
            pat = f"{n_joins}x"

        return f"{pat}{diff}"

    def _build_report(self, classifications: list[QueryClassification]) -> QueryReport:
        r = QueryReport(total_queries=len(classifications))

        for c in classifications:
            if c.is_cejsq:
                r.cejsq_count += 1
                pat = c.pattern_code
                r.pattern_distribution[pat] = r.pattern_distribution.get(pat, 0) + 1
            else:
                reason = c.exclusion_reason
                if reason == "aggregate":
                    r.excluded_aggregate += 1
                elif reason == "group_by":
                    r.excluded_group_by += 1
                elif reason == "set_op":
                    r.excluded_set_op += 1
                elif reason == "subquery":
                    r.excluded_subquery += 1
                elif reason == "cross_join":
                    r.excluded_cross_join += 1
                elif reason == "or_condition":
                    r.excluded_or_condition += 1
                elif reason == "order_only":
                    r.excluded_order_only += 1

            # Per-db stats
            db = c.db_id
            if db not in r.per_db:
                r.per_db[db] = {"total": 0, "cejsq": 0}
            r.per_db[db]["total"] += 1
            if c.is_cejsq:
                r.per_db[db]["cejsq"] += 1

        return r

    def print_summary(self, report: QueryReport) -> None:
        print("\n" + "=" * 60)
        print("  Spider Query Analysis — CEJSQ Coverage")
        print("=" * 60)
        print(f"  Total queries:     {report.total_queries}")
        print(f"  CEJSQ (in scope):  {report.cejsq_count}  ({report.cejsq_pct:.1f}%)")
        total_excl = report.total_queries - report.cejsq_count
        print(f"  Out of scope:      {total_excl}  ({100 - report.cejsq_pct:.1f}%)")
        print()
        print("  Exclusion breakdown:")
        excl = [
            ("Aggregate (COUNT/SUM/...)", report.excluded_aggregate),
            ("GROUP BY", report.excluded_group_by),
            ("Set operations (UNION...)", report.excluded_set_op),
            ("Subqueries", report.excluded_subquery),
            ("OR in WHERE", report.excluded_or_condition),
            ("ORDER BY only (no WHERE)", report.excluded_order_only),
            ("CROSS JOIN", report.excluded_cross_join),
        ]
        for label, count in excl:
            if count:
                pct = count / report.total_queries * 100
                print(f"    {label:35s}: {count:4d}  ({pct:.1f}%)")
        print()
        print("  CEJSQ pattern distribution (by join count × filter complexity):")
        for pat, count in sorted(
            report.pattern_distribution.items(), key=lambda x: -x[1]
        ):
            pct = count / report.cejsq_count * 100 if report.cejsq_count else 0
            print(f"    {pat:15s}: {count:4d}  ({pct:.1f}% of CEJSQ)")
        print("=" * 60 + "\n")
