"""Decompose TPC-H official queries into CEJSQ skeletons and classify them.

Each TPC-H query = CEJSQ skeleton (FROM/JOIN + WHERE) + aggregate layer (SELECT/GROUP BY/HAVING).

This module:
  1. Strips the aggregate layer from each of the 22 official queries.
  2. Classifies the skeleton using the same taxonomy as Spider/BIRD.
  3. Reports which strategy codes TPC-H's official suite covers vs. misses.

Key finding for paper: TPC-H's 22 queries cover 0% CEJSQ as written (all have aggregates
or subqueries). Their skeletons span ~8 strategy codes — FlexBench fills all feasible codes.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from talk2metadata.analysis.spider.query_analyzer import (
    _AGG_PATTERN,
    _GROUP_BY_PATTERN,
    _HAVING_PATTERN,
    _OR_IN_WHERE_PATTERN,
    _SUBQUERY_PATTERN,
)
from talk2metadata.analysis.tpch.queries import TPCH_QUERIES
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)

# TPC-H FK edges (child → parent) — used for skeleton pattern classification
_TPCH_FK_EDGES: list[tuple[str, str]] = [
    ("lineitem", "orders"),
    ("lineitem", "part"),
    ("lineitem", "supplier"),
    ("orders", "customer"),
    ("customer", "nation"),
    ("supplier", "nation"),
    ("nation", "region"),
    ("partsupp", "part"),
    ("partsupp", "supplier"),
]

# Default target table for TPC-H analysis
_TARGET = "orders"

_FROM_CLAUSE_RE = re.compile(
    r"\bFROM\b(.+?)(?:\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_JOIN_SPLIT_RE = re.compile(
    r",|\b(?:INNER\s+|LEFT\s+(?:OUTER\s+)?|RIGHT\s+(?:OUTER\s+)?|FULL\s+(?:OUTER\s+)?|CROSS\s+)?JOIN\b",
    re.IGNORECASE,
)
_WHERE_BODY_RE = re.compile(
    r"\bWHERE\b(.+?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_SQL_KEYWORDS = frozenset(
    {
        "select",
        "where",
        "on",
        "set",
        "as",
        "order",
        "group",
        "having",
        "limit",
        "union",
        "intersect",
        "except",
        "outer",
        "inner",
        "cross",
        "left",
        "right",
        "full",
        "natural",
        "join",
        "from",
        "and",
        "or",
        "not",
        "exists",
        "in",
        "between",
        "like",
        "is",
        "null",
        "true",
        "false",
        "case",
        "when",
        "then",
        "else",
        "end",
        "distinct",
        "all",
        "any",
        "some",
    }
)


@dataclass
class TpcHDecomposition:
    query_num: int
    description: str

    # --- original query flags ---
    has_aggregate: bool = False
    has_group_by: bool = False
    has_having: bool = False
    has_subquery: bool = False
    has_or_in_where: bool = False
    aggregates: list[str] = field(default_factory=list)
    exclusion_reason: str = ""  # why the full query is not CEJSQ

    # --- skeleton info (outer FROM/WHERE after stripping aggregate layer) ---
    tables: list[str] = field(default_factory=list)
    join_count: int = 0
    where_conditions: int = 0
    skeleton_has_subquery: bool = False  # subquery in the skeleton itself
    skeleton_has_or: bool = False

    # --- classification ---
    skeleton_is_cejsq: bool = False  # True only if skeleton is also clean
    skeleton_strategy: str = ""  # e.g. "2iM", "1pE", "Xm", "NOT_CEJSQ"


def decompose_aggregate_query(sql: str) -> dict:
    """Decompose an aggregate SQL query into structural components.

    Returns a dict with:
        aggregates: list of aggregate function names found
        has_group_by, has_having, has_subquery, has_or_in_where: bool flags
        tables: list of table names from outer FROM/JOIN
        join_count: number of joins (len(tables) - 1)
        where_conditions: approximate count of AND-conjuncts in WHERE
        skeleton_has_subquery: True if subquery appears in outer WHERE/FROM
        skeleton_has_or: True if OR appears in outer WHERE
    """
    aggregates = list({m.group(1).upper() for m in _AGG_PATTERN.finditer(sql)})
    has_group_by = bool(_GROUP_BY_PATTERN.search(sql))
    has_having = bool(_HAVING_PATTERN.search(sql))
    has_subquery = bool(_SUBQUERY_PATTERN.search(sql))

    # Extract outer-level tables — strip subqueries first, then parse FROM clause
    # Iteratively remove innermost (SELECT ...) blocks
    sql_no_sub = sql
    for _ in range(5):
        sql_no_sub = re.sub(
            r"\([^()]*\)", "", sql_no_sub, flags=re.IGNORECASE | re.DOTALL
        )

    from_match = _FROM_CLAUSE_RE.search(sql_no_sub)
    tables: list[str] = []
    if from_match:
        from_body = from_match.group(1)
        # Strip any ON ... conditions inside the FROM clause
        from_body = re.sub(
            r"\bON\b.+?(?=,|\bJOIN\b|$)",
            " ",
            from_body,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for part in _JOIN_SPLIT_RE.split(from_body):
            word = (part.strip().split()[0] if part.strip() else "").lower()
            if word and word not in _SQL_KEYWORDS and word not in tables:
                tables.append(word)

    # WHERE body for OR / condition count
    where_match = _WHERE_BODY_RE.search(sql_no_sub)
    where_body = where_match.group(1).strip() if where_match else ""
    has_or = bool(_OR_IN_WHERE_PATTERN.search(where_body))
    n_and = len(re.findall(r"\bAND\b", where_body, re.IGNORECASE))
    where_conditions = n_and + 1 if where_body else 0

    return {
        "aggregates": aggregates,
        "has_group_by": has_group_by,
        "has_having": has_having,
        "has_subquery": has_subquery,
        "tables": tables,
        "join_count": max(len(tables) - 1, 0),
        "where_conditions": where_conditions,
        "skeleton_has_subquery": has_subquery,  # subquery in outer WHERE counts
        "skeleton_has_or": has_or,
    }


def _classify_skeleton(
    tables: list[str], join_count: int, where_conditions: int
) -> str:
    """Classify a CEJSQ skeleton using the TPC-H FK graph."""
    # Filter complexity
    if where_conditions <= 2:
        f = "E"
    elif where_conditions <= 5:
        f = "M"
    else:
        f = "H"

    if join_count == 0:
        return f"0{f}"
    if join_count == 1:
        return f"1p{f}"

    # For multi-table: inspect FK graph to distinguish path vs intersection
    # Build induced subgraph: edges where both endpoints are in query tables
    table_set = set(tables)
    in_sources: dict[str, set[str]] = defaultdict(set)
    for child, parent in _TPCH_FK_EDGES:
        if child in table_set and parent in table_set:
            in_sources[parent].add(child)

    max_in = max((len(v) for v in in_sources.values()), default=0)

    if max_in >= 2:
        k = min(max_in, 4)
        return f"{k}i{f}"

    # Check max path depth from target
    depth = min(join_count, 3)
    return f"{depth}p{f}"


def analyze_tpch_queries() -> list[TpcHDecomposition]:
    """Decompose and classify all 22 TPC-H official queries."""
    results = []

    for query_num, description, sql in TPCH_QUERIES:
        d = TpcHDecomposition(query_num=query_num, description=description)
        info = decompose_aggregate_query(sql)

        d.aggregates = info["aggregates"]
        d.has_group_by = info["has_group_by"]
        d.has_having = info["has_having"]
        d.has_subquery = info["has_subquery"]
        d.has_or_in_where = info["skeleton_has_or"]
        d.tables = info["tables"]
        d.join_count = info["join_count"]
        d.where_conditions = info["where_conditions"]
        d.skeleton_has_subquery = info["skeleton_has_subquery"]
        d.skeleton_has_or = info["skeleton_has_or"]

        # Exclusion reason for the full query (priority order, matches Spider classifier)
        if d.aggregates:
            d.exclusion_reason = "aggregate"
            d.has_aggregate = True
        elif d.has_group_by:
            d.exclusion_reason = "group_by"
        elif d.has_having:
            d.exclusion_reason = "having"
        elif d.has_subquery:
            d.exclusion_reason = "subquery"
        elif d.has_or_in_where:
            d.exclusion_reason = "or_condition"

        # Skeleton classification — only if skeleton itself is clean
        d.skeleton_is_cejsq = not d.skeleton_has_subquery and not d.skeleton_has_or
        if d.skeleton_is_cejsq and d.tables:
            d.skeleton_strategy = _classify_skeleton(
                d.tables, d.join_count, d.where_conditions
            )
        elif not d.skeleton_is_cejsq:
            reasons = []
            if d.skeleton_has_subquery:
                reasons.append("subquery")
            if d.skeleton_has_or:
                reasons.append("OR")
            d.skeleton_strategy = f"NOT_CEJSQ({'|'.join(reasons)})"
        else:
            d.skeleton_strategy = "empty"

        results.append(d)

    return results


def build_report(decompositions: list[TpcHDecomposition]) -> dict:
    """Build summary report for paper table."""
    total = len(decompositions)
    cejsq_full = sum(1 for d in decompositions if not d.exclusion_reason)
    skeleton_cejsq = sum(1 for d in decompositions if d.skeleton_is_cejsq)

    excl = defaultdict(int)
    for d in decompositions:
        if d.exclusion_reason:
            excl[d.exclusion_reason] += 1

    skeleton_strategies: dict[str, list[int]] = defaultdict(list)
    for d in decompositions:
        skeleton_strategies[d.skeleton_strategy].append(d.query_num)

    per_query = []
    for d in decompositions:
        per_query.append(
            {
                "q": d.query_num,
                "description": d.description,
                "tables": d.tables,
                "aggregates": d.aggregates,
                "has_subquery": d.has_subquery,
                "has_or": d.has_or_in_where,
                "has_group_by": d.has_group_by,
                "has_having": d.has_having,
                "exclusion_reason": d.exclusion_reason,
                "join_count": d.join_count,
                "where_conditions": d.where_conditions,
                "skeleton_is_cejsq": d.skeleton_is_cejsq,
                "skeleton_strategy": d.skeleton_strategy,
            }
        )

    return {
        "total_queries": total,
        "cejsq_as_written": cejsq_full,
        "cejsq_pct": round(cejsq_full / total * 100, 1),
        "skeleton_cejsq_count": skeleton_cejsq,
        "exclusion_breakdown": dict(excl),
        "skeleton_strategy_distribution": {
            k: {"count": len(v), "queries": v}
            for k, v in sorted(skeleton_strategies.items())
        },
        "per_query": per_query,
    }


def run_analysis(output_dir: str | Path | None = None, save: bool = True) -> dict:
    """Run full TPC-H skeleton analysis and optionally save results."""
    decompositions = analyze_tpch_queries()
    report = build_report(decompositions)

    _print_report(report)

    if save and output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "skeleton_analysis.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved TPC-H skeleton analysis to {path}")

    return report


def _print_report(report: dict) -> None:
    print(f"\n{'='*60}")
    print("TPC-H Skeleton Analysis")
    print(f"{'='*60}")
    print(f"Total queries:          {report['total_queries']}")
    print(
        f"CEJSQ as written:       {report['cejsq_as_written']} ({report['cejsq_pct']}%)"
    )
    print(f"Clean CEJSQ skeleton:   {report['skeleton_cejsq_count']}")

    print("\nExclusion breakdown (full query):")
    for reason, count in sorted(
        report["exclusion_breakdown"].items(), key=lambda x: -x[1]
    ):
        print(f"  {reason:<20} {count}")

    print("\nSkeleton strategy distribution:")
    for code, info in sorted(report["skeleton_strategy_distribution"].items()):
        qs = ", ".join(f"Q{q}" for q in info["queries"])
        print(f"  {code:<25} {info['count']:2d}  ({qs})")

    print("\nPer-query summary:")
    print(f"  {'Q':<4} {'Tables':<45} {'Excl':<12} {'Skeleton'}")
    print(f"  {'-'*4} {'-'*45} {'-'*12} {'-'*15}")
    for q in report["per_query"]:
        tables_str = ",".join(q["tables"])[:43]
        print(
            f"  Q{q['q']:<3} {tables_str:<45} {q['exclusion_reason']:<12} {q['skeleton_strategy']}"
        )
