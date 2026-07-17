"""Baseline A — Random SQL (no taxonomy) for RQ2 ablation.

Generates NL2SQL QA pairs by randomly sampling tables and columns without
any strategy allocation or proportional weighting. This is the unstructured
baseline that FlexBench's taxonomy is compared against.

Key differences from FlexBench:
- No strategy taxonomy (no 21-code system)
- No proportional allocation (all queries are generated ad-hoc)
- Random JOIN depth (0–3 hops, uniform distribution)
- Random filter count (1–4, uniform distribution)
- No FK-guided path/intersection construction

CAVEAT (comparison validity): this baseline does NOT run the QAVerifier
faithfulness/quality gate and computes answers via its own pandas execution
path with no SQL syntax check — acceptance criteria are LAXER than FlexBench's.
Comparisons are therefore valid for structural coverage/entropy only; for any
validity/quality comparison, re-validate all modes uniformly first.

Expected finding: biased toward 0E pattern (single-table, easy);
lower strategy coverage entropy and lower overall diversity.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from talk2metadata.agent import AgentWrapper
from talk2metadata.core.qa.qa_pair import QAPair, _generate_uid
from talk2metadata.core.qa.query_builder import Filter, JoinPath, QuerySpec
from talk2metadata.core.qa.question_generator import QuestionGenerator
from talk2metadata.core.schema import SchemaMetadata
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)

_NUMERIC_DTYPES = frozenset({"int64", "float64", "int32", "float32"})
_MAX_ATTEMPTS = 20


@dataclass
class _RandomQuerySpec:
    """Minimal query specification without strategy code."""

    target_table: str
    tables: List[str]  # all tables in query (target + joined)
    filters: List[Filter]
    sql: str
    answer_row_ids: List[Any]


class RandomSQLBaseline:
    """Baseline A: generates QA pairs by random SQL construction without taxonomy."""

    def __init__(
        self,
        schema: SchemaMetadata,
        tables: Dict[str, pd.DataFrame],
        agent: Optional[AgentWrapper] = None,
        max_answer_records: int = 15,
    ):
        self.schema = schema
        self.tables = tables
        self.agent = agent or AgentWrapper()
        self.max_answer_records = max_answer_records
        self._question_gen = QuestionGenerator(agent=self.agent, schema=schema)
        self._target_pk = (
            schema.tables.get(schema.target_table.lower(), {}).primary_key or "id"
        )
        # Pre-compute sample values per column to avoid repeated unique() calls
        self._sample_vals_cache: Dict[Tuple[str, str], list] = {}
        for tname, df in self.tables.items():
            pk_col = (self._target_pk or "id").lower()
            for col in df.columns:
                if col.lower() == pk_col:
                    continue
                vals = df[col].dropna().unique()
                if len(vals) > 20:
                    vals = vals[:20]
                self._sample_vals_cache[(tname, col)] = vals.tolist()

    def generate(self, n: int) -> List[QAPair]:
        """Generate n QA pairs using random SQL construction."""
        pairs: List[QAPair] = []
        attempts = 0
        max_total_attempts = n * _MAX_ATTEMPTS

        while len(pairs) < n and attempts < max_total_attempts:
            attempts += 1
            try:
                spec = self._build_random_query()
                if spec is None:
                    continue

                # Generate NL question using LLM (same as FlexBench)
                question = self._generate_question(spec)
                if not question:
                    continue

                pair = QAPair(
                    uid=_generate_uid(),
                    question=question,
                    answer_row_ids=spec.answer_row_ids,
                    sql=spec.sql,
                    strategy="random",  # no taxonomy code
                    difficulty_score=-1.0,
                    involved_tables=spec.tables,
                    involved_columns=[],
                    involved_filters=[],
                )
                pairs.append(pair)

            except Exception as e:
                logger.debug(f"Random SQL attempt failed: {e}")
                continue

        logger.info(
            f"RandomSQL: generated {len(pairs)}/{n} pairs in {attempts} attempts"
        )
        return pairs

    def _build_random_query(self) -> Optional[_RandomQuerySpec]:
        """Build a random SQL query without strategy guidance."""
        target = self.schema.target_table.lower()
        target_df = self.tables.get(target)
        if target_df is None or len(target_df) == 0:
            return None

        # Random JOIN depth: 0–2 hops (uniform)
        max_hop = min(2, len(self.schema.foreign_keys))
        n_hops = random.randint(0, max_hop)

        # Build table chain (random, not FK-guided)
        query_tables = [target]
        join_clauses: List[str] = []

        if n_hops > 0:
            related = self.schema.get_related_tables(target)
            available = [t for t in related if t != target and t in self.tables]
            n_join = min(n_hops, len(available))
            joined = random.sample(available, n_join) if available else []

            for jtable in joined:
                # Find a FK between current last table and joined table
                fk = self._find_fk(query_tables[-1], jtable)
                if fk is None:
                    break
                query_tables.append(jtable)
                child_col = fk.child_column.lower()
                parent_col = fk.parent_column.lower()
                child_tbl = fk.child_table.lower()
                parent_tbl = fk.parent_table.lower()
                join_clauses.append(
                    f"JOIN {jtable} ON {child_tbl}.{child_col} = {parent_tbl}.{parent_col}"
                )

        # Random filters: 1–4 conditions from any query table
        n_filters = random.randint(1, 4)
        filters = self._build_random_filters(query_tables, n_filters)
        if not filters:
            return None

        # Build SQL
        pk = self._target_pk
        from_clause = f"{target} " + " ".join(join_clauses)
        where_parts = [f.to_sql() for f in filters]
        where_clause = " AND ".join(where_parts)
        sql = f"SELECT DISTINCT {target}.{pk} FROM {from_clause} WHERE {where_clause}"

        # Execute and check answer count
        ids = self._execute(query_tables, filters)
        if ids is None or len(ids) == 0 or len(ids) > self.max_answer_records:
            return None

        return _RandomQuerySpec(
            target_table=target,
            tables=query_tables,
            filters=filters,
            sql=sql,
            answer_row_ids=ids,
        )

    def _build_random_filters(self, tables: List[str], n: int) -> List[Filter]:
        """Build n random filter conditions from the given tables."""
        filters = []
        candidates = []

        for tname in tables:
            df = self.tables.get(tname)
            if df is None:
                continue
            for col in df.columns:
                if col.lower() == (self._target_pk or "id"):
                    continue
                sample_vals = self._sample_vals_cache.get((tname, col), [])
                if sample_vals:
                    candidates.append((tname, col, sample_vals))

        if not candidates:
            return []

        random.shuffle(candidates)
        used_cols: set[tuple] = set()

        for tname, col, sample_vals in candidates:
            if len(filters) >= n:
                break
            if (tname, col) in used_cols:
                continue

            val = random.choice(sample_vals[:20])  # cap sample pool
            is_numeric = isinstance(val, (int, float))

            if is_numeric:
                op = random.choice(["=", ">=", "<="])
            else:
                op = "="
                val = str(val)

            filters.append(
                Filter(table=tname, column=col.lower(), operator=op, value=val)
            )
            used_cols.add((tname, col))

        return filters

    def _find_fk(self, table_a: str, table_b: str):
        """Find a FK between two tables (either direction)."""
        a, b = table_a.lower(), table_b.lower()
        for fk in self.schema.foreign_keys:
            if (fk.child_table.lower() == a and fk.parent_table.lower() == b) or (
                fk.child_table.lower() == b and fk.parent_table.lower() == a
            ):
                return fk
        return None

    def _execute(self, tables: List[str], filters: List[Filter]) -> Optional[List[Any]]:
        """Execute query and return matching target table row IDs."""
        target = self.schema.target_table.lower()
        pk = self._target_pk
        df = self.tables.get(target)
        if df is None:
            return None

        # Apply filters via pandas
        result_df = df.copy()
        result_df = result_df.add_suffix(f"_{target}")
        joined: set[str] = {target}

        for tname in tables[1:]:
            tdf = self.tables.get(tname)
            if tdf is None:
                continue
            from_table = list(joined)[-1]
            fk = self._find_fk(from_table, tname)
            if fk is None:
                continue
            join_df = tdf.copy().add_suffix(f"_{tname}")
            # Determine which side of the FK each table is on
            child_tbl = fk.child_table.lower()
            child_col = fk.child_column.lower()
            parent_col = fk.parent_column.lower()
            if child_tbl == from_table:
                left_col = f"{child_col}_{from_table}"
                right_col = f"{parent_col}_{tname}"
            else:
                left_col = f"{parent_col}_{from_table}"
                right_col = f"{child_col}_{tname}"
            if left_col not in result_df.columns or right_col not in join_df.columns:
                continue
            result_df = result_df.merge(
                join_df, left_on=left_col, right_on=right_col, how="inner"
            )
            joined.add(tname)

        # Apply filters (all columns are suffixed with _{table_name})
        for filt in filters:
            tname = filt.table.lower()
            col = filt.column.lower()
            suffix_col = f"{col}_{tname}"
            if suffix_col not in result_df.columns:
                continue
            try:
                if filt.operator == "=":
                    mask = result_df[suffix_col].astype(str) == str(filt.value)
                elif filt.operator == ">=":
                    mask = pd.to_numeric(
                        result_df[suffix_col], errors="coerce"
                    ) >= float(filt.value)
                elif filt.operator == "<=":
                    mask = pd.to_numeric(
                        result_df[suffix_col], errors="coerce"
                    ) <= float(filt.value)
                else:
                    continue
                result_df = result_df[mask]
            except Exception:
                continue

        pk_col = f"{pk}_{target}" if f"{pk}_{target}" in result_df.columns else None
        if pk_col is None:
            return None

        return result_df[pk_col].dropna().unique().tolist()

    def _generate_question(self, spec: _RandomQuerySpec) -> str:
        """Generate NL question from SQL using LLM (same as FlexBench)."""
        # Reuse QuestionGenerator with a minimal QuerySpec
        fake_spec = QuerySpec(
            strategy="random",
            target_table=spec.target_table,
            answer_id_column=self._target_pk,
            join_paths=[JoinPath(tables=spec.tables, join_type="chain")],
            filters=spec.filters,
            sql=spec.sql,
            involved_tables=spec.tables,
            involved_columns=[f"{f.table}.{f.column}" for f in spec.filters],
            answer_row_ids=spec.answer_row_ids,
        )
        return self._question_gen.generate(fake_spec)
