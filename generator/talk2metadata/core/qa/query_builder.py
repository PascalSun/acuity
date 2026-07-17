"""Query builder for generating SQL queries based on difficulty strategies.

This module generates SQL queries by:
1. Randomly selecting tables and columns based on the strategy
2. Generating appropriate filter conditions
3. Building JOIN statements for path/intersection patterns
4. Executing the query to get answer record IDs
"""

import random
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from talk2metadata.core.qa.column_type_inference import (
    FILTERABLE_COLUMN_TYPES,
    infer_all_column_types,
)
from talk2metadata.core.qa.difficulty_classifier import (
    DifficultyClassifier,
    JoinPath,
    QueryPlan,
)
from talk2metadata.core.schema import ForeignKey, SchemaMetadata
from talk2metadata.core.schema.types import ColumnType
from talk2metadata.utils.json_utils import json_safe
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)

# Maximum times a table may appear in a chain via self-referential FK (hierarchy depth)
_MAX_SELF_REF_DEPTH = 3


def _real_table_name(name: str) -> str:
    """Strip alias suffix from a table name.

    When a table appears more than once in a chain (via self-referential FK),
    subsequent occurrences are represented as ``Table__1``, ``Table__2``, etc.
    This helper returns the canonical table name by stripping that suffix.
    """
    return name.split("__")[0] if "__" in name else name


@dataclass
class Filter:
    """Represents a filter condition."""

    table: str
    column: str
    operator: str  # '=', '>', '<', '>=', '<=', 'LIKE'
    value: Any  # The filter value
    column_type: Optional[str] = None  # ColumnType value string for downstream use

    def to_sql(self) -> str:
        """Convert filter to SQL condition."""
        table_lower = self.table.lower()
        column_lower = self.column.lower()
        if self.operator == "LIKE":
            escaped = str(self.value).replace("'", "''")
            return f"{table_lower}.{column_lower} LIKE '%{escaped}%'"
        elif isinstance(self.value, str):
            escaped = self.value.replace("'", "''")
            return f"{table_lower}.{column_lower} {self.operator} '{escaped}'"
        else:
            return f"{table_lower}.{column_lower} {self.operator} {self.value}"

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "table": self.table,
            "column": self.column,
            "operator": self.operator,
            "value": json_safe(self.value),
        }
        if self.column_type:
            result["column_type"] = self.column_type
        return result


@dataclass
class QuerySpec:
    """Specification for a generated query."""

    strategy: str  # Difficulty code (e.g., "2iM")
    target_table: str
    answer_id_column: str
    join_paths: List[JoinPath]
    filters: List[Filter]
    sql: str
    involved_tables: List[str]
    involved_columns: List[str]  # table.column format
    answer_row_ids: List[Any]  # Answer record IDs from query execution
    filter_column_types: Optional[Dict[str, str]] = None  # table.column -> ColumnType


class QueryBuilder:
    """Builds SQL queries based on difficulty strategies."""

    # Max joined frames kept in the per-structure cache
    _JOINED_FRAME_CACHE_MAX = 16
    # Frames bigger than this are rebuilt on demand instead of cached
    # (holding several multi-100k-row frames OOMs workers on large DBs)
    _JOINED_FRAME_CACHE_MAX_ROWS = 200_000
    # Abort a join structure whose intermediate result explodes beyond this
    # (fan-out merges on large DBs previously ate all memory and took the
    # whole process pool down)
    _JOIN_MAX_ROWS = 1_000_000

    def __init__(
        self,
        schema: SchemaMetadata,
        tables: Dict[str, pd.DataFrame],
        max_answer_records: Optional[int] = None,
        engine: Optional[Engine] = None,
        connection_string: Optional[str] = None,
    ):
        """Initialize query builder.

        Args:
            schema: Schema metadata
            tables: Dictionary mapping table names to DataFrames
            max_answer_records: Maximum number of answer records per question.
                               If provided, queries with more results will be regenerated.
            engine: Optional SQLAlchemy engine for SQL validation
            connection_string: Optional database connection string for SQL validation
        """
        self.schema = schema
        self.target_table = schema.target_table
        self.classifier = DifficultyClassifier()
        self.max_answer_records = max_answer_records

        # Normalize tables: convert table names and column names to lowercase
        # to match the schema metadata (which was normalized during database import)
        self.tables = {}
        for table_name, df in tables.items():
            table_name_lower = table_name.lower()
            df_normalized = df.copy()
            # Convert column names to lowercase
            df_normalized.columns = [col.lower() for col in df_normalized.columns]
            self.tables[table_name_lower] = df_normalized

        logger.debug(
            f"Normalized {len(self.tables)} tables to lowercase: {list(self.tables.keys())}"
        )

        target_pk = schema.tables[self.target_table].primary_key
        self.target_pk = target_pk.lower() if isinstance(target_pk, str) else None
        if not self.target_pk:
            target_df = self.tables.get(self.target_table.lower())
            if target_df is not None:
                self.target_pk = self._infer_primary_key(target_df)
        if not self.target_pk:
            raise ValueError(
                f"Target table '{self.target_table}' has no primary key / unique id column"
            )

        # Set up database engine for SQL validation
        self.engine = engine
        if connection_string and not engine:
            self.engine = create_engine(connection_string)

        # Pre-compute FK columns (non-semantic, should not appear in WHERE filters)
        self._fk_columns: Set[str] = set()
        for fk in self.schema.foreign_keys:
            self._fk_columns.add(f"{fk.child_table.lower()}.{fk.child_column.lower()}")
            self._fk_columns.add(
                f"{fk.parent_table.lower()}.{fk.parent_column.lower()}"
            )

        # Infer column types and compute filterable columns per table.
        # Excludes: PKs, FK columns, id-like columns, user-configured non_filterable_columns,
        # and columns inferred as TEXT or IDENTIFIER type.
        self._column_types: Dict[str, Dict[str, ColumnType]] = {}
        self._filterable_columns: Dict[str, List[str]] = {}

        for table_name, table_meta in self.schema.tables.items():
            table_lower = table_name.lower()
            table_pk = (
                table_meta.primary_key.lower() if table_meta.primary_key else None
            )
            table_df = self.tables.get(table_lower)
            table_df_cols = set(table_df.columns) if table_df is not None else set()

            # Infer types for all columns
            col_types = infer_all_column_types(table_meta, table_df)
            self._column_types[table_lower] = col_types

            # User-configured non-filterable columns (from schema JSON)
            user_excluded = {c.lower() for c in table_meta.non_filterable_columns}

            filterable = []
            for col in table_meta.columns:
                col_lower = col.lower()
                if col_lower == table_pk:
                    continue
                if col_lower not in table_df_cols:
                    continue
                if f"{table_lower}.{col_lower}" in self._fk_columns:
                    continue
                if col_lower in user_excluded:
                    continue
                # Skip id-like columns (surrogate keys, internal references)
                if col_lower == "id" or (
                    col_lower.endswith("_id") and col_lower != table_pk
                ):
                    continue
                # Skip columns with non-filterable inferred types
                inferred_type = col_types.get(col_lower, ColumnType.TEXT)
                if inferred_type not in FILTERABLE_COLUMN_TYPES:
                    continue
                filterable.append(col_lower)
            self._filterable_columns[table_lower] = filterable

            # Log type summary for debugging
            type_summary = {}
            for col, ct in col_types.items():
                type_summary.setdefault(ct.value, []).append(col)
            logger.debug(f"Table '{table_name}' column types: {dict(type_summary)}")
            if not filterable:
                logger.debug(
                    f"Table '{table_name}' has no filterable columns after type inference"
                )

        # Pre-enumerate feasible join structures per pattern
        self._feasible_structures: Dict[str, List[List[JoinPath]]] = {}
        self._enumerate_feasible_structures()
        self._last_build_diagnostics: Dict[str, Any] = {}

        # Cache of joined frames per structure (same structures are reused across
        # many attempts; rebuilding the join each attempt dominated runtime).
        self._joined_frame_cache: Dict[Tuple, Optional[pd.DataFrame]] = {}

    def _enumerate_feasible_structures(self) -> None:
        """Pre-enumerate all feasible join structures per pattern.

        Populates self._feasible_structures with pattern -> list of valid structures.
        Each structure is a List[JoinPath] that can be used directly in build_query.
        """
        # Pattern "0": always feasible if target table has filterable columns
        target_lower = self.target_table.lower()
        if self._filterable_columns.get(target_lower):
            self._feasible_structures["0"] = [[]]

        # Path patterns: enumerate all FK chains of length N
        for n in [1, 2, 3]:
            pattern = f"{n}p"
            chains = self._enumerate_chain_paths(n)
            if chains:
                self._feasible_structures[pattern] = chains
                logger.info(
                    f"Pattern '{pattern}': {len(chains)} feasible chain structure(s)"
                )
            else:
                logger.info(f"Pattern '{pattern}': no feasible structures")

        # Intersection patterns: enumerate all C(dim_tables, N) combinations
        # Get all non-self-referential dimension tables connected to target
        dim_fks = [
            fk
            for fk in self.schema.get_foreign_keys_for_table(
                self.target_table, direction="both"
            )
            if fk.child_table != fk.parent_table
        ]
        dim_tables_with_fk = []
        for fk in dim_fks:
            related = (
                fk.parent_table
                if fk.child_table.lower() == self.target_table.lower()
                else fk.child_table
            )
            if related not in [t for t, _ in dim_tables_with_fk]:
                dim_tables_with_fk.append((related, fk))

        for n in [2, 3, 4]:
            pattern = f"{n}i"
            if len(dim_tables_with_fk) < n:
                logger.info(
                    f"Pattern '{pattern}': not enough dimension tables "
                    f"({len(dim_tables_with_fk)} < {n})"
                )
                continue
            structures = []
            for combo in combinations(dim_tables_with_fk, n):
                join_paths = []
                for related_table, _fk in combo:
                    join_paths.append(
                        JoinPath(
                            tables=[self.target_table, related_table],
                            join_type="star",
                        )
                    )
                structures.append(join_paths)
            if structures:
                self._feasible_structures[pattern] = structures
                logger.info(
                    f"Pattern '{pattern}': {len(structures)} feasible intersection structure(s)"
                )

    def _enumerate_chain_paths(
        self, hops: int, max_paths: int = 50
    ) -> List[List[JoinPath]]:
        """Enumerate all distinct FK chain paths of a given length from target table.

        Args:
            hops: Number of hops (JOINs)
            max_paths: Maximum number of paths to enumerate (avoid combinatorial explosion)

        Returns:
            List of structures, each being a list with one JoinPath of type 'chain'
        """
        results: List[List[str]] = []

        def _dfs(
            current: str,
            path: List[str],
            visited_counts: Dict[str, int],
            depth: int,
        ) -> None:
            if depth == hops:
                results.append(list(path))
                return
            if len(results) >= max_paths:
                return

            real_current = _real_table_name(current)
            fks = self.schema.get_foreign_keys_for_table(
                real_current, direction="outgoing"
            )
            if not fks:
                fks = self.schema.get_foreign_keys_for_table(
                    real_current, direction="incoming"
                )
            if not fks:
                return

            for fk in fks:
                next_real = (
                    fk.parent_table
                    if fk.child_table.lower() == real_current.lower()
                    else fk.child_table
                )
                is_self_ref = fk.child_table == fk.parent_table
                count = visited_counts.get(next_real, 0)
                table_meta = self.schema.tables.get(next_real)
                max_depth = (
                    table_meta.self_ref_depth
                    if table_meta and table_meta.self_ref_depth is not None
                    else _MAX_SELF_REF_DEPTH
                )
                if count == 0 or (is_self_ref and count < max_depth):
                    # Use alias for repeated tables
                    alias = f"{next_real}__{count}" if count > 0 else next_real
                    path.append(alias)
                    visited_counts[next_real] = count + 1
                    _dfs(alias, path, visited_counts, depth + 1)
                    path.pop()
                    visited_counts[next_real] = count
                    if len(results) >= max_paths:
                        return

        _dfs(
            self.target_table,
            [self.target_table],
            {self.target_table: 1},
            0,
        )

        return [
            [JoinPath(tables=path_tables, join_type="chain")] for path_tables in results
        ]

    def get_feasible_patterns(self) -> List[str]:
        """Return list of patterns that have at least one feasible join structure."""
        return list(self._feasible_structures.keys())

    def get_feasible_strategies(self) -> List[str]:
        """Return list of strategy codes (pattern+difficulty) that are feasible.

        Feasibility is value-aware, not just topology-aware: a difficulty tier is
        only feasible when at least one of the pattern's join structures offers
        enough filterable columns to reach the tier's minimum predicate count.
        Previously every ``{pattern}{E,M,H}`` was declared feasible, so quota
        parked on unreachable M/H tiers became a guaranteed shortfall and each
        such tier burned its full retry budget on every DB.
        """
        strategies = []
        for pattern, structures in self._feasible_structures.items():
            # Max filterable-column count achievable by any structure of this
            # pattern (mirrors how _generate_filters builds its candidate pool,
            # including alias repeats for self-referential chains).
            max_cols = 0
            for structure in structures:
                involved = self._get_involved_tables(structure)
                if not involved:
                    involved = [self.target_table]
                n = sum(
                    len(
                        self._filterable_columns.get(
                            _real_table_name(t).lower(), []
                        )
                    )
                    for t in involved
                )
                max_cols = max(max_cols, n)
            for diff in ["E", "M", "H"]:
                min_filters, _ = self._get_filter_range(diff)
                if max_cols >= min_filters:
                    strategies.append(f"{pattern}{diff}")
        return strategies

    def get_last_build_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostics from the most recent build_query call."""
        return dict(self._last_build_diagnostics)

    def build_query(
        self,
        strategy: str,
        max_attempts: int = 10,
        structure_index: Optional[int] = None,
    ) -> Optional[QuerySpec]:
        """Build a query based on the difficulty strategy.

        Uses pre-enumerated feasible structures when available for deterministic
        generation. Falls back to random generation for patterns not pre-enumerated.

        Args:
            strategy: Difficulty code (e.g., "2iM")
            max_attempts: Maximum number of attempts to generate a valid query
            structure_index: If provided, select this specific structure from the
                           pre-enumerated list (mod length). Used by generator for
                           round-robin diversity.

        Returns:
            QuerySpec object, or None if generation failed
        """
        # Parse strategy upfront
        pattern, difficulty = self._parse_strategy(strategy)

        # Check if pattern is pre-enumerated and infeasible
        if pattern not in self._feasible_structures:
            logger.debug(f"Pattern '{pattern}' has no feasible structures, skipping")
            self._last_build_diagnostics = {
                "status": "failed",
                "strategy": strategy,
                "pattern": pattern,
                "difficulty": difficulty,
                "attempts_used": 0,
                "no_result_count": 0,
                "too_many_result_count": 0,
                "validation_failed_count": 0,
                "constraint_error_count": 0,
                "primary_reason_code": "pattern_infeasible",
                "summary": f"Pattern '{pattern}' has no feasible pre-enumerated structures",
                "failure_reasons": [],
            }
            return None

        failure_reasons = []
        no_result_count = 0
        validation_failed_count = 0
        too_many_result_count = 0
        constraint_error_count = 0
        insufficient_filter_count = 0
        last_attempt_info = None  # Store info from last attempt for analysis

        structures = self._feasible_structures[pattern]

        gold_mismatch_count = 0

        for attempt in range(max_attempts):
            try:
                # Pick join structure from pre-enumerated list
                if (
                    structures and structures[0]
                ):  # Non-empty structures (pattern != "0")
                    if structure_index is not None:
                        idx = (structure_index + attempt) % len(structures)
                    else:
                        idx = random.randint(0, len(structures) - 1)
                    join_paths = structures[idx]
                else:
                    # Pattern "0" — no joins
                    join_paths = []

                # Build (cached) joined frame once; anchor-row filter sampling
                # derives predicates from a surviving row so the conjunction is
                # non-empty by construction.
                joined_df = self._get_joined_frame(join_paths)
                if joined_df is None or len(joined_df) == 0:
                    no_result_count += 1
                    logger.debug("Join structure yields no rows, retrying...")
                    continue

                # Generate filters (anchor-row sampling)
                filters = self._generate_filters(
                    difficulty, join_paths, joined_df=joined_df
                )
                min_filters, _ = self._get_filter_range(difficulty)
                if len(filters) < min_filters:
                    insufficient_filter_count += 1
                    logger.debug(
                        "Generated too few filters for %s: %s < %s",
                        strategy,
                        len(filters),
                        min_filters,
                    )
                    continue

                # Build SQL
                sql = self._build_sql(join_paths, filters)

                # Execute query to get answer IDs
                max_results = (
                    (self.max_answer_records + 1)
                    if self.max_answer_records is not None
                    else None
                )
                filtered_df = self._apply_filters(joined_df, filters)
                answer_row_ids = (
                    self._extract_answer_ids(filtered_df, max_results=max_results)
                    if filtered_df is not None
                    else []
                )
                logger.debug(answer_row_ids)

                # Check if we got valid results
                if not answer_row_ids or len(answer_row_ids) == 0:
                    no_result_count += 1
                    # Store info from this attempt for later analysis
                    involved_tables = self._get_involved_tables(join_paths)
                    last_attempt_info = {
                        "pattern": pattern,
                        "difficulty": difficulty,
                        "join_paths": join_paths,
                        "filters": filters,
                        "involved_tables": involved_tables,
                        "num_filters": len(filters),
                    }
                    logger.debug("Query returned no results, retrying...")
                    continue

                # If the query matches too many records, it's too general for QA generation.
                if (
                    self.max_answer_records is not None
                    and len(answer_row_ids) > self.max_answer_records
                ):
                    too_many_result_count += 1
                    involved_tables = self._get_involved_tables(join_paths)
                    last_attempt_info = {
                        "pattern": pattern,
                        "difficulty": difficulty,
                        "join_paths": join_paths,
                        "filters": filters,
                        "involved_tables": involved_tables,
                        "num_filters": len(filters),
                    }
                    logger.debug(
                        "Query returned too many results (%s > %s), retrying...",
                        len(answer_row_ids),
                        self.max_answer_records,
                    )
                    continue

                # Get involved tables and columns
                involved_tables = self._get_involved_tables(join_paths)
                involved_columns = [f"{f.table}.{f.column}" for f in filters]

                # Collect column types for the filters
                fct = {
                    f"{f.table}.{f.column}": f.column_type
                    for f in filters
                    if f.column_type
                }

                query_spec = QuerySpec(
                    strategy=strategy,
                    target_table=self.target_table,
                    answer_id_column=self.target_pk,
                    join_paths=join_paths,
                    filters=filters,
                    sql=sql,
                    involved_tables=involved_tables,
                    involved_columns=involved_columns,
                    answer_row_ids=answer_row_ids,
                    filter_column_types=fct if fct else None,
                )

                # Validate the generated query
                if self._validate_query(query_spec):
                    # Gold-answer gate: the stored SQL, executed on the real
                    # engine, must reproduce answer_row_ids exactly. Catches
                    # pandas-vs-SQL divergences (LIKE case, type coercion,
                    # literal corruption) before a wrong gold pair ships.
                    gold_ok, gold_err = self.verify_gold_execution(query_spec)
                    if not gold_ok:
                        gold_mismatch_count += 1
                        if gold_err and gold_err not in failure_reasons:
                            failure_reasons.append(f"gold mismatch: {gold_err}")
                        logger.debug(
                            "Gold-answer verification failed for %s: %s",
                            strategy,
                            gold_err,
                        )
                        continue
                    logger.debug(
                        f"Successfully generated query for {strategy} with {len(answer_row_ids)} results"
                    )
                    self._last_build_diagnostics = {
                        "status": "success",
                        "strategy": strategy,
                        "pattern": pattern,
                        "difficulty": difficulty,
                        "attempts_used": attempt + 1,
                        "no_result_count": no_result_count,
                        "too_many_result_count": too_many_result_count,
                        "validation_failed_count": validation_failed_count,
                        "constraint_error_count": constraint_error_count,
                        "insufficient_filter_count": insufficient_filter_count,
                        "gold_mismatch_count": gold_mismatch_count,
                        "primary_reason_code": None,
                        "summary": "success",
                        "failure_reasons": list(failure_reasons),
                    }
                    return query_spec
                else:
                    validation_failed_count += 1

            except ValueError as e:
                # These are expected errors (e.g., cannot build chain, no foreign keys)
                constraint_error_count += 1
                error_msg = str(e)
                if error_msg not in failure_reasons:
                    failure_reasons.append(error_msg)
                logger.debug(f"Attempt {attempt + 1} failed: {e}")
                continue
            except Exception as e:
                # Unexpected errors
                constraint_error_count += 1
                error_msg = f"Unexpected error: {str(e)}"
                if error_msg not in failure_reasons:
                    failure_reasons.append(error_msg)
                logger.debug(f"Attempt {attempt + 1} failed: {e}")
                continue

        # Explain why this instance failed: each attempt uses random filters/join paths.
        reason_parts = []
        if no_result_count > 0:
            analysis = self._analyze_empty_result_reason(
                strategy, last_attempt_info, no_result_count
            )
            reason_parts.append(
                f"random filter/join choices produced empty results ({analysis})"
            )
        if too_many_result_count > 0:
            reason_parts.append(
                f"random filter choices matched >{self.max_answer_records} rows "
                f"in {too_many_result_count} attempts (query too broad)"
            )
        if validation_failed_count > 0:
            reason_parts.append(
                f"generated query structure did not match {strategy} classification "
                f"({validation_failed_count} attempts)"
            )
        if insufficient_filter_count > 0:
            reason_parts.append(
                f"only {insufficient_filter_count} attempts produced fewer than the "
                f"required {min_filters} filters for {strategy}"
            )
        if gold_mismatch_count > 0:
            reason_parts.append(
                f"stored SQL did not reproduce answer_row_ids in "
                f"{gold_mismatch_count} attempts (gold-answer mismatch)"
            )
        if failure_reasons:
            detailed = [r for r in failure_reasons if " - " in r or "because:" in r]
            msg = (
                detailed[0].replace("because:", "-")
                if detailed
                else "; ".join(list(set(failure_reasons))[:2])
            )
            reason_parts.append(f"constraint error: {msg}")

        reason_str = f" — {'; '.join(reason_parts)}" if reason_parts else ""
        primary_reason_code = self._determine_primary_reason_code(
            pattern=pattern,
            no_result_count=no_result_count,
            too_many_result_count=too_many_result_count,
            validation_failed_count=validation_failed_count,
            constraint_error_count=constraint_error_count,
            insufficient_filter_count=insufficient_filter_count,
            failure_reasons=failure_reasons,
            gold_mismatch_count=gold_mismatch_count,
        )
        self._last_build_diagnostics = {
            "status": "failed",
            "strategy": strategy,
            "pattern": pattern,
            "difficulty": difficulty,
            "attempts_used": max_attempts,
            "no_result_count": no_result_count,
            "too_many_result_count": too_many_result_count,
            "validation_failed_count": validation_failed_count,
            "constraint_error_count": constraint_error_count,
            "insufficient_filter_count": insufficient_filter_count,
            "gold_mismatch_count": gold_mismatch_count,
            "primary_reason_code": primary_reason_code,
            "summary": "; ".join(reason_parts) if reason_parts else "generation failed",
            "failure_reasons": list(failure_reasons),
        }

        logger.warning(
            f"Could not generate query for one {strategy} instance after {max_attempts} attempts{reason_str}"
        )
        return None

    def _infer_primary_key(self, df: pd.DataFrame) -> Optional[str]:
        if "id" in df.columns and df["id"].is_unique and not df["id"].isna().any():
            return "id"

        for col in df.columns:
            if col.endswith("_id") and df[col].is_unique and not df[col].isna().any():
                return col

        for col in df.columns:
            if df[col].is_unique and not df[col].isna().any():
                return col

        return None

    def _analyze_empty_result_reason(
        self, strategy: str, last_attempt_info: Optional[Dict], no_result_count: int
    ) -> str:
        """Analyze why queries returned empty results and provide professional error message.

        Args:
            strategy: Strategy code
            last_attempt_info: Information from last attempt (if available)
            no_result_count: Number of attempts that returned no results

        Returns:
            Professional error message explaining the failure
        """
        if not last_attempt_info:
            return f"all {no_result_count} attempts returned empty result sets"

        pattern = last_attempt_info.get("pattern", "")
        num_filters = last_attempt_info.get("num_filters", 0)
        involved_tables = last_attempt_info.get("involved_tables", [])

        # Analyze the pattern in plain language
        if pattern[-1] == "i":
            # Intersection pattern
            branches = int(pattern[0]) if pattern[0].isdigit() else 0
            analysis_parts = [
                f"all {no_result_count} attempts returned no results",
                f"query requires connecting {branches} different tables to find matching records",
            ]
            if num_filters > 0:
                analysis_parts.append(
                    f"but no records satisfy all {num_filters} filter conditions at the same time"
                )
            else:
                analysis_parts.append(
                    "but the table connections don't match any records in the data"
                )
        elif pattern[-1] == "p":
            # Path pattern
            hops = int(pattern[0]) if pattern[0].isdigit() else 0
            analysis_parts = [
                f"all {no_result_count} attempts returned no results",
                f"query requires following a {hops}-step path through {len(involved_tables)} connected tables",
            ]
            if num_filters > 0:
                analysis_parts.append(
                    f"but no records satisfy all {num_filters} filter conditions along this path"
                )
            else:
                analysis_parts.append(
                    "but this connection path doesn't match any records in the data"
                )
        else:
            # Direct query
            analysis_parts = [
                f"all {no_result_count} attempts returned no results",
            ]
            if num_filters > 0:
                analysis_parts.append(
                    f"no records satisfy all {num_filters} filter conditions"
                )

        return "; ".join(analysis_parts)

    def _determine_primary_reason_code(
        self,
        pattern: str,
        no_result_count: int,
        too_many_result_count: int,
        validation_failed_count: int,
        constraint_error_count: int,
        insufficient_filter_count: int,
        failure_reasons: List[str],
        gold_mismatch_count: int = 0,
    ) -> str:
        """Collapse detailed build failures into a stable shortfall reason code."""
        detailed_reasons = " ".join(failure_reasons).lower()
        counts = {
            "no_results": no_result_count,
            "too_many_results": too_many_result_count,
            "validation_failed": validation_failed_count,
            "constraint_error": constraint_error_count,
            "insufficient_filters": insufficient_filter_count,
            "gold_mismatch": gold_mismatch_count,
        }
        dominant = max(counts, key=counts.get)
        dominant_count = counts[dominant]

        if dominant_count <= 0:
            return "other"
        if dominant == "too_many_results":
            return "result_size_out_of_range"
        if dominant == "validation_failed":
            return "strategy_validation_failed"
        if dominant == "insufficient_filters":
            return "insufficient_filter_columns"
        if dominant == "gold_mismatch":
            return "gold_answer_mismatch"
        if dominant == "constraint_error":
            if "depth" in detailed_reasons or "self-ref" in detailed_reasons:
                return "depth_limits"
            return "other_constraint"
        if pattern.endswith(("p", "i")):
            return "sparse_combinations"
        return "no_valid_values"

    def _parse_strategy(self, strategy: str) -> Tuple[str, str]:
        """Parse strategy into pattern and difficulty.

        Args:
            strategy: Difficulty code (e.g., "2iM")

        Returns:
            Tuple of (pattern, difficulty) (e.g., ("2i", "M"))
        """
        if strategy[0].isdigit():
            if len(strategy) >= 2 and strategy[1] in ["p", "i"]:
                return strategy[:2], strategy[2:]
            else:
                return strategy[0], strategy[1:]
        else:
            return strategy[:2], strategy[2:]

    def _generate_join_structure(self, pattern: str) -> List[JoinPath]:
        """Generate JOIN structure based on pattern.

        Args:
            pattern: Pattern code (e.g., "2i", "1p")

        Returns:
            List of JoinPath objects
        """
        if pattern == "0":
            # No JOINs
            return []

        # Extract number and type
        if pattern[-1] == "p":
            # Path pattern (chain)
            hops = int(pattern[0])
            return self._generate_chain_joins(hops)
        elif pattern[-1] == "i":
            # Intersection pattern (star)
            branches = int(pattern[0])
            return self._generate_star_joins(branches)
        else:
            # Mixed or other patterns - not implemented yet
            raise NotImplementedError(f"Pattern {pattern} not yet implemented")

    def _generate_chain_joins(self, hops: int) -> List[JoinPath]:
        """Generate chain JOIN paths.

        Args:
            hops: Number of hops (JOINs)

        Returns:
            List of JoinPath objects representing the chain

        Raises:
            ValueError: If chain cannot be generated with detailed reason
        """
        # Try multiple attempts to build a valid chain
        max_attempts = 5
        failure_reasons = []
        no_fk_count = 0
        cycle_count = 0

        for attempt in range(max_attempts):
            try:
                # Build a chain from target table
                current_table = self.target_table
                path_tables = [current_table]
                # Track how many times each real table appears (for self-FK depth limit)
                path_table_counts: Dict[str, int] = {current_table: 1}

                for hop in range(hops):
                    # Use the real table name for FK lookup (strip alias suffix)
                    real_current = _real_table_name(current_table)

                    # Get foreign keys from current table
                    fks = self.schema.get_foreign_keys_for_table(
                        real_current, direction="outgoing"
                    )

                    if not fks:
                        # No outgoing FKs, try incoming
                        fks = self.schema.get_foreign_keys_for_table(
                            real_current, direction="incoming"
                        )

                    if not fks:
                        no_fk_count += 1
                        raise ValueError(
                            f"No foreign keys found for table {real_current} "
                            f"(at hop {hop + 1}/{hops})"
                        )

                    # Filter out FKs that would create cycles, but allow self-referential
                    # FKs up to _MAX_SELF_REF_DEPTH occurrences (for hierarchy traversal)
                    valid_fks = []
                    for fk in fks:
                        next_real = (
                            fk.parent_table
                            if fk.child_table == real_current
                            else fk.child_table
                        )
                        is_self_ref = fk.child_table == fk.parent_table
                        current_count = path_table_counts.get(next_real, 0)
                        table_meta = self.schema.tables.get(next_real)
                        max_depth = (
                            table_meta.self_ref_depth
                            if table_meta is not None
                            and table_meta.self_ref_depth is not None
                            else _MAX_SELF_REF_DEPTH
                        )
                        if current_count == 0 or (
                            is_self_ref and current_count < max_depth
                        ):
                            valid_fks.append(fk)

                    if not valid_fks:
                        cycle_count += 1
                        raise ValueError(
                            f"No valid FKs to continue chain from {real_current} "
                            f"(all would create cycles, at hop {hop + 1}/{hops})"
                        )

                    # Randomly select one valid FK
                    fk = random.choice(valid_fks)

                    # Determine the next real table name
                    if fk.child_table == real_current:
                        next_real = fk.parent_table
                    else:
                        next_real = fk.child_table

                    # If the real table already appears in the path, use an alias
                    existing_count = path_table_counts.get(next_real, 0)
                    if existing_count > 0:
                        next_table = f"{next_real}__{existing_count}"
                    else:
                        next_table = next_real

                    path_tables.append(next_table)
                    path_table_counts[next_real] = existing_count + 1
                    current_table = next_table

                # Successfully built a chain
                return [JoinPath(tables=path_tables, join_type="chain")]

            except ValueError as e:
                error_msg = str(e)
                if error_msg not in failure_reasons:
                    failure_reasons.append(error_msg)
                logger.debug(f"Chain generation attempt {attempt + 1} failed: {e}")
                continue

        # Build detailed error message in plain language
        reason_parts = []
        if no_fk_count > 0:
            reason_parts.append(
                f"cannot find enough table relationships to build a {hops}-hop chain "
                f"(tried {no_fk_count} times but no valid connections found)"
            )
        elif cycle_count > 0:
            reason_parts.append(
                f"all possible table connection paths would create circular references "
                f"(tried {cycle_count} different paths, all would loop back to previous tables)"
            )
        elif failure_reasons:
            # Show the most common failure reason
            most_common = max(set(failure_reasons), key=failure_reasons.count)
            if "No foreign keys" in most_common:
                reason_parts.append(
                    f"cannot find enough table relationships to build a {hops}-hop chain"
                )
            elif "cycles" in most_common:
                reason_parts.append(
                    "all possible table connection paths would create circular references"
                )
            else:
                reason_parts.append(most_common)

        reason_str = f" - {', '.join(reason_parts)}" if reason_parts else ""

        raise ValueError(f"Unable to generate {hops}-hop chain query{reason_str}")

    def _generate_star_joins(self, branches: int) -> List[JoinPath]:
        """Generate star JOIN paths (intersection pattern).

        Args:
            branches: Number of branches (JOINs from target table)

        Returns:
            List of JoinPath objects, each representing one branch
        """
        # Get all foreign keys involving the target table, excluding self-referential FKs
        # (self-joins don't produce meaningful intersection queries)
        fks = [
            fk
            for fk in self.schema.get_foreign_keys_for_table(
                self.target_table, direction="both"
            )
            if fk.child_table != fk.parent_table
        ]

        if len(fks) < branches:
            raise ValueError(
                f"Cannot generate {branches}-way intersection because target table "
                f"'{self.target_table}' has only {len(fks)} foreign key relationship(s), "
                f"but {branches} are required"
            )

        # Randomly select branches
        selected_fks = random.sample(fks, branches)

        join_paths = []
        for fk in selected_fks:
            # Determine the related table
            if fk.child_table == self.target_table:
                related_table = fk.parent_table
            else:
                related_table = fk.child_table

            # Each branch is a 2-table path
            join_paths.append(
                JoinPath(tables=[self.target_table, related_table], join_type="star")
            )

        return join_paths

    def _generate_filters(
        self,
        difficulty: str,
        join_paths: List[JoinPath],
        joined_df: Optional[pd.DataFrame] = None,
    ) -> List[Filter]:
        """Generate filter conditions based on difficulty level.

        Only uses semantic (filterable) columns — PKs, FK columns, and id-like
        columns are excluded to produce natural-sounding questions.

        When ``joined_df`` is provided (anchor-row sampling), one surviving row of
        the joined frame is sampled and every predicate is derived from that row's
        actual cell values. This guarantees the anchor row satisfies the full
        conjunction, so the query is non-empty by construction — unlike sampling
        each column's literal independently, whose joint-match probability is
        ~∏(1/cardinality) and produced the empty-result retry thrashing.

        Args:
            difficulty: Difficulty level (E/M/H)
            join_paths: List of JOIN paths
            joined_df: Pre-joined frame for anchor-row sampling (None = legacy
                independent sampling, kept for compatibility)

        Returns:
            List of Filter objects
        """
        # Determine number of filter columns based on difficulty
        min_cols, max_cols = self._get_filter_range(difficulty)

        num_filters = random.randint(min_cols, max_cols)

        # Get all involved tables (normalize to lowercase for table lookup)
        involved_tables = self._get_involved_tables(join_paths)
        if not involved_tables:
            involved_tables = [self.target_table]

        # Build pool of (table, column) candidates from pre-computed filterable columns
        candidate_pool: List[Tuple[str, str]] = []
        for table in involved_tables:
            real_table_lower = _real_table_name(table).lower()
            for col in self._filterable_columns.get(real_table_lower, []):
                candidate_pool.append((table, col))

        if not candidate_pool:
            return []

        # Anchor-row sampling: pick one surviving row of the joined frame
        anchor_row = None
        if joined_df is not None and len(joined_df) > 0:
            anchor_row = joined_df.iloc[random.randint(0, len(joined_df) - 1)]

        # Shuffle and pick distinct columns
        random.shuffle(candidate_pool)

        filters = []
        used_columns: Set[str] = set()

        for table, column in candidate_pool:
            if len(filters) >= num_filters:
                break
            key = f"{table}.{column}"
            if key in used_columns:
                continue
            used_columns.add(key)
            if anchor_row is not None:
                filter_obj = self._filter_from_anchor(table, column, anchor_row)
            else:
                filter_obj = self._generate_filter_condition(table, column)
            if filter_obj:
                filters.append(filter_obj)

        return filters

    def _filter_from_anchor(
        self, table: str, column: str, anchor_row: pd.Series
    ) -> Optional[Filter]:
        """Derive a filter predicate from the anchor row's actual cell value.

        The chosen operator always *includes* the anchor value (=, >=, <=, LIKE
        on a contained token), so the anchor row satisfies the predicate and the
        conjunction of all such predicates is guaranteed non-empty.

        Returns None when the anchor cell is NaN/empty or the value fails
        hygiene checks (e.g. leading/trailing whitespace that would break SQL
        round-tripping) — the caller then simply tries another column.
        """
        try:
            table_lower = _real_table_name(table).lower()
            column_lower = column.lower()
            col_name = self._joined_col_name(table, column_lower)
            if col_name not in anchor_row.index:
                return None
            value = anchor_row[col_name]
            if pd.isna(value):
                return None

            # Literal hygiene: skip values that would not round-trip through SQL
            if isinstance(value, str):
                if not value.strip():
                    return None
                if value != value.strip():
                    # Padded values ('us ') break equality matching downstream
                    return None

            col_type = self._column_types.get(table_lower, {}).get(
                column_lower, ColumnType.TEXT
            )

            operator: str

            if col_type in (ColumnType.NUMERIC, ColumnType.YEAR, ColumnType.DATE):
                # Anchor-inclusive comparison operators only
                operator = random.choice(["=", ">=", "<="])

            elif col_type in (ColumnType.ENUM, ColumnType.BOOLEAN, ColumnType.LABEL):
                # Exact match on the anchor value (labels: names/titles/cities)
                operator = "="

            elif col_type == ColumnType.COMMA_SEPARATED:
                tokens = [t.strip() for t in str(value).split(",") if t.strip()]
                if not tokens:
                    return None
                value = random.choice(tokens)
                operator = "LIKE"

            else:
                # TEXT / IDENTIFIER — should not reach here, but be safe
                return None

            return Filter(
                table=table,
                column=column_lower,
                operator=operator,
                value=value,
                column_type=col_type.value,
            )

        except Exception as e:
            logger.debug(f"Failed to derive anchor filter for {table}.{column}: {e}")
            return None

    def _generate_filter_condition(self, table: str, column: str) -> Optional[Filter]:
        """Generate a type-aware filter condition for a specific column.

        Uses the pre-inferred column type to select appropriate operators and values.

        Args:
            table: Table name
            column: Column name

        Returns:
            Filter object, or None if generation failed
        """
        try:
            # Strip alias suffix and normalize to lowercase for DataFrame lookup
            table_lower = _real_table_name(table).lower()
            df = self.tables[table_lower]
            column_lower = column.lower()
            if column_lower not in df.columns:
                return None
            values = df[column_lower].dropna()

            if len(values) == 0:
                return None

            # Look up inferred column type
            col_type = self._column_types.get(table_lower, {}).get(
                column_lower, ColumnType.TEXT
            )

            operator: str
            value: Any

            if col_type == ColumnType.NUMERIC:
                value = random.choice(values.tolist())
                operator = random.choice(["=", ">", "<", ">=", "<="])

            elif col_type == ColumnType.YEAR:
                value = random.choice(values.tolist())
                operator = random.choice(["=", ">=", "<="])

            elif col_type == ColumnType.DATE:
                value = random.choice(values.tolist())
                operator = random.choice(["=", ">=", "<="])

            elif col_type in (ColumnType.ENUM, ColumnType.LABEL):
                # Pick from the distinct observed values
                distinct = values.unique().tolist()
                value = random.choice(distinct)
                operator = "="

            elif col_type == ColumnType.BOOLEAN:
                distinct = values.unique().tolist()
                value = random.choice(distinct)
                operator = "="

            elif col_type == ColumnType.COMMA_SEPARATED:
                # Pick a random value and extract one comma-token
                raw_value = random.choice(values.tolist())
                tokens = [t.strip() for t in str(raw_value).split(",") if t.strip()]
                if not tokens:
                    return None
                value = random.choice(tokens)
                operator = "LIKE"

            else:
                # TEXT / IDENTIFIER — should not reach here, but be safe
                return None

            return Filter(
                table=table,
                column=column_lower,
                operator=operator,
                value=value,
                column_type=col_type.value,
            )

        except Exception as e:
            logger.debug(f"Failed to generate filter for {table}.{column}: {e}")
            return None

    def _get_involved_tables(self, join_paths: List[JoinPath]) -> List[str]:
        """Get all involved tables from JOIN paths.

        Args:
            join_paths: List of JOIN paths

        Returns:
            List of unique table names (always includes target_table)
        """
        tables = {self.target_table}  # Always include target table
        for path in join_paths:
            tables.update(path.tables)
        return list(tables)

    def _build_sql(self, join_paths: List[JoinPath], filters: List[Filter]) -> str:
        """Build SQL query from JOIN paths and filters.

        Args:
            join_paths: List of JOIN paths
            filters: List of filters

        Returns:
            SQL query string (all table and column names in lowercase)
        """
        # Convert table and column names to lowercase
        target_table_lower = self.target_table.lower()
        target_pk_lower = self.target_pk.lower()

        # SELECT clause - select primary key from target table
        sql = f"SELECT {target_table_lower}.{target_pk_lower} FROM {target_table_lower}"

        # JOIN clauses
        # ``joined_alias`` maps alias → real table for tracking what's been joined.
        # For plain tables the alias equals the real name; for repeated tables (self-FK)
        # the alias is e.g. ``theme__1`` while the real table is ``theme``.
        if join_paths:
            joined_aliases = {target_table_lower}
            for path in join_paths:
                for i in range(len(path.tables) - 1):
                    from_alias = path.tables[i].lower()
                    to_alias = path.tables[i + 1].lower()
                    to_real = _real_table_name(path.tables[i + 1]).lower()
                    from_real = _real_table_name(path.tables[i]).lower()

                    if to_alias in joined_aliases:
                        continue

                    # Find ALL FK column pairs (composite FKs must join on every pair)
                    fks = self._find_foreign_keys(path.tables[i], path.tables[i + 1])
                    if fks:
                        # Use AS alias when the real table name differs from the alias
                        join_clause = (
                            f"{to_real} AS {to_alias}"
                            if to_real != to_alias
                            else to_real
                        )
                        on_conditions = []
                        for fk in fks:
                            child_col_lower = fk.child_column.lower()
                            parent_col_lower = fk.parent_column.lower()
                            if fk.child_table.lower() == from_real:
                                on_conditions.append(
                                    f"{from_alias}.{child_col_lower} = {to_alias}.{parent_col_lower}"
                                )
                            else:
                                on_conditions.append(
                                    f"{from_alias}.{parent_col_lower} = {to_alias}.{child_col_lower}"
                                )
                        sql += f"\nJOIN {join_clause} ON " + " AND ".join(on_conditions)
                        joined_aliases.add(to_alias)

        # WHERE clause. NOTE: do NOT lowercase the rendered conditions — table and
        # column names are already lowercased inside Filter.to_sql(), and lowercasing
        # the whole string would corrupt string literals (e.g. 'Oxford' → 'oxford'),
        # silently diverging the SQL's result set from the stored answer_row_ids.
        if filters:
            where_conditions = [f.to_sql() for f in filters]
            sql += "\nWHERE " + " AND ".join(where_conditions)

        return sql

    def _find_foreign_key(self, table1: str, table2: str) -> Optional[ForeignKey]:
        """Find foreign key relationship between two tables.

        Args:
            table1: First table name (can be original case or lowercase)
            table2: Second table name (can be original case or lowercase)

        Returns:
            ForeignKey object, or None if not found
        """
        fks = self._find_foreign_keys(table1, table2)
        return fks[0] if fks else None

    def _find_foreign_keys(self, table1: str, table2: str) -> List[ForeignKey]:
        """Find ALL foreign-key column pairs between two tables.

        A composite FK is stored as multiple ForeignKey entries sharing the same
        (child_table, parent_table); joining on only the first pair produces
        wrong/empty joins, so callers must join on every returned pair.

        Args:
            table1: First table name (can be original case or lowercase)
            table2: Second table name (can be original case or lowercase)

        Returns:
            List of ForeignKey objects sharing the same direction as the first
            match (all child→parent the same way); empty list if none found.
        """
        # Strip alias suffixes (e.g. "Theme__1" → "Theme") before lookup
        table1_lower = _real_table_name(table1).lower()
        table2_lower = _real_table_name(table2).lower()

        matches: List[ForeignKey] = []
        for fk in self.schema.foreign_keys:
            fk_child_lower = fk.child_table.lower()
            fk_parent_lower = fk.parent_table.lower()
            if (fk_child_lower == table1_lower and fk_parent_lower == table2_lower) or (
                fk_child_lower == table2_lower and fk_parent_lower == table1_lower
            ):
                matches.append(fk)
        if not matches:
            return []
        # Keep only pairs with the same direction as the first match, so the
        # composite join columns line up consistently.
        first = matches[0]
        return [
            fk
            for fk in matches
            if fk.child_table.lower() == first.child_table.lower()
            and fk.parent_table.lower() == first.parent_table.lower()
        ]

    def _structure_cache_key(self, join_paths: List[JoinPath]) -> Tuple:
        """Stable cache key for a join structure."""
        return tuple((path.join_type, tuple(path.tables)) for path in join_paths)

    def _joined_col_name(self, table_alias: str, column: str) -> str:
        """Return the column name of ``table_alias.column`` inside a joined frame.

        All columns carry a ``_{alias}`` suffix; the only exception is the target
        table's PK, which is renamed back to its raw name after ``add_suffix``.
        """
        target_table_lower = self.target_table.lower()
        alias_lower = table_alias.lower()
        col_lower = column.lower()
        if alias_lower == target_table_lower and col_lower == self.target_pk.lower():
            return col_lower
        return f"{col_lower}_{alias_lower}"

    def _get_joined_frame(self, join_paths: List[JoinPath]) -> Optional[pd.DataFrame]:
        """Build (or fetch from cache) the fully joined frame for a structure.

        The same pre-enumerated structures are used across many attempts, so the
        joined frame is cached per structure. Returns None when the structure
        cannot be joined (missing FK/keys); an empty frame means the join is
        valid but yields no surviving rows.
        """
        cache_key = self._structure_cache_key(join_paths)
        if cache_key in self._joined_frame_cache:
            return self._joined_frame_cache[cache_key]

        frame = self._build_joined_frame(join_paths)
        # Admission control: never hold huge frames in memory — rebuild them
        # on demand instead (several cached multi-100k-row frames OOM workers).
        if frame is not None and len(frame) > self._JOINED_FRAME_CACHE_MAX_ROWS:
            return frame
        # Cap cache size; drop oldest entry (insertion order) when full.
        if len(self._joined_frame_cache) >= self._JOINED_FRAME_CACHE_MAX:
            self._joined_frame_cache.pop(next(iter(self._joined_frame_cache)))
        self._joined_frame_cache[cache_key] = frame
        return frame

    def _build_joined_frame(
        self, join_paths: List[JoinPath]
    ) -> Optional[pd.DataFrame]:
        """Execute the join structure on pandas DataFrames (no filters)."""
        try:
            # Start with target table (normalize to lowercase for lookup)
            target_table_lower = self.target_table.lower()
            result_df = self.tables[target_table_lower].copy()

            # Add suffix to avoid column name conflicts
            result_df = result_df.add_suffix(f"_{target_table_lower}")
            result_df = result_df.rename(
                columns={f"{self.target_pk}_{target_table_lower}": self.target_pk}
            )

            # ``joined_aliases`` tracks alias names (lowercase) that have been merged.
            # For plain tables alias == real name; for self-FK the alias is e.g. ``theme__1``.
            joined_aliases = {target_table_lower}

            # Perform JOINs
            for path in join_paths:
                for i in range(len(path.tables) - 1):
                    from_alias = path.tables[i].lower()
                    to_alias = path.tables[i + 1].lower()
                    to_real = _real_table_name(path.tables[i + 1]).lower()
                    from_real = _real_table_name(path.tables[i]).lower()

                    if to_alias in joined_aliases:
                        continue

                    # Find ALL FK column pairs (composite FKs join on every pair)
                    fks = self._find_foreign_keys(path.tables[i], path.tables[i + 1])
                    if not fks:
                        logger.debug(f"No FK found between {from_alias} and {to_alias}")
                        return None

                    # Load the real table data, suffix columns with the alias
                    join_df = self.tables[to_real].copy()
                    join_df = join_df.add_suffix(f"_{to_alias}")

                    target_pk_lower = self.target_pk.lower()

                    def _suffixed(col: str, alias: str) -> str:
                        """Return the suffixed column name, respecting the
                        target-PK rename that happened earlier."""
                        if alias == target_table_lower and col == target_pk_lower:
                            return col  # PK was renamed back
                        return f"{col}_{alias}"

                    left_on: List[str] = []
                    right_on: List[str] = []
                    for fk in fks:
                        child_col = fk.child_column.lower()
                        parent_col = fk.parent_column.lower()
                        if fk.child_table.lower() == from_real:
                            left_on.append(_suffixed(child_col, from_alias))
                            right_on.append(f"{parent_col}_{to_alias}")
                        else:
                            left_on.append(_suffixed(parent_col, from_alias))
                            right_on.append(f"{child_col}_{to_alias}")

                    if any(c not in result_df.columns for c in left_on) or any(
                        c not in join_df.columns for c in right_on
                    ):
                        logger.debug(
                            "Join keys missing for %s -> %s (%s, %s)",
                            from_alias,
                            to_alias,
                            left_on,
                            right_on,
                        )
                        return None

                    # Explosion guard — BEFORE the merge: estimate the exact
                    # inner-join row count from per-key frequencies. Checking
                    # after the merge is too late; the merge allocation itself
                    # OOM-killed workers (and the parent) on fan-out joins.
                    est_rows = self._estimate_merge_rows(
                        result_df, join_df, left_on, right_on
                    )
                    if est_rows > self._JOIN_MAX_ROWS:
                        logger.debug(
                            "Join structure would produce ~%s rows (> %s), aborting",
                            est_rows,
                            self._JOIN_MAX_ROWS,
                        )
                        return None

                    # Perform the join (on all composite-FK column pairs)
                    result_df = result_df.merge(
                        join_df, left_on=left_on, right_on=right_on, how="inner"
                    )

                    joined_aliases.add(to_alias)

            return result_df

        except Exception as e:
            logger.debug(f"Failed to build joined frame: {e}")
            return None

    @staticmethod
    def _estimate_merge_rows(
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        left_on: List[str],
        right_on: List[str],
    ) -> int:
        """Exact inner-join output size: sum over keys of count_l * count_r.

        Cheap (two value_counts + an index join) and computed BEFORE the merge
        so a fan-out join can be rejected without allocating its result.
        """
        try:
            if len(left_on) == 1:
                left_counts = left_df[left_on[0]].value_counts()
                right_counts = right_df[right_on[0]].value_counts()
            else:
                left_counts = left_df.groupby(left_on, dropna=True).size()
                right_counts = right_df.groupby(right_on, dropna=True).size()
                right_counts.index.names = left_counts.index.names
            common = left_counts.index.intersection(right_counts.index)
            if len(common) == 0:
                return 0
            return int((left_counts.loc[common] * right_counts.loc[common]).sum())
        except Exception:
            # Estimation must never break generation; fall back to a coarse
            # upper bound that still rejects pathological products.
            return (
                0
                if len(left_df) == 0 or len(right_df) == 0
                else min(len(left_df) * len(right_df), 10**12)
            )

    def _apply_filters(
        self, joined_df: pd.DataFrame, filters: List[Filter]
    ) -> Optional[pd.DataFrame]:
        """Apply filter conditions to a joined frame.

        Returns the filtered frame, or None when a filter column is missing
        (structurally invalid query).
        """
        result_df = joined_df
        # Apply filters (filter_obj.table and filter_obj.column are already lowercase)
        for filter_obj in filters:
            col_name = self._joined_col_name(filter_obj.table, filter_obj.column)

            # Make sure column exists
            if col_name not in result_df.columns:
                # Try without suffix (fallback)
                if filter_obj.column.lower() in result_df.columns:
                    col_name = filter_obj.column.lower()
                else:
                    logger.debug(
                        f"Column {col_name} not found in result, rejecting query"
                    )
                    return None

            # Apply filter based on operator
            if filter_obj.operator == "LIKE":
                # Pandas equivalent of SQLite LIKE '%value%' (ASCII case-insensitive)
                result_df = result_df[
                    result_df[col_name]
                    .astype(str)
                    .str.contains(
                        str(filter_obj.value), case=False, na=False, regex=False
                    )
                ]
            elif filter_obj.operator == "=":
                result_df = result_df[result_df[col_name] == filter_obj.value]
            elif filter_obj.operator == ">":
                result_df = result_df[result_df[col_name] > filter_obj.value]
            elif filter_obj.operator == "<":
                result_df = result_df[result_df[col_name] < filter_obj.value]
            elif filter_obj.operator == ">=":
                result_df = result_df[result_df[col_name] >= filter_obj.value]
            elif filter_obj.operator == "<=":
                result_df = result_df[result_df[col_name] <= filter_obj.value]

        return result_df

    def _extract_answer_ids(
        self, result_df: pd.DataFrame, max_results: Optional[int] = None
    ) -> List[Any]:
        """Extract distinct target-PK values from a (filtered) joined frame."""
        if self.target_pk in result_df.columns:
            ids = result_df[self.target_pk].dropna()
            unique_ids = ids.drop_duplicates()
            if max_results is not None:
                unique_ids = unique_ids.head(max_results)
            return unique_ids.tolist()
        logger.warning(f"Primary key {self.target_pk} not found in result")
        return []

    def _execute_query_from_spec(
        self,
        join_paths: List[JoinPath],
        filters: List[Filter],
        max_results: Optional[int] = None,
    ) -> List[Any]:
        """Execute query using join paths and filters on pandas DataFrames.

        Args:
            join_paths: List of JOIN paths
            filters: List of filter conditions
            max_results: If provided, return at most this many distinct IDs.

        Returns:
            List of target table row IDs
        """
        joined_df = self._get_joined_frame(join_paths)
        if joined_df is None:
            return []
        filtered = self._apply_filters(joined_df, filters)
        if filtered is None:
            return []
        return self._extract_answer_ids(filtered, max_results=max_results)

    def _validate_query(self, query_spec: QuerySpec) -> bool:
        """Validate that the generated query matches the strategy.

        Args:
            query_spec: QuerySpec object

        Returns:
            True if valid, False otherwise
        """
        # Create QueryPlan for classification
        filter_columns = set(query_spec.involved_columns)

        query_plan = QueryPlan(
            target_table=query_spec.target_table,
            join_paths=query_spec.join_paths,
            filter_columns=filter_columns,
        )

        # Classify the query
        classified_strategy = self.classifier.classify(query_plan)

        # Check if it matches the target strategy
        if classified_strategy != query_spec.strategy:
            logger.debug(
                f"Strategy mismatch: expected {query_spec.strategy}, "
                f"got {classified_strategy}"
            )
            return False

        # Check if we have valid answers
        if not query_spec.answer_row_ids or len(query_spec.answer_row_ids) == 0:
            logger.debug("Query returned no results")
            return False

        return True

    def _get_filter_range(self, difficulty: str) -> Tuple[int, int]:
        """Return min/max number of filters for a difficulty level."""
        if difficulty == "E":
            return 1, 2
        if difficulty == "M":
            return 3, 5
        if difficulty == "H":
            return 6, 8
        return 1, 2

    def validate_sql_execution(self, sql_query: str) -> Tuple[bool, Optional[str]]:
        """Validate that SQL query can be executed successfully.

        Args:
            sql_query: SQL query string to validate

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if SQL executed successfully, False otherwise
            - error_message: Error message if execution failed, None if successful
        """
        if not self.engine:
            # No engine available, skip validation
            logger.debug("No database engine available for SQL validation")
            return True, None

        try:
            # NOTE: do NOT lowercase the SQL here — table/column names are already
            # lowercase, and lowercasing the full statement corrupts string literals.
            # Try to execute the query with LIMIT 0 to avoid fetching data
            test_query = sql_query.rstrip(";")
            if "limit" not in test_query.lower():
                test_query += " limit 0"
            else:
                # Replace existing LIMIT with 0
                test_query = re.sub(
                    r"limit\s+\d+", "limit 0", test_query, flags=re.IGNORECASE
                )

            with self.engine.connect() as conn:
                conn.execute(text(test_query))
                logger.debug(f"SQL validation successful: {sql_query[:100]}...")
                return True, None

        except Exception as e:
            error_msg = str(e)
            logger.debug(f"SQL validation failed: {error_msg}")
            return False, error_msg

    @staticmethod
    def _normalize_gold_id(value: Any) -> Any:
        """Normalize an answer id for engine-vs-pandas comparison.

        SQLite may return ``1`` where pandas holds ``np.int64(1)`` or ``1.0``;
        normalize numeric-ish values to a canonical number, everything else to str.
        """
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (int, float)):
            f = float(value)
            return int(f) if f.is_integer() else round(f, 6)
        s = str(value)
        try:
            f = float(s)
            return int(f) if f.is_integer() else round(f, 6)
        except (ValueError, TypeError):
            return s

    def verify_gold_execution(
        self, query_spec: QuerySpec
    ) -> Tuple[bool, Optional[str]]:
        """Execute the stored SQL and assert it reproduces ``answer_row_ids``.

        The answer ids are computed by a pandas simulation while the SQL string
        is built independently; any semantic divergence between the two engines
        (LIKE case rules, type coercion, literal corruption) would otherwise ship
        as wrong ground truth. Accepted pairs carry the complete id set (results
        larger than ``max_answer_records`` are rejected earlier), so a full
        set-equality check is valid here.

        Returns:
            (True, None) when the SQL result matches (or no engine is available —
            callers relying on this gate should provide an engine);
            (False, reason) on mismatch or execution error.
        """
        if not self.engine:
            logger.debug("No database engine available for gold verification")
            return True, None

        try:
            with self.engine.connect() as conn:
                rows = conn.execute(text(query_spec.sql.rstrip(";"))).fetchall()
            sql_ids = {self._normalize_gold_id(row[0]) for row in rows if row[0] is not None}
            pandas_ids = {
                self._normalize_gold_id(v)
                for v in query_spec.answer_row_ids
                if v is not None
            }
            if sql_ids == pandas_ids:
                return True, None
            missing = list(pandas_ids - sql_ids)[:3]
            extra = list(sql_ids - pandas_ids)[:3]
            return False, (
                f"SQL result != stored answers "
                f"(sql_n={len(sql_ids)}, stored_n={len(pandas_ids)}, "
                f"missing_from_sql={missing}, extra_in_sql={extra})"
            )
        except Exception as e:
            return False, f"gold execution error: {e}"
