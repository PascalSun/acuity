"""Main QA generator class that coordinates all components.

Generates QA pairs based on difficulty strategies by:
1. Selecting strategies based on configured weights
2. Building SQL queries with appropriate JOINs and filters
3. Generating natural language questions using LLM
4. Extracting answer record IDs
5. Validating QA pairs
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy.engine import Engine

from talk2metadata.agent import AgentWrapper
from talk2metadata.core.qa.difficulty_classifier import DifficultyClassifier
from talk2metadata.core.qa.qa_pair import QAPair, _generate_uid
from talk2metadata.core.qa.query_builder import QueryBuilder
from talk2metadata.core.qa.question_generator import QuestionGenerator
from talk2metadata.core.qa.strategy_selector import StrategySelector
from talk2metadata.core.qa.verifier import QAVerifier
from talk2metadata.core.schema import SchemaMetadata
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


class QAGenerator:
    """Main class for generating QA pairs from database schema and data."""

    def __init__(
        self,
        schema: SchemaMetadata,
        tables: Dict[str, pd.DataFrame],
        agent: Optional[AgentWrapper] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        strategy_weights: Optional[Dict[str, int]] = None,
        tier_weights: Optional[Dict[str, int]] = None,
        feasible_strategies: Optional[list] = None,
        max_answer_records: int = 10,
        engine: Optional[Engine] = None,
        connection_string: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """Initialize QA generator.

        Args:
            schema: Schema metadata
            tables: Dictionary mapping table names to DataFrames
            agent: Optional AgentWrapper instance (shared across components)
            provider: LLM provider name (if agent is None)
            model: LLM model name (if agent is None)
            strategy_weights: Optional weights for specific strategies
            tier_weights: Optional weights for tiers
            feasible_strategies: If set, only use these strategies (from schema analysis)
            max_answer_records: Maximum number of answer records per question (default: 10)
                                Questions with more records are considered too general
            engine: Optional SQLAlchemy engine for SQL validation
            connection_string: Optional database connection string for SQL validation
            seed: Optional RNG seed for reproducible generation (seeds the global
                  ``random`` module; derive a per-DB seed in batch runs)
        """
        if seed is not None:
            random.seed(seed)
        self.seed = seed
        self.schema = schema
        self.tables = tables
        self.target_table = schema.target_table

        # Initialize agent
        if agent is None:
            agent = AgentWrapper(provider=provider, model=model)
        self.agent = agent

        # Try to get database connection for SQL validation
        if not engine and not connection_string:
            try:
                config = get_config()
                ingest_config = config.get("ingest", {})
                data_type = ingest_config.get("data_type", "csv")
                source_path = ingest_config.get("source_path")

                if data_type in ("database", "db") and source_path:
                    connection_string = source_path
                    logger.info(
                        "Using database connection from config for SQL validation"
                    )
                elif data_type == "csv":
                    # Try to get SQLite database created from CSV
                    from talk2metadata.utils.csv_to_db import (
                        get_or_create_db_connection,
                    )

                    try:
                        connection_string = get_or_create_db_connection(
                            ingest_config, schema
                        )
                        logger.info("Using SQLite database from CSV for SQL validation")
                    except Exception as e:
                        logger.debug(
                            f"Could not create database for SQL validation: {e}"
                        )
            except Exception as e:
                logger.debug(
                    f"Could not get database connection for SQL validation: {e}"
                )

        # Initialize components
        self.classifier = DifficultyClassifier()
        self.query_builder = QueryBuilder(
            schema,
            tables,
            max_answer_records=max_answer_records,
            engine=engine,
            connection_string=connection_string,
        )
        # Quota must be allocated only over strategies the schema/data can
        # actually support; otherwise every quota unit parked on an infeasible
        # strategy is a permanent shortfall (previously ~70% of the target).
        if feasible_strategies is None:
            feasible_strategies = self.query_builder.get_feasible_strategies()
        self.strategy_selector = StrategySelector(
            strategy_weights=strategy_weights,
            tier_weights=tier_weights,
            allowed_strategies=feasible_strategies,
        )
        self.question_generator = QuestionGenerator(agent, schema)
        self.verifier = QAVerifier(agent, max_answer_records=max_answer_records)
        self._last_generation_report: Optional[Dict[str, Any]] = None

    def generate(
        self,
        total_qa_pairs: int = 100,
        pairs_per_strategy: Optional[int] = None,
        validate: bool = True,
        filter_valid: bool = True,
    ) -> List[QAPair]:
        """Generate QA pairs based on difficulty strategies.

        Args:
            total_qa_pairs: Total number of QA pairs to generate
            pairs_per_strategy: If specified, generate this many pairs per strategy
                              (overrides total_qa_pairs and weights)
            validate: Whether to validate QA pairs
            filter_valid: Whether to filter out invalid QA pairs

        Returns:
            List of QAPair objects
        """
        logger.info(
            f"Generating QA pairs: total={total_qa_pairs}, "
            f"pairs_per_strategy={pairs_per_strategy}"
        )

        # Step 1: Get exact target counts per strategy (proportional to weights)
        logger.info("Step 1: Computing target counts per strategy...")
        if pairs_per_strategy is not None:
            target_counts = {
                s: pairs_per_strategy
                for s in self.strategy_selector.final_weights
                if self.strategy_selector.final_weights.get(s, 0) > 0
            }
            effective_total = sum(target_counts.values())
        else:
            target_counts = self.strategy_selector.get_target_quotas(total_qa_pairs)
            effective_total = total_qa_pairs

        # Step 1.5: Check which strategies are feasible via pre-enumerated structures
        feasible_set = set(self.query_builder.get_feasible_strategies())
        infeasible = [
            s for s in target_counts if target_counts[s] > 0 and s not in feasible_set
        ]
        if infeasible:
            logger.info(
                f"Skipping {len(infeasible)} infeasible strategies "
                f"(no valid join structures): {', '.join(sorted(infeasible))}"
            )

        # Step 2+3+4: Generate per strategy until we have exactly target VALID pairs each.
        # Uses structure_index for round-robin diversity across join structures.
        logger.info("Step 2: Generating QA pairs per strategy until target...")
        collected: Dict[str, List[QAPair]] = {
            s: [] for s in target_counts if target_counts[s] > 0
        }
        strategy_reports: Dict[str, Dict[str, Any]] = {}
        # Intra-batch dedup: previously duplicates were only removed against a
        # pre-existing file at save() time, so a run could accept the same
        # SQL/question repeatedly (observed on tiny schemas).
        seen_sql: set = set()
        seen_questions: set = set()

        def _normalize_question(text: str) -> str:
            return " ".join(text.lower().split()).rstrip("?").strip()

        for strategy in sorted(target_counts.keys()):
            target = target_counts[strategy]
            shortfall_reason_counts: Dict[str, int] = {}
            if target > 0 and strategy not in feasible_set:
                shortfall_reason_counts["pattern_infeasible"] = target
            strategy_reports[strategy] = {
                "target_quota": target,
                "accepted_count": 0,
                "shortfall": target,
                "fulfillment_rate": 0.0,
                "is_feasible": strategy in feasible_set,
                "outer_attempts": 0,
                "inner_attempts": 0,
                "failed_rounds": 0,
                "invalid_pairs_filtered": 0,
                "failure_events": {},
                "shortfall_reason_counts": shortfall_reason_counts,
                "primary_shortfall_reason": (
                    "pattern_infeasible"
                    if target > 0 and strategy not in feasible_set
                    else None
                ),
                "last_failure_summary": None,
            }

        for strategy in sorted(target_counts.keys()):
            target = target_counts[strategy]
            if target <= 0:
                continue
            if strategy not in feasible_set:
                continue  # Skip pre-determined infeasible strategies
            strategy_report = strategy_reports[strategy]
            give_up_at_0 = (
                target * 12
            )  # Stop if still 0 after this (strategy likely infeasible)
            keep_going_limit = (
                target * 80
            )  # When we have 1+, allow many more to hit target
            attempts = 0
            structure_counter = 0  # Round-robin through join structures for diversity
            while len(collected[strategy]) < target:
                # If we have 0 valid after give_up_at_0 attempts: forgive and stop
                if len(collected[strategy]) == 0 and attempts >= give_up_at_0:
                    logger.warning(
                        f"Strategy {strategy}: no valid pairs after {attempts} attempts, skipping"
                    )
                    break
                # If we have 1+ valid: we must hit target; only stop at keep_going_limit
                if len(collected[strategy]) > 0 and attempts >= keep_going_limit:
                    logger.warning(
                        f"Strategy {strategy}: got {len(collected[strategy])}/{target} "
                        f"after {attempts} attempts, continuing would exceed limit"
                    )
                    break
                try:
                    strategy_report["outer_attempts"] += 1
                    query_spec = self.query_builder.build_query(
                        strategy, structure_index=structure_counter
                    )
                    build_diag = self.query_builder.get_last_build_diagnostics()
                    strategy_report["inner_attempts"] += build_diag.get(
                        "attempts_used", 0
                    )
                    structure_counter += 1
                    if not query_spec:
                        self._record_failure_event(
                            strategy_report,
                            self._map_build_reason_to_shortfall_reason(build_diag),
                            build_diag.get("summary"),
                        )
                        attempts += 1
                        continue
                    # Reject duplicate SQL before spending an LLM call on it
                    if query_spec.sql in seen_sql:
                        self._record_failure_event(
                            strategy_report,
                            "duplicate_pair",
                            "identical SQL already generated in this run",
                        )
                        attempts += 1
                        continue
                    question = self.question_generator.generate(query_spec)
                    sql_valid, sql_error = self.query_builder.validate_sql_execution(
                        query_spec.sql
                    )
                    qa_pair = QAPair(
                        question=question,
                        answer_row_ids=query_spec.answer_row_ids,
                        sql=query_spec.sql,
                        strategy=query_spec.strategy,
                        difficulty_score=self.classifier.get_score(query_spec.strategy),
                        involved_tables=query_spec.involved_tables,
                        involved_columns=query_spec.involved_columns,
                        involved_filters=[f.to_dict() for f in query_spec.filters],
                        sql_valid=sql_valid,
                        sql_validation_error=sql_error,
                        metadata={"target_table": self.target_table},
                        answer_table=query_spec.target_table,
                        answer_id_column=query_spec.answer_id_column,
                        uid=_generate_uid(),
                    )
                    if validate:
                        self.verifier.verify(qa_pair)
                    do_add = (
                        not filter_valid
                        or (validate and qa_pair.is_valid)
                        or (not validate)
                    )
                    if do_add and _normalize_question(qa_pair.question) in seen_questions:
                        self._record_failure_event(
                            strategy_report,
                            "duplicate_pair",
                            "near-identical question already generated in this run",
                        )
                        do_add = False
                        attempts += 1
                        continue
                    if do_add:
                        seen_sql.add(qa_pair.sql)
                        seen_questions.add(_normalize_question(qa_pair.question))
                        collected[strategy].append(qa_pair)
                        if len(collected[strategy]) % 5 == 0:
                            logger.debug(
                                f"  {strategy}: {len(collected[strategy])}/{target}"
                            )
                    else:
                        self._record_failure_event(
                            strategy_report,
                            "qa_validation_failed",
                            "QA pair failed verifier and was filtered out",
                        )
                        strategy_report["invalid_pairs_filtered"] += 1
                except Exception as e:
                    self._record_failure_event(
                        strategy_report, "generation_exception", str(e)
                    )
                    logger.warning(f"Failed for {strategy}: {e}")
                attempts += 1

        # Build final list: exactly target per strategy
        qa_pairs = []
        for strategy in sorted(target_counts.keys()):
            target = target_counts[strategy]
            available = collected.get(strategy, [])
            qa_pairs.extend(available[:target])
            strategy_report = strategy_reports[strategy]
            accepted_count = len(available[:target])
            shortfall = max(target - accepted_count, 0)
            strategy_report["accepted_count"] = accepted_count
            strategy_report["shortfall"] = shortfall
            strategy_report["fulfillment_rate"] = (
                accepted_count / target if target > 0 else 1.0
            )
            if shortfall > 0 and not strategy_report["shortfall_reason_counts"]:
                primary_reason = self._select_primary_shortfall_reason(strategy_report)
                strategy_report["primary_shortfall_reason"] = primary_reason
                strategy_report["shortfall_reason_counts"] = {primary_reason: shortfall}
            elif shortfall == 0:
                strategy_report["shortfall_reason_counts"] = {}
                strategy_report["primary_shortfall_reason"] = None
            elif not strategy_report["primary_shortfall_reason"]:
                strategy_report["primary_shortfall_reason"] = (
                    self._select_primary_shortfall_reason(strategy_report)
                )

        realized_counts = {
            strategy: report["accepted_count"]
            for strategy, report in strategy_reports.items()
            if report["accepted_count"] > 0
        }
        shortfalls = {
            strategy: report["shortfall"]
            for strategy, report in strategy_reports.items()
            if report["shortfall"] > 0
        }
        overall_shortfall_reason_counts: Dict[str, int] = {}
        for report in strategy_reports.values():
            for reason, count in report["shortfall_reason_counts"].items():
                overall_shortfall_reason_counts[reason] = (
                    overall_shortfall_reason_counts.get(reason, 0) + count
                )

        self._last_generation_report = {
            "generation_mode": "quota_guided",
            "target_table": self.target_table,
            "pairs_per_strategy": pairs_per_strategy,
            "target_total": effective_total,
            "realized_total": len(qa_pairs),
            "shortfall_total": max(effective_total - len(qa_pairs), 0),
            "overall_fulfillment_rate": (
                len(qa_pairs) / effective_total if effective_total > 0 else 1.0
            ),
            "feasible_strategies": sorted(feasible_set),
            "requested_strategies": sorted(target_counts.keys()),
            "infeasible_requested_strategies": sorted(infeasible),
            "target_quotas": target_counts,
            "realized_counts": realized_counts,
            "shortfalls": shortfalls,
            "shortfall_reason_counts": overall_shortfall_reason_counts,
            "strategy_reports": strategy_reports,
        }

        logger.info(
            f"Generated {len(qa_pairs)} valid QA pairs (target {effective_total})"
        )
        return qa_pairs

    def _record_failure_event(
        self,
        strategy_report: Dict[str, Any],
        reason: str,
        summary: Optional[str] = None,
    ) -> None:
        strategy_report["failed_rounds"] += 1
        failure_events = strategy_report["failure_events"]
        failure_events[reason] = failure_events.get(reason, 0) + 1
        if summary:
            strategy_report["last_failure_summary"] = summary

    def _map_build_reason_to_shortfall_reason(
        self, build_diag: Optional[Dict[str, Any]]
    ) -> str:
        if not build_diag:
            return "other"
        code = build_diag.get("primary_reason_code")
        if code in {
            "pattern_infeasible",
            "no_valid_values",
            "sparse_combinations",
            "result_size_out_of_range",
            "depth_limits",
            "insufficient_filter_columns",
            "strategy_validation_failed",
        }:
            return code
        if code == "other_constraint":
            return "other_constraint"
        return "other"

    def _select_primary_shortfall_reason(self, strategy_report: Dict[str, Any]) -> str:
        if not strategy_report["is_feasible"]:
            return "pattern_infeasible"
        failure_events = strategy_report.get("failure_events", {})
        if failure_events:
            return max(failure_events.items(), key=lambda item: item[1])[0]
        if strategy_report.get("invalid_pairs_filtered", 0) > 0:
            return "qa_validation_failed"
        return "attempt_budget_exhausted"

    def get_last_generation_report(self) -> Optional[Dict[str, Any]]:
        """Return diagnostics for the most recent generate() call."""
        if self._last_generation_report is None:
            return None
        return json.loads(json.dumps(self._last_generation_report))

    def save_generation_report(
        self,
        output_path: Path | str,
    ) -> Optional[Path]:
        """Persist the most recent generation report next to QA outputs."""
        if self._last_generation_report is None:
            return None

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self._last_generation_report, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved generation report to {output_path}")
        return output_path

    def save(
        self,
        qa_pairs: List[QAPair],
        output_path: Optional[Path | str] = None,
        auto_save: bool = True,
        run_id: Optional[str] = None,
    ) -> Path:
        """Save QA pairs to file.

        If the output file already exists, loads existing QA pairs and merges
        them with the new ones, removing duplicates based on SQL query.

        Args:
            qa_pairs: List of QA pairs to save
            output_path: Optional explicit path to save QA pairs
            auto_save: If True and output_path is None, auto-save to qa/qa_pairs.json
            run_id: Optional run ID for auto-save path

        Returns:
            Path where QA pairs were saved
        """
        if output_path is None and auto_save:
            # Auto-save to qa/qa_pairs.json in run directory
            from talk2metadata.utils.paths import get_qa_dir

            qa_dir = get_qa_dir(run_id)
            qa_dir.mkdir(parents=True, exist_ok=True)
            output_path = qa_dir / "qa_pairs.json"
        elif output_path is None:
            raise ValueError(
                "Either output_path must be provided or auto_save must be True"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists and load existing QA pairs
        existing_qa_pairs = []
        if output_path.exists():
            try:
                existing_qa_pairs = self.load(output_path)
                logger.info(
                    f"Found existing QA pairs file with {len(existing_qa_pairs)} pairs, merging..."
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load existing QA pairs from {output_path}: {e}. "
                    "Will create new file."
                )
                existing_qa_pairs = []

        # Ensure every QA pair (existing and new) has a uid
        for qa in existing_qa_pairs:
            qa.ensure_uid()
        for qa in qa_pairs:
            qa.ensure_uid()

        # Merge existing and new QA pairs, removing duplicates by uid then SQL
        if existing_qa_pairs:
            existing_uids = {qa.uid for qa in existing_qa_pairs}
            existing_sqls = {qa.sql for qa in existing_qa_pairs}

            new_qa_pairs = [
                qa
                for qa in qa_pairs
                if qa.uid not in existing_uids and qa.sql not in existing_sqls
            ]

            if len(new_qa_pairs) < len(qa_pairs):
                logger.info(
                    f"Removed {len(qa_pairs) - len(new_qa_pairs)} duplicate QA pairs "
                    f"(based on uid/SQL query)"
                )

            # Combine existing and new (non-duplicate) QA pairs
            merged_qa_pairs = existing_qa_pairs + new_qa_pairs
            logger.info(
                f"Merged {len(existing_qa_pairs)} existing + {len(new_qa_pairs)} new = "
                f"{len(merged_qa_pairs)} total QA pairs"
            )
        else:
            merged_qa_pairs = qa_pairs

        # Compute statistics from merged QA pairs
        total_qa_pairs = len(merged_qa_pairs)
        valid_qa_pairs = sum(1 for qa in merged_qa_pairs if qa.is_valid)

        # Group by strategy
        strategy_distribution = {}
        for qa in merged_qa_pairs:
            strategy_distribution[qa.strategy] = (
                strategy_distribution.get(qa.strategy, 0) + 1
            )

        # Group by tier
        tier_distribution = {}
        for qa in merged_qa_pairs:
            tier = qa.tier
            tier_distribution[tier] = tier_distribution.get(tier, 0) + 1

        data = {
            "target_table": self.target_table,
            "total_qa_pairs": total_qa_pairs,
            "valid_qa_pairs": valid_qa_pairs,
            "strategy_distribution": strategy_distribution,
            "tier_distribution": tier_distribution,
            "qa_pairs": [qa.to_dict() for qa in merged_qa_pairs],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Write CSV alongside JSON for easy review (same directory, .csv extension)
        csv_path = output_path.with_suffix(".csv")
        self._save_qa_pairs_csv(merged_qa_pairs, csv_path)

        logger.info(f"Saved {len(merged_qa_pairs)} QA pairs to {output_path}")
        return output_path

    def _save_qa_pairs_csv(self, qa_pairs: List[QAPair], csv_path: Path | str) -> None:
        """Write QA pairs to a CSV file for review (flat columns, one row per pair)."""
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for qa in qa_pairs:
            # Build involved_values (table.column -> value) from involved_filters, same as to_dict
            involved_values = {}
            for f in qa.involved_filters:
                table = f.get("table")
                column = f.get("column")
                if table and column and "value" in f:
                    involved_values[f"{table}.{column}"] = f.get("value")
            involved_values_str = (
                "; ".join(f"{k}: {v}" for k, v in involved_values.items())
                if involved_values
                else ""
            )

            rows.append(
                {
                    "uid": qa.ensure_uid(),
                    "question": qa.question,
                    "sql": qa.sql,
                    "strategy": qa.strategy,
                    "tier": qa.tier,
                    "difficulty_score": qa.difficulty_score,
                    "answer_count": qa.answer_count,
                    "is_valid": qa.is_valid,
                    "validation_errors": (
                        "; ".join(qa.validation_errors) if qa.validation_errors else ""
                    ),
                    "sql_valid": qa.sql_valid,
                    "sql_validation_error": qa.sql_validation_error or "",
                    "involved_tables": ", ".join(qa.involved_tables),
                    "involved_columns": ", ".join(qa.involved_columns),
                    "involved_values": involved_values_str,
                }
            )
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info(f"Saved QA pairs review CSV to {csv_path}")

    @classmethod
    def load(cls, qa_pairs_path: Path | str) -> List[QAPair]:
        """Load QA pairs from file.

        Args:
            qa_pairs_path: Path to QA pairs JSON file

        Returns:
            List of QAPair objects
        """
        qa_pairs_path = Path(qa_pairs_path)
        with open(qa_pairs_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        qa_pairs = [QAPair.from_dict(qa) for qa in data["qa_pairs"]]
        logger.info(f"Loaded {len(qa_pairs)} QA pairs from {qa_pairs_path}")
        return qa_pairs

    @classmethod
    def from_schema_file(
        cls,
        schema_path: Path | str,
        tables: Dict[str, pd.DataFrame],
        agent: Optional[AgentWrapper] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        strategy_weights: Optional[Dict[str, int]] = None,
        tier_weights: Optional[Dict[str, int]] = None,
        max_answer_records: int = 10,
        engine: Optional[Engine] = None,
        connection_string: Optional[str] = None,
    ) -> QAGenerator:
        """Create QAGenerator from schema file.

        Args:
            schema_path: Path to schema JSON file
            tables: Dictionary mapping table names to DataFrames
            agent: Optional AgentWrapper instance
            provider: LLM provider name (if agent is None)
            model: LLM model name (if agent is None)
            strategy_weights: Optional weights for specific strategies
            tier_weights: Optional weights for tiers
            max_answer_records: Maximum number of answer records per question (default: 10)
            engine: Optional SQLAlchemy engine for SQL validation
            connection_string: Optional database connection string for SQL validation

        Returns:
            QAGenerator instance
        """
        schema = SchemaMetadata.load(schema_path)
        return cls(
            schema,
            tables,
            agent=agent,
            provider=provider,
            model=model,
            strategy_weights=strategy_weights,
            tier_weights=tier_weights,
            max_answer_records=max_answer_records,
            engine=engine,
            connection_string=connection_string,
        )
