"""Schema detection and foreign key inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from talk2metadata.core.schema.fk_detector_agent import AgentBasedFKDetector
from talk2metadata.core.schema.fk_detector_rule import RuleBasedFKDetector
from talk2metadata.core.schema.types import ForeignKey, TableMetadata
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SchemaMetadata:
    """Complete schema metadata."""

    tables: dict[str, TableMetadata]
    foreign_keys: list[ForeignKey]
    target_table: str

    def save(self, path: str | Path) -> None:
        """Save metadata to JSON file.

        Args:
            path: Path to save JSON file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved schema metadata to {path}")

    @classmethod
    def load(cls, path: str | Path) -> SchemaMetadata:
        """Load metadata from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            SchemaMetadata instance
        """
        with open(path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "tables": {
                name: {
                    "columns": meta.columns,
                    "primary_key": meta.primary_key,
                    "row_count": meta.row_count,
                    "sample_values": meta.sample_values,
                    "description": meta.description,
                    "column_descriptions": meta.column_descriptions,
                    **(
                        {"self_ref_depth": meta.self_ref_depth}
                        if meta.self_ref_depth is not None
                        else {}
                    ),
                    **(
                        {"non_filterable_columns": meta.non_filterable_columns}
                        if meta.non_filterable_columns
                        else {}
                    ),
                    **(
                        {"column_types": meta.column_types} if meta.column_types else {}
                    ),
                }
                for name, meta in self.tables.items()
            },
            "foreign_keys": [
                {
                    "child_table": fk.child_table,
                    "child_column": fk.child_column,
                    "parent_table": fk.parent_table,
                    "parent_column": fk.parent_column,
                    "coverage": fk.coverage,
                }
                for fk in self.foreign_keys
            ],
            "target_table": self.target_table,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SchemaMetadata:
        """Create from dictionary.

        Args:
            data: Dictionary with schema data

        Returns:
            SchemaMetadata instance
        """
        tables = {
            name: TableMetadata(
                name=name,
                columns=meta["columns"],
                primary_key=meta.get("primary_key"),
                row_count=meta.get("row_count", 0),
                sample_values=meta.get("sample_values", {}),
                description=meta.get("description"),
                column_descriptions=meta.get("column_descriptions", {}),
                self_ref_depth=meta.get("self_ref_depth"),
                non_filterable_columns=meta.get("non_filterable_columns", []),
                column_types=meta.get("column_types", {}),
            )
            for name, meta in data["tables"].items()
        }

        foreign_keys = [
            ForeignKey(
                child_table=fk["child_table"],
                child_column=fk["child_column"],
                parent_table=fk["parent_table"],
                parent_column=fk["parent_column"],
                coverage=fk["coverage"],
            )
            for fk in data["foreign_keys"]
        ]

        return cls(
            tables=tables,
            foreign_keys=foreign_keys,
            target_table=data["target_table"],
        )

    def get_related_tables(self, table_name: str) -> list[str]:
        """Get all tables related to the given table via foreign keys.

        Args:
            table_name: Table name

        Returns:
            List of related table names
        """
        related = set()

        for fk in self.foreign_keys:
            if fk.child_table == table_name:
                related.add(fk.parent_table)
            elif fk.parent_table == table_name:
                related.add(fk.child_table)

        return list(related)

    def get_foreign_keys_for_table(
        self, table_name: str, direction: str = "both"
    ) -> list[ForeignKey]:
        """Get foreign keys involving a table.

        Args:
            table_name: Table name
            direction: 'outgoing' (child), 'incoming' (parent), or 'both'

        Returns:
            List of ForeignKey objects
        """
        if direction == "outgoing":
            return [fk for fk in self.foreign_keys if fk.child_table == table_name]
        elif direction == "incoming":
            return [fk for fk in self.foreign_keys if fk.parent_table == table_name]
        else:  # both
            return [
                fk
                for fk in self.foreign_keys
                if fk.child_table == table_name or fk.parent_table == table_name
            ]

    def __repr__(self) -> str:
        return (
            f"SchemaMetadata(tables={len(self.tables)}, "
            f"fks={len(self.foreign_keys)}, target={self.target_table})"
        )


class SchemaDetector:
    """Schema detection with foreign key inference."""

    def __init__(self, config: dict | None = None):
        """Initialize schema detector.

        Args:
            config: Configuration dict (uses global config if None)
        """
        self.config = config or get_config().get("schema", {})
        self.fk_config = self.config.get("fk_detection", {})

        # Initialize FK detectors
        self.rule_detector = RuleBasedFKDetector(self.fk_config)
        self.agent_detector = AgentBasedFKDetector(self.fk_config)

    def detect(
        self,
        tables: dict[str, pd.DataFrame],
        target_table: str,
    ) -> SchemaMetadata:
        """Detect schema and infer foreign keys.

        Args:
            tables: Dict mapping table_name -> DataFrame
            target_table: Name of the target table

        Returns:
            SchemaMetadata object

        Example:
            >>> tables = {"orders": orders_df, "customers": customers_df}
            >>> detector = SchemaDetector()
            >>> metadata = detector.detect(tables, target_table="orders")
        """
        logger.info(f"Detecting schema for {len(tables)} tables")

        if target_table not in tables:
            raise ValueError(
                f"Target table '{target_table}' not found in tables: {list(tables.keys())}"
            )

        # 1. Extract table metadata
        table_metadata = self._extract_table_metadata(tables)

        # 2. Detect foreign keys
        logger.info("Inferring foreign keys from data")
        fks = self._detect_foreign_keys(tables, table_metadata, target_table)

        logger.info(f"Detected {len(fks)} foreign key relationships")
        for fk in fks:
            logger.debug(f"  {fk}")

        return SchemaMetadata(
            tables=table_metadata,
            foreign_keys=fks,
            target_table=target_table,
        )

    def _extract_table_metadata(
        self, tables: dict[str, pd.DataFrame]
    ) -> dict[str, TableMetadata]:
        """Extract metadata from tables.

        Args:
            tables: Dict of DataFrames

        Returns:
            Dict of TableMetadata objects
        """
        metadata = {}

        for name, df in tables.items():
            # Collect sample values for each column (first 3 non-null values)
            sample_values = {}
            for col in df.columns:
                non_null = df[col].dropna().head(3).astype(str).tolist()
                if non_null:
                    sample_values[col] = non_null

            metadata[name] = TableMetadata(
                name=name,
                columns={col: str(dtype) for col, dtype in df.dtypes.items()},
                primary_key=self._infer_primary_key(df),
                row_count=len(df),
                sample_values=sample_values,
            )

            logger.debug(
                f"Table {name}: {len(df.columns)} columns, {len(df)} rows, "
                f"PK={metadata[name].primary_key}"
            )

        return metadata

    def _infer_primary_key(self, df: pd.DataFrame) -> str | None:
        """Infer primary key column.

        Args:
            df: DataFrame

        Returns:
            Primary key column name or None
        """
        # Priority 1: Column named 'id'
        if "id" in df.columns and df["id"].is_unique and not df["id"].isna().any():
            return "id"

        # Priority 2: Column ending with '_id' that is unique
        for col in df.columns:
            if col.endswith("_id") and df[col].is_unique and not df[col].isna().any():
                return col

        # Priority 3: Any unique column without nulls
        for col in df.columns:
            if df[col].is_unique and not df[col].isna().any():
                return col

        return None

    def _detect_foreign_keys(
        self,
        tables: dict[str, pd.DataFrame],
        table_metadata: dict[str, TableMetadata],
        target_table: str,
    ) -> list[ForeignKey]:
        """Detect foreign key relationships using two-stage strategy.

        Strategy:
        1. Rule-based detection: Fast heuristic-based detection (always runs first)
        2. Agent-based detection: AI-powered analysis (always runs after rule-based)
        3. Final output: Agent-based results (if available), otherwise rule-based

        Args:
            tables: Dict of DataFrames
            table_metadata: Dict of TableMetadata
            target_table: Name of the target table (for deduplication priority)

        Returns:
            List of ForeignKey objects
        """
        # Stage 1: Rule-based detection (always runs first)
        logger.info("Stage 1: Running rule-based FK detection...")
        rule_based_fks = self.rule_detector.detect(tables, table_metadata, target_table)
        logger.info(f"Rule-based detection found {len(rule_based_fks)} FKs")

        # Stage 2: Agent-based detection (always runs after rule-based)
        logger.info("Stage 2: Running agent-based FK detection...")
        agent_fks = self.agent_detector.detect(
            tables, table_metadata, target_table, rule_based_fks
        )
        logger.info(f"Agent-based detection found {len(agent_fks)} FKs")

        # Stage 3: Use agent-based results as final output
        # If agent found FKs, use them; otherwise fall back to rule-based
        if agent_fks:
            final_fks = agent_fks
            logger.info(
                f"Using agent-based results as final output "
                f"(rule-based: {len(rule_based_fks)}, agent: {len(agent_fks)})"
            )
        else:
            final_fks = rule_based_fks
            logger.info(
                "Agent-based detection returned no results, using rule-based results"
            )

        # Deduplicate and prioritize FKs
        final_fks = self._deduplicate_fks(final_fks, table_metadata, target_table)

        logger.info(f"Final FKs detected: {len(final_fks)}")

        return final_fks

    def _deduplicate_fks(
        self,
        fks: list[ForeignKey],
        table_metadata: dict[str, TableMetadata],
        target_table: str,
    ) -> list[ForeignKey]:
        """Deduplicate foreign keys and resolve conflicts.

        When a child column points to multiple parent columns with similar values,
        keep only the best relationship based on:
        1. Prefer relationships to the target table
        2. Prefer higher coverage
        3. Prefer parent table with matching primary key

        Args:
            fks: List of detected foreign keys
            table_metadata: Dict of TableMetadata
            target_table: Name of the target table (for priority)

        Returns:
            Deduplicated list of ForeignKey objects
        """
        if not fks:
            return fks

        # Group FKs by (child_table, child_column)
        fk_groups = {}
        for fk in fks:
            key = (fk.child_table, fk.child_column)
            if key not in fk_groups:
                fk_groups[key] = []
            fk_groups[key].append(fk)

        deduplicated = []

        for (child_table, child_column), group in fk_groups.items():
            if len(group) == 1:
                # Only one FK for this column, keep it
                deduplicated.append(group[0])
                continue

            # Multiple FKs for the same child column, need to choose
            logger.info(
                f"Found {len(group)} FK candidates for {child_table}.{child_column}, deduplicating..."
            )

            # Sort by priority:
            # 1. Target table first
            # 2. Higher coverage
            # 3. Parent table name (for determinism)
            def fk_priority(fk: ForeignKey) -> tuple:
                is_target = 1 if fk.parent_table == target_table else 0
                return (is_target, fk.coverage, fk.parent_table)

            group_sorted = sorted(group, key=fk_priority, reverse=True)
            best_fk = group_sorted[0]

            # Check if this looks like the same relationship pointing to different tables
            # (e.g., ANumber in abstracts vs wamex_reports)
            parent_columns = {fk.parent_column for fk in group}
            parent_tables = {fk.parent_table for fk in group}

            if len(parent_columns) == 1 and len(parent_tables) > 1:
                # Same column name in different parent tables
                # This might be a case where both parent tables represent the same entity
                logger.info(
                    f"  Detected redundant FK: {child_table}.{child_column} -> "
                    f"{', '.join(sorted(parent_tables))}.{list(parent_columns)[0]}"
                )
                logger.info(f"  Keeping only: {best_fk}")

            deduplicated.append(best_fk)

        logger.info(f"Deduplication: {len(fks)} -> {len(deduplicated)} FKs")
        return deduplicated
