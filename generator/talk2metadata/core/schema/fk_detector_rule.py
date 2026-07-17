"""Rule-based foreign key detector."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from talk2metadata.core.schema.fk_detector_base import FKDetectorBase
from talk2metadata.core.schema.types import ForeignKey, TableMetadata
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


class RuleBasedFKDetector(FKDetectorBase):
    """Rule-based foreign key detector using heuristics."""

    def __init__(self, config: Dict):
        """Initialize rule-based FK detector.

        Args:
            config: Configuration dict
        """
        super().__init__(config)
        self.min_coverage = config.get("min_coverage", 0.9)
        self.tolerance = config.get("inclusion_tolerance", 0.1)

    def detect(
        self,
        tables: Dict[str, pd.DataFrame],
        table_metadata: Dict[str, TableMetadata],
        target_table: str,
    ) -> List[ForeignKey]:
        """Detect foreign keys using rule-based heuristics.

        Prioritizes target table (star schema pattern). For star schema,
        all dimension tables should connect to the target/center table.
        Only if no match to target table is found, check other tables.

        Args:
            tables: Dict of DataFrames
            table_metadata: Dict of TableMetadata
            target_table: Name of the target/center table (for star schema)

        Returns:
            List of ForeignKey objects
        """
        logger.info("Running rule-based FK detection...")

        fks = []
        # Track which (child_table, child_column) pairs we've already found FKs for
        found_fks = set()

        # Check all tables as potential child tables (including target table)
        for child_name, child_df in tables.items():

            for child_col in child_df.columns:
                # Note: A column CAN be both a primary key and a foreign key
                # (foreign primary key pattern, e.g., abstracts.ANumber is PK and FK)
                # We only skip self-referential relationships (same table)

                # Skip if we already found an FK for this column
                if (child_name, child_col) in found_fks:
                    continue

                child_col_lower = child_col.lower()

                # Heuristic 1: Column name ends with _id or _key (case-insensitive)
                has_id_suffix = child_col_lower.endswith(
                    "_id"
                ) or child_col_lower.endswith("_key")

                # Heuristic 2: Column name matches another table's primary key column name (case-insensitive)
                # This handles cases like ANumber -> ANumber
                # BUT: Skip generic column names like "id", "Id" that have no semantic meaning
                # unless it's matching to target table's PK (star schema pattern)
                is_generic_id = child_col_lower in ("id", "key", "pk", "primary_key")

                # Priority 1: Check target table first (star schema pattern)
                # Skip if child table IS the target table (target table can have FKs to other tables)
                target_match = False
                if (
                    child_name != target_table
                    and target_table in tables
                    and target_table in table_metadata
                ):
                    target_meta = table_metadata[target_table]
                    if target_meta.primary_key is not None:
                        target_pk_lower = target_meta.primary_key.lower()
                        # Check if column matches target table's PK (case-insensitive)
                        # For target table, we allow generic names (id, Id) if they match
                        if target_pk_lower == child_col_lower:
                            target_match = True
                        # Or check if column name pattern suggests target table
                        elif has_id_suffix:
                            parent_candidates = self._get_parent_candidates(
                                child_col, [target_table]
                            )
                            if target_table in parent_candidates:
                                target_match = True

                        # If target table matches, check coverage
                        if target_match:
                            target_df = tables[target_table]
                            target_pk = target_meta.primary_key
                            if target_pk in target_df.columns:
                                coverage = self._check_inclusion(
                                    child_df[child_col],
                                    target_df[target_pk],
                                )
                                if coverage >= self.min_coverage:
                                    fks.append(
                                        ForeignKey(
                                            child_table=child_name,
                                            child_column=child_col,
                                            parent_table=target_table,
                                            parent_column=target_pk,
                                            coverage=coverage,
                                        )
                                    )
                                    found_fks.add((child_name, child_col))
                                    continue  # Found FK to target table, skip other candidates

                # Priority 2: If no match to target table, check other tables
                # Skip generic column names for non-target matches (no semantic meaning)
                if is_generic_id and not target_match:
                    continue  # Generic "id" columns shouldn't match other tables' "id" columns

                # Find other matching tables (excluding self only)
                matching_pk_tables = []
                for parent_name, parent_meta in table_metadata.items():
                    if parent_name == child_name:  # Skip self only
                        continue
                    if parent_meta.primary_key is None:
                        continue
                    # Case-insensitive column name match
                    parent_pk_lower = parent_meta.primary_key.lower()
                    if parent_pk_lower == child_col_lower:
                        # Skip if both are generic IDs (no semantic meaning)
                        if is_generic_id and parent_pk_lower in (
                            "id",
                            "key",
                            "pk",
                            "primary_key",
                        ):
                            continue
                        matching_pk_tables.append(
                            (parent_name, parent_meta.primary_key)
                        )

                # Only proceed if at least one heuristic matches
                if not has_id_suffix and not matching_pk_tables:
                    continue

                # Try parent candidates from column name pattern (Heuristic 1)
                parent_candidates = []
                if has_id_suffix:
                    # Check all tables except self (target table can be child)
                    all_tables = [t for t in tables.keys() if t != child_name]
                    parent_candidates = self._get_parent_candidates(
                        child_col, all_tables
                    )

                # Add tables with matching PK column names (Heuristic 2)
                for parent_name, parent_pk in matching_pk_tables:
                    if parent_name not in parent_candidates:
                        parent_candidates.append(parent_name)

                # Check each candidate parent table (excluding self only)
                for parent_name in parent_candidates:
                    if parent_name == child_name:
                        continue

                    parent_df = tables[parent_name]
                    parent_pk = table_metadata[parent_name].primary_key

                    if parent_pk is None or parent_pk not in parent_df.columns:
                        continue

                    # Check inclusion dependency
                    coverage = self._check_inclusion(
                        child_df[child_col],
                        parent_df[parent_pk],
                    )

                    if coverage >= self.min_coverage:
                        fks.append(
                            ForeignKey(
                                child_table=child_name,
                                child_column=child_col,
                                parent_table=parent_name,
                                parent_column=parent_pk,
                                coverage=coverage,
                            )
                        )
                        found_fks.add((child_name, child_col))
                        break  # Found FK, stop searching for this column

        logger.info(f"Rule-based detection found {len(fks)} FKs")
        return fks

    def _get_parent_candidates(
        self, column_name: str, table_names: List[str]
    ) -> List[str]:
        """Get potential parent table names from column name.

        Args:
            column_name: Column name (e.g., 'customer_id')
            table_names: Available table names

        Returns:
            List of candidate parent table names
        """
        # Remove suffixes (case-insensitive, only at the end)
        column_lower = column_name.lower()
        base_name = column_lower
        # Remove suffix if present (in order: _id, _key, _fk)
        if base_name.endswith("_id"):
            base_name = base_name[:-3]  # Remove "_id"
        elif base_name.endswith("_key"):
            base_name = base_name[:-4]  # Remove "_key"
        elif base_name.endswith("_fk"):
            base_name = base_name[:-3]  # Remove "_fk"

        candidates = []
        # Create lowercase lookup for case-insensitive matching
        table_names_lower = {name.lower(): name for name in table_names}

        # Exact match (case-insensitive)
        if base_name in table_names_lower:
            candidates.append(table_names_lower[base_name])

        # Plural forms (case-insensitive)
        plural_base = base_name + "s"
        if plural_base in table_names_lower:
            candidates.append(table_names_lower[plural_base])

        # Try removing common prefixes (already case-insensitive)
        for table_name in table_names:
            if table_name.lower().startswith(base_name):
                if table_name not in candidates:
                    candidates.append(table_name)

        return candidates
