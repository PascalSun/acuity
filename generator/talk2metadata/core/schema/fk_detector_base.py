"""Base class for foreign key detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd

from talk2metadata.core.schema.types import ForeignKey, TableMetadata


class FKDetectorBase(ABC):
    """Base class for foreign key detectors."""

    def __init__(self, config: Dict):
        """Initialize FK detector.

        Args:
            config: Configuration dict
        """
        self.config = config

    @abstractmethod
    def detect(
        self,
        tables: Dict[str, pd.DataFrame],
        table_metadata: Dict[str, TableMetadata],
        target_table: str,
    ) -> List[ForeignKey]:
        """Detect foreign keys.

        Args:
            tables: Dict of DataFrames
            table_metadata: Dict of TableMetadata
            target_table: Name of the target table

        Returns:
            List of ForeignKey objects
        """
        pass

    def _check_inclusion(
        self,
        child_values: pd.Series,
        parent_values: pd.Series,
    ) -> float:
        """Check inclusion dependency (child ⊆ parent).

        Args:
            child_values: Child column values
            parent_values: Parent column values

        Returns:
            Coverage ratio (0.0 to 1.0)
        """
        child_set = set(child_values.dropna())
        parent_set = set(parent_values.dropna())

        if not child_set:
            return 0.0

        overlap = child_set & parent_set
        return len(overlap) / len(child_set)
