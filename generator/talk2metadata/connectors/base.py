"""Base connector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd

from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


class BaseConnector(ABC):
    """Abstract base class for data connectors."""

    def __init__(self, **kwargs):
        """Initialize connector.

        Args:
            **kwargs: Connector-specific configuration
        """
        self.config = kwargs
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def load_tables(self) -> Dict[str, pd.DataFrame]:
        """Load tables into memory.

        Returns:
            Dict mapping table_name -> DataFrame

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass

    @abstractmethod
    def get_table_names(self) -> List[str]:
        """Get list of available table names.

        Returns:
            List of table names

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass

    def validate_target_table(self, target_table: str) -> bool:
        """Validate that target table exists.

        Args:
            target_table: Target table name

        Returns:
            True if table exists

        Raises:
            ValueError: If table doesn't exist
        """
        table_names = self.get_table_names()

        if target_table not in table_names:
            raise ValueError(
                f"Target table '{target_table}' not found. "
                f"Available tables: {table_names}"
            )

        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.config})"
