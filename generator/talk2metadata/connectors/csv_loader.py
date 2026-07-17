"""CSV connector for loading tables from CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from talk2metadata.connectors.base import BaseConnector
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


class CSVLoader(BaseConnector):
    """Load tables from CSV files."""

    def __init__(
        self,
        data_dir: str | Path,
        target_table: Optional[str] = None,
        file_pattern: str = "*.csv",
        **pandas_kwargs,
    ):
        """Initialize CSV loader.

        Args:
            data_dir: Directory containing CSV files
            target_table: Optional target table name (for validation)
            file_pattern: Glob pattern for CSV files
            **pandas_kwargs: Additional arguments passed to pd.read_csv()

        Example:
            >>> loader = CSVLoader(
            ...     data_dir="./data/csv",
            ...     target_table="orders"
            ... )
            >>> tables = loader.load_tables()
        """
        super().__init__(
            data_dir=data_dir,
            target_table=target_table,
            file_pattern=file_pattern,
            **pandas_kwargs,
        )

        self.data_dir = Path(data_dir)
        self.target_table = target_table
        self.file_pattern = file_pattern
        self.pandas_kwargs = pandas_kwargs

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {self.data_dir}")

    def load_tables(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV files in the directory.

        Returns:
            Dict mapping table_name -> DataFrame
        """
        csv_files = list(self.data_dir.glob(self.file_pattern))

        if not csv_files:
            raise ValueError(
                f"No CSV files found in {self.data_dir} "
                f"matching pattern '{self.file_pattern}'"
            )

        self.logger.info(f"Found {len(csv_files)} CSV files in {self.data_dir}")

        tables = {}

        for csv_file in csv_files:
            table_name = csv_file.stem  # Filename without extension
            self.logger.info(f"Loading {csv_file.name} as table '{table_name}'")

            try:
                df = pd.read_csv(csv_file, **self.pandas_kwargs)
                tables[table_name] = df
                self.logger.debug(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                self.logger.error(f"Failed to load {csv_file}: {e}")
                raise

        # Validate target table if specified
        if self.target_table:
            self.validate_target_table(self.target_table)

        self.logger.info(f"Successfully loaded {len(tables)} tables")

        return tables

    def get_table_names(self) -> List[str]:
        """Get list of CSV file names (without .csv extension).

        Returns:
            List of table names
        """
        csv_files = list(self.data_dir.glob(self.file_pattern))
        return [f.stem for f in csv_files]

    def load_single_table(self, table_name: str) -> pd.DataFrame:
        """Load a single table by name.

        Args:
            table_name: Table name (CSV filename without extension)

        Returns:
            DataFrame

        Raises:
            FileNotFoundError: If CSV file doesn't exist
        """
        csv_file = self.data_dir / f"{table_name}.csv"

        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        self.logger.info(f"Loading single table: {table_name}")
        df = pd.read_csv(csv_file, **self.pandas_kwargs)

        self.logger.debug(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

        return df
