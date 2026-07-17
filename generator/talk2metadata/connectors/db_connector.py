"""Database connector using SQLAlchemy."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from talk2metadata.connectors.base import BaseConnector
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


class DBConnector(BaseConnector):
    """Connect to relational databases via SQLAlchemy."""

    def __init__(
        self,
        connection_string: str,
        target_table: Optional[str] = None,
        schema: Optional[str] = None,
        tables: Optional[List[str]] = None,
    ):
        """Initialize database connector.

        Args:
            connection_string: SQLAlchemy connection string
            target_table: Optional target table name
            schema: Database schema name (optional)
            tables: Optional list of specific tables to load (loads all if None)

        Example:
            >>> # PostgreSQL
            >>> connector = DBConnector(
            ...     "postgresql://user:pass@localhost:5432/mydb",
            ...     target_table="orders"
            ... )
            >>>
            >>> # SQLite
            >>> connector = DBConnector(
            ...     "sqlite:///path/to/database.db",
            ...     target_table="orders"
            ... )
        """
        super().__init__(
            connection_string=connection_string,
            target_table=target_table,
            schema=schema,
            tables=tables,
        )

        self.connection_string = connection_string
        self.target_table = target_table
        self.schema = schema
        self.table_filter = tables

        self.logger.info("Creating database engine")
        self.engine: Engine = create_engine(connection_string)

        # Test connection
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self.logger.info("Database connection successful")
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise

    def load_tables(self) -> Dict[str, pd.DataFrame]:
        """Load tables from database.

        Returns:
            Dict mapping table_name -> DataFrame
        """
        table_names = self.get_table_names()

        if not table_names:
            raise ValueError(f"No tables found in database (schema={self.schema})")

        self.logger.info(f"Loading {len(table_names)} tables from database")

        tables = {}

        for table_name in table_names:
            self.logger.info(f"Loading table: {table_name}")

            try:
                df = self.load_single_table(table_name)
                tables[table_name] = df
                self.logger.debug(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                self.logger.error(f"Failed to load table {table_name}: {e}")
                raise

        # Validate target table if specified
        if self.target_table:
            self.validate_target_table(self.target_table)

        self.logger.info(f"Successfully loaded {len(tables)} tables")

        return tables

    def get_table_names(self) -> List[str]:
        """Get list of table names from database.

        Returns:
            List of table names
        """
        inspector = inspect(self.engine)
        all_tables = inspector.get_table_names(schema=self.schema)

        # Filter if specific tables requested
        if self.table_filter:
            tables = [t for t in all_tables if t in self.table_filter]
            self.logger.info(
                f"Filtered to {len(tables)} tables from {len(all_tables)} available"
            )
        else:
            tables = all_tables

        return tables

    def load_single_table(self, table_name: str) -> pd.DataFrame:
        """Load a single table from database.

        Args:
            table_name: Table name

        Returns:
            DataFrame

        Raises:
            ValueError: If table doesn't exist
        """
        # Validate table exists
        all_tables = inspect(self.engine).get_table_names(schema=self.schema)
        if table_name not in all_tables:
            raise ValueError(f"Table '{table_name}' not found. Available: {all_tables}")

        # Build query
        if self.schema:
            query = f'SELECT * FROM "{self.schema}"."{table_name}"'
        else:
            query = f'SELECT * FROM "{table_name}"'

        self.logger.debug(f"Executing: {query}")

        df = pd.read_sql(query, self.engine)

        return df

    def get_foreign_keys(self) -> Dict[str, List[Dict]]:
        """Get foreign key information from database schema.

        Returns:
            Dict mapping table_name -> list of FK dicts

        Example result:
            {
                "orders": [
                    {
                        "child_column": "customer_id",
                        "parent_table": "customers",
                        "parent_column": "id"
                    }
                ]
            }
        """
        inspector = inspect(self.engine)
        table_names = self.get_table_names()

        fks = {}

        for table_name in table_names:
            fks[table_name] = []

            for fk in inspector.get_foreign_keys(table_name, schema=self.schema):
                # SQLAlchemy FK format:
                # {
                #   'constrained_columns': ['customer_id'],
                #   'referred_table': 'customers',
                #   'referred_columns': ['id']
                # }

                fks[table_name].append(
                    {
                        "child_column": fk["constrained_columns"][0],
                        "parent_table": fk["referred_table"],
                        "parent_column": fk["referred_columns"][0],
                    }
                )

        # Filter out tables with no FKs
        fks = {table: fk_list for table, fk_list in fks.items() if fk_list}

        self.logger.info(f"Found foreign keys in {len(fks)} tables")

        return fks

    def close(self):
        """Close database connection."""
        self.engine.dispose()
        self.logger.info("Database connection closed")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass
