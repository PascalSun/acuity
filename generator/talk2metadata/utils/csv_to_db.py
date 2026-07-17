"""Utility to convert CSV data to SQLite database for text2sql mode."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from sqlalchemy import create_engine, text

from talk2metadata.connectors.csv_loader import CSVLoader
from talk2metadata.core.schema.schema import SchemaMetadata
from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.paths import get_db_dir

logger = get_logger(__name__)


def create_sqlite_from_csv(
    csv_data_dir: Path,
    run_id: Optional[str] = None,
    db_path: Optional[Path] = None,
    schema_metadata: Optional[SchemaMetadata] = None,
) -> str:
    """Create a SQLite database from CSV files with foreign key constraints.

    Args:
        csv_data_dir: Directory containing CSV files
        run_id: Optional run ID for cache location (if None, checks config)
        db_path: Optional path to save database (default: uses get_db_dir(run_id) / "text2sql.db")
        schema_metadata: Optional schema metadata for creating foreign key constraints

    Returns:
        SQLite connection string (e.g., "sqlite:///path/to/db.db")
    """
    if db_path is None:
        # Get db_dir from run_id (or config if run_id is None)
        # get_db_dir will check config for run_id if not provided
        db_dir = get_db_dir(run_id)
        db_path = db_dir / "text2sql.db"

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Create SQLite connection
    connection_string = f"sqlite:///{db_path}"
    engine = create_engine(connection_string)

    # Enable foreign key constraints in SQLite
    with engine.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys = ON"))
        conn.commit()

    # Load CSV files
    logger.info(f"Loading CSV files from {csv_data_dir} into SQLite database")
    csv_loader = CSVLoader(data_dir=csv_data_dir, target_table=None)
    tables = csv_loader.load_tables()

    # Create mapping from original to lowercase names for schema metadata update
    table_name_mapping = {}
    column_name_mapping = {}

    # Write tables to database
    # First, create tables without FK constraints (SQLite requires tables to exist before FK)
    logger.info(f"Writing {len(tables)} tables to SQLite database: {db_path}")
    for table_name, df in tables.items():
        # Convert table name and column names to lowercase
        table_name_lower = table_name.lower()
        table_name_mapping[table_name] = table_name_lower

        df_lower = df.copy()
        original_columns = list(df_lower.columns)
        df_lower.columns = [col.lower() for col in df_lower.columns]
        column_name_mapping[table_name] = dict(zip(original_columns, df_lower.columns))

        logger.info(f"  Writing table '{table_name_lower}' ({len(df_lower)} rows)...")
        logger.debug(f"    Columns (lowercased): {list(df_lower.columns)}")

        df_lower.to_sql(
            table_name_lower,
            engine,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=1000,
        )

        # Create indexes for performance
        with engine.connect() as conn:
            # Index 'anumber' if it exists (common join key)
            if "anumber" in df_lower.columns:
                logger.debug(f"    Creating index on {table_name_lower}.anumber")
                conn.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS idx_{table_name_lower}_anumber ON {table_name_lower}(anumber)"
                    )
                )

            # Index 'id' if it exists (common primary key)
            if "id" in df_lower.columns:
                logger.debug(f"    Creating index on {table_name_lower}.id")
                conn.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS idx_{table_name_lower}_id ON {table_name_lower}(id)"
                    )
                )

    # Build a lowercased copy of schema metadata for SQLite (which needs lowercase).
    # IMPORTANT: do NOT mutate the original schema_metadata — other modes (graph,
    # lexical, semantic) rely on the original-cased column names.
    if schema_metadata:
        logger.info("Building lowercase schema metadata copy for SQLite")
        from copy import deepcopy

        from talk2metadata.core.schema.types import ForeignKey as FK
        from talk2metadata.core.schema.types import TableMetadata

        sql_schema = deepcopy(schema_metadata)

        updated_tables = {}
        for table_name, table_meta in schema_metadata.tables.items():
            if table_name in table_name_mapping:
                table_name_lower = table_name_mapping[table_name]
                col_map = column_name_mapping[table_name]

                if isinstance(table_meta.columns, dict):
                    updated_columns = {
                        col_map.get(c, c.lower()): dtype
                        for c, dtype in table_meta.columns.items()
                    }
                else:
                    updated_columns = {
                        col_map.get(col, col.lower()): "text"
                        for col in table_meta.columns
                    }

                updated_pk = None
                if table_meta.primary_key and table_meta.primary_key in col_map:
                    updated_pk = col_map[table_meta.primary_key]

                updated_sample_values = {
                    col_map.get(c, c.lower()): v
                    for c, v in table_meta.sample_values.items()
                }

                updated_tables[table_name_lower] = TableMetadata(
                    name=table_name_lower,
                    columns=updated_columns,
                    primary_key=updated_pk,
                    row_count=table_meta.row_count,
                    sample_values=updated_sample_values,
                )

        sql_schema.tables = updated_tables
        sql_schema.foreign_keys = [
            FK(
                child_table=table_name_mapping.get(
                    fk.child_table, fk.child_table.lower()
                ),
                child_column=column_name_mapping.get(fk.child_table, {}).get(
                    fk.child_column, fk.child_column.lower()
                ),
                parent_table=table_name_mapping.get(
                    fk.parent_table, fk.parent_table.lower()
                ),
                parent_column=column_name_mapping.get(fk.parent_table, {}).get(
                    fk.parent_column, fk.parent_column.lower()
                ),
                coverage=fk.coverage,
            )
            for fk in schema_metadata.foreign_keys
        ]
        if schema_metadata.target_table in table_name_mapping:
            sql_schema.target_table = table_name_mapping[schema_metadata.target_table]

        logger.info("Built lowercase schema metadata copy for SQLite")

    # Log foreign key relationships
    if schema_metadata and schema_metadata.foreign_keys:
        logger.info(
            f"Found {len(schema_metadata.foreign_keys)} foreign key relationships"
        )
        for fk in sql_schema.foreign_keys:
            logger.debug(
                f"FK: {fk.child_table}.{fk.child_column} -> "
                f"{fk.parent_table}.{fk.parent_column} (coverage: {fk.coverage:.2%})"
            )

    logger.info(f"Successfully created SQLite database at {db_path}")
    return connection_string


def get_or_create_db_connection(
    ingest_config: Dict,
    schema_metadata: SchemaMetadata,
    run_id: Optional[str] = None,
) -> str:
    """Get database connection string, creating SQLite from CSV if needed.

    Args:
        ingest_config: Ingest configuration dict
        schema_metadata: Schema metadata
        run_id: Optional run ID

    Returns:
        Database connection string
    """
    data_type = ingest_config.get("data_type") or "csv"
    source_path = ingest_config.get("source_path")
    if not source_path and run_id:
        inferred_csv_dir = Path("./data") / str(run_id) / "raw"
        if inferred_csv_dir.exists():
            source_path = str(inferred_csv_dir)

    if data_type in ("database", "db"):
        # Already a database connection
        if not source_path:
            raise ValueError(
                "Database connection string not found in config. "
                "Please set 'ingest.source_path' in your run config (e.g., configs/wamex.yml)"
            )
        return source_path

    elif data_type == "csv":
        # Need to create database from CSV
        if not source_path:
            raise ValueError(
                "CSV data directory not found in config. "
                "Please set 'ingest.source_path' in your run config (e.g., configs/wamex.yml)"
            )

        csv_dir = Path(source_path)
        if not csv_dir.exists():
            raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

        # Determine database path (same logic as create_sqlite_from_csv)
        if run_id:
            db_dir = get_db_dir(run_id)
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "text2sql.db"
            connection_string = f"sqlite:///{db_path}"
        else:
            # Without run_id, always recreate (temp file)
            connection_string = create_sqlite_from_csv(
                csv_data_dir=csv_dir,
                run_id=run_id,
                schema_metadata=schema_metadata,
            )
            logger.info(
                "Created temporary SQLite database from CSV data for text2sql mode"
            )
            return connection_string

        # Check if database already exists
        db_path = Path(db_path)
        if db_path.exists():
            logger.info(f"Using existing database at {db_path}")
            return connection_string

        # Database doesn't exist, create it
        connection_string = create_sqlite_from_csv(
            csv_data_dir=csv_dir,
            run_id=run_id,
            schema_metadata=schema_metadata,
            db_path=db_path,
        )
        logger.info("Created SQLite database from CSV data for text2sql mode")
        return connection_string

    else:
        raise ValueError(
            f"Unsupported data_type: {data_type}. "
            "Supported types: 'csv', 'database', 'db'"
        )
