"""Business logic for mode preparation commands (indexing or database loading)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

from talk2metadata.core.schema.schema import SchemaMetadata
from talk2metadata.core.solution.modes import get_registry
from talk2metadata.core.solution.modes.registry import get_mode_retriever_config
from talk2metadata.utils.config import Config
from talk2metadata.utils.csv_to_db import create_sqlite_from_csv
from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.paths import (
    find_schema_file,
    get_indexes_dir,
    get_metadata_dir,
    get_processed_dir,
)

logger = get_logger(__name__)


class PrepareHandler:
    """Handler for mode preparation operations.

    Handles different preparation steps for different modes:
    - Index-based modes (e.g., semantic): Build indexes
    - Database-based modes (e.g., text2sql): Load CSV to database
    """

    def __init__(self, config: Config):
        """Initialize handler.

        Args:
            config: Configuration instance
        """
        self.config = config
        self.registry = get_registry()

    def prepare_mode(
        self,
        mode_name: str,
        schema_metadata: SchemaMetadata,
        run_id: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, str]:
        """Prepare a mode for use (index or database loading).

        Automatically determines what needs to be done based on mode type:
        - Index-based modes: Builds index if not exists (uses config for model, batch_size, etc.)
        - Database-based modes: Loads CSV to database if needed

        Args:
            mode_name: Mode name to prepare
            schema_metadata: Schema metadata
            run_id: Optional run ID
            force: If True, rebuild index/database even if it already exists

        Returns:
            Dict with preparation results (e.g., {"status": "success", "message": "..."})

        Raises:
            ValueError: If mode not found or not enabled
        """
        mode_info = self.registry.get(mode_name)
        if not mode_info or not mode_info.enabled:
            raise ValueError(f"Mode '{mode_name}' is not enabled")

        if mode_name == "hybrid" or str(mode_name).startswith("hybrid."):
            mode_cfg = get_mode_retriever_config(mode_name)
            vector_mode = mode_cfg.get("vector_mode", "semantic")
            sql_mode = mode_cfg.get("sql_mode", "text2sql.two_step")
            vector_result = self.prepare_mode(
                vector_mode, schema_metadata, run_id, force=force
            )
            sql_result = self.prepare_mode(
                sql_mode, schema_metadata, run_id, force=force
            )

            if vector_result.get("status") != "success":
                return {
                    "status": "error",
                    "message": f"Hybrid dependency failed: {vector_result.get('message', '')}",
                    "requires_index": True,
                }
            if sql_result.get("status") != "success":
                return {
                    "status": "error",
                    "message": f"Hybrid dependency failed: {sql_result.get('message', '')}",
                    "requires_db": True,
                }

            return {
                "status": "success",
                "message": f"Mode '{mode_name}' is ready ({vector_mode} index + {sql_mode} database prepared)",
                "requires_index": False,
                "requires_db": False,
            }

        # Check if mode needs database preparation
        if str(mode_name).startswith("text2sql"):
            return self._prepare_text2sql_mode(
                mode_name=mode_name,
                schema_metadata=schema_metadata,
                run_id=run_id,
                force=force,
            )
        else:
            # For index-based modes
            # Check if index already exists (unless force is True)
            if not force:
                index_dir = get_indexes_dir(
                    run_id or self.config.get("run_id"), self.config
                )
                mode_index_dir = index_dir / mode_name
                if (mode_index_dir / "schema_metadata.json").exists():
                    return {
                        "status": "success",
                        "message": f"Mode '{mode_name}' is ready (index already exists)",
                        "requires_index": False,
                    }

            # Build index (force rebuild if force=True)
            return self._build_index_for_mode(
                mode_name=mode_name,
                schema_metadata=schema_metadata,
                run_id=run_id,
            )

    def _prepare_text2sql_mode(
        self,
        mode_name: str,
        schema_metadata: SchemaMetadata,
        run_id: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, str]:
        """Prepare text2sql mode by loading CSV to database.

        Args:
            mode_name: Mode name
            schema_metadata: Schema metadata
            run_id: Optional run ID
            force: If True, recreate database even if it exists

        Returns:
            Dict with preparation results
        """
        ingest_config = self.config.get("ingest", {}) or {}
        data_type = ingest_config.get("data_type") or "csv"
        source_path = ingest_config.get("source_path")
        effective_run_id = run_id or self.config.get("run_id")
        if not source_path and effective_run_id:
            inferred_csv_dir = Path("./data") / str(effective_run_id) / "raw"
            if inferred_csv_dir.exists():
                source_path = str(inferred_csv_dir)

        if data_type in ("database", "db"):
            # Already a database, no preparation needed
            return {
                "status": "success",
                "message": f"Mode '{mode_name}' is ready (using existing database: {source_path})",
                "requires_db": False,
            }

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

            # Check if database already exists (unless force is True)
            if not force:
                from talk2metadata.utils.paths import get_db_dir

                db_dir = get_db_dir(run_id or self.config.get("run_id"), self.config)
                db_path = db_dir / "text2sql.db"
                if db_path.exists():
                    return {
                        "status": "success",
                        "message": f"Mode '{mode_name}' is ready (database already exists at {db_path})",
                        "connection_string": f"sqlite:///{db_path}",
                        "requires_db": False,
                    }

            # Create SQLite database from CSV (force recreate if force=True)
            connection_string = create_sqlite_from_csv(
                csv_data_dir=csv_dir,
                run_id=run_id,
                schema_metadata=schema_metadata,
            )

            return {
                "status": "success",
                "message": f"Mode '{mode_name}' is ready (created database from CSV)",
                "connection_string": connection_string,
                "requires_db": True,
            }

        else:
            raise ValueError(
                f"Unsupported data_type: {data_type}. "
                "Supported types: 'csv', 'database', 'db'"
            )

    def _build_index_for_mode(
        self,
        mode_name: str,
        schema_metadata: SchemaMetadata,
        run_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Build index for an index-based mode.

        Args:
            mode_name: Mode name
            schema_metadata: Schema metadata
            run_id: Optional run ID

        Returns:
            Dict with preparation results
        """
        try:
            # Import here to avoid circular imports
            from talk2metadata.cli.handlers.index_handler import IndexHandler

            # Load tables from pickle
            processed_dir = get_processed_dir(
                run_id or self.config.get("run_id"), self.config
            )
            tables_path = processed_dir / "tables.pkl"

            if not tables_path.exists():
                return {
                    "status": "error",
                    "message": f"Tables pickle file not found at {tables_path}. "
                    "Please run 'talk2metadata ingest' first.",
                    "requires_index": True,
                }

            logger.info(f"Loading tables from {tables_path}")
            with open(tables_path, "rb") as f:
                tables = pickle.load(f)

            # Build index using config parameters (model_name, batch_size from config)
            index_handler = IndexHandler(self.config)
            logger.info(
                f"Building index for mode '{mode_name}' using config settings..."
            )
            table_indices, indexer = index_handler.build_index_for_mode(
                mode_name=mode_name,
                tables=tables,
                schema_metadata=schema_metadata,
                model_name=None,  # Use config default
                batch_size=None,  # Use config default
            )

            # Save index
            schema_path = find_schema_file(
                get_metadata_dir(run_id or self.config.get("run_id"), self.config),
                target_table=schema_metadata.target_table,
            )
            index_dir = index_handler.save_index_for_mode(
                mode_name=mode_name,
                table_indices=table_indices,
                indexer=indexer,
                schema_metadata_path=schema_path,
                run_id=run_id,
            )

            # For graph mode: build knowledge graph from DuckDB (no SQL at query time)
            if mode_name == "graph":
                from talk2metadata.core.solution.paths.graph.knowledge_graph import (
                    KnowledgeGraph,
                )

                kg = KnowledgeGraph.build_from_duckdb(
                    index_dir / "metadata.duckdb",
                    schema_metadata,
                )
                kg.save(index_dir / "knowledge_graph.pkl")
                logger.info(
                    f"Built knowledge graph: {len(kg.nodes)} nodes, {len(kg.edges)} edges"
                )

            return {
                "status": "success",
                "message": f"Mode '{mode_name}' is ready (index built and saved to {index_dir})",
                "requires_index": False,
                "index_dir": str(index_dir),
            }
        except Exception as e:
            logger.exception(f"Failed to build index for mode '{mode_name}': {e}")
            return {
                "status": "error",
                "message": f"Failed to build index: {e}",
                "requires_index": True,
            }

    def prepare_all_modes(
        self,
        mode_names: List[str],
        schema_metadata: SchemaMetadata,
        run_id: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Dict[str, str]]:
        """Prepare multiple modes.

        Args:
            mode_names: List of mode names to prepare
            schema_metadata: Schema metadata
            run_id: Optional run ID
            force: If True, rebuild indexes/databases even if they already exist

        Returns:
            Dict mapping mode_name -> preparation results
        """
        results = {}
        for mode_name in mode_names:
            try:
                result = self.prepare_mode(
                    mode_name, schema_metadata, run_id, force=force
                )
                results[mode_name] = result
            except Exception as e:
                logger.error(f"Failed to prepare mode '{mode_name}': {e}")
                results[mode_name] = {
                    "status": "error",
                    "message": str(e),
                }
        return results
