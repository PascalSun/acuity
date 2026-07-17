"""Business logic for index building commands."""

from __future__ import annotations

import inspect
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from talk2metadata.core.schema import SchemaMetadata
from talk2metadata.core.solution.modes import get_mode_indexer_config, get_registry
from talk2metadata.utils.config import Config
from talk2metadata.utils.paths import get_indexes_dir


class IndexHandler:
    """Handler for index building operations.

    Encapsulates business logic for index building commands,
    keeping CLI commands thin and focused on user interaction.

    Example:
        >>> handler = IndexHandler(config)
        >>> tables = {...}
        >>> schema = SchemaMetadata(...)
        >>> table_indices, indexer = handler.build_index_for_mode(
        ...     "semantic", tables, schema
        ... )
    """

    def __init__(self, config: Config):
        """Initialize handler.

        Args:
            config: Configuration instance
        """
        self.config = config
        self.registry = get_registry()

    def build_index_for_mode(
        self,
        mode_name: str,
        tables: Dict[str, pd.DataFrame],
        schema_metadata: SchemaMetadata,
        model_name: Optional[str] = None,
        batch_size: int = 32,
    ) -> Tuple[Dict, Any]:
        """Build index for a specific mode.

        Args:
            mode_name: Mode name
            tables: Dictionary of DataFrames
            schema_metadata: Schema metadata
            model_name: Optional model name (overrides config)
            batch_size: Batch size for embedding generation

        Returns:
            Tuple of (table_indices, indexer)

        Raises:
            Exception: If index building fails
        """
        mode_info = self.registry.get(mode_name)
        if not mode_info or not mode_info.enabled:
            raise ValueError(f"Mode '{mode_name}' is not enabled")

        # Get mode-specific config
        mode_indexer_config = get_mode_indexer_config(mode_name)

        # Get indexer class from mode
        IndexerClass = mode_info.indexer_class

        init_kwargs = {
            "model_name": model_name or mode_indexer_config.get("model_name"),
            "device": mode_indexer_config.get("device"),
            "batch_size": (
                batch_size
                if batch_size is not None
                else mode_indexer_config.get("batch_size", 32)
            ),
            "normalize": mode_indexer_config.get("normalize", True),
        }

        init_params = inspect.signature(IndexerClass.__init__).parameters
        accepted_kwargs = {
            k: v for k, v in init_kwargs.items() if k in init_params and v is not None
        }
        indexer = IndexerClass(**accepted_kwargs) if accepted_kwargs else IndexerClass()

        # Build index
        table_indices = indexer.build_index(tables, schema_metadata)

        return table_indices, indexer

    def save_index_for_mode(
        self,
        mode_name: str,
        table_indices: Dict,
        indexer: Any,
        schema_metadata_path: Path,
        output_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
    ) -> Path:
        """Save index for a specific mode.

        Args:
            mode_name: Mode name
            table_indices: Table indices dictionary
            indexer: Indexer instance
            schema_metadata_path: Path to schema metadata file
            output_dir: Optional base output directory
            run_id: Optional run ID

        Returns:
            Path to saved index directory
        """
        base_index_dir = (
            output_dir
            if output_dir
            else get_indexes_dir(run_id or self.config.get("run_id"), self.config)
        )
        mode_index_dir = Path(base_index_dir) / mode_name
        mode_index_dir.mkdir(parents=True, exist_ok=True)

        # Save index (if method exists)
        if hasattr(indexer, "save_multi_table_index"):
            indexer.save_multi_table_index(table_indices, mode_index_dir)
        elif str(mode_name).startswith("text2sql"):
            # Text2SQL mode doesn't need to save indices, just schema metadata
            pass

        # Copy schema metadata
        schema_copy_path = mode_index_dir / "schema_metadata.json"
        shutil.copy(schema_metadata_path, schema_copy_path)

        return mode_index_dir
