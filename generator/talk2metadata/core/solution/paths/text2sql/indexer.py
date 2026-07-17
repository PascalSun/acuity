"""Indexer for text2sql mode.

For text2sql mode, we don't need to build embeddings or indices.
The indexer is mainly a placeholder to satisfy the mode interface.
"""

from __future__ import annotations

from typing import Any, Dict

from talk2metadata.core.schema.schema import SchemaMetadata
from talk2metadata.utils.logging import get_logger

from ...modes.registry import BaseIndexer

logger = get_logger(__name__)


class Indexer(BaseIndexer):
    """Indexer for text2sql mode.

    This indexer doesn't build any indices since text2sql generates SQL queries
    dynamically. It's mainly a placeholder to satisfy the mode interface.
    """

    def build_index(
        self, tables: Dict, schema_metadata: SchemaMetadata, **kwargs
    ) -> Dict[str, Any]:
        """Build index for text2sql mode.

        For text2sql, we don't need to build indices. We just return the schema metadata
        which will be used by the retriever to generate SQL queries.

        Args:
            tables: Dict of table_name -> DataFrame (not used for text2sql)
            schema_metadata: Schema metadata with table structures
            **kwargs: Additional arguments

        Returns:
            Dict containing schema metadata (for compatibility with mode interface)
        """
        logger.info("Text2SQL mode: No index building needed, using schema metadata")
        return {
            "schema_metadata": schema_metadata,
            "tables": tables,  # Keep tables for reference if needed
        }
