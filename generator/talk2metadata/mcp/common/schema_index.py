"""Shared schema metadata access for MCP tools."""

from __future__ import annotations

from pathlib import Path

from talk2metadata.core.schema import SchemaMetadata
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.paths import find_schema_file, get_metadata_dir

logger = get_logger(__name__)

_schema_cache: dict[str, SchemaMetadata] = {}


def get_schema(
    *,
    run_id: str | None = None,
    metadata_dir: str | Path | None = None,
    schema_path: str | Path | None = None,
) -> SchemaMetadata:
    config = get_config()

    if schema_path is None:
        if metadata_dir is None:
            resolved_run_id = run_id or config.get("run_id")
            metadata_dir_path = get_metadata_dir(resolved_run_id, config)
        else:
            metadata_dir_path = Path(metadata_dir)

        schema_path_obj = find_schema_file(
            metadata_dir_path,
            target_table=config.get("ingest.target_table"),
        )
    else:
        schema_path_obj = Path(schema_path)

    cache_key = str(schema_path_obj.resolve())
    cached = _schema_cache.get(cache_key)
    if cached is not None:
        return cached

    if not schema_path_obj.exists():
        raise FileNotFoundError(
            f"Schema not found at {schema_path_obj}. Please run 'talk2metadata schema ingest' first."
        )

    loaded = SchemaMetadata.load(schema_path_obj)
    _schema_cache[cache_key] = loaded
    logger.info(
        f"Loaded schema with {len(loaded.tables)} tables from {schema_path_obj}"
    )
    return loaded
