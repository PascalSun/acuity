"""Path utilities for managing run_id-based directory structure."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from talk2metadata.utils.config import get_config


def sanitize_run_id(run_id: str) -> str:
    """Sanitize run_id for use in file paths.

    Args:
        run_id: Run ID string

    Returns:
        Sanitized run ID safe for filesystem
    """
    return re.sub(r"[^\w\-_.]", "_", str(run_id))


def infer_run_id_from_path(source_path: str | Path) -> Optional[str]:
    source_path_str = str(source_path)
    if "://" in source_path_str and not source_path_str.startswith("file://"):
        return None

    p = Path(source_path_str.replace("file://", ""))

    if p.suffix:
        if p.parent.name.lower() == "raw":
            return (
                sanitize_run_id(p.parent.parent.name) if p.parent.parent.name else None
            )
        if p.parent.name:
            return sanitize_run_id(p.parent.name)
        return None

    leaf_dirs = {"raw", "processed", "metadata", "indexes", "qa", "benchmark", "db"}
    if p.name.lower() in leaf_dirs:
        return sanitize_run_id(p.parent.name) if p.parent.name else None

    return sanitize_run_id(p.name) if p.name else None


def get_run_id_from_config(config=None) -> Optional[str]:
    """Get run_id from config, supporting both simple and extended formats.

    Supports:
    - Simple format: run_id: "wamex_run"
    - Extended format: run: { id: "wamex_run", ... }

    Args:
        config: Optional config instance. If None, uses get_config()

    Returns:
        Run ID string or None
    """
    if config is None:
        config = get_config()

    # Try extended format first: run.id
    run_config = config.get("run", {})
    if isinstance(run_config, dict) and "id" in run_config:
        return run_config["id"]

    # Fall back to simple format: run_id
    return config.get("run_id")


def get_run_base_dir(
    run_id: Optional[str] = None, base_dir: Optional[Path] = None, config=None
) -> Path:
    """Get base directory for a run_id.

    If run_id is provided, returns {output_dir}/{run_id}/, otherwise returns {output_dir}/.
    Can read output_dir from config.run.output_dir or base_dir from config.run.base_dir.

    Args:
        run_id: Optional run ID. If None, uses default data directory structure.
        base_dir: Optional base directory. If None, tries to read from config.run.output_dir
                  or config.run.base_dir, otherwise defaults to "./data"
        config: Optional config instance. If None, uses get_config()

    Returns:
        Path to run base directory
    """
    if base_dir is None:
        if config is None:
            from talk2metadata.utils.config import get_config

            config = get_config()
        # Try to get output_dir or base_dir from run config
        run_config = config.get("run", {})
        if isinstance(run_config, dict):
            # Prefer output_dir over base_dir
            output_dir_str = run_config.get("output_dir") or run_config.get("base_dir")
            if output_dir_str:
                base_dir = Path(output_dir_str)
            else:
                base_dir = Path("./data")
        else:
            base_dir = Path("./data")

    if run_id:
        run_id_safe = sanitize_run_id(run_id)
        return base_dir / run_id_safe
    return base_dir


def get_metadata_dir(run_id: Optional[str] = None, config=None) -> Path:
    """Get metadata directory path for a run_id.

    Args:
        run_id: Optional run ID
        config: Optional config instance. If None, uses get_config()

    Returns:
        Path to metadata directory
    """
    if config is None:
        config = get_config()

    if run_id:
        # Check for custom metadata_dir in run config
        run_config = config.get("run", {})
        if isinstance(run_config, dict) and "metadata_dir" in run_config:
            return Path(run_config["metadata_dir"])
        run_base = get_run_base_dir(run_id, config=config)
        return run_base / "metadata"
    else:
        return Path(config.get("data.metadata_dir", "./data/metadata"))


def get_processed_dir(run_id: Optional[str] = None, config=None) -> Path:
    """Get processed directory path for a run_id.

    Args:
        run_id: Optional run ID
        config: Optional config instance. If None, uses get_config()

    Returns:
        Path to processed directory
    """
    if config is None:
        config = get_config()

    if run_id:
        # Check for custom processed_dir in run config
        run_config = config.get("run", {})
        if isinstance(run_config, dict) and "processed_dir" in run_config:
            return Path(run_config["processed_dir"])
        run_base = get_run_base_dir(run_id, config=config)
        return run_base / "processed"
    else:
        return Path(config.get("data.processed_dir", "./data/processed"))


def get_indexes_dir(run_id: Optional[str] = None, config=None) -> Path:
    """Get indexes directory path for a run_id.

    Args:
        run_id: Optional run ID
        config: Optional config instance. If None, uses get_config()

    Returns:
        Path to indexes directory
    """
    if config is None:
        config = get_config()

    if run_id:
        # Check for custom indexes_dir in run config
        run_config = config.get("run", {})
        if isinstance(run_config, dict) and "indexes_dir" in run_config:
            return Path(run_config["indexes_dir"])
        run_base = get_run_base_dir(run_id, config=config)
        return run_base / "indexes"
    else:
        return Path(config.get("data.indexes_dir", "./data/indexes"))


def get_qa_dir(run_id: Optional[str] = None, config=None) -> Path:
    """Get QA directory path for a run_id.

    Args:
        run_id: Optional run ID
        config: Optional config instance. If None, uses get_config()

    Returns:
        Path to QA directory
    """
    if config is None:
        config = get_config()

    if run_id:
        # Check for custom qa_dir in run config
        run_config = config.get("run", {})
        if isinstance(run_config, dict) and "qa_dir" in run_config:
            return Path(run_config["qa_dir"])
        run_base = get_run_base_dir(run_id, config=config)
        return run_base / "qa"
    else:
        # Default to data/qa if no run_id
        return Path("./data/qa")


def get_benchmark_dir(run_id: Optional[str] = None, config=None) -> Path:
    """Get benchmark directory path for a run_id.

    Args:
        run_id: Optional run ID. If None, tries to get from config.
        config: Optional config instance. If None, uses get_config()

    Returns:
        Path to benchmark directory
    """
    if config is None:
        config = get_config()

    # Use run_id from config if not provided
    if run_id is None:
        run_id = get_run_id_from_config(config)

    if run_id:
        # Check for custom benchmark_dir in run config
        run_config = config.get("run", {})
        if isinstance(run_config, dict) and "benchmark_dir" in run_config:
            return Path(run_config["benchmark_dir"])
        run_base = get_run_base_dir(run_id, config=config)
        return run_base / "benchmark"
    else:
        # Default to data/benchmark if no run_id in config either
        return Path("./data/benchmark")


def get_db_dir(run_id: Optional[str] = None, config=None) -> Path:
    """Get database directory path for a run_id.

    Args:
        run_id: Optional run ID. If None, tries to get from config.
        config: Optional config instance. If None, uses get_config()

    Returns:
        Path to database directory
    """
    if config is None:
        config = get_config()

    # Use run_id from config if not provided
    if run_id is None:
        run_id = get_run_id_from_config(config)

    if run_id:
        # Check for custom db_dir in run config
        run_config = config.get("run", {})
        if isinstance(run_config, dict) and "db_dir" in run_config:
            return Path(run_config["db_dir"])
        run_base = get_run_base_dir(run_id, config=config)
        return run_base / "db"
    else:
        # Default to data/db if no run_id in config either
        return Path("./data/db")


def find_schema_file(metadata_dir: Path, target_table: Optional[str] = None) -> Path:
    """Find schema JSON file in metadata directory.

    If target_table is provided, looks for schema_{target_table}.json first.
    Otherwise, looks for schema.json first, then falls back to schema_*.json files.

    Args:
        metadata_dir: Path to metadata directory
        target_table: Optional target table name to construct filename

    Returns:
        Path to schema file

    Raises:
        FileNotFoundError: If no schema file is found
    """
    # If target_table is provided, try schema_{target_table}.json first
    if target_table:
        import re

        target_table_safe = re.sub(r"[^\w\-_.]", "_", target_table)
        schema_path = metadata_dir / f"schema_{target_table_safe}.json"
        if schema_path.exists():
            return schema_path

    # Fall back to schema_*.json files
    schema_files = list(metadata_dir.glob("schema_*.json"))
    if schema_files:
        # Return the first one found (or most recent if multiple)
        return sorted(schema_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    raise FileNotFoundError(
        f"No schema file found in {metadata_dir}. "
        "Expected schema.json or schema_*.json"
    )
