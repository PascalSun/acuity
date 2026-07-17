"""Shared retriever instance for MCP tools."""

from __future__ import annotations

from pathlib import Path

from talk2metadata.cli.handlers.search_handler import SearchHandler
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)

_retriever_cache: dict[str, object] = {}


def get_retriever(
    index_dir: str | Path | None = None,
    run_id: str | None = None,
    mode_name: str | None = None,
) -> object:
    config = get_config()
    mode_name = mode_name or "semantic"

    resolved_run_id = run_id or config.get("run_id")
    resolved_index_dir = str(Path(index_dir).resolve()) if index_dir else ""

    cache_key = f"{resolved_run_id or ''}:{resolved_index_dir}:{mode_name}"
    cached = _retriever_cache.get(cache_key)
    if cached is not None:
        return cached

    search_handler = SearchHandler(config)
    loaded = search_handler.load_retriever(
        mode_name=mode_name,
        index_dir=Path(index_dir) if index_dir else None,
        run_id=resolved_run_id,
    )
    _retriever_cache[cache_key] = loaded

    logger.info(f"Loaded {mode_name} retriever (run_id={resolved_run_id})")
    return loaded
