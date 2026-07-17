"""Indexing and retrieval modes for Talk2Metadata."""

# Modes are registered lazily in registry._ensure_modes_registered() to avoid
# loading heavy dependencies (sentence-transformers, etc.) at CLI startup.
from .registry import (
    ModeRegistry,
    get_active_mode,
    get_mode,
    get_mode_config,
    get_mode_indexer_config,
    get_mode_retriever_config,
    get_registry,
    register_mode,
    resolve_index_mode,
)

__all__ = [
    "ModeRegistry",
    "get_active_mode",
    "get_mode",
    "get_mode_config",
    "get_mode_indexer_config",
    "get_mode_retriever_config",
    "get_registry",
    "register_mode",
    "resolve_index_mode",
]
