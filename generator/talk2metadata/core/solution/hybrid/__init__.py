"""Hybrid mode - combine components across paths."""

from __future__ import annotations

from ..modes.registry import register_mode
from ..paths.text2sql.indexer import Indexer as Text2SQLIndexer
from .retriever import HybridRetriever

register_mode(
    name="hybrid",
    description="Hybrid: semantic recall + structured filtering",
    indexer_class=Text2SQLIndexer,
    retriever_class=HybridRetriever,
    enabled=True,
)

__all__ = ["HybridRetriever"]
