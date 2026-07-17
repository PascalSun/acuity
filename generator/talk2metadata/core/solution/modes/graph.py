from __future__ import annotations

from ..paths.graph import GraphRetriever, Indexer
from .registry import register_mode

register_mode(
    name="graph",
    description="Graph retrieval: traverse foreign keys to connect entities across tables",
    indexer_class=Indexer,
    retriever_class=GraphRetriever,
    enabled=True,
)
