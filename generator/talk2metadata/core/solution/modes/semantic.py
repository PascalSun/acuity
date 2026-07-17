from __future__ import annotations

from ..paths.semantic import Indexer, RecordVoter
from .registry import register_mode

register_mode(
    name="semantic",
    description="Semantic retrieval: Record-level embedding with voting-based cross-table search (RecordVoter)",
    indexer_class=Indexer,
    retriever_class=RecordVoter,
    enabled=True,
)
