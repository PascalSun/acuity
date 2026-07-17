from __future__ import annotations

from ..paths.lexical import Indexer, LexicalRetriever
from .registry import register_mode

register_mode(
    name="lexical",
    description="Lexical retrieval: keyword matching over table values",
    indexer_class=Indexer,
    retriever_class=LexicalRetriever,
    enabled=True,
)
