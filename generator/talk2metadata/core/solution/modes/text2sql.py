from __future__ import annotations

from ..paths.text2sql import (
    DirectText2SQLRetriever,
    Indexer,
    TwoStepText2SQLRetriever,
)
from ..paths.text2sql.finetuning import FinetunedRetriever
from .registry import register_mode

register_mode(
    name="text2sql",
    description="Text-to-SQL: Convert natural language questions to SQL queries and execute them",
    indexer_class=Indexer,
    retriever_class=DirectText2SQLRetriever,
    enabled=True,
)

register_mode(
    name="text2sql.two_step",
    description="Text-to-SQL (Two-step): Locate relevant columns/tables first, then generate SQL",
    indexer_class=Indexer,
    retriever_class=TwoStepText2SQLRetriever,
    enabled=True,
)

register_mode(
    name="text2sql.finetuning",
    description="Text-to-SQL using local fine-tuned models",
    indexer_class=Indexer,
    retriever_class=FinetunedRetriever,
    enabled=True,
)
