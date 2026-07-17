"""Direct Text2SQL Retriever - Question + schema → SQL → results."""

from __future__ import annotations

from typing import List, Tuple

from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.timing import timed

from .base import (
    BaseText2SQLRetriever,
    Text2SQLSearchResult,
)

logger = get_logger(__name__)


def _text2sql_system_prompt(target_table_name: str, id_column: str, top_k: int) -> str:
    """Single Text2SQL system prompt (used at inference and for openai finetune export)."""
    return f"""You are a SQL generator. You convert natural language questions into SQLite SQL.

You MUST respond with a JSON object containing exactly two fields:
- "thought": a brief reasoning about the query (1-2 sentences)
- "sql": the complete SQLite SQL query

Rules for the SQL:
- Use lowercase SQL keywords: select/from/join/where/limit.
- Always select {target_table_name}.{id_column} and include limit {top_k}.
- Use joins only when needed (follow FK relationships in schema).
- Text fields: use like '%value%' with lowercase values.
- ID fields ending with 'id' or 'ids': use = 'value' (not like).
- The "sql" field must NEVER be empty. Always generate a valid SQL query.
"""


def _text2sql_user_prompt(
    schema_text: str,
    question: str,
    target_table_name: str,
    id_column: str,
    top_k: int,
) -> str:
    """Single Text2SQL user prompt."""
    return f"""{schema_text}

## Question
{question}

## Task
Generate a SQLite SQL query that answers the question above.
The query must select {target_table_name}.{id_column} and include limit {top_k}.
"""


def build_prompts_for_finetuning(
    schema_text: str,
    question: str,
    target_table_name: str,
    id_column: str,
    top_k: int = 10,
) -> Tuple[str, str]:
    """Build system and user prompts (same format as Text2SQL inference)."""
    system_prompt = _text2sql_system_prompt(target_table_name, id_column, top_k)
    user_prompt = _text2sql_user_prompt(
        schema_text, question, target_table_name, id_column, top_k
    )
    return system_prompt, user_prompt


class DirectText2SQLRetriever(BaseText2SQLRetriever):
    """Direct text2sql retriever: Question + schema → SQL → results.

    This approach directly generates SQL from the question and schema information
    without first locating relevant columns/tables.
    """

    @timed("text2sql_direct.search")
    def search(self, query: str, top_k: int = 5) -> List[Text2SQLSearchResult]:
        """Search by converting question to SQL and executing it.

        Args:
            query: Natural language question
            top_k: Maximum number of results (used as LIMIT in SQL)

        Returns:
            List of Text2SQLSearchResult objects
        """
        # Log question
        logger.info("=" * 80)
        logger.info(f"🔍 QUESTION: {query}")
        logger.info("=" * 80)

        schema_text = BaseText2SQLRetriever.format_schema_for_prompt_compact_static(
            self.schema_metadata, context=getattr(self, "context", "")
        )
        target_table_name = self.target_table
        id_column = self._get_target_id_column() or "id"
        system_prompt = _text2sql_system_prompt(target_table_name, id_column, top_k)
        user_prompt_start = _text2sql_user_prompt(
            schema_text, query, target_table_name, id_column, top_k
        )

        # Execute Agentic Loop
        final_result = self._agentic_loop(
            query=query,
            base_system_prompt=system_prompt,
            user_prompt_start=user_prompt_start,
            top_k=top_k,
            target_table_name=target_table_name,
            id_column=id_column,
        )

        return final_result
