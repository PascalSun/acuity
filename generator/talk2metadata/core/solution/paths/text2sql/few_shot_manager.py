"""Manager for dynamic few-shot examples in Text2SQL."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer, util

from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


class FewShotManager:
    """Manages few-shot examples for Text2SQL.

    Indexes Q&A pairs and retrieves the most similar ones to a new query
    to serve as in-context examples.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize manager.

        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self.model = None
        self.examples: List[Dict[str, str]] = []
        self.embeddings = None

    def _load_model(self):
        """Lazy load the model."""
        if self.model is None:
            logger.info(f"Loading FewShot embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def load_examples_from_file(self, file_path: Path):
        """Load examples from a JSON file.

        File should be a list of dicts with 'question' and 'sql' keys.
        Or the format of the QA pairs: {'question': ..., 'query': ...}
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"Few-shot examples file not found: {file_path}")
            return

        with open(file_path, "r") as f:
            data = json.load(f)

        # Handle wrapper format (dictionary with 'qa_pairs' key)
        if isinstance(data, dict) and "qa_pairs" in data:
            data = data["qa_pairs"]

        # Normalize data format and extract schema metadata
        normalized = []
        for item in data:
            question = item.get("question")
            sql = item.get("query") or item.get("sql")
            if question and sql:
                normalized.append(
                    {
                        "question": question,
                        "sql": sql,
                        "explanation": item.get("explanation", ""),
                        "tables_used": self._extract_tables_from_sql(sql),
                        "num_joins": sql.upper().count("JOIN"),
                        "has_subquery": sql.upper().count("SELECT") > 1,
                    }
                )

        self.add_examples(normalized)

    def add_examples(self, examples: List[Dict[str, str]]):
        """Add examples and rebuild index.

        Args:
            examples: List of dicts with 'question' and 'sql'
        """
        if not examples:
            return

        self._load_model()
        self.examples.extend(examples)

        # Build embeddings for questions
        questions = [ex["question"] for ex in examples]
        new_embeddings = self.model.encode(questions, convert_to_tensor=True)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            import torch

            self.embeddings = torch.cat([self.embeddings, new_embeddings])

        logger.info(f"Indexed {len(examples)} few-shot examples")

    def retrieve(
        self, query: str, k: int = 3, schema_context: Optional[Dict] = None
    ) -> List[Dict[str, str]]:
        """Retrieve k most similar examples for the query.

        Args:
            query: User question
            k: Number of examples to return
            schema_context: Optional dict with 'tables' list for schema-aware ranking

        Returns:
            List of matching examples
        """
        if not self.examples or self.model is None:
            return []

        # Encode query
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Compute cosine similarity
        scores = util.cos_sim(query_embedding, self.embeddings)[0]

        # Get top k*3 candidates for reranking
        candidate_count = min(k * 3, len(self.examples))
        top_candidates_indices = scores.argsort(descending=True)[:candidate_count]

        candidates = []
        for idx in top_candidates_indices:
            idx = int(idx)
            candidates.append((idx, self.examples[idx]))

        # If schema context provided, rerank by schema similarity
        if schema_context and "tables" in schema_context:
            candidates = self._rerank_by_schema(candidates, schema_context)

        # Return top k after reranking
        results = [ex for _, ex in candidates[:k]]
        return results

    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        """Extract table names from SQL query."""
        sql_upper = sql.upper()
        tables = set()

        # Pattern: FROM table_name or JOIN table_name
        patterns = [
            r"FROM\s+(\w+)",
            r"JOIN\s+(\w+)",
            r"INTO\s+(\w+)",
            r"UPDATE\s+(\w+)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, sql_upper)
            tables.update(matches)

        return list(tables)

    def _rerank_by_schema(
        self, candidates: List[tuple], schema_context: Dict
    ) -> List[tuple]:
        """Rerank candidates by schema similarity.

        Args:
            candidates: List of (index, example) tuples
            schema_context: Dict with 'tables' key containing list of table names

        Returns:
            Reranked list of (index, example) tuples
        """
        query_tables = set(t.upper() for t in schema_context.get("tables", []))
        num_query_tables = len(query_tables)
        has_query_joins = num_query_tables > 1

        def schema_score(candidate_tuple):
            _, example = candidate_tuple
            ex_tables = set(t.upper() for t in example.get("tables_used", []))
            ex_num_tables = len(ex_tables)
            ex_has_joins = example.get("num_joins", 0) > 0

            score = 0.0

            # 1. Prefer similar table count (weight: 2)
            table_count_diff = abs(ex_num_tables - num_query_tables)
            score -= table_count_diff * 2

            # 2. Prefer same join pattern (weight: 5)
            if ex_has_joins == has_query_joins:
                score += 5

            # 3. Boost for common tables (weight: 3 per table)
            common_tables = query_tables & ex_tables
            score += len(common_tables) * 3

            # 4. Slight penalty for complexity mismatch
            if example.get("has_subquery", False) and num_query_tables <= 1:
                score -= 1  # Don't show complex examples for simple queries

            return score

        # Sort by schema score (descending)
        reranked = sorted(candidates, key=schema_score, reverse=True)

        logger.debug(f"Reranked {len(candidates)} candidates by schema similarity")
        return reranked

    def format_for_prompt(self, examples: List[Dict[str, str]]) -> str:
        """Format examples for inclusion in prompt."""
        if not examples:
            return ""

        parts = ["## Similar Past Examples"]
        parts.append(
            "Here are similar questions and their CORRECT SQL queries. Use them as a reference for syntax and logic.\n"
        )

        for i, ex in enumerate(examples):
            parts.append(f"### Example {i+1}")
            parts.append(f"Question: {ex['question']}")
            parts.append(f"SQL: {ex['sql']}")
            if ex.get("explanation"):
                parts.append(f"Note: {ex['explanation']}")
            parts.append("")

        return "\n".join(parts)
