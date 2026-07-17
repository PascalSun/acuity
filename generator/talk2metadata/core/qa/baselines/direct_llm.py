"""Baseline B — Direct LLM SQL generation (no taxonomy) for RQ2 ablation.

Prompts an LLM with full schema context to generate both SQL queries and
natural language questions in one shot. The SQL is then executed against
the actual database to extract answer row IDs and validate correctness.

Key differences from FlexBench:
- No strategy taxonomy (LLM decides query structure on its own)
- No proportional allocation (LLM naturally biases toward common patterns)
- No structural diversity control (no round-robin across JOIN patterns)
- Same LLM provider/model as FlexBench

This is a strong baseline: the LLM has full schema + FK + sample data context,
and its SQL is executed against a real sqlite database.

CAVEAT (comparison validity): this baseline does NOT run the QAVerifier
faithfulness/quality gate, and its execution backend (in-memory sqlite3)
differs from FlexBench's pandas-simulation + engine gold gate — acceptance
criteria are NOT identical across modes. Comparisons are therefore valid for
structural coverage/entropy only; for any validity/quality comparison,
re-validate all modes uniformly first.

Expected finding: valid SQL but lower strategy coverage entropy (LLM biases
toward easy single-table queries, similar to human annotators).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from talk2metadata.agent import AgentWrapper
from talk2metadata.core.qa.qa_pair import QAPair, _generate_uid
from talk2metadata.core.schema import SchemaMetadata
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)

_SCHEMA_SUMMARY_MAX_ROWS = 5
_BATCH_SIZE = 5
_MAX_ATTEMPTS = 20


class DirectLLMBaseline:
    """Baseline B: LLM generates both SQL and NL questions from schema context."""

    def __init__(
        self,
        schema: SchemaMetadata,
        tables: Dict[str, pd.DataFrame],
        agent: Optional[AgentWrapper] = None,
        max_answer_records: int = 15,
    ):
        self.schema = schema
        self.tables = tables
        self.agent = agent or AgentWrapper()
        self.max_answer_records = max_answer_records
        self._target_pk = schema.tables.get(schema.target_table.lower(), None)
        if self._target_pk is not None:
            self._target_pk = self._target_pk.primary_key or "id"
        else:
            self._target_pk = "id"

    def generate(self, n: int) -> List[QAPair]:
        """Generate n QA pairs by prompting LLM for SQL + question."""
        schema_summary = self._build_schema_summary()
        pairs: List[QAPair] = []
        rounds = 0
        max_rounds = (n // _BATCH_SIZE + 1) * _MAX_ATTEMPTS

        while len(pairs) < n and rounds < max_rounds:
            rounds += 1
            remaining = n - len(pairs)
            batch = min(_BATCH_SIZE, remaining)

            try:
                raw_pairs = self._prompt_for_qa_pairs(schema_summary, batch)
            except Exception as e:
                logger.warning(f"LLM call failed (round {rounds}): {e}")
                continue

            for rp in raw_pairs:
                if len(pairs) >= n:
                    break

                sql = rp.get("sql", "").strip()
                question = rp.get("question", "").strip()
                if not sql or not question:
                    continue

                # Validate SQL by executing against actual data
                answer_ids = self._execute_sql(sql)
                if answer_ids is None:
                    logger.debug(f"SQL execution failed: {sql[:80]}")
                    continue
                if len(answer_ids) == 0 or len(answer_ids) > self.max_answer_records:
                    logger.debug(
                        f"SQL returned {len(answer_ids)} rows (need 1-{self.max_answer_records}): {sql[:80]}"
                    )
                    continue

                pairs.append(
                    QAPair(
                        uid=_generate_uid(),
                        question=question,
                        answer_row_ids=answer_ids,
                        sql=sql,
                        strategy="direct_llm",
                        difficulty_score=-1.0,
                        involved_tables=self._extract_tables(sql),
                        involved_columns=[],
                        involved_filters=[],
                    )
                )

        logger.info(
            f"DirectLLM: generated {len(pairs)}/{n} valid pairs in {rounds} LLM rounds"
        )
        return pairs

    def _build_schema_summary(self) -> str:
        """Build a detailed schema description for the LLM prompt."""
        lines = [
            f"Database with target table: {self.schema.target_table}",
            f"Target table primary key: {self._target_pk}",
            "",
            "Tables and columns:",
        ]

        for tname, tmeta in self.schema.tables.items():
            pk_marker = f" (PK: {tmeta.primary_key})" if tmeta.primary_key else ""
            lines.append(f"  {tname}{pk_marker} [{tmeta.row_count} rows]:")
            for col, dtype in tmeta.columns.items():
                samples = tmeta.sample_values.get(col, [])[:_SCHEMA_SUMMARY_MAX_ROWS]
                sample_str = (
                    f" -- e.g. {', '.join(str(s) for s in samples)}" if samples else ""
                )
                lines.append(f"    {col} [{dtype}]{sample_str}")

        if self.schema.foreign_keys:
            lines.append("")
            lines.append("Foreign key relationships:")
            for fk in self.schema.foreign_keys:
                lines.append(
                    f"  {fk.child_table}.{fk.child_column} -> "
                    f"{fk.parent_table}.{fk.parent_column}"
                )

        return "\n".join(lines)

    def _prompt_for_qa_pairs(self, schema_summary: str, n: int) -> List[Dict[str, str]]:
        """Prompt LLM to generate n (question, SQL) pairs."""
        target = self.schema.target_table
        pk = self._target_pk

        prompt = f"""You are generating NL2SQL benchmark pairs for a database.

{schema_summary}

Generate exactly {n} pairs. Each pair has a natural language question and the SQL query that answers it.

Rules:
- SQL must be SELECT DISTINCT {target}.{pk} FROM {target} ... WHERE ...
- SQL must use only the tables and columns listed above
- Use proper JOIN syntax with the foreign keys shown above
- WHERE conditions should use real values from the sample data
- Questions should be diverse: vary the number of JOINs (0, 1, 2+), the number of WHERE conditions (1-5), and the tables involved
- Questions must be natural, specific, and unambiguous

Output as JSON array:
[
  {{"question": "...", "sql": "SELECT DISTINCT ..."}},
  ...
]

Output ONLY the JSON array, no other text."""

        response = self.agent.generate(prompt)
        raw = response.content.strip()
        return self._parse_json_pairs(raw)

    @staticmethod
    def _parse_json_pairs(text: str) -> List[Dict[str, str]]:
        """Parse JSON array of {question, sql} pairs from LLM response."""
        # Try to find JSON array in the response
        # Strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [
                    d
                    for d in data
                    if isinstance(d, dict) and "question" in d and "sql" in d
                ]
        except json.JSONDecodeError:
            pass

        # Fallback: try to find JSON array substring
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return [
                        d
                        for d in data
                        if isinstance(d, dict) and "question" in d and "sql" in d
                    ]
            except json.JSONDecodeError:
                pass

        logger.debug(f"Failed to parse LLM JSON response: {text[:200]}")
        return []

    def _execute_sql(self, sql: str) -> Optional[List[Any]]:
        """Execute SQL against in-memory DataFrames and return answer row IDs.

        Uses pandas to simulate SQL execution by parsing the query structure.
        Returns None on error, empty list if no results, or list of PKs.
        """
        target = self.schema.target_table.lower()
        target_df = self.tables.get(target)
        if target_df is None:
            return None

        try:
            # Try using sqlite3 in-memory database for accurate execution
            import sqlite3

            conn = sqlite3.connect(":memory:")

            # Load all tables into sqlite
            for tname, df in self.tables.items():
                df.to_sql(tname, conn, index=False, if_exists="replace")

            # Execute the SQL
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchmany(self.max_answer_records + 1)
            conn.close()

            if not rows:
                return []

            # Extract first column (should be the PK)
            return [row[0] for row in rows]

        except Exception as e:
            logger.debug(f"SQL execution error: {e}")
            return None

    @staticmethod
    def _extract_tables(sql: str) -> List[str]:
        """Extract table names from SQL FROM/JOIN clauses."""
        tables = []
        # FROM table
        from_match = re.search(r"\bFROM\s+(\w+)", sql, re.IGNORECASE)
        if from_match:
            tables.append(from_match.group(1).lower())
        # JOIN table
        for join_match in re.finditer(r"\bJOIN\s+(\w+)", sql, re.IGNORECASE):
            tables.append(join_match.group(1).lower())
        return tables
