"""QA pair definition with comprehensive metadata.

Contains question, answer SQL, answer record IDs, strategy, and related metadata.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _generate_uid() -> str:
    """Generate a new UUID4 string for a QA pair."""
    return str(uuid.uuid4())


def _uid_from_sql(sql: str) -> str:
    """Deterministic UID derived from the SQL query (backward compat)."""
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()[:16]


@dataclass
class QAPair:
    """A question-answer pair for evaluation."""

    # Core QA data
    question: str  # Natural language question
    answer_row_ids: List[Any]  # List of target table row IDs (the answer)
    sql: str  # SQL query that produces the answer

    # Strategy and difficulty
    strategy: str  # Difficulty code (e.g., "2iM")
    difficulty_score: float  # Numeric difficulty score

    # Related tables and columns
    involved_tables: List[str]  # All tables involved in the query
    involved_columns: List[str]  # All columns used in filters (table.column format)
    involved_filters: List[Dict[str, Any]] = field(default_factory=list)

    # Validation status
    is_valid: Optional[bool] = None  # Whether this QA pair passed validation
    validation_errors: List[str] = field(default_factory=list)

    # SQL validation status
    sql_valid: Optional[bool] = None  # Whether SQL executed successfully
    sql_validation_error: Optional[str] = None  # Error message if SQL execution failed

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    answer_table: Optional[str] = None
    answer_id_column: Optional[str] = None

    # Stable unique identifier (assigned at generation, persisted across saves)
    uid: Optional[str] = None

    def ensure_uid(self) -> str:
        """Return the uid, generating one if missing.

        For newly generated QA pairs the uid is set at creation time (UUID4).
        For legacy QA pairs loaded from JSON without a uid field, a
        deterministic uid is derived from the SQL query hash.
        """
        if not self.uid:
            self.uid = _uid_from_sql(self.sql)
        return self.uid

    @property
    def answer_count(self) -> int:
        """Number of answer records."""
        return len(self.answer_row_ids)

    @property
    def tier(self) -> str:
        """Difficulty tier (easy/medium/hard/expert).

        Baseline generators set difficulty_score=-1.0 (no taxonomy score);
        those pairs report "unknown" rather than silently masquerading as easy.
        """
        if self.difficulty_score < 0.0:
            return "unknown"
        if self.difficulty_score < 1.0:
            return "easy"
        elif self.difficulty_score < 2.0:
            return "medium"
        elif self.difficulty_score < 3.0:
            return "hard"
        else:
            return "expert"

    def __repr__(self) -> str:
        return (
            f"QAPair(question='{self.question[:50]}...', "
            f"answers={len(self.answer_row_ids)}, "
            f"strategy={self.strategy}, "
            f"valid={self.is_valid})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        involved_values: Dict[str, Any] = {}
        for f in self.involved_filters:
            table = f.get("table")
            column = f.get("column")
            if table and column and "value" in f:
                involved_values[f"{table}.{column}"] = f.get("value")

        return {
            "uid": self.ensure_uid(),
            "question": self.question,
            "answer_row_ids": self.answer_row_ids,
            "answer_count": self.answer_count,
            "sql": self.sql,
            "strategy": self.strategy,
            "difficulty_score": self.difficulty_score,
            "tier": self.tier,
            "involved_tables": self.involved_tables,
            "involved_columns": self.involved_columns,
            "involved_filters": self.involved_filters,
            "involved_values": involved_values,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "sql_valid": self.sql_valid,
            "sql_validation_error": self.sql_validation_error,
            "metadata": self.metadata,
            "answer_table": self.answer_table,
            "answer_id_column": self.answer_id_column,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QAPair:
        """Create QAPair from dictionary.

        Args:
            data: Dictionary with QA pair data

        Returns:
            QAPair instance
        """
        sql = data["sql"]
        answer_id_column = data.get("answer_id_column")
        if not answer_id_column and isinstance(sql, str):
            m = re.search(r"select\s+([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)", sql, re.I)
            if m:
                answer_id_column = m.group(2).lower()

        # Read uid if present; otherwise derive deterministically from SQL
        uid = data.get("uid") or _uid_from_sql(sql)

        return cls(
            question=data["question"],
            answer_row_ids=data["answer_row_ids"],
            sql=sql,
            strategy=data["strategy"],
            difficulty_score=data["difficulty_score"],
            involved_tables=data.get("involved_tables", []),
            involved_columns=data.get("involved_columns", []),
            involved_filters=data.get("involved_filters", []),
            is_valid=data.get("is_valid"),
            validation_errors=data.get("validation_errors", []),
            sql_valid=data.get("sql_valid"),
            sql_validation_error=data.get("sql_validation_error"),
            metadata=data.get("metadata", {}),
            answer_table=data.get("answer_table")
            or data.get("metadata", {}).get("target_table"),
            answer_id_column=answer_id_column,
            uid=uid,
        )
