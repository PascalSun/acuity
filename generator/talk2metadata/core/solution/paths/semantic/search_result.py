"""Search result data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class SearchResult:
    """Search result for a single record.

    Attributes:
        row_id: Row identifier (int or str)
        table: Table name
        data: Row data as dictionary
        score: Similarity score (lower is better for L2 distance)
        rank: Result rank (1-indexed)
    """

    row_id: int | str
    table: str
    data: Dict
    score: float  # Similarity score (lower is better for L2 distance)
    rank: int  # Result rank (1-indexed)

    def __repr__(self) -> str:
        return f"SearchResult(rank={self.rank}, table={self.table}, row_id={self.row_id}, score={self.score:.4f})"
