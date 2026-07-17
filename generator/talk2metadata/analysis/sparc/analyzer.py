"""Analyze SParC benchmark queries for CEJSQ coverage.

SParC (Yu et al., 2019) — Sequential NL2SQL benchmark.
  - 4,298 interactions / 12,726 question-SQL pairs (train + dev).
  - Same 200 databases as Spider (166 unique in our HF source).
  - Queries come in conversation sequences: Q1 → Q2 (follow-up) → Q3...
  - Earlier turns often lack WHERE clauses; later turns reference prior context.

Key difference from Spider:
  - Each question is part of a sequence, so CEJSQ distribution shifts toward
    simpler patterns (fewer filters in early turns, follow-ups add joins).

HF source: AayushShah/SQL_SparC_Dataset_With_Schema
  - Schema is embedded as simplified DDL (no explicit FK constraints).
  - Topology reuses Spider's FK graph (same databases).
"""

from __future__ import annotations

from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)

HF_DATASET = "AayushShah/SQL_SparC_Dataset_With_Schema"


def load_queries() -> list[dict]:
    """Load SParC queries from HuggingFace, normalised to Spider query format."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise RuntimeError("Install 'datasets': uv pip install datasets") from e

    logger.info(f"Loading {HF_DATASET}...")
    ds = load_dataset(HF_DATASET, split="train")
    logger.info(f"Loaded {len(ds)} SParC examples")

    queries = [
        {
            "query": row["query"],
            "db_id": row["database_id"],
            "question": row["question"],
        }
        for row in ds
    ]
    return queries
