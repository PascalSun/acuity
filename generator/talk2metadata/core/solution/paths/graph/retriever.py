"""Graph retriever: knowledge-graph only, no SQL at query time.

Builds a KG at index time (nodes = rows, edges = FKs). At query time:
tokenize -> seed nodes by token index -> BFS to target nodes -> return top-k.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

from talk2metadata.core.schema.schema import SchemaMetadata

from ...modes.registry import BaseRetriever
from ..lexical.retriever import _clean_bm25_tokens, _tokenize
from ..semantic.search_result import SearchResult
from .knowledge_graph import KnowledgeGraph


@dataclass
class GraphSearchResult(SearchResult):
    """Search result with optional graph metadata."""

    match_count: int = 0
    matched_tables: List[str] = field(default_factory=list)


class GraphRetriever(BaseRetriever):
    """Retriever that uses only the pre-built knowledge graph (no SQL)."""

    def __init__(
        self,
        schema_metadata: SchemaMetadata,
        kg: KnowledgeGraph,
        max_hops: int = 6,
        seed_max_nodes: int = 3000,
    ):
        self.schema_metadata = schema_metadata
        self.kg = kg
        self.max_hops = max(1, int(max_hops))
        self.seed_max_nodes = seed_max_nodes

    @classmethod
    def from_paths(
        cls,
        base_dir: str | Path,
        schema_metadata_path: str | Path,
        per_table_top_k: int = 10,
        max_hops: int = 6,
        traversal_fetch_limit: int = 200,
        seed_max_nodes: int = 3000,
        **kwargs: Any,
    ) -> GraphRetriever:
        schema_metadata = SchemaMetadata.load(schema_metadata_path)
        base_dir = Path(base_dir)
        kg_path = base_dir / "knowledge_graph.pkl"
        if kg_path.exists():
            kg = KnowledgeGraph.load(kg_path)
        else:
            db_path = base_dir / "metadata.duckdb"
            if not db_path.exists():
                raise FileNotFoundError(
                    f"Knowledge graph not found at {kg_path} and no DuckDB at {db_path}. "
                    "Run 'talk2metadata search prepare --mode graph' first."
                )
            kg = KnowledgeGraph.build_from_duckdb(db_path, schema_metadata)
            kg.save(kg_path)
        return cls(
            schema_metadata=schema_metadata,
            kg=kg,
            max_hops=max_hops,
            seed_max_nodes=seed_max_nodes,
        )

    def search(self, query: str, top_k: int = 5) -> List[GraphSearchResult]:
        # Tokenize (no SQL)
        tokens = _clean_bm25_tokens(query)
        if not tokens:
            tokens = _tokenize(query)
        if not tokens:
            return []

        # Seed nodes from token index (no SQL)
        seed_nodes = self.kg.get_seed_nodes(
            tokens,
            min_tokens=1,
            max_nodes=self.seed_max_nodes,
        )
        if not seed_nodes:
            return []

        # Graph search: BFS from seeds to target nodes (no SQL)
        target_scores = self.kg.search_to_target(
            seed_nodes,
            max_hops=self.max_hops,
            top_k=top_k,
        )
        if not target_scores:
            return []

        results: List[GraphSearchResult] = []
        for rank, (row_id, score) in enumerate(target_scores, 1):
            results.append(
                GraphSearchResult(
                    row_id=row_id,
                    table=self.kg.target_table,
                    data={"pk_value": row_id},
                    score=-float(score),
                    rank=rank,
                    match_count=0,
                    matched_tables=[],
                )
            )
        return results
