"""Record Voter Retriever for cross-table search with voting mechanism."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from talk2metadata.core.schema.schema import SchemaMetadata
from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.timing import TimingContext, timed

from .search_result import SearchResult

# Disable multiprocessing on macOS to avoid segmentation fault
# This is a known issue with sentence-transformers on macOS ARM
if os.name == "posix" and os.uname().sysname == "Darwin":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"

logger = get_logger(__name__)


@dataclass
class RecordVoteSearchResult(SearchResult):
    """Search result with voting metadata.

    Attributes:
        match_count: Number of votes (matches) this target row received
        matched_tables: List of table names that voted for this row
    """

    match_count: int  # Number of votes this target row received
    matched_tables: List[str]  # Tables that voted for this row

    def __repr__(self) -> str:
        return (
            f"RecordVoteSearchResult(rank={self.rank}, table={self.table}, "
            f"row_id={self.row_id}, score={self.score:.4f}, "
            f"votes={self.match_count}, voters={self.matched_tables})"
        )


class RecordVoter:
    """Record Voter Retriever for cross-table search with voting mechanism using DuckDB."""

    def __init__(
        self,
        table_indices: Dict[str, Tuple[faiss.IndexFlatL2, Any]],
        schema_metadata: SchemaMetadata,
        db_path: str,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize: bool = True,
        per_table_top_k: int = 5,
        use_reranking: bool = False,
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """Initialize Record Voter retriever.

        Args:
            table_indices: Dict mapping table_name -> (FAISS index, records_placeholder)
            schema_metadata: Schema metadata with FK relationships
            db_path: Path to DuckDB database file
            model_name: Sentence-transformers model name
            device: Device for encoding queries
            normalize: Whether to normalize query embeddings
            per_table_top_k: Number of results per table before voting
            use_reranking: Whether to use Cross-Encoder reranking
            reranker_model_name: Model name for reranker
        """
        # Unwrap indices (records are in DuckDB)
        # Handle case where table_indices might be just indices or tuples
        self.table_indices = {}
        for k, v in table_indices.items():
            if isinstance(v, tuple):
                self.table_indices[k] = v[0]
            else:
                self.table_indices[k] = v

        self.schema_metadata = schema_metadata
        self.target_table = schema_metadata.target_table
        self.per_table_top_k = per_table_top_k
        self.db_path = db_path
        self.use_reranking = use_reranking

        # Use provided parameters or defaults
        self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self.device = device
        self.normalize = normalize if normalize is not None else True

        logger.info(f"Loading embedding model for queries: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)

        if self.use_reranking:
            logger.info(f"Loading reranker model: {reranker_model_name}")
            self.reranker = CrossEncoder(reranker_model_name, device=self.device)
        else:
            self.reranker = None

        self.con = None

    def _connect(self) -> duckdb.DuckDBPyConnection:
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                try:
                    con = duckdb.connect(self.db_path, read_only=True)
                except TypeError:
                    con = duckdb.connect(
                        self.db_path, config={"access_mode": "READ_ONLY"}
                    )

                try:
                    con.execute("LOAD json;")
                except Exception:
                    con.execute("INSTALL json; LOAD json;")

                return con
            except Exception as e:
                last_error = e
                msg = str(e)
                is_lock = (
                    "Could not set lock on file" in msg
                    or "Conflicting lock is held" in msg
                )
                if not is_lock:
                    raise

                time.sleep(min(0.2 * (2**attempt), 2.0))

        raise RuntimeError(
            f"DuckDB is locked and cannot be opened (read-only) at {self.db_path}: {last_error}"
        )

    @classmethod
    @timed("record_voter.load", log_level="info")
    def from_paths(
        cls,
        base_dir: str | Path,
        schema_metadata_path: str | Path,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        per_table_top_k: int = 5,
        use_reranking: bool = False,
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> RecordVoter:
        """Create retriever from saved multi-table indices."""
        from talk2metadata.core.schema.schema import SchemaMetadata

        from .indexer import Indexer

        with TimingContext("multi_table_index_load"):
            table_indices = Indexer.load_multi_table_index(base_dir)

        with TimingContext("schema_metadata_load"):
            schema_metadata = SchemaMetadata.load(schema_metadata_path)

        return cls(
            table_indices=table_indices,
            schema_metadata=schema_metadata,
            db_path=str(Path(base_dir) / "metadata.duckdb"),
            model_name=model_name,
            device=device,
            per_table_top_k=per_table_top_k,
            use_reranking=use_reranking,
            reranker_model_name=reranker_model_name,
        )

    @timed("record_voter.search")
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RecordVoteSearchResult]:
        """Search across all tables and aggregate results via voting mechanism."""
        logger.debug(
            f"Record voter search for: '{query}' (top_k={top_k}, per_table_top_k={self.per_table_top_k})"
        )

        # 1. Encode query
        with TimingContext("query_encoding"):
            query_embedding = self._encode_query(query)

        # 2. Search FAISS indices for matches
        # We use a larger pool from each table to ensure good voting candidates
        search_k = max(top_k * 2, self.per_table_top_k)
        with TimingContext("multi_table_search"):
            all_matches = self._search_faiss_indices(query_embedding, search_k)

        # 3. Aggregate results via DuckDB voting
        # If reranking is enabled, retrieve more candidates first (e.g. 5x top_k)
        initial_k = top_k * 5 if self.use_reranking else top_k

        with TimingContext("voting"):
            con = self._connect()
            try:
                aggregated = self._vote_and_aggregate(con, all_matches, initial_k)
            finally:
                try:
                    con.close()
                except Exception:
                    pass

        # 4. Rerank if enabled
        if self.use_reranking and aggregated:
            with TimingContext("reranking"):
                aggregated = self._rerank_results(query, aggregated, top_k)

        logger.debug(f"Record voter search returned {len(aggregated)} results")
        return aggregated

    def _rerank_results(
        self, query: str, results: List[RecordVoteSearchResult], top_k: int
    ) -> List[RecordVoteSearchResult]:
        """Rerank results using Cross-Encoder."""
        if not results:
            return []

        # Reconstruct text representation for each result
        # We use a simple format similar to indexer but for the target record
        pairs = []
        for result in results:
            text_parts = [f"Table: {result.table}"]
            for k, v in result.data.items():
                if v:
                    text_parts.append(f"{k}: {v}")
            text = "\n".join(text_parts)
            # Truncate text if too long to avoid token limit errors
            if len(text) > 1000:
                text = text[:1000]
            pairs.append([query, text])

        # Predict scores
        scores = self.reranker.predict(pairs)

        # Update scores and re-sort
        for result, score in zip(results, scores):
            # We keep the vote count but use reranker score for final ranking
            # Normalize score might be needed, but raw logits are usually fine for ranking
            result.score = float(score)  # Convert numpy float to python float

        # Sort by new score (descending)
        results.sort(key=lambda x: x.score, reverse=True)

        # Assign new ranks
        for i, result in enumerate(results):
            result.rank = i + 1

        return results[:top_k]

    def _search_faiss_indices(
        self, query_embedding: np.ndarray, k: int
    ) -> Dict[str, List[Tuple[int, float]]]:
        """Search FAISS indices and return matches."""
        all_matches = {}

        for table_name, index in self.table_indices.items():
            try:
                # Debug logging for input shapes
                logger.debug(
                    f"Searching index for {table_name}: shape={query_embedding.shape}, dtype={query_embedding.dtype}"
                )
                if query_embedding.ndim == 1:
                    query_embedding = query_embedding.reshape(1, -1)

                query_dim = int(query_embedding.shape[1])
                index_dim = getattr(index, "d", None)
                if index_dim is not None and int(index_dim) != query_dim:
                    logger.error(
                        "FAISS dimension mismatch for %s: query_dim=%s index_dim=%s model=%s. "
                        "Rebuild indexes (talk2metadata search prepare) or set modes.semantic.indexer.model_name to match the index.",
                        table_name,
                        query_dim,
                        int(index_dim),
                        self.model_name,
                    )
                    continue

                distances, indices = index.search(query_embedding, k)

                matches = []
                for distance, idx in zip(distances[0], indices[0]):
                    if idx == -1:
                        break
                    matches.append((int(idx), float(distance)))

                all_matches[table_name] = matches
                logger.info(f"Found {len(matches)} matches in {table_name}")
            except Exception as e:
                logger.error(f"FAISS search failed for {table_name}: {e}")
                import traceback

                logger.error(traceback.format_exc())
                continue

        return all_matches

    @timed("voting")
    def _vote_and_aggregate(
        self,
        con: duckdb.DuckDBPyConnection,
        all_matches: Dict[str, List[Tuple[int, float]]],
        top_k: int,
    ) -> List[RecordVoteSearchResult]:
        """Aggregate search results via voting mechanism using DuckDB."""
        # Create temp table for matches
        con.execute(
            "CREATE TEMPORARY TABLE IF NOT EXISTS matches (source_table VARCHAR, faiss_id INTEGER, score FLOAT)"
        )
        con.execute("DELETE FROM matches")

        # Prepare batch insert
        match_data = []
        for table, items in all_matches.items():
            for idx, score in items:
                match_data.append((table, idx, score))

        if not match_data:
            return []

        con.executemany("INSERT INTO matches VALUES (?, ?, ?)", match_data)

        # Create temp votes table
        con.execute(
            "CREATE TEMPORARY TABLE IF NOT EXISTS votes (target_row_id VARCHAR, score FLOAT, voter_table VARCHAR)"
        )
        con.execute("DELETE FROM votes")

        # Perform voting logic using SQL joins
        matched_tables = set(m[0] for m in match_data)

        for table in matched_tables:
            # 1. Direct Vote (Target table votes for itself)
            if table == self.target_table:
                # Direct match: Table row IS the target
                # We join on faiss_id to get row_id from the table itself
                con.execute(
                    f"""
                    INSERT INTO votes
                    SELECT T.row_id, m.score, '{table}'
                    FROM matches m
                    JOIN {table} T ON m.faiss_id = T.faiss_id
                    WHERE m.source_table = '{table}'
                    """
                )
                continue

            # 2. FK Votes (Linked tables vote for target lookup)
            for fk in self.schema_metadata.foreign_keys:
                # Case A: Child (Source) -> Parent (Target)
                if fk.child_table == table and fk.parent_table == self.target_table:
                    # Join logic: extract FK from Source.data and match with Target.row_id
                    # Note: We assume Target.row_id IS the PK.
                    # DuckDB JSON extraction: json_extract_string(data, '$.key')
                    con.execute(
                        f"""
                        INSERT INTO votes
                        SELECT Target.row_id, m.score, '{table}'
                        FROM matches m
                        JOIN {table} Source ON m.faiss_id = Source.faiss_id
                        JOIN {self.target_table} Target ON json_extract_string(Source.data, '$.{fk.child_column}') = Target.row_id
                        WHERE m.source_table = '{table}'
                        """
                    )

                # Case B: Parent (Source) -> Child (Target)
                elif fk.parent_table == table and fk.child_table == self.target_table:
                    # Join logic: Source.row_id matches extracted FK from Target.data
                    con.execute(
                        f"""
                        INSERT INTO votes
                        SELECT Target.row_id, m.score, '{table}'
                        FROM matches m
                        JOIN {table} Source ON m.faiss_id = Source.faiss_id
                        JOIN {self.target_table} Target ON Source.row_id = json_extract_string(Target.data, '$.{fk.child_column}')
                        WHERE m.source_table = '{table}'
                        """
                    )

        # Aggregate results
        results = con.execute(
            f"""
            SELECT
                target_row_id,
                COUNT(*) as vote_count,
                MIN(score) as best_score,
                LIST(DISTINCT voter_table) as voters
            FROM votes
            GROUP BY target_row_id
            ORDER BY vote_count DESC, best_score ASC
            LIMIT {top_k}
            """
        ).fetchall()

        # Build final objects
        final_results = []
        for rank, (rid, vote_count, score, voters) in enumerate(results, 1):
            if not rid:
                continue

            # Fetch target row data
            # We use parameterized query for safety
            row_data_json = con.execute(
                f"SELECT data FROM {self.target_table} WHERE row_id = ?", [rid]
            ).fetchone()

            if row_data_json:
                data = json.loads(row_data_json[0])
                final_results.append(
                    RecordVoteSearchResult(
                        row_id=rid,
                        table=self.target_table,
                        data=data,
                        score=score,
                        rank=rank,
                        match_count=vote_count,
                        matched_tables=sorted(list(voters)),
                    )
                )

        return final_results

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a single query to embedding."""
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )

        return embedding.astype("float32")

    def get_stats(self) -> Dict:
        """Get retriever statistics."""
        # Query DuckDB for total records
        total_records = 0
        try:
            con = self._connect()
            for table in self.table_indices:
                count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                total_records += count
            try:
                con.close()
            except Exception:
                pass
        except Exception:
            pass

        return {
            "total_tables": len(self.table_indices),
            "total_records": total_records,
            "target_table": self.target_table,
            "per_table_top_k": self.per_table_top_k,
            "model": self.model_name,
            "db_path": self.db_path,
        }

    def __repr__(self) -> str:
        return (
            f"RecordVoter(tables={len(self.table_indices)}, "
            f"target={self.target_table}, model={self.model_name})"
        )
