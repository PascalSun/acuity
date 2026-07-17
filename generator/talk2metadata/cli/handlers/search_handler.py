"""Business logic for search commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from talk2metadata.core.schema.schema import SchemaMetadata
from talk2metadata.core.solution.modes import (
    get_mode_indexer_config,
    get_mode_retriever_config,
    get_registry,
    resolve_index_mode,
)
from talk2metadata.core.solution.modes.comparison import ModeComparator
from talk2metadata.core.solution.paths.graph.retriever import GraphRetriever
from talk2metadata.core.solution.paths.lexical.retriever import LexicalRetriever
from talk2metadata.core.solution.paths.semantic.retriever import RecordVoter
from talk2metadata.core.solution.paths.semantic.search_result import SearchResult
from talk2metadata.core.solution.paths.text2sql import (
    DirectText2SQLRetriever,
    TwoStepText2SQLRetriever,
)
from talk2metadata.core.solution.paths.text2sql.few_shot_manager import FewShotManager
from talk2metadata.core.solution.paths.text2sql.finetuning import FinetunedRetriever
from talk2metadata.core.solution.preprocess import run_preprocess
from talk2metadata.utils.config import Config
from talk2metadata.utils.csv_to_db import get_or_create_db_connection
from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.paths import (
    find_schema_file,
    get_indexes_dir,
    get_metadata_dir,
)

logger = get_logger(__name__)


class SearchHandler:
    """Handler for search operations.

    Encapsulates business logic for searching and comparing across modes.
    """

    def __init__(self, config: Config):
        """Initialize handler.

        Args:
            config: Configuration instance
        """
        self.config = config
        self.registry = get_registry()

    def _get_text2sql_connection(
        self, schema_metadata: SchemaMetadata, run_id: Optional[str]
    ) -> str:
        """Resolve or build a database connection for text2sql modes."""
        ingest_config = self.config.get("ingest", {})
        return get_or_create_db_connection(
            ingest_config=ingest_config,
            schema_metadata=schema_metadata,
            run_id=run_id or self.config.get("run_id"),
        )

    def load_retriever(
        self,
        mode_name: str,
        index_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
        per_table_top_k: int = 5,
    ):
        """Load retriever for a specific mode.

        Args:
            mode_name: Mode name
            index_dir: Optional index directory
            run_id: Optional run ID
            per_table_top_k: Top-k per table for semantic mode

        Returns:
            Retriever instance

        Raises:
            FileNotFoundError: If index not found
            NotImplementedError: If mode retriever not implemented
        """
        # Determine index directory (aliases use base mode's index, e.g. text2sql.openai52 → text2sql.two_step)
        if not index_dir:
            base_index_dir = get_indexes_dir(
                run_id or self.config.get("run_id"), self.config
            )
            index_dir = base_index_dir / resolve_index_mode(mode_name)
        else:
            index_dir = Path(index_dir)
            if not (index_dir / "schema_metadata.json").exists():
                index_dir = index_dir / resolve_index_mode(mode_name)

        # Find schema file
        schema_path = index_dir / "schema_metadata.json"
        if not schema_path.exists():
            metadata_dir = get_metadata_dir(
                run_id or self.config.get("run_id"), self.config
            )
            # Get target_table from config to find correct schema file
            target_table = self.config.get("ingest.target_table")
            schema_path = find_schema_file(metadata_dir, target_table=target_table)
            if not schema_path or not Path(schema_path).exists():
                raise FileNotFoundError(
                    f"Schema metadata not found for mode '{mode_name}'"
                )

        # Get mode-specific config
        mode_retriever_config = get_mode_retriever_config(mode_name)

        # Load schema metadata (needed for text2sql modes)
        schema_metadata = SchemaMetadata.load(schema_path)

        mode_info = self.registry.get(mode_name)

        # Initialize retriever based on mode
        if mode_name == "semantic":
            # Get indexer config to extract model_name
            mode_indexer_config = get_mode_indexer_config(mode_name)
            return RecordVoter.from_paths(
                index_dir,
                schema_path,
                model_name=mode_indexer_config.get(
                    "model_name"
                ),  # Must match indexer model!
                per_table_top_k=per_table_top_k
                or mode_retriever_config.get("per_table_top_k", 5),
                use_reranking=mode_retriever_config.get("use_reranking", False),
                reranker_model_name=mode_retriever_config.get(
                    "reranker_model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"
                ),
            )
        elif str(mode_name).startswith("text2sql"):
            connection_string = self._get_text2sql_connection(schema_metadata, run_id)
            retriever_cls = mode_info.retriever_class if mode_info else None
            if retriever_cls is None:
                if mode_name == "text2sql":
                    retriever_cls = DirectText2SQLRetriever
                elif mode_name == "text2sql.two_step":
                    retriever_cls = TwoStepText2SQLRetriever
                elif mode_name == "text2sql.finetuning":
                    retriever_cls = FinetunedRetriever
                else:
                    retriever_cls = DirectText2SQLRetriever

            # For TWO_STEP, try to load vector retriever (Record Voter) for VASQL
            vector_retriever = None
            few_shot_manager = None

            is_two_step = False
            try:
                is_two_step = issubclass(retriever_cls, TwoStepText2SQLRetriever)
            except Exception:
                is_two_step = retriever_cls is TwoStepText2SQLRetriever

            if is_two_step:
                # 1. Load Vector Retriever (VASQL)
                try:
                    vector_index_dir = index_dir.parent / "semantic"
                    if (
                        vector_index_dir.exists()
                        and (vector_index_dir / "schema_metadata.json").exists()
                    ):
                        # Load record voter config to respect its settings
                        rv_config = get_mode_retriever_config("semantic")
                        rv_indexer_config = get_mode_indexer_config("semantic")

                        vector_retriever = RecordVoter.from_paths(
                            vector_index_dir,
                            vector_index_dir / "schema_metadata.json",
                            model_name=rv_indexer_config.get(
                                "model_name"
                            ),  # Must match indexer!
                            per_table_top_k=rv_config.get("per_table_top_k", 5),
                            # We can disable reranking for hints to save time, or enable for precision
                            use_reranking=False,
                        )
                except Exception as e:
                    # Non-fatal, just log and proceed without hints
                    logger.warning(f"Failed to load VASQL vector retriever: {e}")
                    pass

                # 2. Load FewShotManager (Agentic SQL)
                try:
                    # Look for qa/qa_pairs.json in the run directory
                    # index_dir is usually data/run_id/indexes/mode
                    # We want data/run_id/qa/qa_pairs.json
                    run_dir = index_dir.parent.parent
                    qa_file = run_dir / "qa" / "qa_pairs.json"

                    if qa_file.exists():
                        few_shot_manager = FewShotManager()
                        few_shot_manager.load_examples_from_file(qa_file)
                        logger.info(
                            f"Loaded {len(few_shot_manager.examples)} few-shot examples from {qa_file}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to load FewShotManager: {e}")

            return retriever_cls(
                schema_metadata=schema_metadata,
                connection_string=connection_string,
                mode_name=mode_name,
                vector_retriever=vector_retriever,  # Will be ignored by DirectText2SQLRetriever but accepted by TwoStep
                few_shot_manager=few_shot_manager,  # Pass manager
            )
        elif mode_name == "lexical":
            per_table_top_k = mode_retriever_config.get("per_table_top_k", 10)
            target_table_only = mode_retriever_config.get("target_table_only", False)
            target_table_boost = mode_retriever_config.get("target_table_boost", 0.0)
            field_match_boost = mode_retriever_config.get("field_match_boost", 1.0)
            id_exact_boost = mode_retriever_config.get("id_exact_boost", 6.0)
            phrase_match_boost = mode_retriever_config.get("phrase_match_boost", 2.5)
            date_year_boost = mode_retriever_config.get("date_year_boost", 6.0)
            date_month_boost = mode_retriever_config.get("date_month_boost", 2.0)
            date_exact_boost = mode_retriever_config.get("date_exact_boost", 10.0)
            enable_structured_recall = mode_retriever_config.get(
                "enable_structured_recall", True
            )
            structured_recall_boost = mode_retriever_config.get(
                "structured_recall_boost", 60.0
            )
            enable_fk_expansion = mode_retriever_config.get(
                "enable_fk_expansion", False
            )
            fk_expansion_boost = mode_retriever_config.get("fk_expansion_boost", 18.0)
            enable_entity_dictionary = mode_retriever_config.get(
                "enable_entity_dictionary", False
            )
            entity_dictionary_per_column_limit = mode_retriever_config.get(
                "entity_dictionary_per_column_limit", 3000
            )
            entity_match_boost = mode_retriever_config.get("entity_match_boost", 4.0)
            entity_dictionary_path = mode_retriever_config.get("entity_dictionary_path")
            persist_entity_dictionary = mode_retriever_config.get(
                "persist_entity_dictionary", True
            )
            rebuild_entity_dictionary = mode_retriever_config.get(
                "rebuild_entity_dictionary", False
            )
            return LexicalRetriever.from_paths(
                index_dir,
                schema_path,
                per_table_top_k=per_table_top_k,
                target_table_only=target_table_only,
                target_table_boost=target_table_boost,
                field_match_boost=field_match_boost,
                id_exact_boost=id_exact_boost,
                phrase_match_boost=phrase_match_boost,
                date_year_boost=date_year_boost,
                date_month_boost=date_month_boost,
                date_exact_boost=date_exact_boost,
                enable_structured_recall=enable_structured_recall,
                structured_recall_boost=structured_recall_boost,
                enable_fk_expansion=enable_fk_expansion,
                fk_expansion_boost=fk_expansion_boost,
                enable_entity_dictionary=enable_entity_dictionary,
                entity_dictionary_per_column_limit=entity_dictionary_per_column_limit,
                entity_match_boost=entity_match_boost,
                entity_dictionary_path=entity_dictionary_path,
                persist_entity_dictionary=persist_entity_dictionary,
                rebuild_entity_dictionary=rebuild_entity_dictionary,
            )
        elif mode_name == "graph":
            max_hops = mode_retriever_config.get("max_hops", 6)
            seed_max_nodes = mode_retriever_config.get("seed_max_nodes", 3000)
            return GraphRetriever.from_paths(
                index_dir,
                schema_path,
                max_hops=max_hops,
                seed_max_nodes=seed_max_nodes,
            )

        elif mode_name == "hybrid" or str(mode_name).startswith("hybrid."):
            vector_mode = mode_retriever_config.get("vector_mode", "semantic")
            sql_mode = mode_retriever_config.get("sql_mode", "text2sql.two_step")

            if str(vector_mode).startswith("text2sql") or str(vector_mode).startswith(
                "hybrid"
            ):
                raise ValueError(
                    f"Hybrid vector_mode must be an index-based mode (got: {vector_mode})"
                )

            vector_index_dir = index_dir.parent / vector_mode
            if not (
                vector_index_dir.exists()
                and (vector_index_dir / "schema_metadata.json").exists()
            ):
                raise FileNotFoundError(
                    f"Hybrid mode requires '{vector_mode}' index at {vector_index_dir}"
                )

            vector_retriever = self.load_retriever(
                mode_name=vector_mode,
                index_dir=index_dir.parent,
                run_id=run_id,
                per_table_top_k=per_table_top_k,
            )

            if not str(sql_mode).startswith("text2sql"):
                raise ValueError(
                    f"Hybrid sql_mode must be a text2sql mode (got: {sql_mode})"
                )

            connection_string = self._get_text2sql_connection(schema_metadata, run_id)
            sql_mode_info = self.registry.get(sql_mode)
            if not sql_mode_info:
                raise ValueError(f"Mode '{sql_mode}' not found")
            sql_retriever_cls = sql_mode_info.retriever_class

            llm_retriever = sql_retriever_cls(
                schema_metadata=schema_metadata,
                connection_string=connection_string,
                mode_name=sql_mode,
                vector_retriever=None,
                few_shot_manager=None,
            )

            # 3. Combine
            from talk2metadata.core.solution.hybrid import HybridRetriever

            return HybridRetriever(
                schema_metadata=schema_metadata,
                connection_string=connection_string,
                vector_retriever=vector_retriever,
                llm_retriever=llm_retriever,
                mode_name=mode_name,
            )

        raise NotImplementedError(f"Retriever for mode '{mode_name}' not implemented")

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode_name: Optional[str] = None,
        index_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
        per_table_top_k: int = 5,
    ) -> List[Union[SearchResult, Any]]:
        """Perform search using specified mode.

        Args:
            query: Search query
            top_k: Number of results to return
            mode_name: Optional mode name (uses active mode if None)
            index_dir: Optional index directory
            run_id: Optional run ID
            per_table_top_k: Top-k per table for semantic mode

        Returns:
            List of SearchResult objects
        """
        # Determine mode
        if not mode_name:
            from talk2metadata.core.solution.modes import get_active_mode

            mode_name = get_active_mode() or "semantic"

        # Load retriever
        retriever = self.load_retriever(
            mode_name=mode_name,
            index_dir=index_dir,
            run_id=run_id,
            per_table_top_k=per_table_top_k,
        )

        schema_metadata = getattr(retriever, "schema_metadata", None)
        query, _ = run_preprocess(
            query,
            config=self.config,
            mode_name=mode_name,
            run_id=run_id or self.config.get("run_id"),
            schema_metadata=(
                schema_metadata if isinstance(schema_metadata, SchemaMetadata) else None
            ),
        )

        # Perform search
        return retriever.search(query, top_k=top_k)

    def load_retrievers_for_comparison(
        self,
        modes_to_compare: Optional[List[str]] = None,
        index_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load retrievers for all modes to compare.

        Args:
            modes_to_compare: Optional list of mode names
            index_dir: Optional base index directory
            run_id: Optional run ID

        Returns:
            Dictionary mapping mode_name -> retriever
        """
        # Determine modes to compare
        if not modes_to_compare:
            comparison_config = self.config.get("modes.compare", {})
            modes_to_compare = comparison_config.get("modes", [])
            if not modes_to_compare:
                modes_to_compare = self.registry.get_all_enabled()

        if not modes_to_compare:
            raise ValueError("No enabled modes found for comparison")

        # Load retrievers
        base_index_dir = (
            index_dir
            if index_dir
            else get_indexes_dir(run_id or self.config.get("run_id"), self.config)
        )
        base_index_dir = Path(base_index_dir)

        retrievers = {}
        for mode_name in modes_to_compare:
            mode_info = self.registry.get(mode_name)
            if not mode_info or not mode_info.enabled:
                continue

            try:
                retriever = self.load_retriever(
                    mode_name=mode_name,
                    index_dir=base_index_dir,
                    run_id=run_id,
                )
                retrievers[mode_name] = retriever
            except (FileNotFoundError, NotImplementedError):
                # Skip modes that can't be loaded
                continue

        return retrievers

    def compare_modes(
        self,
        query: str,
        top_k: int = 5,
        modes_to_compare: Optional[List[str]] = None,
        index_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
    ):
        """Compare search results across multiple modes.

        Args:
            query: Search query
            top_k: Number of results to return per mode
            modes_to_compare: Optional list of mode names
            index_dir: Optional base index directory
            run_id: Optional run ID

        Returns:
            ComparisonResult object
        """
        # Load retrievers
        retrievers = self.load_retrievers_for_comparison(
            modes_to_compare=modes_to_compare,
            index_dir=index_dir,
            run_id=run_id,
        )

        if not retrievers:
            raise ValueError("No retrievers loaded for comparison")

        query_by_mode = {}
        for m, r in retrievers.items():
            schema_metadata = getattr(r, "schema_metadata", None)
            rewritten, _ = run_preprocess(
                query,
                config=self.config,
                mode_name=m,
                run_id=run_id or self.config.get("run_id"),
                schema_metadata=(
                    schema_metadata
                    if isinstance(schema_metadata, SchemaMetadata)
                    else None
                ),
            )
            query_by_mode[m] = rewritten

        # Run comparison
        comparator = ModeComparator(modes=list(retrievers.keys()))
        return comparator.compare(
            query, top_k=top_k, retrievers=retrievers, query_by_mode=query_by_mode
        )
