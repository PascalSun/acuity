"""Comparison mode for running multiple modes and comparing results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

from ..paths.semantic.search_result import SearchResult
from .registry import get_registry

logger = get_logger(__name__)


@dataclass
class ComparisonResult:
    """Result from comparing multiple modes."""

    query: str
    mode_results: Dict[str, List[SearchResult]]  # mode_name -> results
    common_results: List[SearchResult]  # Results appearing in all modes
    unique_results: Dict[str, List[SearchResult]]  # mode_name -> unique results
    overlap_stats: Dict[str, float]  # mode_name -> overlap percentage

    def __repr__(self) -> str:
        return (
            f"ComparisonResult(query='{self.query}', "
            f"modes={list(self.mode_results.keys())}, "
            f"common={len(self.common_results)}, "
            f"unique={sum(len(r) for r in self.unique_results.values())})"
        )


class ModeComparator:
    """Compare results from multiple retrieval modes."""

    def __init__(self, modes: Optional[List[str]] = None):
        """Initialize comparator.

        Args:
            modes: List of mode names to compare. If None, uses all enabled modes.
        """
        registry = get_registry()
        config = get_config()

        if modes is None:
            # Get modes from config or use all enabled
            comparison_config = config.get("modes.compare", {})
            modes = comparison_config.get("modes", [])
            if not modes:
                modes = registry.get_all_enabled()

        self.modes = modes
        self.registry = registry

        # Validate modes
        for mode_name in self.modes:
            mode_info = registry.get(mode_name)
            if not mode_info:
                raise ValueError(f"Mode '{mode_name}' not found")
            if not mode_info.enabled:
                raise ValueError(f"Mode '{mode_name}' is disabled")

        logger.info(f"Initialized ModeComparator with modes: {self.modes}")

    def compare(
        self,
        query: str,
        top_k: int = 5,
        retrievers: Optional[Dict[str, Any]] = None,
        query_by_mode: Optional[Dict[str, str]] = None,
    ) -> ComparisonResult:
        """Run query on all modes and compare results.

        Args:
            query: Search query
            top_k: Number of results per mode
            retrievers: Optional dict of pre-initialized retrievers (mode_name -> retriever)

        Returns:
            ComparisonResult with comparison statistics
        """
        mode_results: Dict[str, List[SearchResult]] = {}

        # Run search on each mode
        for mode_name in self.modes:
            logger.debug(f"Running search on mode: {mode_name}")
            try:
                if retrievers and mode_name in retrievers:
                    retriever = retrievers[mode_name]
                else:
                    # Would need to initialize retriever here
                    # For now, assume retrievers are provided
                    raise ValueError(f"Retriever for mode '{mode_name}' not provided")

                mode_query = (
                    query_by_mode.get(mode_name, query)
                    if isinstance(query_by_mode, dict)
                    else query
                )
                results = retriever.search(mode_query, top_k=top_k)
                mode_results[mode_name] = results

            except Exception as e:
                logger.error(f"Error running mode '{mode_name}': {e}")
                mode_results[mode_name] = []

        # Analyze results
        common_results = self._find_common_results(mode_results)
        unique_results = self._find_unique_results(mode_results, common_results)
        overlap_stats = self._calculate_overlap_stats(mode_results, common_results)

        return ComparisonResult(
            query=query,
            mode_results=mode_results,
            common_results=common_results,
            unique_results=unique_results,
            overlap_stats=overlap_stats,
        )

    def _find_common_results(
        self, mode_results: Dict[str, List[SearchResult]]
    ) -> List[SearchResult]:
        """Find results that appear in all modes.

        Args:
            mode_results: Dict mapping mode_name -> results

        Returns:
            List of common results
        """
        if not mode_results:
            return []

        # Get row IDs from first mode
        first_mode = list(mode_results.keys())[0]
        first_results = mode_results[first_mode]

        # Find results that appear in all modes
        common = []
        for result in first_results:
            row_id = result.row_id
            table = result.table

            # Check if this result appears in all other modes
            in_all = True
            for mode_name, results in mode_results.items():
                if mode_name == first_mode:
                    continue

                found = any(r.row_id == row_id and r.table == table for r in results)
                if not found:
                    in_all = False
                    break

            if in_all:
                common.append(result)

        return common

    def _find_unique_results(
        self,
        mode_results: Dict[str, List[SearchResult]],
        common_results: List[SearchResult],
    ) -> Dict[str, List[SearchResult]]:
        """Find results unique to each mode.

        Args:
            mode_results: Dict mapping mode_name -> results
            common_results: Common results across all modes

        Returns:
            Dict mapping mode_name -> unique results
        """
        common_ids = {(r.row_id, r.table) for r in common_results}
        unique: Dict[str, List[SearchResult]] = {}

        for mode_name, results in mode_results.items():
            unique[mode_name] = [
                r for r in results if (r.row_id, r.table) not in common_ids
            ]

        return unique

    def _calculate_overlap_stats(
        self,
        mode_results: Dict[str, List[SearchResult]],
        common_results: List[SearchResult],
    ) -> Dict[str, float]:
        """Calculate overlap statistics for each mode.

        Args:
            mode_results: Dict mapping mode_name -> results
            common_results: Common results

        Returns:
            Dict mapping mode_name -> overlap percentage
        """
        stats = {}
        total_common = len(common_results)

        for mode_name, results in mode_results.items():
            if len(results) == 0:
                stats[mode_name] = 0.0
            else:
                overlap = total_common / len(results) * 100
                stats[mode_name] = round(overlap, 2)

        return stats
