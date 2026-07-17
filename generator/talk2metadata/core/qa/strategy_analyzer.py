"""Strategy analyzer for QA generation.

Analyzes schema capabilities and determines which strategies can be generated.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from talk2metadata.core.qa.difficulty_classifier import DifficultyClassifier
from talk2metadata.core.schema import SchemaMetadata
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


class StrategyAnalyzer:
    """Analyzes schema to determine which QA strategies are feasible."""

    def __init__(self, schema: SchemaMetadata):
        """Initialize strategy analyzer.

        Args:
            schema: Schema metadata to analyze
        """
        self.schema = schema
        self.target_table = schema.target_table
        self.classifier = DifficultyClassifier()

    def analyze_schema_capabilities(self) -> Dict[str, any]:
        """Analyze schema and determine which strategies are feasible.

        Returns:
            Dictionary with analysis results:
            - supported_strategies: List of strategy codes that are feasible
            - unsupported_strategies: List of strategy codes that are not feasible
            - reasons: Dict mapping strategy -> reason why it's unsupported
            - schema_info: Dict with schema statistics
        """
        all_strategies = self.classifier.get_all_strategies()
        supported = []
        unsupported = {}
        reasons = {}

        # Get schema statistics
        target_table_meta = self.schema.tables[self.target_table]
        num_target_columns = len(target_table_meta.columns)
        num_target_fks = len(
            self.schema.get_foreign_keys_for_table(self.target_table, direction="both")
        )

        # Analyze each strategy
        for strategy in all_strategies:
            is_supported, reason = self._check_strategy_feasibility(strategy)
            if is_supported:
                supported.append(strategy)
            else:
                unsupported[strategy] = True
                reasons[strategy] = reason

        return {
            "supported_strategies": supported,
            "unsupported_strategies": list(unsupported.keys()),
            "reasons": reasons,
            "schema_info": {
                "target_table": self.target_table,
                "target_table_columns": num_target_columns,
                "target_table_fks": num_target_fks,
                "total_tables": len(self.schema.tables),
                "total_foreign_keys": len(self.schema.foreign_keys),
            },
        }

    def _check_strategy_feasibility(self, strategy: str) -> Tuple[bool, str]:
        """Check if a strategy is feasible given the schema.

        Args:
            strategy: Strategy code (e.g., "2iM")

        Returns:
            Tuple of (is_feasible, reason)
        """
        # Parse strategy
        pattern, difficulty = self._parse_strategy(strategy)

        # Check pattern feasibility
        pattern_feasible, pattern_reason = self._check_pattern_feasibility(pattern)
        if not pattern_feasible:
            return False, pattern_reason

        # Check difficulty feasibility (filter columns)
        difficulty_feasible, difficulty_reason = self._check_difficulty_feasibility(
            pattern, difficulty
        )
        if not difficulty_feasible:
            return False, difficulty_reason

        return True, ""

    def _parse_strategy(self, strategy: str) -> Tuple[str, str]:
        """Parse strategy into pattern and difficulty.

        Args:
            strategy: Difficulty code (e.g., "2iM")

        Returns:
            Tuple of (pattern, difficulty) (e.g., ("2i", "M"))
        """
        if strategy[0].isdigit():
            if len(strategy) >= 2 and strategy[1] in ["p", "i"]:
                return strategy[:2], strategy[2:]
            else:
                return strategy[0], strategy[1:]
        else:
            return strategy[:2], strategy[2:]

    def _check_pattern_feasibility(self, pattern: str) -> Tuple[bool, str]:
        """Check if a JOIN pattern is feasible.

        Args:
            pattern: Pattern code (e.g., "2i", "1p")

        Returns:
            Tuple of (is_feasible, reason)
        """
        if pattern == "0":
            # Direct query - no JOINs needed
            # Just need target table to have columns for filtering
            target_table_meta = self.schema.tables[self.target_table]
            if len(target_table_meta.columns) < 2:  # Need at least PK + 1 filter column
                return (
                    False,
                    f"Target table '{self.target_table}' has insufficient columns (< 2)",
                )
            return True, ""

        # Extract number and type
        if pattern[-1] == "p":
            # Path pattern (chain)
            hops = int(pattern[0])
            return self._check_chain_feasibility(hops)
        elif pattern[-1] == "i":
            # Intersection pattern (star)
            branches = int(pattern[0])
            return self._check_star_feasibility(branches)
        else:
            return False, f"Unknown pattern: {pattern}"

    def _check_chain_feasibility(self, hops: int) -> Tuple[bool, str]:
        """Check if a chain pattern with N hops is feasible.

        Args:
            hops: Number of hops (JOINs in chain)

        Returns:
            Tuple of (is_feasible, reason)
        """
        if hops == 0:
            return True, ""

        # Check if we can build a chain starting from target table
        # We need to be able to traverse hops number of foreign keys
        max_chain_length = self._get_max_chain_length(self.target_table, set())

        if max_chain_length < hops:
            return (
                False,
                f"Cannot build {hops}-hop chain (max chain length: {max_chain_length})",
            )

        return True, ""

    def _get_max_chain_length(self, current_table: str, visited: Set[str]) -> int:
        """Get maximum chain length (number of hops) starting from a table.

        Args:
            current_table: Starting table name
            visited: Set of visited tables (to avoid cycles)

        Returns:
            Maximum number of hops (JOINs) possible from this table
        """
        if current_table in visited:
            return 0

        visited.add(current_table)

        # Get foreign keys from current table
        fks = self.schema.get_foreign_keys_for_table(current_table, direction="both")

        if not fks:
            return 0

        max_hops = 0
        for fk in fks:
            # Determine next table
            if fk.child_table == current_table:
                next_table = fk.parent_table
            else:
                next_table = fk.child_table

            if next_table not in visited:
                # 1 hop to next_table, plus any hops from next_table
                hops_from_next = self._get_max_chain_length(next_table, visited.copy())
                total_hops = 1 + hops_from_next
                max_hops = max(max_hops, total_hops)

        return max_hops

    def _check_star_feasibility(self, branches: int) -> Tuple[bool, str]:
        """Check if a star pattern with N branches is feasible.

        Args:
            branches: Number of branches (JOINs from target table)

        Returns:
            Tuple of (is_feasible, reason)
        """
        # Get all foreign keys involving the target table
        fks = self.schema.get_foreign_keys_for_table(
            self.target_table, direction="both"
        )

        if len(fks) < branches:
            return (
                False,
                f"Target table '{self.target_table}' has only {len(fks)} foreign keys, "
                f"but {branches} branches are required",
            )

        return True, ""

    def _check_difficulty_feasibility(
        self, pattern: str, difficulty: str
    ) -> Tuple[bool, str]:
        """Check if difficulty level is feasible (enough filter columns).

        Args:
            pattern: Pattern code (e.g., "2i")
            difficulty: Difficulty level (E/M/H)

        Returns:
            Tuple of (is_feasible, reason)
        """
        # Determine required filter columns
        if difficulty == "E":
            min_cols = 1
        elif difficulty == "M":
            min_cols = 3
        elif difficulty == "H":
            min_cols = 6
        else:
            min_cols = 1

        # Get involved tables for this pattern
        involved_tables = self._get_involved_tables_for_pattern(pattern)

        # Count available filter columns (excluding primary keys)
        available_cols = 0
        for table in involved_tables:
            table_meta = self.schema.tables[table]
            # Exclude primary key
            available_cols += len(table_meta.columns) - 1

        if available_cols < min_cols:
            return (
                False,
                f"Pattern '{pattern}' with difficulty '{difficulty}' requires "
                f"{min_cols} filter columns, but only {available_cols} available "
                f"across involved tables: {', '.join(involved_tables)}",
            )

        return True, ""

    def _get_involved_tables_for_pattern(self, pattern: str) -> List[str]:
        """Get tables that would be involved in a pattern.

        Args:
            pattern: Pattern code (e.g., "2i")

        Returns:
            List of table names
        """
        tables = [self.target_table]

        if pattern == "0":
            return tables

        if pattern[-1] == "p":
            # Path pattern - we can't know exact tables without building,
            # but we know it will include target table + N other tables
            # For feasibility check, we'll be conservative
            hops = int(pattern[0])
            # Get some sample tables that could be in the chain
            fks = self.schema.get_foreign_keys_for_table(
                self.target_table, direction="both"
            )
            for i, fk in enumerate(fks[:hops]):
                if fk.child_table == self.target_table:
                    tables.append(fk.parent_table)
                else:
                    tables.append(fk.child_table)
            return tables[: hops + 1]

        elif pattern[-1] == "i":
            # Star pattern - target table + N related tables
            branches = int(pattern[0])
            fks = self.schema.get_foreign_keys_for_table(
                self.target_table, direction="both"
            )
            for fk in fks[:branches]:
                if fk.child_table == self.target_table:
                    tables.append(fk.parent_table)
                else:
                    tables.append(fk.child_table)
            return tables

        return tables

    def check_config_strategies(
        self,
        strategy_weights: Dict[str, int] = None,
        tier_weights: Dict[str, int] = None,
    ) -> Dict[str, any]:
        """Check which configured strategies are feasible.

        Args:
            strategy_weights: Optional strategy weights from config
            tier_weights: Optional tier weights from config

        Returns:
            Dictionary with:
            - configured_strategies: List of strategies from config
            - feasible_strategies: List of configured strategies that are feasible
            - infeasible_strategies: List of configured strategies that are not feasible
            - reasons: Dict mapping strategy -> reason why it's infeasible
        """
        # Get schema capabilities
        capabilities = self.analyze_schema_capabilities()
        supported = set(capabilities["supported_strategies"])

        # Determine configured strategies
        configured = set()
        if strategy_weights:
            configured = set(strategy_weights.keys())
        elif tier_weights:
            # Get all strategies in configured tiers
            for tier, weight in tier_weights.items():
                if weight > 0:
                    tier_strategies = self.classifier.get_strategies_by_tier(tier)
                    configured.update(tier_strategies)

        # Categorize configured strategies
        feasible = [s for s in configured if s in supported]
        infeasible = [s for s in configured if s not in supported]

        reasons = {
            s: capabilities["reasons"].get(s, "Unknown reason") for s in infeasible
        }

        return {
            "configured_strategies": sorted(configured),
            "feasible_strategies": sorted(feasible),
            "infeasible_strategies": sorted(infeasible),
            "reasons": reasons,
        }
