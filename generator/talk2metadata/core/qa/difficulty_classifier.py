"""Difficulty classifier based on query patterns and filter complexity.

This module implements the difficulty classification system described in
docs/qa/difficulty-classification.md.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Set


class PatternType(Enum):
    """Pattern types based on JOIN structure."""

    DIRECT = "0"  # No JOIN, query only target table
    PATH_1 = "1p"  # Single-hop path (1 JOIN)
    PATH_2 = "2p"  # Two-hop path (2 JOINs in chain)
    PATH_3 = "3p"  # Three-hop path (3 JOINs in chain)
    INTERSECTION_2 = "2i"  # Two-way intersection (2 JOINs in star)
    INTERSECTION_3 = "3i"  # Three-way intersection (3 JOINs in star)
    INTERSECTION_4 = "4i"  # Four-way intersection (4 JOINs in star)
    # NOTE: Xm is RESERVED, not implemented. The query builder never constructs
    # mixed chain+star structures, get_all_strategies() excludes Xm, and
    # _identify_pattern cannot return it for any producible QueryPlan. Do not
    # claim Xm coverage in papers/docs until mixed generation exists.
    MIXED = "Xm"  # Complex combination of chain and star (reserved, unimplemented)


class DifficultyLevel(Enum):
    """Difficulty levels based on filter complexity."""

    EASY = "E"  # 1-2 columns
    MEDIUM = "M"  # 3-5 columns
    HARD = "H"  # 6+ columns


@dataclass
class JoinPath:
    """Represents a JOIN path in the query."""

    tables: List[str]  # List of tables in the path (including target table)
    join_type: str  # 'chain' or 'star'


@dataclass
class QueryPlan:
    """Represents a parsed query plan for difficulty classification."""

    target_table: str
    join_paths: List[JoinPath]
    filter_columns: Set[str]  # All columns used in WHERE clause (table.column format)

    @property
    def num_joins(self) -> int:
        """Return the total number of JOINs."""
        return len(self.join_paths)

    @property
    def num_filter_columns(self) -> int:
        """Return the number of distinct columns used in filters."""
        return len(self.filter_columns)


class DifficultyClassifier:
    """Classifier for query difficulty based on pattern and filter complexity."""

    # Pattern base scores
    PATTERN_SCORES = {
        "0": 0.0,
        "1p": 1.0,
        "2p": 2.0,
        "2i": 2.0,
        "3p": 3.0,
        "3i": 3.0,
        "4i": 4.0,
        "Xm": 5.0,
    }

    # Difficulty modifiers
    DIFFICULTY_MODIFIERS = {
        "E": 0.0,
        "M": 0.3,
        "H": 0.6,
    }

    # Tier boundaries
    TIER_BOUNDARIES = {
        "easy": (0.0, 0.9),  # 0E, 0M, 0H
        "medium": (1.0, 1.9),  # 1pE, 1pM, 1pH
        "hard": (2.0, 2.9),  # 2pE, 2pM, 2pH, 2iE, 2iM, 2iH
        "expert": (3.0, 10.0),  # 3p*, 3i*, 4i*, Xm*
    }

    def classify(self, query_plan: QueryPlan) -> str:
        """Classify a query plan into difficulty code.

        Args:
            query_plan: QueryPlan object representing the query structure

        Returns:
            Difficulty code (e.g., "2iM")
        """
        pattern = self._identify_pattern(query_plan)
        difficulty = self._assess_difficulty(query_plan)

        return f"{pattern.value}{difficulty.value}"

    def _identify_pattern(self, query_plan: QueryPlan) -> PatternType:
        """Identify the pattern type based on JOIN structure."""
        num_joins = query_plan.num_joins

        if num_joins == 0:
            return PatternType.DIRECT

        # Check if all JOINs are direct to target (star pattern)
        is_star = all(
            path.join_type == "star" and len(path.tables) == 2
            for path in query_plan.join_paths
        )

        if is_star:
            # Star/Intersection pattern
            if num_joins == 1:
                return PatternType.PATH_1  # Actually still a path
            elif num_joins == 2:
                return PatternType.INTERSECTION_2
            elif num_joins == 3:
                return PatternType.INTERSECTION_3
            elif num_joins >= 4:
                return PatternType.INTERSECTION_4
        else:
            # Path/Chain pattern
            max_depth = max(len(path.tables) for path in query_plan.join_paths)
            if max_depth == 2:
                return PatternType.PATH_1
            elif max_depth == 3:
                return PatternType.PATH_2
            elif max_depth >= 4:
                return PatternType.PATH_3

        return PatternType.MIXED

    def _assess_difficulty(self, query_plan: QueryPlan) -> DifficultyLevel:
        """Assess difficulty based on filter complexity."""
        num_filter_columns = query_plan.num_filter_columns

        if num_filter_columns <= 2:
            return DifficultyLevel.EASY
        elif num_filter_columns <= 5:
            return DifficultyLevel.MEDIUM
        else:
            return DifficultyLevel.HARD

    def get_score(self, difficulty_code: str) -> float:
        """Convert difficulty code to numeric score.

        Args:
            difficulty_code: Difficulty code (e.g., "2iM")

        Returns:
            Numeric score (e.g., 2.3)
        """
        # Parse pattern and difficulty
        if difficulty_code[0].isdigit():
            if len(difficulty_code) >= 2 and difficulty_code[1] in ["p", "i"]:
                pattern_str = difficulty_code[:2]
                diff_str = difficulty_code[2:]
            else:
                pattern_str = difficulty_code[0]
                diff_str = difficulty_code[1:]
        else:
            pattern_str = difficulty_code[:2]
            diff_str = difficulty_code[2:]

        base = self.PATTERN_SCORES.get(pattern_str, 0.0)
        modifier = self.DIFFICULTY_MODIFIERS.get(diff_str, 0.0)

        return base + modifier

    def get_tier(self, difficulty_code: str) -> str:
        """Get the tier (easy/medium/hard/expert) for a difficulty code.

        Args:
            difficulty_code: Difficulty code (e.g., "2iM")

        Returns:
            Tier name: "easy", "medium", "hard", or "expert"
        """
        score = self.get_score(difficulty_code)

        for tier, (min_score, max_score) in self.TIER_BOUNDARIES.items():
            if min_score <= score <= max_score:
                return tier

        return "expert"  # Default for very high scores

    @classmethod
    def get_all_strategies(cls) -> List[str]:
        """Get all possible difficulty strategies.

        Returns:
            List of all difficulty codes
        """
        patterns = ["0", "1p", "2p", "2i", "3p", "3i", "4i"]
        difficulties = ["E", "M", "H"]

        strategies = []
        for pattern in patterns:
            for difficulty in difficulties:
                strategies.append(f"{pattern}{difficulty}")

        return strategies

    @classmethod
    def get_strategies_by_tier(cls, tier: str) -> List[str]:
        """Get all strategies in a specific tier.

        Args:
            tier: Tier name ("easy", "medium", "hard", "expert")

        Returns:
            List of difficulty codes in that tier
        """
        all_strategies = cls.get_all_strategies()
        classifier = cls()

        return [s for s in all_strategies if classifier.get_tier(s) == tier]
