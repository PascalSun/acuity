"""Agent-based foreign key detector using AI."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import pandas as pd

from talk2metadata.core.schema.fk_detector_base import FKDetectorBase
from talk2metadata.core.schema.types import ForeignKey, TableMetadata
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


class AgentBasedFKDetector(FKDetectorBase):
    """Agent-based foreign key detector using AI analysis."""

    def __init__(self, config: Dict):
        """Initialize agent-based FK detector.

        Args:
            config: Configuration dict
        """
        super().__init__(config)
        self.min_overlap = config.get("min_overlap_ratio", 0.8)

    def detect(
        self,
        tables: Dict[str, pd.DataFrame],
        table_metadata: Dict[str, TableMetadata],
        target_table: str,
        rule_based_fks: Optional[List[ForeignKey]] = None,
    ) -> List[ForeignKey]:
        """Detect foreign keys using AI agent analysis.

        This method analyzes all candidate column pairs and returns the final
        foreign key relationships. Rule-based results are provided as reference
        but agent results override them.

        Args:
            tables: Dict of DataFrames
            table_metadata: Dict of TableMetadata
            target_table: Target/center table name (for star schema prioritization)
            rule_based_fks: FKs found by rule-based detection (for reference)

        Returns:
            List of ForeignKey objects (agent's final determination)
        """
        use_agent = bool(self.config.get("use_agent", True))
        if not use_agent:
            return []

        trigger = str(self.config.get("agent_trigger", "auto")).lower()
        threshold = int(self.config.get("agent_threshold", 2))
        rule_count = len(rule_based_fks or [])

        if trigger == "never":
            return []
        if trigger == "auto" and rule_count >= threshold:
            return []

        try:
            from talk2metadata.utils.config import get_config

            if not bool(get_config().get("agent.enabled", False)):
                return []
        except Exception:
            return []

        logger.info("Running agent-based FK detection...")

        try:
            from talk2metadata.agent import AgentWrapper
        except ImportError:
            logger.warning(
                "Agent module not available, skipping agent-based FK detection"
            )
            return []

        # Find all candidate column pairs (including rule-based ones)
        candidates = self._find_column_candidates(tables, table_metadata, target_table)

        if not candidates:
            logger.info("No candidate column pairs found for agent analysis")
            return []

        logger.info(
            f"Analyzing {len(candidates)} candidate FKs with AI agent "
            f"(rule-based found {len(rule_based_fks or [])} FKs as reference)..."
        )

        # Prepare prompt for agent (include rule-based results as reference)
        prompt = self._build_agent_prompt(
            candidates, table_metadata, target_table, rule_based_fks
        )

        # Initialize agent wrapper
        try:
            agent = AgentWrapper()
        except Exception as e:
            logger.warning(f"Failed to initialize agent: {e}")
            logger.warning("Skipping agent-based FK detection")
            return []

        # Call agent
        try:
            response = agent.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=4096,
            )

            # Parse response
            agent_fks = self._parse_agent_response(response.content, candidates)

            logger.info(f"Agent-based detection found {len(agent_fks)} FKs")
            return agent_fks

        except Exception as e:
            logger.error(f"Agent FK detection failed: {e}")
            return []

    def _find_column_candidates(
        self,
        tables: Dict[str, pd.DataFrame],
        table_metadata: Dict[str, TableMetadata],
        target_table: Optional[str] = None,
    ) -> List[Dict]:
        """Find candidate column pairs with high value overlap.

        Args:
            tables: Dict of DataFrames
            table_metadata: Dict of TableMetadata
            target_table: Target/center table name (prioritized in star schema)

        Returns:
            List of candidate dicts with overlap info, sorted by target table priority
        """
        candidates = []

        # Find all column pairs with potential relationships
        # Include all tables as potential child tables (including target table)
        for child_name, child_df in tables.items():

            for child_col in child_df.columns:
                # Note: A column CAN be both a PK and FK (foreign primary key pattern)
                # We only skip self-referential relationships (same table)

                # Check overlap with all other tables' primary keys
                for parent_name, parent_df in tables.items():
                    if parent_name == child_name:
                        continue

                    parent_pk = table_metadata[parent_name].primary_key
                    if parent_pk is None or parent_pk not in parent_df.columns:
                        continue

                    # Calculate overlap
                    coverage = self._check_inclusion(
                        child_df[child_col], parent_df[parent_pk]
                    )

                    if coverage >= self.min_overlap:
                        # Get sample values for agent analysis
                        child_sample = child_df[child_col].dropna().head(5).tolist()
                        parent_sample = parent_df[parent_pk].dropna().head(5).tolist()

                        # Mark if this is a target table relationship
                        is_target_relationship = parent_name == target_table

                        candidates.append(
                            {
                                "child_table": child_name,
                                "child_column": child_col,
                                "parent_table": parent_name,
                                "parent_column": parent_pk,
                                "coverage": coverage,
                                "child_sample": child_sample,
                                "parent_sample": parent_sample,
                                "child_unique": child_df[child_col].nunique(),
                                "parent_unique": parent_df[parent_pk].nunique(),
                                "is_target_relationship": is_target_relationship,
                            }
                        )

        # Sort candidates: target table relationships first, then by coverage
        candidates.sort(key=lambda x: (not x["is_target_relationship"], -x["coverage"]))

        return candidates

    def _build_agent_prompt(
        self,
        candidates: List[Dict],
        table_metadata: Dict[str, TableMetadata],
        target_table: Optional[str] = None,
        rule_based_fks: Optional[List[ForeignKey]] = None,
    ) -> str:
        """Build prompt for agent FK detection.

        Args:
            candidates: List of candidate column pairs
            table_metadata: Dict of TableMetadata
            target_table: Target/center table name (for star schema context)
            rule_based_fks: Rule-based FK results for reference

        Returns:
            Prompt string
        """
        # Build table schema summary
        schema_summary = "## Database Schema\n\n"
        if target_table:
            schema_summary += f"**Target/Center Table: `{target_table}`** (all tables should relate to this)\n\n"

        for table_name, meta in table_metadata.items():
            marker = " (TARGET)" if table_name == target_table else ""
            schema_summary += f"**{table_name}{marker}**\n"
            schema_summary += f"- Primary Key: {meta.primary_key}\n"
            schema_summary += f"- Columns: {', '.join(meta.columns.keys())}\n"
            schema_summary += f"- Row Count: {meta.row_count}\n\n"

        # Build candidates summary
        candidates_summary = "## Candidate Foreign Key Relationships\n\n"
        for i, c in enumerate(candidates, 1):
            # Mark if this is a target table relationship
            target_marker = (
                " ⭐ TARGET TABLE" if c.get("is_target_relationship") else ""
            )
            candidates_summary += f"### Candidate {i}{target_marker}\n"
            candidates_summary += f"- Child: `{c['child_table']}.{c['child_column']}`\n"
            candidates_summary += (
                f"- Parent: `{c['parent_table']}.{c['parent_column']}`\n"
            )
            candidates_summary += f"- Coverage: {c['coverage']:.2%}\n"
            candidates_summary += f"- Child unique values: {c['child_unique']}\n"
            candidates_summary += f"- Parent unique values: {c['parent_unique']}\n"
            candidates_summary += f"- Child sample values: {c['child_sample']}\n"
            candidates_summary += f"- Parent sample values: {c['parent_sample']}\n\n"

        # Build rule-based reference section
        rule_based_reference = ""
        if rule_based_fks:
            rule_based_reference = "## Rule-Based Detection Results (Reference)\n\n"
            rule_based_reference += (
                "The following foreign keys were detected by rule-based heuristics:\n\n"
            )
            for fk in rule_based_fks:
                rule_based_reference += (
                    f"- `{fk.child_table}.{fk.child_column}` -> "
                    f"`{fk.parent_table}.{fk.parent_column}` "
                    f"(coverage: {fk.coverage:.2%})\n"
                )
            rule_based_reference += (
                "\n**Note**: Review these results and validate or correct them. "
                "Your analysis should be the final determination.\n\n"
            )

        # Build architecture context
        architecture_context = ""
        if target_table:
            architecture_context = f"""
## Architecture Context

This database follows a **star schema** pattern with `{target_table}` as the central table:
- All dimension tables should have foreign keys pointing to `{target_table}`
- Prefer relationships to the target table over intermediate tables
- If a column could reference multiple tables with the same primary key, choose the target table
"""

        prompt = f"""You are a database schema expert analyzing potential foreign key relationships.

{rule_based_reference}

{schema_summary}

{candidates_summary}
{architecture_context}

## Task

Analyze each candidate relationship and determine if it represents a valid foreign key.
Consider:
1. **Star schema architecture**: Prioritize relationships to the target table (`{target_table if target_table else 'N/A'}`)
2. Column name semantics (e.g., similar names across tables)
3. Data type compatibility
4. Value overlap coverage (higher is better)
5. Cardinality patterns (many-to-one relationships)
6. Domain knowledge (e.g., "ANumber" likely means assignment/accession number)

**IMPORTANT**: When multiple candidates point to tables with the same primary key values,
prefer the relationship to the target table (marked with ⭐).

## Output Format

For each candidate that IS a valid foreign key, output ONLY the candidate number (e.g., "1", "2", "3").
Put each number on a separate line.
If a candidate is NOT a valid FK, do not include its number.

Example output:
```
1
3
5
```

Begin your analysis:"""

        return prompt

    def _parse_agent_response(
        self, response_text: str, candidates: List[Dict]
    ) -> List[ForeignKey]:
        """Parse agent response and create ForeignKey objects.

        Args:
            response_text: Agent response text
            candidates: Original candidates list

        Returns:
            List of ForeignKey objects
        """
        # Extract candidate numbers from response
        # Look for lines with just numbers
        lines = response_text.strip().split("\n")
        selected_indices = []

        for line in lines:
            line = line.strip()
            # Match lines that are just numbers (possibly in code blocks)
            if re.match(r"^\d+$", line):
                selected_indices.append(int(line))

        # Create ForeignKey objects
        fks = []
        for idx in selected_indices:
            if 1 <= idx <= len(candidates):
                c = candidates[idx - 1]
                fks.append(
                    ForeignKey(
                        child_table=c["child_table"],
                        child_column=c["child_column"],
                        parent_table=c["parent_table"],
                        parent_column=c["parent_column"],
                        coverage=c["coverage"],
                    )
                )
            else:
                logger.warning(f"Agent returned invalid candidate index: {idx}")

        return fks
