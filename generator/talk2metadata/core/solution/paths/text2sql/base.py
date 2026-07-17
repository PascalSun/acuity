"""Base classes for Text2SQL retrievers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from talk2metadata.agent.factory import LLMProviderFactory
from talk2metadata.core.schema.schema import SchemaMetadata
from talk2metadata.core.solution.modes import get_mode_retriever_config
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

from ...modes.registry import BaseRetriever

logger = get_logger(__name__)


@dataclass
class Text2SQLSearchResult:
    """Search result from text2sql mode.

    Attributes:
        rank: Rank of the result
        table: Table name (or "multiple" if query spans multiple tables)
        data: Result data (list of dicts representing rows)
        sql_query: The SQL query that was executed
        row_count: Number of rows returned
        score: Confidence score (always 1.0 for SQL results)
    """

    rank: int
    table: str
    data: List[Dict[str, Any]]
    sql_query: str
    row_count: int
    score: float = 1.0

    def __repr__(self) -> str:
        return (
            f"Text2SQLSearchResult(rank={self.rank}, table={self.table}, "
            f"rows={self.row_count}, sql={self.sql_query[:50]}...)"
        )


class BaseText2SQLRetriever(BaseRetriever):
    """Base class for text2sql retrievers."""

    def __init__(
        self,
        schema_metadata: SchemaMetadata,
        connection_string: Optional[str] = None,
        engine: Optional[Engine] = None,
        **kwargs: Any,
    ):
        """Initialize text2sql retriever.

        Args:
            schema_metadata: Schema metadata with table structures
            connection_string: Database connection string (SQLAlchemy format)
            engine: Optional pre-created SQLAlchemy engine
            **kwargs: Additional configuration
        """
        self.schema_metadata = schema_metadata
        self.target_table = schema_metadata.target_table

        # Get database connection
        config = get_config()
        if engine:
            self.engine = engine
            self._own_engine = False
        elif connection_string:
            self.engine = create_engine(connection_string)
            self._own_engine = True
        else:
            # Try to get from config
            ingest_config = config.get("ingest", {})
            source_path = ingest_config.get("source_path")
            data_type = ingest_config.get("data_type", "csv")

            if data_type in ("database", "db") and source_path:
                logger.info(f"Using connection string from config: {source_path}")
                self.engine = create_engine(source_path)
                self._own_engine = True
            else:
                raise ValueError(
                    "Either connection_string, engine, or database config must be provided"
                )

        # Initialize LLM provider
        # Use global agent config, merged with mode-specific agent override (for aliases)
        import os

        agent_config = dict(config.get("agent", {}))
        mode_name = kwargs.get("mode_name", "text2sql")
        self.mode_name = mode_name
        modes_cfg = config.get("modes", {})
        mode_block = modes_cfg.get(mode_name, {}) if isinstance(modes_cfg, dict) else {}
        mode_agent = mode_block.get("agent", {}) if isinstance(mode_block, dict) else {}
        if mode_agent:
            # Mode-specific agent overrides global (for text2sql.openai52, text2sql.gemini, etc.)
            agent_config = {**agent_config, **mode_agent}

        provider = agent_config.get("provider", "openai")
        model = agent_config.get("model") or agent_config.get(provider, {}).get("model")

        # Build provider kwargs (merge config, provider-specific config, and keys)
        agent_kwargs = agent_config.get("config", {}).copy()

        # Merge provider-specific config
        provider_config = agent_config.get(provider, {})
        for key, value in provider_config.items():
            if key != "model":  # model handled separately
                agent_kwargs[key] = value

        # Merge API keys from keys section
        keys_config = agent_config.get("keys", {})
        # Check for provider-specific key (e.g., gemini_api_key)
        api_key_name = f"{provider}_api_key"
        # Special case: gemini provider also accepts google_api_key
        if provider == "gemini" and "google_api_key" in keys_config:
            agent_kwargs["api_key"] = keys_config["google_api_key"]
        elif api_key_name in keys_config:
            agent_kwargs["api_key"] = keys_config[api_key_name]

        # Expand environment variables in string values
        for key, value in agent_kwargs.items():
            if (
                isinstance(value, str)
                and value.startswith("${")
                and value.endswith("}")
            ):
                env_var = value[2:-1]
                env_value = os.getenv(env_var)
                if env_value:
                    agent_kwargs[key] = env_value

        logger.info(
            f"Initializing LLM provider: {provider}, model: {model} (mode={mode_name})"
        )
        self.llm = LLMProviderFactory.create_provider(
            provider=provider, model=model, **agent_kwargs
        )

        # Load context from config (if available)
        # get_mode_retriever_config merges base retriever for aliases
        mode_retriever_config = get_mode_retriever_config(mode_name)
        self.mode_retriever_config = mode_retriever_config
        self.context = (mode_retriever_config.get("context") or "").strip()

        if self.context:
            logger.info(f"Loaded context for text2sql ({len(self.context)} chars)")

    def _format_schema_for_prompt(self) -> str:
        """Format schema metadata for LLM prompt.

        Returns:
            Formatted schema string
        """
        parts = ["# Database Schema\n"]
        parts.append(
            "IMPORTANT: Use EXACT table and column names as shown below. Case sensitivity matters!\n"
        )

        # Add table information with emphasis on exact names
        for table_name, table_meta in self.schema_metadata.tables.items():
            parts.append(f"## Table: {table_name}")
            if table_name == self.target_table:
                parts.append(
                    "  ⭐ THIS IS THE TARGET TABLE (use this as the main table)"
                )
            # Add table description if available
            if table_meta.description:
                parts.append(f"  Description: {table_meta.description}")
            if table_meta.row_count > 0:
                parts.append(f"  Row count: {table_meta.row_count:,}")
            if table_meta.primary_key:
                parts.append(f"  Primary Key: {table_meta.primary_key}")

            # Format columns with data types and descriptions - emphasize exact names
            if isinstance(table_meta.columns, dict):
                # columns is a dict: {column_name: dtype}
                column_list = []
                for col_name, dtype in table_meta.columns.items():
                    col_info = f"{col_name} ({dtype})"
                    # Add column description if available
                    if col_name in table_meta.column_descriptions:
                        col_info += f" - {table_meta.column_descriptions[col_name]}"
                    column_list.append(col_info)
                parts.append("  Columns:")
                for col_info in column_list:
                    parts.append(f"    - {col_info}")
            else:
                # Fallback: columns is a list
                parts.append(f"  Columns: {', '.join(table_meta.columns)}")

            # Include more sample values for better understanding
            if table_meta.sample_values:
                parts.append(
                    "  Sample values (use EXACT format including spaces and case):"
                )
                # Show more columns (up to 5) with more samples
                for col, vals in list(table_meta.sample_values.items())[:5]:
                    # Truncate long values but show more samples
                    sample_strs = []
                    for v in vals[:5]:
                        val_str = str(v)
                        if len(val_str) > 100:
                            val_str = val_str[:97] + "..."
                        sample_strs.append(f"'{val_str}'")
                    sample = ", ".join(sample_strs)
                    parts.append(f"    {col}: {sample}")
            parts.append("")

        # Add foreign key relationships with more context
        if self.schema_metadata.foreign_keys:
            parts.append("## Foreign Key Relationships\n")
            parts.append("Use these EXACT relationships to JOIN tables:\n")
            for fk in self.schema_metadata.foreign_keys:
                parts.append(
                    f"  {fk.child_table}.{fk.child_column} = "
                    f"{fk.parent_table}.{fk.parent_column}"
                )
            parts.append("")

        # Add user-provided context if available
        if hasattr(self, "context") and self.context:
            parts.append("# Additional Context\n")
            parts.append(self.context)
            parts.append("")

        return "\n".join(parts)

    @staticmethod
    def get_target_id_column_static(schema_metadata: SchemaMetadata) -> Optional[str]:
        """Get the ID column name for the target table (standalone for export)."""
        target_table = schema_metadata.target_table
        if target_table not in schema_metadata.tables:
            return None
        table_meta = schema_metadata.tables[target_table]
        if table_meta.primary_key:
            return table_meta.primary_key
        common_id_names = ["id", "Id", "ID", "row_id", "RowId", "ROW_ID"]
        for col_name in common_id_names:
            if col_name in table_meta.columns:
                return col_name
        return None

    @staticmethod
    def format_schema_for_prompt_compact_static(
        schema_metadata: SchemaMetadata, context: str = ""
    ) -> str:
        """Compact schema formatter: no sample values / row counts / descriptions.

        This is designed to reduce prompt (and finetune) tokens while still giving
        the model the *complete* list of tables/columns and FK relationships.
        """
        target_table = schema_metadata.target_table
        parts = ["# Database Schema\n"]
        parts.append("Use EXACT table and column names as shown below.\n")
        for table_name, table_meta in schema_metadata.tables.items():
            star = " ⭐ TARGET TABLE" if table_name == target_table else ""
            parts.append(f"## Table: {table_name}{star}")
            # Keep columns only (no types/descriptions)
            if isinstance(table_meta.columns, dict):
                cols = list(table_meta.columns.keys())
            else:
                cols = list(table_meta.columns)
            parts.append("  Columns: " + ", ".join(cols))
            parts.append("")

        if schema_metadata.foreign_keys:
            parts.append("## Foreign Key Relationships\n")
            for fk in schema_metadata.foreign_keys:
                parts.append(
                    f"  {fk.child_table}.{fk.child_column} = {fk.parent_table}.{fk.parent_column}"
                )
            parts.append("")

        if context:
            parts.append("# Additional Context\n")
            parts.append(context)
            parts.append("")

        return "\n".join(parts)

    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL query from LLM response.

        Args:
            response: LLM response text

        Returns:
            Extracted SQL query
        """
        # Try to extract SQL from code blocks
        sql_match = re.search(r"```(?:sql)?\s*\n(.*?)\n```", response, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()

        # Try to find SELECT statement (more flexible pattern)
        # Match SELECT ... until semicolon or end of string
        select_match = re.search(
            r"(SELECT.*?)(?:;|$)", response, re.DOTALL | re.IGNORECASE
        )
        if select_match:
            return select_match.group(1).strip()

        # If no SQL found, return the response as-is (might be plain SQL)
        return response.strip()

    def _get_target_id_column(self) -> Optional[str]:
        """Get the ID column name for the target table.

        Returns:
            Primary key column name, or None if not found
        """
        if self.target_table not in self.schema_metadata.tables:
            return None

        table_meta = self.schema_metadata.tables[self.target_table]
        # Try primary key first
        if table_meta.primary_key:
            return table_meta.primary_key

        # Fallback: look for common ID column names
        common_id_names = ["id", "Id", "ID", "row_id", "RowId", "ROW_ID"]
        for col_name in common_id_names:
            if col_name in table_meta.columns:
                return col_name

        return None

    def _ensure_id_column_in_select(self, sql_query: str) -> str:
        """Ensure target table's ID column is included in SELECT clause if target table is involved.

        Args:
            sql_query: SQL query string

        Returns:
            Modified SQL query with ID column if needed
        """
        id_column = self._get_target_id_column()
        if not id_column:
            # No ID column found, return as-is
            return sql_query

        sql_upper = sql_query.upper()

        # Check if target table is in the query
        target_table_pattern = rf"\b{re.escape(self.target_table.upper())}\b"
        if not re.search(target_table_pattern, sql_upper):
            return sql_query

        # Check if ID column is already in SELECT
        # Match SELECT ... FROM pattern (more flexible)
        select_match = re.search(
            r"(SELECT\s+)(.*?)(\s+FROM)", sql_query, re.DOTALL | re.IGNORECASE
        )
        if not select_match:
            return sql_query

        select_clause = select_match.group(2).strip()
        # Check if ID column (or table.ID_column) is already selected
        id_pattern = (
            r"\b(?:"
            + re.escape(id_column)
            + r"|"
            + re.escape(self.target_table)
            + r"\."
            + re.escape(id_column)
            + r")\b"
        )
        if re.search(id_pattern, select_clause, re.IGNORECASE):
            return sql_query

        # Add ID column to SELECT clause
        # Handle different cases: SELECT * vs SELECT col1, col2
        if "*" in select_clause:
            # Replace * with specific columns including ID column
            # Handle both SELECT * and SELECT table.*
            if f"{self.target_table}.*" in select_clause:
                new_select = select_clause.replace(
                    f"{self.target_table}.*",
                    f"{self.target_table}.{id_column}, {self.target_table}.*",
                )
            elif "*" in select_clause:
                new_select = select_clause.replace(
                    "*", f"{self.target_table}.{id_column}, *"
                )
            else:
                new_select = f"{self.target_table}.{id_column}, {select_clause}"
        else:
            # Add ID column at the beginning
            new_select = f"{self.target_table}.{id_column}, {select_clause}"

        # Reconstruct the query
        return (
            sql_query[: select_match.start(2)]
            + new_select
            + sql_query[select_match.end(2) :]
        )

    def _preemptive_sql_check(self, sql_query: str) -> str:
        """Check and fix SQL completeness BEFORE validation.

        This catches truncation issues early (e.g., missing closing parentheses,
        incomplete WHERE clauses) that often occur with LLM generation.

        Args:
            sql_query: SQL query to check

        Returns:
            Fixed SQL query
        """
        fixed = sql_query.strip()

        # 1. Balance parentheses
        open_count = fixed.count("(")
        close_count = fixed.count(")")
        if open_count > close_count:
            missing = open_count - close_count
            logger.info(f"🔧 Preemptive fix: Adding {missing} missing ')'")
            fixed += ")" * missing

        # 2. Check if SQL appears truncated (ends with incomplete clause)
        if re.search(r"(WHERE|AND|OR|LIKE|JOIN|ON)\s*$", fixed, re.IGNORECASE):
            logger.warning(
                "⚠️ SQL appears truncated (ends with WHERE/AND/OR/LIKE/JOIN/ON)"
            )
            # We can't auto-fix this - will let validation/retry handle it

        # 3. Ensure proper statement termination
        if not fixed.endswith(";"):
            # Only add semicolon if it looks complete
            if not re.search(r"(WHERE|AND|OR|LIKE|JOIN|ON)\s*$", fixed, re.IGNORECASE):
                fixed += ";"

        # 4. Check for basic SQL structure
        sql_lower = fixed.lower()
        if "select" not in sql_lower:
            logger.warning("⚠️ SQL missing SELECT keyword")
        if "from" not in sql_lower:
            logger.warning("⚠️ SQL missing FROM keyword")

        return fixed

    def _validate_and_fix_sql(
        self, sql_query: str, query: str, max_retries: int = 2
    ) -> str:
        """Validate SQL query and attempt to fix common issues.

        Args:
            sql_query: SQL query to validate
            query: Original natural language query
            max_retries: Maximum number of retry attempts

        Returns:
            Fixed SQL query
        """
        # NEW: Run preemptive check first to catch truncation issues
        sql_query = self._preemptive_sql_check(sql_query)

        # Try to execute and catch errors
        for attempt in range(max_retries + 1):
            try:
                # Test execute (with LIMIT 0 to avoid fetching data)
                # Normalize SQL to lowercase before validation
                # (since we preprocess all table/column names to lowercase)
                sql_query_normalized = self._normalize_sql_to_lowercase(sql_query)

                test_query = sql_query_normalized.rstrip(";")
                if "limit" not in test_query:
                    test_query += " limit 0"
                else:
                    # Replace existing LIMIT with 0
                    test_query = re.sub(
                        r"limit\s+\d+", "limit 0", test_query, flags=re.IGNORECASE
                    )

                with self.engine.connect() as conn:
                    conn.execute(text(test_query))

                # If successful, return the query
                return sql_query

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f"SQL validation failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                    )
                    # Try to fix common issues
                    sql_query = self._attempt_sql_fix(sql_query, str(e), query)
                else:
                    logger.error(
                        f"SQL validation failed after {max_retries + 1} attempts: {e}"
                    )
                    # Return original query, let execution handle the error
                    return sql_query

        return sql_query

    def _validate_table_column_names(self, sql_query: str) -> tuple[bool, list[str]]:
        """Validate that all table and column names in SQL exist in schema.

        Args:
            sql_query: SQL query to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        sql_upper = sql_query.upper()

        # Extract table names from SQL (FROM, JOIN clauses)
        table_pattern = r"(?:FROM|JOIN)\s+(\w+)"
        mentioned_tables = set(re.findall(table_pattern, sql_upper))

        # Check if mentioned tables exist in schema
        schema_tables_upper = {t.upper(): t for t in self.schema_metadata.tables.keys()}
        for table_upper in mentioned_tables:
            if table_upper not in schema_tables_upper:
                # Try to find similar table name
                similar = [
                    t
                    for t in schema_tables_upper.keys()
                    if table_upper in t or t in table_upper
                ]
                if similar:
                    issues.append(
                        f"Table '{table_upper}' not found. Did you mean '{schema_tables_upper[similar[0]]}'?"
                    )
                else:
                    issues.append(f"Table '{table_upper}' not found in schema")

        # Extract column references (table.column or just column)
        # This is a simplified check - full parsing would be more complex
        column_pattern = r"(\w+)\.(\w+)"
        column_refs = re.findall(column_pattern, sql_query)

        for table_ref, col_ref in column_refs:
            table_upper = table_ref.upper()
            if table_upper in schema_tables_upper:
                actual_table = schema_tables_upper[table_upper]
                table_meta = self.schema_metadata.tables[actual_table]
                columns_upper = {c.upper(): c for c in table_meta.columns.keys()}
                col_upper = col_ref.upper()
                if col_upper not in columns_upper:
                    similar = [
                        c
                        for c in columns_upper.keys()
                        if col_upper in c or c in col_upper
                    ]
                    if similar:
                        issues.append(
                            f"Column '{table_ref}.{col_ref}' not found. Did you mean '{actual_table}.{columns_upper[similar[0]]}'?"
                        )
                    else:
                        issues.append(
                            f"Column '{table_ref}.{col_ref}' not found in table '{actual_table}'"
                        )

        return len(issues) == 0, issues

    def _attempt_sql_fix(
        self, sql_query: str, error_msg: str, original_query: str
    ) -> str:
        """Attempt to fix SQL query based on error message.

        Args:
            sql_query: SQL query with error
            error_msg: Error message from database
            original_query: Original natural language query

        Returns:
            Potentially fixed SQL query
        """
        error_lower = error_msg.lower()
        fixed_query = sql_query

        # 0. Auto-heal unbalanced parentheses (common LLM truncation issue)
        open_count = fixed_query.count("(")
        close_count = fixed_query.count(")")
        if open_count > close_count:
            missing_closes = open_count - close_count
            logger.info(
                f"Auto-healing: Adding {missing_closes} missing closing parenthesis"
            )
            fixed_query += ")" * missing_closes

        # Common fixes
        # 1. Missing table qualifier for ambiguous columns
        if "ambiguous" in error_lower or "ambiguous column" in error_lower:
            # Try to add table qualifiers - this is complex, might need LLM help
            logger.debug("Ambiguous column detected, may need table qualifiers")

        # 2. Invalid column name - log for debugging
        if "no such column" in error_lower or "invalid column" in error_lower:
            logger.debug(f"Invalid column detected: {error_msg}")
            # Note: Specific fixes should be handled via context in config.yml
            # or through schema validation which provides suggestions

        # 3. Syntax errors - try basic fixes
        if "syntax error" in error_lower:
            # Remove trailing semicolons if present multiple times
            fixed_query = fixed_query.rstrip(";")
            # Ensure proper spacing
            fixed_query = re.sub(r"\s+", " ", fixed_query)

        # 4. Validate table/column names and suggest fixes
        is_valid, issues = self._validate_table_column_names(fixed_query)
        if not is_valid:
            logger.debug(f"Schema validation issues: {issues}")

        return fixed_query

    def _normalize_sql_to_lowercase(self, sql_query: str) -> str:
        """Normalize SQL query to use lowercase table and column names.

        Since we preprocess all table and column names to lowercase when importing
        data, we need to ensure SQL queries also use lowercase names.

        This method converts SQL identifiers (table/column names) to lowercase.
        For LIKE queries, string literals are also converted to lowercase since
        we use fuzzy matching and database data is lowercase.

        Args:
            sql_query: SQL query string

        Returns:
            SQL query with table and column names converted to lowercase
        """
        import re

        # First, identify LIKE patterns and convert their string literals to lowercase
        # Pattern: column LIKE 'pattern' or column LIKE "pattern"
        like_pattern = r'(\w+(?:\.\w+)?)\s+LIKE\s+([\'"])([^\'"]*(?:\'\'[^\'"]*)*)\2'

        def lower_like_string(match):
            column = match.group(1)
            quote = match.group(2)
            pattern = match.group(3)
            # Convert pattern to lowercase
            pattern_lower = pattern.lower()
            return f"{column.lower()} like {quote}{pattern_lower}{quote}"

        # Replace LIKE patterns first
        sql_query = re.sub(
            like_pattern, lower_like_string, sql_query, flags=re.IGNORECASE
        )

        # Now handle the rest: preserve string literals for exact matches (=) but convert identifiers
        result = []
        i = 0
        in_single_quote = False
        in_double_quote = False
        # Track if we're in a LIKE context (for the remaining string literals)
        # Since we already processed LIKE, remaining string literals are likely for exact matches
        # But to be safe, we'll convert all string literals to lowercase since our DB is lowercase

        while i < len(sql_query):
            char = sql_query[i]

            if char == "'" and not in_double_quote:
                # Check if it's an escaped single quote ('')
                if i + 1 < len(sql_query) and sql_query[i + 1] == "'":
                    result.append("''")
                    i += 2
                    continue
                # Toggle single quote state
                in_single_quote = not in_single_quote
                result.append(char)
            elif char == '"' and not in_single_quote:
                # Check if it's an escaped double quote ("")
                if i + 1 < len(sql_query) and sql_query[i + 1] == '"':
                    result.append('""')
                    i += 2
                    continue
                # Toggle double quote state
                in_double_quote = not in_double_quote
                result.append(char)
            elif in_single_quote or in_double_quote:
                # Inside string literal - convert to lowercase since DB data is lowercase
                result.append(char.lower())
            else:
                # Outside string literal - convert to lowercase
                result.append(char.lower())

            i += 1

        return "".join(result)

    def _agentic_loop(
        self,
        query: str,
        base_system_prompt: str,
        user_prompt_start: str,
        top_k: int,
        target_table_name: str,
        id_column: str,
    ) -> List[Text2SQLSearchResult]:
        """Execute the Think-Try-Fix loop."""

        max_retries = 2

        current_user_prompt = user_prompt_start
        last_error = None

        for attempt in range(max_retries + 1):
            logger.info(f"🔄 Agentic Loop: Attempt {attempt+1}/{max_retries+1}")

            # 1. Generate SQL
            if attempt > 0:
                # Add "reflection" instruction for retry with emphasis on completeness
                retry_instruction = f"""
## PREVIOUS ATTEMPT FAILED
Error/Issue: {last_error}

## ACTION REQUIRED
⚠️ CRITICAL: The previous SQL was INCOMPLETE or INCORRECT.

**Root cause analysis**:
- If it was a database error, correct the syntax or column names.
- If it returned 0 rows, RELAX the conditions (e.g., remove a strict filter, use broader LIKE).

**IMPORTANT - Generate COMPLETE SQL**:
- MUST close ALL opening parentheses with matching closing parentheses
- MUST have valid and complete syntax
- DO NOT truncate - return the FULL query from SELECT to the end
- Verify all WHERE/AND/OR clauses are properly closed

Try again with COMPLETE and VALID SQL.
"""
                current_user_prompt += retry_instruction

            # Call LLM
            try:
                response = self.llm.generate(
                    prompt=current_user_prompt
                    + "\\nReturn JSON with `thought` and `sql` fields.",
                    system_prompt=base_system_prompt,
                    temperature=0.0,
                    max_tokens=4096,  # Increased for complex SQL queries
                    response_format="json",
                )
            except Exception as e:
                # Handle Policy/Safety Errors by simplifying prompt
                if "invalid_prompt" in str(e) or "400" in str(e):
                    logger.warning(
                        "⚠️ Policy/Safety Error encountered. Retrying with simplified prompt..."
                    )
                    # Fallback: Strip context/examples to minimize trigger risk
                    # Use generic schema formatting for fallback
                    simple_user_prompt = f"Schema: {self._format_schema_for_prompt()}\\nQuestion: {query}\\nTask: Generate SQL JSON."
                    try:
                        response = self.llm.generate(
                            prompt=simple_user_prompt,
                            system_prompt="You are a SQL generator. Return JSON with thought and sql.",
                            temperature=0.0,
                            max_tokens=2048,  # Increased for fallback safety
                            response_format="json",
                        )
                    except Exception as fatal_e:
                        last_error = f"Fatal Safety Error: {fatal_e}"
                        logger.error(f"❌ {last_error}")
                        break
                else:
                    last_error = f"LLM Error: {e}"
                    logger.error(f"❌ {last_error}")
                    continue

            try:
                content = json.loads(response.content)
                thought = content.get("thought", "No thought provided.")
                sql_query = content.get("sql", "")

                logger.info(f"💭 Thought: {thought}")
                logger.info(f"📝 SQL: {sql_query}")

                # If JSON parsed but sql field is empty, try regex extraction
                if not sql_query or not sql_query.strip():
                    logger.warning(
                        "JSON parsed but 'sql' field is empty, trying regex extraction"
                    )
                    sql_query = self._extract_sql_from_response(response.content)
                    if not sql_query:
                        sql_query = self._extract_sql_from_response(thought)

            except Exception:
                # Fallback to plain text extraction
                logger.warning(
                    "Failed to parse JSON response, falling back to regex extraction"
                )
                sql_query = self._extract_sql_from_response(response.content)

            # Validate table/column names
            is_valid, issues = self._validate_table_column_names(sql_query)
            if not is_valid:
                last_error = f"Schema validation failed: {'; '.join(issues)}"
                logger.warning(f"❌ {last_error}")
                # Try simple fix before giving up on this attempt
                sql_query = self._attempt_sql_fix(sql_query, "; ".join(issues), query)

            # Execute
            try:
                sql_query_normalized = self._normalize_sql_to_lowercase(sql_query)
                # Strip trailing semicolon if present
                sql_query_normalized = sql_query_normalized.strip().rstrip(";")

                sql_query_final = self._ensure_id_column_in_select(sql_query_normalized)

                if "limit" not in sql_query_final.lower():
                    sql_query_final += f" limit {top_k}"

                df = self._execute_sql(sql_query_final)

                # Check for "Silent Failure" (0 rows) and not the last attempt
                if len(df) == 0:
                    if attempt < max_retries:
                        last_error = (
                            "Query returned 0 rows. Conditions might be too strict."
                        )
                        logger.warning(f"⚠️  {last_error}")
                        continue  # Retry loop
                    else:
                        logger.warning("⚠️  Final attempt returned 0 rows.")

                # Success
                if len(df) > 0:
                    logger.info(f"✅ Success: {len(df)} rows found.")
                    return [self._convert_dataframe_to_results(df, sql_query_final)]

            except Exception as e:
                last_error = f"SQL Execution Error: {str(e)}"
                logger.error(f"❌ {last_error}")
                continue

        # If we exit loop without success
        return [
            Text2SQLSearchResult(
                rank=1,
                table=target_table_name,
                data=[],
                sql_query=sql_query if "sql_query" in locals() else "",
                row_count=0,
                score=0.0,
            )
        ]

    def _execute_sql(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL query and return results.

        Args:
            sql_query: SQL query string

        Returns:
            DataFrame with query results

        Raises:
            Exception: If SQL execution fails
        """
        # Normalize SQL to lowercase before execution
        # (since we preprocess all table/column names to lowercase)
        sql_query = self._normalize_sql_to_lowercase(sql_query)

        logger.debug(f"Executing SQL: {sql_query[:100]}...")
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql_query))
                rows = result.fetchall()
                columns = result.keys()

                # Convert to DataFrame
                df = pd.DataFrame(rows, columns=columns)
                logger.debug(f"Query returned {len(df)} rows")
                return df
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            raise

    def _convert_dataframe_to_results(
        self, df: pd.DataFrame, sql_query: str, rank: int = 1
    ) -> Text2SQLSearchResult:
        """Convert DataFrame to Text2SQLSearchResult.

        Args:
            df: DataFrame with query results
            sql_query: The SQL query that was executed
            rank: Rank of the result

        Returns:
            Text2SQLSearchResult object
        """
        # Determine table name (use target table or "multiple" if joins)
        table_name = self.target_table
        if "JOIN" in sql_query.upper() or "FROM" in sql_query.upper():
            # Try to detect if multiple tables are involved
            from_match = re.search(r"FROM\s+(\w+)", sql_query, re.IGNORECASE)
            if from_match:
                table_name = from_match.group(1)

        # Convert DataFrame to list of dicts
        id_column = self._get_target_id_column()
        target_cols = None
        if self.target_table in self.schema_metadata.tables:
            target_cols = self.schema_metadata.tables[self.target_table].columns

        sql_upper = sql_query.upper()
        target_table_in_query = bool(
            re.search(rf"\b{re.escape(self.target_table.upper())}\b", sql_upper)
        )

        if (
            target_table_in_query
            and id_column
            and len(df.columns) >= 1
            and any(c.lower() == id_column.lower() for c in df.columns)
            and target_cols
            and any(c.lower() == id_column.lower() for c in target_cols)
        ):
            df_cols_lower = {c.lower() for c in df.columns}
            target_cols_lower = {c.lower() for c in target_cols}
            should_expand = df_cols_lower != target_cols_lower

            if should_expand:
                id_col_in_df = next(
                    c for c in df.columns if c.lower() == id_column.lower()
                )
                ids = [str(v) for v in df[id_col_in_df].tolist() if v is not None]
                if ids:
                    placeholders = ", ".join(f":id_{i}" for i in range(len(ids)))
                    params = {f"id_{i}": ids[i] for i in range(len(ids))}
                    with self.engine.connect() as conn:
                        expanded = conn.execute(
                            text(
                                f"SELECT * FROM {self.target_table} "
                                f"WHERE {id_column} IN ({placeholders})"
                            ),
                            params,
                        )
                        rows = expanded.fetchall()
                        columns = expanded.keys()
                    expanded_df = pd.DataFrame(rows, columns=columns)
                    expanded_id_col = next(
                        (
                            c
                            for c in expanded_df.columns
                            if c.lower() == id_column.lower()
                        ),
                        None,
                    )
                    if expanded_id_col:
                        by_id = {
                            str(row[expanded_id_col]): row
                            for row in expanded_df.to_dict("records")
                        }
                        ordered = [by_id.get(i) for i in ids]
                        df = pd.DataFrame([r for r in ordered if r is not None])
                    else:
                        df = expanded_df

        data = df.to_dict("records")

        return Text2SQLSearchResult(
            rank=rank,
            table=table_name,
            data=data,
            sql_query=sql_query,
            row_count=len(data),
            score=1.0,
        )

    def close(self):
        """Close database connection."""
        if self._own_engine:
            self.engine.dispose()
            logger.info("Database connection closed")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass
