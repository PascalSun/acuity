"""Two-step Text2SQL Retriever - Question → locate columns/tables → SQL → results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.timing import timed

from .base import (
    BaseText2SQLRetriever,
    Text2SQLSearchResult,
)

logger = get_logger(__name__)


class TwoStepText2SQLRetriever(BaseText2SQLRetriever):
    """Two-step text2sql retriever: Question → locate columns/tables → SQL → results.

    This approach first analyzes the question to identify relevant columns and tables,
    then generates SQL using only the relevant schema information.
    """

    def __init__(
        self,
        schema_metadata,
        connection_string: Optional[str] = None,
        engine=None,
        vector_retriever: Optional[Any] = None,
        few_shot_manager: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(schema_metadata, connection_string, engine, **kwargs)
        self.vector_retriever = vector_retriever
        self.few_shot_manager = few_shot_manager

        # Load domain-specific configuration
        self.domain_config = self._load_domain_config()
        logger.info(
            f"📋 Loaded domain config: {len(self.domain_config.get('enum_patterns', []))} enum patterns, "
            f"{len(self.domain_config.get('sql_rules', []))} SQL rules"
        )

        if self.vector_retriever:
            logger.info("✅ VASQL Enabled: Vector Augmentation ready")
        if self.few_shot_manager:
            logger.info("✅ Agentic SQL Enabled: Dynamic Few-Shot ready")

    def _load_domain_config(self) -> Dict:
        """Load domain-specific configuration from domain_config.yml.

        Returns:
            Dict with domain-specific patterns, rules, and examples
        """
        # Try to find domain_config.yml in project root
        config_paths = [
            Path("domain_config.yml"),  # Current directory (fallback)
            Path("data/wamex/domain_config.yml"),  # Default domain data location
            Path("data/domain_config.yml"),  # Generic data location
            Path(__file__).parent.parent.parent.parent.parent
            / "domain_config.yml",  # Project root
        ]

        for config_path in config_paths:
            if config_path.exists():
                logger.info(f"📁 Loading domain config from: {config_path}")
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                    return config.get("domain", {})

        # Fallback to default config if file not found
        logger.warning("⚠️  domain_config.yml not found, using default patterns")
        return {
            "enum_patterns": ["type", "status", "category", "class"],
            "sql_rules": [],
            "constraint_examples": [],
        }

    @timed("text2sql.two_step.search")
    def search(self, query: str, top_k: int = 5) -> List[Text2SQLSearchResult]:
        """Search using two-step approach: locate relevant schema, then generate SQL.

        Args:
            query: Natural language question
            top_k: Maximum number of results (used as LIMIT in SQL)

        Returns:
            List of Text2SQLSearchResult objects
        """
        # Log question
        logger.info("=" * 80)
        logger.info(f"🔍 QUESTION: {query}")
        logger.info("=" * 80)

        # Step 1: Locate relevant columns and tables
        logger.info("🔎 Step 1: Locating relevant schema elements...")
        relevant_schema = self._locate_relevant_schema(query)
        logger.info(f"   Located: {relevant_schema}")

        # Step 1.5: Get Vector Hints (VASQL) and Few-Shot Examples (Agentic)
        vector_hints = ""
        if self.vector_retriever:
            try:
                logger.info("🧠 VASQL: Retrieving vector hints...")
                vector_results = self.vector_retriever.search(query, top_k=3)
                vector_hints = self._format_vector_hints(vector_results)
                logger.info(f"   Generated hints: {len(vector_hints)} chars")
            except Exception as e:
                logger.warning(f"VASQL failed: {e}")

        # Agentic SQL: Get Dynamic Few-Shot Examples
        few_shot_examples = ""
        if self.few_shot_manager:
            try:
                logger.info("🧠 Agentic: Retrieving similar past examples...")
                # Pass schema context for schema-aware ranking
                schema_context = {"tables": list(relevant_schema.keys())}
                examples = self.few_shot_manager.retrieve(
                    query, k=3, schema_context=schema_context
                )
                few_shot_examples = self.few_shot_manager.format_for_prompt(examples)
                logger.info(f"   Found {len(examples)} examples (schema-aware)")
            except Exception as e:
                logger.warning(f"Agentic Few-Shot failed: {e}")

        # Step 2: Generate SQL using Agentic Loop (Think-Try-Fix)
        schema_text = self._format_relevant_schema(relevant_schema)

        target_table_name = self.target_table
        id_column = self._get_target_id_column() or "id"

        # Classify query complexity and get specific rules
        complexity = self._classify_query_complexity(relevant_schema, query)
        complexity_rules = self._get_complexity_specific_rules(complexity)
        logger.info(f"📊 Query Complexity: {complexity.upper()}")

        # Prepare Base Prompt Parts
        base_system_prompt = f"""You are an expert SQL query generator. Convert natural language questions into accurate, executable SQL queries.

## DATABASE TYPE
**Database: SQLite**
- Use SQLite-compatible SQL syntax
- SQLite LIKE operator condition all be lowercase
- Follow SQLite syntax rules and limitations

## TASK OVERVIEW
Your goal: Find records from the TARGET TABLE ({target_table_name}) that match the question's conditions.
- Always return records identified by {target_table_name}.{id_column} (the primary key)
- You may JOIN other tables to apply filters, but results must be from {target_table_name}
- Use ONLY the tables and columns provided in the relevant schema

{complexity_rules}

## DATA-SPECIFIC RULES
1. **Date Format**: Wamex reports use a special date format: `/date(milliseconds_since_epoch)/`.
   - SQLite `LIKE` operator is required for year/month filtering (e.g., `reportdate LIKE '/date(1233414000000)/'`).
   - If only a year is provided (e.g., "2009"), use `reportdate LIKE '%2009%'` to match the ms-timestamp representation in text.
2. **Joins & Recall (Multi-Entity Constraint)**: If the query mentions multiple entity types (e.g. "geochemistry AND drilling"), you MUST ensure the result report has BOTH.
   - Use `INNER JOIN` if you are confident the related table contains the filter criteria.
   - Alternatively, check if the `Keywords` or `Abstract` columns in `wamex_reports` mention the entity type to avoid missing records not indexed in detail tables.
   - For mandatory intersection (e.g. "reports with both X and Y"), `INNER JOIN` is preferred on `ANumber`.
3. **Primary Selection**: Results MUST come from the target table {target_table_name}. Select `{target_table_name}.{id_column}`.
4. **Lowercase Logic**: Apply `lower()` or `LIKE` with lowercase patterns for all text columns.
5. **Junction Logic**: Use `TargetCommoditiesIds` patterns: `(ids = '49' OR ids LIKE '49,%' OR ids LIKE '%,49' OR ids LIKE '%,49,%')`.

## CHAIN OF THOUGHT (Use 'thought' key in JSON)
Before writing SQL, you must explain your logic:
1. Identify key entities: "User is asking about 'X'..."
2. Map to Schema: "This maps to table T, column C..."
3. Strategy: "I will join T1 and T2 to enforce existence..."
4. Pitfall Check: "I must use LIKE for text and ensure lowercase..."

## STANDARD QUERY STRUCTURE
```sql
select distinct {target_table_name}.{id_column} from {target_table_name} ...
```
"""

        user_prompt_start = f"""{schema_text}

{few_shot_examples}

## VECTOR SEARCH HINTS (Use these values for WHERE clauses)
{vector_hints if vector_hints else "No hints found."}

## Question
{query}

## Your Task
1. Analyze the request.
2. Generate the SQL query.
"""

        # Execute Agentic Loop
        final_result = self._agentic_loop(
            query=query,
            base_system_prompt=base_system_prompt,
            user_prompt_start=user_prompt_start,
            top_k=100,  # Increased to 100 to improve Recall
            target_table_name=target_table_name,
            id_column=id_column,
        )

        return final_result

    def _locate_relevant_schema(self, query: str) -> Dict[str, List[str]]:
        """Locate relevant tables and columns for the query.

        Args:
            query: Natural language question

        Returns:
            Dict mapping table_name -> list of relevant column names
        """
        # Format full schema for analysis
        full_schema = self._format_schema_for_prompt()

        system_prompt = """You are a database schema analyzer. Analyze the question and identify which tables and columns are relevant.
Return a JSON object with table names as keys and lists of relevant column names as values.
Example: {"customers": ["id", "name", "email"], "orders": ["id", "customer_id", "total"]}
If a table is relevant but you're not sure about specific columns, include all columns from that table."""

        user_prompt = f"""{full_schema}

Question: {query}

Identify which tables and columns are relevant to answer this question. Return only the JSON object."""

        logger.info("Locating relevant schema elements...")
        response = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=2048,  # Increased to prevent schema truncation
            response_format="json",
        )

        try:
            relevant_schema = json.loads(response.content)
            if not isinstance(relevant_schema, dict):
                raise ValueError("Response is not a dictionary")
            return relevant_schema
        except Exception as e:
            logger.warning(
                f"Failed to parse relevant schema JSON: {e}, using full schema"
            )
            # Fallback: return all tables and columns
            return {
                table_name: list(table_meta.columns.keys())
                for table_name, table_meta in self.schema_metadata.tables.items()
            }

    def _classify_query_complexity(
        self, relevant_schema: Dict[str, List[str]], query: str
    ) -> str:
        """Classify query complexity based on schema and query content.

        Args:
            relevant_schema: Dict mapping table names to columns
            query: Natural language question

        Returns:
            'simple', 'medium', or 'complex'
        """
        num_tables = len(relevant_schema)
        query_lower = query.lower()

        # Simple: Single table, direct field access
        if num_tables == 1:
            return "simple"

        # Complex: 3+ tables or mentions multiple distinct entity types
        if num_tables >= 3:
            return "complex"

        # Check for complex keywords
        complex_keywords = ["and", "both", "all", "multiple", "various", "different"]
        if any(kw in query_lower for kw in complex_keywords):
            # If query mentions multiple entities with 2+ tables, likely complex
            entity_mentions = sum(
                1
                for table in relevant_schema.keys()
                if table.lower().replace("_", " ") in query_lower
            )
            if entity_mentions >= 2:
                return "complex"

        # Medium: 2 tables, standard joins
        return "medium"

    def _get_complexity_specific_rules(self, complexity: str) -> str:
        """Get SQL generation rules specific to query complexity.

        Args:
            complexity: 'simple', 'medium', or 'complex'

        Returns:
            Additional rules text for the prompt
        """
        if complexity == "simple":
            return """
## SIMPLE QUERY STRATEGY
- **No JOINs needed** - all data is in a single table
- **Direct field matching** - use exact equality or LIKE
- **Keep it minimal** - avoid unnecessary complexity
- **Focus on WHERE conditions** only
"""
        elif complexity == "complex":
            return """
## COMPLEX MULTI-ENTITY QUERY STRATEGY
⚠️ **CRITICAL**: This query involves 3+ tables. Use flexible join patterns.

### RECOMMENDED APPROACH: WHERE EXISTS
Instead of cascading INNER JOINs (which are too restrictive):

❌ WRONG (too strict):
```sql
SELECT wr.ANumber
FROM wamex_reports wr
INNER JOIN geo_chemistry gc ON gc.ANumber = wr.ANumber
INNER JOIN drilling_summaries ds ON ds.ANumber = wr.ANumber
INNER JOIN storages st ON st.ANumber = wr.ANumber
```

✅ CORRECT (flexible):
```sql
SELECT wr.ANumber
FROM wamex_reports wr
WHERE EXISTS (SELECT 1 FROM geo_chemistry WHERE ANumber = wr.ANumber)
  AND EXISTS (SELECT 1 FROM drilling_summaries WHERE ANumber = wr.ANumber)
  AND EXISTS (SELECT 1 FROM storages WHERE ANumber = wr.ANumber)
```

### WHY?
- WHERE EXISTS checks for **existence** without requiring exact matches
- INNER JOIN cascades eliminate rows if ANY join fails
- For multi-entity queries, we want "reports that have X AND Y", not "perfect matches"

### ALTERNATIVE: LEFT JOINs with COUNT
```sql
SELECT wr.ANumber
FROM wamex_reports wr
LEFT JOIN geo_chemistry gc ON gc.ANumber = wr.ANumber
LEFT JOIN drilling_summaries ds ON ds.ANumber = wr.ANumber
WHERE gc.ANumber IS NOT NULL  -- Has geochemistry
  AND ds.ANumber IS NOT NULL  -- Has drilling
```
"""
        else:  # medium
            return """
## MEDIUM QUERY STRATEGY (2 tables)
- **Use INNER JOIN** for guaranteed relationships
- **Verify join keys** match the schema
- **Filter on both tables** if needed
- **Consider LEFT JOIN** if one entity is optional

### Example:
```sql
SELECT wr.ANumber
FROM wamex_reports wr
INNER JOIN geo_chemistry gc ON gc.ANumber = wr.ANumber
WHERE wr.Confidentiality = 'OPEN FILE'
  AND gc.NumberOfSamples > 10
```
"""

    def _is_enum_like_column(self, column_name: str) -> bool:
        """Check if column likely contains categorical/enum values.

        These columns benefit from showing sample values to help LLM
        identify implicit constraints (e.g., 'stream sediment' for sampletype).
        """
        col_lower = column_name.lower()
        enum_patterns = [
            "type",
            "status",
            "category",
            "class",
            "confidentiality",
            "sample",
            "survey",
            "hole",
            "commodity",
            "operator",
            "author",
        ]
        return any(pattern in col_lower for pattern in enum_patterns)

    def _get_column_value_samples(
        self, table_name: str, column_name: str, limit: int = 5
    ) -> List[str]:
        """Get sample distinct values for a column.

        Args:
            table_name: Table name
            column_name: Column name
            limit: Maximum number of samples to return

        Returns:
            List of sample values (as strings)
        """
        try:
            # Use DISTINCT to get unique values, filter out nulls
            sql = f"""
                SELECT DISTINCT {column_name}
                FROM {table_name}
                WHERE {column_name} IS NOT NULL
                  AND {column_name} != ''
                LIMIT {limit}
            """
            df = self._execute_sql(sql)
            if len(df) > 0:
                return [str(val) for val in df[column_name].tolist()]
            return []
        except Exception as e:
            logger.debug(f"Failed to get samples for {table_name}.{column_name}: {e}")
            return []

    def _format_relevant_schema(self, relevant_schema: Dict[str, List[str]]) -> str:
        """Format relevant schema information for SQL generation prompt.

        Args:
            relevant_schema: Dict mapping table_name -> list of column names

        Returns:
            Formatted schema string with only relevant information
        """
        parts = ["# Relevant Database Schema\n"]
        parts.append(
            "IMPORTANT: Use EXACT table and column names as shown below. Case sensitivity matters!\n"
        )

        for table_name in relevant_schema.keys():
            if table_name not in self.schema_metadata.tables:
                logger.warning(f"Table {table_name} not found in schema metadata")
                continue

            table_meta = self.schema_metadata.tables[table_name]
            relevant_columns = relevant_schema[table_name]

            # Ensure relevant_columns is a list (handle case where it might be a dict)
            if isinstance(relevant_columns, dict):
                relevant_columns = list(relevant_columns.keys())
            elif not isinstance(relevant_columns, list):
                relevant_columns = (
                    list(relevant_columns)
                    if hasattr(relevant_columns, "__iter__")
                    else []
                )

            parts.append(f"## Table: {table_name}")
            if table_name == self.target_table:
                parts.append(
                    "  ⭐ THIS IS THE TARGET TABLE (use this as the main table)"
                )
            # Add table description if available
            if table_meta.description:
                parts.append(f"  Description: {table_meta.description}")
            if table_meta.primary_key:
                parts.append(f"  Primary Key: {table_meta.primary_key}")

            # Format columns with descriptions
            parts.append("  Relevant Columns:")
            for col in relevant_columns:
                col_info = f"{col}"
                if col in table_meta.columns:
                    col_info += f" ({table_meta.columns[col]})"
                # Add column description if available
                if col in table_meta.column_descriptions:
                    col_info += f" - {table_meta.column_descriptions[col]}"
                parts.append(f"    - {col_info}")

            # Include sample values for relevant columns (ENHANCED)
            # Priority 1: Dynamic samples for enum-like columns
            # Priority 2: Static samples from schema metadata
            parts.append(
                "  Sample values (use EXACT format including spaces and case):"
            )

            samples_shown = False
            for col in relevant_columns:
                # Try dynamic sampling for enum-like columns first
                if self._is_enum_like_column(col):
                    dynamic_samples = self._get_column_value_samples(
                        table_name, col, limit=5
                    )
                    if dynamic_samples:
                        sample_strs = [f"'{val}'" for val in dynamic_samples[:5]]
                        sample = ", ".join(sample_strs)
                        parts.append(f"    {col}: {sample}")
                        samples_shown = True
                        continue

                # Fallback to static samples from metadata
                if table_meta.sample_values and col in table_meta.sample_values:
                    vals = table_meta.sample_values[col]
                    sample_strs = [f"'{str(v)}'" for v in vals[:3]]
                    sample = ", ".join(sample_strs)
                    parts.append(f"    {col}: {sample}")
                    samples_shown = True

            if not samples_shown:
                parts.append("    (No sample values available)")

            parts.append("")

        # Include relevant foreign keys
        if self.schema_metadata.foreign_keys:
            parts.append("## Relevant Foreign Key Relationships\n")
            parts.append("Use these EXACT relationships to JOIN tables:\n")
            relevant_tables = set(relevant_schema.keys())
            for fk in self.schema_metadata.foreign_keys:
                if (
                    fk.child_table in relevant_tables
                    or fk.parent_table in relevant_tables
                ):
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

    def _format_vector_hints(self, output_results: List[Any]) -> str:
        """Format vector search results as hints."""
        hints = []
        for i, res in enumerate(output_results):
            # res is likely RecordVoteSearchResult or similar
            # Extract key/values from data
            if not hasattr(res, "data") or not res.data:
                continue

            row_hints = []
            for k, v in res.data.items():
                if v and isinstance(v, str):
                    row_hints.append(f"{k}='{v}'")

            if row_hints:
                hints.append(f"- Record {i+1}: {', '.join(row_hints)}")

                hints.append(f"- Record {i+1}: {', '.join(row_hints)}")

        return "\n".join(hints)
