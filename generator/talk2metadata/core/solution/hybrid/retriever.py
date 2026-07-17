from __future__ import annotations

import json
import re
from typing import List

from talk2metadata.core.schema.schema import SchemaMetadata
from talk2metadata.utils.logging import get_logger

from ..modes.registry import (
    BaseRetriever,
    get_mode_retriever_config,
)
from ..paths.text2sql.base import (
    BaseText2SQLRetriever,
    Text2SQLSearchResult,
)

logger = get_logger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Neuro-Symbolic Hybrid Retriever.

    Combines:
    1. Vector Search (RecordVoter) for high RECALL on semantics/text.
    2. SQL Generation (LLM) for high PRECISION on structured filters (Dates, IDs, Categories).

    Strategy:
    - Run Vector Search -> get candidate IDs.
    - Run SQL Gen (Filter Mode) -> get structured filter clause.
    - Execute SQL: `SELECT * FROM table WHERE {structured_filters} AND id IN ({vector_ids})`
    """

    def __init__(
        self,
        schema_metadata: SchemaMetadata,
        connection_string: str,
        vector_retriever: BaseRetriever,
        llm_retriever: BaseText2SQLRetriever,
        mode_name: str = "hybrid",
    ):
        self.schema_metadata = schema_metadata
        self.connection_string = connection_string
        self.vector_retriever = vector_retriever
        self.llm_retriever = llm_retriever
        self.mode_name = mode_name

    def _generate_soft_filter_sql(
        self, query: str, vector_ids: set, top_k: int = 20
    ) -> tuple:
        """Generate SQL with soft filtering (score-based) instead of strict AND conditions.

        Returns:
            (sql_query, threshold): SQL string and minimum score threshold
        """
        # Use the LLM to identify filterable criteria
        target_table = self.llm_retriever.target_table
        id_column = self.llm_retriever._get_target_id_column() or "ANumber"
        schema_text = self.llm_retriever._format_schema_for_prompt()
        domain_config = getattr(self.llm_retriever, "domain_config", {}) or {}
        sql_rules = (
            domain_config.get("sql_rules", [])
            if isinstance(domain_config, dict)
            else []
        )
        sql_rules_text = "\n\n".join(
            f"- {r.get('name', '')}: {r.get('description', '')}\n{r.get('rule', '').strip()}"
            for r in sql_rules
            if isinstance(r, dict)
        ).strip()

        prompt = f"""You are generating a SOFT FILTERING SQL query for hybrid search.

## CRITICAL CONCEPT: Score-Based Filtering
Instead of strict AND conditions (which return 0 results if ANY condition fails),
assign POINTS for each matched criterion and return results with score >= threshold.

## SCHEMA
{schema_text}

## DATA-SPECIFIC RULES
{sql_rules_text if sql_rules_text else "No extra data-specific rules."}

## QUERY
{query}

## YOUR TASK
Identify filterable criteria (dates, status, IDs, exact names) and assign point values:
- **High priority** (exact matches, required fields): 3 points
- **Medium priority** (important but flexible): 2 points
- **Low priority** (nice-to-have): 1 point

## EXAMPLE
Query: "Open file gold reports from 2010"

Instead of:
```sql
WHERE Confidentiality = 'OPEN FILE' AND TargetCommoditiesNames LIKE '%gold%' AND ReportDate LIKE '%2010%'
```

Use:
```sql
SELECT *,
  (CASE WHEN lower(Confidentiality) = 'open file' THEN 3 ELSE 0 END +
   CASE WHEN lower(TargetCommoditiesNames) LIKE '%gold%' THEN 2 ELSE 0 END +
   CASE WHEN ReportDate LIKE '%2010%' THEN 2 ELSE 0 END) as filter_score
FROM {target_table}
WHERE filter_score >= 5  -- At least 2 out of 3 criteria
ORDER BY filter_score DESC
LIMIT {top_k}
```

## RULES
1. Use `lower()` for text comparisons
2. Each CASE statement adds points (0 if no match)
3. Set threshold to ~70% of max possible score
4. If no filterable criteria found, return: SELECT * FROM {target_table} LIMIT {top_k}
5. Always return at most {top_k} rows. Never use a larger LIMIT.

Return JSON with:
- "criteria": List of identified criteria with point values
- "max_score": Maximum possible score
- "threshold": Recommended minimum score (70% of max)
- "sql": The score-based SQL query
"""

        try:
            response = self.llm_retriever.llm.generate(
                prompt=prompt,
                system_prompt="You are a SQL generator specializing in flexible, score-based queries.",
                temperature=0.0,
                max_tokens=2048,
                response_format="json",
            )

            result = json.loads(response.content)
            sql = result.get("sql", "")
            threshold = result.get("threshold", 0)

            logger.info(f"    Generated soft filter SQL (threshold: {threshold})")
            return sql, threshold

        except Exception as e:
            logger.warning(f"    Soft filter generation failed: {e}, using fallback")
            # Fallback: simple SELECT with vector ID filtering
            vector_id_list = ", ".join(f"'{vid}'" for vid in list(vector_ids)[:200])
            fallback_sql = f"SELECT * FROM {target_table} WHERE {id_column} IN ({vector_id_list}) LIMIT {top_k}"
            return fallback_sql, 0

    def _enforce_limit(self, sql: str, top_k: int) -> str:
        if not isinstance(sql, str):
            return sql
        sql_stripped = sql.strip().rstrip(";")
        if re.search(r"\blimit\s+\d+\b", sql_stripped, flags=re.IGNORECASE):
            sql_stripped = re.sub(
                r"\blimit\s+\d+\b",
                f"LIMIT {top_k}",
                sql_stripped,
                flags=re.IGNORECASE,
            )
            return f"{sql_stripped};"
        return f"{sql_stripped} LIMIT {top_k};"

    def _truncate_text2sql_results(
        self, results: List[Text2SQLSearchResult], top_k: int
    ) -> List[Text2SQLSearchResult]:
        truncated: List[Text2SQLSearchResult] = []
        for res in results or []:
            if not hasattr(res, "data") or not isinstance(res.data, list):
                truncated.append(res)
                continue
            data = res.data[:top_k]
            truncated.append(
                Text2SQLSearchResult(
                    rank=res.rank,
                    table=res.table,
                    data=data,
                    sql_query=res.sql_query,
                    row_count=len(data),
                    score=res.score,
                )
            )
        return truncated

    def search(self, query: str, top_k: int = 20) -> List[Text2SQLSearchResult]:
        logger.info(f"🧠 Hybrid Search: '{query}'")

        # Get config
        retriever_config = get_mode_retriever_config(self.mode_name)
        vector_top_k = retriever_config.get("vector_top_k", 100)
        fallback_to_vector = retriever_config.get("fallback_to_vector", True)
        use_soft_filtering = retriever_config.get("use_soft_filtering", True)
        prefer_strict_sql = retriever_config.get("prefer_strict_sql", True)

        vector_results = []
        vector_ids = set()

        should_run_vector = fallback_to_vector or prefer_strict_sql is False
        if should_run_vector:
            logger.info(f"🔍 Running Vector Search (Top-{vector_top_k})...")
            try:
                vector_results = self.vector_retriever.search(query, top_k=vector_top_k)
                vector_ids = {str(res.row_id) for res in vector_results}
                logger.info(f"    Found {len(vector_ids)} vector candidates.")
            except Exception as e:
                logger.warning(f"    Vector search failed: {type(e).__name__}: {e}")
                vector_results = []
                vector_ids = set()

        if prefer_strict_sql:
            logger.info("🛡️ Running STRICT SQL (agentic) for precision...")
            try:
                sql_results = self.llm_retriever.search(query, top_k=top_k)
                sql_results = self._truncate_text2sql_results(sql_results, top_k=top_k)
                if any(r.data for r in sql_results if isinstance(r.data, list)):
                    return sql_results
            except Exception as e:
                logger.warning(f"    Strict SQL failed: {type(e).__name__}: {e}")

        if use_soft_filtering:
            logger.info("🛡️ Running SOFT SQL (score-based) as fallback...")
            try:
                soft_sql, threshold = self._generate_soft_filter_sql(
                    query, vector_ids, top_k
                )
                soft_sql = self._enforce_limit(soft_sql, top_k=top_k)
                df = self.llm_retriever._execute_sql(soft_sql)

                if len(df) == 0 and threshold:
                    relaxed_threshold = max(1, int(threshold * 0.5))
                    relaxed_sql = soft_sql.replace(
                        f"filter_score >= {threshold}",
                        f"filter_score >= {relaxed_threshold}",
                    )
                    relaxed_sql = self._enforce_limit(relaxed_sql, top_k=top_k)
                    df = self.llm_retriever._execute_sql(relaxed_sql)
                    soft_sql = relaxed_sql

                if len(df) > 0:
                    rows = df.head(top_k).to_dict("records")
                    return [
                        Text2SQLSearchResult(
                            rank=1,
                            table=self.llm_retriever.target_table,
                            data=rows,
                            sql_query=soft_sql,
                            row_count=len(rows),
                            score=1.0,
                        )
                    ]
            except Exception as e:
                logger.warning(f"    Soft SQL failed: {type(e).__name__}: {e}")

        if fallback_to_vector and vector_results:
            logger.info("Falling back to pure vector results...")
            return [
                Text2SQLSearchResult(
                    rank=i + 1,
                    table=vr.table,
                    data=[vr.data],
                    sql_query="-- vector search fallback --",
                    row_count=1,
                    score=1.0,
                )
                for i, vr in enumerate(vector_results[:top_k])
            ]

        return []
