from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from mcp.server import Server
from mcp.types import Resource, ResourceTemplate

from talk2metadata.utils.config import get_config
from talk2metadata.utils.paths import find_schema_file, get_run_base_dir

from ..common.schema_index import get_schema

_RUN_DETAILS_RE = re.compile(r"^resource://talk2metadata/run/([^/]+)$")
_RUN_CONTEXT_RE = re.compile(r"^resource://talk2metadata/run/([^/]+)/context$")
_RUN_TABLES_RE = re.compile(r"^resource://talk2metadata/run/([^/]+)/tables$")
_RUN_SCHEMA_RE = re.compile(r"^resource://talk2metadata/run/([^/]+)/schema$")
_RUN_TABLE_RE = re.compile(r"^resource://talk2metadata/run/([^/]+)/table/(.+)$")


def _discover_runs() -> list[dict[str, str]]:
    config = get_config()
    base_dir = get_run_base_dir(run_id=None, config=config)
    if not base_dir.exists():
        return []

    runs: list[dict[str, str]] = []
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            continue

        metadata_dir = child / "metadata"
        if not metadata_dir.exists():
            continue

        schema_path: Path | None
        try:
            schema_path = find_schema_file(metadata_dir, target_table=None)
        except Exception:
            schema_path = None

        if schema_path is None or not schema_path.exists():
            continue

        runs.append(
            {
                "run_id": child.name,
                "schema_path": str(schema_path),
                "updated_at": datetime.fromtimestamp(
                    schema_path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        )

    runs.sort(key=lambda r: r["run_id"])
    return runs


def _read_runs() -> str:
    config = get_config()
    configured_run_id = config.get("run_id")
    runs = _discover_runs()
    return json.dumps(
        {
            "configured_run_id": configured_run_id,
            "run_count": len(runs),
            "runs": [
                {
                    **r,
                    "resources": {
                        "run": f"resource://talk2metadata/run/{r['run_id']}",
                        "context": f"resource://talk2metadata/run/{r['run_id']}/context",
                        "tables": f"resource://talk2metadata/run/{r['run_id']}/tables",
                        "schema": f"resource://talk2metadata/run/{r['run_id']}/schema",
                    },
                }
                for r in runs
            ],
        },
        indent=2,
    )


def _read_run_details(run_id: str) -> str:
    config = get_config()
    run_base = get_run_base_dir(run_id=run_id, config=config)
    try:
        schema = get_schema(run_id=run_id)
        table_names = sorted(schema.tables.keys())
        table_preview = table_names[:10]
        schema_summary: dict[str, object] = {
            "target_table": schema.target_table,
            "table_count": len(schema.tables),
            "foreign_key_count": len(schema.foreign_keys),
            "tables_preview": table_preview,
        }
    except FileNotFoundError as e:
        schema_summary = {
            "error": "Schema not found",
            "error_code": "SCHEMA_NOT_FOUND",
            "message": str(e),
        }

    return json.dumps(
        {
            "id": f"resource://talk2metadata/run/{run_id}",
            "run_id": run_id,
            "paths": {
                "run_base_dir": str(run_base),
                "metadata_dir": str(run_base / "metadata"),
                "processed_dir": str(run_base / "processed"),
                "indexes_dir": str(run_base / "indexes"),
                "raw_dir": str(run_base / "raw"),
                "db_dir": str(run_base / "db"),
            },
            "schema": schema_summary,
            "resources": {
                "context": f"resource://talk2metadata/run/{run_id}/context",
                "tables": f"resource://talk2metadata/run/{run_id}/tables",
                "schema": f"resource://talk2metadata/run/{run_id}/schema",
                "table": f"resource://talk2metadata/run/{run_id}/table/{{table_name}}",
            },
        },
        indent=2,
    )


def _read_table_by_name(table_name: str, run_id: str) -> str:
    try:
        schema = get_schema(run_id=run_id)

        if table_name not in schema.tables:
            available_tables = list(schema.tables.keys())
            return json.dumps(
                {
                    "error": "Table not found",
                    "error_code": "TABLE_NOT_FOUND",
                    "run_id": run_id,
                    "table_name": table_name,
                    "message": f"No table found with name '{table_name}'.",
                    "available_tables": available_tables,
                },
                indent=2,
            )

        meta = schema.tables[table_name]
        related_tables = schema.get_related_tables(table_name)
        fks = schema.get_foreign_keys_for_table(table_name)

        return json.dumps(
            {
                "id": f"resource://talk2metadata/run/{run_id}/table/{table_name}",
                "run_id": run_id,
                "name": table_name,
                "is_target": table_name == schema.target_table,
                "columns": meta.columns,
                "primary_key": meta.primary_key,
                "row_count": meta.row_count,
                "sample_values": meta.sample_values,
                "related_tables": related_tables,
                "foreign_keys": [
                    {
                        "child_table": fk.child_table,
                        "child_column": fk.child_column,
                        "parent_table": fk.parent_table,
                        "parent_column": fk.parent_column,
                        "coverage": fk.coverage,
                    }
                    for fk in fks
                ],
            },
            indent=2,
        )
    except FileNotFoundError as e:
        return json.dumps(
            {
                "error": "Schema not found",
                "error_code": "SCHEMA_NOT_FOUND",
                "run_id": run_id,
                "message": str(e),
                "hint": "Please run 'talk2metadata ingest' to load data first.",
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps(
            {
                "error": "Failed to read table information",
                "error_code": "READ_ERROR",
                "run_id": run_id,
                "message": str(e),
            },
            indent=2,
        )


def _read_schema(run_id: str) -> str:
    try:
        schema = get_schema(run_id=run_id)

        tables: dict[str, object] = {}
        for name, meta in schema.tables.items():
            tables[name] = {
                "id": f"resource://talk2metadata/run/{run_id}/table/{name}",
                "columns": meta.columns,
                "primary_key": meta.primary_key,
                "row_count": meta.row_count,
                "sample_values": meta.sample_values,
            }

        foreign_keys = [
            {
                "child_table": fk.child_table,
                "child_column": fk.child_column,
                "parent_table": fk.parent_table,
                "parent_column": fk.parent_column,
                "coverage": fk.coverage,
            }
            for fk in schema.foreign_keys
        ]

        return json.dumps(
            {
                "id": f"resource://talk2metadata/run/{run_id}/schema",
                "run_id": run_id,
                "target_table": schema.target_table,
                "table_count": len(tables),
                "foreign_key_count": len(foreign_keys),
                "tables": tables,
                "foreign_keys": foreign_keys,
            },
            indent=2,
        )
    except FileNotFoundError as e:
        return json.dumps(
            {
                "error": "Schema not found",
                "error_code": "SCHEMA_NOT_FOUND",
                "run_id": run_id,
                "message": str(e),
                "hint": "Please run 'talk2metadata ingest' to load data first.",
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps(
            {
                "error": "Failed to read schema",
                "error_code": "READ_ERROR",
                "run_id": run_id,
                "message": str(e),
            },
            indent=2,
        )


def _read_tables(run_id: str) -> str:
    try:
        schema = get_schema(run_id=run_id)

        tables = []
        for name, meta in schema.tables.items():
            tables.append(
                {
                    "id": f"resource://talk2metadata/run/{run_id}/table/{name}",
                    "name": name,
                    "columns": list(meta.columns.keys()),
                    "column_count": len(meta.columns),
                    "row_count": meta.row_count,
                    "primary_key": meta.primary_key,
                    "is_target": name == schema.target_table,
                }
            )
        tables.sort(key=lambda x: x["name"])
        return json.dumps(
            {
                "run_id": run_id,
                "table_count": len(tables),
                "target_table": schema.target_table,
                "tables": tables,
            },
            indent=2,
        )
    except FileNotFoundError as e:
        return json.dumps(
            {
                "error": "Schema not found",
                "error_code": "SCHEMA_NOT_FOUND",
                "run_id": run_id,
                "message": str(e),
                "hint": "Please run 'talk2metadata ingest' to load data first.",
            },
            indent=2,
        )


def _read_run_context(run_id: str) -> str:
    try:
        schema = get_schema(run_id=run_id)
        return json.dumps(
            {
                "id": f"resource://talk2metadata/run/{run_id}/context",
                "run_id": run_id,
                "target_table": schema.target_table,
                "table_count": len(schema.tables),
                "foreign_key_count": len(schema.foreign_keys),
                "search_tip": "For higher success with mode=lexical or mode=graph, use concrete terms (author names, IDs, report numbers), name+initial (e.g. 'Standing J'), and top_k 10-20. Read resource://talk2metadata/search-modes for full guidance.",
            },
            indent=2,
        )
    except FileNotFoundError as e:
        return json.dumps(
            {
                "error": "Schema not found",
                "error_code": "SCHEMA_NOT_FOUND",
                "run_id": run_id,
                "message": str(e),
                "hint": "Please run 'talk2metadata ingest' to load data first.",
            },
            indent=2,
        )


_SEARCH_MODES_TEXT = """\
# Talk2Metadata Search Modes

The `search` tool supports multiple retrieval modes via the `mode` parameter.
Default mode is `graph`.

---

## 1. semantic  (Record-Embedding Search)

Encodes the query into a dense vector using a Sentence Transformer model and
searches pre-built FAISS indices across all tables. Results from related tables
are aggregated through a foreign-key voting mechanism: each matched record
"votes" for target-table records it links to. An optional cross-encoder
reranker refines the final ranking.

- **Best for:** conceptual / similarity-based queries, exploratory search.
- **Strengths:** understands meaning, not just keywords; leverages table
  relationships via voting; fast at query time (FAISS lookup).
- **Weaknesses:** requires pre-built vector indices; may miss exact keyword
  matches; quality depends on the embedding model.

---

## 2. lexical  (BM25 / Keyword Search)

Tokenizes the query, removes stopwords, and runs full-text search using BM25
(DuckDB FTS) when available, or falls back to LIKE-based pattern matching.
Applies phrase boost when the full query string appears in the text and numeric
boost for ID fields.

- **Best for:** exact keyword or phrase lookups, ID-based searches.
- **Strengths:** simple, fast, interpretable; excellent for known-term queries.
- **Weaknesses:** no semantic understanding; requires exact or near-exact term
  matches; less effective for free-text conceptual queries.

---

## 3. graph  (Knowledge-Graph Search)

Uses a pre-built knowledge graph (nodes = rows, edges = foreign keys). No SQL at
query time: tokenize query → find seed nodes by token index → BFS to target rows.

- **Best for:** author/entity names, report numbers, IDs, and multi-table
  traversal (e.g. "reports by author X", "reports with commodity Y").
- **Strengths:** fast, no LLM; good hit rate when query contains concrete
  terms (names, numbers, codes) that appear in the data.
- **Weaknesses:** depends on token overlap; very abstract or long narrative
  queries may not match enough seeds.

---

## 4. text2sql  (Direct Text-to-SQL)

Sends the full database schema plus the natural-language question to an LLM,
which generates a single SQL query that is then executed against the database.

- **Best for:** structured queries with specific filter conditions on known
  fields.
- **Strengths:** precise, deterministic SQL output; handles complex multi-table
  JOINs; auditable (the SQL is returned alongside results).
- **Weaknesses:** slower (LLM generation); LLM may hallucinate columns/tables;
  can hit token limits on large schemas.

---

## 5. text2sql_two_step  (Two-Step Text-to-SQL) — DEFAULT

Step 1 — **Schema Location:** an LLM identifies which tables and columns are
relevant to the query, producing a focused schema subset.
Step 2 — **SQL Generation:** a second LLM call generates SQL using only the
relevant schema, with query-complexity classification (simple / medium /
complex) to choose the right SQL pattern (e.g., WHERE EXISTS for complex
multi-table queries).

Optionally incorporates vector hints (VASQL) and dynamic few-shot examples
from past queries.

- **Best for:** general-purpose queries (recommended default).
- **Strengths:** reduced hallucination (focused schema); adaptive complexity
  strategies; few-shot learning from query history.
- **Weaknesses:** slower than direct text2sql (extra LLM round-trip); schema
  identification can still be imperfect.

---

## 6. hybrid  (Neuro-Symbolic Search)

Combines vector search with SQL generation in a cascade:

1. **Strict SQL first** — runs text2sql; if it returns results, done.
2. **Soft filtering** — runs vector search to get candidate IDs, then
   generates a scoring SQL where each criterion earns points (high=3,
   medium=2, low=1). Records with score >= 70% of max are returned;
   relaxes to 50% if needed.
3. **Pure vector fallback** — returns raw vector results if SQL paths fail.

- **Best for:** exploratory queries where recall matters; mixed
  structured + semantic queries.
- **Strengths:** best recall; graceful degradation; adapts to query type.
- **Weaknesses:** slowest mode (multiple search passes); most complex
  configuration.

---

## Quick Comparison

| Mode              | Speed | Precision | Recall | Index Required       |
|-------------------|-------|-----------|--------|----------------------|
| semantic          | Fast  | Medium    | High   | FAISS vectors        |
| lexical           | Fast  | High      | Medium | FTS / text column    |
| graph             | Fast  | High      | High   | Knowledge graph      |
| text2sql          | Medium| High      | Medium | Schema + LLM         |
| text2sql_two_step | Slow  | High      | High   | Schema + LLM         |
| hybrid            | Slow  | High      | Very High | FAISS + Schema + LLM |

---

## How to get higher success with lexical and graph

Use these tips so the model returns more relevant results when `mode` is `lexical` or `graph`.

### For both modes

- **Include concrete terms** that appear in the data: author names (e.g. "Standing J", "CROFT A"), report numbers or ANumber, commodity names, project names, IDs (e.g. author ID 4823).
- **Prefer short, term-focused questions** rather than long narrative. Example: "WAMEX reports authored by Stewart L J" is better than a long paragraph.
- **Use a larger top_k** (e.g. 10–20) when you want more candidates; hit rate improves with higher top_k.

### Lexical

- **Exact or near-exact wording** works best; synonyms may be missed.
- **Phrases and IDs** (e.g. "author 3438", "gold, iron, uranium") are strong; the engine boosts phrase and numeric matches.
- If results are weak, try rephrasing with terms from the schema (e.g. column names or sample values from the run schema).

### Graph

- **Author / entity names**: include both family name and initial (e.g. "Graindorge J M", "Penna P E") so token match and graph seeds work well.
- **IDs and codes**: include numeric IDs or commodity codes (e.g. "author ID 4823", "commodity 49"); they are weighted in seed scoring.
- **Multi-table questions** (e.g. reports by author, reports with a given commodity) suit graph well; the graph traverses foreign keys to the target table.

### Example calls for higher success

```json
{ "query": "WAMEX reports authored by Standing J", "mode": "graph", "top_k": 15 }
{ "query": "report numbers for author ID 3438", "mode": "lexical", "top_k": 10 }
{ "query": "reports targeting gold and base metals", "mode": "graph", "top_k": 20 }
```

---

## Usage in MCP search tool

```json
{ "query": "gold deposits in Kalgoorlie", "mode": "hybrid" }
{ "query": "report A12345", "mode": "text2sql_two_step" }
{ "query": "reports by author Standing J", "mode": "graph", "top_k": 15 }
```

If `mode` is not specified, the default is `graph`.
"""


def _read_search_modes() -> str:
    return _SEARCH_MODES_TEXT


def _read_resource(uri_str: str) -> str:
    if uri_str == "resource://talk2metadata/search-modes":
        return _read_search_modes()

    if uri_str == "resource://talk2metadata/runs":
        return _read_runs()

    run_details_match = _RUN_DETAILS_RE.match(uri_str)
    if run_details_match:
        return _read_run_details(run_details_match.group(1))

    run_context_match = _RUN_CONTEXT_RE.match(uri_str)
    if run_context_match:
        return _read_run_context(run_context_match.group(1))

    run_tables_match = _RUN_TABLES_RE.match(uri_str)
    if run_tables_match:
        return _read_tables(run_tables_match.group(1))

    run_schema_match = _RUN_SCHEMA_RE.match(uri_str)
    if run_schema_match:
        return _read_schema(run_schema_match.group(1))

    run_table_match = _RUN_TABLE_RE.match(uri_str)
    if run_table_match:
        return _read_table_by_name(run_table_match.group(2), run_table_match.group(1))

    return json.dumps(
        {
            "error": "Unknown resource",
            "error_code": "UNKNOWN_RESOURCE",
            "uri": uri_str,
            "message": f"Resource URI '{uri_str}' is not recognized.",
        },
        indent=2,
    )


def register_resources(server: Server) -> None:
    @server.list_resources()
    async def list_resources() -> list[Resource]:
        return [
            Resource(
                uri="resource://talk2metadata/runs",
                name="Runs",
                description="List available dataset runs and run-scoped resources.",
                mimeType="application/json",
            ),
            Resource(
                uri="resource://talk2metadata/search-modes",
                name="Search Modes",
                description="Explains search modes (semantic, lexical, graph, text2sql, text2sql_two_step, hybrid) and how to get higher success with lexical and graph.",
                mimeType="text/markdown",
            ),
        ]

    @server.list_resource_templates()
    async def list_resource_templates() -> list[ResourceTemplate]:
        return [
            ResourceTemplate(
                uriTemplate="resource://talk2metadata/run/{run_id}",
                name="Run Details",
                description="Get details for a specific run_id dataset.",
                mimeType="application/json",
            ),
            ResourceTemplate(
                uriTemplate="resource://talk2metadata/run/{run_id}/context",
                name="Run Context",
                description="Get run-scoped context, including schema summary for a dataset run.",
                mimeType="application/json",
            ),
            ResourceTemplate(
                uriTemplate="resource://talk2metadata/run/{run_id}/tables",
                name="Run Tables",
                description="List tables for a specific run_id dataset.",
                mimeType="application/json",
            ),
            ResourceTemplate(
                uriTemplate="resource://talk2metadata/run/{run_id}/schema",
                name="Run Schema",
                description="Get complete schema metadata for a specific run_id dataset.",
                mimeType="application/json",
            ),
            ResourceTemplate(
                uriTemplate="resource://talk2metadata/run/{run_id}/table/{table_name}",
                name="Run Table Information",
                description="Get detailed table information for a specific run_id dataset.",
                mimeType="application/json",
            ),
        ]

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        return _read_resource(str(uri))
