"""Help prompt for Talk2Metadata MCP server."""

from __future__ import annotations

from mcp.types import GetPromptResult, PromptMessage, TextContent

from talk2metadata.mcp.common.schema_index import get_schema
from talk2metadata.utils.config import get_config


async def get_help_prompt(arguments: dict[str, str]) -> GetPromptResult:
    config = get_config()
    run_id = arguments.get("run_id") or config.get("run_id")

    run_section = ""
    if run_id:
        try:
            schema = get_schema(run_id=run_id)
            table_names = sorted(schema.tables.keys())
            preview_tables = table_names[:10]
            target = schema.target_table
            other_tables = [t for t in table_names if t != target]
            table_a = target
            table_b = other_tables[0] if other_tables else target
            table_c = other_tables[1] if len(other_tables) > 1 else table_b
            run_section = (
                f"## Run Context\n"
                f"- run_id: {run_id}\n"
                f"- target_table: {schema.target_table}\n"
                f"- tables: {len(schema.tables)}\n"
                f"- foreign_keys: {len(schema.foreign_keys)}\n\n"
                f"Example tables: {', '.join(preview_tables)}"
                + (" ..." if len(table_names) > len(preview_tables) else "")
                + "\n\n"
                + "Example prompts:\n"
                + f'- "Find important records in {table_a}"\n'
                + f'- "Records related to {table_b}"\n'
                + f'- "How does {table_c} relate to {table_a}?"\n\n'
            )
        except Exception:
            run_section = f"## Run Context\n- run_id: {run_id}\n\n"

    return GetPromptResult(
        description="Guide for using Talk2Metadata MCP server",
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=(
                        "# Talk2Metadata MCP Server - Quick Start Guide\n\n"
                        + run_section
                        + "## Overview\n"
                        "Talk2Metadata provides semantic search and schema exploration "
                        "for multi-table relational data. It helps you find relevant records "
                        "using natural language queries and understand relationships between tables.\n\n"
                        "## Available Tools\n\n"
                        "### 1. search\n"
                        "Search for relevant records using natural language.\n"
                        "- `run_id`: Which dataset run to query\n"
                        "- `query`: Your search question (e.g., 'customers in healthcare')\n"
                        "- `top_k`: Number of results (default: 5)\n"
                        "- `hybrid`: Use BM25+semantic hybrid search for better results\n"
                        "- WAMEX note: when run_id='wamex', results may include `data.pdfs` which contains presigned URLs you can open to access PDF reports\n\n"
                        "### 2. list_tables\n"
                        "Get a list of all available tables with basic information.\n\n"
                        "### 3. get_schema\n"
                        "Get complete schema metadata including foreign key relationships.\n\n"
                        "### 4. get_table_info\n"
                        "Get detailed information about a specific table.\n"
                        "- `run_id`: Which dataset run to query\n"
                        "- `table_name`: Name of the table to inspect\n\n"
                        "## Available Resources\n\n"
                        "- `resource://talk2metadata/runs` - List available runs\n"
                        "- `resource://talk2metadata/run/{run_id}` - Run details\n"
                        "- `resource://talk2metadata/run/{run_id}/context` - Run summary\n"
                        "- `resource://talk2metadata/run/{run_id}/schema` - Complete schema\n"
                        "- `resource://talk2metadata/run/{run_id}/tables` - Table list\n"
                        "- `resource://talk2metadata/run/{run_id}/table/{name}` - Specific table info\n\n"
                        "## REST Endpoints (Optional)\n\n"
                        "If you are using the REST API (/docs), prefer run-scoped endpoints:\n\n"
                        "```\n"
                        "GET /api/runs\n"
                        "GET /api/run/YOUR_RUN_ID\n"
                        "GET /api/run/YOUR_RUN_ID/tables\n"
                        "GET /api/run/YOUR_RUN_ID/schema\n"
                        "GET /api/run/YOUR_RUN_ID/table/orders\n"
                        'POST /api/search  { "run_id": "YOUR_RUN_ID", "query": "...", "top_k": 5 }\n'
                        "```\n\n"
                        "## Typical Workflow\n\n"
                        "1. **Explore schema**: Use `list_tables` or `get_schema` to understand available data\n"
                        "2. **Inspect tables**: Use `get_table_info` to see columns and relationships\n"
                        "3. **Search records**: Use `search` to find relevant records with natural language\n"
                        "4. **Refine search**: Use `hybrid=true` for better search quality\n\n"
                        "## Example Queries\n\n"
                        "```\n"
                        "# Find customers in a specific industry\n"
                        'search(run_id="YOUR_RUN_ID", query="customers in healthcare industry", top_k=10)\n\n'
                        "# Find high-value orders\n"
                        'search(run_id="YOUR_RUN_ID", query="orders with high value", hybrid=true)\n\n'
                        "# Explore schema relationships\n"
                        'get_schema(run_id="YOUR_RUN_ID")\n'
                        'get_table_info(run_id="YOUR_RUN_ID", table_name="orders")\n'
                        "```\n\n"
                        "## Tips\n\n"
                        "- Use hybrid search for better results (combines keyword and semantic matching)\n"
                        "- Check foreign keys to understand table relationships\n"
                        "- Sample values help understand column content\n"
                        "- The target table is the main table for your queries\n"
                    ),
                ),
            )
        ],
    )


PROMPT_SPEC = {
    "name": "help",
    "description": "Get help and guidance on using Talk2Metadata MCP server",
}
