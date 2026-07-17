"""Search guide prompt for Talk2Metadata."""

from __future__ import annotations

from mcp.types import GetPromptResult, PromptMessage, TextContent

from talk2metadata.mcp.common.schema_index import get_schema
from talk2metadata.utils.config import get_config


async def get_search_guide_prompt(arguments: dict[str, str]) -> GetPromptResult:
    config = get_config()
    run_id = arguments.get("run_id") or config.get("run_id")
    schema_section = ""

    if run_id:
        try:
            schema = get_schema(run_id=run_id)
            table_names = sorted(schema.tables.keys())
            target = schema.target_table
            other_tables = [t for t in table_names if t != target]
            table_a = target
            table_b = other_tables[0] if other_tables else target
            table_c = other_tables[1] if len(other_tables) > 1 else table_b
            schema_section = (
                f"## Run-Aware Examples\n\n"
                f"Use the `run_id` parameter so the server loads the correct dataset.\n\n"
                f"Example tools:\n"
                f"```\n"
                f'list_tables(run_id="{run_id}")\n'
                f'get_schema(run_id="{run_id}")\n'
                f'get_table_info(run_id="{run_id}", table_name="{table_a}")\n'
                f"```\n\n"
                f"Example searches:\n"
                f"```\n"
                f'search(run_id="{run_id}", query="Find important records in {table_a}", top_k=10)\n'
                f'search(run_id="{run_id}", query="Records related to {table_b}", hybrid=true)\n'
                f'search(run_id="{run_id}", query="How does {table_c} relate to {table_a}?")\n'
                f"```\n\n"
                f"Example resources:\n"
                f"```\n"
                f"resource://talk2metadata/runs\n"
                f"resource://talk2metadata/run/{run_id}\n"
                f"resource://talk2metadata/run/{run_id}/context\n"
                f"resource://talk2metadata/run/{run_id}/tables\n"
                f"resource://talk2metadata/run/{run_id}/schema\n"
                f"resource://talk2metadata/run/{run_id}/table/{table_a}\n"
                f"```\n\n"
                f"REST equivalents (if using /docs):\n"
                f"```\n"
                f"GET /api/run/{run_id}/tables\n"
                f'POST /api/search  {{ "run_id": "{run_id}", "query": "Find important records in {table_a}", "top_k": 10 }}\n'
                f"```\n\n"
            )
        except Exception:
            schema_section = (
                "## Run-Aware Examples\n\n"
                "Use the `run_id` parameter so the server loads the correct dataset.\n\n"
                "```\n"
                'search(run_id="YOUR_RUN_ID", query="your question", top_k=10)\n'
                "```\n\n"
            )

    return GetPromptResult(
        description="Guide for effective searching with Talk2Metadata",
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=(
                        "# Effective Searching with Talk2Metadata\n\n"
                        + schema_section
                        + "## Search Modes\n\n"
                        'Use the `mode` parameter. Default is `text2sql_two_step`. For higher success with **lexical** or **graph**, read resource `resource://talk2metadata/search-modes` (section "How to get higher success with lexical and graph").\n\n'
                        "### lexical and graph — tips for higher hit rate\n"
                        "- **lexical**: Use exact or near-exact terms (author names, report numbers, IDs); include phrases and numeric IDs; try `top_k` 10–20.\n"
                        '- **graph**: Use concrete terms (author name + initial e.g. "Standing J", IDs, commodity codes); good for "reports by author X" or "reports with commodity Y"; use `top_k` 15–20.\n'
                        "- Both: Prefer short, term-focused queries; include names/numbers that appear in the data.\n\n"
                        "### Hybrid Search (Recommended for recall)\n"
                        '- Set `hybrid=true` or `mode="hybrid"`\n'
                        "- Combines SQL + vector; best recall, slower.\n\n"
                        "## Query Tips\n\n"
                        "### Good Queries\n"
                        "✓ 'customers in healthcare industry'\n"
                        "✓ 'recent high-value orders'\n"
                        "✓ 'products frequently purchased together'\n"
                        "✓ 'employees with technical skills'\n\n"
                        "### Less Effective Queries\n"
                        "✗ Single keywords: 'healthcare'\n"
                        "✗ Too vague: 'find data'\n"
                        "✗ Too specific: 'customer_id = 12345' (use direct lookup)\n\n"
                        "## Understanding Results\n\n"
                        "Each result includes:\n"
                        "- `rank`: Position in results (1 = best match)\n"
                        "- `score`: Relevance score (higher = better match)\n"
                        "- `table`: Source table name\n"
                        "- `row_id`: Unique record identifier\n"
                        "- `data`: Complete record data\n\n"
                        "WAMEX note:\n"
                        "- When run_id='wamex', rows may include `data.pdfs` (a list of presigned URLs) for accessing the report PDFs\n\n"
                        "For hybrid search:\n"
                        "- `bm25_score`: Keyword matching score\n"
                        "- `semantic_score`: Embedding similarity score\n"
                        "- Final score combines both\n\n"
                        "## Best Practices\n\n"
                        "1. **Start broad, then refine**\n"
                        "   - Initial query: 'customers'\n"
                        "   - Refined: 'enterprise customers with active subscriptions'\n\n"
                        "2. **Use domain terminology**\n"
                        "   - Queries work better with terms from your data\n"
                        "   - Check sample_values in schema to see actual terms used\n\n"
                        "3. **Adjust top_k based on needs**\n"
                        "   - top_k=5: Quick overview\n"
                        "   - top_k=20: Comprehensive results\n"
                        "   - top_k=50: Exhaustive search\n\n"
                        "4. **Leverage foreign keys**\n"
                        "   - Query mentions related data (e.g., 'orders with premium customers')\n"
                        "   - System uses FKs to enrich results\n\n"
                        "## Troubleshooting\n\n"
                        "**No results?**\n"
                        "- Try broader terms\n"
                        "- Use hybrid search\n"
                        "- Increase top_k\n"
                        "- Check available tables with list_tables\n\n"
                        "**Irrelevant results?**\n"
                        "- Make query more specific\n"
                        "- Use exact terms from sample values\n"
                        "- Reduce top_k to see only best matches\n\n"
                        "**Mixed quality results?**\n"
                        "- Enable hybrid search\n"
                        "- Results are ranked by relevance - top results are best\n"
                    ),
                ),
            )
        ],
    )


PROMPT_SPEC = {
    "name": "search_guide",
    "description": "Learn how to effectively search and query data in Talk2Metadata",
}
