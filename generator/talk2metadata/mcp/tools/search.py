"""Search tool for finding relevant records."""

from __future__ import annotations

import json
from typing import Any

from mcp.types import TextContent

from talk2metadata.core.solution.preprocess import run_preprocess
from talk2metadata.metrics.runtime import log_slow_query
from talk2metadata.utils.config import get_config
from talk2metadata.utils.json_utils import json_safe
from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.timing import TimingContext

from ..common.retriever import get_retriever
from ..config import MCPConfig

logger = get_logger(__name__)


_WAMEX_RUN_ID = "wamex"
_WAMEX_S3_BUCKET = "wamex"
_WAMEX_S3_PREFIX_TEMPLATE = "reports/{anumber}/"
_WAMEX_S3_PRESIGN_EXPIRES_SECONDS = 3600


def _extract_anumber(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return str(int(value))
    if isinstance(value, str):
        v = value.strip()
        return v or None
    return str(value).strip() or None


def _extract_anumber_from_row(row: dict[str, Any]) -> str | None:
    for k, v in row.items():
        if str(k).lower() == "anumber":
            return _extract_anumber(v)
    return None


def _list_and_presign_wamex_pdfs(
    anumber: str,
    *,
    bucket: str = _WAMEX_S3_BUCKET,
    expires_in: int = _WAMEX_S3_PRESIGN_EXPIRES_SECONDS,
) -> list[dict[str, str]]:
    try:
        import boto3
    except Exception as e:
        logger.warning(f"boto3 not available, skipping S3 presign: {e}")
        return []

    prefix = _WAMEX_S3_PREFIX_TEMPLATE.format(anumber=anumber)
    mcp_config = MCPConfig.load()
    aws = mcp_config.aws
    try:
        client_kwargs: dict[str, Any] = {}
        if aws.region:
            client_kwargs["region_name"] = aws.region
        if aws.endpoint_url:
            client_kwargs["endpoint_url"] = aws.endpoint_url
        if aws.access_key_id and aws.secret_access_key:
            client_kwargs["aws_access_key_id"] = aws.access_key_id
            client_kwargs["aws_secret_access_key"] = aws.secret_access_key
            if aws.session_token:
                client_kwargs["aws_session_token"] = aws.session_token

        s3 = boto3.client("s3", **client_kwargs)
        paginator = s3.get_paginator("list_objects_v2")
        pdf_keys: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                key = obj.get("Key")
                if not key:
                    continue
                if str(key).lower().endswith(".pdf"):
                    pdf_keys.append(str(key))

        results: list[dict[str, str]] = []
        for key in sorted(set(pdf_keys)):
            try:
                url = s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": bucket, "Key": key},
                    ExpiresIn=expires_in,
                )
                results.append(
                    {"key": key, "filename": key.rsplit("/", 1)[-1], "url": url}
                )
            except Exception as e:
                logger.warning(
                    f"Failed to presign S3 object: bucket={bucket}, key={key}, err={e}"
                )
                continue

        return results
    except Exception as e:
        logger.warning(
            f"Failed to list/presign S3 PDFs: bucket={bucket}, prefix={prefix}, err={e}"
        )
        return []


async def handle_search(args: dict[str, Any]) -> list[TextContent]:
    """Search for relevant records using natural language query.

    Args:
        args: Dictionary with 'query' and optional 'top_k', 'mode' keys

    Returns:
        List of TextContent with search results
    """
    import time

    config = get_config()
    run_id = args.get("run_id") or config.get("run_id") or "wamex"
    query = args.get("query", "")
    top_k = args.get("top_k", 5)
    mode = args.get("mode") or "graph"

    if not query:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": "Query parameter is required"}, indent=2),
            )
        ]

    try:
        start_time = time.perf_counter()

        if run_id:
            config.set("run_id", run_id)

        include_wamex_pdfs = str(run_id).lower() == _WAMEX_RUN_ID
        wamex_pdf_cache: dict[str, list[dict[str, str]]] = {}

        # Get retriever
        with TimingContext("retriever_init"):
            retriever = get_retriever(run_id=run_id, mode_name=str(mode))

        # Search
        with TimingContext("search_execution"):
            schema_metadata = getattr(retriever, "schema_metadata", None)
            query, _ = run_preprocess(
                query,
                config=config,
                mode_name=str(mode),
                run_id=str(run_id) if run_id is not None else None,
                schema_metadata=schema_metadata,
            )
            results = retriever.search(query, top_k=top_k)

        # Format results
        with TimingContext("result_serialization"):
            formatted = []
            for r in results:
                if hasattr(r, "sql_query"):
                    rows = r.data
                    if include_wamex_pdfs and isinstance(rows, list):
                        enriched_rows: list[dict[str, Any]] = []
                        for row in rows:
                            if not isinstance(row, dict):
                                enriched_rows.append(row)
                                continue
                            row_copy = dict(row)
                            anumber = _extract_anumber_from_row(row_copy)
                            if anumber:
                                if anumber not in wamex_pdf_cache:
                                    wamex_pdf_cache[anumber] = (
                                        _list_and_presign_wamex_pdfs(anumber)
                                    )
                                row_copy["pdfs"] = wamex_pdf_cache[anumber]
                            enriched_rows.append(row_copy)
                        rows = enriched_rows

                    formatted.append(
                        {
                            "id": (
                                f"{run_id}:{mode}:{r.rank}"
                                if run_id
                                else f"{mode}:{r.rank}"
                            ),
                            "rank": r.rank,
                            "table": r.table,
                            "score": getattr(r, "score", 1.0),
                            "row_count": getattr(r, "row_count", None),
                            "sql_query": r.sql_query,
                            "data": rows,
                        }
                    )
                else:
                    data = r.data
                    if include_wamex_pdfs and isinstance(data, dict):
                        data_copy = dict(data)
                        anumber = _extract_anumber_from_row(data_copy)
                        if not anumber and str(r.table).lower() == "wamex_reports":
                            anumber = _extract_anumber(getattr(r, "row_id", None))
                        if anumber:
                            if anumber not in wamex_pdf_cache:
                                wamex_pdf_cache[anumber] = _list_and_presign_wamex_pdfs(
                                    anumber
                                )
                            data_copy["pdfs"] = wamex_pdf_cache[anumber]
                        data = data_copy

                    result_dict = {
                        "id": (
                            f"{run_id}:{r.table}:{r.row_id}"
                            if run_id
                            else f"{r.table}:{r.row_id}"
                        ),
                        "rank": r.rank,
                        "table": r.table,
                        "row_id": r.row_id,
                        "score": r.score,
                        "data": data,
                    }

                    if hasattr(r, "bm25_score"):
                        result_dict["bm25_score"] = r.bm25_score
                        result_dict["semantic_score"] = r.semantic_score

                    formatted.append(result_dict)

            output = {
                "run_id": run_id,
                "query": query,
                "top_k": top_k,
                "mode": mode,
                "results_count": len(formatted),
                "results": formatted,
            }

        # Check for slow query
        total_duration = (time.perf_counter() - start_time) * 1000
        slow_query_threshold = 100.0  # milliseconds
        if total_duration > slow_query_threshold:
            log_slow_query(
                query=query,
                duration_ms=total_duration,
                threshold_ms=slow_query_threshold,
                details={
                    "top_k": top_k,
                    "mode": mode,
                    "results_count": len(formatted),
                },
            )

        return [TextContent(type="text", text=json.dumps(json_safe(output), indent=2))]

    except FileNotFoundError as e:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": "Index not found",
                        "message": str(e),
                        "hint": "Please run 'talk2metadata search prepare' to prepare modes first.",
                    },
                    indent=2,
                ),
            )
        ]
    except Exception as e:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {"error": "Search failed", "message": str(e)}, indent=2
                ),
            )
        ]


TOOL_SPEC = {
    "name": "search",
    "description": (
        "Search for relevant records across all tables using natural language queries. "
        "Supports modes: graph (default), lexical, semantic, text2sql, text2sql_two_step, hybrid. "
        "For higher success with lexical or graph: use concrete terms (author names, IDs, report numbers), include name+initial (e.g. 'Standing J'), and top_k 10–20. See resource resource://talk2metadata/search-modes for details. "
        "For run_id='wamex', results may include data.pdfs with presigned URLs for report PDFs."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "Run ID used to locate which dataset to query",
            },
            "query": {
                "type": "string",
                "description": "Natural language search query",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default: 5)",
                "default": 5,
            },
            "mode": {
                "type": "string",
                "description": "Mode: graph (default), lexical, semantic, text2sql.two_step, hybrid, etc. Use lexical/graph with concrete terms and top_k 10–20 for higher success. See resource://talk2metadata/search-modes.",
            },
        },
        "required": ["query"],
    },
}
