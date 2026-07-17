"""MCP tools registration and integration."""

from __future__ import annotations

import json
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

from talk2metadata.metrics.runtime import get_metrics_collector, log_request_metrics
from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.timing import TimingContext

from . import get_schema, get_table_info, list_tables, search

logger = get_logger(__name__)

# Collect all tool modules
TOOL_MODULES = [
    search,
    list_tables,
    get_schema,
    get_table_info,
]

# Collect tool specifications and handlers
TOOL_SPECS = [module.TOOL_SPEC for module in TOOL_MODULES]
TOOL_HANDLERS = {
    module.TOOL_SPEC["name"]: getattr(module, f"handle_{module.TOOL_SPEC['name']}")
    for module in TOOL_MODULES
}


def register_tools(server: Server) -> None:
    """Register all MCP tools with the server."""

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List all available tools."""
        return [Tool(**spec) for spec in TOOL_SPECS]

    @server.call_tool()
    async def call_tool(name: str, arguments: Any) -> list[TextContent]:
        """Handle tool invocation by routing to the appropriate handler."""
        import time
        import uuid

        request_id = str(uuid.uuid4())[:8]
        logger.debug(f"Tool called: {name} (request_id={request_id})")

        # Track request
        metrics = get_metrics_collector()
        metrics.increment_requests(tool_name=name)

        start_time = time.perf_counter()
        success = False

        try:
            handler = TOOL_HANDLERS.get(name)
            if not handler:
                return [
                    TextContent(
                        type="text", text=json.dumps({"error": f"Unknown tool: {name}"})
                    )
                ]

            with TimingContext(f"tool.{name}"):
                result = await handler(arguments or {})

            success = True
            return result

        except Exception as e:  # pragma: no cover - defensive logging
            metrics.increment_errors()
            logger.error(f"Error executing tool {name}: {e}", exc_info=True)
            return [
                TextContent(
                    type="text", text=json.dumps({"error": str(e), "tool": name})
                )
            ]
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            log_request_metrics(
                request_id=request_id,
                tool_name=name,
                duration_ms=duration_ms,
                success=success,
                details={"arguments": arguments} if arguments else {},
            )
