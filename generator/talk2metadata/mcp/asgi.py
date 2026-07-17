"""ASGI app factory for running Talk2Metadata MCP under Uvicorn workers."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from talk2metadata.mcp.config import MCPConfig
from talk2metadata.mcp.server import build_server
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


def create_app():
    config_path = os.getenv("MCP_CONFIG_PATH")
    config_path_obj = Path(config_path) if config_path else None
    config = MCPConfig.load(config_path_obj)

    app, http_transport, mcp_server = build_server(config)

    @asynccontextmanager
    async def lifespan(_app):
        async with http_transport.connect() as (read_stream, write_stream):

            async def run_mcp_server():
                try:
                    await mcp_server.run(
                        read_stream,
                        write_stream,
                        mcp_server.create_initialization_options(),
                    )
                except Exception as e:
                    logger.error(f"MCP server error: {e}")

            task = asyncio.create_task(run_mcp_server())

            yield

            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app.router.lifespan_context = lifespan
    return app
