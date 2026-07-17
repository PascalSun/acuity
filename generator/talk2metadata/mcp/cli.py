"""Command line interface for running Talk2Metadata MCP server."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import click
import uvicorn

from talk2metadata.mcp.config import MCPConfig
from talk2metadata.mcp.server import build_server
from talk2metadata.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def _is_broken_pipe(exc: BaseException) -> bool:
    """Check if exception is a BrokenPipeError (including nested exceptions)."""
    if isinstance(exc, BrokenPipeError):
        return True
    if nested := getattr(exc, "exceptions", None):
        return any(_is_broken_pipe(child) for child in nested)
    return False


def _setup_logging(log_level: str = "info") -> None:
    """Configure logging for MCP server."""
    setup_logging(level=log_level.upper())


@click.group()
def main() -> None:
    """Run Talk2Metadata as an MCP provider."""


@main.command()
@click.option(
    "--host", default=None, help="Bind address (default: from config or 0.0.0.0)"
)
@click.option(
    "--port", default=None, type=int, help="Port to bind (default: from config or 8010)"
)
@click.option("--log-level", default="info", help="Uvicorn log level")
@click.option(
    "--config", default=None, help="Path to config.mcp.yml (default: ./config.mcp.yml)"
)
@click.option(
    "--workers",
    default=4,
    type=int,
    show_default=True,
    help="Number of worker processes",
)
def sse(
    host: str | None,
    port: int | None,
    log_level: str,
    config: str | None,
    workers: int,
) -> None:
    """Start MCP server with StreamableHTTP transport.

    Configuration priority: CLI arguments > environment variables > config file > defaults
    """
    _setup_logging(log_level)

    # Load configuration
    config_path = Path(config) if config else None
    mcp_config = MCPConfig.load(config_path)

    # Override with CLI arguments
    if host:
        mcp_config.server.host = host
    if port:
        mcp_config.server.port = port

    # Build server
    app, http_transport, mcp_server = build_server(mcp_config)

    # Create lifespan context for transport initialization
    @asynccontextmanager
    async def lifespan(_app):
        """Initialize StreamableHTTP transport and run MCP server."""
        async with http_transport.connect() as (read_stream, write_stream):
            # Start MCP server in background
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

            # Cleanup
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app.router.lifespan_context = lifespan

    # Display configuration (concise)
    click.echo(
        f"MCP server: {mcp_config.server.base_url} — StreamableHTTP (MCP 2025-03-26)"
    )
    click.echo("Endpoints: /mcp (OAuth) · /health")
    click.echo(
        f"OAuth: discovery={mcp_config.oauth.discovery_url} · validation={'Introspection' if mcp_config.oauth.use_introspection else 'JWT'} · client_id={mcp_config.oauth.client_id}"
    )
    click.echo(
        "Connect via MCP Inspector with OAuth 2.0; authorize and use Bearer token."
    )
    click.echo("")

    # Run server
    if workers <= 1:
        uvicorn.run(
            app,
            host=mcp_config.server.host,
            port=mcp_config.server.port,
            log_level=log_level,
            access_log=True,
        )
        return

    if config_path:
        os.environ["MCP_CONFIG_PATH"] = str(config_path)
    os.environ["MCP_HOST"] = str(mcp_config.server.host)
    os.environ["MCP_PORT"] = str(mcp_config.server.port)
    os.environ["MCP_BASE_URL"] = str(mcp_config.server.base_url)

    uvicorn.run(
        "talk2metadata.mcp.asgi:create_app",
        factory=True,
        workers=workers,
        host=mcp_config.server.host,
        port=mcp_config.server.port,
        log_level=log_level,
        access_log=True,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
