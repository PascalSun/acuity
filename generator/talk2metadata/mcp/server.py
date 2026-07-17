"""MCP server implementation with StreamableHTTP transport and OAuth 2.0 authentication.

This module provides a Model Context Protocol server for Talk2Metadata with:
- StreamableHTTP transport (MCP protocol 2025-03-26)
- OAuth 2.0/OIDC authentication via Django OAuth Toolkit
- Semantic search and schema exploration tools
"""

from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import io
import json
import os
import time
import uuid
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs

import duckdb
import httpx
from mcp.server import Server
from mcp.server.streamable_http import StreamableHTTPServerTransport
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from talk2metadata import __version__
from talk2metadata.mcp.auth import oauth_proxy
from talk2metadata.mcp.auth.oidc_client import OIDCResourceServer
from talk2metadata.mcp.config import MCPConfig
from talk2metadata.metrics.runtime import (
    MetricsExporter,
    get_metrics_collector,
    log_http_request,
)
from talk2metadata.utils.json_utils import json_safe
from talk2metadata.utils.logging import get_logger

from .prompts import register_prompts
from .resources import register_resources
from .tools import register_tools

logger = get_logger(__name__)

SERVER_NAME = "Talk2Metadata MCP"
SERVER_INSTRUCTIONS = (
    "This MCP server provides search and schema exploration capabilities "
    "for multi-table relational data. Use the search tool to find relevant records "
    "using natural language queries, and schema tools to understand table relationships "
    "and foreign keys. Supports modes: graph (default), lexical, semantic, text2sql, hybrid, etc. "
    "For run_id='wamex', search results may include `pdfs` with presigned URLs to access report PDFs."
)

_LOCAL_TZINFO = datetime.now().astimezone().tzinfo


def _format_ts_utc(value: Any) -> str:
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_LOCAL_TZINFO or timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    return str(value)


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Middleware to validate OAuth tokens for protected endpoints."""

    def __init__(
        self, app, oidc_resource_server: OIDCResourceServer, protected_paths: list[str]
    ):
        super().__init__(app)
        self.oidc_resource_server = oidc_resource_server
        self.protected_paths = protected_paths

    async def dispatch(self, request: Request, call_next):
        """Validate token for protected endpoints."""
        if not any(request.url.path.startswith(p) for p in self.protected_paths):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                {
                    "error": "unauthorized",
                    "error_description": "Missing or invalid Authorization header",
                },
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="Talk2Metadata MCP"'},
            )

        token = auth_header[7:]
        token_data = await self.oidc_resource_server.verify_token(token)

        if not token_data:
            return JSONResponse(
                {
                    "error": "unauthorized",
                    "error_description": "Invalid or expired token",
                },
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="Talk2Metadata MCP"'},
            )

        request.state.token_data = token_data
        request.state.user_id = token_data.get("sub") or token_data.get("username")
        logger.debug(f"Authenticated request from user: {request.state.user_id}")

        return await call_next(request)


class RestAuthTokenVerifier:
    def __init__(
        self,
        verify_url: str | None,
        verify_ssl: bool,
        timeout: float,
        cache_ttl_seconds: float,
    ) -> None:
        self.verify_url = verify_url
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.cache_ttl_seconds = cache_ttl_seconds
        self._token_cache: dict[str, float] = {}

    async def verify(self, token: str) -> bool:
        if not self.verify_url:
            return False

        now = time.monotonic()
        cached_until = self._token_cache.get(token)
        if cached_until is not None and cached_until > now:
            return True

        async with httpx.AsyncClient(verify=self.verify_ssl, timeout=self.timeout) as c:
            r = await c.post(
                self.verify_url, headers={"Authorization": f"Token {token}"}
            )
            if r.status_code == 405:
                r = await c.get(
                    self.verify_url, headers={"Authorization": f"Token {token}"}
                )
            if not (200 <= r.status_code < 300):
                r = await c.post(self.verify_url, json={"token": token})

        if 200 <= r.status_code < 300:
            self._token_cache[token] = now + self.cache_ttl_seconds
            return True

        return False


class RestAuthMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        token_verifier: RestAuthTokenVerifier,
        protected_paths: list[str],
        static_token: str | None = None,
    ):
        super().__init__(app)
        self.token_verifier = token_verifier
        self.protected_paths = protected_paths
        self.static_token = static_token

    async def dispatch(self, request: Request, call_next):
        if not any(request.url.path.startswith(p) for p in self.protected_paths):
            return await call_next(request)

        if not self.token_verifier.verify_url and not self.static_token:
            return JSONResponse(
                {
                    "error": "server_error",
                    "error_description": "REST auth is not configured",
                },
                status_code=500,
            )

        auth_header = request.headers.get("Authorization", "")
        token: str | None = None
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
        elif auth_header.startswith("Token "):
            token = auth_header[6:]
        if not token:
            return JSONResponse(
                {
                    "error": "unauthorized",
                    "error_description": "Missing or invalid Authorization header",
                },
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="Talk2Metadata REST"'},
            )

        if self.static_token and hmac.compare_digest(token, self.static_token):
            request.state.user_id = None
            return await call_next(request)

        if not self.token_verifier.verify_url:
            return JSONResponse(
                {
                    "error": "unauthorized",
                    "error_description": "Invalid or expired token",
                },
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="Talk2Metadata REST"'},
            )

        ok = await self.token_verifier.verify(token)
        if not ok:
            return JSONResponse(
                {
                    "error": "unauthorized",
                    "error_description": "Invalid or expired token",
                },
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="Talk2Metadata REST"'},
            )

        request.state.user_id = None
        return await call_next(request)


def _collect_query_params(request: Request) -> dict[str, Any]:
    qp: dict[str, Any] = {}
    for k, v in request.query_params.multi_items():
        if k in qp:
            if isinstance(qp[k], list):
                qp[k].append(v)
            else:
                qp[k] = [qp[k], v]
        else:
            qp[k] = v
    return qp


def _extract_mcp_call(payload: Any) -> tuple[str | None, dict[str, Any] | None]:
    if not isinstance(payload, dict):
        return None, None
    params = payload.get("params")
    if not isinstance(params, dict):
        return None, None
    tool_name = params.get("name")
    arguments = params.get("arguments")
    if isinstance(tool_name, str) and isinstance(arguments, dict):
        return tool_name, arguments
    return None, None


def _normalize_response_to_json(raw: str) -> str:
    """Normalize a captured response body to valid JSON when possible.

    - If the string is valid JSON, re-serialize it so we store canonical JSON.
    - If it has a "content" field (at any level) with text items containing JSON strings, parse them.
    - If it looks like SSE (lines "data: {...}"), extract JSON from each event
      and return a JSON array.
    - Otherwise return the raw string.
    """
    if not raw or not isinstance(raw, str):
        return raw or ""
    stripped = raw.strip()
    if not stripped:
        return raw

    def _normalize_content_field(obj: Any) -> Any:
        """Recursively normalize content fields in nested structures."""
        if isinstance(obj, dict):
            # Check if this dict has a "content" field
            if "content" in obj:
                content = obj.get("content")
                if isinstance(content, list):
                    normalized_content = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_value = item.get("text")
                            if isinstance(text_value, str) and text_value.strip():
                                # Try to parse the text field as JSON
                                try:
                                    parsed_text = json.loads(text_value)
                                    # Recursively normalize the parsed JSON in case it has nested content
                                    parsed_text = _normalize_content_field(parsed_text)
                                    # Replace the string with parsed JSON object
                                    normalized_item = dict(item)
                                    normalized_item["text"] = parsed_text
                                    normalized_content.append(normalized_item)
                                except (json.JSONDecodeError, TypeError):
                                    # If parsing fails, keep original text
                                    normalized_content.append(item)
                            else:
                                normalized_content.append(item)
                        else:
                            # Recursively process nested items
                            normalized_content.append(_normalize_content_field(item))
                    obj = dict(obj)
                    obj["content"] = normalized_content
                else:
                    # Recursively process the content value if it's not a list
                    obj = dict(obj)
                    obj["content"] = _normalize_content_field(content)
            else:
                # Recursively process all values in the dict
                obj = {k: _normalize_content_field(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Recursively process all items in the list
            obj = [_normalize_content_field(item) for item in obj]
        return obj

    # Try as single JSON
    try:
        parsed = json.loads(stripped)
        # Recursively normalize any content fields
        parsed = _normalize_content_field(parsed)
        return json.dumps(json_safe(parsed), ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        pass
    # Try SSE: event stream with "data: " lines
    try:
        collected: list[Any] = []
        for line in stripped.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    collected.append(json.loads(payload))
        if collected:
            # Normalize any nested content fields inside SSE events as well
            collected = _normalize_content_field(collected)
            return json.dumps(json_safe(collected), ensure_ascii=False)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return raw


class UsageLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        if not (
            request.url.path.startswith("/api") or request.url.path.startswith("/mcp")
        ):
            return await call_next(request)

        request_id = str(uuid.uuid4())[:8]
        request.state.t2m_request_id = request_id
        start = time.perf_counter()

        route_obj = request.scope.get("route")
        route = getattr(route_obj, "path", None) or request.url.path

        query_params = _collect_query_params(request)
        path_params = dict(getattr(request, "path_params", {}) or {})

        body_obj: Any | None = None
        content_type = (request.headers.get("content-type") or "").lower()
        if request.method in {"POST", "PUT", "PATCH"}:
            try:
                body_bytes = await request.body()
            except Exception:
                body_bytes = b""
            if body_bytes and len(body_bytes) <= 64 * 1024:
                if "application/json" in content_type:
                    try:
                        body_obj = json.loads(
                            body_bytes.decode("utf-8", errors="ignore")
                        )
                    except Exception:
                        body_obj = None
                elif "application/x-www-form-urlencoded" in content_type:
                    try:
                        body_obj = {
                            k: (v[0] if len(v) == 1 else v)
                            for k, v in parse_qs(
                                body_bytes.decode("utf-8", errors="ignore")
                            ).items()
                        }
                    except Exception:
                        body_obj = None

        run_id = None
        if isinstance(path_params.get("run_id"), str):
            run_id = path_params.get("run_id")
        elif isinstance(query_params.get("run_id"), str):
            run_id = query_params.get("run_id")
        elif isinstance(body_obj, dict) and isinstance(body_obj.get("run_id"), str):
            run_id = body_obj.get("run_id")

        query_text = None
        if isinstance(query_params.get("query"), str):
            query_text = query_params.get("query")
        elif isinstance(body_obj, dict) and isinstance(body_obj.get("query"), str):
            query_text = body_obj.get("query")
        elif request.url.path.startswith("/mcp") and isinstance(body_obj, dict):
            tool_name, arguments = _extract_mcp_call(body_obj)
            if isinstance(arguments, dict):
                if isinstance(arguments.get("run_id"), str):
                    run_id = arguments.get("run_id")
                if isinstance(arguments.get("query"), str):
                    query_text = arguments.get("query")

        params: dict[str, Any] = {"query": query_params, "path": path_params}
        if isinstance(body_obj, dict):
            params["body"] = body_obj

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            log_http_request(
                request_id=request_id,
                route=str(route),
                path=request.url.path,
                method=request.method,
                status_code=500,
                duration_ms=duration_ms,
                success=False,
                run_id=run_id,
                query_text=query_text,
                params=params,
                response_json=None,
            )
            raise

        mcp_status_code = getattr(request.state, "t2m_status_code", None)
        status_code = (
            int(mcp_status_code)
            if isinstance(mcp_status_code, int)
            else int(getattr(response, "status_code", 200))
        )
        success = status_code < 400

        state_response_json = getattr(request.state, "t2m_response_json", None)
        limit = 200_000

        logged = False

        def _log_once(response_json: str | None) -> None:
            nonlocal logged
            if logged:
                return
            logged = True
            duration_ms = (time.perf_counter() - start) * 1000
            log_http_request(
                request_id=request_id,
                route=str(route),
                path=request.url.path,
                method=request.method,
                status_code=status_code,
                duration_ms=duration_ms,
                success=success,
                run_id=run_id,
                query_text=query_text,
                params=params,
                response_json=response_json,
            )

        if isinstance(state_response_json, str) and state_response_json:
            _log_once(_normalize_response_to_json(state_response_json))
            return response

        body_bytes = getattr(response, "body", None)
        if isinstance(body_bytes, (bytes, bytearray)) and body_bytes:
            captured = body_bytes[:limit]
            try:
                _log_once(
                    _normalize_response_to_json(
                        captured.decode("utf-8", errors="replace")
                    )
                )
            except Exception:
                _log_once(None)
            return response

        body_iterator = getattr(response, "body_iterator", None)
        if body_iterator is None:
            _log_once(None)
            return response

        captured_stream = bytearray()

        async def _iter_and_log():
            try:
                async for chunk in body_iterator:
                    if (
                        isinstance(chunk, (bytes, bytearray))
                        and chunk
                        and len(captured_stream) < limit
                    ):
                        remaining = limit - len(captured_stream)
                        captured_stream.extend(chunk[:remaining])
                    yield chunk
            finally:
                if captured_stream:
                    try:
                        _log_once(
                            _normalize_response_to_json(
                                captured_stream.decode("utf-8", errors="replace")
                            )
                        )
                    except Exception:
                        _log_once(None)
                else:
                    _log_once(None)

        response.body_iterator = _iter_and_log()
        return response


def _console_password(config: MCPConfig) -> str | None:
    env_password = os.getenv("TALK2METADATA_CONSOLE_PASSWORD")
    if env_password:
        return env_password
    dev_password = getattr(config.dev, "console_password", None)
    if dev_password:
        return dev_password
    if config.rest_auth.token:
        return config.rest_auth.token
    return None


def _console_signing_key(config: MCPConfig) -> bytes:
    env_secret = os.getenv("TALK2METADATA_CONSOLE_SECRET")
    if env_secret:
        return env_secret.encode("utf-8")
    dev_secret = getattr(config.dev, "console_secret", None)
    if dev_secret:
        return dev_secret.encode("utf-8")
    if config.oauth.client_secret:
        return config.oauth.client_secret.encode("utf-8")
    return b"talk2metadata"


def _sign_console_cookie(payload_b64: str, signing_key: bytes) -> str:
    sig = hmac.new(signing_key, payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{sig}"


def _verify_console_cookie(
    cookie_value: str, signing_key: bytes
) -> dict[str, Any] | None:
    if "." not in cookie_value:
        return None
    payload_b64, sig = cookie_value.rsplit(".", 1)
    expected = hmac.new(
        signing_key, payload_b64.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        payload_json = base64.urlsafe_b64decode(payload_b64 + "===").decode("utf-8")
        payload = json.loads(payload_json)
    except Exception:
        return None
    iat = payload.get("iat")
    if not isinstance(iat, int):
        return None
    if time.time() - iat > 12 * 60 * 60:
        return None
    return payload


class ConsoleAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, config: MCPConfig):
        super().__init__(app)
        self.config = config

    async def dispatch(self, request: Request, call_next):
        if not request.url.path.startswith("/console"):
            return await call_next(request)

        if request.url.path in {"/console/login", "/console/login/"}:
            return await call_next(request)

        password = _console_password(self.config)
        if not password:
            return Response(
                content="Console auth is not configured",
                status_code=503,
                media_type="text/plain",
            )

        cookie_value = request.cookies.get("t2m_console")
        if cookie_value:
            payload = _verify_console_cookie(
                cookie_value, _console_signing_key(self.config)
            )
            if payload:
                request.state.console_user = payload.get("user")
                return await call_next(request)

        if request.url.path.startswith("/console/api"):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        return Response(status_code=303, headers={"Location": "/console/login"})


def _usage_db_path() -> Path:
    return Path("./data/logs/usage.duckdb")


def _query_usage_summary() -> dict[str, Any]:
    db_path = _usage_db_path()
    if not db_path.exists():
        return {
            "has_data": False,
            "total": {"requests": 0, "errors": 0},
            "top_tools": [],
            "requests_by_day": [],
            "top_endpoints": [],
            "run_ids": [],
            "top_queries": [],
            "recent_queries": [],
        }

    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS request_metrics (
              ts TIMESTAMP,
              request_id VARCHAR,
              tool_name VARCHAR,
              duration_ms DOUBLE,
              success BOOLEAN,
              details_json VARCHAR
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS http_requests (
              ts TIMESTAMP,
              request_id VARCHAR,
              route VARCHAR,
              path VARCHAR,
              method VARCHAR,
              status_code INTEGER,
              duration_ms DOUBLE,
              success BOOLEAN,
              run_id VARCHAR,
              query_text VARCHAR,
              params_json VARCHAR,
              response_json VARCHAR
            )
            """
        )
        try:
            con.execute(
                "ALTER TABLE http_requests ADD COLUMN IF NOT EXISTS response_json VARCHAR"
            )
        except Exception:
            pass
        total = con.execute(
            """
            SELECT
              COUNT(*) AS requests,
              SUM(CASE WHEN success THEN 0 ELSE 1 END) AS errors,
              AVG(duration_ms) AS avg_ms,
              quantile_cont(duration_ms, 0.5) AS p50_ms,
              quantile_cont(duration_ms, 0.95) AS p95_ms
            FROM http_requests
            """
        ).fetchone()
        top_tools = con.execute(
            """
            SELECT
              tool_name,
              COUNT(*) AS requests,
              SUM(CASE WHEN success THEN 0 ELSE 1 END) AS errors,
              AVG(duration_ms) AS avg_ms,
              quantile_cont(duration_ms, 0.5) AS p50_ms,
              quantile_cont(duration_ms, 0.95) AS p95_ms
            FROM request_metrics
            GROUP BY tool_name
            ORDER BY requests DESC
            LIMIT 50
            """
        ).fetchall()
        by_day = con.execute(
            """
            SELECT
              date_trunc('day', ts) AS day,
              COUNT(*) AS requests,
              SUM(CASE WHEN success THEN 0 ELSE 1 END) AS errors
            FROM http_requests
            GROUP BY day
            ORDER BY day ASC
            """
        ).fetchall()
        endpoints = con.execute(
            """
            SELECT
              route,
              method,
              COUNT(*) AS requests,
              SUM(CASE WHEN success THEN 0 ELSE 1 END) AS errors,
              AVG(duration_ms) AS avg_ms,
              quantile_cont(duration_ms, 0.95) AS p95_ms
            FROM http_requests
            GROUP BY route, method
            ORDER BY requests DESC
            LIMIT 50
            """
        ).fetchall()
        run_ids = con.execute(
            """
            SELECT
              run_id,
              COUNT(*) AS requests,
              SUM(CASE WHEN success THEN 0 ELSE 1 END) AS errors
            FROM http_requests
            WHERE run_id IS NOT NULL AND run_id <> ''
            GROUP BY run_id
            ORDER BY requests DESC
            LIMIT 50
            """
        ).fetchall()
        queries = con.execute(
            """
            SELECT
              query_text,
              COUNT(*) AS requests,
              COUNT(DISTINCT run_id) AS run_ids
            FROM http_requests
            WHERE query_text IS NOT NULL AND query_text <> ''
            GROUP BY query_text
            ORDER BY requests DESC
            LIMIT 50
            """
        ).fetchall()
        recent_queries = con.execute(
            """
            SELECT
              ts,
              route,
              run_id,
              query_text,
              success,
              duration_ms
            FROM http_requests
            WHERE query_text IS NOT NULL AND query_text <> ''
            ORDER BY ts DESC
            LIMIT 50
            """
        ).fetchall()
    finally:
        con.close()

    def _row_to_float(v: Any) -> float | None:
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    return {
        "has_data": True,
        "total": {
            "requests": int(total[0] or 0),
            "errors": int(total[1] or 0),
            "avg_ms": _row_to_float(total[2]),
            "p50_ms": _row_to_float(total[3]),
            "p95_ms": _row_to_float(total[4]),
        },
        "top_tools": [
            {
                "tool_name": r[0],
                "requests": int(r[1] or 0),
                "errors": int(r[2] or 0),
                "avg_ms": _row_to_float(r[3]),
                "p50_ms": _row_to_float(r[4]),
                "p95_ms": _row_to_float(r[5]),
            }
            for r in top_tools
        ],
        "requests_by_day": [
            {
                "day": (
                    r[0].date().isoformat() if isinstance(r[0], datetime) else str(r[0])
                ),
                "requests": int(r[1] or 0),
                "errors": int(r[2] or 0),
            }
            for r in by_day
        ],
        "top_endpoints": [
            {
                "route": r[0],
                "method": r[1],
                "requests": int(r[2] or 0),
                "errors": int(r[3] or 0),
                "avg_ms": _row_to_float(r[4]),
                "p95_ms": _row_to_float(r[5]),
            }
            for r in endpoints
        ],
        "run_ids": [
            {"run_id": r[0], "requests": int(r[1] or 0), "errors": int(r[2] or 0)}
            for r in run_ids
        ],
        "top_queries": [
            {"query": r[0], "requests": int(r[1] or 0), "run_ids": int(r[2] or 0)}
            for r in queries
        ],
        "recent_queries": [
            {
                "ts": _format_ts_utc(r[0]),
                "route": r[1],
                "run_id": r[2],
                "query": r[3],
                "success": bool(r[4]),
                "duration_ms": _row_to_float(r[5]),
            }
            for r in recent_queries
        ],
        "generated_at": _format_ts_utc(datetime.now(tz=timezone.utc)),
    }


_CONSOLE_LOGIN_HTML = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Talk2Metadata - Console Login</title>
    <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
    <style>
      * { box-sizing: border-box; }
      body {
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
        margin: 0;
        color: #0f172a;
        background: #f8fafc;
      }
      .header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 14px;
        padding: 18px 20px;
        border-bottom: 1px solid #e2e8f0;
        background: #ffffff;
      }
      .header img { height: 40px; width: auto; }
      .header-title { font-weight: 800; font-size: 18px; color: #0f172a; }
      main { max-width: 520px; margin: 0 auto; padding: 28px 16px; }
      .card {
        border: 1px solid #e2e8f0;
        background: #ffffff;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 12px 30px rgba(15,23,42,0.08);
      }
      h1 { font-size: 20px; margin: 0; }
      .sub { color: #475569; margin-top: 6px; line-height: 1.55; font-size: 13px; }
      label { display:block; font-weight: 650; margin-top: 14px; color: #0f172a; }
      input {
        width: 100%;
        padding: 11px 12px;
        margin-top: 7px;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        font-size: 14px;
        color: #0f172a;
        background: #ffffff;
        outline: none;
      }
      input:focus { border-color: #60a5fa; box-shadow: 0 0 0 3px rgba(96,165,250,0.18); }
      button {
        width: 100%;
        margin-top: 14px;
        padding: 11px 14px;
        border: 0;
        border-radius: 10px;
        background: #2563eb;
        color: #ffffff;
        font-weight: 750;
        cursor: pointer;
      }
      button:hover { filter: brightness(0.98); }
      .hint { margin-top: 12px; color: #475569; line-height: 1.6; font-size: 13px; }
      .err { margin-top: 12px; color: #b91c1c; font-weight: 650; }
      code { background: #f1f5f9; padding: 2px 6px; border-radius: 8px; border: 1px solid #e2e8f0; }
    </style>
  </head>
  <body>
    <div class="header">
      <img src="/assets/logo.svg" alt="Talk2Metadata" />
      <div class="header-title">Dev Console</div>
    </div>
    <main>
      <div class="card">
        <h1>Console Login</h1>
        <div class="sub">Talk2Metadata 开发中控台（只建议内网/开发环境使用）。</div>
        <form method="post" action="/console/login">
          <label>Password</label>
          <input name="password" type="password" autocomplete="current-password" autofocus />
          <button type="submit">Sign in</button>
        </form>
        __ERROR_BLOCK__
      </div>
    </main>
  </body>
</html>
"""


_CONSOLE_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Talk2Metadata Console</title>
    <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
    <style>
      :root {
        --bg: #f8fafc; --surface: #ffffff; --text: #0f172a; --text-light: #64748b;
        --border: #e2e8f0; --primary: #3b82f6; --primary-hover: #2563eb;
        --danger: #ef4444; --success: #22c55e;
        --radius: 12px; --shadow: 0 1px 3px rgba(0,0,0,0.05);
      }
      * { box-sizing: border-box; }
      body {
        font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
        margin: 0; background: var(--bg); color: var(--text); -webkit-font-smoothing: antialiased;
      }

      /* Layout */
      .app { display: flex; flex-direction: column; min-height: 100vh; }
      .header {
        background: var(--surface); border-bottom: 1px solid var(--border);
        padding: 0 20px; height: 64px; display: flex; align-items: center; justify-content: space-between;
        position: sticky; top: 0; z-index: 50;
      }
      .brand { font-weight: 700; font-size: 18px; display: flex; align-items: center; gap: 12px; color: var(--text); text-decoration: none; }
      .brand img { height: 32px; width: auto; }
      .nav { display: flex; gap: 24px; }
      .nav a { color: var(--text-light); text-decoration: none; font-size: 14px; font-weight: 600; transition: color 0.2s; }
      .nav a:hover { color: var(--primary); }

      .main { flex: 1; width: 100%; max-width: none; margin: 0; padding: 32px 20px; }

      /* Tabs */
      .tabs { display: flex; gap: 4px; background: #e2e8f0; padding: 4px; border-radius: 10px; width: fit-content; margin-bottom: 32px; }
      .tab {
        padding: 8px 20px; border-radius: 8px; border: none; background: transparent;
        color: var(--text-light); font-weight: 600; font-size: 14px; cursor: pointer; transition: all 0.2s;
      }
      .tab:hover { color: var(--text); }
      .tab.active { background: var(--surface); color: var(--primary); box-shadow: 0 1px 2px rgba(0,0,0,0.05); }

      /* Stats */
      .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 24px; margin-bottom: 32px; }
      .card {
        background: var(--surface); padding: 24px; border-radius: var(--radius);
        border: 1px solid var(--border); box-shadow: var(--shadow);
      }
      .stat-label { font-size: 13px; font-weight: 600; color: var(--text-light); text-transform: uppercase; letter-spacing: 0.05em; }
      .stat-val { font-size: 32px; font-weight: 700; margin-top: 8px; color: var(--text); letter-spacing: -0.02em; }
      .stat-sub { font-size: 13px; color: var(--text-light); margin-top: 4px; display: flex; align-items: center; gap: 6px; }

      /* Panels */
      .panel {
        background: var(--surface); border-radius: var(--radius); border: 1px solid var(--border);
        box-shadow: var(--shadow); overflow: hidden; display: flex; flex-direction: column; margin-bottom: 24px;
      }
      .panel-head {
        padding: 16px 24px; border-bottom: 1px solid var(--border);
        font-weight: 700; font-size: 15px; display: flex; justify-content: space-between; align-items: center;
      }
      .panel-body { padding: 0; overflow-x: auto; }

      .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
      @media (max-width: 1024px) { .grid-2 { grid-template-columns: 1fr; } }

      /* Tables */
      table { width: 100%; border-collapse: collapse; font-size: 14px; }
      th { text-align: left; padding: 12px 24px; color: var(--text-light); font-weight: 600; border-bottom: 1px solid var(--border); background: #f8fafc; white-space: nowrap; font-size: 12px; text-transform: uppercase; letter-spacing: 0.03em; }
      td { padding: 12px 24px; border-bottom: 1px solid var(--border); color: var(--text); vertical-align: middle; }
      tr:last-child td { border-bottom: none; }
      tr:hover td { background: #f8fafc; }
      .num { text-align: right; font-feature-settings: "tnum"; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; }
      .badge {
        display: inline-flex; align-items: center; padding: 2px 8px; border-radius: 999px;
        font-size: 12px; font-weight: 600; background: #f1f5f9; color: var(--text-light);
      }
      .badge.success { background: #dcfce7; color: #166534; }
      .badge.error { background: #fee2e2; color: #991b1b; }
      .cell-truncate { max-width: 240px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

      /* Controls */
      .filters { display: flex; gap: 12px; padding: 16px 24px; border-bottom: 1px solid var(--border); background: #f8fafc; flex-wrap: wrap; align-items: center; }
      .input {
        padding: 8px 12px; border: 1px solid var(--border); border-radius: 8px; font-size: 14px;
        outline: none; min-width: 240px; background: white;
      }
      .input:focus { border-color: var(--primary); box-shadow: 0 0 0 3px rgba(59,130,246,0.1); }
      .btn {
        padding: 8px 16px; border-radius: 8px; font-weight: 600; font-size: 14px; cursor: pointer;
        border: 1px solid transparent; background: var(--primary); color: white; transition: opacity 0.2s;
      }
      .btn:hover { opacity: 0.9; }
      .btn.secondary { background: white; border-color: var(--border); color: var(--text); }
      .btn.secondary:hover { background: #f1f5f9; }

      /* Chart */
      .chart-container { padding: 24px; height: 300px; position: relative; }
      canvas { width: 100%; height: 100%; display: block; }

      /* Modal */
      .modal-overlay {
        position: fixed; inset: 0; background: rgba(0,0,0,0.5); z-index: 100;
        display: none; align-items: center; justify-content: center; backdrop-filter: blur(2px);
      }
      .modal {
        background: var(--surface); width: 90vw; max-width: 900px; height: 85vh;
        border-radius: 16px; display: flex; flex-direction: column; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.25);
      }
      .modal-head { padding: 20px 24px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
      .modal-title { font-weight: 700; font-size: 18px; }
      .modal-body { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 24px; }
      .code-block {
        background: #0f172a; color: #e2e8f0; padding: 16px; border-radius: 8px;
        overflow-x: auto; font-family: monospace; font-size: 13px; line-height: 1.5;
      }
      .code-pre { margin: 0; white-space: pre; }
      .json-key { color: #93c5fd; }
      .json-string { color: #a7f3d0; }
      .json-number { color: #fcd34d; }
      .json-boolean { color: #c4b5fd; }
      .json-null { color: #fca5a5; }
      .label { font-size: 12px; font-weight: 700; color: var(--text-light); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
    </style>
  </head>
  <body>
    <div class="app">
      <header class="header">
        <a href="/console" class="brand">
          <img src="/assets/logo.svg" alt="Talk2Metadata" />
          <span>Dev Console</span>
        </a>
        <nav class="nav">
          <a href="/docs" target="_blank">API Docs</a>
          <a href="/console/logout">Logout</a>
        </nav>
      </header>

      <main class="main">
        <div class="tabs">
          <button class="tab active" id="tabOverview" type="button">Overview</button>
          <button class="tab" id="tabEndpoints" type="button">Endpoints</button>
          <button class="tab" id="tabRuns" type="button">Runs</button>
          <button class="tab" id="tabLogs" type="button">Logs</button>
        </div>

        <!-- OVERVIEW -->
        <div id="viewOverview">
          <div class="stats">
            <div class="card">
              <div class="stat-label">Total Requests</div>
              <div class="stat-val" id="sTotal">-</div>
              <div class="stat-sub">Lifetime volume</div>
            </div>
            <div class="card">
              <div class="stat-label">Success Rate</div>
              <div class="stat-val" id="sSuccess">-</div>
              <div class="stat-sub" id="sErrors">-</div>
            </div>
            <div class="card">
              <div class="stat-label">Avg Latency</div>
              <div class="stat-val" id="sLatency">-</div>
              <div class="stat-sub" id="sP95">-</div>
            </div>
            <div class="card">
              <div class="stat-label">Active Run IDs</div>
              <div class="stat-val" id="sRuns">-</div>
              <div class="stat-sub">Unique sessions</div>
            </div>
          </div>

          <div class="panel">
            <div class="panel-head">Traffic Volume (Daily)</div>
            <div class="chart-container">
              <canvas id="trafficChart"></canvas>
            </div>
          </div>

          <div class="grid-2">
            <div class="panel">
              <div class="panel-head">Recent Queries</div>
              <div class="panel-body">
                <table id="tRecent">
                  <thead><tr><th>Time</th><th>Route</th><th>Run ID</th><th>Query</th><th class="num">Sec</th><th>Status</th></tr></thead>
                  <tbody></tbody>
                </table>
              </div>
            </div>
            <div class="panel">
              <div class="panel-head">Top Tools</div>
              <div class="panel-body">
                <table id="tTools">
                  <thead><tr><th>Tool</th><th class="num">Reqs</th><th class="num">Errs</th></tr></thead>
                  <tbody></tbody>
                </table>
              </div>
            </div>
          </div>

          <div class="grid-2">
            <div class="panel">
              <div class="panel-head">Top Endpoints</div>
              <div class="panel-body">
                <table id="tEndpoints">
                  <thead><tr><th>Route</th><th>Method</th><th class="num">Reqs</th><th class="num">Avg s</th></tr></thead>
                  <tbody></tbody>
                </table>
              </div>
            </div>
            <div class="panel">
              <div class="panel-head">Top Run IDs</div>
              <div class="panel-body">
                <table id="tRunIds">
                  <thead><tr><th>Run ID</th><th class="num">Reqs</th><th class="num">Errs</th></tr></thead>
                  <tbody></tbody>
                </table>
              </div>
            </div>
          </div>
        </div>

        <!-- ENDPOINTS -->
        <div id="viewEndpoints" style="display: none;">
          <div class="panel">
            <div class="panel-head">
              <span>Endpoints</span>
            </div>
            <div class="panel-body">
              <table id="tEndpoints2">
                <thead><tr><th>Route</th><th>Method</th><th class="num">Reqs</th><th class="num">Errs</th><th class="num">Avg s</th><th class="num">P95 s</th><th style="width:1px"></th></tr></thead>
                <tbody></tbody>
              </table>
            </div>
          </div>
        </div>

        <!-- RUNS -->
        <div id="viewRuns" style="display: none;">
          <div class="panel">
            <div class="panel-head">
              <span>Runs</span>
            </div>
            <div class="panel-body">
              <table id="tRuns2">
                <thead><tr><th>Run ID</th><th class="num">Reqs</th><th class="num">Errs</th><th style="width:1px"></th></tr></thead>
                <tbody></tbody>
              </table>
            </div>
          </div>
        </div>

        <!-- LOGS -->
        <div id="viewLogs" style="display: none;">
          <div class="panel">
            <div class="panel-head">
              <span>Logs</span>
              <span class="badge" id="detailCount" style="margin-left:12px">0 loaded</span>
            </div>
            <div class="filters">
              <select class="input" id="fRange" style="min-width:200px">
                <option value="all" selected>All time</option>
                <option value="24h">Last 24 hours</option>
                <option value="7d">Last 7 days</option>
                <option value="custom">Custom range</option>
              </select>
              <input class="input" id="fSince" type="datetime-local" style="min-width:220px;display:none" />
              <input class="input" id="fUntil" type="datetime-local" style="min-width:220px;display:none" />
              <input class="input" id="fRoute" placeholder="Route contains..." />
              <select class="input" id="fMethod" style="min-width:140px">
                <option value="">Any method</option>
                <option value="GET">GET</option>
                <option value="POST">POST</option>
                <option value="PUT">PUT</option>
                <option value="PATCH">PATCH</option>
                <option value="DELETE">DELETE</option>
              </select>
              <select class="input" id="fStatus" style="min-width:140px">
                <option value="">Any status</option>
                <option value="OK">OK</option>
                <option value="ERR">ERR</option>
              </select>
              <input class="input" id="fRunId" placeholder="Run ID..." />
              <select class="input" id="fRunIdMode" style="min-width:140px">
                <option value="exact" selected>Exact</option>
                <option value="prefix">Prefix</option>
              </select>
              <input class="input" id="fQuery" placeholder="Query contains..." />
              <label style="display:flex;align-items:center;gap:8px;font-size:14px;font-weight:600;color:var(--text-light);cursor:pointer">
                <input type="checkbox" id="fOnlyQuery" checked /> Only Queries
              </label>
              <div style="flex:1"></div>
              <select class="input" id="fExportFmt" style="min-width:140px">
                <option value="json" selected>Export JSON</option>
                <option value="ndjson">Export NDJSON</option>
              </select>
              <label style="display:flex;align-items:center;gap:8px;font-size:14px;font-weight:600;color:var(--text-light);cursor:pointer">
                <input type="checkbox" id="fExportParams" /> Params
              </label>
              <label style="display:flex;align-items:center;gap:8px;font-size:14px;font-weight:600;color:var(--text-light);cursor:pointer">
                <input type="checkbox" id="fExportResp" /> Response
              </label>
              <button class="btn secondary" id="btnExport" type="button">Export</button>
              <button class="btn" id="btnLoad" type="button">Search</button>
            </div>
            <div class="panel-body">
              <table id="tDetailed">
                <thead><tr><th>Time</th><th>Route</th><th>Method</th><th>Status</th><th>Run ID</th><th>Query</th><th class="num">Sec</th><th style="width:1px"></th></tr></thead>
                <tbody></tbody>
              </table>
              <div style="padding:16px;text-align:center">
                <button class="btn secondary" id="btnMore" type="button">Load More</button>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>

    <!-- MODAL -->
    <div class="modal-overlay" id="modal">
      <div class="modal">
        <div class="modal-head">
          <div class="modal-title">Request Details</div>
          <button class="btn secondary" id="btnModalClose" type="button">Close</button>
        </div>
        <div class="modal-body">
          <div>
            <div class="label">Query / Params</div>
            <div class="code-block" id="mQuery"></div>
          </div>
          <div>
            <div class="label">Response</div>
            <div class="code-block" id="mResponse"></div>
          </div>
        </div>
      </div>
    </div>

    <script src="/assets/console.js?v=__CONSOLE_JS_VERSION__" defer></script>
  </body>
</html>
"""


async def _handle_console_login(request: Request, *, config: MCPConfig) -> Response:
    password = _console_password(config)
    if not password:
        html = _CONSOLE_LOGIN_HTML.replace(
            "__ERROR_BLOCK__", '<p class="err">Console auth is not configured.</p>'
        )
        return Response(content=html, media_type="text/html", status_code=503)

    if request.method == "GET":
        err = request.query_params.get("err")
        error_block = '<p class="err">Invalid password</p>' if err == "1" else ""
        html = _CONSOLE_LOGIN_HTML.replace("__ERROR_BLOCK__", error_block)
        return Response(content=html, media_type="text/html")

    body = (await request.body()).decode("utf-8", errors="ignore")
    form = parse_qs(body)
    submitted = (form.get("password", [""])[0] or "").strip()
    if not hmac.compare_digest(submitted, password):
        return Response(status_code=303, headers={"Location": "/console/login?err=1"})

    payload = {"iat": int(time.time()), "user": "console"}
    payload_b64 = (
        base64.urlsafe_b64encode(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        )
        .decode("utf-8")
        .rstrip("=")
    )
    cookie_value = _sign_console_cookie(payload_b64, _console_signing_key(config))

    resp = Response(status_code=303, headers={"Location": "/console"})
    resp.set_cookie(
        "t2m_console",
        cookie_value,
        httponly=True,
        samesite="lax",
        max_age=12 * 60 * 60,
        path="/",
    )
    return resp


async def _handle_console_logout(_request: Request) -> Response:
    resp = Response(status_code=303, headers={"Location": "/console/login"})
    resp.delete_cookie("t2m_console", path="/")
    return resp


async def _handle_console(request: Request) -> Response:
    if request.url.path.endswith("/"):
        return Response(status_code=307, headers={"Location": "/console"})
    html = _CONSOLE_HTML.replace("__CONSOLE_JS_VERSION__", __version__)
    return Response(content=html, media_type="text/html")


async def _handle_console_api_summary(_request: Request) -> JSONResponse:
    return JSONResponse(_query_usage_summary())


def _parse_console_ts(value: str) -> str | None:
    v = (value or "").strip()
    if not v:
        return None
    try:
        if v.endswith("Z"):
            v = v[:-1] + "+00:00"
        dt = datetime.fromisoformat(v)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_LOCAL_TZINFO or timezone.utc)
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt.isoformat(sep=" ", timespec="seconds")
    except Exception:
        return None


def _normalize_bool_param(value: str | None) -> bool | None:
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y"}:
        return True
    if v in {"0", "false", "no", "n"}:
        return False
    return None


async def _handle_console_api_requests(request: Request) -> JSONResponse:
    try:
        limit = int(request.query_params.get("limit") or "50")
    except Exception:
        limit = 50
    try:
        offset = int(request.query_params.get("offset") or "0")
    except Exception:
        offset = 0

    limit = max(1, min(200, limit))
    offset = max(0, offset)

    run_id = (request.query_params.get("run_id") or "").strip()
    run_id_mode = (request.query_params.get("run_id_mode") or "exact").strip().lower()
    route = (request.query_params.get("route") or "").strip()
    method = (request.query_params.get("method") or "").strip().upper()
    status = (request.query_params.get("status") or "").strip().upper()
    since = _parse_console_ts(request.query_params.get("since") or "")
    until = _parse_console_ts(request.query_params.get("until") or "")
    success_filter = _normalize_bool_param(request.query_params.get("success"))
    q = (request.query_params.get("q") or "").strip()
    only_query = (request.query_params.get("only_query") or "1").strip() not in {
        "0",
        "false",
        "False",
        "no",
    }

    db_path = _usage_db_path()
    if not db_path.exists():
        return JSONResponse(
            {"has_data": False, "rows": [], "limit": limit, "offset": offset}
        )

    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS http_requests (
              ts TIMESTAMP,
              request_id VARCHAR,
              route VARCHAR,
              path VARCHAR,
              method VARCHAR,
              status_code INTEGER,
              duration_ms DOUBLE,
              success BOOLEAN,
              run_id VARCHAR,
              query_text VARCHAR,
              params_json VARCHAR,
              response_json VARCHAR
            )
            """
        )
        try:
            con.execute(
                "ALTER TABLE http_requests ADD COLUMN IF NOT EXISTS response_json VARCHAR"
            )
        except Exception:
            pass

        where: list[str] = []
        params: list[Any] = []
        if run_id:
            if run_id_mode == "prefix":
                where.append("run_id ILIKE ?")
                params.append(f"{run_id}%")
            else:
                where.append("run_id = ?")
                params.append(run_id)
        if route:
            where.append("route ILIKE ?")
            params.append(f"%{route}%")
        if method:
            where.append("method = ?")
            params.append(method)
        if status in {"OK", "ERR"}:
            where.append("success = ?")
            params.append(status == "OK")
        elif success_filter is not None:
            where.append("success = ?")
            params.append(success_filter)
        if q:
            where.append("query_text ILIKE ?")
            params.append(f"%{q}%")
        if only_query:
            where.append("query_text IS NOT NULL AND query_text <> ''")
        if since:
            where.append("ts >= ?::TIMESTAMP")
            params.append(since)
        if until:
            where.append("ts <= ?::TIMESTAMP")
            params.append(until)

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        rows = con.execute(
            f"""
            SELECT
              ts,
              request_id,
              route,
              path,
              method,
              status_code,
              duration_ms,
              success,
              run_id,
              query_text
            FROM http_requests
            {where_sql}
            ORDER BY ts DESC
            LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        ).fetchall()
    finally:
        con.close()

    def _ts(v: Any) -> str:
        return _format_ts_utc(v)

    return JSONResponse(
        {
            "has_data": True,
            "limit": limit,
            "offset": offset,
            "rows": [
                {
                    "ts": _ts(r[0]),
                    "request_id": r[1],
                    "route": r[2],
                    "path": r[3],
                    "method": r[4],
                    "status_code": int(r[5] or 0),
                    "duration_ms": float(r[6]) if r[6] is not None else None,
                    "success": bool(r[7]),
                    "run_id": r[8],
                    "query": r[9],
                }
                for r in rows
            ],
        }
    )


async def _handle_console_api_request_detail(request: Request) -> JSONResponse:
    request_id = (request.path_params.get("request_id") or "").strip()
    if not request_id:
        return JSONResponse({"error": "missing request_id"}, status_code=400)

    db_path = _usage_db_path()
    if not db_path.exists():
        return JSONResponse({"error": "no data"}, status_code=404)

    con = duckdb.connect(str(db_path))
    try:
        row = con.execute(
            """
            SELECT
              ts,
              request_id,
              route,
              path,
              method,
              status_code,
              duration_ms,
              success,
              run_id,
              query_text,
              params_json,
              response_json
            FROM http_requests
            WHERE request_id = ?
            ORDER BY ts DESC
            LIMIT 1
            """,
            [request_id],
        ).fetchone()
    finally:
        con.close()

    if not row:
        return JSONResponse({"error": "not found"}, status_code=404)

    def _ts(v: Any) -> str:
        return _format_ts_utc(v)

    return JSONResponse(
        {
            "ts": _ts(row[0]),
            "request_id": row[1],
            "route": row[2],
            "path": row[3],
            "method": row[4],
            "status_code": int(row[5] or 0),
            "duration_ms": float(row[6]) if row[6] is not None else None,
            "success": bool(row[7]),
            "run_id": row[8],
            "query": row[9],
            "params_json": row[10],
            "response_json": row[11],
        }
    )


def _console_export_format(request: Request) -> str:
    fmt = (request.query_params.get("format") or "json").strip().lower()
    return fmt


def _console_export_limit(request: Request) -> int:
    try:
        limit = int(request.query_params.get("limit") or "50000")
    except Exception:
        limit = 50000
    return max(1, min(200000, limit))


def _console_export_filters(request: Request) -> tuple[str, list[Any]]:
    run_id = (request.query_params.get("run_id") or "").strip()
    run_id_mode = (request.query_params.get("run_id_mode") or "exact").strip().lower()
    route = (request.query_params.get("route") or "").strip()
    method = (request.query_params.get("method") or "").strip().upper()
    status = (request.query_params.get("status") or "").strip().upper()
    since = _parse_console_ts(request.query_params.get("since") or "")
    until = _parse_console_ts(request.query_params.get("until") or "")
    success_filter = _normalize_bool_param(request.query_params.get("success"))
    q = (request.query_params.get("q") or "").strip()
    only_query = (request.query_params.get("only_query") or "1").strip() not in {
        "0",
        "false",
        "False",
        "no",
    }

    where: list[str] = []
    params: list[Any] = []
    if run_id:
        if run_id_mode == "prefix":
            where.append("run_id ILIKE ?")
            params.append(f"{run_id}%")
        else:
            where.append("run_id = ?")
            params.append(run_id)
    if route:
        where.append("route ILIKE ?")
        params.append(f"%{route}%")
    if method:
        where.append("method = ?")
        params.append(method)
    if status in {"OK", "ERR"}:
        where.append("success = ?")
        params.append(status == "OK")
    elif success_filter is not None:
        where.append("success = ?")
        params.append(success_filter)
    if q:
        where.append("query_text ILIKE ?")
        params.append(f"%{q}%")
    if only_query:
        where.append("query_text IS NOT NULL AND query_text <> ''")
    if since:
        where.append("ts >= ?::TIMESTAMP")
        params.append(since)
    if until:
        where.append("ts <= ?::TIMESTAMP")
        params.append(until)

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    return where_sql, params


def _console_export_columns(request: Request) -> list[str]:
    include_params = (
        _normalize_bool_param(request.query_params.get("include_params")) is True
    )
    include_response = (
        _normalize_bool_param(request.query_params.get("include_response")) is True
    )

    cols: list[str] = [
        "ts",
        "request_id",
        "route",
        "path",
        "method",
        "status_code",
        "duration_ms",
        "success",
        "run_id",
        "query_text",
    ]
    if include_params:
        cols.append("params_json")
    if include_response:
        cols.append("response_json")
    return cols


def _console_export_jsonl_iter(
    cur: duckdb.DuckDBPyConnection, cols: list[str]
) -> Iterable[bytes]:
    while True:
        batch = cur.fetchmany(1000)
        if not batch:
            break
        for r in batch:
            obj = {}
            for i, k in enumerate(cols):
                v = r[i]
                if isinstance(v, datetime):
                    v = _format_ts_utc(v)
                # For params/response columns, try to parse JSON so exports contain real JSON
                if k in {"params_json", "response_json"} and isinstance(v, str) and v:
                    try:
                        v = json.loads(v)
                    except Exception:
                        pass
                obj[k] = v
            yield (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def _console_export_json_iter(
    cur: duckdb.DuckDBPyConnection, cols: list[str]
) -> Iterable[bytes]:
    first = True
    yield b"[\n"
    while True:
        batch = cur.fetchmany(1000)
        if not batch:
            break
        for r in batch:
            obj = {}
            for i, k in enumerate(cols):
                v = r[i]
                if isinstance(v, datetime):
                    v = _format_ts_utc(v)
                # For params/response columns, try to parse JSON so exports contain real JSON
                if k in {"params_json", "response_json"} and isinstance(v, str) and v:
                    try:
                        v = json.loads(v)
                    except Exception:
                        pass
                obj[k] = v
            if not first:
                yield b",\n"
            first = False
            yield json.dumps(obj, ensure_ascii=False).encode("utf-8")
    yield b"\n]\n"


def _console_export_csv_iter(
    cur: duckdb.DuckDBPyConnection, cols: list[str]
) -> Iterable[bytes]:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    yield buf.getvalue().encode("utf-8")
    buf.seek(0)
    buf.truncate(0)

    while True:
        batch = cur.fetchmany(2000)
        if not batch:
            break
        for r in batch:
            row_out = []
            for v in r:
                if isinstance(v, datetime):
                    v = _format_ts_utc(v)
                row_out.append(v)
            w.writerow(row_out)
        yield buf.getvalue().encode("utf-8")
        buf.seek(0)
        buf.truncate(0)


def _console_export_stream(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    params: list[Any],
    limit: int,
    cols: list[str],
    fmt: str,
) -> Iterable[bytes]:
    cur = con.execute(sql, [*params, limit])
    if fmt == "ndjson":
        return _console_export_jsonl_iter(cur, cols)
    if fmt == "json":
        return _console_export_json_iter(cur, cols)
    return _console_export_csv_iter(cur, cols)


def _console_export_filename_media(fmt: str, ts_suffix: str) -> tuple[str, str]:
    if fmt == "ndjson":
        return f"talk2metadata_logs_{ts_suffix}.jsonl", "application/x-ndjson"
    if fmt == "json":
        return f"talk2metadata_logs_{ts_suffix}.json", "application/json"
    return f"talk2metadata_logs_{ts_suffix}.csv", "text/csv"


async def _handle_console_api_export(request: Request) -> Response:
    fmt = _console_export_format(request)
    if fmt not in {"json", "ndjson", "csv"}:
        return JSONResponse({"error": "invalid format"}, status_code=400)
    limit = _console_export_limit(request)

    db_path = _usage_db_path()
    if not db_path.exists():
        return JSONResponse({"error": "no data"}, status_code=404)

    con = duckdb.connect(str(db_path), read_only=True)
    where_sql, params = _console_export_filters(request)
    cols = _console_export_columns(request)

    sql = f"""
        SELECT {", ".join(cols)}
        FROM http_requests
        {where_sql}
        ORDER BY ts DESC
        LIMIT ?
    """

    ts_suffix = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    filename, media_type = _console_export_filename_media(fmt, ts_suffix)

    def _iter_rows():
        try:
            yield from _console_export_stream(con, sql, params, limit, cols, fmt)
        finally:
            con.close()

    return StreamingResponse(
        _iter_rows(),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _json_from_text_content(
    contents: list[Any],
) -> tuple[dict[str, Any] | None, str | None]:
    if not contents:
        return None, None
    first = contents[0]
    text = getattr(first, "text", None)
    if not isinstance(text, str):
        return None, None
    try:
        return json.loads(text), None
    except Exception:
        return None, text


def _discover_runs() -> list[dict[str, str]]:
    from datetime import datetime, timezone

    from talk2metadata.utils.config import get_config
    from talk2metadata.utils.paths import find_schema_file, get_run_base_dir

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

        schema_path = None
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
                )
                .astimezone()
                .isoformat(),
            }
        )

    runs.sort(key=lambda r: r["run_id"])
    return runs


async def _handle_mcp_endpoint(
    request: Request, *, http_transport: StreamableHTTPServerTransport
) -> Response:
    captured = bytearray()
    limit = 200_000

    async def _send(message: Any) -> None:
        if isinstance(message, dict):
            if message.get("type") == "http.response.start":
                status = message.get("status")
                if isinstance(status, int):
                    request.state.t2m_status_code = status
            elif message.get("type") == "http.response.body":
                body = message.get("body")
                if (
                    isinstance(body, (bytes, bytearray))
                    and body
                    and len(captured) < limit
                ):
                    remaining = limit - len(captured)
                    captured.extend(body[:remaining])
        await request._send(message)

    await http_transport.handle_request(request.scope, request.receive, _send)
    if captured:
        try:
            request.state.t2m_response_json = _normalize_response_to_json(
                captured.decode("utf-8", errors="replace")
            )
        except Exception:
            pass
    return Response()


async def _handle_api_queue_status(request: Request) -> JSONResponse:
    """Queue status for external monitoring (REST token auth via /api/ path)."""
    return JSONResponse({
        "service": SERVER_NAME,
        "version": __version__,
        "counts": {},
        "tasks": [],
    })


async def _handle_health(_request: Request) -> JSONResponse:
    return JSONResponse(
        {"status": "healthy", "service": SERVER_NAME, "version": __version__}
    )


async def _handle_api_list_tables(request: Request) -> JSONResponse:
    from talk2metadata.mcp.tools.list_tables import handle_list_tables

    run_id = request.query_params.get("run_id")
    result = await handle_list_tables({"run_id": run_id} if run_id else {})
    payload, raw = _json_from_text_content(result)
    if payload is not None:
        return JSONResponse(payload, status_code=400 if "error" in payload else 200)
    return JSONResponse({"raw": raw}, status_code=200)


async def _handle_api_get_schema(request: Request) -> JSONResponse:
    from talk2metadata.mcp.tools.get_schema import handle_get_schema

    run_id = request.query_params.get("run_id")
    result = await handle_get_schema({"run_id": run_id} if run_id else {})
    payload, raw = _json_from_text_content(result)
    if payload is not None:
        return JSONResponse(payload, status_code=400 if "error" in payload else 200)
    return JSONResponse({"raw": raw}, status_code=200)


async def _handle_api_get_table_info(request: Request) -> JSONResponse:
    from talk2metadata.mcp.tools.get_table_info import handle_get_table_info

    run_id = request.query_params.get("run_id")
    table_name = request.path_params.get("table_name")
    args: dict[str, Any] = {"table_name": table_name}
    if run_id:
        args["run_id"] = run_id
    result = await handle_get_table_info(args)
    payload, raw = _json_from_text_content(result)
    if payload is not None:
        return JSONResponse(payload, status_code=400 if "error" in payload else 200)
    return JSONResponse({"raw": raw}, status_code=200)


async def _handle_api_search(request: Request) -> JSONResponse:
    from talk2metadata.mcp.tools.search import handle_search

    try:
        body = await request.json()
    except Exception:
        body = {}

    args = {
        "run_id": body.get("run_id"),
        "query": body.get("query"),
        "top_k": body.get("top_k", 5),
        "mode": body.get("mode"),
    }
    result = await handle_search(args)
    payload, raw = _json_from_text_content(result)
    if payload is not None:
        return JSONResponse(payload, status_code=400 if "error" in payload else 200)
    return JSONResponse({"raw": raw}, status_code=200)


async def _handle_api_sql(request: Request) -> JSONResponse:
    """
    POST /api/sql
    Body: { "run_id": str, "sql": str, "limit"?: int }

    Execute a raw SQL SELECT directly against the run's SQLite DB — bypasses
    the text2sql LLM entirely. Intended for trusted internal callers (e.g.
    KAIAPlatform proxy) where SQL is already structured, e.g. filter-only
    search from DrSun.

    Safety: only SELECT statements allowed. Rejects anything containing
    DDL/DML keywords. Callers must still sanitize user input that gets
    interpolated into the SQL.
    """
    from talk2metadata.mcp.common.retriever import get_retriever

    try:
        body = await request.json()
    except Exception:
        body = {}

    run_id = body.get("run_id")
    sql = (body.get("sql") or "").strip()
    limit = body.get("limit")

    if not sql:
        return JSONResponse({"error": "sql is required"}, status_code=400)

    # Guard: only allow SELECT (or WITH ... SELECT). Forbid multi-statement and
    # any DDL/DML verbs that could mutate the DB.
    lowered = sql.lower()
    first_word = lowered.lstrip("(").split(None, 1)[0] if lowered else ""
    if first_word not in ("select", "with"):
        return JSONResponse(
            {"error": "only SELECT/WITH queries are allowed"}, status_code=400,
        )
    forbidden = (
        " insert ", " update ", " delete ", " drop ", " alter ", " create ",
        " truncate ", " attach ", " detach ", " pragma ", " vacuum ",
    )
    padded = f" {lowered.replace(chr(10), ' ').replace(chr(9), ' ')} "
    if any(kw in padded for kw in forbidden):
        return JSONResponse(
            {"error": "mutation keywords are not allowed"}, status_code=400,
        )
    if ";" in sql.rstrip(";"):
        return JSONResponse(
            {"error": "multiple statements not allowed"}, status_code=400,
        )

    # Optional LIMIT append (only if caller provided, and user SQL doesn't
    # already have one)
    if isinstance(limit, int) and limit > 0 and " limit " not in lowered:
        sql = f"{sql.rstrip(';').rstrip()} LIMIT {limit}"

    try:
        # Reuse the cached text2sql retriever — gives us a ready SQLAlchemy
        # engine pointing at the run's SQLite DB, without spinning up the LLM.
        retriever = get_retriever(run_id=run_id, mode_name="text2sql")
        df = retriever._execute_sql(sql)  # type: ignore[attr-defined]
        # Clean up DataFrame values for JSON serialization: replace NaN/Inf
        # (which are not JSON-compliant) with None.
        if df is not None and not df.empty:
            import math
            import numpy as np  # noqa: F401 — used via df.replace

            df = df.replace({float("inf"): None, float("-inf"): None})
            df = df.where(df.notna(), None)
            rows = df.to_dict(orient="records")
            for row in rows:
                for k, v in list(row.items()):
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        row[k] = None
        else:
            rows = []
        return JSONResponse(
            {
                "run_id": run_id,
                "sql": sql,
                "row_count": len(rows),
                "rows": rows,
            },
            status_code=200,
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


async def _handle_api_list_runs(_request: Request) -> JSONResponse:
    from talk2metadata.utils.config import get_config

    config = get_config()
    configured_run_id = config.get("run_id")
    runs = _discover_runs()
    return JSONResponse(
        {
            "configured_run_id": configured_run_id,
            "run_count": len(runs),
            "runs": [
                {
                    **r,
                    "endpoints": {
                        "run": f"/api/run/{r['run_id']}",
                        "context": f"/api/run/{r['run_id']}/context",
                        "tables": f"/api/run/{r['run_id']}/tables",
                        "schema": f"/api/run/{r['run_id']}/schema",
                        "table": f"/api/run/{r['run_id']}/table/{{table_name}}",
                    },
                }
                for r in runs
            ],
        }
    )


async def _handle_api_run_details(request: Request) -> JSONResponse:
    from talk2metadata.mcp.common.schema_index import get_schema
    from talk2metadata.utils.config import get_config
    from talk2metadata.utils.paths import get_run_base_dir

    run_id = request.path_params.get("run_id")
    config = get_config()
    run_base = get_run_base_dir(run_id=run_id, config=config)

    schema_summary: dict[str, object]
    try:
        schema = get_schema(run_id=run_id)
        table_names = sorted(schema.tables.keys())
        schema_summary = {
            "target_table": schema.target_table,
            "table_count": len(schema.tables),
            "foreign_key_count": len(schema.foreign_keys),
            "tables_preview": table_names[:10],
        }
    except FileNotFoundError as e:
        schema_summary = {
            "error": "Schema not found",
            "error_code": "SCHEMA_NOT_FOUND",
            "message": str(e),
        }

    return JSONResponse(
        {
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
        }
    )


async def _handle_api_run_context(request: Request) -> JSONResponse:
    from talk2metadata.mcp.tools.get_schema import handle_get_schema

    run_id = request.path_params.get("run_id")
    result = await handle_get_schema({"run_id": run_id})
    payload, raw = _json_from_text_content(result)
    if payload is not None:
        if "error" in payload:
            return JSONResponse(payload, status_code=400)
        return JSONResponse(
            {
                "run_id": run_id,
                "target_table": payload.get("target_table"),
                "table_count": payload.get("table_count"),
                "foreign_key_count": payload.get("foreign_key_count"),
            }
        )
    return JSONResponse({"run_id": run_id, "raw": raw}, status_code=200)


async def _handle_api_run_tables(request: Request) -> JSONResponse:
    from talk2metadata.mcp.tools.list_tables import handle_list_tables

    run_id = request.path_params.get("run_id")
    result = await handle_list_tables({"run_id": run_id})
    payload, raw = _json_from_text_content(result)
    if payload is not None:
        return JSONResponse(payload, status_code=400 if "error" in payload else 200)
    return JSONResponse({"run_id": run_id, "raw": raw}, status_code=200)


async def _handle_api_run_schema(request: Request) -> JSONResponse:
    from talk2metadata.mcp.tools.get_schema import handle_get_schema

    run_id = request.path_params.get("run_id")
    result = await handle_get_schema({"run_id": run_id})
    payload, raw = _json_from_text_content(result)
    if payload is not None:
        return JSONResponse(payload, status_code=400 if "error" in payload else 200)
    return JSONResponse({"run_id": run_id, "raw": raw}, status_code=200)


async def _handle_api_run_table_info(request: Request) -> JSONResponse:
    from talk2metadata.mcp.tools.get_table_info import handle_get_table_info

    run_id = request.path_params.get("run_id")
    table_name = request.path_params.get("table_name")
    result = await handle_get_table_info({"run_id": run_id, "table_name": table_name})
    payload, raw = _json_from_text_content(result)
    if payload is not None:
        return JSONResponse(payload, status_code=400 if "error" in payload else 200)
    return JSONResponse({"run_id": run_id, "raw": raw}, status_code=200)


def _openapi_spec(*, config: MCPConfig) -> dict[str, Any]:
    return {
        "openapi": "3.0.3",
        "info": {
            "title": SERVER_NAME,
            "version": __version__,
            "termsOfService": f"{config.server.base_url}/terms",
            "x-privacyPolicy": f"{config.server.base_url}/privacy",
        },
        "servers": [{"url": config.server.base_url}],
        "components": {
            "securitySchemes": {
                "TokenAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "Authorization",
                    "description": (
                        "Use Django REST framework token auth: Authorization: Token <key>"
                    ),
                }
            }
        },
        "security": [{"TokenAuth": []}],
        "paths": {
            "/api/runs": {
                "get": {
                    "summary": "List available runs",
                    "security": [{"TokenAuth": []}],
                    "responses": {"200": {"description": "OK"}},
                }
            },
            "/api/run/{run_id}": {
                "get": {
                    "summary": "Get run details",
                    "security": [{"TokenAuth": []}],
                    "parameters": [
                        {
                            "name": "run_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string", "example": "wamex"},
                        }
                    ],
                    "responses": {"200": {"description": "OK"}},
                }
            },
            "/api/run/{run_id}/context": {
                "get": {
                    "summary": "Get run context summary",
                    "security": [{"TokenAuth": []}],
                    "parameters": [
                        {
                            "name": "run_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string", "example": "wamex"},
                        }
                    ],
                    "responses": {"200": {"description": "OK"}},
                }
            },
            "/api/run/{run_id}/tables": {
                "get": {
                    "summary": "List tables for a run",
                    "security": [{"TokenAuth": []}],
                    "parameters": [
                        {
                            "name": "run_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string", "example": "wamex"},
                        }
                    ],
                    "responses": {"200": {"description": "OK"}},
                }
            },
            "/api/run/{run_id}/schema": {
                "get": {
                    "summary": "Get schema for a run",
                    "security": [{"TokenAuth": []}],
                    "parameters": [
                        {
                            "name": "run_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string", "example": "wamex"},
                        }
                    ],
                    "responses": {"200": {"description": "OK"}},
                }
            },
            "/api/run/{run_id}/table/{table_name}": {
                "get": {
                    "summary": "Get table info for a run",
                    "security": [{"TokenAuth": []}],
                    "parameters": [
                        {
                            "name": "run_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string", "example": "wamex"},
                        },
                        {
                            "name": "table_name",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string", "example": "orders"},
                        },
                    ],
                    "responses": {"200": {"description": "OK"}},
                }
            },
            "/api/tables": {
                "get": {
                    "summary": "List tables",
                    "security": [{"TokenAuth": []}],
                    "parameters": [
                        {
                            "name": "run_id",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string", "example": "wamex"},
                        }
                    ],
                    "responses": {"200": {"description": "OK"}},
                }
            },
            "/api/schema": {
                "get": {
                    "summary": "Get schema",
                    "security": [{"TokenAuth": []}],
                    "parameters": [
                        {
                            "name": "run_id",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string", "example": "wamex"},
                        }
                    ],
                    "responses": {"200": {"description": "OK"}},
                }
            },
            "/api/table/{table_name}": {
                "get": {
                    "summary": "Get table info",
                    "security": [{"TokenAuth": []}],
                    "parameters": [
                        {
                            "name": "table_name",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string", "example": "orders"},
                        },
                        {
                            "name": "run_id",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string", "example": "wamex"},
                        },
                    ],
                    "responses": {"200": {"description": "OK"}},
                }
            },
            "/api/search": {
                "post": {
                    "summary": "Search",
                    "security": [{"TokenAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "run_id": {
                                            "type": "string",
                                            "example": "wamex",
                                            "default": "wamex",
                                        },
                                        "query": {"type": "string"},
                                        "top_k": {"type": "integer", "default": 5},
                                        "mode": {
                                            "type": "string",
                                            "example": "graph",
                                            "default": "graph",
                                        },
                                    },
                                    "required": ["query"],
                                }
                            }
                        },
                    },
                    "responses": {"200": {"description": "OK"}},
                }
            },
        },
    }


async def _handle_openapi(_request: Request, *, config: MCPConfig) -> JSONResponse:
    return JSONResponse(_openapi_spec(config=config))


_DOCS_HTML = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Talk2Metadata REST API Docs</title>
    <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
    <style>
      body {
        margin: 0;
      }
      .docs-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 16px;
        padding: 18px 20px;
        border-bottom: 1px solid #e2e8f0;
        background: #ffffff;
      }
      .docs-header img {
        height: 44px;
        width: auto;
      }
      .docs-header-title {
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
        font-weight: 800;
        font-size: 18px;
        color: #0f172a;
      }
      .swagger-ui .topbar {
        background: #ffffff;
      }
      .swagger-ui .topbar-wrapper img {
        content: url("/assets/logo.svg");
        width: 170px;
        height: auto;
      }
      .swagger-ui .topbar-wrapper a {
        max-width: 240px;
      }
    </style>
  </head>
  <body>
    <div class="docs-header">
      <img src="/assets/logo.svg" alt="Talk2Metadata" />
      <div class="docs-header-title">Talk2Metadata REST API Docs</div>
    </div>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script>
      window.ui = SwaggerUIBundle({
        url: "/openapi.json",
        dom_id: "#swagger-ui",
        presets: [SwaggerUIBundle.presets.apis],
        layout: "BaseLayout",
        persistAuthorization: true,
        requestInterceptor: (req) => {
          const headers = req.headers || {};
          const authHeaderKey = headers.Authorization ? "Authorization" : (headers.authorization ? "authorization" : null);
          if (authHeaderKey) {
            const v = (headers[authHeaderKey] || "").trim();
            if (v && !v.toLowerCase().startsWith("token ") && !v.toLowerCase().startsWith("bearer ")) {
              headers[authHeaderKey] = "Token " + v;
              req.headers = headers;
            }
          }
          return req;
        }
      });
    </script>
  </body>
</html>
"""

_PRIVACY_HTML = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Talk2Metadata - Privacy Policy</title>
    <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
    <style>
      body {
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
        margin: 0;
        padding: 24px;
        color: #0f172a;
        background: #ffffff;
      }
      main {
        max-width: 820px;
        margin: 0 auto;
      }
      h1 {
        font-size: 28px;
        margin: 0 0 12px 0;
      }
      p, li {
        line-height: 1.55;
        color: #334155;
      }
      a { color: #2563eb; }
    </style>
  </head>
  <body>
    <main>
      <h1>Privacy Policy</h1>
      <p>This MCP server provides semantic search and schema exploration tools. Your data handling depends on your deployment and configuration.</p>
      <ul>
        <li>Requests may be logged for operational purposes (e.g., errors, metrics).</li>
        <li>Authentication tokens are processed for authorization and are not intentionally stored in plaintext.</li>
        <li>Do not send secrets in prompts or tool inputs.</li>
      </ul>
      <p>For questions, contact the server operator.</p>
    </main>
  </body>
</html>
"""

_TERMS_HTML = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Talk2Metadata - Terms of Service</title>
    <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
    <style>
      body {
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
        margin: 0;
        padding: 24px;
        color: #0f172a;
        background: #ffffff;
      }
      main {
        max-width: 820px;
        margin: 0 auto;
      }
      h1 {
        font-size: 28px;
        margin: 0 0 12px 0;
      }
      p, li {
        line-height: 1.55;
        color: #334155;
      }
      a { color: #2563eb; }
    </style>
  </head>
  <body>
    <main>
      <h1>Terms of Service</h1>
      <ul>
        <li>This service is provided as-is, without warranty of any kind.</li>
        <li>You are responsible for complying with applicable laws and your organization policies.</li>
        <li>You must have authorization to access any data you query through this server.</li>
      </ul>
      <p>By using this service, you agree to these terms.</p>
    </main>
  </body>
</html>
"""


async def _handle_docs(_request: Request) -> Response:
    return Response(content=_DOCS_HTML, media_type="text/html")


async def _handle_privacy(_request: Request) -> Response:
    return Response(content=_PRIVACY_HTML, media_type="text/html")


async def _handle_terms(_request: Request) -> Response:
    return Response(content=_TERMS_HTML, media_type="text/html")


async def _handle_metrics(_request: Request) -> JSONResponse:
    collector = get_metrics_collector()
    snapshot = collector.get_snapshot()
    tool_counts = collector.get_tool_counts()

    metrics_dict = snapshot.to_dict()
    metrics_dict["tool_usage"] = tool_counts

    return JSONResponse(metrics_dict)


async def _handle_metrics_prometheus(_request: Request) -> Response:
    collector = get_metrics_collector()
    snapshot = collector.get_snapshot()
    prometheus_text = MetricsExporter.to_prometheus(snapshot)

    return Response(
        content=prometheus_text,
        media_type="text/plain; version=0.0.4",
    )


async def _handle_favicon(request: Request) -> Response:
    favicon_name = request.url.path.lstrip("/")
    static_dir = Path(__file__).parent / "static"
    favicon_path = static_dir / favicon_name

    if favicon_path.exists() and favicon_path.is_file():
        content_type_map = {
            ".ico": "image/x-icon",
            ".png": "image/png",
            ".svg": "image/svg+xml",
        }
        content_type = content_type_map.get(favicon_path.suffix.lower(), "image/x-icon")

        return FileResponse(
            str(favicon_path),
            media_type=content_type,
            headers={"Cache-Control": "public, max-age=31536000"},
        )

    if favicon_name in {"favicon.ico", "favicon.png"}:
        svg_path = static_dir / "favicon.svg"
        if svg_path.exists() and svg_path.is_file():
            return Response(
                status_code=307,
                headers={"Location": "/favicon.svg"},
            )

    logger.debug(f"Favicon not found: {favicon_path}, returning 204")
    return Response(status_code=204)


async def _handle_asset(request: Request) -> Response:
    asset_rel = request.path_params.get("path", "")
    static_dir = Path(__file__).parent / "static"
    static_resolved = static_dir.resolve()
    asset_path = (static_dir / asset_rel).resolve()

    if static_resolved not in asset_path.parents and asset_path != static_resolved:
        return Response(status_code=404)

    if not asset_path.exists() or not asset_path.is_file():
        return Response(status_code=404)

    content_type_map = {
        ".js": "application/javascript",
        ".ico": "image/x-icon",
        ".png": "image/png",
        ".svg": "image/svg+xml",
        ".webmanifest": "application/manifest+json",
    }
    content_type = content_type_map.get(
        asset_path.suffix.lower(), "application/octet-stream"
    )

    cache_control = "public, max-age=31536000"
    if asset_path.suffix.lower() == ".js":
        cache_control = "no-cache"

    return FileResponse(
        str(asset_path),
        media_type=content_type,
        headers={"Cache-Control": cache_control},
    )


async def _handle_metadata(_request: Request, *, config: MCPConfig) -> JSONResponse:
    auth_required = config.oauth.protect_mcp
    return JSONResponse(
        {
            "name": SERVER_NAME,
            "version": __version__,
            "instructions": SERVER_INSTRUCTIONS,
            "capabilities": {
                "tools": True,
                "resources": True,
                "resourceTemplates": True,
                "prompts": True,
            },
            "authentication": {
                "required": auth_required,
                "type": "oauth2" if auth_required else "none",
                "discovery": (
                    {
                        "oauth": "/.well-known/oauth-authorization-server",
                        "oidc": "/.well-known/openid-configuration",
                        "resource": "/.well-known/oauth-protected-resource",
                    }
                    if auth_required
                    else {}
                ),
            },
            "endpoints": {"mcp": "/mcp", "health": "/health"},
            "transport": {
                "type": "streamable-http",
                "description": "StreamableHTTP transport (MCP protocol 2025-03-26)",
            },
        }
    )


async def _handle_oauth_metadata(
    request: Request, *, config: MCPConfig
) -> JSONResponse:
    return await oauth_proxy.handle_oauth_metadata(config, request)


async def _handle_protected_resource(
    request: Request, *, config: MCPConfig
) -> JSONResponse:
    return await oauth_proxy.handle_protected_resource_metadata(config, request)


async def _handle_openid_config(request: Request, *, config: MCPConfig) -> JSONResponse:
    return await oauth_proxy.handle_openid_configuration(config, request)


async def _handle_client_reg(request: Request, *, config: MCPConfig) -> JSONResponse:
    logger.debug(
        f"Client registration endpoint hit: method={request.method}, path={request.url.path}"
    )
    return await oauth_proxy.handle_client_registration(config, request)


async def _handle_callback(request: Request, *, config: MCPConfig) -> Response:
    return await oauth_proxy.handle_oauth_callback(config, request)


async def _handle_oauth_proxy(request: Request, *, config: MCPConfig) -> Response:
    return await oauth_proxy.proxy_oauth_request(config, request)


async def _handle_accounts_proxy(request: Request, *, config: MCPConfig) -> Response:
    return await oauth_proxy.proxy_accounts_request(config, request)


async def _handle_login_proxy(request: Request, *, config: MCPConfig) -> Response:
    return await oauth_proxy.proxy_login_request(config, request)


async def _handle_root_oauth_redirect(request: Request) -> Response:
    """Redirect root-level OAuth paths to /oauth/* for MCP client compatibility."""
    from starlette.responses import RedirectResponse

    query = request.url.query
    target = f"/oauth{request.url.path}"
    if query:
        target = f"{target}?{query}"
    return RedirectResponse(url=target, status_code=307)


async def _handle_static_proxy(request: Request, *, config: MCPConfig) -> Response:
    return await oauth_proxy.proxy_static_request(config, request)


def create_mcp_server() -> Server:
    server = Server(
        name=SERVER_NAME, version=__version__, instructions=SERVER_INSTRUCTIONS
    )

    register_tools(server)
    register_resources(server)
    register_prompts(server)
    return server


def create_asgi_app(
    mcp_server: Server,
    oidc_resource_server: OIDCResourceServer,
    http_transport: StreamableHTTPServerTransport,
    config: MCPConfig,
) -> Starlette:
    """Create the ASGI application with MCP endpoints protected by OAuth."""

    rest_token_verifier = RestAuthTokenVerifier(
        verify_url=config.rest_auth.verify_url,
        verify_ssl=config.rest_auth.verify_ssl,
        timeout=config.rest_auth.timeout,
        cache_ttl_seconds=config.rest_auth.cache_ttl_seconds,
    )

    routes = [
        Route(
            "/openapi.json",
            endpoint=partial(_handle_openapi, config=config),
            methods=["GET"],
        ),
        Route("/docs", endpoint=_handle_docs, methods=["GET"]),
        Route("/docs/", endpoint=_handle_docs, methods=["GET"]),
        Route(
            "/console",
            endpoint=_handle_console,
            methods=["GET"],
        ),
        Route(
            "/console/",
            endpoint=_handle_console,
            methods=["GET"],
        ),
        Route(
            "/console/login",
            endpoint=partial(_handle_console_login, config=config),
            methods=["GET", "POST"],
        ),
        Route(
            "/console/login/",
            endpoint=partial(_handle_console_login, config=config),
            methods=["GET", "POST"],
        ),
        Route("/console/logout", endpoint=_handle_console_logout, methods=["GET"]),
        Route("/console/logout/", endpoint=_handle_console_logout, methods=["GET"]),
        Route(
            "/console/api/summary",
            endpoint=_handle_console_api_summary,
            methods=["GET"],
        ),
        Route(
            "/console/api/summary/",
            endpoint=_handle_console_api_summary,
            methods=["GET"],
        ),
        Route(
            "/console/api/requests",
            endpoint=_handle_console_api_requests,
            methods=["GET"],
        ),
        Route(
            "/console/api/requests/",
            endpoint=_handle_console_api_requests,
            methods=["GET"],
        ),
        Route(
            "/console/api/request/{request_id}",
            endpoint=_handle_console_api_request_detail,
            methods=["GET"],
        ),
        Route(
            "/console/api/request/{request_id}/",
            endpoint=_handle_console_api_request_detail,
            methods=["GET"],
        ),
        Route(
            "/console/api/export",
            endpoint=_handle_console_api_export,
            methods=["GET"],
        ),
        Route(
            "/console/api/export/",
            endpoint=_handle_console_api_export,
            methods=["GET"],
        ),
        Route("/privacy", endpoint=_handle_privacy, methods=["GET"]),
        Route("/privacy/", endpoint=_handle_privacy, methods=["GET"]),
        Route("/terms", endpoint=_handle_terms, methods=["GET"]),
        Route("/terms/", endpoint=_handle_terms, methods=["GET"]),
        Route("/terms-of-service", endpoint=_handle_terms, methods=["GET"]),
        Route("/terms-of-service/", endpoint=_handle_terms, methods=["GET"]),
        Route("/assets/{path:path}", endpoint=_handle_asset, methods=["GET"]),
        Route("/health", endpoint=_handle_health, methods=["GET"]),
        Route("/health/", endpoint=_handle_health, methods=["GET"]),
        Route("/api/runs", endpoint=_handle_api_list_runs, methods=["GET"]),
        Route("/api/run/{run_id}", endpoint=_handle_api_run_details, methods=["GET"]),
        Route(
            "/api/run/{run_id}/context",
            endpoint=_handle_api_run_context,
            methods=["GET"],
        ),
        Route(
            "/api/run/{run_id}/tables",
            endpoint=_handle_api_run_tables,
            methods=["GET"],
        ),
        Route(
            "/api/run/{run_id}/schema",
            endpoint=_handle_api_run_schema,
            methods=["GET"],
        ),
        Route(
            "/api/run/{run_id}/table/{table_name}",
            endpoint=_handle_api_run_table_info,
            methods=["GET"],
        ),
        Route("/api/tables", endpoint=_handle_api_list_tables, methods=["GET"]),
        Route("/api/schema", endpoint=_handle_api_get_schema, methods=["GET"]),
        Route(
            "/api/table/{table_name}",
            endpoint=_handle_api_get_table_info,
            methods=["GET"],
        ),
        Route("/api/search", endpoint=_handle_api_search, methods=["POST"]),
        Route("/api/sql", endpoint=_handle_api_sql, methods=["POST"]),
        Route("/api/queue_status", endpoint=_handle_api_queue_status, methods=["GET"]),
        Route("/api/queue_status/", endpoint=_handle_api_queue_status, methods=["GET"]),
        Route(
            "/static/{path:path}",
            endpoint=partial(_handle_static_proxy, config=config),
            methods=["GET"],
        ),
        Route("/favicon.ico", endpoint=_handle_favicon, methods=["GET"]),
        Route("/favicon.svg", endpoint=_handle_favicon, methods=["GET"]),
        Route("/favicon.png", endpoint=_handle_favicon, methods=["GET"]),
        Route(
            "/mcp",
            endpoint=partial(_handle_mcp_endpoint, http_transport=http_transport),
            methods=["GET", "POST", "DELETE"],
        ),
    ]
    if config.oauth.protect_mcp:
        routes.extend(
            [
                Route(
                    "/.well-known/oauth-authorization-server",
                    endpoint=partial(_handle_oauth_metadata, config=config),
                    methods=["GET", "OPTIONS"],
                ),
                Route(
                    "/.well-known/oauth-authorization-server/mcp",
                    endpoint=partial(_handle_oauth_metadata, config=config),
                    methods=["GET", "OPTIONS"],
                ),
                Route(
                    "/.well-known/oauth-authorization-server/oauth",
                    endpoint=partial(_handle_oauth_metadata, config=config),
                    methods=["GET", "OPTIONS"],
                ),
                Route(
                    "/.well-known/oauth-protected-resource",
                    endpoint=partial(_handle_protected_resource, config=config),
                    methods=["GET", "OPTIONS"],
                ),
                Route(
                    "/.well-known/oauth-protected-resource/mcp",
                    endpoint=partial(_handle_protected_resource, config=config),
                    methods=["GET", "OPTIONS"],
                ),
                Route(
                    "/.well-known/openid-configuration",
                    endpoint=partial(_handle_openid_config, config=config),
                    methods=["GET", "OPTIONS"],
                ),
                Route(
                    "/.well-known/openid-configuration/mcp",
                    endpoint=partial(_handle_openid_config, config=config),
                    methods=["GET", "OPTIONS"],
                ),
                Route(
                    "/.well-known/openid-configuration/oauth",
                    endpoint=partial(_handle_openid_config, config=config),
                    methods=["GET", "OPTIONS"],
                ),
                Route(
                    "/oauth/register",
                    endpoint=partial(_handle_client_reg, config=config),
                    methods=["GET", "POST", "OPTIONS"],
                ),
                Route(
                    "/oauth/callback",
                    endpoint=partial(_handle_callback, config=config),
                    methods=["GET", "OPTIONS"],
                ),
                Route(
                    "/oauth/{path:path}",
                    endpoint=partial(_handle_oauth_proxy, config=config),
                    methods=["GET", "POST", "OPTIONS"],
                ),
                Route(
                    "/accounts/{path:path}",
                    endpoint=partial(_handle_accounts_proxy, config=config),
                    methods=["GET", "POST", "OPTIONS"],
                ),
                Route(
                    "/login",
                    endpoint=partial(_handle_login_proxy, config=config),
                    methods=["GET", "POST", "OPTIONS"],
                ),
                Route(
                    "/login/",
                    endpoint=partial(_handle_login_proxy, config=config),
                    methods=["GET", "POST", "OPTIONS"],
                ),
                Route(
                    "/authorize",
                    endpoint=_handle_root_oauth_redirect,
                    methods=["GET", "OPTIONS"],
                ),
                Route(
                    "/authorize/",
                    endpoint=_handle_root_oauth_redirect,
                    methods=["GET", "OPTIONS"],
                ),
                Route(
                    "/token",
                    endpoint=_handle_root_oauth_redirect,
                    methods=["POST", "OPTIONS"],
                ),
                Route(
                    "/token/",
                    endpoint=_handle_root_oauth_redirect,
                    methods=["POST", "OPTIONS"],
                ),
            ]
        )

    middleware = [
        Middleware(UsageLoggingMiddleware),
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        ),
    ]
    middleware.append(Middleware(ConsoleAuthMiddleware, config=config))

    if rest_token_verifier.verify_url or config.rest_auth.token:
        middleware.append(
            Middleware(
                RestAuthMiddleware,
                token_verifier=rest_token_verifier,
                protected_paths=["/api"],
                static_token=config.rest_auth.token,
            )
        )
    if config.oauth.protect_mcp:
        middleware.append(
            Middleware(
                JWTAuthMiddleware,
                oidc_resource_server=oidc_resource_server,
                protected_paths=["/mcp"],
            )
        )

    app = Starlette(routes=routes, middleware=middleware)
    logger.info("ASGI application created with OAuth authentication")
    return app


def build_server(
    config: MCPConfig | None = None,
) -> tuple[Starlette, StreamableHTTPServerTransport, Server]:
    """Build and configure the complete MCP server with OAuth integration.

    Args:
        config: MCP configuration. If None, loads from config.mcp.yml and environment variables.

    Returns:
        Tuple of (Starlette ASGI app, StreamableHTTP transport, MCP Server)
    """
    if config is None:
        config = MCPConfig.load()

    logger.info(f"Building MCP server at {config.server.base_url}")
    logger.info("Using StreamableHTTP transport (MCP protocol 2025-03-26)")
    logger.info(
        f"Token validation: {'Introspection' if config.oauth.use_introspection else 'JWT'}"
    )

    oidc_resource_server = OIDCResourceServer(
        oidc_discovery_url=config.oauth.discovery_url,
        client_id=config.oauth.client_id,
        client_secret=config.oauth.client_secret,
        use_introspection=config.oauth.use_introspection,
        verify_ssl=config.oauth.verify_ssl,
        timeout=config.oauth.timeout,
    )

    mcp_server = create_mcp_server()
    http_transport = StreamableHTTPServerTransport(
        mcp_session_id=None,
        is_json_response_enabled=False,
        event_store=None,
    )

    app = create_asgi_app(mcp_server, oidc_resource_server, http_transport, config)
    return app, http_transport, mcp_server
