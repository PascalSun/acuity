"""Runtime metrics collection and export for monitoring."""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import duckdb

from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.timing import get_latency_tracker

logger = get_logger(__name__)


@dataclass
class MetricsSnapshot:
    """Snapshot of current metrics state."""

    timestamp: float = field(default_factory=time.time)
    uptime_seconds: float = 0.0
    total_requests: int = 0
    error_count: int = 0
    latency_stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_requests == 0:
            return 0.0
        return self.error_count / self.total_requests

    @property
    def requests_per_minute(self) -> float:
        """Calculate requests per minute."""
        if self.uptime_seconds == 0:
            return 0.0
        return (self.total_requests / self.uptime_seconds) * 60

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "error_rate": round(self.error_rate, 4),
            "requests_per_minute": round(self.requests_per_minute, 2),
            "latency_stats": self.latency_stats,
        }


class MetricsCollector:
    """Collects and aggregates metrics from various sources."""

    def __init__(self):
        """Initialize metrics collector."""
        self._lock = Lock()
        self._request_count = 0
        self._error_count = 0
        self._tool_counts: Dict[str, int] = {}

    def increment_requests(self, tool_name: Optional[str] = None) -> None:
        """Increment request counter."""
        with self._lock:
            self._request_count += 1
            if tool_name:
                self._tool_counts[tool_name] = self._tool_counts.get(tool_name, 0) + 1

    def increment_errors(self) -> None:
        """Increment error counter."""
        with self._lock:
            self._error_count += 1

    def get_snapshot(self) -> MetricsSnapshot:
        """Get current metrics snapshot."""
        tracker = get_latency_tracker()

        with self._lock:
            return MetricsSnapshot(
                uptime_seconds=tracker.get_uptime_seconds(),
                total_requests=self._request_count,
                error_count=self._error_count,
                latency_stats=tracker.get_stats(),
            )

    def get_tool_counts(self) -> Dict[str, int]:
        """Get per-tool request counts."""
        with self._lock:
            return self._tool_counts.copy()

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._request_count = 0
            self._error_count = 0
            self._tool_counts.clear()
        get_latency_tracker().reset()


_global_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return _global_collector


class MetricsExporter:
    """Export metrics to various formats and destinations."""

    @staticmethod
    def to_json(snapshot: MetricsSnapshot) -> str:
        """Export metrics snapshot as JSON string."""
        return json.dumps(snapshot.to_dict(), indent=2)

    @staticmethod
    def to_prometheus(snapshot: MetricsSnapshot) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            Prometheus-formatted metrics string
        """
        lines = [
            "# HELP talk2metadata_uptime_seconds Server uptime in seconds",
            "# TYPE talk2metadata_uptime_seconds gauge",
            f"talk2metadata_uptime_seconds {snapshot.uptime_seconds:.2f}",
            "",
            "# HELP talk2metadata_requests_total Total number of requests",
            "# TYPE talk2metadata_requests_total counter",
            f"talk2metadata_requests_total {snapshot.total_requests}",
            "",
            "# HELP talk2metadata_errors_total Total number of errors",
            "# TYPE talk2metadata_errors_total counter",
            f"talk2metadata_errors_total {snapshot.error_count}",
            "",
            "# HELP talk2metadata_error_rate Current error rate",
            "# TYPE talk2metadata_error_rate gauge",
            f"talk2metadata_error_rate {snapshot.error_rate:.4f}",
            "",
        ]

        for operation, stats in snapshot.latency_stats.items():
            safe_op = operation.replace(".", "_").replace("-", "_")

            lines.extend(
                [
                    f"# HELP talk2metadata_latency_ms_{safe_op} Latency for {operation}",
                    f"# TYPE talk2metadata_latency_ms_{safe_op} summary",
                    f'talk2metadata_latency_ms_{safe_op}{{quantile="0.5"}} {stats["p50_ms"]:.3f}',
                    f'talk2metadata_latency_ms_{safe_op}{{quantile="0.95"}} {stats["p95_ms"]:.3f}',
                    f'talk2metadata_latency_ms_{safe_op}{{quantile="0.99"}} {stats["p99_ms"]:.3f}',
                    f"talk2metadata_latency_ms_{safe_op}_sum {stats['total_ms']:.3f}",
                    f"talk2metadata_latency_ms_{safe_op}_count {stats['count']}",
                    "",
                ]
            )

        return "\n".join(lines)

    @staticmethod
    def write_to_file(
        snapshot: MetricsSnapshot, file_path: Path, format: str = "json"
    ) -> None:
        """
        Write metrics snapshot to file.

        Args:
            snapshot: Metrics snapshot to export
            file_path: Destination file path
            format: Export format ("json" or "prometheus")
        """
        try:
            if format == "json":
                content = MetricsExporter.to_json(snapshot)
            elif format == "prometheus":
                content = MetricsExporter.to_prometheus(snapshot)
            else:
                raise ValueError(f"Unsupported format: {format}")

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            logger.debug(f"Metrics exported to {file_path}")

        except Exception as e:
            logger.error(f"Failed to export metrics to {file_path}: {e}")


def log_slow_query(
    query: str,
    duration_ms: float,
    threshold_ms: float,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log a slow query warning.

    Args:
        query: The query text
        duration_ms: Query duration in milliseconds
        threshold_ms: Threshold that was exceeded
        details: Additional context details
    """
    logger.warning(
        f"Slow query detected: {duration_ms:.1f}ms (threshold: {threshold_ms}ms)",
        extra={
            "event": "slow_query",
            "query": query[:100],  # Truncate for logging
            "duration_ms": round(duration_ms, 3),
            "threshold_ms": threshold_ms,
            **(details or {}),
        },
    )


def log_request_metrics(
    request_id: str,
    tool_name: str,
    duration_ms: float,
    success: bool,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log structured request metrics.

    Args:
        request_id: Unique request identifier
        tool_name: Name of the tool/operation
        duration_ms: Request duration in milliseconds
        success: Whether the request succeeded
        details: Additional context details
    """
    level = "info" if success else "error"
    log_fn = getattr(logger, level)

    try:
        db_path = Path("./data/logs/usage.duckdb")
        db_path.parent.mkdir(parents=True, exist_ok=True)
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
                "INSERT INTO request_metrics VALUES (?, ?, ?, ?, ?, ?)",
                [
                    datetime.fromtimestamp(time.time(), tz=timezone.utc),
                    request_id,
                    tool_name,
                    float(duration_ms),
                    bool(success),
                    json.dumps(details, ensure_ascii=False) if details else None,
                ],
            )
        finally:
            con.close()
    except Exception:
        pass

    log_fn(
        f"Request {request_id}: {tool_name} ({'success' if success else 'failed'})",
        extra={
            "event": "request_metrics",
            "request_id": request_id,
            "tool_name": tool_name,
            "duration_ms": round(duration_ms, 3),
            "success": success,
            **(details or {}),
        },
    )


def _redact_params(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    redacted: dict[str, Any] = {}
    for k, v in value.items():
        key = str(k).lower()
        if any(s in key for s in ("token", "password", "secret", "authorization")):
            redacted[k] = "***"
        else:
            redacted[k] = v
    return redacted


def log_http_request(
    request_id: str,
    route: str,
    path: str,
    method: str,
    status_code: int,
    duration_ms: float,
    success: bool,
    run_id: str | None = None,
    query_text: str | None = None,
    params: Optional[Dict[str, Any]] = None,
    response_json: str | None = None,
) -> None:
    try:
        db_path = Path("./data/logs/usage.duckdb")
        db_path.parent.mkdir(parents=True, exist_ok=True)
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
            cleaned_params = _redact_params(params or {})
            con.execute(
                """
                INSERT INTO http_requests (
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
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    datetime.fromtimestamp(time.time(), tz=timezone.utc),
                    request_id,
                    route,
                    path,
                    method,
                    int(status_code),
                    float(duration_ms),
                    bool(success),
                    run_id,
                    (query_text[:5000] if isinstance(query_text, str) else None),
                    (
                        json.dumps(cleaned_params, ensure_ascii=False)
                        if cleaned_params
                        else None
                    ),
                    (
                        response_json[:100000]
                        if isinstance(response_json, str)
                        else None
                    ),
                ],
            )
        finally:
            con.close()
    except Exception:
        pass
