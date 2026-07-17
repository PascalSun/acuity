"""Timing utilities for performance monitoring and latency tracking."""

import functools
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TimingStat:
    """Statistics for a timed operation."""

    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    timings: List[float] = field(default_factory=list)

    def add(self, duration_ms: float) -> None:
        """Add a timing measurement."""
        self.count += 1
        self.total_ms += duration_ms
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)
        self.timings.append(duration_ms)

    @property
    def mean_ms(self) -> float:
        """Calculate mean latency."""
        return self.total_ms / self.count if self.count > 0 else 0.0

    def percentile(self, p: float) -> float:
        """Calculate percentile (p in [0, 100])."""
        if not self.timings:
            return 0.0
        sorted_timings = sorted(self.timings)
        idx = int(len(sorted_timings) * p / 100)
        idx = min(idx, len(sorted_timings) - 1)
        return sorted_timings[idx]

    @property
    def p50_ms(self) -> float:
        """Median latency."""
        return self.percentile(50)

    @property
    def p95_ms(self) -> float:
        """95th percentile latency."""
        return self.percentile(95)

    @property
    def p99_ms(self) -> float:
        """99th percentile latency."""
        return self.percentile(99)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "count": self.count,
            "total_ms": round(self.total_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "min_ms": round(self.min_ms, 3) if self.min_ms != float("inf") else 0.0,
            "max_ms": round(self.max_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
        }


class LatencyTracker:
    """Thread-safe tracker for latency metrics across components."""

    def __init__(self, window_size: Optional[int] = None):
        """
        Initialize latency tracker.

        Args:
            window_size: Maximum number of timings to keep per operation.
                        If None, keeps all timings.
        """
        self._stats: Dict[str, TimingStat] = defaultdict(TimingStat)
        self._lock = Lock()
        self._window_size = window_size
        self._start_time = time.time()

    def record(self, operation: str, duration_ms: float) -> None:
        """
        Record a timing measurement.

        Args:
            operation: Name of the operation (e.g., "query_encoding")
            duration_ms: Duration in milliseconds
        """
        with self._lock:
            stat = self._stats[operation]
            stat.add(duration_ms)

            # Trim old timings if window size is set
            if self._window_size and len(stat.timings) > self._window_size:
                stat.timings = stat.timings[-self._window_size :]

    def get_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for operation(s).

        Args:
            operation: Specific operation name, or None for all operations

        Returns:
            Dictionary of statistics
        """
        with self._lock:
            if operation:
                return {operation: self._stats[operation].to_dict()}
            return {op: stat.to_dict() for op, stat in self._stats.items()}

    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._stats.clear()
            self._start_time = time.time()

    def get_uptime_seconds(self) -> float:
        """Get uptime since tracker initialization."""
        return time.time() - self._start_time


# Global latency tracker instance
_global_tracker = LatencyTracker(window_size=10000)


def get_latency_tracker() -> LatencyTracker:
    """Get the global latency tracker instance."""
    return _global_tracker


@contextmanager
def TimingContext(operation: str, log_level: str = "debug"):
    """
    Context manager for timing a block of code.

    Example:
        with TimingContext("database_query"):
            result = db.query(...)

    Args:
        operation: Name of the operation being timed
        log_level: Logging level for the timing message
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        _global_tracker.record(operation, duration_ms)

        # Log the timing
        log_fn = getattr(logger, log_level, logger.debug)
        log_fn(f"{operation} completed in {duration_ms:.3f}ms")


def timed(operation: Optional[str] = None, log_level: str = "debug"):
    """
    Decorator for timing function execution.

    Example:
        @timed("expensive_operation")
        def process_data():
            ...

    Args:
        operation: Name of the operation (defaults to function name)
        log_level: Logging level for the timing message
    """

    def decorator(func: Callable) -> Callable:
        op_name = operation or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                _global_tracker.record(op_name, duration_ms)

                # Log the timing
                log_fn = getattr(logger, log_level, logger.debug)
                log_fn(f"{op_name} completed in {duration_ms:.3f}ms")

        return wrapper

    return decorator


@contextmanager
def RequestTimingContext(request_id: str):
    """
    Context manager for tracking end-to-end request timing with structured logging.

    Example:
        with RequestTimingContext("req_123") as ctx:
            ctx["query"] = "customer search"
            result = process_request()
            ctx["results_count"] = len(result)

    Args:
        request_id: Unique request identifier
    """
    start_time = time.perf_counter()
    context_data = {"request_id": request_id}

    try:
        yield context_data
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        _global_tracker.record("request_total", duration_ms)

        # Log structured request completion
        logger.info(
            "Request completed",
            extra={
                "event": "request_completed",
                "request_id": request_id,
                "duration_ms": round(duration_ms, 3),
                **context_data,
            },
        )
