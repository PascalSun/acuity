"""Logging utilities for Talk2Metadata using loguru."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from loguru import logger


# Intercept standard library logging and route it through loguru
class _InterceptHandler(logging.Handler):
    """Redirect stdlib logging records to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        # Find the loguru level that matches
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find the caller frame (skip this handler + logging internals)
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """Setup logging configuration using loguru.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        format_string: Optional custom format string (loguru syntax)
    """
    # Remove any existing loguru handlers
    logger.remove()

    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

    # Console sink with colors
    logger.add(
        sys.stderr,
        format=format_string,
        level=level.upper(),
        colorize=True,
    )

    # File sink (no colors)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format=format_string,
            level=level.upper(),
            colorize=False,
            rotation="10 MB",
        )

    # Intercept stdlib logging so third-party libraries also go through loguru
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)


def get_logger(name: str) -> logger.__class__:
    """Get a loguru logger bound to a module name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Loguru logger instance bound with the given name
    """
    return logger.bind(name=name)
