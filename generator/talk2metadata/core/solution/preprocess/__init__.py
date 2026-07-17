"""Preprocess modules (schema linking, query rewrite, etc.)."""

from __future__ import annotations

from .builtins import register_builtin_preprocessors
from .pipeline import run_preprocess
from .registry import (
    get_preprocessor_registry,
    register_preprocessor,
)

register_builtin_preprocessors()

__all__ = ["get_preprocessor_registry", "register_preprocessor", "run_preprocess"]
