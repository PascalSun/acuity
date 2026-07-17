"""Built-in preprocessors."""

from __future__ import annotations

import re

from .base import (
    BasePreprocessor,
    PreprocessContext,
    PreprocessResult,
)
from .registry import register_preprocessor


class IdentityPreprocessor(BasePreprocessor):
    def preprocess(self, query: str, *, context: PreprocessContext) -> PreprocessResult:
        return PreprocessResult(query=query, metadata={})


class NormalizeWhitespacePreprocessor(BasePreprocessor):
    def preprocess(self, query: str, *, context: PreprocessContext) -> PreprocessResult:
        normalized = re.sub(r"\s+", " ", query).strip()
        return PreprocessResult(query=normalized, metadata={})


def register_builtin_preprocessors() -> None:
    register_preprocessor("identity", IdentityPreprocessor)
    register_preprocessor("normalize_whitespace", NormalizeWhitespacePreprocessor)
