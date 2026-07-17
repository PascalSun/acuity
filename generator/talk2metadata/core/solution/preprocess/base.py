"""Preprocess interfaces for query-time transformations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

from talk2metadata.core.schema.schema import SchemaMetadata
from talk2metadata.utils.config import Config


@dataclass(frozen=True)
class PreprocessContext:
    config: Config
    mode_name: str
    run_id: str | None = None
    schema_metadata: SchemaMetadata | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreprocessResult:
    query: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BasePreprocessor(ABC):
    @abstractmethod
    def preprocess(
        self, query: str, *, context: PreprocessContext
    ) -> PreprocessResult: ...
