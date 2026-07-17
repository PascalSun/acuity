"""Registry for preprocessors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Type

from .base import BasePreprocessor


@dataclass(frozen=True)
class PreprocessorInfo:
    name: str
    preprocessor_class: Type[BasePreprocessor]


class PreprocessorRegistry:
    def __init__(self) -> None:
        self._items: Dict[str, PreprocessorInfo] = {}

    def register(self, name: str, preprocessor_class: Type[BasePreprocessor]) -> None:
        self._items[name] = PreprocessorInfo(
            name=name, preprocessor_class=preprocessor_class
        )

    def get(self, name: str) -> Optional[PreprocessorInfo]:
        return self._items.get(name)

    def list(self) -> Dict[str, PreprocessorInfo]:
        return self._items.copy()


_registry = PreprocessorRegistry()


def register_preprocessor(
    name: str, preprocessor_class: Type[BasePreprocessor]
) -> None:
    _registry.register(name, preprocessor_class)


def get_preprocessor(name: str) -> Optional[Type[BasePreprocessor]]:
    info = _registry.get(name)
    return info.preprocessor_class if info else None


def get_preprocessor_registry() -> PreprocessorRegistry:
    return _registry
