"""Mode registry for managing different indexing and retrieval strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from talk2metadata.core.schema.schema import SchemaMetadata
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModeInfo:
    """Information about a mode."""

    name: str
    description: str
    indexer_class: Type
    retriever_class: Type
    enabled: bool = True


class BaseIndexer(ABC):
    """Base class for mode-specific indexers."""

    @abstractmethod
    def build_index(
        self, tables: Dict, schema_metadata: SchemaMetadata, **kwargs
    ) -> Any:
        """Build index for this mode."""
        pass


class BaseRetriever(ABC):
    """Base class for mode-specific retrievers."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[Any]:
        """Search using this mode."""
        pass


class ModeRegistry:
    """Registry for indexing and retrieval modes."""

    def __init__(self):
        self._modes: Dict[str, ModeInfo] = {}
        self._active_mode: Optional[str] = None

    def register(
        self,
        name: str,
        description: str,
        indexer_class: Type[BaseIndexer],
        retriever_class: Type[BaseRetriever],
        enabled: bool = True,
    ) -> None:
        """Register a new mode.

        Args:
            name: Mode name (e.g., "semantic")
            description: Human-readable description
            indexer_class: Indexer class for this mode
            retriever_class: Retriever class for this mode
            enabled: Whether this mode is enabled
        """
        self._modes[name] = ModeInfo(
            name=name,
            description=description,
            indexer_class=indexer_class,
            retriever_class=retriever_class,
            enabled=enabled,
        )
        logger.info(f"Registered mode: {name}")

    def get(self, name: str) -> Optional[ModeInfo]:
        """Get mode information by name.

        Resolves mode aliases from config: if name is not registered but exists
        in modes config with a 'base' key, returns the base mode's ModeInfo.
        This enables variants like text2sql.openai52, text2sql.gemini for
        comparing different LLM text2sql capabilities.

        Args:
            name: Mode name (e.g., "text2sql.openai52" or "semantic")

        Returns:
            ModeInfo or None if not found
        """
        # Direct lookup
        info = self._modes.get(name)
        if info is not None:
            return info

        # Resolve alias: check config for base mode
        config = get_config()
        modes_cfg = config.get("modes", {})
        mode_config = modes_cfg.get(name) if isinstance(modes_cfg, dict) else None
        if isinstance(mode_config, dict) and mode_config.get("base"):
            base_name = mode_config["base"]
            base_info = self._modes.get(base_name)
            if base_info is not None:
                return base_info

        return None

    def get_active(self) -> Optional[str]:
        """Get active mode name from config.

        Returns:
            Active mode name or None
        """
        if self._active_mode:
            return self._active_mode

        config = get_config()
        modes_cfg = config.get("modes", {})
        if isinstance(modes_cfg, dict):
            active = modes_cfg.get("active", "semantic")
        else:
            active = "semantic"
        # Support both registered modes and config aliases (base)
        if self.get(active) and self.get(active).enabled:
            return active
        return None

    def set_active(self, name: str) -> None:
        """Set active mode.

        Args:
            name: Mode name to activate
        """
        if name not in self._modes:
            raise ValueError(f"Mode '{name}' not registered")
        if not self._modes[name].enabled:
            raise ValueError(f"Mode '{name}' is disabled")
        self._active_mode = name
        logger.info(f"Set active mode to: {name}")

    def list_modes(self, enabled_only: bool = False) -> Dict[str, ModeInfo]:
        """List all registered modes.

        Args:
            enabled_only: If True, only return enabled modes

        Returns:
            Dict mapping mode name -> ModeInfo
        """
        if enabled_only:
            return {name: info for name, info in self._modes.items() if info.enabled}
        return self._modes.copy()

    def get_all_enabled(self) -> list[str]:
        """Get list of all enabled mode names.

        Returns:
            List of enabled mode names
        """
        return [name for name, info in self._modes.items() if info.enabled]


# Global registry instance
_registry = ModeRegistry()
_modes_registered = False


def _ensure_modes_registered() -> None:
    """Lazily import and register modes. Avoids loading heavy mode deps at CLI startup."""
    global _modes_registered
    if _modes_registered:
        return
    from .. import hybrid  # noqa: F401
    from . import graph  # noqa: F401
    from . import lexical  # noqa: F401
    from . import semantic  # noqa: F401
    from . import text2sql  # noqa: F401

    _modes_registered = True


def register_mode(
    name: str,
    description: str,
    indexer_class: Type[BaseIndexer],
    retriever_class: Type[BaseRetriever],
    enabled: bool = True,
) -> None:
    """Register a mode in the global registry.

    Args:
        name: Mode name
        description: Mode description
        indexer_class: Indexer class
        retriever_class: Retriever class
        enabled: Whether enabled
    """
    _registry.register(name, description, indexer_class, retriever_class, enabled)


def get_mode(name: str) -> Optional[ModeInfo]:
    """Get mode information.

    Args:
        name: Mode name

    Returns:
        ModeInfo or None
    """
    _ensure_modes_registered()
    return _registry.get(name)


def get_active_mode() -> Optional[str]:
    """Get active mode name.

    Returns:
        Active mode name or None
    """
    _ensure_modes_registered()
    return _registry.get_active()


def get_registry() -> ModeRegistry:
    """Get the global mode registry.

    Returns:
        ModeRegistry instance
    """
    _ensure_modes_registered()
    return _registry


def get_mode_config(mode_name: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a specific mode.

    Supports mode aliases: if the mode has a 'base' key, it inherits from that
    base mode (e.g., text2sql.openai52 with base: text2sql.two_step).
    Mode config can also include 'agent' for per-mode LLM overrides.

    Args:
        mode_name: Mode name

    Returns:
        Mode configuration dict with 'indexer', 'retriever', 'base', and/or
        'agent' keys, or None
    """
    config = get_config()
    modes_cfg = config.get("modes", {})
    mode_config = modes_cfg.get(mode_name) if isinstance(modes_cfg, dict) else None
    if not isinstance(mode_config, dict):
        return None
    # Valid if it has any mode-specific config
    has_config = (
        "indexer" in mode_config
        or "retriever" in mode_config
        or "base" in mode_config
        or "agent" in mode_config
    )
    return mode_config if has_config else None


def get_mode_indexer_config(mode_name: str) -> Dict[str, Any]:
    """Get indexer configuration for a specific mode.

    Args:
        mode_name: Mode name

    Returns:
        Indexer configuration dict with defaults
    """
    mode_config = get_mode_config(mode_name)
    if mode_config and "indexer" in mode_config:
        indexer_config = mode_config["indexer"]
        if isinstance(indexer_config, dict):
            return indexer_config

    # Default configuration if mode-specific config not found
    return {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "device": None,
        "batch_size": 32,
        "normalize": True,
    }


def get_mode_retriever_config(mode_name: str) -> Dict[str, Any]:
    """Get retriever configuration for a specific mode.

    Merges retriever config from mode config and base mode (for aliases).

    Args:
        mode_name: Mode name

    Returns:
        Retriever configuration dict (falls back to global retrieval config)
    """
    mode_config = get_mode_config(mode_name)
    result = {}
    # For aliases, merge base mode's retriever config first
    if mode_config and mode_config.get("base"):
        base_config = get_mode_config(mode_config["base"])
        if base_config and "retriever" in base_config:
            base_retriever = base_config.get("retriever")
            if isinstance(base_retriever, dict):
                result.update(base_retriever)
    if mode_config and "retriever" in mode_config:
        retriever_config = mode_config["retriever"]
        if isinstance(retriever_config, dict):
            result.update(retriever_config)
    if result:
        return result

    # Default configuration if mode-specific config not found
    return {
        "top_k": 5,
        "similarity_metric": "cosine",
        "per_table_top_k": 5,
        "use_reranking": False,
    }


def resolve_index_mode(mode_name: str) -> str:
    """Resolve mode name to the one used for index directory.

    For aliases (modes with 'base' in config), returns the base mode name
    since aliases share the same index as their base (e.g., text2sql.openai52
    and text2sql.gemini both use text2sql.two_step index).

    Args:
        mode_name: Mode name (can be alias)

    Returns:
        Mode name for index path (base for aliases, else mode_name)
    """
    mode_config = get_mode_config(mode_name)
    if mode_config and mode_config.get("base"):
        return mode_config["base"]
    return mode_name
