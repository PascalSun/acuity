"""Query preprocess pipeline runner."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from talk2metadata.core.schema.schema import SchemaMetadata
from talk2metadata.utils.config import Config
from talk2metadata.utils.logging import get_logger

from .base import PreprocessContext
from .registry import get_preprocessor

logger = get_logger(__name__)


def _get_mode_block(config: Config, mode_name: str) -> Dict[str, Any]:
    modes_cfg = config.get("modes", {})
    if not isinstance(modes_cfg, dict):
        return {}
    mode_block = modes_cfg.get(mode_name, {})
    return mode_block if isinstance(mode_block, dict) else {}


def _get_preprocess_config(config: Config, mode_name: str) -> Dict[str, Any]:
    mode_block = _get_mode_block(config, mode_name)
    preprocess_cfg = mode_block.get("preprocess")
    if isinstance(preprocess_cfg, dict):
        return preprocess_cfg
    global_cfg = config.get("preprocess", {})
    return global_cfg if isinstance(global_cfg, dict) else {}


def run_preprocess(
    query: str,
    *,
    config: Config,
    mode_name: str,
    run_id: str | None = None,
    schema_metadata: SchemaMetadata | None = None,
) -> Tuple[str, Dict[str, Any]]:
    preprocess_cfg = _get_preprocess_config(config, mode_name)
    enabled = preprocess_cfg.get("enabled", False)
    if not enabled:
        return query, {}

    steps = preprocess_cfg.get("steps", [])
    if not isinstance(steps, list) or not steps:
        return query, {}

    metadata: Dict[str, Any] = {}
    ctx = PreprocessContext(
        config=config,
        mode_name=mode_name,
        run_id=run_id,
        schema_metadata=schema_metadata,
        metadata={},
    )

    current = query
    for step_name in steps:
        if not isinstance(step_name, str) or not step_name:
            continue
        preprocessor_cls = get_preprocessor(step_name)
        if preprocessor_cls is None:
            logger.warning(f"Unknown preprocessor: {step_name}")
            continue
        preprocessor = preprocessor_cls()
        result = preprocessor.preprocess(current, context=ctx)
        current = result.query
        if isinstance(result.metadata, dict) and result.metadata:
            metadata.update(result.metadata)
        ctx = PreprocessContext(
            config=config,
            mode_name=mode_name,
            run_id=run_id,
            schema_metadata=schema_metadata,
            metadata=metadata,
        )

    return current, metadata
