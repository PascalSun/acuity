"""CLI utility modules."""

from __future__ import annotations

import os
from pathlib import Path

import click

from talk2metadata.cli.utils.loaders import CLIDataLoader
from talk2metadata.utils.config import Config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


def get_yaml_config(config_path: str | None = None) -> Config:
    """Load Config from YAML file(s).

    When loading a run config (e.g. configs/wamex.yml), merges with config.yml
    so agent.keys and provider settings are included. Same logic as Config.from_yaml.

    Args:
        config_path: Explicit path to YAML config file (from ``--config`` CLI option).

    Resolution order:
    1. ``config_path`` argument
    2. ``TALK2METADATA_CONFIG`` environment variable
    3. ``./config.yml`` in the current working directory
    """
    config_path = config_path or os.getenv("TALK2METADATA_CONFIG")
    if not config_path and Path("config.yml").exists():
        config_path = "config.yml"
    if not config_path:
        logger.error(
            "No YAML config found: pass --config, set TALK2METADATA_CONFIG, or provide ./config.yml"
        )
        raise click.Abort()

    path = Path(config_path)
    if path.suffix.lower() not in {".yml", ".yaml"}:
        logger.error(f"Config file must be YAML (.yml/.yaml): {path}")
        raise click.Abort()

    if not path.exists():
        logger.error(f"Config file does not exist: {path}")
        raise click.Abort()

    return Config.from_yaml(path)


def resolve_config(config: str | None, run_id: str | None) -> Config:
    """Resolve config: explicit path, or configs/{run_id}.yml when run_id given.

    Use from commands that accept either --config <path> or --run-id <id>
    (loads configs/{run_id}.yml).
    """
    if config:
        return get_yaml_config(config)
    if run_id:
        for ext in (".yml", ".yaml"):
            path = Path("configs") / f"{run_id}{ext}"
            if path.exists():
                return get_yaml_config(str(path))
        logger.error(
            f"No config found for run_id '{run_id}'. Expected configs/{run_id}.yml"
        )
        raise click.Abort()
    logger.error(
        "Provide --config <path> or --run-id <id> (loads configs/{run_id}.yml)"
    )
    raise click.Abort()


__all__ = ["CLIDataLoader", "get_yaml_config", "resolve_config"]
