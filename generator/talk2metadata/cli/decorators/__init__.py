"""CLI decorators for common options and error handling."""

from talk2metadata.cli.decorators.error_handling import handle_errors, require_schema
from talk2metadata.cli.decorators.options import (
    with_agent_config,
    with_config,
    with_output_file,
    with_run_id,
    with_schema_file,
    with_standard_options,
)

__all__ = [
    "handle_errors",
    "require_schema",
    "with_config",
    "with_agent_config",
    "with_output_file",
    "with_run_id",
    "with_schema_file",
    "with_standard_options",
]
