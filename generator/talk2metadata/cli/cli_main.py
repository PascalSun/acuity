"""CLI entry point for Talk2Metadata."""

from __future__ import annotations

import click

from talk2metadata import __version__

# Import command groups
from talk2metadata.cli.commands import (
    analysis_group,
    qa,
    schema_group,
    search_group,
    utils_group,
)
from talk2metadata.utils.config import load_config
from talk2metadata.utils.logging import setup_logging


@click.group()
@click.version_option(version=__version__)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to run config YAML (e.g., configs/wamex.yml)",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Logging level",
)
@click.pass_context
def cli(ctx, config, log_level):
    """Talk2Metadata - Question-driven multi-table record retrieval.

    \b
    Examples:
        # Ingest CSV files
        talk2metadata schema ingest csv ./data/csv --target orders

        # Prepare modes (build indexes or load databases)
        talk2metadata search prepare

        # Search for records
        talk2metadata search retrieve "customers in healthcare industry"

        # Generate QA pairs
        talk2metadata qa generate
    """
    ctx.ensure_object(dict)

    # Setup logging
    setup_logging(level=log_level)

    # Load config if provided
    if config:
        ctx.obj["config"] = load_config(config)


# Register command groups
cli.add_command(schema_group.schema_group)
cli.add_command(search_group.search_group)
cli.add_command(qa.qa_group)
cli.add_command(utils_group.utils_group)
cli.add_command(analysis_group.analysis_group)


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
