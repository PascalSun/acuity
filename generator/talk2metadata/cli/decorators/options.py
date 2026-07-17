"""Common CLI option decorators."""

from __future__ import annotations

from functools import wraps

import click


def _load_config_callback(
    ctx: click.Context, param: click.Parameter, value: str | None
):
    if value:
        from talk2metadata.utils.config import load_config

        load_config(value)
        ctx.ensure_object(dict)
        ctx.obj["config_path"] = value
    return value


def with_config(f):
    return click.option(
        "--config",
        type=click.Path(exists=True),
        help="Path to run config YAML (e.g., configs/wamex.yml)",
        is_eager=True,
        callback=_load_config_callback,
        expose_value=False,
    )(f)


def with_run_id(f):
    """Add --run-id option to command.

    Example:
        @click.command()
        @with_run_id
        def my_command(run_id):
            pass
    """
    return click.option(
        "--run-id",
        type=str,
        help="Run ID for organizing multiple runs",
    )(f)


def with_schema_file(f):
    """Add --schema-file option to command.

    Example:
        @click.command()
        @with_schema_file
        def my_command(schema_file):
            pass
    """
    return click.option(
        "--schema-file",
        "-s",
        type=click.Path(exists=True),
        help="Path to schema JSON file (default: auto-detected from run_id)",
    )(f)


def with_output_file(f):
    """Add --output option to command.

    Example:
        @click.command()
        @with_output_file
        def my_command(output):
            pass
    """
    return click.option(
        "--output",
        "-o",
        type=click.Path(),
        help="Output file path",
    )(f)


def with_agent_config(f):
    """Add agent configuration options to command.

    Example:
        @click.command()
        @with_agent_config
        def my_command(provider, model):
            pass
    """

    @click.option(
        "--provider",
        type=str,
        help="LLM provider (openai, anthropic, gemini, etc.)",
    )
    @click.option(
        "--model",
        type=str,
        help="Model name (e.g., gpt-4o-mini, claude-3-5-sonnet-20241022)",
    )
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


def with_standard_options(f):
    """Add standard options: run-id, schema-file, output.

    Convenience decorator that combines multiple common options.

    Example:
        @click.command()
        @with_standard_options
        def my_command(run_id, schema_file, output):
            pass
    """
    f = with_output_file(f)
    f = with_schema_file(f)
    f = with_run_id(f)
    return f


def with_data_dir(f):
    """Add --data-dir option to command.

    Example:
        @click.command()
        @with_data_dir
        def my_command(data_dir):
            pass
    """
    return click.option(
        "--data-dir",
        type=click.Path(exists=True),
        help="Directory containing data files",
    )(f)
