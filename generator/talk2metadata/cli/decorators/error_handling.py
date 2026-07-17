"""Error handling decorators for CLI commands."""

from __future__ import annotations

import signal
import sys
from functools import wraps

import click

from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)

# Handle SIGPIPE gracefully (prevent BrokenPipeError when piping to head, etc.)
try:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
except AttributeError:
    # Windows doesn't have SIGPIPE
    pass


def handle_errors(f):
    """Decorator to handle common errors in CLI commands.

    Catches exceptions and displays user-friendly error messages,
    then aborts the command gracefully.

    Example:
        @click.command()
        @handle_errors
        def my_command():
            # Your command logic
            pass
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except click.Abort:
            # Re-raise abort to let Click handle it
            raise
        except BrokenPipeError:
            # Handle broken pipe (e.g., piping to head)
            # Close stdout/stderr to avoid further errors
            devnull = open("/dev/null", "w")
            sys.stdout = devnull
            sys.stderr = devnull
            sys.exit(0)
        except FileNotFoundError as e:
            click.echo(f"❌ File not found: {e}", err=True)
            raise click.Abort()
        except PermissionError as e:
            click.echo(f"❌ Permission denied: {e}", err=True)
            raise click.Abort()
        except ValueError as e:
            click.echo(f"❌ Invalid value: {e}", err=True)
            logger.debug("ValueError details", exc_info=True)
            raise click.Abort()
        except KeyError as e:
            click.echo(f"❌ Missing key: {e}", err=True)
            logger.debug("KeyError details", exc_info=True)
            raise click.Abort()
        except Exception as e:
            click.echo(f"❌ Unexpected error: {e}", err=True)
            logger.exception("Unexpected error in command")
            raise click.Abort()

    return wrapper


def require_schema(f):
    """Decorator to ensure schema is loaded before command runs.

    This decorator checks if a schema file exists in the expected location.
    If not, it displays an error and aborts.

    Example:
        @click.command()
        @require_schema
        def my_command():
            # Schema is guaranteed to exist
            pass
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        from talk2metadata.utils.config import get_config
        from talk2metadata.utils.paths import find_schema_file, get_metadata_dir

        config = get_config()
        run_id = config.get("run_id")

        try:
            metadata_dir = get_metadata_dir(run_id, config)
            schema_path = find_schema_file(metadata_dir)

            if not schema_path.exists():
                click.echo(f"❌ Schema file not found: {schema_path}", err=True)
                click.echo(
                    "   Run 'talk2metadata ingest' first to generate schema.", err=True
                )
                raise click.Abort()

        except FileNotFoundError:
            click.echo("❌ No schema file found", err=True)
            click.echo(
                "   Run 'talk2metadata ingest' first to generate schema.", err=True
            )
            raise click.Abort()

        return f(*args, **kwargs)

    return wrapper
