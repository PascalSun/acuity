"""Schema management commands - ingest, validate, visualize."""

from __future__ import annotations

import click

from talk2metadata.cli.decorators import (
    handle_errors,
    with_run_id,
)
from talk2metadata.cli.handlers import IngestHandler, SchemaHandler
from talk2metadata.cli.utils import CLIDataLoader, resolve_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


@click.group(name="schema")
def schema_group():
    """Schema management commands for data ingestion and validation.

    This command group provides tools for ingesting data, validating schema,
    and visualizing database relationships.
    """
    pass


@schema_group.command(name="ingest")
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to run config YAML (default: configs/{run_id}.yml when --run-id given)",
)
@with_run_id
@handle_errors
@click.pass_context
def ingest_cmd(ctx, config, run_id):  # noqa: ARG001
    """Ingest data from CSV files or database.

    All configuration is read from the run config YAML file.

    \b
    Examples:
        talk2metadata schema ingest --run-id wamex
        talk2metadata schema ingest --config configs/wamex.yml
    """
    cfg = resolve_config(config, run_id)
    run_id = run_id or cfg.get("run_id")

    if not run_id:
        logger.error("You must specify a run_id")
        raise click.Abort()

    source_type = cfg.get("ingest.data_type")
    source_path = cfg.get("ingest.source_path")
    target_table = cfg.get("ingest.target_table")
    visualize = cfg.get("ingest.visualize", True)
    skip_validation = cfg.get("ingest.skip_validation", False)

    # Validate required config parameters
    if source_type is None:
        logger.error("'ingest.data_type' must be set in the run config")
        raise click.Abort()
    if source_path is None:
        logger.error("'ingest.source_path' must be set in the run config")
        raise click.Abort()
    if target_table is None:
        logger.error("'ingest.target_table' must be set in the run config")
        raise click.Abort()

    logger.info(f"Ingesting from {source_type}: {source_path}")
    logger.info(f"Target table: {target_table}")

    # Initialize handler
    handler = IngestHandler(cfg)

    # 1. Create connector and load tables
    logger.info("Loading tables...")
    try:
        connector = handler.create_connector(source_type, source_path, target_table)
        tables = connector.load_tables()

        logger.info(f"Loaded {len(tables)} tables:")
        for name, df in tables.items():
            logger.info(f"   ✓ {name}: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"Failed to load tables: {e}")
        raise click.Abort()

    # 2. Detect schema and FKs
    logger.info("Detecting schema and foreign keys...")
    try:
        metadata = handler.detect_schema(tables, target_table)

        logger.info("Schema detection complete:")
        logger.info(f"  Tables: {len(metadata.tables)}")
        logger.info(f"  Foreign keys: {len(metadata.foreign_keys)}")

        if metadata.foreign_keys:
            logger.info("   Foreign key relationships:")
            for fk in metadata.foreign_keys:
                coverage_icon = "✓" if fk.coverage >= 0.9 else "⚠"
                msg = (
                    f"     {coverage_icon} {fk.child_table}.{fk.child_column} → "
                    f"{fk.parent_table}.{fk.parent_column} (coverage: {fk.coverage:.1%})"
                )
                logger.info(msg)
    except Exception as e:
        logger.error(f"Schema detection failed: {e}")
        raise click.Abort()

    # 3. Validate schema (unless skipped in config)
    if not skip_validation:
        logger.info("🔍 Validating schema...")
        validation_result = handler.validate_schema_metadata(metadata)

        if validation_result["errors"]:
            logger.info("❌ Schema validation found errors:")
            for err in validation_result["errors"]:
                logger.info(f"   - {err}")
            logger.warning("Schema has errors. Please review and fix before indexing.")
            logger.info(
                "Use 'talk2metadata schema validate --edit' to modify the schema file."
            )

            if not click.confirm("\n   Continue anyway?", default=False):
                raise click.Abort()

        if validation_result["warnings"]:
            logger.info("⚠️  Schema validation warnings:")
            for w in validation_result["warnings"]:
                logger.info(f"   - {w}")

        if not validation_result["errors"] and not validation_result["warnings"]:
            logger.info("Schema validation passed!")

    # 4. Save metadata
    logger.info("💾 Saving metadata...")
    try:
        metadata_path = handler.save_metadata(metadata, run_id=run_id)
        logger.info(f"Metadata saved to {metadata_path}")
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")
        raise click.Abort()

    # 5. Generate visualization if enabled in config
    if visualize:
        logger.info("🎨 Generating schema visualization...")
        try:
            viz_path = handler.generate_visualization(metadata, run_id=run_id)
            logger.info(f"Visualization saved to {viz_path}")
            logger.info(f"   Open in browser: file://{viz_path.absolute()}")
        except Exception as e:
            logger.warning(f"Failed to generate visualization: {e}")

    # 6. Save tables for indexing
    logger.info("💾 Saving processed tables...")
    try:
        tables_path = handler.save_tables(tables, run_id)
        logger.info(f"Tables saved to {tables_path}")
    except Exception as e:
        logger.error(f"Failed to save tables: {e}")
        raise click.Abort()

    # Success
    logger.info("✅ Ingestion complete!")
    next_steps = ["Review schema: talk2metadata schema validate"]
    if not visualize:
        next_steps.append("Visualize schema: talk2metadata schema visualize")
    next_steps.append("Prepare modes: talk2metadata search prepare")
    logger.info("Next steps:")
    for step in next_steps:
        logger.info(f"   - {step}")


@schema_group.command(name="validate")
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to run config YAML (e.g., configs/wamex.yml)",
)
@with_run_id
@handle_errors
@click.pass_context
def validate_cmd(ctx, run_id, config):  # noqa: ARG001
    """Validate schema metadata and check for errors.

    Loads the schema produced by ingest, shows summary and validation
    results. Edit the schema file manually if needed, then re-run validate
    or proceed to search prepare.

    \b
    Examples:
        # Validate schema (after ingest)
        talk2metadata schema validate --config configs/wamex.yml

        # With run-id (loads configs/{run_id}.yml)
        talk2metadata schema validate --run-id wamex
    """
    cfg = resolve_config(config, run_id)
    run_id = run_id or cfg.get("run_id")
    if not run_id:
        logger.error("You must specify a run_id")
        raise click.Abort()

    # Load schema (from ingest output)
    loader = CLIDataLoader(cfg)
    schema = loader.load_schema(run_id=run_id)

    # Display schema summary
    handler = SchemaHandler(cfg)
    summary = handler.get_schema_summary(schema)

    logger.info("📊 Schema Summary:")
    logger.info(f"   Target Table: {summary['target_table']}")
    logger.info(f"   Tables: {summary['num_tables']}")
    logger.info(f"   Foreign Keys: {summary['num_foreign_keys']}")

    if schema.tables:
        logger.info("   Tables:")
        for name, meta in schema.tables.items():
            is_target = " (target)" if name == schema.target_table else ""
            logger.info(
                f"     - {name}{is_target}: {meta.row_count} rows, "
                f"{len(meta.columns)} columns, PK={meta.primary_key or 'None'}"
            )

    if schema.foreign_keys:
        logger.info("   Foreign Keys:")
        for fk in schema.foreign_keys:
            coverage_icon = "✓" if fk.coverage >= 0.9 else "⚠"
            msg = (
                f"     {coverage_icon} {fk.child_table}.{fk.child_column} → "
                f"{fk.parent_table}.{fk.parent_column} (coverage: {fk.coverage:.1%})"
            )
            logger.info(msg)

    # Validate schema
    logger.info("🔍 Validating schema...")
    validation_result = handler.validate(schema)

    if validation_result["errors"]:
        logger.info("❌ Errors found:")
        for err in validation_result["errors"]:
            logger.info(f"   - {err}")

    if validation_result["warnings"]:
        logger.info("⚠️  Warnings:")
        for w in validation_result["warnings"]:
            logger.info(f"   - {w}")

    if not validation_result["errors"] and not validation_result["warnings"]:
        logger.info("Schema validation passed with no errors or warnings!")


@schema_group.command(name="visualize")
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to run config YAML (e.g., configs/wamex.yml)",
)
@with_run_id
@handle_errors
@click.pass_context
def visualize_cmd(ctx, run_id, config):  # noqa: ARG001
    """Generate HTML visualization of schema relationships.

    All configuration is read from the run config YAML file.

    \b
    Examples:
        # Generate visualization
        talk2metadata schema visualize

        # Generate with a specific config file
        talk2metadata schema visualize --config configs/wamex.yml
    """
    cfg = resolve_config(config, run_id)
    run_id = run_id or cfg.get("run_id")
    if not run_id:
        logger.error("You must specify a run_id")
        raise click.Abort()

    # Load schema
    loader = CLIDataLoader(cfg)
    schema = loader.load_schema(run_id=run_id)

    # Get schema path
    from talk2metadata.utils.paths import find_schema_file, get_metadata_dir

    metadata_dir = get_metadata_dir(run_id, cfg)
    target_table = cfg.get("ingest.target_table")
    schema_path = find_schema_file(metadata_dir, target_table=target_table)

    # Generate visualization
    logger.info("🎨 Generating visualization...")
    handler = SchemaHandler(cfg)
    try:
        viz_path = handler.generate_visualization(
            schema=schema,
            schema_path=schema_path,
        )
        logger.info(f"Visualization saved to {viz_path}")
        logger.info(f"   Open in browser: file://{viz_path.absolute()}")
    except Exception as e:
        logger.error(f"Failed to generate visualization: {e}")
        raise click.Abort()
