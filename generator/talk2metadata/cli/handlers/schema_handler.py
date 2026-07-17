"""Business logic for schema commands."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from talk2metadata.core.schema import SchemaMetadata
from talk2metadata.core.schema.schema_viz import (
    generate_html_visualization,
    validate_schema,
)
from talk2metadata.utils.config import Config


class SchemaHandler:
    """Handler for schema operations.

    Encapsulates business logic for schema commands,
    keeping CLI commands thin and focused on user interaction.

    Example:
        >>> handler = SchemaHandler(config)
        >>> validation_result = handler.validate(schema)
    """

    def __init__(self, config: Config):
        """Initialize handler.

        Args:
            config: Configuration instance
        """
        self.config = config

    def validate(self, schema: SchemaMetadata) -> Dict[str, list]:
        """Validate schema and return errors/warnings.

        Args:
            schema: Schema metadata to validate

        Returns:
            Dictionary with 'errors' and 'warnings' keys
        """
        return validate_schema(schema)

    def generate_visualization(
        self,
        schema: SchemaMetadata,
        output_file: Optional[str] = None,
        schema_path: Optional[Path] = None,
    ) -> Path:
        """Generate HTML visualization of schema.

        Args:
            schema: Schema metadata
            output_file: Optional output file path
            schema_path: Optional path to schema file (used for default naming)

        Returns:
            Path to generated HTML file
        """
        import re

        # Determine output path
        if output_file:
            viz_path = Path(output_file)
        else:
            # Generate filename with target table name
            target_table_safe = re.sub(r"[^\w\-_.]", "_", schema.target_table)
            if schema_path:
                viz_path = (
                    schema_path.parent
                    / f"schema_visualization_{target_table_safe}.html"
                )
            else:
                viz_path = Path(f"schema_visualization_{target_table_safe}.html")

        generate_html_visualization(schema, viz_path)
        return viz_path

    def get_schema_summary(self, schema: SchemaMetadata) -> Dict:
        """Get summary statistics for schema.

        Args:
            schema: Schema metadata

        Returns:
            Dictionary with summary statistics
        """
        return {
            "target_table": schema.target_table,
            "num_tables": len(schema.tables),
            "num_foreign_keys": len(schema.foreign_keys),
            "tables": {
                name: {
                    "row_count": meta.row_count,
                    "num_columns": len(meta.columns),
                    "primary_key": meta.primary_key,
                }
                for name, meta in schema.tables.items()
            },
        }
