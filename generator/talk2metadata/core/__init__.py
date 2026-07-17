"""Core modules for Talk2Metadata."""

from talk2metadata.core.schema import (
    ForeignKey,
    SchemaDetector,
    SchemaMetadata,
    TableMetadata,
    generate_html_visualization,
    validate_schema,
)

__all__ = [
    # Schema
    "ForeignKey",
    "SchemaDetector",
    "SchemaMetadata",
    "TableMetadata",
    "generate_html_visualization",
    "validate_schema",
]
