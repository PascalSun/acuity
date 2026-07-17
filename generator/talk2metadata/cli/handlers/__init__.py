"""CLI command handlers containing business logic."""

from talk2metadata.cli.handlers.evaluation_handler import EvaluationHandler
from talk2metadata.cli.handlers.index_handler import IndexHandler
from talk2metadata.cli.handlers.ingest_handler import IngestHandler
from talk2metadata.cli.handlers.prepare_handler import PrepareHandler
from talk2metadata.cli.handlers.qa_handler import QAHandler
from talk2metadata.cli.handlers.schema_handler import SchemaHandler
from talk2metadata.cli.handlers.search_handler import SearchHandler

__all__ = [
    "EvaluationHandler",
    "IngestHandler",
    "IndexHandler",
    "PrepareHandler",
    "QAHandler",
    "SchemaHandler",
    "SearchHandler",
]
