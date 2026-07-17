"""Data connectors for Talk2Metadata."""

from talk2metadata.connectors.base import BaseConnector
from talk2metadata.connectors.csv_loader import CSVLoader
from talk2metadata.connectors.db_connector import DBConnector
from talk2metadata.connectors.registry import CONNECTOR_REGISTRY, ConnectorFactory

__all__ = [
    "BaseConnector",
    "CSVLoader",
    "DBConnector",
    "ConnectorFactory",
    "CONNECTOR_REGISTRY",
]
