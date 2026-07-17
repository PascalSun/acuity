"""Connector registry and factory."""

from __future__ import annotations

from typing import Dict, Type

from talk2metadata.connectors.base import BaseConnector
from talk2metadata.connectors.csv_loader import CSVLoader
from talk2metadata.connectors.db_connector import DBConnector
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)

# Registry of available connectors
CONNECTOR_REGISTRY: Dict[str, Type[BaseConnector]] = {
    "csv": CSVLoader,
    "database": DBConnector,
    "db": DBConnector,  # Alias
}


class ConnectorFactory:
    """Factory for creating connector instances."""

    @staticmethod
    def create_connector(
        connector_type: str,
        **kwargs,
    ) -> BaseConnector:
        """Create connector instance.

        Args:
            connector_type: Connector type ('csv', 'database', 'db')
            **kwargs: Connector-specific configuration

        Returns:
            BaseConnector instance

        Raises:
            ValueError: If connector type is not supported

        Example:
            >>> # CSV connector
            >>> connector = ConnectorFactory.create_connector(
            ...     "csv",
            ...     data_dir="./data/csv",
            ...     target_table="orders"
            ... )
            >>>
            >>> # Database connector
            >>> connector = ConnectorFactory.create_connector(
            ...     "database",
            ...     connection_string="postgresql://localhost/mydb",
            ...     target_table="orders"
            ... )
        """
        connector_type_lower = connector_type.lower().strip()

        if connector_type_lower not in CONNECTOR_REGISTRY:
            available = ", ".join(sorted(CONNECTOR_REGISTRY.keys()))
            raise ValueError(
                f"Unknown connector type: {connector_type}. " f"Available: {available}"
            )

        connector_class = CONNECTOR_REGISTRY[connector_type_lower]
        logger.info(f"Creating {connector_class.__name__} connector")

        return connector_class(**kwargs)

    @staticmethod
    def register_connector(name: str, connector_class: Type[BaseConnector]) -> None:
        """Register a custom connector.

        Args:
            name: Connector name
            connector_class: Connector class (must inherit from BaseConnector)

        Example:
            >>> class MyConnector(BaseConnector):
            ...     pass
            >>>
            >>> ConnectorFactory.register_connector("myconnector", MyConnector)
        """
        if not issubclass(connector_class, BaseConnector):
            raise TypeError(
                f"Connector class must inherit from BaseConnector, "
                f"got {connector_class}"
            )

        CONNECTOR_REGISTRY[name.lower()] = connector_class
        logger.info(f"Registered custom connector: {name}")

    @staticmethod
    def list_connectors() -> list[str]:
        """List available connector types.

        Returns:
            List of connector names
        """
        return sorted(set(CONNECTOR_REGISTRY.keys()))
