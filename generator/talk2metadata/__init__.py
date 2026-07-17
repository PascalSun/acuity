"""Talk2Metadata - Question-driven multi-table record retrieval."""

__version__ = "0.1.0"

# Connectors
from talk2metadata.connectors import (
    BaseConnector,
    ConnectorFactory,
    CSVLoader,
    DBConnector,
)

# Core modules
from talk2metadata.core import (
    ForeignKey,
    SchemaDetector,
    SchemaMetadata,
    TableMetadata,
)

# Utils
from talk2metadata.utils.config import Config, get_config, load_config


# Lazy imports for optional dependencies (if needed in future)
def __getattr__(name):
    """Lazy import for optional modules."""
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Version
    "__version__",
    # Core
    "SchemaDetector",
    "SchemaMetadata",
    "TableMetadata",
    "ForeignKey",
    # Connectors
    "BaseConnector",
    "ConnectorFactory",
    "CSVLoader",
    "DBConnector",
    # Config
    "Config",
    "get_config",
    "load_config",
]
