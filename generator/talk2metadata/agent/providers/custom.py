"""Custom OpenAI-compatible provider (e.g. for vLLM/Ollama)."""

from __future__ import annotations

import os
from typing import Any, Optional

from talk2metadata.agent.providers.openai import OpenAIProvider
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


class CustomOpenAIProvider(OpenAIProvider):
    """Custom OpenAI-compatible provider for local/self-hosted models.

    This provider simplifies connecting to vLLM, Ollama, or other OpenAI-compatible APIs
    by relaxing API key requirements and providing better defaults for local inference.
    """

    def __init__(self, model: Optional[str] = None, **kwargs: Any):
        """Initialize custom provider.

        Args:
            model: Model name (e.g. "defog/sqlcoder-7b-2")
            **kwargs: Configuration options including:
                - base_url: Custom API base URL (REQUIRED)
                - api_key: OpenAI API key (defaults to "EMPTY" if not provided)
        """
        # Ensure base_url is present or helpfully suggest it
        if "base_url" not in kwargs:
            logger.warning(
                "Custom provider initialized without `base_url`. Defaulting to OpenAI default if not set."
            )

        # Default API key to "EMPTY" for local servers if not provided
        # This prevents the OpenAI client from raising "api_key not found" errors
        if "api_key" not in kwargs and not os.getenv("OPENAI_API_KEY"):
            logger.info(
                "No API key provided for custom provider. Using dummy key 'EMPTY'."
            )
            kwargs["api_key"] = "EMPTY"

        super().__init__(model=model, **kwargs)
        logger.info(
            f"Initialized CustomOpenAIProvider with model={model}, base_url={kwargs.get('base_url')}"
        )
