"""Factory for creating LLM provider instances."""

from __future__ import annotations

from typing import Any, Dict, Optional

from talk2metadata.agent.base import BaseLLMProvider
from talk2metadata.agent.providers import PROVIDER_REGISTRY
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


class LLMProviderFactory:
    """Factory for creating LLM providers."""

    @staticmethod
    def create_provider(
        provider: str, model: Optional[str] = None, **kwargs: Any
    ) -> BaseLLMProvider:
        """Create an LLM provider instance.

        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            model: Model name (provider-specific default if None)
            **kwargs: Provider-specific configuration

        Returns:
            BaseLLMProvider instance

        Raises:
            ValueError: If provider is not supported

        Example:
            >>> provider = LLMProviderFactory.create_provider(
            ...     provider="openai",
            ...     model="gpt-4",
            ...     api_key="sk-...",
            ...     temperature=0.7,
            ... )
        """
        provider_lower = provider.lower()

        if provider_lower not in PROVIDER_REGISTRY:
            available = ", ".join(sorted(PROVIDER_REGISTRY.keys()))
            raise ValueError(
                f"Unsupported provider: {provider}. Available providers: {available}"
            )

        provider_class = PROVIDER_REGISTRY[provider_lower]
        logger.info(f"Creating {provider_class.__name__} with model={model}")

        # Create provider instance
        if model is not None:
            return provider_class(model=model, **kwargs)
        else:
            return provider_class(**kwargs)

    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> BaseLLMProvider:
        """Create provider from configuration dictionary.

        Args:
            config: Configuration dict with keys:
                - provider: Provider name (required)
                - model: Model name (optional)
                - Other provider-specific config

        Returns:
            BaseLLMProvider instance

        Raises:
            ValueError: If provider is missing

        Example:
            >>> config = {
            ...     "provider": "openai",
            ...     "model": "gpt-4",
            ...     "api_key": "sk-...",
            ...     "temperature": 0.7,
            ... }
            >>> provider = LLMProviderFactory.create_from_config(config)
        """
        if "provider" not in config:
            raise ValueError("Configuration must include 'provider' key")

        provider = config["provider"]
        model = config.get("model")

        # Extract kwargs (everything except provider and model)
        kwargs = {k: v for k, v in config.items() if k not in ("provider", "model")}

        return LLMProviderFactory.create_provider(provider, model, **kwargs)

    @staticmethod
    def list_providers() -> list[str]:
        """Get list of available provider names.

        Returns:
            List of provider names
        """
        return sorted(PROVIDER_REGISTRY.keys())
