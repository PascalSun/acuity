"""High-level agent wrapper with config management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from talk2metadata.agent.base import LLMResponse
from talk2metadata.agent.factory import LLMProviderFactory
from talk2metadata.utils.config import Config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


class AgentWrapper:
    """High-level wrapper for LLM agents with config management.

    Handles configuration resolution from multiple sources:
    1. Explicit parameters
    2. Config file
    3. Environment variables
    4. Provider defaults

    Example:
        >>> agent = AgentWrapper(provider="anthropic", model="claude-3-5-sonnet-20241022")
        >>> response = agent.generate("What is the capital of France?")
        >>> print(response.content)
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        config_path: Optional[str | Path] = None,
        **kwargs: Any,
    ):
        """Initialize agent wrapper.

        Args:
            provider: Provider name ("openai", "anthropic", etc.)
            model: Model name (provider default if None)
            config_path: Path to config file (uses default if None)
            **kwargs: Provider-specific configuration
        """
        # Load configuration
        agent_config = self._load_agent_config(config_path)

        # Resolve provider name
        provider = self._resolve_provider_name(provider, kwargs, agent_config)

        # Resolve model name
        model = self._resolve_model_name(model, kwargs, agent_config, provider)

        # Build provider kwargs
        provider_kwargs = self._build_provider_kwargs(kwargs, agent_config, provider)

        # Create provider instance
        self.provider = LLMProviderFactory.create_provider(
            provider=provider, model=model, **provider_kwargs
        )

        logger.info(f"Initialized AgentWrapper with {provider}/{model}")

    def _load_agent_config(self, config_path: Optional[str | Path]) -> Dict[str, Any]:
        """Load agent configuration from file.

        Args:
            config_path: Path to config file

        Returns:
            Agent configuration dict
        """
        try:
            if config_path:
                config = Config.from_yaml(config_path)
            else:
                # Try to use global config
                from talk2metadata.utils.config import get_config

                config = get_config()

            return config.get("agent", {})
        except Exception as e:
            logger.warning(f"Failed to load agent config: {e}, using defaults")
            return {}

    def _resolve_provider_name(
        self,
        provider: Optional[str],
        kwargs: Dict[str, Any],
        agent_config: Dict[str, Any],
    ) -> str:
        """Resolve provider name from multiple sources.

        Priority: explicit parameter > kwargs > config > error

        Args:
            provider: Explicit provider parameter
            kwargs: Additional kwargs
            agent_config: Agent config from file

        Returns:
            Provider name

        Raises:
            ValueError: If provider cannot be resolved
        """
        resolved = (
            provider or kwargs.pop("provider", None) or agent_config.get("provider")
        )

        if resolved is None:
            raise ValueError(
                "Provider must be specified via parameter, config, or kwargs"
            )

        return resolved

    def _resolve_model_name(
        self,
        model: Optional[str],
        kwargs: Dict[str, Any],
        agent_config: Dict[str, Any],
        provider: str,
    ) -> Optional[str]:
        """Resolve model name from multiple sources.

        Priority: explicit parameter > kwargs > provider config > agent config > None (use provider default)

        Args:
            model: Explicit model parameter
            kwargs: Additional kwargs
            agent_config: Agent config from file
            provider: Provider name

        Returns:
            Model name or None (for provider default)
        """
        resolved = (
            model
            or kwargs.pop("model", None)
            or agent_config.get(provider, {}).get("model")
            or agent_config.get("model")
        )

        return resolved

    def _build_provider_kwargs(
        self,
        kwargs: Dict[str, Any],
        agent_config: Dict[str, Any],
        provider: str,
    ) -> Dict[str, Any]:
        """Build provider-specific kwargs from multiple sources.

        Priority: explicit kwargs > provider config > agent config

        Args:
            kwargs: Explicit kwargs
            agent_config: Agent config from file
            provider: Provider name

        Returns:
            Merged provider kwargs
        """
        # Start with agent-level config
        merged = agent_config.get("config", {}).copy()

        # Merge provider-specific config
        provider_config = agent_config.get(provider, {})
        for key, value in provider_config.items():
            if key != "model":  # model handled separately
                merged[key] = value

        # Merge API keys from keys section
        keys_config = agent_config.get("keys", {})
        # Check for provider-specific key (e.g., gemini_api_key)
        api_key_name = f"{provider}_api_key"
        # Special case: gemini provider also accepts google_api_key
        if provider == "gemini" and "google_api_key" in keys_config:
            merged["api_key"] = keys_config["google_api_key"]
        elif api_key_name in keys_config:
            merged["api_key"] = keys_config[api_key_name]

        # Explicit kwargs override everything
        merged.update(kwargs)

        # Expand environment variables in string values
        for key, value in merged.items():
            if (
                isinstance(value, str)
                and value.startswith("${")
                and value.endswith("}")
            ):
                env_var = value[2:-1]
                env_value = os.getenv(env_var)
                if env_value:
                    merged[key] = env_value

        return merged

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text from a prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            response_format: Response format ("json" or None)
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse object
        """
        return self.provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            **kwargs,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Multi-turn chat completion.

        Args:
            messages: List of message dicts with "role" and "content"
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            response_format: Response format ("json" or None)
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse object
        """
        return self.provider.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            **kwargs,
        )

    def switch_provider(
        self, provider: str, model: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Switch to a different provider at runtime.

        Args:
            provider: New provider name
            model: New model name (provider default if None)
            **kwargs: Provider-specific configuration
        """
        self.provider = LLMProviderFactory.create_provider(
            provider=provider, model=model, **kwargs
        )
        logger.info(f"Switched to {provider}/{model or 'default'}")

    @property
    def model(self) -> str:
        """Get current model name."""
        return self.provider.model

    @property
    def provider_type(self) -> str:
        """Get current provider type."""
        return self.provider.__class__.__name__.replace("Provider", "").lower()

    def __repr__(self) -> str:
        return f"AgentWrapper({self.provider_type}/{self.model})"
