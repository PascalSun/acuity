"""Base classes for LLM agent providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Standardized response from LLM providers."""

    content: str
    model: str
    usage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        tokens = self.usage.get("total_tokens", "?")
        return f"LLMResponse(model={self.model}, tokens={tokens})"


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model: str, **kwargs: Any):
        """Initialize LLM provider.

        Args:
            model: Model name/identifier
            **kwargs: Provider-specific configuration
        """
        self.model = model
        self.config = kwargs
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
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
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
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
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
