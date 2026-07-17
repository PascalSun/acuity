"""vLLM provider implementation for local high-performance LLM inference."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

from talk2metadata.agent.base import BaseLLMProvider, LLMResponse


class VLLMProvider(BaseLLMProvider):
    """vLLM provider for local high-performance LLM inference.

    Connects to a running vLLM OpenAI-compatible API server.

    Start vLLM server:
        python -m vllm.entrypoints.openai.api_server --model <model_name>

    Default endpoint: http://localhost:8000/v1

    See https://github.com/vllm-project/vllm for more information.
    """

    def __init__(
        self,
        model: str = "meta-llama/Llama-2-7b-chat-hf",
        base_url: str = "http://localhost:8000/v1",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize vLLM provider in server mode.

        Args:
            model: Model name/identifier
            base_url: vLLM server URL (default: http://localhost:8000/v1)
            api_key: API key (optional, vLLM server typically doesn't require this)
            **kwargs: Additional parameters for OpenAI client
        """
        # Persist full provider config (defaults for generation)
        super().__init__(model, **kwargs)

        # Server mode: use OpenAI-compatible HTTP API
        if OpenAI is None:
            raise ImportError(
                "openai package is required for vLLM provider. Install with: pip install openai"
            )

        # Build OpenAI client with ONLY supported client options
        allowed_client_keys = {
            "api_key",
            "base_url",
            "organization",
            "project",
            "timeout",
            "http_client",
        }
        client_kwargs: Dict[str, Any] = {
            k: v for k, v in kwargs.items() if k in allowed_client_keys
        }
        # Set base_url (vLLM default endpoint)
        if "base_url" not in client_kwargs:
            client_kwargs["base_url"] = base_url
        # vLLM typically doesn't require API keys, but OpenAI client requires the parameter
        # Use provided api_key or a dummy value (vLLM server will ignore it)
        if "api_key" not in client_kwargs:
            client_kwargs["api_key"] = api_key if api_key else "EMPTY"

        self.logger.info(f"Connecting to vLLM server at {client_kwargs['base_url']}")
        self.client = OpenAI(**client_kwargs)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text using vLLM server mode."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Merge default generation kwargs from provider config
        default_keys = {
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "logprobs",
            "logit_bias",
            "seed",
        }
        merged_kwargs: Dict[str, Any] = {
            k: v for k, v in self.config.items() if k in default_keys and v is not None
        }
        # Call-time args override defaults
        if temperature is not None:
            merged_kwargs["temperature"] = temperature
        if max_tokens is not None:
            merged_kwargs["max_tokens"] = max_tokens
        # Filter out None values from kwargs
        merged_kwargs.update({k: v for k, v in kwargs.items() if v is not None})

        # Support JSON mode
        if response_format == "json":
            merged_kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **merged_kwargs,
        )

        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": (
                response.usage.completion_tokens if response.usage else 0
            ),
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=self.model,
            usage=usage,
            metadata={"finish_reason": response.choices[0].finish_reason},
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Chat completion with message history using vLLM server mode."""
        # Merge default generation kwargs from provider config
        default_keys = {
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "logprobs",
            "logit_bias",
            "seed",
        }
        merged_kwargs: Dict[str, Any] = {
            k: v for k, v in self.config.items() if k in default_keys and v is not None
        }
        if temperature is not None:
            merged_kwargs["temperature"] = temperature
        if max_tokens is not None:
            merged_kwargs["max_tokens"] = max_tokens
        # Filter out None values from kwargs
        merged_kwargs.update({k: v for k, v in kwargs.items() if v is not None})

        # Support JSON mode
        if response_format == "json":
            merged_kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **merged_kwargs,
        )

        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": (
                response.usage.completion_tokens if response.usage else 0
            ),
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=self.model,
            usage=usage,
            metadata={"finish_reason": response.choices[0].finish_reason},
        )
