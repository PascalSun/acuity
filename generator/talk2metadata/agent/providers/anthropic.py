"""Anthropic/Claude LLM provider."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

try:
    import anthropic as _anthropic_pkg
    from anthropic import Anthropic

    _ANTHROPIC_VERSION = getattr(_anthropic_pkg, "__version__", "0.0.0")
    # output_config for structured JSON was added in 0.77.0
    _SUPPORTS_OUTPUT_CONFIG = _ANTHROPIC_VERSION >= "0.77.0"
except ImportError:
    Anthropic = None  # type: ignore
    _SUPPORTS_OUTPUT_CONFIG = False

from talk2metadata.agent.base import BaseLLMProvider, LLMResponse


def _extract_json_from_response(raw: str) -> str:
    """Strip markdown code fences so raw JSON can be parsed."""
    text = raw.strip()
    for pattern in (
        r"^```(?:json)?\s*\n(.*?)\n```\s*$",
        r"^```(?:json)?\s*\n(.*)\n```",
    ):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return text


def _sanitize_sql_value(sql: str) -> str:
    """Remove trailing JSON/markdown artifacts from a sql string (e.g. \" } ```)."""
    if not sql:
        return sql
    s = sql.strip()
    while s and s[-1] in '"}\\]\n\r`':
        s = s[:-1].rstrip()
    return s.strip()


def _normalize_json_response_content(content: str) -> str:
    """For Anthropic JSON responses: strip markdown, parse, clean 'sql' field, re-serialize."""
    raw = _extract_json_from_response(content)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "sql" in obj and isinstance(obj["sql"], str):
            obj["sql"] = _sanitize_sql_value(obj["sql"])
        return json.dumps(obj)
    except (json.JSONDecodeError, TypeError):
        return content


class AnthropicProvider(BaseLLMProvider):
    """Anthropic/Claude provider implementation."""

    def __init__(self, model: Optional[str] = None, **kwargs: Any):
        """Initialize Anthropic provider.

        Args:
            model: Model name (defaults to claude-sonnet-4-5-20250929)
            **kwargs: Configuration options including:
                - api_key: Anthropic API key (or ANTHROPIC_API_KEY env var)
                - base_url: Optional custom API base URL
                - temperature: Default temperature
                - max_tokens: Default max tokens
        """
        if Anthropic is None:
            raise ImportError(
                "anthropic package is required. Install with: pip install anthropic"
            )

        # Default model
        model = model or "claude-sonnet-4-5-20250929"

        # Extract client-specific kwargs
        allowed_client_keys = {"api_key", "base_url", "timeout", "max_retries"}
        client_kwargs = {k: v for k, v in kwargs.items() if k in allowed_client_keys}

        # Use env var as fallback for API key
        if "api_key" not in client_kwargs:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                client_kwargs["api_key"] = api_key

        # Initialize client
        self.client = Anthropic(**client_kwargs)

        # Store config (non-client kwargs become defaults)
        config_kwargs = {
            k: v for k, v in kwargs.items() if k not in allowed_client_keys
        }

        super().__init__(model=model, **config_kwargs)

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
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_format: Response format ("json" or None)
            **kwargs: Additional parameters

        Returns:
            LLMResponse object
        """
        # Build messages
        messages = [{"role": "user", "content": prompt}]

        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            system_prompt=system_prompt,
            **kwargs,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = "json",
        **kwargs: Any,
    ) -> LLMResponse:
        """Multi-turn chat completion.

        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_format: Response format ("json" or None)
            **kwargs: Additional parameters including system_prompt

        Returns:
            LLMResponse object
        """
        # Extract system_prompt from kwargs
        system_prompt = kwargs.pop("system_prompt", None)

        # Merge config defaults with call-time parameters
        default_keys = {
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "stop_sequences",
        }
        merged_kwargs = {
            k: v for k, v in self.config.items() if k in default_keys and v is not None
        }

        # Call-time parameters override defaults
        if temperature is not None:
            merged_kwargs["temperature"] = temperature
        if max_tokens is not None:
            merged_kwargs["max_tokens"] = max_tokens

        # Set default max_tokens if not specified
        if "max_tokens" not in merged_kwargs:
            merged_kwargs["max_tokens"] = 4096

        # Force JSON output: use API structured output when SDK supports it (0.77+),
        # otherwise enforce via system prompt for older SDKs.
        output_config = None
        if response_format == "json":
            if _SUPPORTS_OUTPUT_CONFIG:
                output_config = {
                    "format": {
                        "type": "json_schema",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "thought": {
                                    "type": "string",
                                    "description": "Reasoning or explanation.",
                                },
                                "sql": {
                                    "type": "string",
                                    "description": "The SQL query.",
                                },
                            },
                            "required": ["thought", "sql"],
                            "additionalProperties": False,
                        },
                    }
                }
            else:
                if system_prompt:
                    system_prompt = (
                        f"{system_prompt}\n\nIMPORTANT: You must respond with valid JSON only "
                        '(a single object with "thought" and "sql" keys), no markdown, no extra text.'
                    )
                else:
                    system_prompt = (
                        "You must respond with valid JSON only "
                        '(a single object with "thought" and "sql" keys), no markdown, no extra text.'
                    )

        # Build system blocks
        system_blocks = None
        if system_prompt:
            system_blocks = [{"type": "text", "text": system_prompt}]

        # Prepare create kwargs
        create_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **merged_kwargs,
        }
        if system_blocks:
            create_kwargs["system"] = system_blocks
        if output_config is not None:
            create_kwargs["output_config"] = output_config

        # Call API
        try:
            response = self.client.messages.create(**create_kwargs)

            # Extract content
            content = response.content[0].text if response.content else ""

            # Normalize JSON response (Anthropic-only): strip markdown, clean sql value
            if response_format == "json" and content:
                content = _normalize_json_response_content(content)

            # Build usage dict
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens
                + response.usage.output_tokens,
            }

            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                metadata={"stop_reason": response.stop_reason},
            )

        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            raise
