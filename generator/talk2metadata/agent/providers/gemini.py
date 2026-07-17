"""Google Gemini provider implementation."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore

from talk2metadata.agent.base import BaseLLMProvider, LLMResponse


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize Gemini provider.

        Args:
            model: Model name (e.g., 'gemini-2.0-flash', 'gemini-2.5-pro')
            api_key: Google API key (or set GOOGLE_API_KEY / GEMINI_API_KEY env var)
            **kwargs: Additional Gemini configuration
        """
        if genai is None:
            raise ImportError(
                "google-genai package is required. Install with: pip install google-genai"
            )

        super().__init__(model, **kwargs)
        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key is required. "
                "Set it in config.yml (agent.gemini.api_key) or set GOOGLE_API_KEY / GEMINI_API_KEY env var."
            )

        self.client = genai.Client(api_key=api_key)

    def _build_generation_config(
        self,
        temperature: Optional[float],
        max_tokens: Optional[int],
        system_prompt: Optional[str] = None,
        response_format: Optional[str] = None,
        **kwargs: Any,
    ) -> types.GenerateContentConfig:
        """Build generation config for Gemini API."""
        config_kwargs: Dict[str, Any] = {}

        # Pull defaults from self.config
        default_keys = {"temperature", "max_output_tokens", "top_p", "top_k"}
        for k, v in self.config.items():
            if k in default_keys and v is not None:
                config_kwargs[k] = v

        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens

        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt

        if response_format == "json":
            config_kwargs["response_mime_type"] = "application/json"

        # Merge any extra kwargs
        config_kwargs.update({k: v for k, v in kwargs.items() if v is not None})

        return types.GenerateContentConfig(**config_kwargs)

    def _extract_usage_metadata(self, response: Any) -> Dict[str, int]:
        """Extract usage metadata from Gemini response."""
        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "prompt_tokens": getattr(
                    response.usage_metadata, "prompt_token_count", 0
                ),
                "completion_tokens": getattr(
                    response.usage_metadata, "response_token_count", 0
                ),
                "total_tokens": getattr(
                    response.usage_metadata, "total_token_count", 0
                ),
            }
        return usage

    def _extract_response_content(self, response: Any) -> tuple[str, Optional[str]]:
        """Extract content and finish reason from Gemini response."""
        finish_reason = None
        content = ""
        if not response.candidates:
            return content, finish_reason

        candidate = response.candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)

        safety_reasons = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT"}
        finish_str = str(finish_reason) if finish_reason else ""

        if finish_str in safety_reasons:
            try:
                content = response.text
            except ValueError:
                content = "[Content filtered by safety filters]"
                self.logger.warning(f"Gemini response was filtered: {finish_reason}")
        else:
            try:
                content = response.text
            except (ValueError, AttributeError) as e:
                self.logger.warning(f"Failed to extract text: {e}")
                content = "[Failed to extract response content]"

        return content, finish_str or None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text using Gemini API."""
        config = self._build_generation_config(
            temperature,
            max_tokens,
            system_prompt=system_prompt,
            response_format=response_format,
            **kwargs,
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        usage = self._extract_usage_metadata(response)
        content, finish_reason = self._extract_response_content(response)

        return LLMResponse(
            content=content,
            model=self.model,
            usage=usage,
            metadata={"finish_reason": finish_reason},
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Chat completion with message history."""
        # Separate system messages from conversation history
        system_prompt = None
        conversation = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                role = "user" if msg["role"] == "user" else "model"
                conversation.append(
                    types.Content(
                        role=role,
                        parts=[types.Part(text=msg["content"])],
                    )
                )

        config = self._build_generation_config(
            temperature,
            max_tokens,
            system_prompt=system_prompt,
            response_format=response_format,
            **kwargs,
        )

        # Build history from all but last message, send last message
        history = conversation[:-1] if len(conversation) > 1 else []
        last_message = conversation[-1].parts[0].text if conversation else ""

        chat = self.client.chats.create(
            model=self.model,
            config=config,
            history=history,
        )

        response = chat.send_message(message=last_message)

        usage = self._extract_usage_metadata(response)
        content, finish_reason = self._extract_response_content(response)

        return LLMResponse(
            content=content,
            model=self.model,
            usage=usage,
            metadata={"finish_reason": finish_reason},
        )
