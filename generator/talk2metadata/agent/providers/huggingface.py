"""Hugging Face Inference API provider implementation."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None  # type: ignore

from talk2metadata.agent.base import BaseLLMProvider, LLMResponse


class HuggingFaceProvider(BaseLLMProvider):
    """Hugging Face Inference API provider.

    Uses the Hugging Face Inference API for serverless model inference.
    https://huggingface.co/inference-api
    """

    def __init__(
        self,
        model: str = "meta-llama/Llama-2-7b-chat-hf",
        api_key: Optional[str] = None,
        base_url: str = "https://api-inference.huggingface.co/models",
        **kwargs: Any,
    ):
        """Initialize HuggingFace provider.

        Args:
            model: Model identifier (e.g., 'meta-llama/Llama-2-7b-chat-hf')
            api_key: HuggingFace API token (or HF_TOKEN env var)
            base_url: API base URL
            **kwargs: Additional configuration
        """
        if requests is None:
            raise ImportError(
                "requests package is required. Install with: pip install requests"
            )

        # Try to get API key from parameter first, then env var
        api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError(
                "HuggingFace API token is required. "
                "Set it in config.yml (agent.huggingface.api_key) or set HF_TOKEN env var."
            )

        super().__init__(model, **kwargs)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/{model}"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text using HuggingFace Inference API."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # HuggingFace JSON mode via prompt engineering
        if response_format == "json":
            full_prompt = f"{full_prompt}\n\nIMPORTANT: You must respond with valid JSON only, no additional text or markdown."

        # Build parameters
        default_keys = {
            "temperature",
            "max_new_tokens",
            "top_p",
            "top_k",
            "repetition_penalty",
            "do_sample",
        }
        merged_params = {k: v for k, v in self.config.items() if k in default_keys}
        if temperature is not None:
            merged_params["temperature"] = temperature
        if max_tokens is not None:
            merged_params["max_new_tokens"] = max_tokens
        elif "max_new_tokens" not in merged_params:
            merged_params["max_new_tokens"] = 512

        if merged_params.get("temperature", 0) > 0:
            merged_params["do_sample"] = True

        # Make request
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": full_prompt,
            "parameters": merged_params,
        }

        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=kwargs.get("timeout", 120),
        )
        response.raise_for_status()
        data = response.json()

        # Extract generated text
        if isinstance(data, list) and len(data) > 0:
            generated_text = data[0].get("generated_text", "")
        elif isinstance(data, dict):
            generated_text = data.get("generated_text", "")
        else:
            generated_text = str(data)

        # Remove the prompt from the output if it's included
        if generated_text.startswith(full_prompt):
            generated_text = generated_text[len(full_prompt) :].strip()

        return LLMResponse(
            content=generated_text,
            model=self.model,
            usage={},  # HF Inference API doesn't provide token counts
            metadata={},
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
        # Format messages into prompt
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt = "\n".join(prompt_parts) + "\nAssistant:"

        return self.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            **kwargs,
        )
