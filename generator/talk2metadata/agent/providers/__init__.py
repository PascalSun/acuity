"""LLM provider registry."""

from talk2metadata.agent.providers.anthropic import AnthropicProvider
from talk2metadata.agent.providers.custom import CustomOpenAIProvider
from talk2metadata.agent.providers.doubao import DoubaoProvider
from talk2metadata.agent.providers.gemini import GeminiProvider
from talk2metadata.agent.providers.huggingface import HuggingFaceProvider
from talk2metadata.agent.providers.ollama import OllamaProvider
from talk2metadata.agent.providers.openai import OpenAIProvider
from talk2metadata.agent.providers.vllm import VLLMProvider

# Provider registry maps provider name to class
PROVIDER_REGISTRY = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
    "gemini": GeminiProvider,
    "vllm": VLLMProvider,
    "doubao": DoubaoProvider,
    "huggingface": HuggingFaceProvider,
    "custom": CustomOpenAIProvider,
}

__all__ = [
    "PROVIDER_REGISTRY",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "GeminiProvider",
    "VLLMProvider",
    "DoubaoProvider",
    "HuggingFaceProvider",
]
