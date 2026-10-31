# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""LLM client interfaces and implementations."""

from typing import Any

from rlmstudio.ui.data.providers_catalog import DEFAULT_ANTHROPIC_MODEL

from .auto import (
    ConfigurationError,
    auto_detect_provider,
    get_default_client,
    get_default_client_config,
    get_provider_info,
    list_available_providers,
)
from .base import BaseLLMProvider, LLMResponse
from .mock_client import MockLLMClient

# Optional external providers (lazy import to avoid requiring dependencies)
__all__ = [
    "MockLLMClient",
    "BaseLLMProvider",
    "LLMResponse",
    "auto_detect_provider",
    "list_available_providers",
    "get_provider_info",
    "get_default_client",
    "get_default_client_config",
    "ConfigurationError",
]

# Try to import OpenAI client
try:
    from .openai_client import OpenAIClient

    __all__.append("OpenAIClient")
    _OpenAIClient: Any = OpenAIClient
except ImportError:
    OpenAIClient = None  # type: ignore[assignment,misc]  # noqa: F401
    _OpenAIClient = None

# Try to import Claude client
try:
    from .anthropic_client import ClaudeClient

    __all__.append("ClaudeClient")
    _ClaudeClient: Any = ClaudeClient
except ImportError:
    ClaudeClient = None  # type: ignore[assignment,misc]  # noqa: F401
    _ClaudeClient = None

# Try to import Ollama client
try:
    from .ollama_client import OllamaClient

    __all__.append("OllamaClient")
    _OllamaClient: Any = OllamaClient
except ImportError:
    OllamaClient = None  # type: ignore[assignment,misc]  # noqa: F401
    _OllamaClient = None

# Try to import LM Studio client
try:
    from .lmstudio_client import LMStudioClient

    __all__.append("LMStudioClient")
    _LMStudioClient: Any = LMStudioClient
except ImportError:
    LMStudioClient = None  # type: ignore[assignment,misc]  # noqa: F401
    _LMStudioClient = None

# Try to import vLLM client
try:
    from .vllm_client import vLLMClient

    __all__.append("vLLMClient")
    _vLLMClient: Any = vLLMClient  # noqa: N816
except ImportError:
    vLLMClient = None  # type: ignore[assignment,misc]  # noqa: F401,N816
    _vLLMClient = None  # noqa: N816


def get_llm_client(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> BaseLLMProvider:
    """
    Factory function to create LLM client based on provider name.

    Args:
        provider: Provider name ('openai', 'anthropic', 'ollama', 'lmstudio', 'vllm', 'mock')
        model: Model name (provider-specific)
        api_key: API key for the provider (optional, will use environment variable if not provided)
        **kwargs: Additional arguments for the client

    Returns:
        LLM client instance

    Raises:
        ValueError: If provider is not supported or not installed
    """
    provider = provider.lower()

    if provider == "mock":
        return MockLLMClient(**kwargs)  # type: ignore[return-value]

    elif provider == "openai":
        if _OpenAIClient is None:
            raise ValueError("OpenAI client not available. Install with: pip install openai")
        if api_key:
            kwargs["api_key"] = api_key
        return _OpenAIClient(model=model or "gpt-4", **kwargs)  # type: ignore[no-any-return]

    elif provider == "anthropic":
        if _ClaudeClient is None:
            raise ValueError("Anthropic client not available. Install with: pip install anthropic")
        if api_key:
            kwargs["api_key"] = api_key
        return _ClaudeClient(model=model or DEFAULT_ANTHROPIC_MODEL, **kwargs)  # type: ignore[no-any-return]

    elif provider == "ollama":
        if _OllamaClient is None:
            raise ValueError("Ollama client not available. Install with: pip install ollama")
        return _OllamaClient(model=model or "llama2", **kwargs)  # type: ignore[no-any-return]

    elif provider == "lmstudio" or provider == "lm studio":
        if _LMStudioClient is None:
            raise ValueError("LM Studio client not available")
        return _LMStudioClient(model=model, **kwargs)  # type: ignore[no-any-return]

    elif provider == "vllm":
        if _vLLMClient is None:
            raise ValueError("vLLM client not available")
        return _vLLMClient(model=model, **kwargs)  # type: ignore[no-any-return]

    else:
        raise ValueError(
            f"Unknown provider: {provider}. Supported: openai, anthropic, ollama, lmstudio, vllm, mock"
        )


__all__.append("get_llm_client")
