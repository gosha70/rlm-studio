# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""Auto-detection and configuration of LLM providers from environment variables."""

import os
from typing import Any

from rlmstudio.domain.model_catalog import DEFAULT_ANTHROPIC_MODEL

from .base import BaseLLMProvider


class ConfigurationError(Exception):
    """Raised when no LLM provider is properly configured."""

    pass


def auto_detect_provider() -> str | None:
    """
    Auto-detect the first available LLM provider from environment variables.

    Checks in order:
    1. OPENAI_API_KEY → openai
    2. ANTHROPIC_API_KEY → anthropic
    3. OLLAMA_BASE_URL → ollama
    4. LMSTUDIO_BASE_URL → lmstudio
    5. VLLM_BASE_URL → vllm

    Returns:
        Provider name if found, None otherwise
    """
    # Check OpenAI
    if os.getenv("OPENAI_API_KEY"):
        return "openai"

    # Check Anthropic/Claude
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"

    # Check Ollama (local)
    if os.getenv("OLLAMA_BASE_URL"):
        return "ollama"

    # Check LM Studio (local)
    if os.getenv("LMSTUDIO_BASE_URL"):
        return "lmstudio"

    # Check vLLM
    if os.getenv("VLLM_BASE_URL"):
        return "vllm"

    return None


def list_available_providers() -> list[str]:
    """
    List all LLM providers that are currently configured via environment variables.

    Returns:
        List of available provider names

    Example:
        >>> providers = list_available_providers()
        >>> print(providers)
        ['openai', 'anthropic', 'ollama']
    """
    available = []

    if os.getenv("OPENAI_API_KEY"):
        available.append("openai")

    if os.getenv("ANTHROPIC_API_KEY"):
        available.append("anthropic")

    if os.getenv("OLLAMA_BASE_URL"):
        available.append("ollama")

    if os.getenv("LMSTUDIO_BASE_URL"):
        available.append("lmstudio")

    if os.getenv("VLLM_BASE_URL"):
        available.append("vllm")

    return available


def get_provider_info() -> dict[str, dict[str, Any]]:
    """
    Get detailed information about each provider's configuration status.

    Returns:
        Dictionary mapping provider name to configuration details

    Example:
        >>> info = get_provider_info()
        >>> print(info['openai'])
        {'configured': True, 'env_var': 'OPENAI_API_KEY', 'status': 'ready'}
    """
    info = {}

    # OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    info["openai"] = {
        "configured": bool(openai_key),
        "env_var": "OPENAI_API_KEY",
        "status": "ready" if openai_key else "missing",
        "key_preview": f"{openai_key[:8]}..." if openai_key else None,
    }

    # Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    info["anthropic"] = {
        "configured": bool(anthropic_key),
        "env_var": "ANTHROPIC_API_KEY",
        "status": "ready" if anthropic_key else "missing",
        "key_preview": f"{anthropic_key[:8]}..." if anthropic_key else None,
    }

    # Ollama
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    info["ollama"] = {
        "configured": bool(os.getenv("OLLAMA_BASE_URL")),
        "env_var": "OLLAMA_BASE_URL",
        "status": "ready" if os.getenv("OLLAMA_BASE_URL") else "default",
        "base_url": ollama_url,
    }

    # LM Studio
    lmstudio_url = os.getenv("LMSTUDIO_BASE_URL")
    info["lmstudio"] = {
        "configured": bool(lmstudio_url),
        "env_var": "LMSTUDIO_BASE_URL",
        "status": "ready" if lmstudio_url else "missing",
        "base_url": lmstudio_url,
    }

    # vLLM
    vllm_url = os.getenv("VLLM_BASE_URL")
    info["vllm"] = {
        "configured": bool(vllm_url),
        "env_var": "VLLM_BASE_URL",
        "status": "ready" if vllm_url else "missing",
        "base_url": vllm_url,
    }

    return info


def get_default_client_config() -> tuple[str, str | None]:
    """
    Get the default client configuration (provider and model).

    Returns:
        Tuple of (provider, default_model)

    Raises:
        ConfigurationError: If no provider is configured
    """
    provider = auto_detect_provider()

    if provider is None:
        raise ConfigurationError(
            "No LLM provider configured. Please set one of:\n"
            "  • OPENAI_API_KEY=sk-...\n"
            "  • ANTHROPIC_API_KEY=sk-ant-...\n"
            "  • OLLAMA_BASE_URL=http://localhost:11434\n"
            "  • LMSTUDIO_BASE_URL=http://localhost:1234\n"
            "  • VLLM_BASE_URL=http://localhost:8000\n\n"
            "Or explicitly pass provider='openai' and api_key='...' to interact()"
        )

    # Default models for each provider
    default_models = {
        "openai": "gpt-4o-mini",
        "anthropic": DEFAULT_ANTHROPIC_MODEL,
        "ollama": "llama3.2",
        "lmstudio": None,  # User must specify
        "vllm": None,  # User must specify
    }

    return provider, default_models.get(provider)


def get_default_client() -> BaseLLMProvider:
    """
    Get a configured LLM client using auto-detection.

    This is a convenience function that auto-detects the provider
    from environment variables and returns a ready-to-use client.

    Returns:
        Configured LLM client

    Raises:
        ConfigurationError: If no provider is configured

    Example:
        >>> from rlmstudio.llm import get_default_client
        >>> client = get_default_client()  # Auto-detects from env
        >>> response = client.complete([{"role": "user", "content": "Hello"}])
    """
    from . import get_llm_client

    provider, model = get_default_client_config()
    return get_llm_client(provider=provider, model=model)
