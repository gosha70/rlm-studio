# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0
"""
Provider catalog — single source of truth for provider metadata.

All display names, model lists, pricing, env-var mappings, and placeholder
hints live here so that UI rendering code stays free of hard-coded data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a single model."""

    name: str
    """Model identifier sent to the API (e.g. ``gpt-4o``)."""

    input_cost_per_1k: float = 0.0
    """Cost in USD per 1 000 input tokens."""

    output_cost_per_1k: float = 0.0
    """Cost in USD per 1 000 output tokens."""


@dataclass(frozen=True)
class ProviderEntry:
    """Everything the UI needs to know about a single provider type."""

    key: str
    """Internal key (``openai``, ``anthropic``, ``ollama``, ``lmstudio``, ``vllm``)."""

    display_name: str
    """Human-readable label shown in the UI."""

    models: list[ModelInfo] = field(default_factory=list)
    """Pre-defined models (empty for local providers where user types a name)."""

    env_var: str | None = None
    """Default environment variable for the API key (``None`` for local)."""

    api_key_placeholder: str = "Paste your API key here"
    """Placeholder hint for the key input field."""

    requires_api_key: bool = True
    """Whether this provider needs an API key at all."""

    default_endpoint: str | None = None
    """Default API endpoint (useful for local providers)."""

    model_input_hint: str = ""
    """Placeholder for the free-text model input (local providers only)."""

    # Embedding support
    embedding_models: list[ModelInfo] = field(default_factory=list)
    """Embedding models available from this provider."""


# ---------------------------------------------------------------------------
# Catalog data
# ---------------------------------------------------------------------------

OPENAI_MODELS: list[ModelInfo] = [
    ModelInfo("gpt-4o", 0.005, 0.015),
    ModelInfo("gpt-4o-mini", 0.00015, 0.0006),
    ModelInfo("gpt-4-turbo", 0.01, 0.03),
    ModelInfo("gpt-4", 0.03, 0.06),
    ModelInfo("gpt-3.5-turbo", 0.0005, 0.0015),
    ModelInfo("o1", 0.015, 0.06),
    ModelInfo("o1-mini", 0.003, 0.012),
]

ANTHROPIC_MODELS: list[ModelInfo] = [
    ModelInfo("claude-opus-4-6", 0.003, 0.015),
    ModelInfo("claude-sonnet-4-6", 0.003, 0.015),
    ModelInfo("claude-haiku-4-5-20251001", 0.008, 0.024),
]

# Default model per provider for the legacy `llm/` clients, which previously
# carried their own hardcoded literals. Those drifted: they still named
# Claude 3 models that Anthropic has since retired, so `provider="anthropic"`
# on the public client failed with not_found_error out of the box. Deriving
# the default from this list keeps one source of truth — when a model is
# retired it is removed here once, and every caller follows.
# Sonnet, not ANTHROPIC_MODELS[0] (Opus): the legacy defaults these replace
# were Sonnet-tier, and silently moving an existing caller onto Opus pricing
# would be a surprise cost rather than a retirement fix.
DEFAULT_ANTHROPIC_MODEL = next(m.name for m in ANTHROPIC_MODELS if "sonnet" in m.name)
DEFAULT_OPENAI_MODEL = OPENAI_MODELS[0].name
# Cheap tier, for the cost-optimised presets and recursive sub-calls.
DEFAULT_ANTHROPIC_CHEAP_MODEL = next(m.name for m in ANTHROPIC_MODELS if "haiku" in m.name)


OPENAI_EMBEDDING_MODELS: list[ModelInfo] = [
    ModelInfo("text-embedding-3-small", 0.00002, 0.0),
    ModelInfo("text-embedding-3-large", 0.00013, 0.0),
    ModelInfo("text-embedding-ada-002", 0.0001, 0.0),
]

# ---------------------------------------------------------------------------
# Provider catalog (ordered — first entry is the UI default)
# ---------------------------------------------------------------------------

PROVIDERS: list[ProviderEntry] = [
    ProviderEntry(
        key="openai",
        display_name="OpenAI",
        models=OPENAI_MODELS,
        env_var="OPENAI_API_KEY",
        api_key_placeholder="sk-... (from platform.openai.com)",
        requires_api_key=True,
        embedding_models=OPENAI_EMBEDDING_MODELS,
    ),
    ProviderEntry(
        key="anthropic",
        display_name="Anthropic",
        models=ANTHROPIC_MODELS,
        env_var="ANTHROPIC_API_KEY",
        api_key_placeholder="sk-ant-... (from console.anthropic.com)",
        requires_api_key=True,
    ),
    ProviderEntry(
        key="ollama",
        display_name="Ollama (Local)",
        env_var=None,
        api_key_placeholder="Not required for local Ollama",
        requires_api_key=False,
        default_endpoint="http://localhost:11434",
        model_input_hint="e.g., llama2, neural-chat, mistral",
    ),
    ProviderEntry(
        key="lmstudio",
        display_name="LM Studio (Local)",
        env_var=None,
        api_key_placeholder="Not required for local LM Studio",
        requires_api_key=False,
        default_endpoint="http://localhost:1234/v1",
        model_input_hint="e.g., mistral, neural-chat",
    ),
    ProviderEntry(
        key="vllm",
        display_name="vLLM (Self-hosted)",
        env_var=None,
        api_key_placeholder="Optional — leave blank for a local/private server",
        requires_api_key=False,
        default_endpoint="http://localhost:8000/v1",
        model_input_hint="e.g., Qwen/Qwen2.5-7B-Instruct (HuggingFace model ID)",
    ),
]

# ---------------------------------------------------------------------------
# Convenience look-ups
# ---------------------------------------------------------------------------

PROVIDERS_BY_KEY: dict[str, ProviderEntry] = {p.key: p for p in PROVIDERS}
"""Look up a ``ProviderEntry`` by its internal key."""

PROVIDER_OPTIONS: list[tuple[str, str]] = [(p.display_name, p.key) for p in PROVIDERS]
"""``(display_name, key)`` pairs for select-box widgets."""


def get_provider(key: str) -> ProviderEntry | None:
    """Return the catalog entry for *key*, or ``None``."""
    return PROVIDERS_BY_KEY.get(key)


def get_model_pricing(provider_key: str, model_name: str) -> tuple[float, float]:
    """Return ``(input_cost_per_1k, output_cost_per_1k)`` for a model.

    Falls back to ``(0.0, 0.0)`` for unknown models / local providers.
    """
    entry = PROVIDERS_BY_KEY.get(provider_key)
    if not entry:
        return (0.0, 0.0)
    for m in entry.models:
        if m.name == model_name:
            return (m.input_cost_per_1k, m.output_cost_per_1k)
    return (0.0, 0.0)


def get_model_names(provider_key: str) -> list[str]:
    """Return the list of model names for a provider."""
    entry = PROVIDERS_BY_KEY.get(provider_key)
    if not entry:
        return []
    return [m.name for m in entry.models]


def get_env_var(provider_key: str) -> str | None:
    """Return the expected env-var name for a provider's API key."""
    entry = PROVIDERS_BY_KEY.get(provider_key)
    return entry.env_var if entry else None


def get_embedding_model_names() -> list[str]:
    """Return all available embedding model names (across all providers)."""
    names: list[str] = []
    for p in PROVIDERS:
        for m in p.embedding_models:
            names.append(m.name)
    return names


def build_pricing_table() -> dict[str, dict[str, dict[str, str]]]:
    """Build the pricing reference table for display.

    Returns a nested dict::

        {
            "OpenAI": {
                "gpt-4o": {"input": "$0.005", "output": "$0.015", "per": "1K tokens"},
                ...
            },
            ...
        }
    """
    table: dict[str, dict[str, dict[str, str]]] = {}
    for p in PROVIDERS:
        if not p.models:
            # Local providers get a single "free" row
            table[p.display_name] = {
                "Any Model": {"input": "Free", "output": "Free", "per": "Local execution"},
            }
        else:
            models_dict: dict[str, dict[str, str]] = {}
            for m in p.models:
                models_dict[m.name] = {
                    "input": f"${m.input_cost_per_1k}",
                    "output": f"${m.output_cost_per_1k}",
                    "per": "1K tokens",
                }
            table[p.display_name] = models_dict
    return table


def build_metrics_pricing() -> dict[str, dict[str, dict[str, float]]]:
    """Build pricing dict in the format expected by ``MetricsCollector``.

    Returns::

        {
            "openai": {
                "gpt-4o": {"input": 0.005, "output": 0.015},
                ...
            },
            ...
        }
    """
    result: dict[str, dict[str, dict[str, float]]] = {}
    for p in PROVIDERS:
        models: dict[str, dict[str, float]] = {}
        for m in p.models:
            models[m.name] = {
                "input": m.input_cost_per_1k,
                "output": m.output_cost_per_1k,
            }
        if models:
            result[p.key] = models
    return result
