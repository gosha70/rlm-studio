"""Provider management endpoints."""

from __future__ import annotations

import logging
import os
import re
import time

import httpx
from fastapi import APIRouter, Depends, HTTPException

from rlmstudio.application.services.provider_failure import redact_secrets
from rlmstudio.server.dependencies import AppState, get_state
from rlmstudio.server.models import (
    ModelInfo,
    ProviderConfig,
    ProviderInfo,
    ProviderSaveRequest,
    ProviderSaveResponse,
    ProviderTestRequest,
    ProviderTestResponse,
)
from rlmstudio.ui.data.providers_catalog import PROVIDERS, PROVIDERS_BY_KEY
from rlmstudio.ui.services.secret_store import FileSecretStore, KeyringSecretStore

logger = logging.getLogger(__name__)

# Cache for dynamic model lists: provider_key -> (timestamp, models)
_model_cache: dict[str, tuple[float, list[ModelInfo]]] = {}
_MODEL_CACHE_TTL = 300  # 5 minutes

router = APIRouter()

# LiteLLM provider prefixes: models must be prefixed for non-OpenAI providers
_LITELLM_PREFIXES: dict[str, str] = {
    "anthropic": "anthropic/",
    "ollama": "ollama/",
    "lmstudio": "openai/",  # LM Studio uses OpenAI-compatible API
    # hosted_vllm/, not openai/ — see docs/hosts/dgx-spark-vllm.md §7 Blocker #4.
    # Must stay in sync with the tables in server/dependencies.py and api.py.
    "vllm": "hosted_vllm/",
}


def _litellm_model_name(provider_key: str, model: str) -> str:
    """Prepend the LiteLLM provider prefix if needed.

    E.g. 'ollama' + 'qwen3' -> 'ollama/qwen3'
    OpenAI models don't need a prefix. Anthropic models handled by LiteLLM
    but explicit prefix is safer.

    HuggingFace-style model IDs (e.g. 'Qwen/Qwen2.5-7B-Instruct') contain a
    slash but are NOT already prefixed — only skip prefixing when the model
    already starts with this provider's own LiteLLM prefix.
    """
    prefix = _LITELLM_PREFIXES.get(provider_key, "")
    if not prefix or model.startswith(prefix):
        return model
    return f"{prefix}{model}"


def _build_model_infos(provider_key: str) -> list[ModelInfo]:
    """Convert catalog ModelInfo dataclasses to Pydantic ModelInfo."""
    entry = PROVIDERS_BY_KEY.get(provider_key)
    if not entry or not entry.models:
        return []
    return [
        ModelInfo(
            name=m.name,
            input_cost_per_1k=m.input_cost_per_1k,
            output_cost_per_1k=m.output_cost_per_1k,
        )
        for m in entry.models
    ]


@router.get("/api/providers")
async def list_providers(
    state: AppState = Depends(get_state),  # noqa: B008
) -> list[ProviderInfo]:
    """List configured providers with status.

    Status values:
    - "connected": API key present AND health check passed (cached)
    - "configured": API key present but not yet verified
    - "not_configured": no API key set
    """
    # Build a lookup of persisted provider configs for status + masked key
    persisted: dict[str, ProviderConfig] = {pc.provider: pc for pc in state.config.provider_configs}

    result = []
    for p in PROVIDERS:
        configured = False
        raw_key: str | None = None
        if p.env_var:
            raw_key = os.environ.get(p.env_var)
            configured = bool(raw_key)
        elif not p.requires_api_key:
            # Local providers (Ollama, LM Studio) are always "configured"
            configured = True

        # Determine status:
        # - "connected" if test passed (in-memory or persisted) or key present + enabled
        # - "configured" if key present but not enabled and never tested
        # - "not_configured" if no key
        cached = _provider_status_cache.get(p.key)
        pc = persisted.get(p.key)
        if cached == "offline":
            status = "offline"
        elif cached == "connected":
            status = "connected"
        elif pc and pc.last_tested_status == "connected" and configured:
            status = "connected"
            _provider_status_cache[p.key] = "connected"
        elif configured and pc and pc.enabled:
            # Active provider with API key — treat as connected
            status = "connected"
        elif configured:
            status = "configured"
        else:
            status = "not_configured"

        # Mask the API key: show first 6 and last 4 chars
        masked: str | None = None
        if raw_key and len(raw_key) > 12:
            masked = f"{raw_key[:6]}...{raw_key[-4:]}"
        elif raw_key:
            masked = "****"

        default_model = p.models[0].name if p.models else None

        # Use persisted endpoint if set, otherwise catalog default
        effective_endpoint = pc.endpoint if pc and pc.endpoint else p.default_endpoint

        result.append(
            ProviderInfo(
                name=p.key,
                display_name=p.display_name,
                status=status,
                models=_build_model_infos(p.key),
                default_model=default_model,
                configured=configured,
                requires_api_key=p.requires_api_key,
                default_endpoint=effective_endpoint,
                model_input_hint=p.model_input_hint,
                masked_api_key=masked,
            )
        )
    return result


# Cache of verified provider statuses (populated by test_provider endpoint)
_provider_status_cache: dict[str, str] = {}


def _persist_test_status(state: AppState, provider_name: str, status: str) -> None:
    """Save last_tested_status to the provider config on disk."""
    for pc in state.config.provider_configs:
        if pc.provider == provider_name:
            pc.last_tested_status = status
            state.save_config()
            return


@router.post("/api/providers/test")
async def test_provider(
    req: ProviderTestRequest,
    state: AppState = Depends(get_state),  # noqa: B008, PT028
) -> ProviderTestResponse:
    """Test a provider connection."""
    model = req.model
    if not model:
        entry = PROVIDERS_BY_KEY.get(req.provider)
        if entry and entry.models:
            model = entry.models[0].name
    if not model:
        model = "gpt-4o"

    # LiteLLM requires provider prefix for non-OpenAI models
    litellm_model = _litellm_model_name(req.provider, model)
    logger.info(
        "Testing provider=%s model=%s (litellm_model=%s)", req.provider, model, litellm_model
    )

    import litellm

    params: dict = {
        "model": litellm_model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 5,
        "timeout": 30,
        "drop_params": True,
    }
    if req.api_key:
        params["api_key"] = req.api_key

    # Use explicit endpoint or fall back to provider's default
    endpoint = req.endpoint
    if not endpoint:
        entry = PROVIDERS_BY_KEY.get(req.provider)
        if entry and entry.default_endpoint:
            endpoint = entry.default_endpoint
    if endpoint:
        params["api_base"] = endpoint
        logger.debug("Using api_base=%s", endpoint)

    start = time.time()
    try:
        response = litellm.completion(**params)
        latency_ms = int((time.time() - start) * 1000)
        if response.choices:
            logger.info("Provider %s connected OK (%dms)", req.provider, latency_ms)
            _provider_status_cache[req.provider] = "connected"
            _persist_test_status(state, req.provider, "connected")
            return ProviderTestResponse(
                connected=True,
                latency_ms=latency_ms,
                model=model,
            )
        logger.warning("Provider %s returned no choices", req.provider)
        _provider_status_cache[req.provider] = "offline"
        _persist_test_status(state, req.provider, "offline")
        return ProviderTestResponse(
            connected=False,
            error="No response from model",
        )
    except Exception as exc:
        logger.error("Provider %s test failed: %s", req.provider, exc)
        _provider_status_cache[req.provider] = "offline"
        # Extract the useful part of the error message
        msg = str(exc)
        # LiteLLM wraps errors like "AnthropicException - {json}"
        if " - " in msg:
            msg = msg.split(" - ", 1)[1]
        return ProviderTestResponse(
            connected=False,
            error=msg[:300],
        )


def _persist_api_key(provider_name: str, api_key: str) -> None:
    """Persist an API key using the most secure available backend.

    Prefers the OS keyring (encrypted, no plain-text file) when the
    ``keyring`` package is installed; falls back to the JSON file store
    (<state dir>/api_keys.json, chmod 600).  Both backends also inject the
    key into ``os.environ`` so the current process can use it immediately.
    """
    if KeyringSecretStore.is_available():
        KeyringSecretStore().set(provider_name, api_key)
    else:
        FileSecretStore().set(provider_name, api_key)


@router.put("/api/providers/{provider_name}")
async def save_provider(
    provider_name: str,
    req: ProviderSaveRequest,
    state: AppState = Depends(get_state),  # noqa: B008
) -> ProviderSaveResponse:
    """Save provider configuration and persist the API key via SecretStore.

    Uses the OS keyring when available; falls back to the JSON file store
    (<state dir>/api_keys.json).  Also persists runtime_settings and enabled
    state in AppState.config.
    """
    entry = PROVIDERS_BY_KEY.get(provider_name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider_name}")

    env_var = entry.env_var

    if req.api_key and env_var:
        # Persist via SecretStore (keyring when available, else JSON file).
        # The store also sets os.environ[env_var] for the current process.
        _persist_api_key(provider_name, req.api_key)
        os.environ[env_var] = req.api_key  # explicit fallback in case env_var lookup differs
        logger.info("Saved API key for %s (%s)", provider_name, env_var)

    # Update provider config in AppState
    _update_provider_config(state, provider_name, req)

    # If this provider is being enabled, set it as the active provider
    # and disable all others (only one active at a time)
    if req.enabled:
        state.config.active_provider = provider_name
        if req.model:
            state.config.active_model = req.model
        for pc in state.config.provider_configs:
            if pc.provider != provider_name:
                pc.enabled = False

    state.save_config()

    return ProviderSaveResponse(
        saved=True,
        provider=provider_name,
        env_var=env_var,
        message=(
            f"API key saved ({env_var})" if env_var and req.api_key else "Configuration saved"
        ),
    )


def _update_provider_config(state: AppState, provider_name: str, req: ProviderSaveRequest) -> None:
    """Upsert provider config in state.config.provider_configs."""
    configs = state.config.provider_configs

    # Find existing or create new
    existing: ProviderConfig | None = None
    for pc in configs:
        if pc.provider == provider_name:
            existing = pc
            break

    if existing is None:
        entry = PROVIDERS_BY_KEY.get(provider_name)
        default_model = entry.models[0].name if entry and entry.models else ""
        existing = ProviderConfig(
            provider=provider_name,
            model=req.model or default_model,
        )
        configs.append(existing)

    if req.model is not None:
        existing.model = req.model
    if req.endpoint is not None:
        existing.endpoint = req.endpoint
    if req.runtime_settings is not None:
        existing.runtime_settings = req.runtime_settings
    if req.enabled is not None:
        existing.enabled = req.enabled


_ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"
_OPENAI_MODELS_URL = "https://api.openai.com/v1/models"


# Families that cannot serve chat completions.  This is a DENYLIST, not an
# allowlist: a chat model the account is entitled to must appear in the
# picker without this code learning its name first.  An allowlist here is
# the same aging-list failure as hardcoding example model ids — it silently
# hides every model released after the list was written.
_NON_CHAT_MODEL_PREFIXES = (
    "text-embedding-",
    "whisper-",
    "tts-",
    "dall-e-",
    "omni-moderation-",
    "text-moderation-",
    "babbage-",
    "davinci-",
    "curie-",
    "ada-",
)


def _is_chat_capable_model(model_id: str) -> bool:
    """Whether an OpenAI model id can serve a chat completion.

    ``/v1/models`` carries no capability metadata, so the only local signal
    is the id.  Unknown ids are kept: showing a non-chat model costs one
    failed probe with a classified error, while hiding a chat model makes
    it impossible to configure.  Fine-tune ids (``ft:<base>:<org>:<suffix>``)
    are judged by their base model.
    """
    base = model_id
    if base.startswith("ft:"):
        base = base[len("ft:") :].split(":", 1)[0]
    return not base.startswith(_NON_CHAT_MODEL_PREFIXES)


def _models_endpoint_url(provider_name: str, endpoint_override: str | None) -> str:
    """Return the models URL, honoring a configured compatible gateway.

    A cloud provider may point at a compatible gateway.  Discovery must ask
    that gateway rather than the vendor — otherwise the gateway's key is
    sent to the vendor and the returned list describes the wrong service.
    The endpoint may be given with or without a trailing ``/v1``.
    """
    default = _ANTHROPIC_MODELS_URL if provider_name == "anthropic" else _OPENAI_MODELS_URL
    if not endpoint_override:
        return default
    base = re.sub(r"/v1/?$", "", endpoint_override.rstrip("/"))
    return f"{base}/v1/models"


async def _fetch_models_from_api(
    provider_name: str,
    endpoint_override: str | None = None,
    api_key_override: str | None = None,
) -> list[ModelInfo]:
    """Fetch live model list from a provider's API.

    For local providers (Ollama, LM Studio), *endpoint_override* takes
    precedence over the catalog default so users on non-standard ports
    hit the correct server.

    For cloud providers, *api_key_override* takes precedence over the
    process environment: the Settings form tests a key that has not been
    saved yet, and the account's model list must be the one that key sees.
    """
    entry = PROVIDERS_BY_KEY.get(provider_name)
    if not entry:
        return []

    # An environment key is only used for the vendor's own API.  When the
    # caller supplies a destination it must also supply the credential —
    # otherwise an arbitrary ?endpoint= would receive the process key.
    def _key_for(default_env: str) -> str:
        if api_key_override:
            return api_key_override
        return "" if endpoint_override else os.environ.get(default_env, "")

    async with httpx.AsyncClient(timeout=15) as client:
        if provider_name == "anthropic":
            api_key = _key_for("ANTHROPIC_API_KEY")
            if not api_key:
                return []
            resp = await client.get(
                _models_endpoint_url("anthropic", endpoint_override),
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            return [
                ModelInfo(
                    name=m["id"],
                    input_cost_per_1k=0.0,
                    output_cost_per_1k=0.0,
                )
                for m in data
            ]

        elif provider_name == "openai":
            api_key = _key_for("OPENAI_API_KEY")
            if not api_key:
                return []
            resp = await client.get(
                _models_endpoint_url("openai", endpoint_override),
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if endpoint_override:
                # A compatible gateway defines its own catalogue: its ids
                # are not bound by OpenAI's naming (Qwen, custom names, …),
                # so return the live list unchanged rather than filtering
                # it away.
                return [
                    ModelInfo(
                        name=m["id"],
                        input_cost_per_1k=0.0,
                        output_cost_per_1k=0.0,
                    )
                    for m in data
                ]
            # Official OpenAI endpoint: drop known non-chat families only.
            # Everything else is kept, so a model released tomorrow is not
            # hidden by a list written today.
            return [
                ModelInfo(
                    name=m["id"],
                    input_cost_per_1k=0.0,
                    output_cost_per_1k=0.0,
                )
                for m in data
                if _is_chat_capable_model(m["id"])
            ]

        elif provider_name == "ollama":
            endpoint = endpoint_override or entry.default_endpoint or "http://localhost:11434"
            resp = await client.get(f"{endpoint}/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return [
                ModelInfo(name=m["name"], input_cost_per_1k=0.0, output_cost_per_1k=0.0)
                for m in models
            ]

        elif provider_name == "lmstudio":
            endpoint = endpoint_override or entry.default_endpoint or "http://localhost:1234/v1"
            resp = await client.get(f"{endpoint}/models")
            resp.raise_for_status()
            data = resp.json().get("data", [])
            return [
                ModelInfo(name=m["id"], input_cost_per_1k=0.0, output_cost_per_1k=0.0) for m in data
            ]

        elif provider_name == "vllm":
            endpoint = endpoint_override or entry.default_endpoint or "http://localhost:8000/v1"
            resp = await client.get(f"{endpoint}/models")
            resp.raise_for_status()
            data = resp.json().get("data", [])
            return [
                ModelInfo(name=m["id"], input_cost_per_1k=0.0, output_cost_per_1k=0.0) for m in data
            ]

    return []


@router.get("/api/providers/{provider_name}/models")
async def list_provider_models(
    provider_name: str,
    endpoint: str | None = None,
    state: AppState = Depends(get_state),  # noqa: B008
) -> list[ModelInfo]:
    """Fetch models live from a provider API, with 5-minute cache and catalog fallback.

    Legacy: accepts no credential and cannot report provenance.  The Settings
    form uses ``POST /api/llm-providers/models``, which discovers with the key
    being configured, never substitutes the catalog, and is uncached.
    """
    # Cache key includes endpoint so different endpoints don't share stale data
    cache_key = f"{provider_name}:{endpoint or ''}"
    cached = _model_cache.get(cache_key)
    if cached and time.time() - cached[0] < _MODEL_CACHE_TTL:
        return cached[1]

    try:
        models = await _fetch_models_from_api(provider_name, endpoint_override=endpoint)
        if models:
            _model_cache[cache_key] = (time.time(), models)
            return models
    except Exception as exc:
        # This route can authenticate with a vendor environment key, so a
        # provider error that echoes credentials must not reach the log raw.
        logger.warning(
            "Failed to fetch models for %s: %s",
            provider_name,
            redact_secrets(str(exc))[:300],
        )

    # Fall back to hardcoded catalog
    return _build_model_infos(provider_name)
