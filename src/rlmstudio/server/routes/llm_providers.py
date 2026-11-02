"""CRUD endpoints for named LLM Provider instances."""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from rlmstudio.server.dependencies import AppState, get_state
from rlmstudio.server.models import (
    LLMProviderConfig,
    LLMProviderCreateRequest,
    LLMProviderModelsRequest,
    LLMProviderModelsResponse,
    LLMProviderUpdateRequest,
    ModelInfo,
    ProviderTestResponse,
    RuntimeSettings,
)
from rlmstudio.ui.data.providers_catalog import PROVIDERS_BY_KEY
from rlmstudio.ui.services.secret_store import FileSecretStore, KeyringSecretStore

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory status cache: llm_provider_id -> status
_status_cache: dict[str, str] = {}


def _persist_api_key(llm_provider_id: str, backend: str, api_key: str) -> None:
    """Persist API key for a named LLM Provider instance."""
    key_name = f"llm_provider:{llm_provider_id}"
    # Also set the backend env var so the process can use the key immediately
    entry = PROVIDERS_BY_KEY.get(backend)
    if entry and entry.env_var:
        os.environ[entry.env_var] = api_key

    if KeyringSecretStore.is_available():
        KeyringSecretStore().set(key_name, api_key)
    else:
        FileSecretStore().set(key_name, api_key)


def _get_api_key(llm_provider_id: str, backend: str) -> str | None:
    """Retrieve API key for a named LLM Provider (instance key first, then env var)."""
    key_name = f"llm_provider:{llm_provider_id}"
    store = KeyringSecretStore() if KeyringSecretStore.is_available() else FileSecretStore()
    raw_key: str | None = store.get(key_name)
    if raw_key:
        return raw_key
    # Fall back to env var
    entry = PROVIDERS_BY_KEY.get(backend)
    if entry and entry.env_var:
        return os.environ.get(entry.env_var)
    return None


def _discover_context_window(backend: str, model: str, endpoint: str | None) -> int | None:
    """Best-effort discovery of a model's total context window in tokens.

    Tries litellm model info first, then queries the provider's
    ``/v1/models`` endpoint for vLLM / Ollama / OpenAI-compatible backends.
    Returns None when the context window cannot be determined.
    """
    # 1. litellm model info (works for well-known models)
    try:
        import litellm

        # LiteLLM uses prefixed model names for some backends
        model_names = [model, f"{backend}/{model}"]
        if "/" in model:
            model_names.append(model.split("/", 1)[1])
        for name in model_names:
            try:
                info = litellm.get_model_info(model=name)
                ctx = info.get("max_input_tokens") or info.get("max_tokens")
                if ctx and isinstance(ctx, int) and ctx > 0:
                    logger.info("Discovered context_window=%d for %s via litellm", ctx, name)
                    return ctx
            except Exception:
                continue
    except ImportError:
        pass

    # 2. Query /v1/models endpoint (vLLM, Ollama, LM Studio)
    if endpoint and backend in ("vllm", "ollama", "lmstudio", "openai"):
        import re
        import urllib.request

        # Normalize base URL — strip trailing /v1 or /v1/ to avoid doubling
        base = re.sub(r"/v1/?$", "", endpoint.rstrip("/"))
        url = f"{base}/v1/models"

        # Build candidate model IDs for matching.  LiteLLM prefixes like
        # "openai/" are not part of the id vLLM/Ollama actually report.
        model_candidates = {model}
        if "/" in model:
            # "openai/Qwen/Qwen2.5-7B-Instruct" → "Qwen/Qwen2.5-7B-Instruct"
            model_candidates.add(model.split("/", 1)[1])
            # Also try just the final segment: "Qwen2.5-7B-Instruct"
            model_candidates.add(model.rsplit("/", 1)[-1])

        try:
            import json
            from urllib.parse import urlparse

            # B310: only allow http/https schemes to prevent file:// access
            if urlparse(url).scheme not in ("http", "https"):
                return None

            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310 — scheme validated above
                data = json.loads(resp.read())
                for m in data.get("data", []):
                    mid = m.get("id", "")
                    if mid in model_candidates or mid in {
                        c.split("/", 1)[1] for c in model_candidates if "/" in c
                    }:
                        # vLLM exposes max_model_len; Ollama may use context_length
                        ctx = (  # type: ignore[assignment]  # reuse outer name
                            m.get("max_model_len")
                            or m.get("context_length")
                            or m.get("context_window")
                        )
                        if ctx and isinstance(ctx, int) and ctx > 0:
                            logger.info(
                                "Discovered context_window=%d for %s (matched %s) via %s",
                                ctx,
                                model,
                                mid,
                                url,
                            )
                            return int(ctx)
        except Exception as exc:
            logger.debug("Could not query %s for context window: %s", url, exc)

    return None


def _compute_status(lp: LLMProviderConfig) -> str:
    """Compute current status for an LLM Provider instance."""
    # Check in-memory cache first
    cached = _status_cache.get(lp.id)
    if cached:
        return cached
    # Persisted status from last test
    if lp.status == "connected":
        return "connected"
    entry = PROVIDERS_BY_KEY.get(lp.backend)
    if not entry:
        return "not_configured"
    if entry.requires_api_key:
        key = _get_api_key(lp.id, lp.backend)
        if key:
            return lp.status if lp.status in ("connected", "configured") else "configured"
        return "not_configured"
    # Local provider (Ollama, LM Studio) — configured by default
    return lp.status if lp.status == "connected" else "configured"


@router.get("/api/llm-providers")
async def list_llm_providers(
    state: AppState = Depends(get_state),  # noqa: B008
) -> list[LLMProviderConfig]:
    """List all named LLM Provider instances with computed status."""
    result = []
    for lp in state.config.llm_providers:
        copy = lp.model_copy()
        copy.status = _compute_status(lp)
        result.append(copy)
    return result


@router.post("/api/llm-providers", status_code=201)
async def create_llm_provider(
    req: LLMProviderCreateRequest,
    state: AppState = Depends(get_state),  # noqa: B008
) -> LLMProviderConfig:
    """Create a new named LLM Provider instance."""
    if req.backend not in PROVIDERS_BY_KEY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown backend: {req.backend}. Available: {list(PROVIDERS_BY_KEY.keys())}",
        )

    with state._config_lock:
        # Name must be unique
        for existing_lp in state.config.llm_providers:
            if existing_lp.name.lower() == req.name.lower():
                raise HTTPException(
                    status_code=409,
                    detail=f"LLM Provider with name '{req.name}' already exists",
                )

        now = datetime.now(timezone.utc)
        lp = LLMProviderConfig(
            id=str(uuid.uuid4()),
            name=req.name,
            backend=req.backend,
            model=req.model,
            endpoint=req.endpoint,
            runtime_settings=req.runtime_settings or RuntimeSettings(),
            context_window=req.context_window,
            status="not_configured",
            created_at=now,
            updated_at=now,
        )

        if req.api_key:
            _persist_api_key(lp.id, req.backend, req.api_key)
            lp.status = "configured"

        # Auto-discover context window if not explicitly provided
        if lp.context_window is None:
            entry = PROVIDERS_BY_KEY.get(req.backend)
            endpoint = req.endpoint or (entry.default_endpoint if entry else None)
            discovered = _discover_context_window(req.backend, req.model, endpoint)
            if discovered:
                lp.context_window = discovered

        state.config.llm_providers.append(lp)
        state.save_config()

        result = lp.model_copy()
        result.status = _compute_status(lp)
    return result


@router.get("/api/llm-providers/{llm_provider_id}")
async def get_llm_provider(
    llm_provider_id: str,
    state: AppState = Depends(get_state),  # noqa: B008
) -> LLMProviderConfig:
    """Get a single named LLM Provider by ID."""
    lp = state.get_llm_provider(llm_provider_id)
    if not lp:
        raise HTTPException(status_code=404, detail="LLM Provider not found")
    result = lp.model_copy()
    result.status = _compute_status(lp)
    return result


@router.put("/api/llm-providers/{llm_provider_id}")
async def update_llm_provider(
    llm_provider_id: str,
    req: LLMProviderUpdateRequest,
    state: AppState = Depends(get_state),  # noqa: B008
) -> LLMProviderConfig:
    """Update an existing named LLM Provider."""
    with state._config_lock:
        lp = state.get_llm_provider(llm_provider_id)
        if not lp:
            raise HTTPException(status_code=404, detail="LLM Provider not found")

        if req.name is not None and req.name.lower() != lp.name.lower():
            for existing in state.config.llm_providers:
                if existing.id != llm_provider_id and existing.name.lower() == req.name.lower():
                    raise HTTPException(
                        status_code=409,
                        detail=f"LLM Provider with name '{req.name}' already exists",
                    )
            lp.name = req.name

        # An explicit JSON null for ``endpoint`` means "clear the override";
        # an omitted field means "leave it alone".  ``model_fields_set`` is
        # the only way to tell them apart — treating both as "omitted" kept
        # the old endpoint (and its green badge) after the user cleared it.
        endpoint_provided = "endpoint" in req.model_fields_set

        # Track whether model or endpoint changed — triggers context_window re-discovery
        model_or_endpoint_changed = (req.model is not None and req.model != lp.model) or (
            endpoint_provided and (req.endpoint or None) != lp.endpoint
        )

        if req.model is not None:
            lp.model = req.model
        if endpoint_provided:
            lp.endpoint = req.endpoint or None
        if req.runtime_settings is not None:
            lp.runtime_settings = req.runtime_settings
        if req.context_window is not None:
            lp.context_window = req.context_window if req.context_window > 0 else None
        elif model_or_endpoint_changed:
            # Model or endpoint changed but no explicit context_window override —
            # re-discover so the provider doesn't keep a stale limit.
            entry = PROVIDERS_BY_KEY.get(lp.backend)
            endpoint = lp.endpoint or (entry.default_endpoint if entry else None)
            discovered = _discover_context_window(lp.backend, lp.model, endpoint)
            lp.context_window = discovered  # None if discovery fails — safe default
            if discovered:
                logger.info(
                    "Re-discovered context_window=%d for updated provider %s (%s)",
                    discovered,
                    lp.name,
                    lp.model,
                )
        if model_or_endpoint_changed:
            # The previous probe described the old model/endpoint.  Keeping
            # "connected" would show a green badge — and a last-tested time —
            # for a configuration that has never been tested.
            lp.status = "configured"
            lp.last_tested_at = None
            lp.last_tested_by = None
            lp.consecutive_failures = 0
            _status_cache.pop(lp.id, None)

        if req.api_key:
            # A replacement key affects every probe: the stored result
            # belongs to the credentials that were tested, not these.
            _persist_api_key(lp.id, lp.backend, req.api_key)
            lp.status = "configured"
            lp.last_tested_at = None
            lp.last_tested_by = None
            lp.consecutive_failures = 0
            _status_cache.pop(lp.id, None)

        lp.updated_at = datetime.now(timezone.utc)
        state.save_config()

        result = lp.model_copy()
        result.status = _compute_status(lp)
    return result


@router.delete("/api/llm-providers/{llm_provider_id}", status_code=204)
async def delete_llm_provider(
    llm_provider_id: str,
    state: AppState = Depends(get_state),  # noqa: B008
) -> None:
    """Delete a named LLM Provider. Fails if any Chat Provider references it."""
    with state._config_lock:
        lp = state.get_llm_provider(llm_provider_id)
        if not lp:
            raise HTTPException(status_code=404, detail="LLM Provider not found")

        # Check referential integrity
        refs = [
            cp.name for cp in state.config.chat_providers if cp.llm_provider_id == llm_provider_id
        ]
        if refs:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot delete: referenced by Chat Provider(s): {', '.join(refs)}",
            )

        state.config.llm_providers = [
            lp for lp in state.config.llm_providers if lp.id != llm_provider_id
        ]
        _status_cache.pop(llm_provider_id, None)
        state.save_config()


def _resolve_saved_provider(
    state: AppState, backend: str, llm_provider_id: str | None
) -> LLMProviderConfig | None:
    """Load the provider a form request references, and verify its backend.

    A caller-supplied id must not be able to borrow one provider's stored
    secret while claiming another backend.
    """
    if not llm_provider_id:
        return None
    saved = state.get_llm_provider(llm_provider_id)
    if saved is None:
        raise HTTPException(status_code=404, detail="LLM Provider not found")
    if saved.backend != backend:
        raise HTTPException(
            status_code=400,
            detail=(
                f"LLM Provider '{saved.name}' uses backend '{saved.backend}', not '{backend}'."
            ),
        )
    return saved


@router.post("/api/llm-providers/models")
async def discover_llm_provider_models(
    req: LLMProviderModelsRequest,
    state: AppState = Depends(get_state),  # noqa: B008
) -> LLMProviderModelsResponse:
    """Load the model list for the values currently in the Settings form.

    Unlike ``GET /api/providers/{name}/models`` this route:

    * uses the API key being configured (or the saved key in edit mode),
      not only the process environment;
    * never substitutes the static catalog — a list the provider did not
      return is reported as ``source="unavailable"`` with an error, so the
      UI cannot present fallback entries as the account's live list;
    * does not cache, so the form's Refresh button really re-reads the
      endpoint.

    A saved key is only used for the destination it was saved with.  When
    the form changes the endpoint, the user must re-enter the key rather
    than have the old secret sent to a new host.
    """
    from rlmstudio.application.services.provider_failure import redact_secrets
    from rlmstudio.server.routes.providers import _fetch_models_from_api

    entry = PROVIDERS_BY_KEY.get(req.backend)
    if entry is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown backend: {req.backend}. Available: {list(PROVIDERS_BY_KEY.keys())}",
        )

    saved = _resolve_saved_provider(state, req.backend, req.llm_provider_id)
    requested_endpoint = req.endpoint or None
    endpoint_changed = saved is not None and requested_endpoint != saved.endpoint

    api_key = req.api_key or None
    if not api_key and entry.requires_api_key and endpoint_changed:
        return LLMProviderModelsResponse(
            source="unavailable",
            error=(
                "The endpoint changed, so the saved API key was not sent to it. "
                "Re-enter the API key to load models from the new destination."
            ),
        )
    if not api_key and saved is not None:
        # Edit mode with a blank field: use the key saved for this provider.
        api_key = _get_api_key(saved.id, saved.backend)
    if not api_key and entry.env_var and not (entry.requires_api_key and requested_endpoint):
        # Never send a process-environment key to an arbitrary destination.
        api_key = os.environ.get(entry.env_var)
    if entry.requires_api_key and not api_key:
        return LLMProviderModelsResponse(
            source="unavailable",
            error=f"No API key available for {entry.display_name}. Enter one above.",
        )

    try:
        models = await _fetch_models_from_api(
            req.backend,
            endpoint_override=req.endpoint,
            api_key_override=api_key,
        )
    except Exception as exc:  # noqa: BLE001 — any discovery failure is a result, not a 500
        # Sanitize once: the log and the response must not disagree about
        # whether a key-shaped value was redacted.
        error_text = redact_secrets(str(exc))[:300]
        logger.warning("Model discovery failed for %s: %s", req.backend, error_text)
        return LLMProviderModelsResponse(source="unavailable", error=error_text)

    if not models:
        return LLMProviderModelsResponse(
            source="unavailable",
            error="No models returned by the endpoint.",
        )
    return LLMProviderModelsResponse(models=models, source="live")


class LLMProviderTestRequest(LLMProviderCreateRequest):
    """Body for ``POST /api/llm-providers/test``.

    Extends the create-request shape with an optional
    ``llm_provider_id`` used by the Edit form: when the user edits a
    provider and leaves the API-key field blank (respecting the
    "leave blank to keep current key" contract), the frontend sends
    the id so the backend can resolve the saved key without the user
    having to re-paste it.
    """

    llm_provider_id: str | None = None


@router.post("/api/llm-providers/test")
async def test_llm_provider_config(
    req: LLMProviderTestRequest,
    state: AppState = Depends(get_state),  # noqa: B008
) -> ProviderTestResponse:
    """Test connection against an in-memory LLM Provider config.

    Nothing is persisted:

    * the API key is passed through ``test_provider(api_key_override=)``,
      never written to the secret store or ``os.environ``;
    * the request-scoped UUID prevents concurrent form tests from
      racing on a shared secret-store slot;
    * edit-mode requests that leave ``api_key`` blank resolve the
      saved key via ``llm_provider_id``, matching the
      "leave blank to keep current key" save-path contract.

    Credential boundaries:

    * a referenced provider must exist and must use the request's backend;
    * its saved key is never sent to an endpoint that differs from the one
      it was saved with — the user re-enters the key for a new destination;
    * a process-environment key is likewise never sent to a caller-supplied
      destination.

    Defined BEFORE the ``/{llm_provider_id}/test`` route so FastAPI's
    path resolver picks this fixed-prefix match rather than treating
    ``test`` as a literal id.
    """
    from rlmstudio.application.services.provider_tester import test_provider

    entry = PROVIDERS_BY_KEY.get(req.backend)
    if entry is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown backend: {req.backend}. Available: {list(PROVIDERS_BY_KEY.keys())}",
        )

    saved = _resolve_saved_provider(state, req.backend, req.llm_provider_id)
    requested_endpoint = req.endpoint or None
    endpoint_changed = saved is not None and requested_endpoint != saved.endpoint

    # Resolve the API key precedence:
    # 1. Explicit api_key in the request (user typed a new key).
    # 2. Saved key for the referenced provider (edit mode, field blank)
    #    — only for the endpoint it was saved with.
    # 3. None (the tester falls back to env-var lookup for providers
    #    where that makes sense; see `_build_probe_params`).
    api_key_override: str | None = req.api_key or None
    if api_key_override is None and entry.requires_api_key and endpoint_changed:
        return ProviderTestResponse(
            connected=False,
            failure_cause=("The endpoint changed, so the saved API key was not sent to it."),
            failure_fix="Re-enter the API key to test the new destination.",
        )
    if api_key_override is None and saved is not None:
        api_key_override = _get_api_key(saved.id, saved.backend)
    if api_key_override is None and entry.requires_api_key and requested_endpoint:
        return ProviderTestResponse(
            connected=False,
            failure_cause="A custom endpoint needs its API key entered in the form.",
            failure_fix="Enter the API key for this destination and test again.",
        )

    # Build an in-memory LLMProviderConfig. Request-scoped uuid avoids
    # any chance of concurrent form tests colliding on shared state.
    now = datetime.now(timezone.utc)
    lp = LLMProviderConfig(
        id=f"__ephemeral_test_{uuid.uuid4().hex}__",
        name=req.name or "ephemeral",
        backend=req.backend,
        model=req.model,
        endpoint=req.endpoint,
        runtime_settings=req.runtime_settings or RuntimeSettings(),
        context_window=req.context_window,
        status="not_configured",
        created_at=now,
        updated_at=now,
    )

    result = test_provider(lp, timeout_s=15.0, api_key_override=api_key_override)

    if result.status == "connected":
        return ProviderTestResponse(
            connected=True,
            latency_ms=result.latency_ms,
            model=lp.model,
            tested_at=result.tested_at,
        )
    return ProviderTestResponse(
        connected=False,
        error=result.error_message,
        tested_at=result.tested_at,
        failure_kind=result.failure_kind.value if result.failure_kind else None,
        failure_cause=result.failure_cause,
        failure_fix=result.failure_fix,
    )


@router.post("/api/llm-providers/{llm_provider_id}/test")
async def test_llm_provider(
    llm_provider_id: str,
    state: AppState = Depends(get_state),  # noqa: B008
) -> ProviderTestResponse:
    """Test connection to a named LLM Provider (manual / user-initiated).

    Thin wrapper over :func:`test_provider`.  Persists the result back to the
    provider (status, last_tested_at, last_tested_by, consecutive_failures)
    under the manual-path semantics defined in the spec:

    * success → ``status=connected``, ``consecutive_failures=0``.
    * failure → ``status=offline`` (immediate flip, no threshold wait),
      ``consecutive_failures=1``.  The user is actively looking at the
      result and should see the flip immediately.
    """
    from rlmstudio.application.services.provider_tester import test_provider

    lp = state.get_llm_provider(llm_provider_id)
    if not lp:
        raise HTTPException(status_code=404, detail="LLM Provider not found")

    # Manual route uses a longer timeout (15s) than the background default
    # (10s) because a user clicking "Test" will wait a bit longer for a
    # definitive answer.  Matches pre-refactor behavior.
    result = test_provider(lp, timeout_s=15.0)

    # Manual path is a distinct code path from the background threshold
    # logic (see spec §Failure Semantics).  On failure, flip immediately
    # without waiting for N consecutive failures — the user is actively
    # looking at the result and expects it to reflect the observation now.
    with state._config_lock:
        if result.status == "connected":
            lp.status = "connected"
            lp.consecutive_failures = 0
        elif result.status == "error":
            lp.status = "error"
            lp.consecutive_failures = 1
        else:
            lp.status = "offline"
            lp.consecutive_failures = 1
        lp.last_tested_at = result.tested_at
        lp.last_tested_by = "manual"
        _status_cache[lp.id] = lp.status
        state.save_config()

    if result.status == "connected":
        return ProviderTestResponse(
            connected=True,
            latency_ms=result.latency_ms,
            model=lp.model,
            tested_at=result.tested_at,
        )
    return ProviderTestResponse(
        connected=False,
        error=result.error_message,
        tested_at=result.tested_at,
        failure_kind=result.failure_kind.value if result.failure_kind else None,
        failure_cause=result.failure_cause,
        failure_fix=result.failure_fix,
    )


@router.get("/api/llm-providers/{llm_provider_id}/models")
async def list_llm_provider_models(
    llm_provider_id: str,
    state: AppState = Depends(get_state),  # noqa: B008
) -> list[ModelInfo]:
    """Fetch live models for a named LLM Provider instance.

    Uses the provider's own saved key and endpoint, so a cloud provider
    backed only by the SecretStore — or one pointed at a compatible
    gateway — gets its real list rather than the catalog fallback.
    """
    from rlmstudio.application.services.provider_failure import redact_secrets
    from rlmstudio.server.routes.providers import (
        _build_model_infos,
        _fetch_models_from_api,
    )

    lp = state.get_llm_provider(llm_provider_id)
    if not lp:
        raise HTTPException(status_code=404, detail="LLM Provider not found")

    try:
        live: list[ModelInfo] = await _fetch_models_from_api(
            lp.backend,
            endpoint_override=lp.endpoint,
            api_key_override=_get_api_key(lp.id, lp.backend),
        )
        if live:
            # A model list proves the endpoint answers, not that the configured
            # model completes a request.  It must not mark the provider
            # connected: that left a green badge with no last_tested_at.
            return live
    except Exception as exc:
        error_text = redact_secrets(str(exc))[:300]
        logger.warning("Failed to fetch models for LLM Provider %s: %s", lp.name, error_text)

    catalog: list[ModelInfo] = _build_model_infos(lp.backend)
    return catalog
