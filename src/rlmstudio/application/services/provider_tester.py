"""Shared LLM Provider connection-test logic.

Single source of truth for probing an `LLMProviderConfig` for connectivity.
Both the manual-test route (`POST /api/llm-providers/{id}/test`) and the
background connection-test thread call :func:`test_provider`.

Timeout contract (critical)
---------------------------
:func:`test_provider` MUST return within approximately ``timeout_s`` seconds
regardless of network behavior.  Python cannot cancel running threads from
outside, so callers rely on this guarantee: if the function ever hangs,
worker threads in the caller's pool leak forever.  Enforcement inside this
function:

* ``litellm.completion(timeout=timeout_s, ...)`` — the authoritative bound.
  litellm forwards this to httpx which enforces per-request connect/read
  timeouts, so an accepts-but-never-responds server still times out.
* No retries (``num_retries=0``).

We intentionally do NOT use ``socket.setdefaulttimeout``: it mutates
process-wide state, races under concurrent callers (the background thread
pool has max_workers=5), and also affects unrelated sockets in the process
(metrics, other HTTP clients) for the duration of the probe.

Any violation of the timeout contract is a bug in this module, not in the
caller.  The test ``test_test_provider_returns_within_timeout_even_when_
server_hangs`` pins the contract with an accepts-but-hangs TCP listener.

Layer note
----------
This module lives in ``application/`` but currently imports from
``server/`` (for ``LLMProviderConfig`` and ``_litellm_model_name``) and
``ui/`` (for the provider catalog and secret stores).  Those imports are
transitional; see review of commit 5791483 for the layer-cleanup
follow-up.  Until that cleanup lands, do not take the imports as a
pattern for new application-layer code.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from rlmstudio.application.services.provider_failure import (
    ProviderFailureKind,
    classify_provider_failure,
    describe_provider_failure,
    redact_secrets,
)
from rlmstudio.server.models import LLMProviderConfig
from rlmstudio.ui.data.providers_catalog import PROVIDERS_BY_KEY, request_api_key
from rlmstudio.ui.services.secret_store import FileSecretStore, KeyringSecretStore

logger = logging.getLogger("rlmstudio.application.provider_tester")

# Literal statuses — matches LLMProviderConfig.status vocabulary.
ProviderStatus = Literal["connected", "offline", "error"]

# Max characters of upstream error text we copy into `error_message`.
# Matches pre-refactor behavior.  The frontend tooltip truncates further
# visually, but the full 300 chars is useful in logs and API responses.
_ERROR_MESSAGE_MAX_LEN = 300

# litellm repeats its message after this label ("Request timed out. Error_str:
# Request timed out."); the label is internal and the repeat is noise.
_LITELLM_ERROR_STR_LABEL = " Error_str: "

# What to do when the probe times out.  The run path's advice (raise the
# runtime timeout) does not apply: the probe has its own fixed timeout, and
# the usual cause is a local server still loading the model.
_PROBE_TIMEOUT_FIX = (
    "A local server may still be loading the model, or the provider may be busy. "
    "Wait a moment, then test again."
)

# Hard cap on test duration: timeout_s + this small overhead.  Used only as a
# diagnostic log threshold; the real bound is enforced by litellm's timeout
# kwarg.
_TIMEOUT_OVERHEAD_S = 2.0

# Patterns matching common API-key shapes.  Anything matching gets replaced
# with ``<redacted>`` in error messages before the cap is applied.  Not a
# complete list — defense-in-depth on top of "callers log error_message,
# not the raw exception."  See test_error_message_does_not_contain_api_key.
_API_KEY_PATTERNS = (
    # OpenAI-style sk-..., sk-proj-..., etc.
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    # Anthropic-style
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    # Generic Bearer token in an HTTP header echoed back
    re.compile(r"[Bb]earer\s+[A-Za-z0-9_.~+/=-]{20,}"),
    # Hugging Face tokens
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    # Google-style
    re.compile(r"AIza[A-Za-z0-9_-]{20,}"),
)


@dataclass(frozen=True)
class ProviderTestResult:
    """Outcome of a single connection probe.

    Fields
    ------
    status:
        ``"connected"`` — probe reached the server and got a valid response.
        ``"offline"``   — reachable or not, but the probe failed gracefully
        (timeout, 4xx/5xx, auth error, no-choices response).  Normal failure.
        ``"error"``     — unhandled exception during probe.  Distinct from
        ``offline`` so the UI can surface a different affordance.
    tested_at:
        UTC timestamp of when the probe completed.
    latency_ms:
        Round-trip milliseconds for a successful probe.  ``None`` when the
        probe never reached the server (timeout, DNS failure, connection
        refused) or when the test errored out.
    error_message:
        Sanitized upstream error text, truncated to ~200 chars.  Never
        contains API keys, bearer tokens, or raw response bodies.
    failure_kind:
        Classification of the failure (#95), from the same classifier the
        run path uses, so both surfaces name the same cause.  ``None`` on
        success.  ``status`` stays coarse for existing consumers; this is
        the field that says *why*.
    failure_cause / failure_fix:
        User-facing text for ``failure_kind`` — what went wrong and what to
        do.  ``None`` on success.
    """

    status: ProviderStatus
    tested_at: datetime
    latency_ms: int | None
    error_message: str | None
    failure_kind: ProviderFailureKind | None = None
    failure_cause: str | None = None
    failure_fix: str | None = None


def _get_saved_key(llm_provider_id: str) -> str | None:
    """Retrieve the instance-scoped key from the secret store only."""
    key_name = f"llm_provider:{llm_provider_id}"
    store: KeyringSecretStore | FileSecretStore = (
        KeyringSecretStore() if KeyringSecretStore.is_available() else FileSecretStore()
    )
    return store.get(key_name)


def _get_env_key(backend: str) -> str | None:
    """Retrieve the backend's process-environment key, if any."""
    entry = PROVIDERS_BY_KEY.get(backend)
    if entry and entry.env_var:
        return os.environ.get(entry.env_var)
    return None


def _get_api_key(llm_provider_id: str, backend: str) -> str | None:
    """Retrieve the API key (instance-scoped first, then env var)."""
    return _get_saved_key(llm_provider_id) or _get_env_key(backend)


def resolve_provider_key(provider: LLMProviderConfig, override: str | None = None) -> str | None:
    """Resolve the key for a provider's own destination.

    Instance-scoped first; a process-environment key only when the provider
    has no custom endpoint.  Shared by the connection probe and the save-time
    model check so both enforce the same credential boundary.
    """
    api_key = override or _get_saved_key(provider.id)
    if not api_key and not provider.endpoint:
        api_key = _get_env_key(provider.backend)
    return api_key


def _sanitize_error_message(exc: BaseException) -> str:
    """Extract a short, safe description of an upstream failure.

    Sanitization steps (in order):

    1. Strip the litellm "ExceptionName - " prefix (noise for the UI).
    2. Redact anything matching the API-key patterns in
       ``_API_KEY_PATTERNS``.  litellm's auth errors sometimes echo the
       key that was rejected; httpx-layer errors can echo request
       headers including ``Authorization: Bearer ...``.
    3. Truncate to ``_ERROR_MESSAGE_MAX_LEN``.

    Callers must never log the raw exception object; use this helper.
    """
    msg = str(exc)
    if " - " in msg:
        msg = msg.split(" - ", 1)[1]
    head, label, tail = msg.partition(_LITELLM_ERROR_STR_LABEL)
    if label:
        msg = head if tail.strip() in head else f"{head} {tail}"
    # Redaction lives in provider_failure so the run path, which also shows
    # upstream text now, shares exactly these patterns (#95).
    redacted = redact_secrets(msg)
    # Hard cap — upstream may echo request bodies into error strings.
    return redacted[:_ERROR_MESSAGE_MAX_LEN]


def _build_probe_params(
    provider: LLMProviderConfig,
    timeout_s: float,
    api_key_override: str | None = None,
) -> dict[str, object]:
    """Build the kwargs passed to ``litellm.completion``.

    Mirrors the logic previously inline in ``POST /api/llm-providers/{id}/test``.
    When ``api_key_override`` is set it wins over the secret-store lookup —
    used by the stateless form-test endpoint to avoid persisting an unsaved
    key to the secret store or ``os.environ``.
    """
    # Local import so importing this module does not force litellm load
    # (keeps unit tests that don't probe fast and avoids circular imports).
    from rlmstudio.server.routes.providers import _litellm_model_name

    litellm_model = _litellm_model_name(provider.backend, provider.model)
    params: dict[str, object] = {
        "model": litellm_model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 5,
        "timeout": timeout_s,
        "drop_params": True,
        # Explicit: no retries.  Retries multiply the effective timeout and
        # can make the function overshoot its contract.
        "num_retries": 0,
    }

    # Same key chat would use: an instance key for any backend (a local server
    # may sit behind its own key), else the keyless placeholder for LM Studio.
    entry = PROVIDERS_BY_KEY.get(provider.backend)
    api_key = request_api_key(provider.backend, resolve_provider_key(provider, api_key_override))
    if api_key:
        params["api_key"] = api_key

    endpoint = provider.endpoint
    if not endpoint and entry and entry.default_endpoint:
        endpoint = entry.default_endpoint
    if endpoint:
        params["api_base"] = endpoint

    return params


def test_provider(
    provider: LLMProviderConfig,
    timeout_s: float = 10.0,  # noqa: PT028  # not a pytest function — name is a domain verb
    api_key_override: str | None = None,  # noqa: PT028
) -> ProviderTestResult:
    """Probe an LLM Provider for connectivity.

    Never raises.  All failure modes (timeout, auth, network, unknown) are
    collapsed into a :class:`ProviderTestResult`.  See module docstring for
    the timeout contract — violating it leaks threads.

    Parameters
    ----------
    provider:
        The provider config to probe.  Read fields: ``id``, ``backend``,
        ``model``, ``endpoint``.  API keys resolved via the usual precedence
        (instance-scoped secret → backend env var) UNLESS
        ``api_key_override`` is set.
    timeout_s:
        Upper bound on probe duration.  Passed to ``litellm.completion`` and
        enforced at the socket level.  The function returns within
        ``timeout_s + ~2s`` on a well-behaved client library.
    api_key_override:
        When non-None, use this key directly instead of looking one up.
        Intended for the stateless form-test path, where the user's
        unsaved key must be probed without persisting it to the secret
        store or ``os.environ``.

    Returns
    -------
    ProviderTestResult
        Always; never raises.
    """
    logger.debug("probe attempt provider_id=%s timeout_s=%s", provider.id, timeout_s)
    start = time.monotonic()

    try:
        try:
            import litellm
        except ImportError as exc:
            # Should never happen in a configured RLMKit install, but do not
            # let it crash the caller.
            tested_at = datetime.now(timezone.utc)
            logger.error("litellm import failed: %s", exc)
            return ProviderTestResult(
                status="error",
                tested_at=tested_at,
                latency_ms=None,
                error_message="litellm not installed",
            )

        params = _build_probe_params(provider, timeout_s, api_key_override=api_key_override)
        try:
            response = litellm.completion(**params)
        except Exception as exc:  # noqa: BLE001 — any litellm failure = offline probe result
            tested_at = datetime.now(timezone.utc)
            elapsed = time.monotonic() - start
            msg = _sanitize_error_message(exc)
            logger.debug(
                "probe result provider_id=%s status=offline latency_ms=None error=%s elapsed=%.2fs",
                provider.id,
                msg,
                elapsed,
            )
            if elapsed > timeout_s + _TIMEOUT_OVERHEAD_S:
                logger.warning(
                    "test_provider overshot timeout_s=%.1fs by %.1fs — "
                    "contract violation, check timeout enforcement in the probe",
                    timeout_s,
                    elapsed - timeout_s,
                )
            # Classify through the shared classifier (#95) so the probe names
            # the same cause the run path does. `status` stays coarse for
            # existing consumers; `failure_kind` is what says why.
            kind = classify_provider_failure(exc)
            failure = describe_provider_failure(
                kind,
                provider_name=provider.name,
                model=provider.model,
                api_base=provider.endpoint,
            )
            fix = _PROBE_TIMEOUT_FIX if kind is ProviderFailureKind.TIMEOUT else failure.fix
            return ProviderTestResult(
                status="offline",
                tested_at=tested_at,
                latency_ms=None,
                error_message=msg,
                failure_kind=kind,
                failure_cause=failure.cause,
                failure_fix=fix,
            )

        # Success path
        tested_at = datetime.now(timezone.utc)
        latency_ms = int((time.monotonic() - start) * 1000)
        if not getattr(response, "choices", None):
            logger.debug(
                "probe result provider_id=%s status=offline (empty choices)",
                provider.id,
            )
            return ProviderTestResult(
                status="offline",
                tested_at=tested_at,
                latency_ms=latency_ms,
                error_message="No response from model",
            )
        logger.debug(
            "probe result provider_id=%s status=connected latency_ms=%d",
            provider.id,
            latency_ms,
        )
        return ProviderTestResult(
            status="connected",
            tested_at=tested_at,
            latency_ms=latency_ms,
            error_message=None,
        )
    except Exception as exc:  # noqa: BLE001 — never raise from this function
        # Truly unexpected (non-litellm) exception.  Distinct "error" status.
        tested_at = datetime.now(timezone.utc)
        logger.exception(
            "unhandled error probing provider_id=%s — classified as error status",
            provider.id,
        )
        return ProviderTestResult(
            status="error",
            tested_at=tested_at,
            latency_ms=None,
            error_message=_sanitize_error_message(exc),
        )


# Pytest auto-collects any callable whose name matches ``python_functions``
# ("test_*") from the test-module namespace.  Because :func:`test_provider`
# is a domain verb that tests import directly, mark it as non-test so pytest
# skips it instead of treating it as a test case missing fixtures.
test_provider.__test__ = False  # type: ignore[attr-defined]


def record_failure_classification(provider: LLMProviderConfig, result: ProviderTestResult) -> None:
    """Persist (or clear) the last failure classification on *provider*.

    Called by the manual test route and the background cycle.  A successful
    probe clears the fields; any failure stores the classifier's kind, cause
    and fix, so a row can state them after the transient test banner is gone.
    """
    if result.status == "connected":
        provider.last_failure_kind = None
        provider.last_failure_cause = None
        provider.last_failure_fix = None
        return
    provider.last_failure_kind = result.failure_kind.value if result.failure_kind else None
    provider.last_failure_cause = result.failure_cause
    provider.last_failure_fix = result.failure_fix


def record_model_check(
    provider: LLMProviderConfig,
    model_names: list[str],
    *,
    checked_at: datetime | None = None,
) -> None:
    """Record whether the provider's configured model is in its live list.

    An empty list means discovery did not return anything — the check is
    ``unknown`` rather than claiming the model is gone.
    """
    if not model_names:
        provider.model_check_status = "unknown"
        provider.model_check_alternatives = None
    elif provider.model in model_names:
        provider.model_check_status = "available"
        provider.model_check_alternatives = None
    else:
        provider.model_check_status = "unavailable"
        provider.model_check_alternatives = model_names[:5]
    provider.model_checked_at = checked_at or datetime.now(timezone.utc)


def record_model_check_unavailable(
    provider: LLMProviderConfig,
    model_names: list[str],
    *,
    checked_at: datetime | None = None,
) -> None:
    """Record a probe-proven missing model, keeping alternatives if fetched."""
    provider.model_check_status = "unavailable"
    provider.model_check_alternatives = model_names[:5] or None
    provider.model_checked_at = checked_at or datetime.now(timezone.utc)


def record_model_check_from_probe(provider: LLMProviderConfig, result: ProviderTestResult) -> None:
    """Update the model check from a probe outcome, with no network call.

    A bad key or an unreachable endpoint says nothing about the model, so
    those outcomes leave the previous check untouched.
    """
    if result.status == "connected":
        provider.model_check_status = "available"
        provider.model_check_alternatives = None
        provider.model_checked_at = result.tested_at
    elif result.failure_kind is ProviderFailureKind.MODEL_UNAVAILABLE:
        provider.model_check_status = "unavailable"
        provider.model_checked_at = result.tested_at
