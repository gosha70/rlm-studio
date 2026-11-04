"""Diagnostics endpoint — powers the Learn tab persistent strip.

Returns four lightweight health checks (backend, provider, judge,
storage) so the client can render a status row on every Learn page.

Check semantics match the pitch's red-dot badge policy:
``error`` states are hard blockers (no LLM provider configured,
storage unreachable); ``warn`` is a soft issue (judge missing);
``ok`` passes. Only ``error`` drives the sidebar badge — see
``pitch-learn-tab.md`` §Diagnostics red-dot badge.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from rlmstudio.server.dependencies import AppState, get_state
from rlmstudio.server.models import DiagnosticCheck, DiagnosticsResponse
from rlmstudio.server.routes.llm_providers import _compute_status

router = APIRouter()


@router.get("/api/diagnostics")
async def get_diagnostics(
    state: AppState = Depends(get_state),  # noqa: B008
) -> DiagnosticsResponse:
    """Run the four Learn-tab diagnostic checks and return their status.

    Budget: sub-500 ms. Per-check costs:

    * ``backend``: no I/O; the endpoint itself is the probe.
    * ``provider``: iterates in-memory ``llm_providers``. Calls
      :func:`_compute_status` per entry. For an untested provider it reads the
      SecretStore (file or keyring) once per API-key provider —
      typically <1 ms for the file store and well under 100 ms for the
      macOS keyring on already-authenticated processes. Local backends
      (Ollama, LM Studio) short-circuit without any I/O. No network
      calls or LLM pings.
    * ``judge``: in-memory config only.
    * ``storage``: one ``SELECT COUNT(*)`` against SQLite.
    """
    return DiagnosticsResponse(
        backend=_check_backend(),
        provider=_check_provider(state),
        judge=_check_judge(state),
        storage=_check_storage(state),
    )


def _check_backend() -> DiagnosticCheck:
    # If this code runs, the backend is up — the endpoint itself is the probe.
    return DiagnosticCheck(status="ok", message="Backend reachable")


def _check_provider(state: AppState) -> DiagnosticCheck:
    # Source of truth: the same _compute_status() the LLM-provider
    # listing and chat UI use — not the raw persisted status field.
    # That distinction matters: local backends (Ollama, LM Studio) and
    # API-key backends whose key is in an env var (or SecretStore) are
    # effectively "configured" even though the persisted record still
    # says "not_configured". Diagnostics must not false-negative on
    # setups that Settings already accepts as usable.
    #
    # Legacy provider_configs are checked as a backward-compat fallback
    # for installs that predate named LLM Providers.
    llm_configured = [
        lp for lp in state.config.llm_providers if _compute_status(lp) != "not_configured"
    ]
    legacy_enabled = [pc for pc in state.config.provider_configs if pc.enabled]

    if llm_configured or legacy_enabled:
        total = len(llm_configured) + len(legacy_enabled)
        return DiagnosticCheck(
            status="ok",
            message=f"{total} LLM provider(s) configured",
        )
    return DiagnosticCheck(
        status="error",
        message="No LLM provider configured",
        fixUrl="/settings",
    )


def _check_judge(state: AppState) -> DiagnosticCheck:
    if state.config.judge_chat_provider_id:
        return DiagnosticCheck(status="ok", message="Judge configured")
    # Judge missing is a soft warning, not an error — see pitch §Resolved
    # Decisions #6. It does not drive the red-dot sidebar badge.
    return DiagnosticCheck(
        status="warn",
        message="Judge not configured",
        fixUrl="/settings",
    )


def _check_storage(state: AppState) -> DiagnosticCheck:
    # Read-only probe: the telemetry store's count_runs() issues a
    # SELECT COUNT(*). That confirms the DB is reachable and readable,
    # not that it is writable. The "ok" message reflects that narrower
    # guarantee honestly; a genuine write-broken store still trips
    # "error" via the actual write paths (record_run) in normal use.
    #
    # fixUrl is intentionally omitted until /learn/troubleshoot ships
    # in Step 6 — a clickable fixUrl that lands on 404 would be worse
    # than no link at all.
    try:
        state.telemetry.count_runs()
    except Exception as exc:  # noqa: BLE001 — any failure means storage is unhealthy
        return DiagnosticCheck(
            status="error",
            message=f"Storage unavailable: {exc.__class__.__name__}",
        )
    return DiagnosticCheck(status="ok", message="Storage reachable")
