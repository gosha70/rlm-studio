"""Model-vs-account validation for LLM Providers (#95 Scope 2).

Server layer by design: it orchestrates the discovery route's fetch with the
application-layer recorders.  ``provider_tester.py`` stays network-free.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from rlmstudio.application.services.provider_failure import redact_secrets
from rlmstudio.application.services.provider_tester import (
    record_model_check,
    record_model_check_unavailable,
    resolve_provider_key,
)
from rlmstudio.server.routes import providers as _providers

if TYPE_CHECKING:
    from rlmstudio.server.dependencies import AppState

logger = logging.getLogger(__name__)

# Bound for the save-time and load-time model check.  Long enough for a cloud
# list call, short enough that a dead endpoint does not stall a save.
MODEL_CHECK_TIMEOUT_S = 5.0
_ERROR_MESSAGE_MAX_LEN = 300


async def refresh_model_check(
    state: AppState,
    llm_provider_id: str,
    *,
    force_unavailable: bool = False,
    stop_event: threading.Event | None = None,
) -> None:
    """Compare the provider's model with its live list and persist the result.

    Bounded to ``MODEL_CHECK_TIMEOUT_S``.  A failed fetch records
    ``unknown``; this never raises, so a slow or dead provider cannot fail a
    save or the startup probe.

    The guard discards a result whose provider changed while the request was
    in flight.  ``updated_at`` is part of the comparison on purpose: rotating
    the saved key does not change the model, endpoint or backend, so without
    it an old-key discovery could overwrite the state the edit just cleared.
    """
    lp = state.get_llm_provider(llm_provider_id)
    if lp is None:
        return
    backend, model, endpoint, revision = lp.backend, lp.model, lp.endpoint, lp.updated_at

    names: list[str] = []
    try:
        # Attribute lookup at call time, so tests and the autouse suite stub
        # that patch the providers module take effect here too.
        models = await _providers._fetch_models_from_api(
            backend,
            endpoint_override=endpoint,
            # Same credential boundary as the probe: an environment key is
            # never sent to a custom endpoint.
            api_key_override=resolve_provider_key(lp),
            timeout=MODEL_CHECK_TIMEOUT_S,
        )
        names = [m.name for m in models]
    except Exception as exc:  # noqa: BLE001 — a failed check is a recorded state
        # Provider errors can echo authorization headers or response bodies.
        logger.debug(
            "Model check failed for provider %s: %s",
            lp.name,
            redact_secrets(str(exc))[:_ERROR_MESSAGE_MAX_LEN],
        )

    if stop_event is not None and stop_event.is_set():
        return
    with state._config_lock:
        if stop_event is not None and stop_event.is_set():
            return
        current = state.get_llm_provider(llm_provider_id)
        if (
            current is None
            or current.model != model
            or current.endpoint != endpoint
            or current.updated_at != revision
        ):
            return
        if force_unavailable:
            record_model_check_unavailable(current, names)
        else:
            record_model_check(current, names)
        state.save_config()
