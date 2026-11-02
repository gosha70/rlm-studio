"""Unit tests for the model-vs-account check orchestration (#95).

The route and startup tests exercise it end to end; these pin the two
concurrency contracts directly: the snapshot and its credential are taken
atomically, and a provider edited during the fetch is discarded.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

from rlmstudio.server import dependencies
from rlmstudio.server.dependencies import reset_state
from rlmstudio.server.models import LLMProviderConfig, ModelInfo
from rlmstudio.server.provider_checks import refresh_model_check

_FETCH = "rlmstudio.server.routes.providers._fetch_models_from_api"


def _state_with_provider() -> tuple[dependencies.AppState, LLMProviderConfig]:
    reset_state()
    state = dependencies.get_state()
    lp = LLMProviderConfig(
        id="lp-1",
        name="gw",
        backend="openai",
        model="gpt-4o-mini",
        endpoint="https://old.example/v1",
    )
    state.config.llm_providers.append(lp)
    return state, lp


def test_snapshot_and_credential_are_taken_under_the_lock() -> None:
    """Endpoint and key must be paired atomically.

    Resolving them separately lets a concurrent update send the new key to
    the old destination.
    """
    state, lp = _state_with_provider()
    captured: dict[str, Any] = {}

    def _fake_resolve(provider: LLMProviderConfig) -> str:
        # RLock ownership is thread-local; True here means the snapshot and
        # the credential lookup happen inside the same critical section.
        captured["lock_held"] = state._config_lock._is_owned()  # type: ignore[attr-defined]
        captured["resolved_endpoint"] = provider.endpoint
        return "sk-old"

    async def _fake_fetch(
        backend: str,
        *,
        endpoint_override: str | None = None,
        api_key_override: str | None = None,
        timeout: float = 15.0,
    ) -> list[ModelInfo]:
        captured["fetch_endpoint"] = endpoint_override
        captured["fetch_key"] = api_key_override
        return [ModelInfo(name="gpt-4o-mini")]

    with (
        patch("rlmstudio.server.provider_checks.resolve_provider_key", _fake_resolve),
        patch(_FETCH, _fake_fetch),
    ):
        asyncio.run(refresh_model_check(state, lp.id))

    assert captured["lock_held"] is True
    assert captured["resolved_endpoint"] == "https://old.example/v1"
    assert captured["fetch_endpoint"] == captured["resolved_endpoint"]
    assert captured["fetch_key"] == "sk-old"
    assert lp.model_check_status == "available"


def test_an_edit_during_the_fetch_discards_the_result() -> None:
    """The fetch uses the snapshot, and the revision guard drops its result."""
    state, lp = _state_with_provider()

    async def _fake_fetch(
        backend: str,
        *,
        endpoint_override: str | None = None,
        api_key_override: str | None = None,
        timeout: float = 15.0,
    ) -> list[ModelInfo]:
        assert endpoint_override == "https://old.example/v1"
        # A key rotation lands while the request is in flight.
        lp.updated_at = datetime.now(timezone.utc)
        return [ModelInfo(name="gpt-4o-mini")]

    with (
        patch("rlmstudio.server.provider_checks.resolve_provider_key", lambda _p: None),
        patch(_FETCH, _fake_fetch),
    ):
        asyncio.run(refresh_model_check(state, lp.id))

    assert lp.model_check_status is None
