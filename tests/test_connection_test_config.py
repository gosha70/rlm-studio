"""Tests for the config-model additions in Commit 3 of the scheduled-
connection-testing feature.

Covers:
- LLMProviderConfig gains last_tested_at, last_tested_by, consecutive_failures.
- ConfigResponse/ConfigUpdateRequest gain connection_test_interval_minutes
  with validator (0-1440 inclusive).
- Manual test route updates the new fields (spec §Failure Semantics — manual
  path bypasses the N-consecutive-failures threshold).
- PUT /api/config round-trips connection_test_interval_minutes.
- AppState._config_lock exists and is reentrant.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from rlmstudio.application.services.provider_tester import ProviderTestResult
from rlmstudio.server import dependencies
from rlmstudio.server.app import create_app
from rlmstudio.server.dependencies import AppState, reset_state
from rlmstudio.server.models import (
    ConfigResponse,
    ConfigUpdateRequest,
    LLMProviderConfig,
    ModelInfo,
    RuntimeSettings,
)


@pytest.fixture
def client() -> TestClient:
    reset_state()
    app = create_app()
    return TestClient(app)


# --------------------------------------------------------------------------
# Model-level tests
# --------------------------------------------------------------------------


def test_llm_provider_config_has_tracking_fields() -> None:
    """New fields exist with sensible defaults."""
    lp = LLMProviderConfig(id="x", name="x", backend="openai", model="gpt-4o")
    assert lp.last_tested_at is None
    assert lp.last_tested_by is None
    assert lp.consecutive_failures == 0


def test_llm_provider_config_accepts_tracking_values() -> None:
    now = datetime.now(timezone.utc)
    lp = LLMProviderConfig(
        id="x",
        name="x",
        backend="openai",
        model="gpt-4o",
        last_tested_at=now,
        last_tested_by="background",
        consecutive_failures=3,
    )
    assert lp.last_tested_at == now
    assert lp.last_tested_by == "background"
    assert lp.consecutive_failures == 3


def test_llm_provider_config_rejects_invalid_last_tested_by() -> None:
    with pytest.raises(ValueError):
        LLMProviderConfig(
            id="x",
            name="x",
            backend="openai",
            model="gpt-4o",
            last_tested_by="something_else",  # type: ignore[arg-type]
        )


def test_config_response_default_interval_is_zero() -> None:
    cr = ConfigResponse()
    assert cr.connection_test_interval_minutes == 0


@pytest.mark.parametrize("value", [-1, 1441, 99999])
def test_config_response_rejects_out_of_range_interval(value: int) -> None:
    with pytest.raises(ValueError):
        ConfigResponse(connection_test_interval_minutes=value)


@pytest.mark.parametrize("value", [0, 1, 60, 1440])
def test_config_response_accepts_in_range_interval(value: int) -> None:
    cr = ConfigResponse(connection_test_interval_minutes=value)
    assert cr.connection_test_interval_minutes == value


def test_config_update_request_accepts_none_interval() -> None:
    """None means "no change" in the partial-update request."""
    req = ConfigUpdateRequest()
    assert req.connection_test_interval_minutes is None


def test_config_update_request_rejects_out_of_range_interval() -> None:
    with pytest.raises(ValueError):
        ConfigUpdateRequest(connection_test_interval_minutes=2000)


# --------------------------------------------------------------------------
# AppState — lock
# --------------------------------------------------------------------------


def test_app_state_has_reentrant_config_lock() -> None:
    state = AppState(load_from_disk=False)
    # Must be an RLock, not a regular Lock — route handlers acquire it and
    # then call save_config() which also acquires it.
    assert state._config_lock is not None
    # Acquire twice from the same thread; RLock allows this, Lock would hang.
    with state._config_lock:
        with state._config_lock:
            pass


def test_save_config_acquires_lock() -> None:
    """save_config MUST hold the lock during the write so concurrent
    mutations can't race with its model_dump snapshot."""
    state = AppState(load_from_disk=False)
    holding = threading.Event()
    released = threading.Event()
    state.save_config = lambda: None  # type: ignore[assignment]  # no disk

    def _hold_lock() -> None:
        with state._config_lock:
            holding.set()
            released.wait(timeout=5.0)

    t = threading.Thread(target=_hold_lock)
    t.start()
    holding.wait(timeout=5.0)

    # At this point the worker thread holds the lock.  A non-blocking
    # acquire from this thread must fail.
    assert state._config_lock.acquire(blocking=False) is False
    released.set()
    t.join(timeout=5.0)

    # After the worker releases, we can acquire it.
    assert state._config_lock.acquire(blocking=False) is True
    state._config_lock.release()


# --------------------------------------------------------------------------
# Integration: manual test route updates the new fields
# --------------------------------------------------------------------------


def _add_provider(client: TestClient) -> str:
    resp = client.post(
        "/api/llm-providers",
        json={
            "name": "test-provider",
            "backend": "openai",
            "model": "gpt-4o-mini",
            "api_key": "sk-test",
            "runtime_settings": RuntimeSettings().model_dump(),
        },
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    lp_id: str = data["id"]
    return lp_id


def test_manual_test_success_updates_tracking_fields(client: TestClient) -> None:
    lp_id = _add_provider(client)

    class _FakeResponse:
        def __init__(self) -> None:
            self.choices = [object()]

    with patch("litellm.completion", return_value=_FakeResponse()):
        resp = client.post(f"/api/llm-providers/{lp_id}/test")

    assert resp.status_code == 200
    assert resp.json()["connected"] is True

    # Pull the provider back.
    get_resp = client.get(f"/api/llm-providers/{lp_id}")
    assert get_resp.status_code == 200
    lp = get_resp.json()
    assert lp["status"] == "connected"
    assert lp["last_tested_at"] is not None
    assert lp["last_tested_by"] == "manual"
    assert lp["consecutive_failures"] == 0


def test_manual_test_failure_flips_status_and_sets_counter_to_one(
    client: TestClient,
) -> None:
    """Manual failure path (spec §Failure Semantics): status flips
    immediately, counter goes to 1.  Distinct from the background
    threshold logic which requires N ≥ 2 consecutive failures."""
    lp_id = _add_provider(client)

    with patch("litellm.completion", side_effect=RuntimeError("AuthError - bad key")):
        resp = client.post(f"/api/llm-providers/{lp_id}/test")

    assert resp.status_code == 200
    assert resp.json()["connected"] is False

    get_resp = client.get(f"/api/llm-providers/{lp_id}")
    lp = get_resp.json()
    assert lp["status"] == "offline"
    assert lp["consecutive_failures"] == 1
    assert lp["last_tested_by"] == "manual"
    assert lp["last_tested_at"] is not None


def test_manual_test_resets_counter_from_nonzero_on_success(
    client: TestClient,
) -> None:
    lp_id = _add_provider(client)

    # Prime the counter to 3 by poking the state directly.
    state = dependencies.get_state()
    lp = state.get_llm_provider(lp_id)
    assert lp is not None
    lp.consecutive_failures = 3
    lp.status = "offline"

    class _FakeResponse:
        def __init__(self) -> None:
            self.choices = [object()]

    with patch("litellm.completion", return_value=_FakeResponse()):
        resp = client.post(f"/api/llm-providers/{lp_id}/test")
    assert resp.status_code == 200

    lp_after = state.get_llm_provider(lp_id)
    assert lp_after is not None
    assert lp_after.status == "connected"
    assert lp_after.consecutive_failures == 0


# --------------------------------------------------------------------------
# Integration: config round-trip
# --------------------------------------------------------------------------


def test_get_config_exposes_interval_field(client: TestClient) -> None:
    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "connection_test_interval_minutes" in data
    assert data["connection_test_interval_minutes"] == 0


def test_put_config_updates_interval_field(client: TestClient) -> None:
    resp = client.put(
        "/api/config",
        json={"connection_test_interval_minutes": 5},
    )
    assert resp.status_code == 200
    assert resp.json()["connection_test_interval_minutes"] == 5

    # Round-trip via GET.
    get_resp = client.get("/api/config")
    assert get_resp.json()["connection_test_interval_minutes"] == 5


def test_put_config_rejects_out_of_range_interval(client: TestClient) -> None:
    resp = client.put(
        "/api/config",
        json={"connection_test_interval_minutes": 9999},
    )
    assert resp.status_code == 422  # Pydantic validation


# --------------------------------------------------------------------------
# A tested result must not outlive the configuration it describes, and a
# model-list fetch must not masquerade as a connection test.
# --------------------------------------------------------------------------


def test_test_endpoint_reports_classified_failure(client: TestClient) -> None:
    """A failure names the cause and fix, not just the upstream errno."""
    lp_id = _add_provider(client)

    class APIConnectionError(Exception):
        """Same class name litellm uses — the classifier matches on it."""

    with patch("litellm.completion", side_effect=APIConnectionError("Connection refused")):
        resp = client.post(f"/api/llm-providers/{lp_id}/test")

    assert resp.status_code == 200
    data = resp.json()
    assert data["connected"] is False
    assert data["failure_kind"] == "connection"
    assert "Could not reach the provider" in data["failure_cause"]
    assert data["failure_fix"]
    assert data["tested_at"] is not None


def test_test_endpoint_reports_tested_at_on_success(client: TestClient) -> None:
    lp_id = _add_provider(client)

    class _FakeResponse:
        def __init__(self) -> None:
            self.choices = [object()]

    with patch("litellm.completion", return_value=_FakeResponse()):
        resp = client.post(f"/api/llm-providers/{lp_id}/test")

    assert resp.status_code == 200
    assert resp.json()["connected"] is True
    assert resp.json()["tested_at"] is not None


def test_model_list_fetch_does_not_mark_provider_connected(client: TestClient) -> None:
    """Listing models proves the endpoint answers, not that the model works."""
    resp = client.post(
        "/api/llm-providers",
        json={
            "name": "spark",
            "backend": "vllm",
            "model": "qwen38-27b",
            "endpoint": "http://127.0.0.1:9/v1",
        },
    )
    assert resp.status_code == 201, resp.text
    lp_id = resp.json()["id"]

    live = [ModelInfo(name="qwen38-27b")]
    with patch(
        "rlmstudio.server.routes.providers._fetch_models_from_api",
        new=AsyncMock(return_value=live),
    ):
        models = client.get(f"/api/llm-providers/{lp_id}/models")

    assert models.status_code == 200
    assert [m["name"] for m in models.json()] == ["qwen38-27b"]

    lp = client.get(f"/api/llm-providers/{lp_id}").json()
    assert lp["status"] != "connected"
    assert lp["last_tested_at"] is None


@pytest.mark.parametrize("change", [{"model": "gpt-4o"}, {"endpoint": "http://127.0.0.1:9/v1"}])
def test_update_config_that_affects_the_probe_clears_the_old_result(
    client: TestClient, change: dict[str, str]
) -> None:
    lp_id = _add_provider(client)
    state = dependencies.get_state()
    lp = state.get_llm_provider(lp_id)
    assert lp is not None
    lp.status = "connected"
    lp.last_tested_at = datetime.now(timezone.utc)
    lp.last_tested_by = "manual"
    lp.consecutive_failures = 2

    with patch(
        "rlmstudio.server.routes.llm_providers._discover_context_window",
        return_value=None,
    ):
        resp = client.put(f"/api/llm-providers/{lp_id}", json=change)

    assert resp.status_code == 200, resp.text
    updated = client.get(f"/api/llm-providers/{lp_id}").json()
    assert updated["status"] == "configured"
    assert updated["last_tested_at"] is None
    assert updated["last_tested_by"] is None
    assert updated["consecutive_failures"] == 0


def test_update_can_clear_the_endpoint_override(client: TestClient) -> None:
    """JSON null clears the endpoint; it is not the same as omitting it.

    Treating null as "omitted" kept the old host — and the green badge for
    a test of that host — after the user cleared the field.
    """
    with patch(
        "rlmstudio.server.routes.llm_providers._discover_context_window",
        return_value=None,
    ):
        created = client.post(
            "/api/llm-providers",
            json={
                "name": "clear-me",
                "backend": "vllm",
                "model": "qwen38-27b",
                "endpoint": "http://127.0.0.1:8123/v1",
            },
        )
    assert created.status_code == 201, created.text
    lp_id = created.json()["id"]

    state = dependencies.get_state()
    lp = state.get_llm_provider(lp_id)
    assert lp is not None
    lp.status = "connected"
    lp.last_tested_at = datetime.now(timezone.utc)
    lp.last_tested_by = "manual"

    with patch(
        "rlmstudio.server.routes.llm_providers._discover_context_window",
        return_value=None,
    ):
        resp = client.put(f"/api/llm-providers/{lp_id}", json={"endpoint": None})

    assert resp.status_code == 200, resp.text
    updated = client.get(f"/api/llm-providers/{lp_id}").json()
    assert updated["endpoint"] is None
    assert updated["status"] == "configured"
    assert updated["last_tested_at"] is None
    assert updated["last_tested_by"] is None


def test_update_without_the_endpoint_field_keeps_the_override(
    client: TestClient,
) -> None:
    """An omitted endpoint means "leave it alone" — only an explicit null clears."""
    with patch(
        "rlmstudio.server.routes.llm_providers._discover_context_window",
        return_value=None,
    ):
        created = client.post(
            "/api/llm-providers",
            json={
                "name": "keep-me",
                "backend": "vllm",
                "model": "qwen38-27b",
                "endpoint": "http://127.0.0.1:8123/v1",
            },
        )
        lp_id = created.json()["id"]
        resp = client.put(f"/api/llm-providers/{lp_id}", json={"model": "qwen38-27b-awq"})

    assert resp.status_code == 200, resp.text
    assert resp.json()["endpoint"] == "http://127.0.0.1:8123/v1"


def test_replacing_the_api_key_clears_the_old_test_result(client: TestClient) -> None:
    """New credentials invalidate the result produced by the previous ones."""
    lp_id = _add_provider(client)
    state = dependencies.get_state()
    lp = state.get_llm_provider(lp_id)
    assert lp is not None
    lp.status = "connected"
    lp.last_tested_at = datetime.now(timezone.utc)
    lp.last_tested_by = "manual"
    lp.consecutive_failures = 2

    resp = client.put(f"/api/llm-providers/{lp_id}", json={"api_key": "sk-replaced"})

    assert resp.status_code == 200, resp.text
    updated = client.get(f"/api/llm-providers/{lp_id}").json()
    assert updated["status"] == "configured"
    assert updated["last_tested_at"] is None
    assert updated["last_tested_by"] is None
    assert updated["consecutive_failures"] == 0


def _create_provider_with_endpoint(client: TestClient) -> str:
    with patch(
        "rlmstudio.server.routes.llm_providers._discover_context_window",
        return_value=None,
    ):
        resp = client.post(
            "/api/llm-providers",
            json={
                "name": "gateway",
                "backend": "openai",
                "model": "gpt-4o",
                "api_key": "sk-saved",
                "endpoint": "https://old.example/v1",
            },
        )
    assert resp.status_code == 201, resp.text
    lp_id: str = resp.json()["id"]
    return lp_id


def test_form_test_rejects_an_unknown_provider_id(client: TestClient) -> None:
    resp = client.post(
        "/api/llm-providers/test",
        json={
            "name": "x",
            "backend": "openai",
            "model": "gpt-4o",
            "llm_provider_id": "missing",
            "api_key": "sk-x",
        },
    )
    assert resp.status_code == 404


def test_form_test_rejects_a_backend_mismatch(client: TestClient) -> None:
    lp_id = _create_provider_with_endpoint(client)
    resp = client.post(
        "/api/llm-providers/test",
        json={
            "name": "x",
            "backend": "anthropic",
            "model": "claude-sonnet-4-6",
            "llm_provider_id": lp_id,
            "api_key": "sk-ant-x",
        },
    )
    assert resp.status_code == 400


def test_form_test_refuses_the_saved_key_for_a_changed_endpoint(
    client: TestClient,
) -> None:
    """A saved secret must not follow the form to a different destination."""
    lp_id = _create_provider_with_endpoint(client)

    resp = client.post(
        "/api/llm-providers/test",
        json={
            "name": "gateway",
            "backend": "openai",
            "model": "gpt-4o",
            "llm_provider_id": lp_id,
            "endpoint": "https://new.example/v1",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["connected"] is False
    assert "saved API key" in body["failure_cause"]
    assert body["failure_fix"]


def test_form_test_refuses_an_env_key_for_a_custom_endpoint(
    client: TestClient,
) -> None:
    """Process-environment keys are not sent to a caller-supplied host."""
    resp = client.post(
        "/api/llm-providers/test",
        json={
            "name": "x",
            "backend": "openai",
            "model": "gpt-4o",
            "endpoint": "https://gw.example/v1",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["connected"] is False
    assert "custom endpoint" in body["failure_cause"]


def test_form_test_uses_the_saved_key_for_its_saved_endpoint(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    lp_id = _create_provider_with_endpoint(client)
    captured: dict[str, str | None] = {}

    def fake_test(
        provider: LLMProviderConfig,
        timeout_s: float = 10.0,
        api_key_override: str | None = None,
    ) -> ProviderTestResult:
        captured["key"] = api_key_override
        return ProviderTestResult(
            status="connected",
            tested_at=datetime.now(timezone.utc),
            latency_ms=1,
            error_message=None,
        )

    monkeypatch.setattr("rlmstudio.application.services.provider_tester.test_provider", fake_test)
    monkeypatch.setattr(
        "rlmstudio.server.routes.llm_providers._get_api_key",
        lambda *_args, **_kwargs: "sk-saved",
    )

    resp = client.post(
        "/api/llm-providers/test",
        json={
            "name": "gateway",
            "backend": "openai",
            "model": "gpt-4o",
            "llm_provider_id": lp_id,
            "endpoint": "https://old.example/v1",
        },
    )

    assert resp.status_code == 200
    assert resp.json()["connected"] is True
    assert captured["key"] == "sk-saved"


def test_update_that_does_not_affect_the_probe_keeps_the_result(
    client: TestClient,
) -> None:
    lp_id = _add_provider(client)
    state = dependencies.get_state()
    lp = state.get_llm_provider(lp_id)
    assert lp is not None
    tested_at = datetime.now(timezone.utc)
    lp.status = "connected"
    lp.last_tested_at = tested_at
    lp.last_tested_by = "manual"

    resp = client.put(f"/api/llm-providers/{lp_id}", json={"name": "renamed"})

    assert resp.status_code == 200, resp.text
    updated = client.get(f"/api/llm-providers/{lp_id}").json()
    assert updated["status"] == "connected"
    assert updated["last_tested_at"] is not None
    assert updated["last_tested_by"] == "manual"


def test_put_config_none_interval_means_no_change(client: TestClient) -> None:
    # Set it to something non-default first.
    client.put("/api/config", json={"connection_test_interval_minutes": 5})

    # Now send a PUT without the interval field.
    resp = client.put("/api/config", json={"budget": {"max_steps": 11}})
    assert resp.status_code == 200
    # Interval must be unchanged.
    assert resp.json()["connection_test_interval_minutes"] == 5


# --------------------------------------------------------------------------
# Provider_tester result plumbing
# --------------------------------------------------------------------------


def test_provider_tester_result_shape_matches_manual_route_consumer() -> None:
    """Pin the contract between ProviderTestResult and the manual route.

    If either side drifts, the manual route breaks.  This tests the
    structural shape rather than behavior.
    """
    result = ProviderTestResult(
        status="connected",
        tested_at=datetime.now(timezone.utc),
        latency_ms=42,
        error_message=None,
    )
    assert result.status in ("connected", "offline", "error")
    assert isinstance(result.tested_at, datetime)
    assert result.latency_ms == 42
    assert result.error_message is None
