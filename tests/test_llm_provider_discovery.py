"""Tests for POST /api/llm-providers/models — form model discovery.

The endpoint exists because the legacy discovery route could not see the
key being configured, silently substituted the static catalog, and cached
for five minutes.  These tests pin all three properties.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from rlmstudio.server.app import create_app
from rlmstudio.server.dependencies import reset_state
from rlmstudio.server.models import ModelInfo

_FETCH = "rlmstudio.server.routes.providers._fetch_models_from_api"


@pytest.fixture
def client() -> TestClient:
    reset_state()
    return TestClient(create_app())


def _post(client: TestClient, **body: object) -> dict[str, Any]:
    resp = client.post("/api/llm-providers/models", json=body)
    assert resp.status_code == 200, resp.text
    return dict(resp.json())


def test_live_list_is_marked_live(client: TestClient) -> None:
    live = [ModelInfo(name="qwen38-27b"), ModelInfo(name="qwen38-27b-awq")]
    with patch(_FETCH, new=AsyncMock(return_value=live)):
        data = _post(client, backend="vllm", endpoint="http://127.0.0.1:8123/v1")

    assert data["source"] == "live"
    assert [m["name"] for m in data["models"]] == ["qwen38-27b", "qwen38-27b-awq"]
    assert data["error"] is None


def test_failure_is_unavailable_and_never_falls_back_to_the_catalog(
    client: TestClient,
) -> None:
    """A failed discovery must not be dressed up as the account's model list."""
    with patch(_FETCH, new=AsyncMock(side_effect=RuntimeError("Connection refused"))):
        data = _post(client, backend="openai", api_key="sk-form")

    assert data["source"] == "unavailable"
    assert data["models"] == []
    assert "Connection refused" in data["error"]


def test_cloud_discovery_without_a_key_says_so(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    data = _post(client, backend="openai")

    assert data["source"] == "unavailable"
    assert data["models"] == []
    assert "No API key" in data["error"]


def test_the_key_being_configured_is_used(client: TestClient) -> None:
    """Discovery authenticates with the form's key, not the process env."""
    fetch = AsyncMock(return_value=[ModelInfo(name="gpt-4o")])
    with patch(_FETCH, new=fetch):
        _post(client, backend="openai", api_key="sk-form-key")

    assert fetch.await_args is not None
    assert fetch.await_args.kwargs["api_key_override"] == "sk-form-key"


def test_edit_mode_falls_back_to_the_saved_key(client: TestClient) -> None:
    lp_id = _create_cloud_provider(client)
    fetch = AsyncMock(return_value=[ModelInfo(name="gpt-4o")])
    with (
        patch(
            "rlmstudio.server.routes.llm_providers._get_api_key",
            return_value="sk-saved",
        ),
        patch(_FETCH, new=fetch),
    ):
        _post(
            client,
            backend="openai",
            llm_provider_id=lp_id,
            endpoint="https://old.example/v1",
        )

    assert fetch.await_args is not None
    assert fetch.await_args.kwargs["api_key_override"] == "sk-saved"


def test_discovery_is_not_cached(client: TestClient) -> None:
    """The form's Refresh button must really re-read the endpoint."""
    fetch = AsyncMock(return_value=[ModelInfo(name="qwen38-27b")])
    with patch(_FETCH, new=fetch):
        _post(client, backend="vllm", endpoint="http://127.0.0.1:8123/v1")
        _post(client, backend="vllm", endpoint="http://127.0.0.1:8123/v1")

    assert fetch.await_count == 2


def test_unknown_backend_is_rejected(client: TestClient) -> None:
    resp = client.post("/api/llm-providers/models", json={"backend": "nope"})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Credential boundaries
# ---------------------------------------------------------------------------


def _create_cloud_provider(client: TestClient, **overrides: object) -> str:
    body: dict[str, object] = {
        "name": "gw",
        "backend": "openai",
        "model": "gpt-4o",
        "api_key": "sk-saved",
        "endpoint": "https://old.example/v1",
    }
    body.update(overrides)
    with patch(
        "rlmstudio.server.routes.llm_providers._discover_context_window",
        return_value=None,
    ):
        resp = client.post("/api/llm-providers", json=body)
    assert resp.status_code == 201, resp.text
    lp_id: str = resp.json()["id"]
    return lp_id


def test_unknown_provider_id_is_rejected(client: TestClient) -> None:
    resp = client.post(
        "/api/llm-providers/models",
        json={"backend": "openai", "llm_provider_id": "missing", "api_key": "sk-x"},
    )
    assert resp.status_code == 404


def test_backend_mismatch_is_rejected(client: TestClient) -> None:
    lp_id = _create_cloud_provider(client, backend="anthropic", model="claude-sonnet-4-6")

    resp = client.post(
        "/api/llm-providers/models",
        json={"backend": "openai", "llm_provider_id": lp_id, "api_key": "sk-x"},
    )
    assert resp.status_code == 400


def test_changed_endpoint_does_not_reuse_the_saved_key(client: TestClient) -> None:
    lp_id = _create_cloud_provider(client)

    data = _post(
        client,
        backend="openai",
        llm_provider_id=lp_id,
        endpoint="https://new.example/v1",
    )

    assert data["source"] == "unavailable"
    assert data["models"] == []
    assert "endpoint changed" in data["error"].lower()


class _FakeResponse:
    def __init__(self, data: dict[str, object]) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._data


class _RecordingClient:
    """Captures the URL discovery asks for, without any network."""

    last_url: str | None = None
    last_headers: dict[str, str] | None = None
    payload: dict[str, object] = {"data": [{"id": "gpt-4o"}]}

    def __init__(self, *args: object, **kwargs: object) -> None:
        return None

    async def __aenter__(self) -> _RecordingClient:
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False

    async def get(self, url: str, headers: dict[str, str] | None = None) -> _FakeResponse:
        _RecordingClient.last_url = url
        _RecordingClient.last_headers = headers
        return _FakeResponse(_RecordingClient.payload)


def test_cloud_discovery_honors_a_custom_endpoint(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gateway provider must be asked at its own URL, not the vendor's."""
    monkeypatch.setattr("rlmstudio.server.routes.providers.httpx.AsyncClient", _RecordingClient)
    _RecordingClient.last_url = None
    _RecordingClient.payload = {"data": [{"id": "gpt-4o"}]}

    data = _post(client, backend="openai", api_key="sk-x", endpoint="https://gw.example/v1")

    assert data["source"] == "live"
    assert _RecordingClient.last_url == "https://gw.example/v1/models"
    assert _RecordingClient.last_headers == {"Authorization": "Bearer sk-x"}


def test_cloud_discovery_endpoint_without_v1_still_gets_the_models_path(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rlmstudio.server.routes.providers.httpx.AsyncClient", _RecordingClient)
    _RecordingClient.last_url = None
    _RecordingClient.payload = {"data": [{"id": "claude-sonnet-4-6"}]}

    data = _post(client, backend="anthropic", api_key="sk-ant-x", endpoint="https://gw.example")

    assert data["source"] == "live"
    assert _RecordingClient.last_url == "https://gw.example/v1/models"


def test_env_key_is_not_used_for_a_custom_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy discovery path must not forward a process key to ?endpoint=."""
    from rlmstudio.server.routes.providers import _fetch_models_from_api

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setattr("rlmstudio.server.routes.providers.httpx.AsyncClient", _RecordingClient)
    _RecordingClient.last_url = None

    result = asyncio.run(
        _fetch_models_from_api("openai", endpoint_override="https://gw.example/v1")
    )

    assert result == []
    assert _RecordingClient.last_url is None


def test_gateway_model_ids_are_not_filtered_as_openai_vendor_ids(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A compatible gateway's own ids (Qwen, custom) must survive discovery."""
    monkeypatch.setattr("rlmstudio.server.routes.providers.httpx.AsyncClient", _RecordingClient)
    _RecordingClient.payload = {
        "data": [{"id": "qwen38-27b"}, {"id": "custom-model"}, {"id": "gpt-4o"}]
    }

    data = _post(client, backend="openai", api_key="sk-x", endpoint="https://gw.example/v1")

    assert data["source"] == "live"
    assert [m["name"] for m in data["models"]] == ["qwen38-27b", "custom-model", "gpt-4o"]


def test_official_openai_keeps_unlisted_chat_models_and_drops_non_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The official filter is a denylist: a model released tomorrow survives."""
    from rlmstudio.server.routes.providers import _fetch_models_from_api

    monkeypatch.setattr("rlmstudio.server.routes.providers.httpx.AsyncClient", _RecordingClient)
    _RecordingClient.payload = {
        "data": [
            {"id": "gpt-4o"},
            {"id": "chatgpt-4o-latest"},
            {"id": "brand-new-chat-model"},
            {"id": "ft:gpt-4o-mini:acme:123"},
            {"id": "ft:babbage-002:acme:456"},
            {"id": "text-embedding-3-small"},
            {"id": "whisper-1"},
            {"id": "dall-e-3"},
        ]
    }

    result = asyncio.run(_fetch_models_from_api("openai", api_key_override="sk-x"))

    assert [m.name for m in result] == [
        "gpt-4o",
        "chatgpt-4o-latest",
        "brand-new-chat-model",
        "ft:gpt-4o-mini:acme:123",
    ]


def test_saved_provider_discovery_uses_its_saved_key(client: TestClient) -> None:
    """The by-id route has the provider, so it can use its own credential."""
    lp_id = _create_cloud_provider(client)
    fetch = AsyncMock(return_value=[ModelInfo(name="gpt-4o")])
    with (
        patch(_FETCH, new=fetch),
        patch(
            "rlmstudio.server.routes.llm_providers._get_api_key",
            return_value="sk-saved",
        ),
    ):
        resp = client.get(f"/api/llm-providers/{lp_id}/models")

    assert resp.status_code == 200
    assert [m["name"] for m in resp.json()] == ["gpt-4o"]
    assert fetch.await_args is not None
    assert fetch.await_args.kwargs["api_key_override"] == "sk-saved"


def test_saved_provider_discovery_logs_a_redacted_error(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    lp_id = _create_cloud_provider(client)
    secret = "sk-" + "B" * 24
    with (
        patch(
            _FETCH,
            new=AsyncMock(side_effect=RuntimeError(f"Authorization: Bearer {secret}")),
        ),
        caplog.at_level(logging.WARNING),
    ):
        resp = client.get(f"/api/llm-providers/{lp_id}/models")

    assert resp.status_code == 200  # catalog fallback for this legacy route
    assert secret not in caplog.text
    assert "<redacted>" in caplog.text


def test_legacy_discovery_logs_a_redacted_error(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """The legacy route can use a vendor env key, so its log must be redacted."""
    from rlmstudio.server.routes import providers as providers_mod

    providers_mod._model_cache.clear()
    secret = "sk-" + "C" * 24
    with (
        patch(
            "rlmstudio.server.routes.providers._fetch_models_from_api",
            new=AsyncMock(side_effect=RuntimeError(f"Authorization: Bearer {secret}")),
        ),
        caplog.at_level(logging.WARNING),
    ):
        resp = client.get("/api/providers/openai/models")

    assert resp.status_code == 200  # catalog fallback for this legacy route
    assert secret not in caplog.text
    assert "<redacted>" in caplog.text


def test_discovery_logs_and_returns_the_same_redacted_error(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "sk-" + "A" * 24
    with (
        patch(
            _FETCH,
            new=AsyncMock(side_effect=RuntimeError(f"Authorization: Bearer {secret}")),
        ),
        caplog.at_level(logging.WARNING),
    ):
        data = _post(client, backend="openai", api_key=secret)

    assert data["source"] == "unavailable"
    assert secret not in caplog.text
    assert secret not in (data["error"] or "")
    assert "<redacted>" in caplog.text
    assert "<redacted>" in (data["error"] or "")
