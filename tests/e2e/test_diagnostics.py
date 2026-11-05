"""E2E tests for the /api/diagnostics endpoint.

Covers the four checks (backend, provider, judge, storage) against the
real FastAPI app via TestClient. State is reset between tests via the
autouse ``_clean_state`` fixture in ``conftest.py``.

Provider-check tests isolate themselves from the ambient environment
(real API keys in env vars, real SecretStore contents) via the
``isolate_provider_env`` fixture. That lets each test declare exactly one signal at a time.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rlmstudio.server.dependencies import get_state
from rlmstudio.server.models import LLMProviderConfig, ProviderConfig
from rlmstudio.server.routes import llm_providers as llm_providers_mod

pytestmark = [pytest.mark.e2e]


@pytest.fixture
def isolate_provider_env(monkeypatch: pytest.MonkeyPatch):
    """Pin provider-status inputs so tests are deterministic.

    - ``_get_api_key`` defaults to returning ``None``; individual
      tests can override via the returned ``set_api_key`` callable.
    """
    current_key: dict[str, str | None] = {"value": None}

    def _fake_get_api_key(_llm_provider_id: str, _backend: str) -> str | None:
        return current_key["value"]

    monkeypatch.setattr(llm_providers_mod, "_get_api_key", _fake_get_api_key)

    def set_api_key(value: str | None) -> None:
        current_key["value"] = value

    return set_api_key


class TestDiagnosticsEndpoint:
    """GET /api/diagnostics."""

    def test_returns_200_with_all_four_checks(self, client: TestClient) -> None:
        resp = client.get("/api/diagnostics")
        assert resp.status_code == 200
        data = resp.json()
        assert set(data.keys()) == {"backend", "provider", "judge", "storage"}
        for key in ("backend", "provider", "judge", "storage"):
            assert data[key]["status"] in {"ok", "warn", "error"}
            assert isinstance(data[key]["message"], str)

    def test_backend_check_is_ok(self, client: TestClient) -> None:
        data = client.get("/api/diagnostics").json()
        assert data["backend"]["status"] == "ok"

    def test_provider_error_when_no_configured_provider(
        self, client: TestClient, isolate_provider_env
    ) -> None:
        state = get_state()
        state.config.llm_providers = []
        state.config.provider_configs = []
        data = client.get("/api/diagnostics").json()
        assert data["provider"]["status"] == "error"
        assert data["provider"]["fixUrl"] == "/settings"

    def test_provider_error_when_api_key_backend_has_no_key(
        self, client: TestClient, isolate_provider_env
    ) -> None:
        # API-key backend with no key anywhere (SecretStore, env):
        # _compute_status returns "not_configured", which must NOT pass.
        state = get_state()
        state.config.llm_providers = [
            LLMProviderConfig(
                id="lp-openai-nokey",
                name="OpenAI (no key)",
                backend="openai",
                model="gpt-4o",
                status="not_configured",
            ),
        ]
        state.config.provider_configs = []
        data = client.get("/api/diagnostics").json()
        assert data["provider"]["status"] == "error"

    def test_provider_ok_when_api_key_backend_has_env_key(
        self,
        client: TestClient,
        isolate_provider_env,
    ) -> None:
        # The reviewer's P1 case: persisted status="not_configured",
        # but the effective key is available via env var / SecretStore.
        # _compute_status must lift this to "configured", and the
        # diagnostics endpoint must count it as usable.
        isolate_provider_env("sk-test-abc123")
        state = get_state()
        state.config.llm_providers = [
            LLMProviderConfig(
                id="lp-openai-env",
                name="OpenAI (env)",
                backend="openai",
                model="gpt-4o",
                status="not_configured",
            ),
        ]
        state.config.provider_configs = []
        data = client.get("/api/diagnostics").json()
        assert data["provider"]["status"] == "ok", data["provider"]
        assert "1" in data["provider"]["message"]

    def test_provider_ok_for_local_backend_with_not_configured_status(
        self, client: TestClient, isolate_provider_env
    ) -> None:
        # The reviewer's other P1 case: a local backend (Ollama) never
        # needs an API key, so _compute_status returns "configured"
        # even when the persisted record still says "not_configured".
        state = get_state()
        state.config.llm_providers = [
            LLMProviderConfig(
                id="lp-ollama",
                name="Ollama local",
                backend="ollama",
                model="llama3.1:8b",
                status="not_configured",
            ),
        ]
        state.config.provider_configs = []
        data = client.get("/api/diagnostics").json()
        assert data["provider"]["status"] == "ok", data["provider"]

    def test_provider_ok_when_configured_llm_provider_exists(
        self, client: TestClient, isolate_provider_env
    ) -> None:
        # Persisted status="configured" is preserved by _compute_status
        # for API-key backends only if a key is present. Provide one via
        # the fixture to keep this test independent of the real env.
        isolate_provider_env("sk-test-xyz")
        state = get_state()
        state.config.llm_providers = [
            LLMProviderConfig(
                id="lp-openai-cfg",
                name="My OpenAI",
                backend="openai",
                model="gpt-4o",
                status="configured",
            ),
        ]
        state.config.provider_configs = []
        data = client.get("/api/diagnostics").json()
        assert data["provider"]["status"] == "ok"
        assert "1" in data["provider"]["message"]
        assert data["provider"]["fixUrl"] is None

    @pytest.mark.parametrize("failed_status", ["offline", "error"])
    def test_provider_warn_when_every_provider_failed_its_test(
        self, client: TestClient, isolate_provider_env, failed_status: str
    ) -> None:
        # M16: a provider whose last connection test failed is not usable,
        # so the check must not report "ok".
        state = get_state()
        state.config.llm_providers = [
            LLMProviderConfig(
                id="lp-lmstudio-down",
                name="LM Studio",
                backend="lmstudio",
                model="qwen2.5-7b-instruct",
                status=failed_status,
            ),
        ]
        state.config.provider_configs = []
        data = client.get("/api/diagnostics").json()
        assert data["provider"]["status"] == "warn", data["provider"]
        assert "connection test" in data["provider"]["message"]
        assert data["provider"]["fixUrl"] == "/settings"

    def test_provider_ok_when_one_of_several_providers_is_reachable(
        self, client: TestClient, isolate_provider_env
    ) -> None:
        state = get_state()
        state.config.llm_providers = [
            LLMProviderConfig(
                id="lp-lmstudio-down",
                name="LM Studio",
                backend="lmstudio",
                model="qwen2.5-7b-instruct",
                status="offline",
            ),
            LLMProviderConfig(
                id="lp-ollama-up",
                name="Ollama",
                backend="ollama",
                model="llama3.2",
                status="connected",
            ),
        ]
        state.config.provider_configs = []
        data = client.get("/api/diagnostics").json()
        assert data["provider"]["status"] == "ok", data["provider"]
        assert "1" in data["provider"]["message"]

    def test_provider_ok_via_legacy_provider_configs_fallback(
        self, client: TestClient, isolate_provider_env
    ) -> None:
        # Backward compat: installs that predate named LLM Providers still
        # carry an enabled entry in the legacy provider_configs list.
        state = get_state()
        state.config.llm_providers = []
        state.config.provider_configs = [
            ProviderConfig(provider="openai", model="gpt-4o", enabled=True),
        ]
        data = client.get("/api/diagnostics").json()
        assert data["provider"]["status"] == "ok"

    def test_judge_warn_when_not_configured(self, client: TestClient) -> None:
        state = get_state()
        state.config.judge_chat_provider_id = None
        data = client.get("/api/diagnostics").json()
        assert data["judge"]["status"] == "warn"
        assert data["judge"]["fixUrl"] == "/settings"

    def test_judge_ok_when_configured(self, client: TestClient) -> None:
        state = get_state()
        state.config.judge_chat_provider_id = "cp-123"
        data = client.get("/api/diagnostics").json()
        assert data["judge"]["status"] == "ok"

    def test_storage_ok_on_healthy_telemetry_store(self, client: TestClient) -> None:
        data = client.get("/api/diagnostics").json()
        assert data["storage"]["status"] == "ok"
        # Message reflects the narrower guarantee: read-only probe.
        assert data["storage"]["message"] == "Storage reachable"
        # fixUrl is intentionally omitted until /learn/troubleshoot ships.
        assert data["storage"]["fixUrl"] is None

    def test_storage_error_when_telemetry_raises(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = get_state()

        def _boom() -> int:
            raise RuntimeError("disk full")

        monkeypatch.setattr(state.telemetry, "count_runs", _boom)
        data = client.get("/api/diagnostics").json()
        assert data["storage"]["status"] == "error"
        assert "RuntimeError" in data["storage"]["message"]
        # No fixUrl until /learn/troubleshoot ships in Step 6.
        assert data["storage"]["fixUrl"] is None
