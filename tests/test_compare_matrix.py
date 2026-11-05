"""Tests for POST /api/chat/compare-matrix — V2 (LLM Provider) path."""

from __future__ import annotations

import uuid
from collections.abc import Generator, Iterator
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rlmstudio.application.dto import LLMResponseDTO
from rlmstudio.server.app import app
from rlmstudio.server.dependencies import AppState, get_state, reset_state
from rlmstudio.server.models import (
    ChatProviderConfig,
    LLMProviderConfig,
    RuntimeSettings,
)

# ---------------------------------------------------------------------------
# Fake LLM adapter
# ---------------------------------------------------------------------------


class _FakeLLM:
    def __init__(self, response: str = "fake answer") -> None:
        self._response = response
        self.active_model = "fake-model"

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        return LLMResponseDTO(
            content=self._response, model="fake", input_tokens=10, output_tokens=5
        )

    def complete_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        yield self._response

    def count_tokens(self, text: str = "", *, messages: list[dict[str, str]] | None = None) -> int:
        return max(1, len(text) // 4)

    def get_pricing(self) -> dict[str, float]:
        return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}

    def get_completion_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state() -> Generator[None, None, None]:
    reset_state()
    yield
    reset_state()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _make_llm_provider(
    name: str,
    backend: str = "openai",
    model: str = "gpt-4o",
    status: str = "connected",
) -> LLMProviderConfig:
    return LLMProviderConfig(
        id=str(uuid.uuid4()),
        name=name,
        backend=backend,
        model=model,
        status=status,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _make_chat_provider(name: str) -> ChatProviderConfig:
    return ChatProviderConfig(
        id=str(uuid.uuid4()),
        name=name,
        llm_provider="openai",
        llm_model="gpt-4o",
        execution_mode="direct",
        runtime_settings=RuntimeSettings(),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _register_fake_adapter(state: AppState, response: str = "fake answer") -> None:
    def _factory(cp_id: str, num_retries: int | None = None) -> Any:
        return _FakeLLM(response=response)

    state.create_llm_adapter_for_chat_provider = _factory  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# V2 Validation
# ---------------------------------------------------------------------------


class TestV2Validation:
    def test_missing_both_provider_fields_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/api/chat/compare-matrix",
            json={"query": "q", "content": "doc", "modes": ["direct"]},
        )
        assert resp.status_code == 422

    def test_unknown_llm_provider_returns_404(self, client: TestClient) -> None:
        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "q",
                "content": "doc",
                "llm_provider_ids": ["nonexistent"],
                "modes": ["direct"],
            },
        )
        assert resp.status_code == 404

    def test_disconnected_llm_provider_returns_400(self, client: TestClient) -> None:
        state = get_state()
        lp = _make_llm_provider("Offline", status="offline")
        state.config.llm_providers.append(lp)

        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "q",
                "content": "doc",
                "llm_provider_ids": [lp.id],
                "modes": ["direct"],
            },
        )
        assert resp.status_code == 400
        assert "not connected" in resp.json()["error"]["message"]

    def test_an_untested_local_provider_is_accepted(self, client: TestClient) -> None:
        """Browser pass D1: a local provider saved without pressing Test keeps
        the stored default status "not_configured" while Settings shows it as
        "configured"; Compare refused it with a 400."""
        state = get_state()
        lp = _make_llm_provider("Local", backend="lmstudio", model="qwen", status="not_configured")
        state.config.llm_providers.append(lp)
        _register_fake_adapter(state)

        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "q",
                "content": "doc",
                "llm_provider_ids": [lp.id],
                "modes": ["direct"],
            },
        )
        assert resp.status_code == 200, resp.text

    def test_missing_content_returns_400(self, client: TestClient) -> None:
        state = get_state()
        lp = _make_llm_provider("A")
        state.config.llm_providers.append(lp)
        _register_fake_adapter(state)

        resp = client.post(
            "/api/chat/compare-matrix",
            json={"query": "q", "llm_provider_ids": [lp.id], "modes": ["direct"]},
        )
        assert resp.status_code == 400

    def test_too_many_slots_returns_400(self, client: TestClient) -> None:
        state = get_state()
        providers = [_make_llm_provider(f"P{i}") for i in range(6)]
        for lp in providers:
            state.config.llm_providers.append(lp)
        _register_fake_adapter(state)

        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "q",
                "content": "doc",
                "llm_provider_ids": [lp.id for lp in providers],
                "modes": ["direct", "rlm"],
            },
        )
        assert resp.status_code == 400
        assert "Too many slots" in resp.json()["error"]["message"]


# ---------------------------------------------------------------------------
# V2 Execution
# ---------------------------------------------------------------------------


class TestV2Execution:
    def test_single_provider_single_mode(self, client: TestClient) -> None:
        state = get_state()
        lp = _make_llm_provider("GPT-4o", backend="openai", model="gpt-4o")
        state.config.llm_providers.append(lp)
        _register_fake_adapter(state, response="the answer")

        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "what is it?",
                "content": "doc content",
                "llm_provider_ids": [lp.id],
                "modes": ["direct"],
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["slots"]) == 1
        slot = data["slots"][0]
        assert slot["success"] is True
        assert "GPT-4o" in slot["label"]
        assert "direct" in slot["label"]
        assert slot["provider"] == "openai"
        assert slot["model"] == "gpt-4o"

    def test_two_providers_two_modes(self, client: TestClient) -> None:
        state = get_state()
        lp_a = _make_llm_provider("ProviderA", backend="openai", model="gpt-4o")
        lp_b = _make_llm_provider("ProviderB", backend="anthropic", model="claude")
        state.config.llm_providers.append(lp_a)
        state.config.llm_providers.append(lp_b)
        _register_fake_adapter(state)

        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "q",
                "content": "doc",
                "llm_provider_ids": [lp_a.id, lp_b.id],
                "modes": ["direct", "rlm"],
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["slots"]) == 4
        labels = {s["label"] for s in data["slots"]}
        assert "ProviderA \u00b7 direct" in labels
        assert "ProviderA \u00b7 rlm" in labels
        assert "ProviderB \u00b7 direct" in labels
        assert "ProviderB \u00b7 rlm" in labels

    def test_inline_runtime_settings_applied(self, client: TestClient) -> None:
        state = get_state()
        lp = _make_llm_provider("Test")
        state.config.llm_providers.append(lp)
        _register_fake_adapter(state)

        client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "q",
                "content": "doc",
                "llm_provider_ids": [lp.id],
                "modes": ["direct"],
                "runtime_settings": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "max_output_tokens": 2048,
                    "timeout_seconds": 60,
                },
            },
        )

        ephemeral = [cp for cp in state.config.chat_providers if cp.ephemeral]
        assert len(ephemeral) == 1
        assert ephemeral[0].runtime_settings.temperature == 0.3
        assert ephemeral[0].runtime_settings.top_p == 0.9

    def test_inline_budget_applied(self, client: TestClient) -> None:
        state = get_state()
        lp = _make_llm_provider("Test")
        state.config.llm_providers.append(lp)
        _register_fake_adapter(state)

        client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "q",
                "content": "doc",
                "llm_provider_ids": [lp.id],
                "modes": ["rlm"],
                "budget": {
                    "max_steps": 32,
                    "max_tokens": 50000,
                    "max_cost_usd": 10.0,
                    "max_time_seconds": 300,
                    "max_recursion_depth": 5,
                    "repeat_limit": 5,
                    "nudge_at_fraction": 0.8,
                },
            },
        )

        ephemeral = [cp for cp in state.config.chat_providers if cp.ephemeral]
        assert len(ephemeral) == 1
        assert ephemeral[0].rlm_max_steps == 32
        assert ephemeral[0].rlm_repeat_limit == 5


# ---------------------------------------------------------------------------
# Ephemeral CPs
# ---------------------------------------------------------------------------


class TestEphemeralCPs:
    def test_ephemeral_cp_created(self, client: TestClient) -> None:
        state = get_state()
        lp = _make_llm_provider("Test")
        state.config.llm_providers.append(lp)
        _register_fake_adapter(state)

        client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "q",
                "content": "doc",
                "llm_provider_ids": [lp.id],
                "modes": ["direct"],
            },
        )

        ephemeral = [cp for cp in state.config.chat_providers if cp.ephemeral]
        assert len(ephemeral) == 1
        assert ephemeral[0].llm_provider_id == lp.id
        assert "[compare]" in ephemeral[0].name

    def test_ephemeral_cp_excluded_from_listing(self, client: TestClient) -> None:
        state = get_state()
        lp = _make_llm_provider("Test")
        state.config.llm_providers.append(lp)
        _register_fake_adapter(state)

        client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "q",
                "content": "doc",
                "llm_provider_ids": [lp.id],
                "modes": ["direct"],
            },
        )

        resp = client.get("/api/chat-providers")
        assert resp.status_code == 200
        names = [cp["name"] for cp in resp.json()]
        assert not any("[compare]" in n for n in names)

    def test_ephemeral_cp_dedup_on_repeat_run(self, client: TestClient) -> None:
        state = get_state()
        lp = _make_llm_provider("Test")
        state.config.llm_providers.append(lp)
        _register_fake_adapter(state)

        req = {
            "query": "q",
            "content": "doc",
            "llm_provider_ids": [lp.id],
            "modes": ["direct"],
        }
        client.post("/api/chat/compare-matrix", json=req)
        client.post("/api/chat/compare-matrix", json=req)

        ephemeral = [cp for cp in state.config.chat_providers if cp.ephemeral]
        assert len(ephemeral) == 1

    def test_ephemeral_cp_not_in_serialized_config(self, client: TestClient) -> None:
        state = get_state()
        lp = _make_llm_provider("Test")
        state.config.llm_providers.append(lp)
        _register_fake_adapter(state)

        client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "q",
                "content": "doc",
                "llm_provider_ids": [lp.id],
                "modes": ["direct"],
            },
        )

        dump = state.config.model_dump()
        persistent = [cp for cp in dump["chat_providers"] if not cp.get("ephemeral")]
        assert len(persistent) == 0


# ---------------------------------------------------------------------------
# Session & Telemetry
# ---------------------------------------------------------------------------


class TestV2SessionAndTelemetry:
    def test_session_has_messages(self, client: TestClient) -> None:
        state = get_state()
        lp = _make_llm_provider("Test")
        state.config.llm_providers.append(lp)
        _register_fake_adapter(state)

        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "q",
                "content": "doc",
                "llm_provider_ids": [lp.id],
                "modes": ["direct"],
            },
        )
        session_id = resp.json()["session_id"]
        session = state.sessions[session_id]
        assert len(session.messages) >= 2  # 1 user + 1 assistant

    def test_execution_records_created(self, client: TestClient) -> None:
        state = get_state()
        lp = _make_llm_provider("Test")
        state.config.llm_providers.append(lp)
        _register_fake_adapter(state)

        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "q",
                "content": "doc",
                "llm_provider_ids": [lp.id],
                "modes": ["direct", "rlm"],
            },
        )
        data = resp.json()
        for slot in data["slots"]:
            assert slot["execution_id"] in state.executions

    def test_telemetry_recorded(self, client: TestClient) -> None:
        state = get_state()
        lp = _make_llm_provider("Test")
        state.config.llm_providers.append(lp)
        _register_fake_adapter(state)

        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "q",
                "content": "doc",
                "llm_provider_ids": [lp.id],
                "modes": ["direct"],
            },
        )
        group_id = resp.json()["comparison_group_id"]
        runs = state.telemetry.list_runs(comparison_group_id=group_id, limit=100)
        assert len(runs) == 1


# ---------------------------------------------------------------------------
# V1 Backward Compatibility
# ---------------------------------------------------------------------------


class TestV1Backward:
    def test_v1_path_still_works(self, client: TestClient) -> None:
        state = get_state()
        cp = _make_chat_provider("Legacy CP")
        state.config.chat_providers.append(cp)
        _register_fake_adapter(state)

        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "q",
                "content": "doc",
                "chat_provider_ids": [cp.id],
                "modes": ["direct"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["slots"]) == 1
        assert data["slots"][0]["chat_provider_id"] == cp.id


def test_ephemeral_official_provider_is_named_with_the_mode_label() -> None:
    """M22: the Compare-only Chat Provider shown in Traces names the official
    engine by its label, while its execution mode stays the raw id."""
    from rlmstudio.server.routes.compare_matrix import _get_or_create_ephemeral_cp

    state = get_state()
    lp = _make_llm_provider("LM Studio Qwen")
    cp = _get_or_create_ephemeral_cp(state, lp, "rlm_official", None, None, None)

    assert cp.name.endswith("LM Studio Qwen · Official RLM (rlms)")
    assert "rlm_official" not in cp.name
    assert cp.execution_mode == "rlm_official"
