"""Route-level tests for the official engine (interop FR-6).

``GET /api/engines`` reports availability; ``POST /api/chat/compare-matrix``
and ``POST /api/chat`` accept ``rlm_official`` and return 400 with the
user-facing reason when the engine cannot run.  The engine itself is the
shared :class:`FakeRLMEngine`, injected through the ``AppState`` factories
exactly the way the LLM adapter fakes are.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator, Iterator
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from rlmstudio.application.dto import LLMResponseDTO
from rlmstudio.application.sandbox_vars import MODE_DIRECT, MODE_RLM_OFFICIAL
from rlmstudio.infrastructure.engines.rlms_adapter import UNAVAILABLE_REASON
from rlmstudio.server.app import app
from rlmstudio.server.dependencies import AppState, get_state, reset_state
from rlmstudio.server.models import ChatProviderConfig, RuntimeSettings
from tests.fakes.fake_rlm_engine import FakeRLMEngine

_AVAILABLE = (True, "rlms 0.1.3", "0.1.3")
_UNAVAILABLE = (False, UNAVAILABLE_REASON, None)


class _FakeLLMAdapter:
    def __init__(self, response: str = "direct answer") -> None:
        self._response = response
        self.active_model = "fake"

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        return LLMResponseDTO(
            content=self._response, model="fake", input_tokens=10, output_tokens=5
        )

    def complete_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        yield self._response

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def get_pricing(self) -> dict[str, float]:
        return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}

    def get_completion_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0


@pytest.fixture(autouse=True)
def _clean_state() -> Generator[None, None, None]:
    reset_state()
    yield
    reset_state()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _make_chat_provider(name: str = "CP") -> ChatProviderConfig:
    return ChatProviderConfig(
        id=str(uuid.uuid4()),
        name=name,
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        execution_mode=MODE_DIRECT,
        runtime_settings=RuntimeSettings(),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _install_fakes(
    state: AppState,
    *,
    engine: FakeRLMEngine,
    availability: tuple[bool, str, str | None] = _AVAILABLE,
) -> None:
    state.create_llm_adapter_for_chat_provider = (  # type: ignore[method-assign]
        lambda cp_id, num_retries=None: _FakeLLMAdapter()
    )
    state.create_rlm_engine_for_chat_provider = (  # type: ignore[method-assign]
        lambda cp_id, llm=None: engine
    )
    state.rlm_engine_availability = lambda: availability  # type: ignore[method-assign]


class TestEnginesEndpoint:
    def test_reports_the_install_hint_when_the_extra_is_missing(self, client: TestClient) -> None:
        get_state().rlm_engine_availability = lambda: _UNAVAILABLE  # type: ignore[method-assign]

        resp = client.get("/api/engines")

        assert resp.status_code == 200
        body = resp.json()[MODE_RLM_OFFICIAL]
        assert body["available"] is False
        assert "rlm-studio[interop]" in body["reason"]
        assert body["version"] is None

    def test_reports_the_version_when_installed(self, client: TestClient) -> None:
        get_state().rlm_engine_availability = lambda: _AVAILABLE  # type: ignore[method-assign]

        body = client.get("/api/engines").json()[MODE_RLM_OFFICIAL]

        assert body == {"available": True, "reason": "rlms 0.1.3", "version": "0.1.3"}

    def test_real_probe_never_raises(self, client: TestClient) -> None:
        """Whatever is installed, the endpoint answers with a bool and a non-empty reason."""
        body = client.get("/api/engines").json()[MODE_RLM_OFFICIAL]

        assert isinstance(body["available"], bool)
        assert body["reason"]


class TestCompareMatrixWithOfficialEngine:
    def test_official_slot_runs_next_to_a_direct_slot(self, client: TestClient) -> None:
        state = get_state()
        cp = _make_chat_provider()
        state.config.chat_providers.append(cp)
        engine = FakeRLMEngine(answer="official answer")
        _install_fakes(state, engine=engine)

        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "content": "doc",
                "query": "q",
                "chat_provider_ids": [cp.id],
                "modes": [MODE_DIRECT, MODE_RLM_OFFICIAL],
            },
        )

        assert resp.status_code == 200, resp.text
        slots = {s["mode"]: s for s in resp.json()["slots"]}
        assert slots[MODE_DIRECT]["answer"] == "direct answer"
        official = slots[MODE_RLM_OFFICIAL]
        assert official["success"] is True
        assert official["answer"] == "official answer"
        assert official["steps"] == 3
        assert official["total_tokens"] == 235
        assert engine.calls[0][:2] == ("doc", "q")

    def test_unavailable_engine_is_a_400_with_the_reason(self, client: TestClient) -> None:
        state = get_state()
        cp = _make_chat_provider()
        state.config.chat_providers.append(cp)
        _install_fakes(state, engine=FakeRLMEngine(), availability=_UNAVAILABLE)

        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "content": "doc",
                "query": "q",
                "chat_provider_ids": [cp.id],
                "modes": [MODE_RLM_OFFICIAL],
            },
        )

        assert resp.status_code == 400
        assert "rlm-studio[interop]" in resp.json()["error"]["message"]
        # Nothing was recorded for the rejected request.
        assert state.executions == {}

    def test_availability_is_not_consulted_for_built_in_modes(self, client: TestClient) -> None:
        state = get_state()
        cp = _make_chat_provider()
        state.config.chat_providers.append(cp)
        _install_fakes(state, engine=FakeRLMEngine(), availability=_UNAVAILABLE)

        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "content": "doc",
                "query": "q",
                "chat_provider_ids": [cp.id],
                "modes": ["direct"],
            },
        )

        assert resp.status_code == 200, resp.text


class TestChatWithOfficialEngine:
    def test_unavailable_engine_is_a_400_before_any_execution(self, client: TestClient) -> None:
        state = get_state()
        state.rlm_engine_availability = lambda: _UNAVAILABLE  # type: ignore[method-assign]

        resp = client.post(
            "/api/chat",
            json={"query": "q", "content": "doc", "mode": MODE_RLM_OFFICIAL},
        )

        assert resp.status_code == 400
        assert "rlm-studio[interop]" in resp.json()["error"]["message"]
        assert state.executions == {}

    def test_chat_provider_mode_is_honoured(self, client: TestClient) -> None:
        state = get_state()
        cp = _make_chat_provider()
        cp.execution_mode = MODE_RLM_OFFICIAL
        state.config.chat_providers.append(cp)
        state.rlm_engine_availability = lambda: _UNAVAILABLE  # type: ignore[method-assign]

        resp = client.post(
            "/api/chat",
            json={"query": "q", "content": "doc", "chat_provider_id": cp.id},
        )

        assert resp.status_code == 400
        assert "rlm-studio[interop]" in resp.json()["error"]["message"]

    def test_available_engine_is_accepted(self, client: TestClient) -> None:
        state = get_state()
        engine = FakeRLMEngine(answer="official answer")
        state.create_llm_adapter = lambda num_retries=None: _FakeLLMAdapter()  # type: ignore[method-assign]
        state.create_rlm_engine = lambda llm=None: engine  # type: ignore[method-assign]
        state.rlm_engine_availability = lambda: _AVAILABLE  # type: ignore[method-assign]

        resp = client.post(
            "/api/chat",
            json={"query": "q", "content": "doc", "mode": MODE_RLM_OFFICIAL},
        )

        assert resp.status_code == 202, resp.text
        assert resp.json()["status"] == "running"
