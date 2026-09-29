"""Route-level tests for the official engine (interop FR-6).

``GET /api/engines`` reports availability; ``POST /api/chat/compare-matrix``,
``POST /api/chat`` and ``WS /ws/chat/{session_id}`` accept ``rlm_official``
and return the engine's result, or 400 / an ``error`` frame with the
user-facing reason when the engine cannot run.  The engine itself is the
shared :class:`FakeRLMEngine`, injected through the ``AppState`` factories
exactly the way the LLM adapter fakes are.

Two things this file deliberately does *not* do:

- It never asserts a :class:`FakeRLMEngine` default as though it were
  production behaviour.  Every number under test is passed into the fake
  from ``_ENGINE_*`` below, so the assertion pins the route's plumbing
  rather than the fake's constructor defaults.
- It never tears state down while a background task is still running.
  ``POST /api/chat`` hands the work to ``asyncio.create_task`` (see
  ``server/routes/chat.py:520``), so the teardown fixture drains in-flight
  executions before calling ``reset_state()``.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Generator, Iterator
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rlmstudio.application.dto import LLMResponseDTO
from rlmstudio.application.sandbox_vars import (
    MODE_DIRECT,
    MODE_RLM_OFFICIAL,
    TRACE_KEY_CONTENT,
    TRACE_KEY_ROLE,
)
from rlmstudio.infrastructure.engines.rlms_adapter import UNAVAILABLE_REASON
from rlmstudio.server.app import app
from rlmstudio.server.dependencies import AppState, SessionRecord, get_state, reset_state
from rlmstudio.server.models import ChatProviderConfig, RuntimeSettings
from tests.fakes.fake_rlm_engine import FakeRLMEngine, scripted_trajectory

_AVAILABLE = (True, "rlms 0.1.3", "0.1.3")
_UNAVAILABLE = (False, UNAVAILABLE_REASON, None)

# Engine totals under test.  Passed *into* the fake so the assertions below
# pin what the route surfaces, not FakeRLMEngine's defaults.
_ENGINE_ANSWER = "official answer"
_ENGINE_INPUT_TOKENS = 411
_ENGINE_OUTPUT_TOKENS = 27
_ENGINE_TOTAL_TOKENS = _ENGINE_INPUT_TOKENS + _ENGINE_OUTPUT_TOKENS
_ENGINE_COST = 0.0031
_ENGINE_STEPS = 5
_ENGINE_TRACE_LENGTH = len(scripted_trajectory(_ENGINE_ANSWER))

_EXECUTION_STATUS_RUNNING = "running"
_EXECUTION_STATUS_COMPLETE = "complete"

_DRAIN_TIMEOUT_SECONDS = 5.0
_POLL_INTERVAL_SECONDS = 0.01
_WS_TIMEOUT_SECONDS = 10.0


def _engine(**overrides: Any) -> FakeRLMEngine:
    """A :class:`FakeRLMEngine` with every asserted total set explicitly."""
    kwargs: dict[str, Any] = {
        "answer": _ENGINE_ANSWER,
        "input_tokens": _ENGINE_INPUT_TOKENS,
        "output_tokens": _ENGINE_OUTPUT_TOKENS,
        "total_cost": _ENGINE_COST,
        "steps": _ENGINE_STEPS,
    }
    kwargs.update(overrides)
    return FakeRLMEngine(**kwargs)


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


def _drain_executions(timeout: float = _DRAIN_TIMEOUT_SECONDS) -> None:
    """Wait (bounded) for background executions to leave ``running``.

    ``POST /api/chat`` returns before its ``asyncio.create_task`` work
    finishes.  Resetting state underneath that task is the race this guards:
    the task keeps a reference to the old ``AppState`` and would otherwise
    still be writing to it while the next test builds a fresh one.
    """
    state = get_state()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(
            record.status == _EXECUTION_STATUS_RUNNING for record in list(state.executions.values())
        ):
            return
        time.sleep(_POLL_INTERVAL_SECONDS)


def _await_execution(execution_id: str, timeout: float = _DRAIN_TIMEOUT_SECONDS) -> Any:
    """Return the execution record once the background task has finished with it."""
    state = get_state()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = state.executions.get(execution_id)
        if record is not None and record.status != _EXECUTION_STATUS_RUNNING:
            return record
        time.sleep(_POLL_INTERVAL_SECONDS)
    raise AssertionError(f"execution {execution_id} did not finish within {timeout}s")


def _receive_until(
    ws: Any, wanted: str, collect: tuple[str, ...] = ()
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read frames until one of type *wanted*, returning it plus collected frames."""
    gathered: list[dict[str, Any]] = []
    deadline = time.monotonic() + _WS_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        frame: dict[str, Any] = ws.receive_json()
        frame_type = frame.get("type")
        if frame_type in collect:
            gathered.append(frame)
        if frame_type == wanted:
            return frame, gathered
    raise AssertionError(f"no {wanted!r} frame within {_WS_TIMEOUT_SECONDS}s")


@pytest.fixture(autouse=True)
def _clean_state() -> Generator[None, None, None]:
    reset_state()
    yield
    _drain_executions()
    reset_state()


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """A client whose portal stays alive, so background tasks can finish."""
    with TestClient(app) as test_client:
        yield test_client


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


def _install_global_fakes(
    state: AppState,
    *,
    engine: FakeRLMEngine,
    availability: tuple[bool, str, str | None] = _AVAILABLE,
) -> None:
    """Install the no-Chat-Provider factories used by the global-mode paths."""
    state.create_llm_adapter = lambda num_retries=None: _FakeLLMAdapter()  # type: ignore[method-assign]
    state.create_rlm_engine = lambda llm=None: engine  # type: ignore[method-assign]
    state.rlm_engine_availability = lambda: availability  # type: ignore[method-assign]


def _seed_session(session_id: str = "engines-session") -> str:
    state = get_state()
    now = datetime.now(timezone.utc)
    state.sessions[session_id] = SessionRecord(
        id=session_id, name="Test", created_at=now, updated_at=now
    )
    return session_id


class TestEnginesEndpoint:
    """Pins ``GET /api/engines`` (server/routes/engines.py)."""

    def test_reports_the_install_hint_when_the_extra_is_missing(self, client: TestClient) -> None:
        get_state().rlm_engine_availability = lambda: _UNAVAILABLE  # type: ignore[method-assign]

        resp = client.get("/api/engines")

        assert resp.status_code == 200
        body = resp.json()[MODE_RLM_OFFICIAL]
        assert body["available"] is False
        assert "rlm-studio[interop]" in body["reason"]
        assert body["version"] is None

    def test_reports_the_version_when_installed(self, client: TestClient) -> None:
        available, reason, version = _AVAILABLE
        get_state().rlm_engine_availability = lambda: _AVAILABLE  # type: ignore[method-assign]

        body = client.get("/api/engines").json()[MODE_RLM_OFFICIAL]

        assert body == {"available": available, "reason": reason, "version": version}

    def test_real_probe_never_raises(self, client: TestClient) -> None:
        """Whatever is installed, the endpoint answers with a bool and a non-empty reason."""
        body = client.get("/api/engines").json()[MODE_RLM_OFFICIAL]

        assert isinstance(body["available"], bool)
        assert body["reason"]


class TestCompareMatrixWithOfficialEngine:
    """Pins the matrix dispatch in ``server/routes/compare_matrix.py``."""

    def test_official_slot_runs_next_to_a_direct_slot(self, client: TestClient) -> None:
        state = get_state()
        cp = _make_chat_provider()
        state.config.chat_providers.append(cp)
        engine = _engine()
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
        assert official["answer"] == _ENGINE_ANSWER
        assert official["steps"] == _ENGINE_STEPS
        assert official["total_tokens"] == _ENGINE_TOTAL_TOKENS
        assert engine.calls[0][:2] == ("doc", "q")

    def test_unavailable_engine_is_a_400_with_the_reason(self, client: TestClient) -> None:
        state = get_state()
        cp = _make_chat_provider()
        state.config.chat_providers.append(cp)
        _install_fakes(state, engine=_engine(), availability=_UNAVAILABLE)

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
        _install_fakes(state, engine=_engine(), availability=_UNAVAILABLE)

        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "content": "doc",
                "query": "q",
                "chat_provider_ids": [cp.id],
                "modes": [MODE_DIRECT],
            },
        )

        assert resp.status_code == 200, resp.text


class TestChatRejectsUnavailableEngine:
    """Pins the pre-flight availability gate at ``server/routes/chat.py:483``."""

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


class TestChatDispatchesToTheEngine:
    """Pins the REST background dispatch at ``server/routes/chat.py:761-768``.

    ``POST /api/chat`` returns 202 before the engine has run, so asserting
    only on the response body proves nothing about the dispatch.  These
    tests wait for the background task and assert on what it recorded.
    """

    def test_accepted_request_reports_running(self, client: TestClient) -> None:
        state = get_state()
        _install_global_fakes(state, engine=_engine())

        resp = client.post(
            "/api/chat",
            json={"query": "q", "content": "doc", "mode": MODE_RLM_OFFICIAL},
        )

        assert resp.status_code == 202, resp.text
        assert resp.json()["status"] == _EXECUTION_STATUS_RUNNING
        assert resp.json()["execution_id"]

    def test_engine_result_is_recorded_on_the_execution(self, client: TestClient) -> None:
        state = get_state()
        engine = _engine()
        _install_global_fakes(state, engine=engine)

        resp = client.post(
            "/api/chat",
            json={"query": "the question", "content": "the document", "mode": MODE_RLM_OFFICIAL},
        )
        record = _await_execution(resp.json()["execution_id"])

        assert record.status == _EXECUTION_STATUS_COMPLETE
        assert record.mode == MODE_RLM_OFFICIAL
        assert record.result is not None
        assert record.result["success"] is True
        assert record.result["answer"] == _ENGINE_ANSWER
        assert record.result["input_tokens"] == _ENGINE_INPUT_TOKENS
        assert record.result["output_tokens"] == _ENGINE_OUTPUT_TOKENS
        assert record.result["total_tokens"] == _ENGINE_TOTAL_TOKENS
        assert record.result["total_cost"] == _ENGINE_COST
        assert record.result["steps_count"] == _ENGINE_STEPS

    def test_engine_receives_the_content_and_query(self, client: TestClient) -> None:
        state = get_state()
        engine = _engine()
        _install_global_fakes(state, engine=engine)

        resp = client.post(
            "/api/chat",
            json={"query": "the question", "content": "the document", "mode": MODE_RLM_OFFICIAL},
        )
        _await_execution(resp.json()["execution_id"])

        assert engine.calls[0][:2] == ("the document", "the question")
        assert engine.calls[0][2].mode == MODE_RLM_OFFICIAL

    def test_engine_trace_is_stored_as_the_execution_steps(self, client: TestClient) -> None:
        state = get_state()
        _install_global_fakes(state, engine=_engine())

        resp = client.post(
            "/api/chat",
            json={"query": "q", "content": "doc", "mode": MODE_RLM_OFFICIAL},
        )
        record = _await_execution(resp.json()["execution_id"])

        assert len(record.steps) == _ENGINE_TRACE_LENGTH
        assert record.steps[-1][TRACE_KEY_ROLE] == "assistant"
        assert record.steps[-1][TRACE_KEY_CONTENT] == _ENGINE_ANSWER

    def test_answer_reaches_the_session_transcript(self, client: TestClient) -> None:
        state = get_state()
        _install_global_fakes(state, engine=_engine())

        resp = client.post(
            "/api/chat",
            json={"query": "q", "content": "doc", "mode": MODE_RLM_OFFICIAL},
        )
        body = resp.json()
        _await_execution(body["execution_id"])

        session = state.sessions[body["session_id"]]
        assistant = [m for m in session.messages if m["role"] == "assistant"]
        assert len(assistant) == 1
        assert assistant[0]["content"] == _ENGINE_ANSWER
        assert assistant[0]["mode_used"] == MODE_RLM_OFFICIAL
        assert assistant[0]["metrics"]["total_tokens"] == _ENGINE_TOTAL_TOKENS

    def test_engine_failure_is_recorded_as_an_errored_execution(self, client: TestClient) -> None:
        state = get_state()
        _install_global_fakes(state, engine=_engine(error=RuntimeError("backend down")))

        resp = client.post(
            "/api/chat",
            json={"query": "q", "content": "doc", "mode": MODE_RLM_OFFICIAL},
        )
        record = _await_execution(resp.json()["execution_id"])

        assert record.status == "error"
        assert record.result is not None
        assert record.result["success"] is False
        assert record.result["error"] == "backend down"


class TestWebSocketOfficialEngine:
    """Pins the WebSocket dispatch at ``server/routes/chat.py:1203-1213``.

    This is the only production caller of ``execute_async`` with an event
    emitter, so it is the only place the engine's trace is replayed as
    ``step`` frames and its totals as a ``metrics`` frame.
    """

    def test_unavailable_engine_is_an_error_frame(self, client: TestClient) -> None:
        state = get_state()
        session_id = _seed_session()
        state.rlm_engine_availability = lambda: _UNAVAILABLE  # type: ignore[method-assign]

        with client.websocket_connect(f"/ws/chat/{session_id}") as ws:
            assert ws.receive_json()["type"] == "connected"
            ws.send_json(
                {
                    "type": "query",
                    "id": "q1",
                    "query": "q",
                    "content": "doc",
                    "mode": MODE_RLM_OFFICIAL,
                }
            )
            frame, _ = _receive_until(ws, "error")

        assert frame["id"] == "q1"
        assert frame["data"]["code"] == "ENGINE_UNAVAILABLE"
        assert "rlm-studio[interop]" in frame["data"]["message"]
        assert state.executions == {}

    def test_run_emits_one_step_frame_per_trace_entry_then_metrics(
        self, client: TestClient
    ) -> None:
        state = get_state()
        session_id = _seed_session()
        engine = _engine()
        _install_global_fakes(state, engine=engine)

        with client.websocket_connect(f"/ws/chat/{session_id}") as ws:
            assert ws.receive_json()["type"] == "connected"
            ws.send_json(
                {
                    "type": "query",
                    "id": "q2",
                    "query": "the question",
                    "content": "the document",
                    "mode": MODE_RLM_OFFICIAL,
                }
            )
            complete, streamed = _receive_until(
                ws, "complete", collect=("token", "step", "metrics")
            )

        steps = [f for f in streamed if f["type"] == "step"]
        metrics = [f for f in streamed if f["type"] == "metrics"]
        tokens = [f for f in streamed if f["type"] == "token"]

        assert len(steps) == _ENGINE_TRACE_LENGTH
        assert all(f["id"] == "q2" for f in steps)
        assert steps[-1]["data"][TRACE_KEY_CONTENT] == _ENGINE_ANSWER
        assert tokens == []  # the engine does not stream
        assert len(metrics) == 1
        assert metrics[0]["data"]["input_tokens"] == _ENGINE_INPUT_TOKENS
        assert metrics[0]["data"]["output_tokens"] == _ENGINE_OUTPUT_TOKENS
        assert metrics[0]["data"]["total_tokens"] == _ENGINE_TOTAL_TOKENS
        assert metrics[0]["data"]["cost_usd"] == _ENGINE_COST
        assert metrics[0]["data"]["steps"] == _ENGINE_STEPS

        assert complete["id"] == "q2"
        assert complete["data"]["answer"] == _ENGINE_ANSWER
        assert complete["data"]["mode"] == MODE_RLM_OFFICIAL
        assert complete["data"]["metrics"]["total_tokens"] == _ENGINE_TOTAL_TOKENS
        assert engine.calls[0][:2] == ("the document", "the question")

    def test_run_records_an_execution_for_traces_and_dashboard(self, client: TestClient) -> None:
        state = get_state()
        session_id = _seed_session()
        _install_global_fakes(state, engine=_engine())

        with client.websocket_connect(f"/ws/chat/{session_id}") as ws:
            assert ws.receive_json()["type"] == "connected"
            ws.send_json(
                {
                    "type": "query",
                    "id": "q3",
                    "query": "q",
                    "content": "doc",
                    "mode": MODE_RLM_OFFICIAL,
                }
            )
            complete, _ = _receive_until(ws, "complete", collect=("token", "step", "metrics"))

        record = state.executions[complete["data"]["execution_id"]]
        assert record.mode == MODE_RLM_OFFICIAL
        assert record.status == _EXECUTION_STATUS_COMPLETE
        assert record.result is not None
        assert record.result["answer"] == _ENGINE_ANSWER
        assert len(record.steps) == _ENGINE_TRACE_LENGTH

    def test_engine_failure_is_an_execution_error_frame(self, client: TestClient) -> None:
        state = get_state()
        session_id = _seed_session()
        _install_global_fakes(state, engine=_engine(error=RuntimeError("async boom")))

        with client.websocket_connect(f"/ws/chat/{session_id}") as ws:
            assert ws.receive_json()["type"] == "connected"
            ws.send_json(
                {
                    "type": "query",
                    "id": "q4",
                    "query": "q",
                    "content": "doc",
                    "mode": MODE_RLM_OFFICIAL,
                }
            )
            frame, _ = _receive_until(ws, "error", collect=("token", "step", "metrics"))

        assert frame["id"] == "q4"
        assert frame["data"]["code"] == "EXECUTION_ERROR"
        assert frame["data"]["message"] == "async boom"
        assert frame["data"]["mode"] == MODE_RLM_OFFICIAL
