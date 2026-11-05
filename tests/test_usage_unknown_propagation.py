"""A run whose usage was never reported is "unknown" everywhere it is served (SC3-02).

The official engine's own timeout carries no token or cost figures, so the
run is recorded with 0 tokens and $0.  ``cost_known`` / ``tokens_known`` on
the result say those zeros are not measurements; before this they stopped at
the Compare slot and the Chat footer, Traces and Dashboard showed
"0 tokens · $0.0000" as if the run had been free.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rlmstudio.application.sandbox_vars import (
    MODE_DIRECT,
    MODE_RLM_OFFICIAL,
    RESULT_KEY_COST_KNOWN,
    RESULT_KEY_TOKENS_KNOWN,
)
from rlmstudio.server.app import app
from rlmstudio.server.dependencies import get_state, reset_state
from rlmstudio.telemetry import store as store_module
from rlmstudio.telemetry.store import TelemetryStore
from tests.fakes.fake_rlm_engine import FakeRLMEngine
from tests.test_engines_route import (
    _await_execution,
    _drain_executions,
    _engine,
    _install_fakes,
    _install_global_fakes,
    _make_chat_provider,
    _receive_until,
    _seed_session,
)

_UNKNOWN = {RESULT_KEY_COST_KNOWN: False, RESULT_KEY_TOKENS_KNOWN: False}


@pytest.fixture(autouse=True)
def _clean_state() -> Generator[None, None, None]:
    reset_state()
    yield
    _drain_executions()
    reset_state()


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as test_client:
        yield test_client


def _unknown_usage_engine() -> FakeRLMEngine:
    return _engine(input_tokens=0, output_tokens=0, total_cost=0.0, metadata=dict(_UNKNOWN))


class TestChatRunWithUnknownUsage:
    def test_every_surface_carries_the_flags(self, client: TestClient) -> None:
        state = get_state()
        session_id = _seed_session()
        _install_global_fakes(state, engine=_unknown_usage_engine())

        resp = client.post(
            "/api/chat",
            json={
                "query": "q",
                "content": "doc",
                "mode": MODE_RLM_OFFICIAL,
                "session_id": session_id,
            },
        )
        execution_id = resp.json()["execution_id"]
        record = _await_execution(execution_id)

        # The execution record and the trace served from it.
        assert record.result is not None
        assert record.result[RESULT_KEY_COST_KNOWN] is False
        assert record.result[RESULT_KEY_TOKENS_KNOWN] is False
        trace = client.get(f"/api/traces/{execution_id}").json()["result"]
        assert (trace["cost_known"], trace["tokens_known"]) == (False, False)
        listed = {e["execution_id"]: e for e in client.get("/api/executions").json()}
        assert (listed[execution_id]["cost_known"], listed[execution_id]["tokens_known"]) == (
            False,
            False,
        )

        # The Chat footer's metrics on the transcript message.
        message = next(m for m in state.sessions[session_id].messages if m["role"] == "assistant")
        assert message["metrics"][RESULT_KEY_COST_KNOWN] is False
        assert message["metrics"][RESULT_KEY_TOKENS_KNOWN] is False

        # The telemetry row, and the trace served from it once memory is gone.
        run = state.telemetry.get_run(execution_id)
        assert run is not None
        assert (run.cost_known, run.tokens_known) == (False, False)
        state.executions.pop(execution_id)
        trace = client.get(f"/api/traces/{execution_id}").json()["result"]
        assert (trace["cost_known"], trace["tokens_known"]) == (False, False)

        # The Dashboard: the timeline row, and the totals say they omit it.
        metrics = client.get(f"/api/metrics/{session_id}").json()
        entry = next(e for e in metrics["timeline"] if e["execution_id"] == execution_id)
        assert (entry["cost_known"], entry["tokens_known"]) == (False, False)
        assert metrics["summary"]["cost_unknown_runs"] == 1
        assert metrics["summary"]["tokens_unknown_runs"] == 1

    def test_a_measured_run_stays_known(self, client: TestClient) -> None:
        state = get_state()
        _install_global_fakes(state, engine=_engine())

        resp = client.post(
            "/api/chat", json={"query": "q", "content": "doc", "mode": MODE_RLM_OFFICIAL}
        )
        record = _await_execution(resp.json()["execution_id"])

        assert record.result is not None
        assert record.result[RESULT_KEY_COST_KNOWN] is True
        assert record.result[RESULT_KEY_TOKENS_KNOWN] is True

    def test_the_websocket_frames_carry_the_flags(self, client: TestClient) -> None:
        state = get_state()
        session_id = _seed_session()
        _install_global_fakes(state, engine=_unknown_usage_engine())

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
            complete, streamed = _receive_until(ws, "complete", collect=("metrics",))

        for metrics in (complete["data"]["metrics"], streamed[0]["data"]):
            assert metrics[RESULT_KEY_COST_KNOWN] is False
            assert metrics[RESULT_KEY_TOKENS_KNOWN] is False


class TestCompareSlotWithUnknownUsage:
    def test_the_slot_and_its_telemetry_row_carry_the_flags(self, client: TestClient) -> None:
        state = get_state()
        cp = _make_chat_provider()
        state.config.chat_providers.append(cp)
        _install_fakes(state, engine=_unknown_usage_engine())

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
        official, direct = slots[MODE_RLM_OFFICIAL], slots[MODE_DIRECT]
        assert (official["cost_known"], official["tokens_known"]) == (False, False)
        assert (direct["cost_known"], direct["tokens_known"]) == (True, True)
        run = state.telemetry.get_run(official["execution_id"])
        assert run is not None
        assert (run.cost_known, run.tokens_known) == (False, False)
        # Pass 4 (W3C-1): Traces reads the in-memory record first; it showed
        # 0 / $0.0000 until a restart re-read the run from telemetry.
        trace = client.get(f"/api/traces/{official['execution_id']}").json()
        assert (trace["result"]["cost_known"], trace["result"]["tokens_known"]) == (False, False)
        listed = {e["execution_id"]: e for e in client.get("/api/executions").json()}
        assert (
            listed[official["execution_id"]]["cost_known"],
            listed[official["execution_id"]]["tokens_known"],
        ) == (False, False)


def test_rows_recorded_before_the_flags_read_as_known(tmp_path: Path) -> None:
    """A v4 row never said its usage was unknown, so it must not start to."""
    db = tmp_path / "telemetry.db"
    conn = sqlite3.connect(db)
    conn.executescript(store_module._SCHEMA_SQL)
    for version in (2, 3, 4):
        for stmt in store_module._MIGRATIONS[version]:
            conn.execute(stmt)
    conn.execute("PRAGMA user_version = 4")
    conn.execute("INSERT INTO runs (id, created_at, mode) VALUES ('legacy', 1.0, 'rlm')")
    conn.commit()
    conn.close()

    store = TelemetryStore(db)
    legacy = store.get_run("legacy")
    assert legacy is not None
    assert (legacy.cost_known, legacy.tokens_known) == (True, True)
    (summary,) = store.list_runs()
    assert (summary.cost_known, summary.tokens_known) == (True, True)
