"""E2E tests for WebSocket streaming endpoints."""

from __future__ import annotations

import time
from collections.abc import Generator
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rlmstudio.server.app import create_app
from rlmstudio.server.dependencies import SessionRecord, get_state, reset_state

pytestmark = pytest.mark.e2e


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
    return TestClient(create_app())


@pytest.fixture
def session_id() -> str:
    return "test-session"


@pytest.fixture
def seeded_session(session_id: str) -> str:
    """Insert a SessionRecord into state so the WS endpoint can resolve it."""
    state = get_state()
    now = datetime.now(timezone.utc)
    state.sessions[session_id] = SessionRecord(
        id=session_id,
        name="Test",
        created_at=now,
        updated_at=now,
    )
    return session_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def receive_non_ping(
    ws: Any,
    timeout: float = 5.0,
    skip_types: tuple[str, ...] = ("ping",),
) -> dict[str, Any]:
    """Return the next message whose type is not in skip_types."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg: dict[str, Any] = ws.receive_json()
        if msg.get("type") not in skip_types:
            return msg
    raise TimeoutError("No matching message received within timeout")


# ---------------------------------------------------------------------------
# TestWebSocketConnection
# ---------------------------------------------------------------------------


class TestWebSocketConnection:
    def test_websocket_connect(self, client: TestClient, seeded_session: str) -> None:
        """Connecting yields a 'connected' message with the correct session_id."""
        with client.websocket_connect(f"/ws/chat/{seeded_session}") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connected"
            assert msg["session_id"] == seeded_session

    def test_websocket_invalid_json(self, client: TestClient, seeded_session: str) -> None:
        """Sending malformed JSON returns an INVALID_JSON error (without closing)."""
        with client.websocket_connect(f"/ws/chat/{seeded_session}") as ws:
            # Consume the connected handshake first
            connected = ws.receive_json()
            assert connected["type"] == "connected"

            ws.send_text("not valid json {{")
            err = receive_non_ping(ws)
            assert err["type"] == "error"
            assert err["id"] == ""
            assert err["data"]["code"] == "INVALID_JSON"


# ---------------------------------------------------------------------------
# TestWebSocketQuery
# ---------------------------------------------------------------------------


class TestWebSocketQuery:
    def test_query_without_chat_provider(self, client: TestClient, seeded_session: str) -> None:
        """A query with no chat_provider_id returns either complete or error (no crash)."""
        with client.websocket_connect(f"/ws/chat/{seeded_session}") as ws:
            connected = ws.receive_json()
            assert connected["type"] == "connected"

            ws.send_json(
                {
                    "type": "query",
                    "id": "q1",
                    "query": "hello",
                    "content": "some content",
                    "mode": "direct",
                }
            )
            resp = receive_non_ping(ws, timeout=10, skip_types=("ping", "token", "step", "metrics"))
            assert resp["type"] in ("complete", "error")
            assert resp["id"] == "q1"

    def test_query_with_invalid_chat_provider(
        self, client: TestClient, seeded_session: str
    ) -> None:
        """A query referencing a non-existent chat_provider_id returns NOT_FOUND."""
        with client.websocket_connect(f"/ws/chat/{seeded_session}") as ws:
            connected = ws.receive_json()
            assert connected["type"] == "connected"

            ws.send_json(
                {
                    "type": "query",
                    "id": "q2",
                    "query": "hello",
                    "content": "some content",
                    "mode": "direct",
                    "chat_provider_id": "nonexistent",
                }
            )
            err = receive_non_ping(ws)
            assert err["type"] == "error"
            assert err["id"] == "q2"
            assert err["data"]["code"] == "NOT_FOUND"


# ---------------------------------------------------------------------------
# TestWebSocketCancel
# ---------------------------------------------------------------------------


class TestWebSocketCompaction:
    def test_compaction_step_is_streamed(
        self, client: TestClient, seeded_session: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A compaction row reaches the client as an ordinary ``step`` event."""
        from rlmstudio.application.sandbox_vars import (
            TRACE_KEY_ROLE,
            TRACE_ROLE_COMPACTION,
        )
        from tests.test_rlm_working_state_loop import CompactableLLM
        from tests.test_use_cases import FakeSandbox

        def inspect_peek(start: int) -> str:
            import json

            return json.dumps(
                {
                    "type": "inspect",
                    "tool": "peek",
                    "args": {"start": start, "end": start + 20},
                }
            )

        import json

        llm = CompactableLLM(
            [
                inspect_peek(0),
                inspect_peek(20),
                json.dumps({"type": "final", "answer": "done"}),
            ],
            context_window=100,
        )
        import rlmstudio.server.dependencies as deps

        monkeypatch.setattr(
            deps,
            "_registry_config_kwargs",
            lambda: {"compaction_keep_last": 0, "compaction_reserve_tokens": 0},
        )
        state = get_state()
        monkeypatch.setattr(state, "create_llm_adapter", lambda *a, **k: llm)
        monkeypatch.setattr(state, "create_sandbox", lambda *a, **k: FakeSandbox())

        with client.websocket_connect(f"/ws/chat/{seeded_session}") as ws:
            assert ws.receive_json()["type"] == "connected"
            ws.send_json(
                {
                    "type": "query",
                    "id": "q-compact",
                    "query": "what is here?",
                    "content": "alpha beta gamma delta epsilon zeta eta theta",
                    "mode": "rlm",
                }
            )
            seen: list[dict[str, Any]] = []
            while True:
                message = receive_non_ping(ws, timeout=10)
                seen.append(message)
                if message["type"] in ("complete", "error"):
                    break

        assert any(
            m["type"] == "step" and m["data"].get(TRACE_KEY_ROLE) == TRACE_ROLE_COMPACTION
            for m in seen
        )


class TestWebSocketCancel:
    def test_cancel_nonexistent(self, client: TestClient, seeded_session: str) -> None:
        """Cancelling an unknown id is silently ignored — no crash or error frame."""
        with client.websocket_connect(f"/ws/chat/{seeded_session}") as ws:
            connected = ws.receive_json()
            assert connected["type"] == "connected"

            ws.send_json({"type": "cancel", "id": "fake"})
            # No response is expected; send a follow-up query and confirm the
            # connection is still alive.
            ws.send_json(
                {
                    "type": "query",
                    "id": "q3",
                    "query": "ping",
                    "content": "test",
                    "mode": "direct",
                    "chat_provider_id": "nonexistent",
                }
            )
            # We expect a NOT_FOUND error proving the socket is still alive.
            resp = receive_non_ping(ws)
            assert resp["type"] == "error"
            assert resp["data"]["code"] == "NOT_FOUND"
