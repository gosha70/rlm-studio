"""Phase B verification tests: polling metrics, session persistence, dashboard data.

These tests verify the fixes for Bugs #2, #4, and #5 from the outstanding bugs handoff.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from rlmstudio.server.app import app
from rlmstudio.server.dependencies import AppState, ExecutionRecord, reset_state


@pytest.fixture(autouse=True)
def _clean_state() -> None:
    reset_state()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# Bug #2: Trace result exposes input/output tokens for polling path
# ---------------------------------------------------------------------------


class TestTraceTokenMetrics:
    def _create_execution_with_result(self, state: AppState) -> str:
        """Create a completed execution with token metrics."""
        session = state.get_or_create_session()
        exec_id = "test-exec-001"
        now = datetime.now(timezone.utc)
        execution = ExecutionRecord(
            execution_id=exec_id,
            session_id=session.id,
            query="test query",
            mode="direct",
            status="complete",
            started_at=now,
            completed_at=now,
            result={
                "answer": "test answer",
                "success": True,
                "input_tokens": 150,
                "output_tokens": 75,
                "total_tokens": 225,
                "total_cost": 0.0042,
                "elapsed_time": 1.5,
                "steps_count": 0,
            },
            steps=[
                {
                    "step": 0,
                    "role": "assistant",
                    "content": "test answer",
                    "input_tokens": 150,
                    "output_tokens": 75,
                    "elapsed_seconds": 1.5,
                }
            ],
        )
        state.executions[exec_id] = execution
        return exec_id

    def test_trace_result_includes_token_breakdown(self, client: TestClient) -> None:
        """GET /api/traces/{id} includes input_tokens, output_tokens, total_cost in result."""
        from rlmstudio.server.dependencies import get_state

        state = get_state()
        exec_id = self._create_execution_with_result(state)

        resp = client.get(f"/api/traces/{exec_id}")
        assert resp.status_code == 200
        data = resp.json()

        result = data["result"]
        assert result["input_tokens"] == 150
        assert result["output_tokens"] == 75
        assert result["total_cost"] == 0.0042

    def test_trace_budget_still_has_totals(self, client: TestClient) -> None:
        """Budget fields are still populated for backward compatibility."""
        from rlmstudio.server.dependencies import get_state

        state = get_state()
        exec_id = self._create_execution_with_result(state)

        resp = client.get(f"/api/traces/{exec_id}")
        data = resp.json()

        budget = data["budget"]
        assert budget["tokens_used"] == 225
        assert budget["cost_used"] == 0.0042

    def test_trace_result_defaults_to_zero(self, client: TestClient) -> None:
        """Missing token data defaults to 0, not None."""
        from rlmstudio.server.dependencies import get_state

        state = get_state()
        session = state.get_or_create_session()
        execution = ExecutionRecord(
            execution_id="test-exec-empty",
            session_id=session.id,
            query="test",
            mode="direct",
            status="complete",
            result={"answer": "ok", "success": True},
            steps=[],
        )
        state.executions["test-exec-empty"] = execution

        resp = client.get("/api/traces/test-exec-empty")
        result = resp.json()["result"]
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0
        assert result["total_cost"] == 0.0


# ---------------------------------------------------------------------------
# Bug #4: Dashboard empty state shown incorrectly
# (Backend side: metrics endpoint works with populated sessions)
# ---------------------------------------------------------------------------


class TestDashboardMetrics:
    def test_metrics_aggregates_by_provider(self, client: TestClient) -> None:
        """Metrics endpoint populates by_provider from the telemetry store."""
        from rlmstudio.server.dependencies import get_state

        state = get_state()
        session = state.get_or_create_session()
        now = datetime.now(timezone.utc)

        # Two completed runs from different providers, persisted via telemetry.
        state.telemetry.record_run(
            created_at=now.timestamp(),
            mode="direct",
            provider="openai",
            model="gpt-4o",
            query="q1",
            answer="answer 1",
            total_tokens=200,
            total_cost=0.003,
            elapsed_seconds=1.0,
            success=True,
            session_id=session.id,
            chat_provider_name="GPT-4o Config",
        )
        state.telemetry.record_run(
            created_at=now.timestamp() + 1,
            mode="direct",
            provider="anthropic",
            model="claude-sonnet-4-6",
            query="q2",
            answer="answer 2",
            total_tokens=300,
            total_cost=0.005,
            elapsed_seconds=2.0,
            success=True,
            session_id=session.id,
            chat_provider_name="Claude Config",
        )

        resp = client.get(f"/api/metrics/{session.id}")
        assert resp.status_code == 200
        data = resp.json()

        assert data["summary"]["total_queries"] == 2
        assert data["summary"]["total_tokens"] == 500
        assert "openai" in data["by_provider"]
        assert "anthropic" in data["by_provider"]
        assert data["by_provider"]["openai"]["queries"] == 1
        assert data["by_provider"]["anthropic"]["queries"] == 1

    def test_metrics_aggregates_by_chat_provider(self, client: TestClient) -> None:
        """Metrics endpoint populates by_chat_provider from the telemetry store."""
        from rlmstudio.server.dependencies import get_state

        state = get_state()
        session = state.get_or_create_session()
        now = datetime.now(timezone.utc)

        state.telemetry.record_run(
            created_at=now.timestamp(),
            mode="direct",
            provider="openai",
            model="gpt-4o",
            query="q",
            answer="answer",
            total_tokens=100,
            total_cost=0.001,
            elapsed_seconds=0.5,
            success=True,
            session_id=session.id,
            chat_provider_name="My GPT Config",
        )

        resp = client.get(f"/api/metrics/{session.id}")
        data = resp.json()

        assert "My GPT Config" in data["by_chat_provider"]
        assert data["by_chat_provider"]["My GPT Config"]["queries"] == 1


# ---------------------------------------------------------------------------
# Bug #5: Session persistence survives restart
# ---------------------------------------------------------------------------


class TestSessionPersistence:
    def test_default_session_name_uses_local_time_to_the_second(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default names show the user's clock, not UTC, and include seconds."""
        monkeypatch.setenv("TZ", "America/New_York")
        time.tzset()
        try:
            session = AppState(load_from_disk=False).get_or_create_session()
            local = session.created_at.astimezone()
        finally:
            monkeypatch.undo()
            time.tzset()
        assert local.utcoffset() != timedelta(0)
        assert session.name == f"Session {local.strftime('%Y-%m-%d %H:%M:%S')}"

    def test_sessions_loaded_from_disk(self) -> None:
        """AppState._load_sessions() correctly restores sessions."""
        state = AppState(load_from_disk=False)
        session = state.get_or_create_session()
        now = datetime.now(timezone.utc)
        state.add_message(
            session.id,
            {
                "role": "user",
                "content": "Hello",
                "timestamp": now.isoformat(),
            },
        )
        state.add_message(
            session.id,
            {
                "role": "assistant",
                "content": "Hi there",
                "mode_used": "direct",
                "metrics": {"total_tokens": 50, "cost_usd": 0.001, "elapsed_seconds": 0.3},
                "timestamp": now.isoformat(),
            },
        )

        # Manually serialize and deserialize (simulates restart)
        serialized = [
            {
                "id": session.id,
                "name": session.name,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "messages": session.messages,
                "conversations": session.conversations,
            }
        ]

        state2 = AppState(load_from_disk=False)
        assert len(state2.sessions) == 0

        # Simulate loading from disk
        with patch("rlmstudio.server.dependencies._SESSIONS_FILE") as mock_file:
            mock_file.exists.return_value = True
            mock_file.read_text.return_value = json.dumps(serialized, default=str)
            state2._load_sessions()

        assert len(state2.sessions) == 1
        loaded = state2.sessions[session.id]
        assert len(loaded.messages) == 2
        assert loaded.messages[0]["content"] == "Hello"
        assert loaded.messages[1]["content"] == "Hi there"

    def test_rest_error_path_saves_sessions(self) -> None:
        """_run_execution error path calls save_sessions() so errors persist."""
        import inspect

        from rlmstudio.server.routes import chat

        source = inspect.getsource(chat._run_execution)
        except_idx = source.rfind("except Exception")
        assert except_idx != -1
        after_except = source[except_idx:]
        assert "save_sessions()" in after_except, (
            "_run_execution error path must call save_sessions()"
        )

    def test_ws_error_path_saves_sessions(self) -> None:
        """_ws_execute error path calls save_sessions() so WS errors persist."""
        import inspect

        from rlmstudio.server.routes import chat

        source = inspect.getsource(chat.websocket_chat)
        # Find the _ws_execute inner function's except block
        ws_exec_idx = source.find("async def _ws_execute")
        assert ws_exec_idx != -1, "websocket_chat should contain _ws_execute"
        ws_source = source[ws_exec_idx:]
        except_idx = ws_source.rfind("except Exception")
        assert except_idx != -1
        after_except = ws_source[except_idx:]
        assert "save_sessions()" in after_except, "_ws_execute error path must call save_sessions()"
        assert "add_message" in after_except, (
            "_ws_execute error path must add error message to session"
        )


# ---------------------------------------------------------------------------
# File persistence (regression: uploaded files must survive server restarts)
# ---------------------------------------------------------------------------


class TestFilePersistence:
    """Uploaded files must survive ``uvicorn --reload`` and process restarts,
    otherwise any session whose messages reference ``file_ids`` will 404 on
    the next chat turn (the "File not found" bug)."""

    def test_files_round_trip_through_disk(self, tmp_path: object) -> None:
        """AppState.save_files() + _load_files() round-trip preserves all
        FileRecord fields so ``/api/chat`` can still resolve file_ids after
        a restart."""
        from unittest.mock import patch

        from rlmstudio.server.dependencies import AppState, FileRecord

        files_path = tmp_path / "files.json"  # type: ignore[operator]
        with patch("rlmstudio.server.dependencies._FILES_FILE", files_path):
            # Producer: upload a file, persist.
            src = AppState(load_from_disk=False)
            now = datetime.now(timezone.utc)
            src.files["f-1"] = FileRecord(
                id="f-1",
                name="doc.pdf",
                size_bytes=1024,
                content_type="application/pdf",
                text_content="Hello world",
                token_count=3,
                created_at=now,
            )
            src.save_files()
            assert files_path.exists()

            # Consumer: fresh AppState (simulates restart), load from disk.
            dst = AppState(load_from_disk=False)
            assert dst.files == {}  # fresh
            dst._load_files()

            assert "f-1" in dst.files
            rec = dst.files["f-1"]
            assert rec.name == "doc.pdf"
            assert rec.size_bytes == 1024
            assert rec.content_type == "application/pdf"
            assert rec.text_content == "Hello world"
            assert rec.token_count == 3
            # created_at round-trips via isoformat
            assert rec.created_at == now

    def test_upload_endpoint_persists_to_disk(self, tmp_path: object, client: TestClient) -> None:
        """POST /api/files/upload must call save_files() so the file
        survives an immediate restart (the actual bug the user reported)."""
        from unittest.mock import patch

        from rlmstudio.server.dependencies import AppState, get_state

        # reset_state() stubs save_files = lambda: None on the singleton to
        # avoid polluting the user's ~/.rlmkit/ during test runs.  For this
        # test we need the real method so the upload actually writes disk.
        state = get_state()
        state.save_files = AppState.save_files.__get__(state, AppState)  # type: ignore[method-assign]

        files_path = tmp_path / "files.json"  # type: ignore[operator]
        with patch("rlmstudio.server.dependencies._FILES_FILE", files_path):
            resp = client.post(
                "/api/files/upload",
                files={"file": ("doc.txt", b"Hello world", "text/plain")},
            )
            assert resp.status_code == 201
            file_id = resp.json()["id"]

            # Disk file should now contain the uploaded record.
            assert files_path.exists()
            data = json.loads(files_path.read_text())
            assert len(data) == 1
            assert data[0]["id"] == file_id
            assert data[0]["name"] == "doc.txt"
            assert data[0]["text_content"] == "Hello world"

    def test_chat_resolves_file_after_reload(self, tmp_path: object, client: TestClient) -> None:
        """End-to-end regression: upload a file, wipe in-memory state to
        simulate a restart, reload from disk, and verify the file can still
        be resolved for a chat request."""
        from unittest.mock import patch

        from rlmstudio.server.dependencies import AppState, get_state

        # Restore the real save_files on the singleton (reset_state stubs it).
        state = get_state()
        state.save_files = AppState.save_files.__get__(state, AppState)  # type: ignore[method-assign]

        files_path = tmp_path / "files.json"  # type: ignore[operator]
        with patch("rlmstudio.server.dependencies._FILES_FILE", files_path):
            # Step 1: upload a file (the real upload path persists it).
            upload_resp = client.post(
                "/api/files/upload",
                files={"file": ("doc.txt", b"The quick brown fox", "text/plain")},
            )
            assert upload_resp.status_code == 201
            file_id = upload_resp.json()["id"]

            # Step 2: wipe state.files in place to simulate a process restart.
            # (The real AppState singleton persists across requests, so we
            # clear its dict rather than replacing the instance.)
            state = get_state()
            assert file_id in state.files
            state.files.clear()
            assert file_id not in state.files

            # Step 3: load from disk and assert the file came back.
            state._load_files()
            assert file_id in state.files, (
                "File persistence broken: restarted state did not restore "
                "the uploaded file from disk"
            )
            rec = state.files[file_id]
            assert rec.text_content == "The quick brown fox"
