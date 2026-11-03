"""Phase A foundations for prompt refinement (#78).

Covers spec FR-1, FR-2, FR-4, FR-9, FR-11, FR-11a and FR-12 at the
storage layer: telemetry schema v4, run attribution and document identity,
the refinement record store (raising writers + status compare-and-set),
the run writers, and the checked config save.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from rlmstudio.application.dto import RunResultDTO
from rlmstudio.application.refinement_constants import (
    REFINEMENT_INTERRUPTED_ON_STARTUP_ERROR,
    REFINEMENT_STATUS_ACCEPTED,
    REFINEMENT_STATUS_EVALUATED,
    REFINEMENT_STATUS_EVALUATING,
    REFINEMENT_STATUS_INTERRUPTED,
    REFINEMENT_STATUS_PROPOSAL_READY,
    REFINEMENT_STATUS_REJECTED,
)
from rlmstudio.application.use_cases.run_matrix_comparison import MatrixSlotResultDTO
from rlmstudio.server import dependencies
from rlmstudio.server.dependencies import AppState, ExecutionRecord
from rlmstudio.server.routes._helpers import _record_slot_telemetry, _SlotMeta
from rlmstudio.server.routes.chat import _record_telemetry
from rlmstudio.telemetry import store as store_module
from rlmstudio.telemetry.store import (
    RefinementConflictError,
    RefinementStoreError,
    TelemetryStore,
    content_sha256,
)

DOC = "The quarterly report names Harbour Platform as the launch."
PROFILE = "prof-1"


def _abort_trigger(store: TelemetryStore, event: str) -> None:
    """Make every *event* (INSERT/UPDATE) on ``refinements`` fail in SQLite."""
    with store._lock:
        conn = store._connect()
        conn.execute(
            f"CREATE TRIGGER inject_{event.lower()} BEFORE {event} ON refinements "
            "BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
        )
        conn.commit()


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


# ---------------------------------------------------------------------------
# Schema v4
# ---------------------------------------------------------------------------


class TestSchemaV4:
    def test_fresh_database_has_v4_columns_and_table(self) -> None:
        store = TelemetryStore(":memory:")
        conn = store._connect()
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 4
        assert {"profile_id", "content_sha256"} <= _columns(conn, "runs")
        assert {
            "id",
            "profile_id",
            "status",
            "created_at",
            "updated_at",
            "proposal_json",
            "evaluation_json",
            "decision_json",
            "error",
        } == _columns(conn, "refinements")

    def test_v3_database_upgrades_and_keeps_legacy_rows_unattributed(self, tmp_path: Path) -> None:
        db = tmp_path / "telemetry.db"
        conn = sqlite3.connect(db)
        conn.executescript(store_module._SCHEMA_SQL)
        for version in (2, 3):
            for stmt in store_module._MIGRATIONS[version]:
                conn.execute(stmt)
        conn.execute("PRAGMA user_version = 3")
        conn.execute("INSERT INTO runs (id, created_at, mode) VALUES ('legacy', 1.0, 'rlm')")
        conn.commit()
        conn.close()

        store = TelemetryStore(db)
        assert int(store._connect().execute("PRAGMA user_version").fetchone()[0]) == 4
        legacy = store.get_run("legacy")
        assert legacy is not None
        assert legacy.profile_id is None
        assert legacy.content_sha256 is None
        store.close()

        # Reopening an already-migrated database is a no-op.
        reopened = TelemetryStore(db)
        assert reopened.get_run("legacy") is not None
        reopened.close()


# ---------------------------------------------------------------------------
# Run attribution (FR-1, FR-2)
# ---------------------------------------------------------------------------


class TestRunAttribution:
    def test_content_sha256_is_utf8_sha256(self) -> None:
        assert content_sha256(DOC) == hashlib.sha256(DOC.encode("utf-8")).hexdigest()

    def test_record_run_round_trips_profile_and_hash(self) -> None:
        store = TelemetryStore(":memory:")
        store.record_run(
            run_id="r1",
            created_at=1.0,
            mode="rlm",
            outcome_category="timeout",
            profile_id=PROFILE,
            content_sha256=content_sha256(DOC),
        )
        detail = store.get_run("r1")
        assert detail is not None
        assert detail.profile_id == PROFILE
        assert detail.content_sha256 == content_sha256(DOC)
        # get_run now carries the outcome category the digest needs (FR-3).
        assert detail.outcome_category == "timeout"
        summary = store.list_runs()[0]
        assert summary.profile_id == PROFILE
        assert summary.content_sha256 == content_sha256(DOC)

    def test_list_runs_filters_by_profile_and_requires_hash(self) -> None:
        store = TelemetryStore(":memory:")
        store.record_run(
            run_id="mine", created_at=3.0, mode="rlm", profile_id=PROFILE, content_sha256="h"
        )
        store.record_run(run_id="mine-legacy", created_at=2.0, mode="rlm", profile_id=PROFILE)
        store.record_run(
            run_id="other", created_at=1.0, mode="rlm", profile_id="prof-2", content_sha256="h"
        )

        ids = [r.id for r in store.list_runs(profile_id=PROFILE)]
        assert ids == ["mine", "mine-legacy"]
        ids = [r.id for r in store.list_runs(profile_id=PROFILE, require_content_hash=True)]
        assert ids == ["mine"]


# ---------------------------------------------------------------------------
# Refinement records (FR-4, FR-11a, FR-12)
# ---------------------------------------------------------------------------


class TestRefinementStore:
    def test_create_get_list(self) -> None:
        store = TelemetryStore(":memory:")
        first = store.create_refinement(
            profile_id=PROFILE,
            status=REFINEMENT_STATUS_PROPOSAL_READY,
            proposal={"candidate": "A"},
            created_at=1.0,
        )
        second = store.create_refinement(
            profile_id=PROFILE,
            status=REFINEMENT_STATUS_PROPOSAL_READY,
            proposal={"candidate": "B"},
            created_at=2.0,
        )
        store.create_refinement(
            profile_id="prof-2", status=REFINEMENT_STATUS_PROPOSAL_READY, proposal={}
        )

        fetched = store.get_refinement(first.id)
        assert fetched is not None
        assert fetched.proposal == {"candidate": "A"}
        assert fetched.evaluation is None
        assert fetched.decision is None
        assert [r.id for r in store.list_refinements(PROFILE)] == [second.id, first.id]
        assert store.get_refinement("missing") is None

    def test_update_is_a_status_compare_and_set(self) -> None:
        store = TelemetryStore(":memory:")
        rec = store.create_refinement(
            profile_id=PROFILE, status=REFINEMENT_STATUS_PROPOSAL_READY, proposal={}
        )
        updated = store.update_refinement(
            rec.id,
            expected_status=REFINEMENT_STATUS_PROPOSAL_READY,
            status=REFINEMENT_STATUS_EVALUATING,
        )
        assert updated.status == REFINEMENT_STATUS_EVALUATING

        done = store.update_refinement(
            rec.id,
            expected_status=[REFINEMENT_STATUS_EVALUATING],
            status=REFINEMENT_STATUS_EVALUATED,
            evaluation={"complete": True},
        )
        assert done.evaluation == {"complete": True}

        with pytest.raises(RefinementConflictError, match="'evaluated'"):
            store.update_refinement(
                rec.id,
                expected_status=REFINEMENT_STATUS_PROPOSAL_READY,
                status=REFINEMENT_STATUS_REJECTED,
            )
        # The losing CAS changed nothing.
        after = store.get_refinement(rec.id)
        assert after is not None
        assert after.status == REFINEMENT_STATUS_EVALUATED

    def test_update_missing_row_raises_conflict(self) -> None:
        store = TelemetryStore(":memory:")
        with pytest.raises(RefinementConflictError, match="not found"):
            store.update_refinement(
                "nope",
                expected_status=REFINEMENT_STATUS_EVALUATED,
                status=REFINEMENT_STATUS_ACCEPTED,
            )

    def test_update_requires_an_expected_status(self) -> None:
        store = TelemetryStore(":memory:")
        with pytest.raises(ValueError):
            store.update_refinement("x", expected_status=[], status=REFINEMENT_STATUS_ACCEPTED)

    def test_create_propagates_database_failure(self) -> None:
        store = TelemetryStore(":memory:")
        _abort_trigger(store, "INSERT")
        with pytest.raises(RefinementStoreError, match="injected failure"):
            store.create_refinement(
                profile_id=PROFILE, status=REFINEMENT_STATUS_PROPOSAL_READY, proposal={}
            )

    def test_update_propagates_database_failure(self) -> None:
        store = TelemetryStore(":memory:")
        rec = store.create_refinement(
            profile_id=PROFILE, status=REFINEMENT_STATUS_EVALUATED, proposal={}
        )
        _abort_trigger(store, "UPDATE")
        with pytest.raises(RefinementStoreError, match="injected failure") as info:
            store.update_refinement(
                rec.id,
                expected_status=REFINEMENT_STATUS_EVALUATED,
                status=REFINEMENT_STATUS_ACCEPTED,
                decision={"accepted": True},
            )
        assert not isinstance(info.value, RefinementConflictError)
        after = store.get_refinement(rec.id)
        assert after is not None
        assert after.status == REFINEMENT_STATUS_EVALUATED
        assert after.decision is None

    def test_mark_evaluating_as_interrupted(self) -> None:
        store = TelemetryStore(":memory:")
        running = store.create_refinement(
            profile_id=PROFILE, status=REFINEMENT_STATUS_EVALUATING, proposal={}
        )
        ready = store.create_refinement(
            profile_id=PROFILE, status=REFINEMENT_STATUS_PROPOSAL_READY, proposal={}
        )
        assert store.mark_evaluating_as_interrupted() == 1
        interrupted = store.get_refinement(running.id)
        assert interrupted is not None
        assert interrupted.status == REFINEMENT_STATUS_INTERRUPTED
        assert interrupted.error == REFINEMENT_INTERRUPTED_ON_STARTUP_ERROR
        untouched = store.get_refinement(ready.id)
        assert untouched is not None
        assert untouched.status == REFINEMENT_STATUS_PROPOSAL_READY

    def test_mark_interrupted_propagates_database_failure(self) -> None:
        store = TelemetryStore(":memory:")
        store.create_refinement(
            profile_id=PROFILE, status=REFINEMENT_STATUS_EVALUATING, proposal={}
        )
        _abort_trigger(store, "UPDATE")
        with pytest.raises(RefinementStoreError):
            store.mark_evaluating_as_interrupted()


# ---------------------------------------------------------------------------
# Run writers (FR-1)
# ---------------------------------------------------------------------------


def _result(mode: str = "rlm") -> RunResultDTO:
    return RunResultDTO(
        answer="Harbour Platform",
        mode_used=mode,
        success=True,
        steps=1,
        input_tokens=10,
        output_tokens=5,
        total_cost=0.001,
        elapsed_time=0.2,
        trace=[],
    )


class TestRunWriters:
    def test_chat_writer_records_profile_and_content_hash(self) -> None:
        state = AppState(load_from_disk=False)
        execution = ExecutionRecord(
            execution_id="exec-chat",
            session_id="s",
            query="q",
            mode="rlm",
            status="complete",
            started_at=datetime.now(timezone.utc),
            chat_provider_id="cp-1",
        )
        _record_telemetry(
            state,
            execution,
            _result(),
            provider="openai",
            model="gpt-4o",
            profile_id=PROFILE,
            content=DOC,
        )
        detail = state.telemetry.get_run("exec-chat")
        assert detail is not None
        assert detail.profile_id == PROFILE
        assert detail.content_sha256 == content_sha256(DOC)

    def test_chat_writer_without_content_leaves_hash_empty(self) -> None:
        state = AppState(load_from_disk=False)
        execution = ExecutionRecord(
            execution_id="exec-legacy-call",
            session_id="s",
            query="q",
            mode="rlm",
            status="complete",
            started_at=datetime.now(timezone.utc),
        )
        _record_telemetry(state, execution, _result(), provider="openai", model="gpt-4o")
        detail = state.telemetry.get_run("exec-legacy-call")
        assert detail is not None
        assert detail.profile_id is None
        assert detail.content_sha256 is None

    def test_slot_writer_records_profile_and_content_hash(self) -> None:
        state = AppState(load_from_disk=False)
        slot = MatrixSlotResultDTO(
            slot_id="slot-1",
            label="baseline",
            mode="rlm",
            provider="openai",
            model="gpt-4o",
            result=_result(),
        )
        _record_slot_telemetry(
            state=state,
            slot_result=slot,
            meta=_SlotMeta(chat_provider_id="cp-1", execution_id="exec-slot", profile_id=PROFILE),
            session_id="s",
            comparison_group_id="g",
            query="q",
            created_at=datetime.now(timezone.utc),
            content_sha256=content_sha256(DOC),
        )
        detail = state.telemetry.get_run("exec-slot")
        assert detail is not None
        assert detail.profile_id == PROFILE
        assert detail.content_sha256 == content_sha256(DOC)


# ---------------------------------------------------------------------------
# Checked config save (FR-9, FR-11)
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(dependencies, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(dependencies, "_CONFIG_FILE", config_path)
    return config_path


_REAL_NAMED_TEMPORARY_FILE = dependencies.tempfile.NamedTemporaryFile


class _FailingWrite:
    """Delegates to a real temp file but raises on ``write`` (ENOSPC on write())."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._tmp = _REAL_NAMED_TEMPORARY_FILE(*args, **kwargs)
        self.name = self._tmp.name

    def write(self, _data: str) -> int:
        raise OSError("simulated write failure")

    def flush(self) -> None:
        self._tmp.flush()

    def fileno(self) -> int:
        return self._tmp.fileno()

    def close(self) -> None:
        self._tmp.close()


def _fault_patch(fault: str) -> Any:
    if fault == "write":
        return patch("rlmstudio.server.dependencies.tempfile.NamedTemporaryFile", _FailingWrite)
    if fault == "fsync":
        return patch(
            "rlmstudio.server.dependencies.os.fsync", side_effect=OSError("simulated fsync")
        )
    return patch(
        "rlmstudio.server.dependencies.os.replace", side_effect=OSError("simulated replace")
    )


class TestSaveConfigChecked:
    def _primed_state(self) -> tuple[AppState, str]:
        state = AppState(load_from_disk=False)
        state.config.active_model = "before"
        state.save_config_checked()
        return state, "before"

    @pytest.mark.parametrize("fault", ["write", "fsync", "replace"])
    def test_checked_save_raises_and_keeps_previous_file(
        self, isolated_config: Path, fault: str
    ) -> None:
        state, before = self._primed_state()
        good = isolated_config.read_text()
        state.config.active_model = "after"

        with _fault_patch(fault), pytest.raises(OSError, match="simulated"):
            state.save_config_checked()

        assert isolated_config.read_text() == good
        assert json.loads(good)["config"]["active_model"] == before
        leaked = [p.name for p in isolated_config.parent.glob(".config.*.tmp")]
        assert leaked == []

    def test_unchecked_save_still_swallows(self, isolated_config: Path) -> None:
        state, _before = self._primed_state()
        with patch(
            "rlmstudio.server.dependencies.os.replace", side_effect=OSError("simulated replace")
        ):
            state.save_config()  # must not raise

    def test_checked_save_writes_when_healthy(self, isolated_config: Path) -> None:
        state = AppState(load_from_disk=False)
        state.config.active_model = "written"
        state.save_config_checked()
        assert json.loads(isolated_config.read_text())["config"]["active_model"] == "written"
