"""Phase 2 tests for the TelemetryStore schema-migration harness.

Covers AC-5 (v1 → v2 ALTERs add six steps columns + runs.outcome_category),
the v3 `steps.cost_known` flag, the v4 `runs.profile_id` /
`runs.content_sha256` columns (#78), AC-20 (idempotent reopen, PRAGMA
user_version bumped), and AC-22 (fresh install reaches the current version
via the migration path; `_SCHEMA_SQL` itself stays at the v1 baseline).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rlmstudio.telemetry import store as store_module
from rlmstudio.telemetry.store import _CURRENT_SCHEMA_VERSION, _SCHEMA_SQL, TelemetryStore

_V2_STEP_COLUMNS = {
    "prompt_tokens",
    "completion_tokens",
    "ttft_ms",
    "decode_ms",
    "cached_tokens",
    "cache_write_tokens",
}
_V2_RUN_COLUMNS = {"outcome_category"}
_V3_STEP_COLUMNS = {"cost_known"}
_V4_RUN_COLUMNS = {"profile_id", "content_sha256"}


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()  # nosec B608
    return {row[1] for row in rows}


def _user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


class TestFreshInstall:
    """AC-22 — fresh install runs through `_SCHEMA_SQL` (v1) + migrations."""

    def test_fresh_install_reaches_current_version_via_migrations(self, tmp_path: Path) -> None:
        db_path = tmp_path / "fresh.db"
        assert not db_path.exists()

        store = TelemetryStore(db_path=db_path)
        try:
            conn = store._connect()
            assert _user_version(conn) == _CURRENT_SCHEMA_VERSION
            assert _V2_STEP_COLUMNS.issubset(_columns(conn, "steps"))
            assert _V2_RUN_COLUMNS.issubset(_columns(conn, "runs"))
            assert _V3_STEP_COLUMNS.issubset(_columns(conn, "steps"))
            assert _V4_RUN_COLUMNS.issubset(_columns(conn, "runs"))
        finally:
            store.close()

    def test_schema_sql_stays_at_v1_baseline(self) -> None:
        """`_SCHEMA_SQL` must not grow migrated columns — reject the v1.2 bug
        where fresh DBs would carry them but have user_version=0,
        causing later ALTERs to fail as duplicate adds."""
        for column in _V2_STEP_COLUMNS | _V2_RUN_COLUMNS | _V3_STEP_COLUMNS | _V4_RUN_COLUMNS:
            assert column not in _SCHEMA_SQL, (
                f"_SCHEMA_SQL must not contain migrated column {column!r}; "
                f"migrations are the single source of truth post-v1."
            )


class TestUpgradeFromV1:
    """AC-5, AC-20 — opening a v1 DB runs the v2 ALTERs, idempotent on reopen."""

    def _seed_v1_db(self, db_path: Path) -> None:
        """Write the v1 baseline schema + pin user_version=1, simulating a
        DB that was last touched by the pre-Phase-2 code path."""
        conn = sqlite3.connect(db_path)
        try:
            conn.executescript(_SCHEMA_SQL)
            # Seed one row per table so we can prove migration leaves them intact.
            conn.execute(
                "INSERT INTO runs (id, created_at, mode) VALUES (?, ?, ?)",
                ("r1", 1000.0, "rlm"),
            )
            conn.execute(
                "INSERT INTO steps (id, run_id, step_index) VALUES (?, ?, ?)",
                ("s1", "r1", 0),
            )
            conn.execute("PRAGMA user_version = 1")
            conn.commit()
        finally:
            conn.close()

    def test_v1_to_current_migration_adds_columns_and_preserves_rows(self, tmp_path: Path) -> None:
        db_path = tmp_path / "legacy.db"
        self._seed_v1_db(db_path)

        store = TelemetryStore(db_path=db_path)
        try:
            conn = store._connect()
            # user_version advanced through every migration
            assert _user_version(conn) == _CURRENT_SCHEMA_VERSION
            # New columns exist
            assert _V2_STEP_COLUMNS.issubset(_columns(conn, "steps"))
            assert _V2_RUN_COLUMNS.issubset(_columns(conn, "runs"))
            assert _V3_STEP_COLUMNS.issubset(_columns(conn, "steps"))
            assert _V4_RUN_COLUMNS.issubset(_columns(conn, "runs"))
            # Legacy rows preserved
            run_row = conn.execute("SELECT id, mode FROM runs WHERE id = 'r1'").fetchone()
            assert run_row["id"] == "r1"
            assert run_row["mode"] == "rlm"
            step_row = conn.execute(
                "SELECT id, run_id, prompt_tokens, cost_known FROM steps WHERE id = 's1'"
            ).fetchone()
            assert step_row["run_id"] == "r1"
            assert step_row["prompt_tokens"] == 0  # default on pre-migration row
            assert step_row["cost_known"] is None  # legacy rows keep no verdict
        finally:
            store.close()

    def test_migration_is_idempotent(self, tmp_path: Path) -> None:
        """Reopening an already-migrated DB must not re-run the ALTERs
        (which would fail as duplicate-column adds)."""
        db_path = tmp_path / "reopen.db"

        first = TelemetryStore(db_path=db_path)
        first.close()

        # Reopen — migration path runs again but finds the current user_version
        # and skips. No exceptions, no duplicate-column errors.
        second = TelemetryStore(db_path=db_path)
        try:
            conn = second._connect()
            assert _user_version(conn) == _CURRENT_SCHEMA_VERSION
        finally:
            second.close()


class TestRunSummaryOutcomeCategory:
    """AC-20 follow-on: list_runs reads the new outcome_category column."""

    def test_default_is_none_on_fresh_runs(self, tmp_path: Path) -> None:
        store = TelemetryStore(db_path=tmp_path / "runs.db")
        try:
            store.record_run(
                run_id="r1",
                created_at=1000.0,
                mode="rlm",
                query="q",
            )
            runs = store.list_runs(limit=10)
            assert len(runs) == 1
            assert runs[0].outcome_category is None
        finally:
            store.close()


@pytest.mark.parametrize(
    "column_name",
    sorted(_V2_STEP_COLUMNS | _V2_RUN_COLUMNS | _V3_STEP_COLUMNS | _V4_RUN_COLUMNS),
)
def test_every_migrated_column_reachable_via_migration(tmp_path: Path, column_name: str) -> None:
    """Parametrized sanity: every migrated column is queryable after migration."""
    store = TelemetryStore(db_path=tmp_path / "all.db")
    try:
        conn = store._connect()
        table = "runs" if column_name in _V2_RUN_COLUMNS | _V4_RUN_COLUMNS else "steps"
        # A SELECT on the column must not raise.
        conn.execute(f"SELECT {column_name} FROM {table} LIMIT 1").fetchall()  # nosec B608
    finally:
        store.close()


class TestMigrationAtomicity:
    """A failed migration block rolls back with its ``user_version`` bump."""

    def test_failure_after_first_statement_rolls_back_and_retry_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db_path = tmp_path / "partial.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(_SCHEMA_SQL)
        conn.execute("INSERT INTO runs (id, created_at, mode) VALUES ('r1', 1.0, 'rlm')")
        conn.commit()
        conn.close()

        real_v4 = list(store_module._MIGRATIONS[4])
        failing = dict(store_module._MIGRATIONS)
        failing[4] = [real_v4[0], "ALTER TABLE no_such_table ADD COLUMN boom TEXT"]
        monkeypatch.setattr(store_module, "_MIGRATIONS", failing)
        closed: list[bool] = []
        real_close = TelemetryStore.close

        def tracking_close(self: TelemetryStore) -> None:
            closed.append(True)
            real_close(self)

        monkeypatch.setattr(TelemetryStore, "close", tracking_close)

        with pytest.raises(sqlite3.OperationalError, match="no_such_table"):
            TelemetryStore(db_path=db_path)
        # The failed constructor does not leave its connection open.
        assert closed == [True]

        raw = sqlite3.connect(db_path)
        try:
            # Earlier versions committed; the failed v4 block left nothing behind.
            assert _user_version(raw) == 3
            assert _V3_STEP_COLUMNS.issubset(_columns(raw, "steps"))
            assert not (_V4_RUN_COLUMNS & _columns(raw, "runs"))
            assert raw.execute("SELECT id FROM runs").fetchall() == [("r1",)]
        finally:
            raw.close()

        monkeypatch.undo()
        store = TelemetryStore(db_path=db_path)
        try:
            conn = store._connect()
            assert _user_version(conn) == _CURRENT_SCHEMA_VERSION
            assert _V4_RUN_COLUMNS.issubset(_columns(conn, "runs"))
            run = store.get_run("r1")
            assert run is not None
            assert run.profile_id is None
        finally:
            store.close()
