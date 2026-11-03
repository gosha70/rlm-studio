"""SQLite-backed telemetry store for execution history and analytics.

Provides persistent, queryable storage for runs, steps, provider calls,
and ratings.  All writes are synchronous (<1 ms for single rows) and
wrapped in try/finally so telemetry failures never block execution.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rlmstudio.branding import state_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    created_at      REAL NOT NULL,
    mode            TEXT NOT NULL,
    provider        TEXT NOT NULL DEFAULT '',
    model           TEXT NOT NULL DEFAULT '',
    query           TEXT NOT NULL DEFAULT '',
    content_length  INTEGER NOT NULL DEFAULT 0,
    answer_length   INTEGER NOT NULL DEFAULT 0,
    total_tokens    INTEGER NOT NULL DEFAULT 0,
    total_cost      REAL NOT NULL DEFAULT 0.0,
    elapsed_seconds REAL NOT NULL DEFAULT 0.0,
    success         INTEGER NOT NULL DEFAULT 1,
    error           TEXT,
    session_id      TEXT,
    chat_provider_id TEXT,
    chat_provider_name TEXT,
    comparison_group_id TEXT,
    answer          TEXT NOT NULL DEFAULT '',
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    steps_count     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);
CREATE INDEX IF NOT EXISTS idx_runs_mode ON runs(mode);
CREATE INDEX IF NOT EXISTS idx_runs_session ON runs(session_id);

CREATE TABLE IF NOT EXISTS steps (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    step_index      INTEGER NOT NULL,
    action_type     TEXT NOT NULL DEFAULT 'inspect',
    code            TEXT,
    output          TEXT,
    tokens          INTEGER NOT NULL DEFAULT 0,
    cost            REAL NOT NULL DEFAULT 0.0,
    duration        REAL NOT NULL DEFAULT 0.0,
    model           TEXT,
    recursion_depth INTEGER NOT NULL DEFAULT 0,
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id, step_index);

CREATE TABLE IF NOT EXISTS provider_calls (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    step_index      INTEGER NOT NULL DEFAULT 0,
    provider        TEXT NOT NULL DEFAULT '',
    model           TEXT NOT NULL DEFAULT '',
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    cost            REAL NOT NULL DEFAULT 0.0,
    latency_ms      INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'ok',
    error           TEXT
);
CREATE INDEX IF NOT EXISTS idx_calls_run ON provider_calls(run_id);

CREATE TABLE IF NOT EXISTS ratings (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    rating          INTEGER NOT NULL,
    comment         TEXT,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ratings_run ON ratings(run_id);
"""


# ---------------------------------------------------------------------------
# Schema migrations
# ---------------------------------------------------------------------------
#
# `_SCHEMA_SQL` above is **frozen at the v1 baseline** and must not grow
# new columns. Every post-v1 schema change goes in `_MIGRATIONS` keyed by
# target version; both fresh installs and upgrades run through the same
# migration path via `_run_migrations()` using SQLite's `PRAGMA user_version`.
# This keeps the bootstrap rule pinned: one code path, no branching.

_CURRENT_SCHEMA_VERSION = 3

_MIGRATIONS: dict[int, list[str]] = {
    2: [
        # Prefill/decode telemetry fields on steps.
        "ALTER TABLE steps ADD COLUMN prompt_tokens INTEGER DEFAULT 0",
        "ALTER TABLE steps ADD COLUMN completion_tokens INTEGER DEFAULT 0",
        "ALTER TABLE steps ADD COLUMN ttft_ms INTEGER",
        "ALTER TABLE steps ADD COLUMN decode_ms INTEGER DEFAULT 0",
        "ALTER TABLE steps ADD COLUMN cached_tokens INTEGER DEFAULT 0",
        "ALTER TABLE steps ADD COLUMN cache_write_tokens INTEGER DEFAULT 0",
        # Run-level outcome category (wired to writers in phase 4).
        "ALTER TABLE runs ADD COLUMN outcome_category TEXT",
    ],
    3: [
        # Whether a step's stored cost is a measurement (1) or zero by
        # ignorance (0).  NULL on legacy rows means "not recorded", so
        # read-time pricing falls back to the LiteLLM map.
        "ALTER TABLE steps ADD COLUMN cost_known INTEGER",
    ],
}


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Apply every migration whose target version exceeds PRAGMA user_version.

    Uses SQLite's built-in ``user_version`` pragma as the schema-version
    store — no extra table required. Idempotent: reopening an
    already-migrated DB is a no-op because each migration block bumps
    the pragma only after its statements commit successfully.

    Fresh installs run the v1 `_SCHEMA_SQL` (pragma starts at 0) and then
    the v2 migration adds the new columns. Upgrades from pre-Phase-2 DBs
    also start at 0 and follow the same path.
    """
    current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    for target_version in sorted(_MIGRATIONS):
        if target_version <= current:
            continue
        for stmt in _MIGRATIONS[target_version]:
            conn.execute(stmt)
        conn.execute(f"PRAGMA user_version = {target_version}")
        conn.commit()


# ---------------------------------------------------------------------------
# Data classes for query results
# ---------------------------------------------------------------------------


@dataclass
class RunSummary:
    """Summary of a single run returned by list_runs()."""

    id: str
    created_at: float
    mode: str
    provider: str
    model: str
    query: str
    total_tokens: int
    total_cost: float
    elapsed_seconds: float
    success: bool
    error: str | None
    session_id: str | None
    chat_provider_id: str | None
    chat_provider_name: str | None
    steps_count: int
    answer_length: int
    comparison_group_id: str | None = None
    answer: str = ""
    outcome_category: str | None = None


@dataclass
class RunDetail(RunSummary):
    """Full run detail including steps."""

    input_tokens: int = 0
    output_tokens: int = 0
    content_length: int = 0
    steps: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AggregateRow:
    """A single row from an aggregation query."""

    key: str
    run_count: int
    total_tokens: int
    total_cost: float
    avg_elapsed: float
    success_rate: float


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class TelemetryStore:
    """SQLite-backed persistent store for execution telemetry.

    Args:
        db_path: Path to the SQLite database file.
            Defaults to ``<state dir>/telemetry.db``.
            Use ``":memory:"`` for tests.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = state_dir() / "telemetry.db"
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        # Serializes all reads and writes so the store is safe to use from
        # multiple threads (e.g. FastAPI background tasks + WebSocket handlers).
        # SQLite's own locking is page-level; Python-level RLock ensures that
        # "execute + commit" sequences remain atomic.
        self._lock = threading.RLock()
        # Ensure the parent directory exists before sqlite3.connect, so first
        # runs on a clean machine don't fail with "unable to open database file".
        if self._db_path != ":memory:":
            Path(self._db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
            _run_migrations(conn)

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def record_run(
        self,
        *,
        run_id: str | None = None,
        created_at: float,
        mode: str,
        provider: str = "",
        model: str = "",
        query: str = "",
        content_length: int = 0,
        answer: str = "",
        answer_length: int | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        total_cost: float = 0.0,
        elapsed_seconds: float = 0.0,
        success: bool = True,
        error: str | None = None,
        session_id: str | None = None,
        chat_provider_id: str | None = None,
        chat_provider_name: str | None = None,
        comparison_group_id: str | None = None,
        steps_count: int = 0,
        outcome_category: str | None = None,
    ) -> str:
        """Record a completed run. Returns the run ID.

        ``outcome_category`` is an already-classified string from the
        caller (values are expected to be valid
        :class:`OutcomeCategory` values, but the store is a dumb sink
        and does not validate). Passing ``None`` is permitted; legacy
        rows surface ``None`` on read and re-derive via the no-trace
        classifier path on the dashboard.
        """
        rid = run_id or uuid.uuid4().hex
        if answer_length is None:
            answer_length = len(answer)
        with self._lock:
            try:
                conn = self._connect()
                conn.execute(
                    """INSERT INTO runs
                       (id, created_at, mode, provider, model, query,
                        content_length, answer, answer_length, input_tokens,
                        output_tokens, total_tokens, total_cost, elapsed_seconds,
                        success, error, session_id, chat_provider_id,
                        chat_provider_name, comparison_group_id, steps_count,
                        outcome_category)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        rid,
                        created_at,
                        mode,
                        provider,
                        model,
                        query,
                        content_length,
                        answer,
                        answer_length,
                        input_tokens,
                        output_tokens,
                        total_tokens,
                        total_cost,
                        elapsed_seconds,
                        int(success),
                        error,
                        session_id,
                        chat_provider_id,
                        chat_provider_name,
                        comparison_group_id,
                        steps_count,
                        outcome_category,
                    ),
                )
                conn.commit()
            except Exception:
                logger.exception("Failed to record run %s", rid)
        return rid

    def record_step(
        self,
        *,
        run_id: str,
        step_index: int,
        action_type: str = "inspect",
        code: str | None = None,
        output: str | None = None,
        tokens: int = 0,
        cost: float = 0.0,
        duration: float = 0.0,
        model: str | None = None,
        recursion_depth: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        ttft_ms: int | None = None,
        decode_ms: int = 0,
        cached_tokens: int = 0,
        cache_write_tokens: int = 0,
        cost_known: bool | None = None,
    ) -> str:
        """Record a single step within a run. Returns the step ID.

        The six prefill/decode fields (``prompt_tokens``,
        ``completion_tokens``, ``ttft_ms``, ``decode_ms``,
        ``cached_tokens``, ``cache_write_tokens``) ship with safe
        defaults so older callers that don't pass them stay compatible.
        The underlying columns live on schema v2 (see ``_MIGRATIONS``).

        ``cost_known`` is ``None`` for legacy/not-applicable rows, ``True``
        when ``cost`` is a measurement (including a known zero), and
        ``False`` when cost is zero by ignorance.  The column is added by
        schema v3.
        """
        sid = uuid.uuid4().hex
        with self._lock:
            try:
                conn = self._connect()
                conn.execute(
                    """INSERT INTO steps
                       (id, run_id, step_index, action_type, code, output,
                        tokens, cost, duration, model, recursion_depth,
                        input_tokens, output_tokens,
                        prompt_tokens, completion_tokens, ttft_ms,
                        decode_ms, cached_tokens, cache_write_tokens,
                        cost_known)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        sid,
                        run_id,
                        step_index,
                        action_type,
                        code,
                        output,
                        tokens,
                        cost,
                        duration,
                        model,
                        recursion_depth,
                        input_tokens,
                        output_tokens,
                        prompt_tokens,
                        completion_tokens,
                        ttft_ms,
                        decode_ms,
                        cached_tokens,
                        cache_write_tokens,
                        1 if cost_known is True else 0 if cost_known is False else None,
                    ),
                )
                conn.commit()
            except Exception:
                logger.exception("Failed to record step %d for run %s", step_index, run_id)
        return sid

    def record_call(
        self,
        *,
        run_id: str,
        step_index: int = 0,
        provider: str = "",
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
        latency_ms: int = 0,
        status: str = "ok",
        error: str | None = None,
    ) -> str:
        """Record a provider API call. Returns the call ID."""
        cid = uuid.uuid4().hex
        with self._lock:
            try:
                conn = self._connect()
                conn.execute(
                    """INSERT INTO provider_calls
                       (id, run_id, step_index, provider, model,
                        input_tokens, output_tokens, cost, latency_ms,
                        status, error)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        cid,
                        run_id,
                        step_index,
                        provider,
                        model,
                        input_tokens,
                        output_tokens,
                        cost,
                        latency_ms,
                        status,
                        error,
                    ),
                )
                conn.commit()
            except Exception:
                logger.exception("Failed to record call for run %s step %d", run_id, step_index)
        return cid

    def add_rating(
        self,
        *,
        run_id: str,
        rating: int,
        comment: str | None = None,
        created_at: float,
    ) -> str:
        """Add a user rating for a run. Returns the rating ID."""
        rid = uuid.uuid4().hex
        with self._lock:
            try:
                conn = self._connect()
                conn.execute(
                    """INSERT INTO ratings (id, run_id, rating, comment, created_at)
                       VALUES (?,?,?,?,?)""",
                    (rid, run_id, rating, comment, created_at),
                )
                conn.commit()
            except Exception:
                logger.exception("Failed to add rating for run %s", run_id)
        return rid

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def list_runs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        mode: str | None = None,
        session_id: str | None = None,
        chat_provider_id: str | None = None,
        comparison_group_id: str | None = None,
        success: bool | None = None,
    ) -> list[RunSummary]:
        """List runs ordered by created_at descending."""
        clauses: list[str] = []
        params: list[Any] = []
        if mode is not None:
            clauses.append("mode = ?")
            params.append(mode)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if chat_provider_id is not None:
            clauses.append("chat_provider_id = ?")
            params.append(chat_provider_id)
        if comparison_group_id is not None:
            clauses.append("comparison_group_id = ?")
            params.append(comparison_group_id)
        if success is not None:
            clauses.append("success = ?")
            params.append(int(success))

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.extend([limit, offset])

        with self._lock:
            rows = (
                self._connect()
                .execute(
                    f"SELECT * FROM runs{where} ORDER BY created_at DESC LIMIT ? OFFSET ?",  # nosec B608
                    tuple(params),
                )
                .fetchall()
            )

        return [self._row_to_summary(r) for r in rows]

    def get_run(self, run_id: str) -> RunDetail | None:
        """Get full detail for a single run including steps."""
        with self._lock:
            row = self._connect().execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                return None

            step_rows = (
                self._connect()
                .execute(
                    "SELECT * FROM steps WHERE run_id = ? ORDER BY step_index",
                    (run_id,),
                )
                .fetchall()
            )

        detail = RunDetail(
            id=row["id"],
            created_at=row["created_at"],
            mode=row["mode"],
            provider=row["provider"],
            model=row["model"],
            query=row["query"],
            total_tokens=row["total_tokens"],
            total_cost=row["total_cost"],
            elapsed_seconds=row["elapsed_seconds"],
            success=bool(row["success"]),
            error=row["error"],
            session_id=row["session_id"],
            chat_provider_id=row["chat_provider_id"],
            chat_provider_name=row["chat_provider_name"],
            steps_count=row["steps_count"],
            answer_length=row["answer_length"],
            comparison_group_id=row["comparison_group_id"],
            answer=row["answer"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            content_length=row["content_length"],
            steps=[dict(r) for r in step_rows],
        )
        return detail

    def count_runs(self) -> int:
        """Return total number of runs."""
        with self._lock:
            row = self._connect().execute("SELECT COUNT(*) AS cnt FROM runs").fetchone()
        return int(row["cnt"]) if row else 0

    # ------------------------------------------------------------------
    # Aggregation queries
    # ------------------------------------------------------------------

    def aggregate_by_mode(self) -> list[AggregateRow]:
        """Aggregate run metrics grouped by mode."""
        return self._aggregate("mode")

    def aggregate_by_provider(self) -> list[AggregateRow]:
        """Aggregate run metrics grouped by provider."""
        return self._aggregate("provider")

    def _aggregate(self, group_col: str) -> list[AggregateRow]:
        if group_col not in ("mode", "provider", "model"):
            raise ValueError(f"Invalid group column: {group_col!r}")
        with self._lock:
            rows = (
                self._connect()
                .execute(
                    f"""SELECT {group_col} AS key,
                           COUNT(*) AS run_count,
                           SUM(total_tokens) AS total_tokens,
                           SUM(total_cost) AS total_cost,
                           AVG(elapsed_seconds) AS avg_elapsed,
                           AVG(success) AS success_rate
                    FROM runs
                    GROUP BY {group_col}
                    ORDER BY run_count DESC""",  # nosec B608
                )
                .fetchall()
            )
        return [
            AggregateRow(
                key=r["key"],
                run_count=r["run_count"],
                total_tokens=r["total_tokens"] or 0,
                total_cost=r["total_cost"] or 0.0,
                avg_elapsed=r["avg_elapsed"] or 0.0,
                success_rate=r["success_rate"] or 0.0,
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # JSONL export
    # ------------------------------------------------------------------

    def export_jsonl(self, run_id: str) -> str | None:
        """Export a run as JSONL compatible with the upstream RLM visualizer.

        The output schema matches ``visualizer/src/lib/types.ts`` in
        ``alexzhang13/rlm``: one optional ``{"type": "metadata", ...}`` line
        at the top, followed by one line per iteration with ``iteration``,
        ``timestamp``, ``prompt``, ``response``, ``code_blocks``,
        ``final_answer``, and ``iteration_time`` fields.

        Our internal step model is finer-grained than upstream iterations:
        each assistant turn is an ``inspect`` or ``final`` step, and each
        code execution is a ``subcall`` step.  We group them into upstream
        iterations here — one iteration per assistant turn, with any
        trailing ``subcall`` steps collapsed into its ``code_blocks``.

        Fields we don't capture in telemetry are populated with empty
        stand-ins the visualizer accepts:
        - ``prompt`` → ``[]`` (we never stored the full message array)
        - ``result.stderr`` → ``""`` (stdout+stderr are merged in ``output``)
        - ``result.locals`` → ``{}`` (not tracked)
        - ``result.rlm_calls`` → ``[]`` (nested-RLM telemetry is separate)

        Returns ``None`` if the run is not found.
        """
        detail = self.get_run(run_id)
        if detail is None:
            return None

        # Group first so the metadata line can cite the real iteration
        # count (not the raw step count, which differs whenever
        # inspect/subcall steps collapse into one upstream iteration).
        iterations = _group_steps_into_iterations(
            detail.steps,
            run_created_at=detail.created_at,
            final_answer=detail.answer,
        )

        lines: list[str] = []

        # --- Metadata line (upstream schema: {"type": "metadata", ...}) ---
        metadata: dict[str, Any] = {
            "type": "metadata",
            "root_model": detail.model or None,
            "max_depth": None,
            # Must match the number of iteration lines that follow or
            # consumers using this for progress/validation see an
            # internally inconsistent trace.
            "max_iterations": len(iterations) or None,
            "backend": detail.provider or None,
            "backend_kwargs": None,
            "environment_type": "subprocess",
            "environment_kwargs": None,
            "other_backends": None,
            # Non-standard extras — the visualizer's parseJSONL ignores
            # unknown fields, and keeping them lets us round-trip without
            # losing telemetry context.
            "rlm_studio_query": detail.query,
            "rlm_studio_success": detail.success,
            "rlm_studio_total_tokens": detail.total_tokens,
            "rlm_studio_total_cost": detail.total_cost,
            "rlm_studio_elapsed_seconds": detail.elapsed_seconds,
            # The raw (pre-grouping) step count is still useful for
            # cross-referencing against the telemetry store, so keep it
            # as a non-standard extra.
            "rlm_studio_raw_steps_count": detail.steps_count,
        }
        lines.append(json.dumps(metadata))

        # --- Iteration lines ---
        for iteration in iterations:
            lines.append(json.dumps(iteration))

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def delete_run(self, run_id: str) -> None:
        """Delete a run and all associated steps, calls, and ratings."""
        with self._lock:
            try:
                conn = self._connect()
                conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
                conn.commit()
            except Exception:
                logger.exception("Failed to delete run %s", run_id)

    def delete_all_runs(self) -> int:
        """Delete all runs (and cascaded children) in a single statement.

        Returns the number of rows deleted.
        """
        with self._lock:
            try:
                conn = self._connect()
                cursor = conn.execute("DELETE FROM runs")
                conn.commit()
                return cursor.rowcount
            except Exception:
                logger.exception("Failed to delete all runs")
                return 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_summary(row: sqlite3.Row) -> RunSummary:
        # outcome_category was added in schema v2; read defensively so legacy
        # rows on databases that haven't migrated yet don't blow up reads.
        keys = row.keys()
        outcome_category = row["outcome_category"] if "outcome_category" in keys else None
        return RunSummary(
            id=row["id"],
            created_at=row["created_at"],
            mode=row["mode"],
            provider=row["provider"],
            model=row["model"],
            query=row["query"],
            total_tokens=row["total_tokens"],
            total_cost=row["total_cost"],
            elapsed_seconds=row["elapsed_seconds"],
            success=bool(row["success"]),
            error=row["error"],
            session_id=row["session_id"],
            chat_provider_id=row["chat_provider_id"],
            chat_provider_name=row["chat_provider_name"],
            steps_count=row["steps_count"],
            answer_length=row["answer_length"],
            comparison_group_id=row["comparison_group_id"],
            answer=row["answer"] or "",
            outcome_category=outcome_category,
        )


# ---------------------------------------------------------------------------
# Upstream-visualizer JSONL helpers
# ---------------------------------------------------------------------------


def _group_steps_into_iterations(
    steps: list[dict[str, Any]],
    *,
    run_created_at: float,
    final_answer: str,
) -> list[dict[str, Any]]:
    """Group flat RLMKit steps into upstream-visualizer iterations.

    Our step model stores one row per action (``inspect`` / ``subcall`` /
    ``final`` / ``error``).  The upstream visualizer groups execution
    rounds into iterations where each iteration is *one LLM turn plus
    the code blocks executed from that turn's response*.

    Mapping:
        - Assistant turns (``inspect`` / ``final`` / ``error``) start a
          new iteration.
        - Following ``subcall`` steps attach to the current iteration as
          ``code_blocks`` entries.
        - ``final`` iterations carry ``final_answer``; others set it to
          ``null``.

    Fields we don't have in telemetry are populated with empty
    stand-ins the visualizer accepts:
        - ``prompt`` is always ``[]`` (we never stored the message array)
        - ``result.stderr`` is always ``""``
        - ``result.locals`` is always ``{}``
        - ``result.rlm_calls`` is always ``[]``
    """
    from datetime import datetime, timezone

    iterations: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    # Running offset (in seconds) from the run's start.  Every completed
    # iteration advances this by its own ``iteration_time``, so the next
    # iteration's ``timestamp`` reflects when it actually started rather
    # than repeating ``run_created_at`` for every line.  This preserves
    # per-iteration timing/order in the visualizer.
    time_offset: float = 0.0

    def _iso_at_offset(offset: float) -> str:
        return datetime.fromtimestamp(run_created_at + offset, tz=timezone.utc).isoformat()

    def _new_iteration(index: int, step: dict[str, Any], start_offset: float) -> dict[str, Any]:
        return {
            "iteration": index,
            "timestamp": _iso_at_offset(start_offset),
            "prompt": [],  # not captured by telemetry
            "response": step.get("output") or "",
            "code_blocks": [],
            "final_answer": None,
            "iteration_time": float(step.get("duration") or 0.0),
        }

    for step in steps:
        action = step.get("action_type") or "inspect"

        if action in ("inspect", "final", "error"):
            # Assistant turn → new iteration.  Advance the time offset
            # by the previous iteration's accumulated time first so the
            # new iteration's start timestamp is monotonic.
            if current is not None:
                time_offset += float(current.get("iteration_time") or 0.0)
                iterations.append(current)
            current = _new_iteration(len(iterations), step, time_offset)
            if action == "final":
                # Prefer the run-level answer when available; falls back
                # to the step's output so a trimmed answer in the store
                # is still visible in the visualizer.
                current["final_answer"] = final_answer or (step.get("output") or "")
            elif action == "error":
                current["final_answer"] = None  # explicit: no answer

            # Inspect steps often carry ``code`` — treat that as the
            # first code_block of this iteration so the visualizer
            # shows it alongside the LLM response.
            code = step.get("code")
            if action == "inspect" and code:
                current["code_blocks"].append(
                    {
                        "code": code,
                        "result": {
                            "stdout": "",
                            "stderr": "",
                            "locals": {},
                            "execution_time": 0.0,
                            "rlm_calls": [],
                        },
                    }
                )
        elif action == "subcall":
            # Execution result → attach to current iteration.
            if current is None:
                # Orphaned subcall with no preceding assistant turn.
                # Synthesize a minimal iteration wrapper so nothing is lost.
                current = _new_iteration(len(iterations), {}, time_offset)
            code_block: dict[str, Any] = {
                "code": step.get("code") or "",
                "result": {
                    "stdout": step.get("output") or "",
                    "stderr": "",  # merged into stdout by our sandbox
                    "locals": {},
                    "execution_time": float(step.get("duration") or 0.0),
                    "rlm_calls": [],
                },
            }
            current["code_blocks"].append(code_block)
            # Fold the subcall's duration into the iteration's total.
            current["iteration_time"] = float(current.get("iteration_time") or 0.0) + float(
                step.get("duration") or 0.0
            )
        else:
            # Unknown action — skip rather than crash.
            logger.debug("Skipping step with unknown action_type=%r", action)

    if current is not None:
        iterations.append(current)

    return iterations
