"""SQLite-backed telemetry store for execution history and analytics.

Provides persistent, queryable storage for runs, steps, provider calls,
and ratings.  All writes are synchronous (<1 ms for single rows) and
wrapped in try/finally so telemetry failures never block execution.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
import uuid
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rlmstudio.application.refinement_constants import (
    REFINEMENT_INTERRUPTED_ON_STARTUP_ERROR,
    REFINEMENT_STATUS_EVALUATING,
    REFINEMENT_STATUS_INTERRUPTED,
)
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

_CURRENT_SCHEMA_VERSION = 5

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
    4: [
        # Prompt refinement (#78): run attribution and document identity.
        # Legacy rows stay NULL and are not selectable as refinement sources.
        "ALTER TABLE runs ADD COLUMN profile_id TEXT",
        "ALTER TABLE runs ADD COLUMN content_sha256 TEXT",
        "CREATE INDEX IF NOT EXISTS idx_runs_profile ON runs(profile_id, created_at)",
        # One row per refinement; the JSON columns hold the proposal,
        # evaluation and decision records.  No FK to runs: the evidence
        # must survive run deletion.
        """CREATE TABLE IF NOT EXISTS refinements (
            id              TEXT PRIMARY KEY,
            profile_id      TEXT NOT NULL,
            status          TEXT NOT NULL,
            created_at      REAL NOT NULL,
            updated_at      REAL NOT NULL,
            proposal_json   TEXT NOT NULL,
            evaluation_json TEXT,
            decision_json   TEXT,
            error           TEXT
        )""",
        "CREATE INDEX IF NOT EXISTS idx_refinements_profile ON refinements(profile_id, created_at)",
    ],
    5: [
        # Whether a run's stored cost / token totals are measurements (1) or
        # zero by ignorance (0) -- an official-engine timeout reports neither.
        # NULL on legacy rows reads as known, which is what they claimed.
        "ALTER TABLE runs ADD COLUMN cost_known INTEGER",
        "ALTER TABLE runs ADD COLUMN tokens_known INTEGER",
    ],
}


def content_sha256(content: str) -> str:
    """Return the SHA-256 hex digest of *content* encoded as UTF-8.

    Single definition of the document identity stored in
    ``runs.content_sha256``; the held-out check in prompt refinement
    compares benchmark case content against it, so every writer must
    hash the same way.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class RefinementStoreError(Exception):
    """A refinement read or write failed in the database.

    Unlike the run/step writers, refinement writers never swallow
    failures: accept and rollback must know whether the status write
    landed (spec FR-11a).
    """


class RefinementConflictError(RefinementStoreError):
    """The refinement is missing or its status no longer matches the expected one."""


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Apply every migration whose target version exceeds PRAGMA user_version.

    Uses SQLite's built-in ``user_version`` pragma as the schema-version
    store — no extra table required. Idempotent: reopening an
    already-migrated DB is a no-op because each migration block bumps
    the pragma only after its statements commit successfully.

    Fresh installs run the v1 `_SCHEMA_SQL` (pragma starts at 0) and then
    the v2 migration adds the new columns. Upgrades from pre-Phase-2 DBs
    also start at 0 and follow the same path.

    Each version block runs in one explicit transaction together with its
    ``user_version`` bump.  Python's ``sqlite3`` does not open an implicit
    transaction before DDL, so without the explicit ``BEGIN`` a failure
    after the first ``ALTER TABLE`` would leave the columns added at the
    old version and every later startup would fail on a duplicate column.
    SQLite DDL and ``PRAGMA user_version`` are both transactional, so a
    failed block rolls back completely and the next open retries it.
    """
    current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    for target_version in sorted(_MIGRATIONS):
        if target_version <= current:
            continue
        conn.execute("BEGIN")
        try:
            for stmt in _MIGRATIONS[target_version]:
                conn.execute(stmt)
            conn.execute(f"PRAGMA user_version = {target_version}")
            conn.commit()
        except BaseException:
            conn.rollback()
            raise


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
    profile_id: str | None = None
    content_sha256: str | None = None
    # False when total_cost / the token totals are 0 because none was
    # reported (schema v5); legacy rows read as known.
    cost_known: bool = True
    tokens_known: bool = True


@dataclass
class RefinementRecord:
    """One prompt refinement row (schema v4)."""

    id: str
    profile_id: str
    status: str
    created_at: float
    updated_at: float
    proposal: dict[str, Any]
    evaluation: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    error: str | None = None


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
            try:
                conn.executescript(_SCHEMA_SQL)
                conn.commit()
                _run_migrations(conn)
            except BaseException:
                # The constructor is about to raise; do not leave the
                # connection open behind an unreachable store.
                self.close()
                raise

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
        profile_id: str | None = None,
        content_sha256: str | None = None,
        cost_known: bool = True,
        tokens_known: bool = True,
    ) -> str:
        """Record a completed run. Returns the run ID.

        ``outcome_category`` is an already-classified string from the
        caller (values are expected to be valid
        :class:`OutcomeCategory` values, but the store is a dumb sink
        and does not validate). Passing ``None`` is permitted; legacy
        rows surface ``None`` on read and re-derive via the no-trace
        classifier path on the dashboard.

        ``profile_id`` is the profile of the Chat Provider that ran, and
        ``content_sha256`` is :func:`content_sha256` of the document the run
        saw.  Both are schema v4 and drive prompt-refinement source selection.

        ``cost_known`` / ``tokens_known`` are ``False`` when the run reported
        no cost / no token count, so the stored 0 is not a measurement
        (schema v5).
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
                        outcome_category, profile_id, content_sha256,
                        cost_known, tokens_known)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                        profile_id,
                        content_sha256,
                        int(cost_known),
                        int(tokens_known),
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
        profile_id: str | None = None,
        require_content_hash: bool = False,
    ) -> list[RunSummary]:
        """List runs ordered by created_at descending.

        ``require_content_hash=True`` drops runs without ``content_sha256``
        (every run recorded before schema v4), which prompt refinement
        cannot use as a verified source.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if profile_id is not None:
            clauses.append("profile_id = ?")
            params.append(profile_id)
        if require_content_hash:
            clauses.append("content_sha256 IS NOT NULL")
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
            outcome_category=row["outcome_category"],
            profile_id=row["profile_id"],
            content_sha256=row["content_sha256"],
            cost_known=_known(row["cost_known"]),
            tokens_known=_known(row["tokens_known"]),
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
    # Prompt refinements (#78)
    #
    # These writers raise RefinementStoreError instead of logging and
    # continuing: accept and rollback must know whether a status write
    # landed after the profile was saved (spec FR-11a).
    # ------------------------------------------------------------------

    def create_refinement(
        self,
        *,
        profile_id: str,
        status: str,
        proposal: dict[str, Any],
        refinement_id: str | None = None,
        created_at: float | None = None,
    ) -> RefinementRecord:
        """Insert a refinement row and return it.

        Raises:
            RefinementStoreError: The insert failed.
        """
        rid = refinement_id or uuid.uuid4().hex
        now = time.time() if created_at is None else created_at
        proposal_json = json.dumps(proposal, sort_keys=True)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO refinements
                       (id, profile_id, status, created_at, updated_at, proposal_json)
                       VALUES (?,?,?,?,?,?)""",
                    (rid, profile_id, status, now, now, proposal_json),
                )
                conn.commit()
            except sqlite3.Error as exc:
                conn.rollback()
                raise RefinementStoreError(f"Failed to create refinement {rid}: {exc}") from exc
        return RefinementRecord(
            id=rid,
            profile_id=profile_id,
            status=status,
            created_at=now,
            updated_at=now,
            proposal=proposal,
        )

    def update_refinement(
        self,
        refinement_id: str,
        *,
        expected_status: str | Collection[str],
        status: str,
        evaluation: dict[str, Any] | None = None,
        decision: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> RefinementRecord:
        """Move a refinement to *status* if its current status is expected.

        The update is a compare-and-set on ``status``.  ``evaluation``,
        ``decision`` and ``error`` are written only when not ``None``.

        Raises:
            RefinementConflictError: The row is missing or its status is not
                one of *expected_status*.
            RefinementStoreError: The database write failed.
        """
        expected = [expected_status] if isinstance(expected_status, str) else list(expected_status)
        if not expected:
            raise ValueError("expected_status must name at least one status")
        assignments = ["status = ?", "updated_at = ?"]
        params: list[Any] = [status, time.time()]
        if evaluation is not None:
            assignments.append("evaluation_json = ?")
            params.append(json.dumps(evaluation, sort_keys=True))
        if decision is not None:
            assignments.append("decision_json = ?")
            params.append(json.dumps(decision, sort_keys=True))
        if error is not None:
            assignments.append("error = ?")
            params.append(error)
        placeholders = ",".join("?" for _ in expected)
        params.append(refinement_id)
        params.extend(expected)
        # Column names and placeholders are literals built above; every
        # value is bound as a parameter.
        sql = (
            f"UPDATE refinements SET {', '.join(assignments)}"  # nosec B608
            f" WHERE id = ? AND status IN ({placeholders})"
        )
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(sql, tuple(params))
                # Read inside the same transaction so the returned record
                # is exactly what commits; a failed read rolls back and is
                # never reported as a landed write.
                row = conn.execute(
                    "SELECT * FROM refinements WHERE id = ?", (refinement_id,)
                ).fetchone()
                if cursor.rowcount != 1:
                    conn.rollback()
                    if row is None:
                        raise RefinementConflictError(f"Refinement {refinement_id} not found")
                    raise RefinementConflictError(
                        f"Refinement {refinement_id} is {row['status']!r},"
                        f" expected one of {expected!r}"
                    )
                conn.commit()
            except sqlite3.Error as exc:
                conn.rollback()
                raise RefinementStoreError(
                    f"Failed to update refinement {refinement_id}: {exc}"
                ) from exc
        return self._row_to_refinement(row)

    def get_refinement(self, refinement_id: str) -> RefinementRecord | None:
        """Return one refinement, or ``None`` when it does not exist."""
        with self._lock:
            row = (
                self._connect()
                .execute("SELECT * FROM refinements WHERE id = ?", (refinement_id,))
                .fetchone()
            )
        return self._row_to_refinement(row) if row is not None else None

    def list_refinements(self, profile_id: str) -> list[RefinementRecord]:
        """Return a profile's refinements, newest first."""
        with self._lock:
            rows = (
                self._connect()
                .execute(
                    "SELECT * FROM refinements WHERE profile_id = ? ORDER BY created_at DESC",
                    (profile_id,),
                )
                .fetchall()
            )
        return [self._row_to_refinement(r) for r in rows]

    def mark_evaluating_as_interrupted(self) -> int:
        """Mark every ``evaluating`` refinement ``interrupted``; return the count.

        Called once at server start: an evaluation runs on an in-process
        thread, so a row still ``evaluating`` at startup has no worker.

        Raises:
            RefinementStoreError: The database write failed.
        """
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    "UPDATE refinements SET status = ?, updated_at = ?, error = ? WHERE status = ?",
                    (
                        REFINEMENT_STATUS_INTERRUPTED,
                        time.time(),
                        REFINEMENT_INTERRUPTED_ON_STARTUP_ERROR,
                        REFINEMENT_STATUS_EVALUATING,
                    ),
                )
                count = int(cursor.rowcount)
                conn.commit()
            except sqlite3.Error as exc:
                conn.rollback()
                raise RefinementStoreError(
                    f"Failed to mark interrupted refinements: {exc}"
                ) from exc
        return count

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
    def _row_to_refinement(row: sqlite3.Row) -> RefinementRecord:
        evaluation_json = row["evaluation_json"]
        decision_json = row["decision_json"]
        return RefinementRecord(
            id=row["id"],
            profile_id=row["profile_id"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            proposal=json.loads(row["proposal_json"]),
            evaluation=json.loads(evaluation_json) if evaluation_json else None,
            decision=json.loads(decision_json) if decision_json else None,
            error=row["error"],
        )

    @staticmethod
    def _row_to_summary(row: sqlite3.Row) -> RunSummary:
        # outcome_category was added in schema v2; read defensively so legacy
        # rows on databases that haven't migrated yet don't blow up reads.
        keys = row.keys()
        outcome_category = row["outcome_category"] if "outcome_category" in keys else None
        profile_id = row["profile_id"] if "profile_id" in keys else None
        run_content_sha256 = row["content_sha256"] if "content_sha256" in keys else None
        cost_known = _known(row["cost_known"] if "cost_known" in keys else None)
        tokens_known = _known(row["tokens_known"] if "tokens_known" in keys else None)
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
            profile_id=profile_id,
            content_sha256=run_content_sha256,
            cost_known=cost_known,
            tokens_known=tokens_known,
        )


def _known(stored: int | None) -> bool:
    """Read a v5 ``*_known`` column: NULL (a legacy row) counts as known."""
    return stored is None or bool(stored)


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
