"""Metrics endpoints.

Aggregates per-session metrics from two sources:

1. **Telemetry store** (primary): ``state.telemetry.list_runs(session_id=…)``
   is the source of truth for new runs going forward.  Data survives
   restarts and stays consistent with ``/api/traces`` and
   ``/api/executions``.

2. **Legacy session messages** (fallback): sessions created before Bet 2
   stored per-run metrics inline on assistant messages
   (``msg['metrics']``).  Those rows were never backfilled into
   telemetry, so we still walk ``session.messages`` for any assistant
   message whose ``execution_id`` is not already represented in the
   store and fold it into the aggregation.

The fallback is keyed by ``execution_id`` so messages that DO have a
telemetry row are never double-counted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from rlmstudio.application.sandbox_vars import EPHEMERAL_CP_PREFIX
from rlmstudio.application.services.outcome_classifier import (
    ExecutionOutcome,
    OutcomeCategory,
    classify_execution_outcome,
)
from rlmstudio.server.dependencies import AppState, get_state
from rlmstudio.server.models import (
    FailureCategory,
    FailureCategorySummary,
    FailureMetricsResponse,
    MetricsResponse,
    MetricsSummary,
    ModeSummary,
    ProviderSummary,
    TimelineEntry,
)

router = APIRouter()

# Non-SUCCESS categories — used to filter failure points.
_FAILURE_CATEGORIES = frozenset(c for c in OutcomeCategory if c != OutcomeCategory.SUCCESS)

# Enum value -> the response literal.  A mapping rather than a cast: mypy
# checks each value against FailureCategory, so a renamed or added category
# fails here instead of at the response boundary.
_FAILURE_CATEGORY_BY_VALUE: dict[str, FailureCategory] = {
    "timeout": "timeout",
    "prefill_timeout": "prefill_timeout",
    "budget_exhausted": "budget_exhausted",
    "context_overflow": "context_overflow",
    "general_error": "general_error",
}


@dataclass
class _RunPoint:
    """Normalized per-run data point used for aggregation.

    Sits between the two input sources (telemetry rows, legacy messages)
    and the output aggregators, so :func:`_build_metrics_response` never
    has to care where a row came from.
    """

    execution_id: str | None
    timestamp: datetime
    tokens: int
    cost: float
    latency: float
    mode: str
    provider: str
    chat_provider_name: str | None
    success: bool = True
    error: str | None = None
    answer: str = ""
    # Persisted outcome category (spec v1.7 §8c). Populated on rows
    # written after Phase 4; ``None`` for legacy rows (column NULL) and
    # for legacy assistant-message points that were never classified
    # at write time. Fallback to the no-trace classifier path when
    # None — see ``_aggregate_failures``.
    outcome_category: str | None = None


@router.get("/api/metrics/failures/{session_id}")
async def get_failure_metrics(
    session_id: str,
    state: AppState = Depends(get_state),  # noqa: B008
) -> FailureMetricsResponse:
    """Get categorized failure breakdown for a session."""
    session = state.sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    telemetry_runs = state.telemetry.list_runs(session_id=session_id, limit=10_000)
    telemetry_exec_ids: set[str] = {r.id for r in telemetry_runs}

    points: list[_RunPoint] = [
        _point_from_telemetry(r)
        for r in telemetry_runs
        if not (r.chat_provider_name or "").startswith(EPHEMERAL_CP_PREFIX)
    ]

    for msg in session.messages:
        if msg.get("role") != "assistant":
            continue
        msg_metrics = msg.get("metrics")
        if not isinstance(msg_metrics, dict):
            continue
        exec_id = msg.get("execution_id")
        if exec_id and exec_id in telemetry_exec_ids:
            continue
        point = _point_from_message(msg, msg_metrics)
        if point is not None:
            if (point.chat_provider_name or "").startswith(EPHEMERAL_CP_PREFIX):
                continue
            points.append(point)

    return _build_failure_response(session_id, points)


@router.get("/api/metrics/{session_id}")
async def get_metrics(
    session_id: str,
    state: AppState = Depends(get_state),  # noqa: B008
) -> MetricsResponse:
    """Get aggregated metrics for a session (telemetry + legacy messages)."""
    session = state.sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Source 1: telemetry store. A high limit (10_000) avoids accidental
    # truncation for long sessions; at ~1 KB per row this stays well under
    # memory limits.
    telemetry_runs = state.telemetry.list_runs(session_id=session_id, limit=10_000)
    telemetry_exec_ids: set[str] = {r.id for r in telemetry_runs}

    points: list[_RunPoint] = [_point_from_telemetry(r) for r in telemetry_runs]

    # Filter out ephemeral compare-matrix CPs — they skew dashboard
    # metrics (e.g., RLM token savings becomes -1600% when a 6-slot
    # matrix dumps 100K+ RLM tokens into the session).
    points = [p for p in points if not (p.chat_provider_name or "").startswith(EPHEMERAL_CP_PREFIX)]

    # Source 2: legacy assistant messages not already represented in telemetry.
    # This preserves metrics visibility for sessions persisted before Bet 2.
    for msg in session.messages:
        if msg.get("role") != "assistant":
            continue
        msg_metrics = msg.get("metrics")
        if not isinstance(msg_metrics, dict):
            continue
        exec_id = msg.get("execution_id")
        if exec_id and exec_id in telemetry_exec_ids:
            # Already counted via telemetry — avoid double-counting.
            continue
        point = _point_from_message(msg, msg_metrics)
        if point is not None:
            # Skip ephemeral compare-matrix entries from legacy path too.
            if (point.chat_provider_name or "").startswith(EPHEMERAL_CP_PREFIX):
                continue
            points.append(point)

    return _build_metrics_response(session_id, points)


def _resolve_outcome(point: _RunPoint) -> ExecutionOutcome:
    """Prefer the point's persisted outcome_category; fall back to re-derive.

    Spec alias: the prefill/decode spec §8c calls the caller of this
    helper ``_aggregate_failures``. The dashboard's two failure-
    aggregation loops are in ``_build_metrics_response`` and
    ``_build_failure_response`` (grep for ``_resolve_outcome`` to find
    them). Both prefer the persisted category over re-classification.

    Phase 4 classifies at write time, so most rows carry a persisted
    category by the time they reach the dashboard. Legacy rows (pre-
    Phase-4, column NULL) and legacy assistant-message points never
    had one; for those we re-derive with no trace (spec §8c). Legacy
    TIMEOUT rows are never retroactively promoted to PREFILL_TIMEOUT
    — we don't have per-step timing for them.
    """
    cat_str = point.outcome_category
    if cat_str is None:
        return classify_execution_outcome(point.success, point.error, point.answer)
    try:
        category = OutcomeCategory(cat_str)
    except ValueError:
        # Unknown persisted value — treat as legacy and re-derive.
        return classify_execution_outcome(point.success, point.error, point.answer)
    return ExecutionOutcome(category, is_usable=(category == OutcomeCategory.SUCCESS))


def _point_from_telemetry(run: Any) -> _RunPoint:
    """Convert a telemetry :class:`RunSummary` into a :class:`_RunPoint`."""
    return _RunPoint(
        execution_id=run.id,
        timestamp=datetime.fromtimestamp(run.created_at, tz=timezone.utc),
        tokens=run.total_tokens,
        cost=run.total_cost,
        latency=run.elapsed_seconds,
        mode=run.mode or "unknown",
        provider=run.provider or "unknown",
        chat_provider_name=run.chat_provider_name,
        success=run.success,
        error=run.error,
        answer=run.answer,
        outcome_category=getattr(run, "outcome_category", None),
    )


def _point_from_message(
    msg: dict[str, Any],
    msg_metrics: dict[str, Any],
) -> _RunPoint | None:
    """Convert a legacy assistant message into a :class:`_RunPoint`, or None."""
    ts_raw = msg.get("timestamp")
    if isinstance(ts_raw, str):
        try:
            ts = datetime.fromisoformat(ts_raw)
        except ValueError:
            ts = datetime.now(timezone.utc)
    elif isinstance(ts_raw, datetime):
        ts = ts_raw
    else:
        ts = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    return _RunPoint(
        execution_id=msg.get("execution_id"),
        timestamp=ts,
        tokens=int(msg_metrics.get("total_tokens", 0)),
        cost=float(msg_metrics.get("cost_usd", 0.0)),
        latency=float(msg_metrics.get("elapsed_seconds", 0.0)),
        mode=msg.get("mode_used", "unknown") or "unknown",
        provider=msg.get("provider", "unknown") or "unknown",
        chat_provider_name=msg.get("chat_provider_name"),
        success=bool(msg_metrics.get("success", True)),
        error=msg_metrics.get("error"),
        answer=msg.get("content", ""),
        # Legacy messages pre-date the outcome_category column; the
        # dashboard re-derives via the no-trace classifier fallback.
        outcome_category=None,
    )


def _build_metrics_response(
    session_id: str,
    points: list[_RunPoint],
) -> MetricsResponse:
    """Aggregate a list of :class:`_RunPoint` rows into a ``MetricsResponse``."""
    total_tokens = 0
    total_cost = 0.0
    total_latency = 0.0
    query_count = 0
    usable_count = 0

    by_mode: dict[str, _GroupAcc] = {}
    by_provider: dict[str, _GroupAcc] = {}
    by_chat_provider: dict[str, _GroupAcc] = {}
    timeline: list[TimelineEntry] = []

    for point in points:
        query_count += 1

        # Classify to determine if this run should contribute to averages.
        # Non-usable runs (failures, degraded warnings) are counted but
        # excluded from cost/latency/token aggregations to avoid skewing.
        # Prefer the persisted outcome_category (Phase 4 write-time
        # classification); fall back to re-deriving from scalars for
        # legacy rows with NULL category (spec v1.7 §8c).
        outcome = _resolve_outcome(point)
        if outcome.is_usable:
            usable_count += 1
            total_tokens += point.tokens
            total_cost += point.cost
            total_latency += point.latency

            _acc(by_mode, point.mode, point.tokens, point.cost, point.latency)
            _acc(by_provider, point.provider, point.tokens, point.cost, point.latency)
            if point.chat_provider_name:
                _acc(
                    by_chat_provider,
                    point.chat_provider_name,
                    point.tokens,
                    point.cost,
                    point.latency,
                )

        timeline.append(
            TimelineEntry(
                timestamp=point.timestamp,
                tokens=point.tokens,
                cost_usd=point.cost,
                latency_seconds=point.latency,
                mode=point.mode,
                provider=point.provider,
                chat_provider_name=point.chat_provider_name,
                execution_id=point.execution_id,
            )
        )

    # Sort timeline chronologically (oldest → newest) so UI charts draw
    # left-to-right regardless of which source the row came from.
    timeline.sort(key=lambda e: e.timestamp)

    avg_latency = total_latency / usable_count if usable_count else 0.0

    mode_summaries = {key: _to_mode_summary(acc) for key, acc in by_mode.items()}
    provider_summaries = {key: _to_provider_summary(acc) for key, acc in by_provider.items()}
    chat_provider_summaries = {
        key: _to_provider_summary(acc) for key, acc in by_chat_provider.items()
    }

    # Token savings: compare RLM vs Direct total tokens (lower is better).
    rlm_tokens = mode_summaries.get("rlm", ModeSummary()).total_tokens
    direct_tokens = mode_summaries.get("direct", ModeSummary()).total_tokens
    if direct_tokens > 0 and rlm_tokens > 0:
        savings: float | None = round((1 - rlm_tokens / direct_tokens) * 100, 1)
    else:
        savings = None

    return MetricsResponse(
        session_id=session_id,
        summary=MetricsSummary(
            total_queries=query_count,
            total_tokens=total_tokens,
            total_cost_usd=round(total_cost, 4),
            avg_latency_seconds=round(avg_latency, 2),
            avg_token_savings_percent=savings,
        ),
        by_mode=mode_summaries,
        by_provider=provider_summaries,
        by_chat_provider=chat_provider_summaries,
        timeline=timeline,
    )


def _build_failure_response(
    session_id: str,
    points: list[_RunPoint],
) -> FailureMetricsResponse:
    """Aggregate failure data from a list of :class:`_RunPoint` rows."""
    total_runs = len(points)

    # category -> count
    by_category: dict[str, int] = {}
    # provider -> category -> count
    by_provider: dict[str, dict[str, int]] = {}
    # mode -> category -> count
    by_mode: dict[str, dict[str, int]] = {}

    for point in points:
        outcome = _resolve_outcome(point)
        if outcome.category not in _FAILURE_CATEGORIES:
            continue

        cat = outcome.category.value
        by_category[cat] = by_category.get(cat, 0) + 1

        prov = point.chat_provider_name or point.provider
        prov_cats = by_provider.setdefault(prov, {})
        prov_cats[cat] = prov_cats.get(cat, 0) + 1

        mode_cats = by_mode.setdefault(point.mode, {})
        mode_cats[cat] = mode_cats.get(cat, 0) + 1

    total_failures = sum(by_category.values())
    failure_rate = total_failures / total_runs if total_runs else 0.0

    def _to_summaries(bucket: dict[str, int]) -> list[FailureCategorySummary]:
        return [
            FailureCategorySummary(category=_FAILURE_CATEGORY_BY_VALUE[k], count=v)
            for k, v in sorted(bucket.items())
        ]

    return FailureMetricsResponse(
        session_id=session_id,
        total_runs=total_runs,
        total_failures=total_failures,
        failure_rate=round(failure_rate, 4),
        by_category=_to_summaries(by_category),
        by_provider={k: _to_summaries(v) for k, v in sorted(by_provider.items())},
        by_mode={k: _to_summaries(v) for k, v in sorted(by_mode.items())},
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


class _GroupAcc:
    """Mutable accumulator for one aggregation bucket."""

    __slots__ = ("queries", "total_tokens", "total_cost_usd", "latencies")

    def __init__(self) -> None:
        self.queries: int = 0
        self.total_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.latencies: list[float] = []


def _acc(
    bucket: dict[str, _GroupAcc],
    key: str,
    tokens: int,
    cost: float,
    latency: float,
) -> None:
    """Add one run's numbers into the right bucket, creating it on demand."""
    acc = bucket.get(key)
    if acc is None:
        acc = _GroupAcc()
        bucket[key] = acc
    acc.queries += 1
    acc.total_tokens += tokens
    acc.total_cost_usd += cost
    acc.latencies.append(latency)


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _to_mode_summary(acc: _GroupAcc) -> ModeSummary:
    return ModeSummary(
        queries=acc.queries,
        total_tokens=acc.total_tokens,
        total_cost_usd=acc.total_cost_usd,
        avg_latency_seconds=round(_avg(acc.latencies), 2),
    )


def _to_provider_summary(acc: _GroupAcc) -> ProviderSummary:
    return ProviderSummary(
        queries=acc.queries,
        total_tokens=acc.total_tokens,
        total_cost_usd=acc.total_cost_usd,
        avg_latency_seconds=round(_avg(acc.latencies), 2),
    )
