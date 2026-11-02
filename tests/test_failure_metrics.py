"""Tests for the failure metrics endpoint and metrics filtering."""

from __future__ import annotations

import time

from starlette.testclient import TestClient

from rlmstudio.server.app import create_app
from rlmstudio.server.dependencies import SessionRecord, get_state, reset_state
from rlmstudio.server.routes.metrics import (
    _build_failure_response,
    _build_metrics_response,
    _point_from_telemetry,
    _RunPoint,
)


def _now() -> float:
    return time.time()


def test_every_outcome_category_maps_to_a_failure_literal() -> None:
    """The dict is not exhaustiveness-checked; this pins its coverage.

    Without it, adding an ``OutcomeCategory`` member would leave the
    response mapping short an entry, and the first run of that category
    would raise ``KeyError`` inside the metrics endpoint.
    """
    from rlmstudio.application.services.outcome_classifier import OutcomeCategory
    from rlmstudio.server.routes.metrics import _FAILURE_CATEGORY_BY_VALUE

    expected = {c.value for c in OutcomeCategory if c is not OutcomeCategory.SUCCESS}
    assert set(_FAILURE_CATEGORY_BY_VALUE) == expected


def _ts():  # noqa: ANN202
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Unit tests for _build_failure_response
# ---------------------------------------------------------------------------


class TestBuildFailureResponse:
    """Unit tests for the failure aggregation builder."""

    def test_no_points(self) -> None:
        resp = _build_failure_response("s1", [])
        assert resp.total_runs == 0
        assert resp.total_failures == 0
        assert resp.failure_rate == 0.0
        assert resp.by_category == []
        assert resp.by_provider == {}
        assert resp.by_mode == {}

    def test_all_success(self) -> None:
        points = [
            _RunPoint(
                execution_id="e1",
                timestamp=_ts(),
                tokens=100,
                cost=0.01,
                latency=1.0,
                mode="direct",
                provider="openai",
                chat_provider_name="GPT-4o",
                success=True,
                error=None,
                answer="Valid answer",
            ),
        ]
        resp = _build_failure_response("s1", points)
        assert resp.total_runs == 1
        assert resp.total_failures == 0
        assert resp.failure_rate == 0.0

    def test_mixed_failures(self) -> None:
        points = [
            _RunPoint(
                execution_id="e1",
                timestamp=_ts(),
                tokens=100,
                cost=0.01,
                latency=1.0,
                mode="direct",
                provider="openai",
                chat_provider_name="GPT-4o",
                success=True,
                error=None,
                answer="Valid answer",
            ),
            _RunPoint(
                execution_id="e2",
                timestamp=_ts(),
                tokens=0,
                cost=0.0,
                latency=30.0,
                mode="rlm",
                provider="anthropic",
                chat_provider_name="Claude",
                success=False,
                error="Request timeout",
                answer="",
            ),
            _RunPoint(
                execution_id="e3",
                timestamp=_ts(),
                tokens=50,
                cost=0.005,
                latency=5.0,
                mode="direct",
                provider="openai",
                chat_provider_name="GPT-4o",
                success=False,
                error="context window exceeded",
                answer="",
            ),
        ]
        resp = _build_failure_response("s1", points)
        assert resp.total_runs == 3
        assert resp.total_failures == 2
        assert resp.failure_rate == round(2 / 3, 4)

        # by_category
        cat_map = {c.category: c.count for c in resp.by_category}
        assert cat_map["timeout"] == 1
        assert cat_map["context_overflow"] == 1

        # by_provider
        assert "Claude" in resp.by_provider
        assert "GPT-4o" in resp.by_provider
        claude_cats = {c.category: c.count for c in resp.by_provider["Claude"]}
        assert claude_cats["timeout"] == 1

        # by_mode
        assert "rlm" in resp.by_mode
        assert "direct" in resp.by_mode

    def test_degraded_counted_as_failure(self) -> None:
        points = [
            _RunPoint(
                execution_id="e1",
                timestamp=_ts(),
                tokens=500,
                cost=0.05,
                latency=10.0,
                mode="rlm",
                provider="openai",
                chat_provider_name="GPT-4o",
                success=True,
                error=None,
                answer="⚠️ **Step budget exhausted** (16/16 steps used).\n\nPartial...",
            ),
        ]
        resp = _build_failure_response("s1", points)
        assert resp.total_failures == 1
        cat_map = {c.category: c.count for c in resp.by_category}
        assert cat_map["budget_exhausted"] == 1


# ---------------------------------------------------------------------------
# Unit tests for metrics filtering
# ---------------------------------------------------------------------------


class TestMetricsFiltering:
    """Verify that non-usable runs are excluded from averages."""

    def test_failed_runs_excluded_from_averages(self) -> None:
        points = [
            _RunPoint(
                execution_id="e1",
                timestamp=_ts(),
                tokens=100,
                cost=0.01,
                latency=2.0,
                mode="direct",
                provider="openai",
                chat_provider_name="GPT-4o",
                success=True,
                error=None,
                answer="Good answer",
            ),
            _RunPoint(
                execution_id="e2",
                timestamp=_ts(),
                tokens=0,
                cost=0.0,
                latency=120.0,
                mode="direct",
                provider="openai",
                chat_provider_name="GPT-4o",
                success=False,
                error="Request timeout",
                answer="",
            ),
        ]
        resp = _build_metrics_response("s1", points)
        # Total queries count includes all runs
        assert resp.summary.total_queries == 2
        # But averages only use the usable run
        assert resp.summary.total_tokens == 100
        assert resp.summary.avg_latency_seconds == 2.0
        assert resp.summary.total_cost_usd == 0.01

    def test_degraded_runs_excluded_from_averages(self) -> None:
        points = [
            _RunPoint(
                execution_id="e1",
                timestamp=_ts(),
                tokens=200,
                cost=0.02,
                latency=3.0,
                mode="rlm",
                provider="openai",
                chat_provider_name="GPT-4o",
                success=True,
                error=None,
                answer="Good answer",
            ),
            _RunPoint(
                execution_id="e2",
                timestamp=_ts(),
                tokens=500,
                cost=0.05,
                latency=60.0,
                mode="rlm",
                provider="openai",
                chat_provider_name="GPT-4o",
                success=True,
                error=None,
                answer="⚠️ **Execution timed out** after 5 steps.\n\nPartial...",
            ),
        ]
        resp = _build_metrics_response("s1", points)
        assert resp.summary.total_queries == 2
        # Only the usable run contributes to totals
        assert resp.summary.total_tokens == 200
        assert resp.summary.total_cost_usd == 0.02

    def test_all_failures_gives_zero_averages(self) -> None:
        points = [
            _RunPoint(
                execution_id="e1",
                timestamp=_ts(),
                tokens=0,
                cost=0.0,
                latency=30.0,
                mode="direct",
                provider="openai",
                chat_provider_name="GPT-4o",
                success=False,
                error="timeout",
                answer="",
            ),
        ]
        resp = _build_metrics_response("s1", points)
        assert resp.summary.total_queries == 1
        assert resp.summary.total_tokens == 0
        assert resp.summary.avg_latency_seconds == 0.0

    def test_timeline_includes_all_runs(self) -> None:
        points = [
            _RunPoint(
                execution_id="e1",
                timestamp=_ts(),
                tokens=100,
                cost=0.01,
                latency=2.0,
                mode="direct",
                provider="openai",
                chat_provider_name="GPT-4o",
                success=True,
                error=None,
                answer="Good",
            ),
            _RunPoint(
                execution_id="e2",
                timestamp=_ts(),
                tokens=0,
                cost=0.0,
                latency=30.0,
                mode="direct",
                provider="openai",
                chat_provider_name="GPT-4o",
                success=False,
                error="timeout",
                answer="",
            ),
        ]
        resp = _build_metrics_response("s1", points)
        # Timeline should include both (success and failure)
        assert len(resp.timeline) == 2


# ---------------------------------------------------------------------------
# Telemetry point builder
# ---------------------------------------------------------------------------


class TestPointFromTelemetry:
    """Verify _point_from_telemetry populates outcome fields."""

    def test_includes_success_error_answer(self) -> None:
        class _FakeRun:
            id = "r1"
            created_at = _now()
            total_tokens = 100
            total_cost = 0.01
            elapsed_seconds = 2.0
            mode = "direct"
            provider = "openai"
            chat_provider_name = "GPT-4o"
            success = False
            error = "timeout"
            answer = ""

        point = _point_from_telemetry(_FakeRun())
        assert point.success is False
        assert point.error == "timeout"
        assert point.answer == ""


# ---------------------------------------------------------------------------
# Integration test: failure endpoint via TestClient
# ---------------------------------------------------------------------------


class TestFailureEndpointIntegration:
    """Test GET /api/metrics/failures/{session_id} via TestClient."""

    def test_failure_endpoint_404_for_missing_session(self) -> None:
        reset_state()
        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/metrics/failures/no-such-session")
        assert resp.status_code == 404

    def test_failure_endpoint_returns_empty_for_new_session(self) -> None:
        reset_state()
        state = get_state()
        state.sessions["s1"] = SessionRecord(
            id="s1", name="Test", created_at=_ts(), updated_at=_ts()
        )
        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/metrics/failures/s1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "s1"
        assert data["total_runs"] == 0
        assert data["total_failures"] == 0

    def test_failure_endpoint_with_telemetry(self) -> None:
        reset_state()
        state = get_state()
        state.sessions["s1"] = SessionRecord(
            id="s1", name="Test", created_at=_ts(), updated_at=_ts()
        )
        # Record a failed run in telemetry
        state.telemetry.record_run(
            run_id="r1",
            created_at=_now(),
            mode="direct",
            provider="openai",
            chat_provider_name="GPT-4o",
            session_id="s1",
            success=False,
            error="Request timeout",
            answer="",
        )
        # Record a successful run
        state.telemetry.record_run(
            run_id="r2",
            created_at=_now(),
            mode="direct",
            provider="openai",
            chat_provider_name="GPT-4o",
            session_id="s1",
            success=True,
            answer="Good answer",
        )

        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/metrics/failures/s1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_runs"] == 2
        assert data["total_failures"] == 1
        assert data["failure_rate"] == 0.5
        cats = {c["category"]: c["count"] for c in data["by_category"]}
        assert cats["timeout"] == 1
