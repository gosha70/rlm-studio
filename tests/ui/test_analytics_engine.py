# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0
"""Tests for AnalyticsEngine session-wide metrics aggregation."""

import pytest

from rlmstudio.ui.services.analytics_engine import AnalyticsEngine
from rlmstudio.ui.services.models import ExecutionMetrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_execution_metrics(
    total_tokens=500,
    input_tokens=300,
    output_tokens=200,
    cost_usd=0.015,
    execution_time_seconds=2.0,
    steps_taken=0,
    memory_used_mb=45.0,
    memory_peak_mb=60.0,
    success=True,
    execution_type="direct",
) -> ExecutionMetrics:
    return ExecutionMetrics(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        cost_breakdown={"input": cost_usd * 0.4, "output": cost_usd * 0.6},
        execution_time_seconds=execution_time_seconds,
        steps_taken=steps_taken,
        memory_used_mb=memory_used_mb,
        memory_peak_mb=memory_peak_mb,
        success=success,
        execution_type=execution_type,
    )


def _make_assistant_msg(
    mode="compare",
    rlm_metrics=None,
    direct_metrics=None,
    rag_metrics=None,
    content="Response",
):
    return {
        "role": "assistant",
        "content": content,
        "mode": mode,
        "rlm_metrics": rlm_metrics,
        "direct_metrics": direct_metrics,
        "rag_metrics": rag_metrics,
        "rlm_response": None,
        "direct_response": None,
        "rag_response": None,
        "comparison_metrics": None,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmptyMessages:
    def test_empty_list_returns_zeroes(self):
        analytics = AnalyticsEngine([]).compute()
        assert analytics.total_tokens == 0
        assert analytics.total_cost_usd == 0.0
        assert analytics.total_requests == 0
        assert analytics.avg_response_time_seconds == 0.0
        assert analytics.peak_memory_mb == 0.0
        assert analytics.efficiency_score == 0.0
        assert analytics.token_trends == []
        assert analytics.query_details == []
        assert analytics.outlier_indices == []

    def test_only_user_messages_returns_zeroes(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "World"},
        ]
        analytics = AnalyticsEngine(messages).compute()
        assert analytics.total_requests == 0
        assert analytics.total_tokens == 0


class TestSingleCompareMessage:
    def test_both_modes_aggregated(self):
        rlm = _make_execution_metrics(
            total_tokens=850,
            cost_usd=0.050,
            execution_time_seconds=3.0,
            steps_taken=3,
            memory_peak_mb=80.0,
            execution_type="rlm",
        )
        direct = _make_execution_metrics(
            total_tokens=270,
            cost_usd=0.015,
            execution_time_seconds=1.0,
            memory_peak_mb=50.0,
        )
        msg = _make_assistant_msg(mode="compare", rlm_metrics=rlm, direct_metrics=direct)

        analytics = AnalyticsEngine([msg]).compute()

        assert analytics.total_tokens == 850 + 270
        assert analytics.total_cost_usd == pytest.approx(0.050 + 0.015)
        assert analytics.total_requests == 1
        assert analytics.rlm_count == 1
        assert analytics.direct_count == 1
        assert analytics.rlm_total_tokens == 850
        assert analytics.direct_total_tokens == 270
        assert analytics.peak_memory_mb == 80.0
        assert analytics.avg_response_time_seconds == pytest.approx(4.0)

    def test_rlm_savings_when_direct_more_expensive(self):
        rlm = _make_execution_metrics(total_tokens=200, cost_usd=0.01, execution_type="rlm")
        direct = _make_execution_metrics(total_tokens=800, cost_usd=0.05)
        msg = _make_assistant_msg(rlm_metrics=rlm, direct_metrics=direct)

        analytics = AnalyticsEngine([msg]).compute()

        assert analytics.rlm_savings_tokens == 600
        assert analytics.rlm_savings_cost == pytest.approx(0.04)

    def test_no_savings_when_rlm_more_expensive(self):
        rlm = _make_execution_metrics(total_tokens=800, cost_usd=0.05, execution_type="rlm")
        direct = _make_execution_metrics(total_tokens=200, cost_usd=0.01)
        msg = _make_assistant_msg(rlm_metrics=rlm, direct_metrics=direct)

        analytics = AnalyticsEngine([msg]).compute()

        assert analytics.rlm_savings_tokens == 0
        assert analytics.rlm_savings_cost == 0.0


class TestSingleModeMessages:
    def test_rlm_only(self):
        rlm = _make_execution_metrics(
            total_tokens=500,
            cost_usd=0.03,
            execution_type="rlm",
            steps_taken=2,
            memory_peak_mb=70.0,
        )
        msg = _make_assistant_msg(mode="rlm_only", rlm_metrics=rlm)

        analytics = AnalyticsEngine([msg]).compute()

        assert analytics.total_tokens == 500
        assert analytics.rlm_count == 1
        assert analytics.direct_count == 0
        assert analytics.rag_count == 0
        assert analytics.rlm_total_tokens == 500

    def test_direct_only(self):
        direct = _make_execution_metrics(total_tokens=300, cost_usd=0.01)
        msg = _make_assistant_msg(mode="direct_only", direct_metrics=direct)

        analytics = AnalyticsEngine([msg]).compute()

        assert analytics.total_tokens == 300
        assert analytics.direct_count == 1
        assert analytics.rlm_count == 0

    def test_rag_only(self):
        rag = _make_execution_metrics(
            total_tokens=400,
            cost_usd=0.02,
            execution_type="rag",
            steps_taken=1,
            memory_peak_mb=55.0,
        )
        msg = _make_assistant_msg(mode="rag_only", rag_metrics=rag)

        analytics = AnalyticsEngine([msg]).compute()

        assert analytics.rag_count == 1
        assert analytics.rag_total_tokens == 400


class TestMixedModes:
    def test_three_messages_mixed(self):
        msgs = [
            _make_assistant_msg(
                mode="rlm_only",
                rlm_metrics=_make_execution_metrics(
                    total_tokens=500,
                    cost_usd=0.03,
                    execution_type="rlm",
                ),
            ),
            _make_assistant_msg(
                mode="direct_only",
                direct_metrics=_make_execution_metrics(
                    total_tokens=200,
                    cost_usd=0.01,
                ),
            ),
            _make_assistant_msg(
                mode="compare",
                rlm_metrics=_make_execution_metrics(
                    total_tokens=600,
                    cost_usd=0.04,
                    execution_type="rlm",
                ),
                direct_metrics=_make_execution_metrics(
                    total_tokens=250,
                    cost_usd=0.012,
                ),
            ),
        ]

        analytics = AnalyticsEngine(msgs).compute()

        assert analytics.total_requests == 3
        assert analytics.rlm_count == 2
        assert analytics.direct_count == 2
        assert analytics.total_tokens == 500 + 200 + 600 + 250


class TestTimeSeries:
    def test_token_trends_ordered(self):
        msgs = [
            _make_assistant_msg(
                rlm_metrics=_make_execution_metrics(total_tokens=100, execution_type="rlm"),
                direct_metrics=_make_execution_metrics(total_tokens=200),
            ),
            _make_assistant_msg(
                rlm_metrics=_make_execution_metrics(total_tokens=300, execution_type="rlm"),
                direct_metrics=_make_execution_metrics(total_tokens=400),
            ),
        ]

        analytics = AnalyticsEngine(msgs).compute()

        assert len(analytics.token_trends) == 2
        assert analytics.token_trends[0]["index"] == 1
        assert analytics.token_trends[0]["rlm_tokens"] == 100
        assert analytics.token_trends[0]["direct_tokens"] == 200
        assert analytics.token_trends[1]["index"] == 2
        assert analytics.token_trends[1]["rlm_tokens"] == 300

    def test_memory_timeline(self):
        msgs = [
            _make_assistant_msg(
                rlm_metrics=_make_execution_metrics(memory_peak_mb=50.0, execution_type="rlm"),
            ),
            _make_assistant_msg(
                rlm_metrics=_make_execution_metrics(memory_peak_mb=100.0, execution_type="rlm"),
            ),
        ]

        analytics = AnalyticsEngine(msgs).compute()

        assert analytics.memory_timeline[0]["memory_peak"] == 50.0
        assert analytics.memory_timeline[1]["memory_peak"] == 100.0
        assert analytics.peak_memory_mb == 100.0


class TestNoneMetrics:
    def test_all_none_metrics_no_crash(self):
        msg = _make_assistant_msg(mode="compare")
        analytics = AnalyticsEngine([msg]).compute()

        assert analytics.total_tokens == 0
        assert analytics.total_requests == 1
        assert analytics.query_details[0]["tokens"] == 0

    def test_partial_none(self):
        msg = _make_assistant_msg(
            mode="compare",
            rlm_metrics=_make_execution_metrics(total_tokens=500, execution_type="rlm"),
            direct_metrics=None,
        )

        analytics = AnalyticsEngine([msg]).compute()

        assert analytics.total_tokens == 500
        assert analytics.rlm_count == 1
        assert analytics.direct_count == 0


class TestEfficiencyScore:
    def test_score_in_range(self):
        msg = _make_assistant_msg(
            rlm_metrics=_make_execution_metrics(
                total_tokens=500,
                cost_usd=0.005,
                execution_time_seconds=1.0,
                execution_type="rlm",
            ),
        )
        analytics = AnalyticsEngine([msg]).compute()

        assert 0.0 <= analytics.efficiency_score <= 100.0

    def test_cheap_fast_is_high_score(self):
        msg = _make_assistant_msg(
            direct_metrics=_make_execution_metrics(
                total_tokens=100,
                cost_usd=0.001,
                execution_time_seconds=0.5,
            ),
        )
        analytics = AnalyticsEngine([msg]).compute()
        assert analytics.efficiency_score >= 80.0

    def test_expensive_slow_is_lower_score(self):
        msg = _make_assistant_msg(
            direct_metrics=_make_execution_metrics(
                total_tokens=20000,
                cost_usd=0.50,
                execution_time_seconds=15.0,
            ),
        )
        analytics = AnalyticsEngine([msg]).compute()
        assert analytics.efficiency_score < 60.0


class TestOutlierDetection:
    def test_no_outliers_with_few_messages(self):
        msgs = [
            _make_assistant_msg(
                direct_metrics=_make_execution_metrics(total_tokens=100),
            ),
            _make_assistant_msg(
                direct_metrics=_make_execution_metrics(total_tokens=110),
            ),
        ]
        analytics = AnalyticsEngine(msgs).compute()
        assert analytics.outlier_indices == []

    def test_detects_large_outlier(self):
        # Need enough "normal" messages so the outlier clearly exceeds mean + 2*stddev
        msgs = [
            _make_assistant_msg(direct_metrics=_make_execution_metrics(total_tokens=100)),
            _make_assistant_msg(direct_metrics=_make_execution_metrics(total_tokens=110)),
            _make_assistant_msg(direct_metrics=_make_execution_metrics(total_tokens=105)),
            _make_assistant_msg(direct_metrics=_make_execution_metrics(total_tokens=95)),
            _make_assistant_msg(direct_metrics=_make_execution_metrics(total_tokens=102)),
            _make_assistant_msg(direct_metrics=_make_execution_metrics(total_tokens=10000)),
        ]
        analytics = AnalyticsEngine(msgs).compute()
        assert 5 in analytics.outlier_indices


class TestQueryDetails:
    def test_structure(self):
        msg = _make_assistant_msg(
            mode="rlm_only",
            content="Response text",
            rlm_metrics=_make_execution_metrics(
                total_tokens=500,
                cost_usd=0.03,
                execution_time_seconds=2.5,
                steps_taken=3,
                memory_peak_mb=65.0,
                execution_type="rlm",
            ),
        )
        analytics = AnalyticsEngine([msg]).compute()

        assert len(analytics.query_details) == 1
        detail = analytics.query_details[0]
        assert detail["mode"] == "rlm_only"
        assert detail["tokens"] == 500
        assert detail["cost"] == pytest.approx(0.03)
        assert detail["time"] == pytest.approx(2.5)
        assert detail["steps"] == 3
        assert detail["memory"] == 65.0
        assert detail["status"] == "success"

    def test_error_status(self):
        msg = _make_assistant_msg(
            rlm_metrics=_make_execution_metrics(success=False, execution_type="rlm"),
        )
        analytics = AnalyticsEngine([msg]).compute()
        assert analytics.query_details[0]["status"] == "error"
