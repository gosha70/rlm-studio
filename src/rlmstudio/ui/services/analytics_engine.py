# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0
"""
AnalyticsEngine - Session-wide metrics aggregation.

Pure Python module with zero Streamlit dependency. Takes a list of
chat message dicts (as stored in st.session_state.chat_messages) and
produces a SessionAnalytics dataclass with all aggregated data needed
by the Metrics Dashboard and sidebar summary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from rlmstudio.application.sandbox_vars import MODE_DIRECT, MODE_RAG, MODE_RLM

from .models import ExecutionMetrics


@dataclass
class SessionAnalytics:
    """Aggregated analytics for the entire chat session."""

    # Overview cards
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_requests: int = 0
    avg_response_time_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    efficiency_score: float = 0.0  # 0-100

    # Per-mode breakdowns
    rlm_total_tokens: int = 0
    rlm_total_cost: float = 0.0
    rlm_count: int = 0
    direct_total_tokens: int = 0
    direct_total_cost: float = 0.0
    direct_count: int = 0
    rag_total_tokens: int = 0
    rag_total_cost: float = 0.0
    rag_count: int = 0

    # Savings (from compare-mode messages where both RLM and Direct ran)
    rlm_savings_tokens: int = 0
    rlm_savings_cost: float = 0.0

    # Time series data (ordered by assistant message index)
    token_trends: list[dict[str, Any]] = field(default_factory=list)
    cost_trends: list[dict[str, Any]] = field(default_factory=list)
    memory_timeline: list[dict[str, Any]] = field(default_factory=list)
    time_trends: list[dict[str, Any]] = field(default_factory=list)

    # Per-query detail rows for the insights table
    query_details: list[dict[str, Any]] = field(default_factory=list)

    # Advanced analytics
    outlier_indices: list[int] = field(default_factory=list)


class AnalyticsEngine:
    """Aggregate session-wide metrics from chat message dicts.

    Usage::

        engine = AnalyticsEngine(st.session_state.chat_messages)
        analytics = engine.compute()
    """

    def __init__(self, messages: list[dict]) -> None:
        # Keep only assistant messages (user messages have no metrics)
        self._messages = [m for m in messages if m.get("role") == "assistant"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self) -> SessionAnalytics:
        """Single-pass aggregation over all assistant messages."""
        a = SessionAnalytics()

        total_time = 0.0
        all_tokens: list[int] = []  # for outlier detection

        for idx, msg in enumerate(self._messages):
            rlm_m = self._get_metrics(msg, MODE_RLM)
            direct_m = self._get_metrics(msg, MODE_DIRECT)
            rag_m = self._get_metrics(msg, MODE_RAG)

            # --- Per-message token totals (across all modes in this message) ---
            msg_tokens = 0
            msg_cost = 0.0
            msg_time = 0.0
            msg_memory_peak = 0.0
            msg_steps = 0
            msg_status = "success"

            # RLM
            rlm_tokens = 0
            rlm_cost = 0.0
            rlm_time = 0.0
            if rlm_m is not None:
                rlm_tokens = rlm_m.total_tokens
                rlm_cost = rlm_m.cost_usd
                rlm_time = rlm_m.execution_time_seconds
                a.rlm_total_tokens += rlm_tokens
                a.rlm_total_cost += rlm_cost
                a.rlm_count += 1
                msg_tokens += rlm_tokens
                msg_cost += rlm_cost
                msg_time += rlm_time
                msg_steps = max(msg_steps, rlm_m.steps_taken)
                msg_memory_peak = max(msg_memory_peak, rlm_m.memory_peak_mb)
                if not rlm_m.success:
                    msg_status = "error"

            # Direct
            direct_tokens = 0
            direct_cost = 0.0
            direct_time = 0.0
            if direct_m is not None:
                direct_tokens = direct_m.total_tokens
                direct_cost = direct_m.cost_usd
                direct_time = direct_m.execution_time_seconds
                a.direct_total_tokens += direct_tokens
                a.direct_total_cost += direct_cost
                a.direct_count += 1
                msg_tokens += direct_tokens
                msg_cost += direct_cost
                msg_time += direct_time
                msg_memory_peak = max(msg_memory_peak, direct_m.memory_peak_mb)
                if not direct_m.success:
                    msg_status = "error"

            # RAG
            rag_tokens = 0
            rag_cost = 0.0
            rag_time = 0.0
            if rag_m is not None:
                rag_tokens = rag_m.total_tokens
                rag_cost = rag_m.cost_usd
                rag_time = rag_m.execution_time_seconds
                a.rag_total_tokens += rag_tokens
                a.rag_total_cost += rag_cost
                a.rag_count += 1
                msg_tokens += rag_tokens
                msg_cost += rag_cost
                msg_time += rag_time
                msg_memory_peak = max(msg_memory_peak, rag_m.memory_peak_mb)
                if not rag_m.success:
                    msg_status = "error"

            # --- Savings (compare mode: both RLM and Direct present) ---
            if rlm_m is not None and direct_m is not None:
                token_diff = direct_m.total_tokens - rlm_m.total_tokens
                cost_diff = direct_m.cost_usd - rlm_m.cost_usd
                if token_diff > 0:
                    a.rlm_savings_tokens += token_diff
                if cost_diff > 0:
                    a.rlm_savings_cost += cost_diff

            # --- Accumulate totals ---
            a.total_tokens += msg_tokens
            a.total_cost_usd += msg_cost
            total_time += msg_time
            a.peak_memory_mb = max(a.peak_memory_mb, msg_memory_peak)
            all_tokens.append(msg_tokens)

            # --- Time series ---
            a.token_trends.append(
                {
                    "index": idx + 1,
                    "rlm_tokens": rlm_tokens,
                    "direct_tokens": direct_tokens,
                    "rag_tokens": rag_tokens,
                }
            )
            a.cost_trends.append(
                {
                    "index": idx + 1,
                    "rlm_cost": rlm_cost,
                    "direct_cost": direct_cost,
                    "rag_cost": rag_cost,
                }
            )
            a.memory_timeline.append(
                {
                    "index": idx + 1,
                    "memory_peak": msg_memory_peak,
                }
            )
            a.time_trends.append(
                {
                    "index": idx + 1,
                    "rlm_time": rlm_time,
                    "direct_time": direct_time,
                    "rag_time": rag_time,
                }
            )

            # --- Query details row ---
            # Try to find the user query from the preceding user message
            query_text = msg.get("content", "")
            mode = msg.get("mode", "unknown")
            a.query_details.append(
                {
                    "query": query_text,
                    "mode": mode,
                    "status": msg_status,
                    "tokens": msg_tokens,
                    "cost": msg_cost,
                    "time": msg_time,
                    "steps": msg_steps,
                    "memory": msg_memory_peak,
                }
            )

        # --- Derived totals ---
        a.total_requests = len(self._messages)
        if a.total_requests > 0:
            a.avg_response_time_seconds = total_time / a.total_requests

        # --- Efficiency score ---
        a.efficiency_score = self._compute_efficiency_score(a)

        # --- Outlier detection ---
        a.outlier_indices = self._detect_outliers(all_tokens)

        return a

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_metrics(msg: dict, mode: str) -> ExecutionMetrics | None:
        """Safely extract ExecutionMetrics from a message dict."""
        metrics = msg.get(f"{mode}_metrics")
        if isinstance(metrics, ExecutionMetrics):
            return metrics
        return None

    @staticmethod
    def _compute_efficiency_score(a: SessionAnalytics) -> float:
        """Compute 0-100 efficiency score.

        Heuristic:
        - Start at 100
        - Deduct for high average cost (>$0.10 per request → -30)
        - Deduct for high average tokens (>5000 per request → -20)
        - Deduct for slow average speed (>10s per request → -20)
        - Deduct for errors (any error → -10)
        - Bonus for RLM savings (+10 if savings > 0)
        """
        if a.total_requests == 0:
            return 0.0

        score = 100.0
        avg_cost = a.total_cost_usd / a.total_requests
        avg_tokens = a.total_tokens / a.total_requests
        avg_time = a.avg_response_time_seconds

        # Cost penalty (scaled 0-30)
        if avg_cost > 0.10:
            score -= min(30, (avg_cost - 0.10) / 0.10 * 10)
        elif avg_cost > 0.01:
            score -= min(15, (avg_cost - 0.01) / 0.01 * 2)

        # Token penalty (scaled 0-20)
        if avg_tokens > 5000:
            score -= min(20, (avg_tokens - 5000) / 5000 * 10)

        # Speed penalty (scaled 0-20)
        if avg_time > 10:
            score -= min(20, (avg_time - 10) / 10 * 10)
        elif avg_time > 5:
            score -= min(10, (avg_time - 5) / 5 * 5)

        # Error penalty
        error_count = sum(1 for d in a.query_details if d.get("status") == "error")
        if error_count > 0:
            score -= min(10, error_count * 5)

        # RLM savings bonus
        if a.rlm_savings_tokens > 0 or a.rlm_savings_cost > 0:
            score += 10

        return max(0.0, min(100.0, score))

    @staticmethod
    def _detect_outliers(token_counts: list[int]) -> list[int]:
        """Flag messages with token count > mean + 2*stddev."""
        if len(token_counts) < 3:
            return []

        mean = sum(token_counts) / len(token_counts)
        variance = sum((x - mean) ** 2 for x in token_counts) / len(token_counts)
        stddev = math.sqrt(variance)

        if stddev == 0:
            return []

        threshold = mean + 2 * stddev
        return [i for i, count in enumerate(token_counts) if count > threshold]
