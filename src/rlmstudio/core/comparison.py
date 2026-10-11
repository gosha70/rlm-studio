# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""Comparison metrics and utilities for RLM vs Direct mode.

This module provides tools to compare RLM exploration mode against
traditional direct LLM queries, helping users understand when RLM
provides value and measure efficiency gains.
"""

from dataclasses import dataclass, field
from typing import Any

from ..application.sandbox_vars import MODE_DIRECT, MODE_RLM
from .budget import TokenUsage


@dataclass
class ExecutionMetrics:
    """Metrics for a single execution mode (RLM or Direct)."""

    mode: str  # "rlm" or "direct"
    """Execution mode"""

    answer: str
    """Final answer produced"""

    steps: int
    """Number of execution steps (0 for direct mode)"""

    tokens: TokenUsage
    """Token usage statistics"""

    elapsed_time: float
    """Execution time in seconds"""

    cost: float = 0.0
    """Total API cost in dollars"""

    success: bool = True
    """Whether execution completed successfully"""

    error: str | None = None
    """Error message if execution failed"""

    trace: list[dict[str, Any]] = field(default_factory=list)
    """Execution trace (detailed for RLM mode)"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mode": self.mode,
            "answer": self.answer,
            "steps": self.steps,
            "tokens": self.tokens.to_dict(),
            "elapsed_time": self.elapsed_time,
            "cost": self.cost,
            "success": self.success,
            "error": self.error,
            "trace_length": len(self.trace),
        }


@dataclass
class ComparisonResult:
    """
    Comparison results between RLM and Direct modes.

    Provides detailed metrics and analysis to help users understand
    the tradeoffs between exploration (RLM) and direct querying.
    """

    rlm_metrics: ExecutionMetrics | None = None
    """Metrics from RLM mode execution"""

    direct_metrics: ExecutionMetrics | None = None
    """Metrics from Direct mode execution"""

    def get_token_savings(self) -> dict[str, Any] | None:
        """
        Calculate token savings when using RLM vs Direct.

        Returns:
            Dictionary with savings metrics, or None if comparison not available
        """
        if not self.rlm_metrics or not self.direct_metrics:
            return None

        rlm_total = self.rlm_metrics.tokens.total_tokens
        direct_total = self.direct_metrics.tokens.total_tokens

        savings = direct_total - rlm_total
        savings_percent = (savings / direct_total * 100) if direct_total > 0 else 0

        return {
            "rlm_tokens": rlm_total,
            "direct_tokens": direct_total,
            "savings_tokens": savings,
            "savings_percent": savings_percent,
            "rlm_is_better": savings > 0,
        }

    def get_cost_savings(self) -> dict[str, Any] | None:
        """
        Calculate cost savings when using RLM vs Direct.

        Returns:
            Dictionary with cost metrics, or None if comparison not available
        """
        if not self.rlm_metrics or not self.direct_metrics:
            return None

        rlm_cost = self.rlm_metrics.cost
        direct_cost = self.direct_metrics.cost

        savings = direct_cost - rlm_cost
        savings_percent = (savings / direct_cost * 100) if direct_cost > 0 else 0

        return {
            "rlm_cost": rlm_cost,
            "direct_cost": direct_cost,
            "savings": savings,
            "savings_percent": savings_percent,
            "rlm_is_better": savings > 0,
        }

    def get_time_comparison(self) -> dict[str, Any] | None:
        """
        Compare execution times.

        Returns:
            Dictionary with time metrics, or None if comparison not available
        """
        if not self.rlm_metrics or not self.direct_metrics:
            return None

        rlm_time = self.rlm_metrics.elapsed_time
        direct_time = self.direct_metrics.elapsed_time

        difference = rlm_time - direct_time
        ratio = rlm_time / direct_time if direct_time > 0 else 0

        return {
            "rlm_time": rlm_time,
            "direct_time": direct_time,
            "difference": difference,
            "ratio": ratio,
            "direct_is_faster": difference > 0,
        }

    def get_summary(self) -> dict[str, Any]:
        """
        Get comprehensive comparison summary.

        Returns:
            Dictionary with all comparison metrics and recommendations
        """
        summary: dict[str, Any] = {
            "has_rlm": self.rlm_metrics is not None,
            "has_direct": self.direct_metrics is not None,
            "can_compare": self.rlm_metrics is not None and self.direct_metrics is not None,
        }

        if summary["can_compare"]:
            summary.update(
                {
                    "token_savings": self.get_token_savings(),
                    "cost_savings": self.get_cost_savings(),
                    "time_comparison": self.get_time_comparison(),
                }
            )

            # Add recommendation
            token_savings = self.get_token_savings()
            cost_savings = self.get_cost_savings()

            if token_savings and cost_savings:
                if token_savings["savings_percent"] > 20:
                    summary["recommendation"] = MODE_RLM
                    summary["recommendation_reason"] = (
                        f"RLM saves {token_savings['savings_percent']:.1f}% tokens"
                    )
                elif token_savings["savings_percent"] < -20:
                    summary["recommendation"] = MODE_DIRECT
                    summary["recommendation_reason"] = (
                        f"Direct mode uses {-token_savings['savings_percent']:.1f}% fewer tokens"
                    )
                else:
                    summary["recommendation"] = "similar"
                    summary["recommendation_reason"] = "Both modes perform similarly"

        if self.rlm_metrics:
            summary["rlm_metrics"] = self.rlm_metrics.to_dict()

        if self.direct_metrics:
            summary["direct_metrics"] = self.direct_metrics.to_dict()

        return summary

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.get_summary()
