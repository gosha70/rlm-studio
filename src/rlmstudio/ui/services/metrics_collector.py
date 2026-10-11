# mypy: disable-error-code="no-untyped-def,index,no-any-return,assignment"
# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0
"""
MetricsCollector - Collect and aggregate execution metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rlmstudio.application.sandbox_vars import MODE_DIRECT, MODE_RLM

from .models import ComparisonMetrics, ExecutionMetrics

if TYPE_CHECKING:
    from .memory_monitor import MemoryMonitor


class MetricsCollector:
    """
    Collect and manage execution metrics.

    Responsibilities:
    - Count tokens from execution traces
    - Calculate costs based on provider pricing
    - Monitor memory usage
    - Compare metrics between execution paths
    - Aggregate session-wide metrics
    """

    def __init__(self):
        """Initialize MetricsCollector."""
        from rlmstudio.ui.data.providers_catalog import build_metrics_pricing

        self.provider_pricing: dict[str, dict[str, float]] = build_metrics_pricing()

    async def collect_rlm_metrics(
        self,
        rlm_result: dict[str, Any],
        rlm_trace: list[dict[str, Any]],
        memory_monitor: MemoryMonitor,
        provider: str = "openai",
        model: str = "gpt-4",
    ) -> ExecutionMetrics:
        """
        Collect metrics from RLM execution.

        Args:
            rlm_result: Result dict from RLM execution
            rlm_trace: Step-by-step execution trace
            memory_monitor: MemoryMonitor instance
            provider: LLM provider name
            model: Model name

        Returns:
            ExecutionMetrics for RLM execution

        Example:
            >>> result = {"response": Response(...), "metrics": {...}}
            >>> trace = [{"step": 1, "input_tokens": 200, ...}, ...]
            >>> metrics = await collector.collect_rlm_metrics(result, trace, monitor)
            >>> print(metrics.total_tokens)
            850
            >>> print(metrics.cost_usd)
            0.050

        Implementation notes:
        - Sums tokens across all steps in trace
        - Calculates cost using provider pricing
        - Gets memory peak from MemoryMonitor
        - Handles missing data gracefully
        """
        # Sum tokens from trace
        input_tokens = sum(step.get("input_tokens", 0) for step in rlm_trace)
        output_tokens = sum(step.get("output_tokens", 0) for step in rlm_trace)
        total_tokens = input_tokens + output_tokens

        # Calculate cost
        cost = self._calculate_cost(input_tokens, output_tokens, provider, model)
        cost_breakdown = self._cost_breakdown(input_tokens, output_tokens, provider, model)

        # Get execution time from trace
        execution_time = sum(step.get("duration_seconds", 0) for step in rlm_trace)

        # Get memory metrics from monitor
        memory_stats = memory_monitor.get_stats()
        memory_used = memory_stats.get("current", 45.2)
        memory_peak = memory_stats.get("peak", 62.1)

        return ExecutionMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            cost_breakdown=cost_breakdown,
            execution_time_seconds=execution_time,
            steps_taken=len(rlm_trace),
            memory_used_mb=memory_used,
            memory_peak_mb=memory_peak,
            success=True,
            execution_type=MODE_RLM,
        )

    async def collect_direct_metrics(
        self,
        direct_result: dict[str, Any],
        memory_monitor: MemoryMonitor,
        provider: str = "openai",
        model: str = "gpt-4",
    ) -> ExecutionMetrics:
        """
        Collect metrics from direct LLM execution.

        Args:
            direct_result: Result dict from direct LLM call
            memory_monitor: MemoryMonitor instance
            provider: LLM provider name
            model: Model name

        Returns:
            ExecutionMetrics for direct execution

        Example:
            >>> result = {"response": Response(...)}
            >>> metrics = await collector.collect_direct_metrics(result, monitor)
            >>> print(metrics.total_tokens)
            270
            >>> print(metrics.steps_taken)
            0

        Implementation notes:
        - Extracts token counts from LLM response
        - Calculates cost using provider pricing
        - Gets memory usage from MemoryMonitor
        - No trace for direct execution (it's a single call)
        """
        # Extract tokens from result (in simulation, we use fixed values)
        # In production, these would come from the LLM response
        input_tokens = direct_result.get("input_tokens", 150)
        output_tokens = direct_result.get("output_tokens", 120)
        total_tokens = input_tokens + output_tokens

        # Calculate cost
        cost = self._calculate_cost(input_tokens, output_tokens, provider, model)
        cost_breakdown = self._cost_breakdown(input_tokens, output_tokens, provider, model)

        # Get execution time
        execution_time = direct_result.get("execution_time", 0.8)

        # Get memory metrics from monitor
        memory_stats = memory_monitor.get_stats()
        memory_used = memory_stats.get("current", 45.2)
        memory_peak = memory_stats.get("peak", 62.1)

        return ExecutionMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            cost_breakdown=cost_breakdown,
            execution_time_seconds=execution_time,
            steps_taken=0,  # Direct has no exploration steps
            memory_used_mb=memory_used,
            memory_peak_mb=memory_peak,
            success=True,
            execution_type=MODE_DIRECT,
        )

    def compare_metrics(
        self,
        rlm_metrics: ExecutionMetrics,
        direct_metrics: ExecutionMetrics,
    ) -> ComparisonMetrics:
        """
        Compare metrics between RLM and Direct execution.

        Args:
            rlm_metrics: Metrics from RLM
            direct_metrics: Metrics from Direct

        Returns:
            ComparisonMetrics with savings and recommendation

        Example:
            >>> rlm_m = ExecutionMetrics(cost_usd=0.050, total_tokens=850)
            >>> direct_m = ExecutionMetrics(cost_usd=0.015, total_tokens=270)
            >>> comparison = collector.compare_metrics(rlm_m, direct_m)
            >>> print(comparison.cost_delta_percent)
            233
            >>> print("RLM is more expensive")

        Implementation notes:
        - Calculates token savings (absolute and %)
        - Calculates cost savings (absolute and %)
        - Notes time difference
        - Generates recommendation based on trade-offs
        """
        # Calculate token deltas
        token_delta = rlm_metrics.total_tokens - direct_metrics.total_tokens

        # Calculate cost deltas
        cost_delta = rlm_metrics.cost_usd - direct_metrics.cost_usd
        cost_delta_pct = (
            (cost_delta / direct_metrics.cost_usd * 100) if direct_metrics.cost_usd > 0 else 0
        )

        # Calculate time deltas
        time_delta = rlm_metrics.execution_time_seconds - direct_metrics.execution_time_seconds
        time_delta_pct = (
            (time_delta / direct_metrics.execution_time_seconds * 100)
            if direct_metrics.execution_time_seconds > 0
            else 0
        )

        # Generate recommendation based on trade-offs
        # Consider multiple factors: cost, tokens, time, and absolute values

        # Both very cheap (< $0.01)? Recommend simpler approach
        if rlm_metrics.cost_usd < 0.01 and direct_metrics.cost_usd < 0.01:
            recommendation = (
                "Both approaches are very affordable (<$0.01). "
                "Recommend Direct for simple queries like this - it's faster and simpler."
            )
        # RLM significantly more expensive (>2x AND >$0.01 difference)?
        elif cost_delta > 0.01 and cost_delta_pct > 100:
            if token_delta < 0:  # But RLM saved tokens
                token_savings_pct = abs(token_delta / direct_metrics.total_tokens * 100)
                recommendation = (
                    f"Direct is {abs(cost_delta_pct):.0f}% cheaper (${abs(cost_delta):.4f} saved), "
                    f"but RLM saved {token_savings_pct:.0f}% tokens. "
                    f"Use Direct for cost-sensitive work, RLM for token efficiency."
                )
            else:
                recommendation = (
                    f"Recommend Direct: {abs(cost_delta_pct):.0f}% cheaper "
                    f"(${abs(cost_delta):.4f} saved) and "
                    f"{abs(time_delta_pct):.0f}% faster. "
                    f"RLM's {rlm_metrics.steps_taken} steps didn't provide clear benefit here."
                )
        # RLM saved significant tokens (>50%)?
        elif token_delta < 0 and abs(token_delta / direct_metrics.total_tokens * 100) > 50:
            token_savings_pct = abs(token_delta / direct_metrics.total_tokens * 100)
            recommendation = (
                f"RLM saved {token_savings_pct:.0f}% tokens ({abs(token_delta)} tokens), "
                f"worth the {abs(time_delta_pct):.0f}% time increase for complex analysis."
            )
        # Similar costs
        else:
            recommendation = (
                f"Similar costs (${rlm_metrics.cost_usd:.4f} vs ${direct_metrics.cost_usd:.4f}). "
                f"Use Direct for simple queries, RLM for complex document analysis."
            )

        return ComparisonMetrics(
            rlm_cost_usd=rlm_metrics.cost_usd,
            direct_cost_usd=direct_metrics.cost_usd,
            cost_delta_usd=cost_delta,
            cost_delta_percent=cost_delta_pct,
            rlm_time_seconds=rlm_metrics.execution_time_seconds,
            direct_time_seconds=direct_metrics.execution_time_seconds,
            time_delta_seconds=time_delta,
            time_delta_percent=time_delta_pct,
            rlm_tokens=rlm_metrics.total_tokens,
            direct_tokens=direct_metrics.total_tokens,
            token_delta=token_delta,
            rlm_steps=rlm_metrics.steps_taken,
            direct_steps=direct_metrics.steps_taken,
            recommendation=recommendation,
        )

    def _calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        provider: str,
        model: str,
    ) -> float:
        """
        Calculate cost for token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            provider: Provider name (openai, anthropic, etc)
            model: Model name

        Returns:
            Cost in USD

        Example:
            >>> collector = MetricsCollector()
            >>> cost = collector._calculate_cost(1000, 500, "openai", "gpt-4")
            >>> print(f"${cost:.4f}")
            $0.0450
        """
        # LATER: Handle edge cases with logging/monitoring
        if provider not in self.provider_pricing:
            return 0.0

        if model not in self.provider_pricing[provider]:
            return 0.0

        pricing = self.provider_pricing[provider][model]
        input_cost_per_1k = pricing["input"]
        output_cost_per_1k = pricing["output"]

        # Calculate: (tokens / 1000) * cost_per_1k
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k

        total_cost = input_cost + output_cost
        return round(total_cost, 6)  # Round to 6 decimals for USD precision

    def _cost_breakdown(
        self,
        input_tokens: int,
        output_tokens: int,
        provider: str,
        model: str,
    ) -> dict[str, float]:
        """
        Get cost breakdown (input vs output).

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            provider: Provider name
            model: Model name

        Returns:
            Dict with {'input': $x, 'output': $y}

        Example:
            >>> collector = MetricsCollector()
            >>> breakdown = collector._cost_breakdown(1000, 500, "openai", "gpt-4")
            >>> print(breakdown)
            {'input': 0.03, 'output': 0.03}
        """
        # LATER: Handle edge cases with logging/monitoring
        if provider not in self.provider_pricing:
            return {"input": 0.0, "output": 0.0}

        if model not in self.provider_pricing[provider]:
            return {"input": 0.0, "output": 0.0}

        pricing = self.provider_pricing[provider][model]
        input_cost_per_1k = pricing["input"]
        output_cost_per_1k = pricing["output"]

        # Calculate separately: (tokens / 1000) * cost_per_1k
        input_cost = round((input_tokens / 1000) * input_cost_per_1k, 6)
        output_cost = round((output_tokens / 1000) * output_cost_per_1k, 6)

        return {
            "input": input_cost,
            "output": output_cost,
        }

    def add_provider_pricing(
        self,
        provider: str,
        model: str,
        input_cost_per_1k: float,
        output_cost_per_1k: float,
    ) -> None:
        """
        Add or update pricing for a provider/model combination.

        Args:
            provider: Provider name
            model: Model name
            input_cost_per_1k: Input cost per 1K tokens
            output_cost_per_1k: Output cost per 1K tokens
        """
        if provider not in self.provider_pricing:
            self.provider_pricing[provider] = {}

        self.provider_pricing[provider][model] = {
            "input": input_cost_per_1k,
            "output": output_cost_per_1k,
        }
