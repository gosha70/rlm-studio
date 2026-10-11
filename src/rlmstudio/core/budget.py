# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""Budget tracking and enforcement for RLM execution.

This module provides comprehensive tracking of:
- Execution steps
- Token usage (input/output)
- API costs
- Execution time
- Recursion depth
"""

import time
from dataclasses import dataclass
from typing import Any

from .errors import BudgetExceeded


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Uses a simple heuristic: ~4 characters per token (GPT tokenizer average).
    For production use, integrate with tiktoken or similar.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    # Simple heuristic: 1 token ≈ 4 characters
    # This is approximate but good enough for tracking
    return max(1, len(text) // 4)


@dataclass
class TokenUsage:
    """Token usage statistics."""

    input_tokens: int = 0
    """Total input tokens sent to LLM"""

    output_tokens: int = 0
    """Total output tokens received from LLM"""

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.input_tokens + self.output_tokens

    def add_input(self, tokens: int) -> None:
        """Add input tokens."""
        self.input_tokens += tokens

    def add_output(self, tokens: int) -> None:
        """Add output tokens."""
        self.output_tokens += tokens

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class CostTracker:
    """Track API costs."""

    input_cost_per_1k: float = 0.0
    """Cost per 1000 input tokens"""

    output_cost_per_1k: float = 0.0
    """Cost per 1000 output tokens"""

    total_cost: float = 0.0
    """Total accumulated cost"""

    def add_usage(self, input_tokens: int, output_tokens: int) -> float:
        """
        Add token usage and calculate cost.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost for this usage
        """
        input_cost = (input_tokens / 1000.0) * self.input_cost_per_1k
        output_cost = (output_tokens / 1000.0) * self.output_cost_per_1k
        cost = input_cost + output_cost

        self.total_cost += cost
        return cost

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "input_cost_per_1k": self.input_cost_per_1k,
            "output_cost_per_1k": self.output_cost_per_1k,
            "total_cost": self.total_cost,
        }


@dataclass
class BudgetLimits:
    """Budget limits for RLM execution."""

    max_steps: int | None = None
    """Maximum number of execution steps (None = unlimited)"""

    max_tokens: int | None = None
    """Maximum total tokens (None = unlimited)"""

    max_cost: float | None = None
    """Maximum cost in dollars (None = unlimited)"""

    max_time_seconds: float | None = None
    """Maximum execution time in seconds (None = unlimited)"""

    max_recursion_depth: int | None = None
    """Maximum recursion depth for subcalls (None = unlimited)"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_steps": self.max_steps,
            "max_tokens": self.max_tokens,
            "max_cost": self.max_cost,
            "max_time_seconds": self.max_time_seconds,
            "max_recursion_depth": self.max_recursion_depth,
        }


class BudgetTracker:
    """
    Track and enforce execution budgets.

    Monitors:
    - Execution steps
    - Token usage (input/output)
    - API costs
    - Execution time
    - Recursion depth

    Example:
        >>> limits = BudgetLimits(max_steps=10, max_tokens=5000)
        >>> tracker = BudgetTracker(limits)
        >>>
        >>> tracker.start()
        >>> tracker.add_step()
        >>> tracker.add_llm_call("prompt text", "response text")
        >>> tracker.check_limits()  # Raises BudgetExceeded if over
        >>>
        >>> stats = tracker.get_stats()
        >>> print(f"Used {stats['steps']}/{limits.max_steps} steps")
    """

    def __init__(
        self,
        limits: BudgetLimits | None = None,
        cost_tracker: CostTracker | None = None,
    ):
        """
        Initialize budget tracker.

        Args:
            limits: Budget limits to enforce
            cost_tracker: Cost tracker for API pricing
        """
        self.limits = limits or BudgetLimits()
        self.cost_tracker = cost_tracker or CostTracker()

        # Usage tracking
        self.steps = 0
        self.tokens = TokenUsage()
        self.recursion_depth = 0

        # Timing
        self.start_time: float | None = None
        self.end_time: float | None = None

        # History
        self.step_history: list[dict[str, Any]] = []

    def start(self) -> None:
        """Start timing execution."""
        self.start_time = time.time()

    def stop(self) -> None:
        """Stop timing execution."""
        self.end_time = time.time()

    @property
    def elapsed_time(self) -> float | None:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return None

        end = self.end_time or time.time()
        return end - self.start_time

    def add_step(self) -> None:
        """Increment step counter."""
        self.steps += 1

    def add_llm_call(
        self,
        input_text: str,
        output_text: str,
        actual_input_tokens: int | None = None,
        actual_output_tokens: int | None = None,
    ) -> None:
        """
        Record an LLM API call.

        Args:
            input_text: Input text sent to LLM
            output_text: Output text received from LLM
            actual_input_tokens: Actual input tokens (if known from API)
            actual_output_tokens: Actual output tokens (if known from API)
        """
        # Use actual tokens if provided, otherwise estimate
        input_tokens = actual_input_tokens or estimate_tokens(input_text)
        output_tokens = actual_output_tokens or estimate_tokens(output_text)

        self.tokens.add_input(input_tokens)
        self.tokens.add_output(output_tokens)

        # Track cost
        cost = self.cost_tracker.add_usage(input_tokens, output_tokens)

        # Record in history
        self.step_history.append(
            {
                "step": self.steps,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "timestamp": time.time() if self.start_time else None,
            }
        )

    def enter_recursion(self) -> None:
        """Increment recursion depth."""
        self.recursion_depth += 1

    def exit_recursion(self) -> None:
        """Decrement recursion depth."""
        self.recursion_depth = max(0, self.recursion_depth - 1)

    def check_limits(self) -> None:
        """
        Check if any limits are exceeded.

        Raises:
            BudgetExceeded: If any limit is exceeded
        """
        limits = self.limits

        # Check step limit
        if limits.max_steps is not None and self.steps >= limits.max_steps:
            raise BudgetExceeded(
                f"Maximum steps ({limits.max_steps}) exceeded. Current: {self.steps}"
            )

        # Check token limit
        if limits.max_tokens is not None and self.tokens.total_tokens >= limits.max_tokens:
            raise BudgetExceeded(
                f"Maximum tokens ({limits.max_tokens}) exceeded. "
                f"Current: {self.tokens.total_tokens}"
            )

        # Check cost limit
        if limits.max_cost is not None and self.cost_tracker.total_cost >= limits.max_cost:
            raise BudgetExceeded(
                f"Maximum cost (${limits.max_cost:.4f}) exceeded. "
                f"Current: ${self.cost_tracker.total_cost:.4f}"
            )

        # Check time limit
        if limits.max_time_seconds is not None:
            elapsed = self.elapsed_time
            if elapsed is not None and elapsed >= limits.max_time_seconds:
                raise BudgetExceeded(
                    f"Maximum time ({limits.max_time_seconds}s) exceeded. Current: {elapsed:.2f}s"
                )

        # Check recursion depth
        if (
            limits.max_recursion_depth is not None
            and self.recursion_depth >= limits.max_recursion_depth
        ):
            raise BudgetExceeded(
                f"Maximum recursion depth ({limits.max_recursion_depth}) exceeded. "
                f"Current: {self.recursion_depth}"
            )

    def get_stats(self) -> dict[str, Any]:
        """
        Get current usage statistics.

        Returns:
            Dictionary with usage stats
        """
        return {
            "steps": self.steps,
            "tokens": self.tokens.to_dict(),
            "cost": self.cost_tracker.to_dict(),
            "elapsed_time": self.elapsed_time,
            "recursion_depth": self.recursion_depth,
            "limits": self.limits.to_dict(),
        }

    def get_utilization(self) -> dict[str, float | None]:
        """
        Get budget utilization as percentages.

        Returns:
            Dictionary with utilization percentages (0.0-1.0) or None if no limit
        """
        limits = self.limits

        return {
            "steps": self.steps / limits.max_steps if limits.max_steps else None,
            "tokens": self.tokens.total_tokens / limits.max_tokens if limits.max_tokens else None,
            "cost": self.cost_tracker.total_cost / limits.max_cost if limits.max_cost else None,
            "time": (
                self.elapsed_time / limits.max_time_seconds
                if limits.max_time_seconds and self.elapsed_time
                else None
            ),
            "recursion": (
                self.recursion_depth / limits.max_recursion_depth
                if limits.max_recursion_depth
                else None
            ),
        }
