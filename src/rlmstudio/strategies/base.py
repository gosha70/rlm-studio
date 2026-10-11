# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""Strategy protocol and unified result type for multi-strategy evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from rlmstudio.core.budget import TokenUsage

if TYPE_CHECKING:
    from rlmstudio.core.comparison import ExecutionMetrics


@runtime_checkable
class LLMStrategy(Protocol):
    """Common interface for all evaluation strategies."""

    @property
    def name(self) -> str: ...

    def run(self, content: str, query: str) -> StrategyResult: ...


@dataclass
class StrategyResult:
    """Unified result from any strategy execution."""

    strategy: str
    answer: str
    success: bool = True
    error: str | None = None
    steps: int = 0
    tokens: TokenUsage = field(default_factory=TokenUsage)
    cost: float = 0.0
    elapsed_time: float = 0.0
    trace: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "answer": self.answer,
            "success": self.success,
            "error": self.error,
            "steps": self.steps,
            "tokens": self.tokens.to_dict(),
            "cost": self.cost,
            "elapsed_time": self.elapsed_time,
            "trace_length": len(self.trace),
            "metadata": self.metadata,
        }

    def to_execution_metrics(self) -> ExecutionMetrics:  # noqa: F821
        """Convert to existing comparison.ExecutionMetrics for UI compat."""
        from rlmstudio.core.comparison import ExecutionMetrics

        return ExecutionMetrics(
            mode=self.strategy,
            answer=self.answer,
            steps=self.steps,
            tokens=self.tokens,
            elapsed_time=self.elapsed_time,
            cost=self.cost,
            success=self.success,
            error=self.error,
            trace=self.trace,
        )
