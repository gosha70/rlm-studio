"""Outcome classifier — single source of truth for execution failure categorization.

Classifies an execution result into one of five outcome categories
based on the full run result (success flag, error string, and answer content).
No other module should re-parse error strings for failure categorization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlmstudio.domain.entities import ExecutionTrace

_logger = logging.getLogger(__name__)


class OutcomeCategory(Enum):
    """Categorization of an execution outcome."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    PREFILL_TIMEOUT = "prefill_timeout"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CONTEXT_OVERFLOW = "context_overflow"
    GENERAL_ERROR = "general_error"


# Prefill-dominance threshold (spec v1.7 §4): a step is "prefill-dominated"
# when ttft_ms / (duration * 1000) > _PREFILL_RATIO. A run promotes from
# TIMEOUT to PREFILL_TIMEOUT when >= _MIN_PREFILL_STEPS steps satisfy this.
_PREFILL_RATIO = 0.7
_MIN_PREFILL_STEPS = 2


# Warning prefix used by RLM/Direct/RAG use cases for degraded outcomes
_DEGRADED_WARNING_PREFIX = "\u26a0\ufe0f"


def is_degraded_answer(success: bool, answer: str) -> bool:
    """True when a run reported success but its answer carries the warning prefix.

    Such runs are stored as successes, so a caller selecting problem runs
    (prompt refinement's source list) must check the answer as well.
    """
    return success and answer.startswith(_DEGRADED_WARNING_PREFIX)


@dataclass(frozen=True)
class ExecutionOutcome:
    """Result of classifying an execution."""

    category: OutcomeCategory
    is_usable: bool


def classify_execution_outcome(
    success: bool,
    error: str | None,
    answer: str,
    *,
    trace: ExecutionTrace | None = None,
) -> ExecutionOutcome:
    """Classify from the full result, not just the error string.

    Handles three states:
    1. ``success=False`` — hard failure, classify from error string keywords.
       When ``trace`` is provided and the failure is a timeout, a
       prefill-dominance check may promote ``TIMEOUT`` to
       ``PREFILL_TIMEOUT`` (spec v1.7 §4).
    2. ``success=True`` + answer starts with warning prefix — degraded,
       classify from answer content.
    3. ``success=True`` + normal answer — real success.

    Args:
        success: Run-level success flag.
        error: Error message string, if any.
        answer: Answer text (may carry a ``⚠️`` warning prefix).
        trace: Optional materialized :class:`ExecutionTrace`. When
            provided, enables the PREFILL_TIMEOUT refinement for
            timeout failures. Callers that can't materialize a trace
            (e.g. dashboard legacy-row fallback) keep working by
            passing ``None``.
    """
    if not success:
        return _classify_hard_failure(error, trace=trace)

    # success=True but answer may be a degraded warning
    if is_degraded_answer(success, answer):
        return _classify_degraded(answer)

    return ExecutionOutcome(OutcomeCategory.SUCCESS, is_usable=True)


def _classify_hard_failure(
    error: str | None,
    *,
    trace: ExecutionTrace | None = None,
) -> ExecutionOutcome:
    """Classify a hard failure (success=False) from the error string.

    Keyword priority: timeout > context overflow > budget > general.
    If an error contains multiple keywords (e.g. "budget timeout"),
    the first match wins. When the failure is a timeout and the trace
    is prefill-dominated, promote to PREFILL_TIMEOUT.
    """
    error_lower = (error or "").lower()
    if "timeout" in error_lower or "timed out" in error_lower:
        if trace is not None and _is_prefill_dominated(trace):
            return ExecutionOutcome(OutcomeCategory.PREFILL_TIMEOUT, is_usable=False)
        return ExecutionOutcome(OutcomeCategory.TIMEOUT, is_usable=False)
    if "context window" in error_lower or "context_length" in error_lower:
        return ExecutionOutcome(OutcomeCategory.CONTEXT_OVERFLOW, is_usable=False)
    if "budget" in error_lower:
        return ExecutionOutcome(OutcomeCategory.BUDGET_EXHAUSTED, is_usable=False)
    return ExecutionOutcome(OutcomeCategory.GENERAL_ERROR, is_usable=False)


def _is_prefill_dominated(trace: ExecutionTrace) -> bool:
    """Return True when the trace shows signs of prefill-dominated timing.

    A step is prefill-dominated when ``ttft_ms / (duration * 1000) >
    _PREFILL_RATIO`` (default 0.7 — 70% of wall-time spent on prefill).
    A run is prefill-dominated when at least ``_MIN_PREFILL_STEPS``
    (default 2) steps satisfy this — two or more prefill-heavy steps
    is the signature of prefix-cache failure or runaway history
    replay, distinct from a single cold-start TTFT spike.
    """
    prefill_steps = 0
    for step in trace.steps:
        if step.ttft_ms is None or step.duration <= 0:
            continue
        duration_ms = step.duration * 1000
        if duration_ms <= 0:
            continue
        if (step.ttft_ms / duration_ms) > _PREFILL_RATIO:
            prefill_steps += 1
            if prefill_steps >= _MIN_PREFILL_STEPS:
                return True
    return False


# Keywords that indicate budget/step exhaustion in degraded warnings.
# "step" alone is too broad — it false-positives on unrelated text
# like "Follow these steps to recover."
_BUDGET_KEYWORDS = ("budget", "step budget", "steps used")


def _classify_degraded(answer: str) -> ExecutionOutcome:
    """Classify a degraded outcome (success=True, answer starts with warning prefix).

    Keyword priority matches _classify_hard_failure: timeout > context
    overflow > budget > unrecognized (treated as usable success with log).
    """
    answer_lower = answer.lower()
    if "timed out" in answer_lower:
        return ExecutionOutcome(OutcomeCategory.TIMEOUT, is_usable=False)
    if "context window" in answer_lower:
        return ExecutionOutcome(OutcomeCategory.CONTEXT_OVERFLOW, is_usable=False)
    if any(kw in answer_lower for kw in _BUDGET_KEYWORDS):
        return ExecutionOutcome(OutcomeCategory.BUDGET_EXHAUSTED, is_usable=False)
    if "execution error" in answer_lower or "cannot connect" in answer_lower:
        return ExecutionOutcome(OutcomeCategory.GENERAL_ERROR, is_usable=False)
    # Unrecognized ⚠️ prefix — treat as usable but log so new warning
    # types are surfaced during development.
    _logger.warning(
        "Unrecognized degraded warning prefix (treating as usable): %.200s",
        answer,
    )
    return ExecutionOutcome(OutcomeCategory.SUCCESS, is_usable=True)
