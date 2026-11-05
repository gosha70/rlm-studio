"""Tests for the outcome classifier — single source of truth for failure categorization."""

import pytest

from rlmstudio.application.services.outcome_classifier import (
    ExecutionOutcome,
    OutcomeCategory,
    classify_execution_outcome,
)

# ---------------------------------------------------------------------------
# Hard failures (success=False)
# ---------------------------------------------------------------------------


class TestHardFailureClassification:
    """Tests for success=False outcomes classified from error strings."""

    def test_timeout_from_error(self) -> None:
        result = classify_execution_outcome(success=False, error="Request timeout", answer="")
        assert result == ExecutionOutcome(OutcomeCategory.TIMEOUT, is_usable=False)

    def test_timed_out_from_error(self) -> None:
        result = classify_execution_outcome(
            success=False, error="LLM request timed out after 120s", answer=""
        )
        assert result == ExecutionOutcome(OutcomeCategory.TIMEOUT, is_usable=False)

    def test_context_window_from_error(self) -> None:
        result = classify_execution_outcome(
            success=False, error="context window exceeded", answer=""
        )
        assert result == ExecutionOutcome(OutcomeCategory.CONTEXT_OVERFLOW, is_usable=False)

    def test_context_length_from_error(self) -> None:
        result = classify_execution_outcome(
            success=False, error="context_length_exceeded", answer=""
        )
        assert result == ExecutionOutcome(OutcomeCategory.CONTEXT_OVERFLOW, is_usable=False)

    def test_budget_from_error(self) -> None:
        result = classify_execution_outcome(success=False, error="Token budget exceeded", answer="")
        assert result == ExecutionOutcome(OutcomeCategory.BUDGET_EXHAUSTED, is_usable=False)

    def test_general_error_fallback(self) -> None:
        result = classify_execution_outcome(
            success=False, error="Something unexpected happened", answer=""
        )
        assert result == ExecutionOutcome(OutcomeCategory.GENERAL_ERROR, is_usable=False)

    def test_none_error_is_general_error(self) -> None:
        result = classify_execution_outcome(success=False, error=None, answer="")
        assert result == ExecutionOutcome(OutcomeCategory.GENERAL_ERROR, is_usable=False)

    def test_empty_error_is_general_error(self) -> None:
        result = classify_execution_outcome(success=False, error="", answer="")
        assert result == ExecutionOutcome(OutcomeCategory.GENERAL_ERROR, is_usable=False)

    def test_all_hard_failures_are_not_usable(self) -> None:
        errors = [
            "timeout",
            "timed out",
            "context window",
            "budget",
            "random error",
            None,
            "",
        ]
        for error in errors:
            result = classify_execution_outcome(success=False, error=error, answer="")
            assert not result.is_usable, f"Expected is_usable=False for error={error!r}"


# ---------------------------------------------------------------------------
# Degraded outcomes (success=True, answer starts with ⚠️)
# ---------------------------------------------------------------------------


class TestDegradedClassification:
    """Tests for success=True outcomes with warning answer prefix."""

    def test_degraded_timeout(self) -> None:
        result = classify_execution_outcome(
            success=True,
            error=None,
            answer="⚠️ **Execution timed out** after 5 steps.\n\nPartial results...",
        )
        assert result == ExecutionOutcome(OutcomeCategory.TIMEOUT, is_usable=False)

    def test_degraded_context_window(self) -> None:
        result = classify_execution_outcome(
            success=True,
            error=None,
            answer="⚠️ **Context window exceeded** at step 3.\n\nPartial answer...",
        )
        assert result == ExecutionOutcome(OutcomeCategory.CONTEXT_OVERFLOW, is_usable=False)

    def test_degraded_budget_exhausted(self) -> None:
        result = classify_execution_outcome(
            success=True,
            error=None,
            answer="⚠️ **Token budget exceeded** after 4 steps.\n\nPartial...",
        )
        assert result == ExecutionOutcome(OutcomeCategory.BUDGET_EXHAUSTED, is_usable=False)

    def test_degraded_step_budget(self) -> None:
        result = classify_execution_outcome(
            success=True,
            error=None,
            answer="⚠️ **Step budget exhausted** (16/16 steps used).\n\nBest answer...",
        )
        assert result == ExecutionOutcome(OutcomeCategory.BUDGET_EXHAUSTED, is_usable=False)

    def test_degraded_lmm_timeout(self) -> None:
        result = classify_execution_outcome(
            success=True,
            error=None,
            answer="⚠️ **LLM request timed out** after 3 steps.\n\nNo answer...",
        )
        assert result == ExecutionOutcome(OutcomeCategory.TIMEOUT, is_usable=False)

    def test_degraded_unknown_warning_is_success(self) -> None:
        """A ⚠️ prefix without known keywords is treated as success."""
        result = classify_execution_outcome(
            success=True,
            error=None,
            answer="⚠️ Warning: FINAL provided after execution failure. Using partial...",
        )
        # This doesn't match timeout/context/budget/step keywords specifically
        # but it does contain no known failure keywords → success
        assert result.is_usable is True

    def test_degraded_rag_timeout(self) -> None:
        result = classify_execution_outcome(
            success=True,
            error=None,
            answer="⚠️ **RAG request timed out**.\n\nPlease try again.",
        )
        assert result == ExecutionOutcome(OutcomeCategory.TIMEOUT, is_usable=False)

    def test_degraded_direct_timeout(self) -> None:
        result = classify_execution_outcome(
            success=True,
            error=None,
            answer="⚠️ **LLM request timed out**.\n\nThe provider did not respond...",
        )
        assert result == ExecutionOutcome(OutcomeCategory.TIMEOUT, is_usable=False)


# ---------------------------------------------------------------------------
# Success outcomes (success=True, normal answer)
# ---------------------------------------------------------------------------


class TestSuccessClassification:
    """Tests for genuine success outcomes."""

    def test_normal_success(self) -> None:
        result = classify_execution_outcome(
            success=True, error=None, answer="Here is my answer to your question."
        )
        assert result == ExecutionOutcome(OutcomeCategory.SUCCESS, is_usable=True)

    def test_empty_answer_success(self) -> None:
        result = classify_execution_outcome(success=True, error=None, answer="")
        assert result == ExecutionOutcome(OutcomeCategory.SUCCESS, is_usable=True)

    def test_success_with_error_none(self) -> None:
        result = classify_execution_outcome(
            success=True, error=None, answer="Valid response content"
        )
        assert result.is_usable is True
        assert result.category == OutcomeCategory.SUCCESS


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for the classifier."""

    def test_case_insensitive_error_matching(self) -> None:
        result = classify_execution_outcome(
            success=False, error="REQUEST TIMEOUT exceeded", answer=""
        )
        assert result.category == OutcomeCategory.TIMEOUT

    def test_case_insensitive_degraded_matching(self) -> None:
        result = classify_execution_outcome(
            success=True,
            error=None,
            answer="⚠️ CONTEXT WINDOW issue detected",
        )
        assert result.category == OutcomeCategory.CONTEXT_OVERFLOW

    def test_outcome_is_frozen(self) -> None:
        result = classify_execution_outcome(success=True, error=None, answer="ok")
        with pytest.raises(AttributeError):
            result.is_usable = False  # type: ignore[misc]

    def test_all_categories_have_distinct_values(self) -> None:
        values = [c.value for c in OutcomeCategory]
        assert len(values) == len(set(values))


class TestContextLengthWording:
    """M6: LM Studio reports an overflow as "greater than the context length"."""

    def test_lm_studio_context_length_is_an_overflow(self) -> None:
        error = (
            "LiteLLM completion failed (after streaming fallback): litellm.BadRequestError: "
            "OpenAIException - Error code: 400 - {'error': 'The number of tokens to keep "
            "from the initial prompt is greater than the context length. Try to load the "
            "model with a larger context length, or provide a shorter input'}"
        )
        result = classify_execution_outcome(success=False, error=error, answer="")
        assert result.category == OutcomeCategory.CONTEXT_OVERFLOW
