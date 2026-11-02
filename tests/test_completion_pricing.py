"""Unit tests for the trace-view completion pricing (#94)."""

from __future__ import annotations

from rlmstudio.infrastructure.llm.pricing import completion_cost


def test_a_known_model_is_priced_from_its_token_counts() -> None:
    cost = completion_cost("gpt-4o", 1000, 500)

    assert cost is not None
    assert cost > 0


def test_a_prefixed_model_name_is_also_priced() -> None:
    """Trace rows carry the adapter's routing prefix (anthropic/…)."""
    cost = completion_cost("anthropic/claude-sonnet-4-6", 1000, 500)

    assert cost is not None
    assert cost > 0


def test_a_zero_token_row_is_a_genuine_zero() -> None:
    """Subcall and runtime-fingerprint rows carry no tokens of their own."""
    assert completion_cost("gpt-4o", 0, 0) == 0.0
    assert completion_cost(None, 0, 0) == 0.0


def test_an_unpriced_model_is_unavailable_not_free() -> None:
    assert completion_cost("some-model-the-cost-map-has-never-heard-of", 100, 50) is None
    assert completion_cost(None, 100, 50) is None
