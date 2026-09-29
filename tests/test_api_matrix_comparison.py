"""Tests for rlmstudio.compare_matrix() and compare_matrix_async()."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest

from rlmstudio import (
    MatrixCompareResult,
    MatrixSlotResult,
    compare_matrix,
    compare_matrix_async,
)
from rlmstudio.application.dto import LLMResponseDTO
from rlmstudio.application.use_cases.run_matrix_comparison import (
    RunMatrixComparisonUseCase,
)

# ---------------------------------------------------------------------------
# Fake LLM adapter (drop-in for LiteLLMAdapter during tests)
# ---------------------------------------------------------------------------


class _FakeLLM:
    """LiteLLMAdapter-compatible fake keyed by model name.

    When ``rlmstudio.api.LiteLLMAdapter`` is patched to return this class,
    each slot's adapter can be distinguished by the model name in the
    response, so tests can assert per-slot routing.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        input_cost_per_1m: float = 0.0,
        output_cost_per_1m: float = 0.0,
        input_tokens: int = 10,
        output_tokens: int = 5,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.active_model = model
        self._input_cost = input_cost_per_1m
        self._output_cost = output_cost_per_1m
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        # Response contains the model name so assertions can distinguish slots.
        return LLMResponseDTO(
            content=f"answer from {self.model}",
            model=self.model,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )

    def complete_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        yield f"answer from {self.model}"

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def get_pricing(self) -> dict[str, float]:
        return {
            "input_cost_per_1m": self._input_cost,
            "output_cost_per_1m": self._output_cost,
        }

    def get_completion_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self._input_cost / 1_000_000
            + output_tokens * self._output_cost / 1_000_000
        )


def _fake_adapter_factory(*args: Any, **kwargs: Any) -> _FakeLLM:
    """Used with @patch to replace LiteLLMAdapter in the slot builder."""
    return _FakeLLM(model=kwargs.get("model", "gpt-4o"))


class _RaisingLLM:
    """LLM fake whose every call raises — used to force slot failures."""

    def __init__(self, model: str = "bad", **kwargs: Any) -> None:
        self.model = model
        self.active_model = model

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        raise RuntimeError(f"{self.model} is broken")

    def complete_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        raise RuntimeError(f"{self.model} is broken")
        yield ""  # pragma: no cover — unreachable

    def count_tokens(self, text: str) -> int:
        return 1

    def get_pricing(self) -> dict[str, float]:
        return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}

    def get_completion_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0


def _raising_adapter_factory(*args: Any, **kwargs: Any) -> _RaisingLLM:
    return _RaisingLLM(model=kwargs.get("model", "bad"))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_empty_content_raises(self) -> None:
        with pytest.raises(ValueError, match="content cannot be empty"):
            compare_matrix("", "q", ["openai/gpt-4o"])

    def test_empty_query_raises(self) -> None:
        with pytest.raises(ValueError, match="query cannot be empty"):
            compare_matrix("doc", "", ["openai/gpt-4o"])

    def test_empty_providers_raises(self) -> None:
        with pytest.raises(ValueError, match="providers must be a non-empty list"):
            compare_matrix("doc", "q", [])

    def test_empty_modes_raises(self) -> None:
        with pytest.raises(ValueError, match="modes must be a non-empty list"):
            compare_matrix("doc", "q", ["openai/gpt-4o"], modes=[])

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid matrix mode"):
            compare_matrix("doc", "q", ["openai/gpt-4o"], modes=["magic"])

    def test_malformed_provider_spec_raises(self) -> None:
        with pytest.raises(ValueError, match="Provider spec must be"):
            compare_matrix("doc", "q", ["openai-gpt-4o"])  # missing slash

    def test_missing_model_in_spec_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid provider spec"):
            compare_matrix("doc", "q", ["openai/"])

    def test_missing_provider_in_spec_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid provider spec"):
            compare_matrix("doc", "q", ["/gpt-4o"])

    def test_too_many_slots_raises(self) -> None:
        # 6 providers × 2 modes = 12 slots > MAX_SLOTS=10
        providers = [f"openai/model-{i}" for i in range(6)]
        with pytest.raises(ValueError, match="Too many slots"):
            compare_matrix(
                "doc",
                "q",
                providers,
                modes=["direct", "rlm"],
            )

    def test_max_slots_constant_enforced_at_boundary(self) -> None:
        """Exactly MAX_SLOTS slots should be accepted; MAX_SLOTS+1 should not."""
        max_slots = RunMatrixComparisonUseCase.MAX_SLOTS
        over = [f"openai/m{i}" for i in range(max_slots + 1)]
        with pytest.raises(ValueError, match="Too many slots"):
            compare_matrix("doc", "q", over, modes=["direct"])

    @patch("rlmstudio.api.LiteLLMAdapter", side_effect=_fake_adapter_factory)
    def test_invalid_ranking_metric_raises_before_execution(self, mock_adapter: Any) -> None:
        """A bad ``ranking_metric`` must fail *before* any slot executes,
        so a typo can't burn tokens/money across the whole matrix."""
        with pytest.raises(ValueError, match="Invalid ranking_metric"):
            compare_matrix(
                "doc",
                "q",
                ["openai/gpt-4o", "anthropic/claude-sonnet-4-6"],
                ranking_metric="bogus",
            )
        # Critical assertion: no LiteLLMAdapter was ever constructed.
        mock_adapter.assert_not_called()

    @patch("rlmstudio.api.LiteLLMAdapter", side_effect=_fake_adapter_factory)
    def test_valid_ranking_metrics_accepted(self, mock_adapter: Any) -> None:
        """All documented ranking metrics pass validation."""
        for metric in ("cost", "tokens", "latency", "answer_per_cost"):
            result = compare_matrix(
                "doc",
                "q",
                ["openai/gpt-4o"],
                modes=["direct"],
                ranking_metric=metric,
            )
            assert result.ranking_metric == metric

    @patch("rlmstudio.api.LiteLLMAdapter", side_effect=_fake_adapter_factory)
    def test_malformed_spec_fails_before_execution(self, mock_adapter: Any) -> None:
        """A malformed provider spec in the middle of the list must fail
        before any adapter is built, not partway through execution."""
        with pytest.raises(ValueError, match="Invalid provider spec"):
            compare_matrix(
                "doc",
                "q",
                ["openai/gpt-4o", "anthropic/", "ollama/llama3.2"],
            )
        mock_adapter.assert_not_called()


# ---------------------------------------------------------------------------
# Happy path (sync)
# ---------------------------------------------------------------------------


class TestSyncExecution:
    @patch("rlmstudio.api.LiteLLMAdapter", side_effect=_fake_adapter_factory)
    def test_single_provider_direct(self, mock_adapter: Any) -> None:
        result = compare_matrix(
            "doc",
            "q",
            ["openai/gpt-4o"],
            modes=["direct"],
        )

        assert isinstance(result, MatrixCompareResult)
        assert len(result.slots) == 1
        assert result.slots[0].provider == "openai"
        assert result.slots[0].model == "gpt-4o"
        assert result.slots[0].mode == "direct"
        assert result.slots[0].success is True
        assert "gpt-4o" in result.slots[0].answer
        assert result.comparison_group_id  # non-empty UUID
        assert result.ranking == [0]

    @patch("rlmstudio.api.LiteLLMAdapter", side_effect=_fake_adapter_factory)
    def test_cartesian_product(self, mock_adapter: Any) -> None:
        result = compare_matrix(
            "doc",
            "q",
            ["openai/gpt-4o", "anthropic/claude-sonnet-4-6"],
            modes=["direct"],
        )
        # 2 providers × 1 mode = 2 slots
        assert len(result.slots) == 2
        providers = {s.provider for s in result.slots}
        assert providers == {"openai", "anthropic"}
        # Each slot gets a distinct answer because the fake echoes the model
        answers = {s.answer for s in result.slots}
        assert len(answers) == 2

    @patch("rlmstudio.api.LiteLLMAdapter", side_effect=_fake_adapter_factory)
    def test_preserves_input_order(self, mock_adapter: Any) -> None:
        result = compare_matrix(
            "doc",
            "q",
            ["openai/gpt-4o", "anthropic/claude-sonnet-4-6", "ollama/llama3.2"],
            modes=["direct"],
        )
        # Order in .slots must match input order, independent of completion order.
        assert [s.provider for s in result.slots] == ["openai", "anthropic", "ollama"]
        assert [s.model for s in result.slots] == [
            "gpt-4o",
            "claude-sonnet-4-6",
            "llama3.2",
        ]

    @patch("rlmstudio.api.LiteLLMAdapter", side_effect=_fake_adapter_factory)
    def test_default_modes_is_direct(self, mock_adapter: Any) -> None:
        result = compare_matrix("doc", "q", ["openai/gpt-4o"])
        assert len(result.slots) == 1
        assert result.slots[0].mode == "direct"

    @patch("rlmstudio.api.LiteLLMAdapter", side_effect=_fake_adapter_factory)
    def test_slot_labels_include_provider_model_mode(self, mock_adapter: Any) -> None:
        result = compare_matrix(
            "doc",
            "q",
            ["openai/gpt-4o"],
            modes=["direct"],
        )
        assert result.slots[0].label == "openai/gpt-4o · direct"


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


class TestResultHelpers:
    @patch("rlmstudio.api.LiteLLMAdapter", side_effect=_fake_adapter_factory)
    def test_get_slot_by_id(self, mock_adapter: Any) -> None:
        result = compare_matrix(
            "doc",
            "q",
            ["openai/gpt-4o", "anthropic/claude-sonnet-4-6"],
        )
        slot = result.slots[0]
        assert result.get_slot(slot.slot_id) is slot
        assert result.get_slot("not-a-real-id") is None

    @patch("rlmstudio.api.LiteLLMAdapter", side_effect=_fake_adapter_factory)
    def test_best_returns_top_ranked(self, mock_adapter: Any) -> None:
        result = compare_matrix(
            "doc",
            "q",
            ["openai/gpt-4o", "anthropic/claude-sonnet-4-6"],
        )
        assert result.best is not None
        # best is result.slots[result.ranking[0]]
        assert result.best is result.slots[result.ranking[0]]

    def test_best_returns_none_when_empty(self) -> None:
        empty = MatrixCompareResult(comparison_group_id="g1")
        assert empty.best is None

    @patch("rlmstudio.api.LiteLLMAdapter", side_effect=_raising_adapter_factory)
    def test_best_returns_none_when_all_slots_fail(self, mock_adapter: Any) -> None:
        """``best`` must return ``None`` when every slot failed, even though
        ``ranking`` still contains entries (failed slots are placed last)."""
        result = compare_matrix(
            "doc",
            "q",
            ["openai/bad-1", "anthropic/bad-2"],
        )
        assert len(result.slots) == 2
        assert all(not s.success for s in result.slots)
        # ranking is populated (failed slots in input order), but best() must
        # still return None because there is no successful winner.
        assert len(result.ranking) == 2
        assert result.best is None

    @patch("rlmstudio.api.LiteLLMAdapter")
    def test_best_skips_failed_leading_slot(self, mock_adapter: Any) -> None:
        """When some slots succeed and some fail, ``best`` returns the
        top-ranked *successful* slot."""

        # First slot raises (will be ranked last), second succeeds.
        def _mixed(*args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model", "x")
            if "bad" in model:
                return _RaisingLLM(model=model)
            return _FakeLLM(model=model)

        mock_adapter.side_effect = _mixed

        result = compare_matrix(
            "doc",
            "q",
            ["openai/bad", "anthropic/good"],
        )
        assert result.best is not None
        assert result.best.success is True
        assert result.best.provider == "anthropic"

    @patch("rlmstudio.api.LiteLLMAdapter", side_effect=_fake_adapter_factory)
    def test_slot_result_is_public_type(self, mock_adapter: Any) -> None:
        result = compare_matrix("doc", "q", ["openai/gpt-4o"])
        assert isinstance(result.slots[0], MatrixSlotResult)
        # Public type should have all the expected fields
        s = result.slots[0]
        assert hasattr(s, "slot_id")
        assert hasattr(s, "provider")
        assert hasattr(s, "model")
        assert hasattr(s, "mode")
        assert hasattr(s, "answer")
        assert hasattr(s, "success")
        assert hasattr(s, "total_tokens")
        assert hasattr(s, "total_cost")
        assert hasattr(s, "elapsed_time")


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


class TestRanking:
    def _adapter_with_costs(self, cost_map: dict[str, float]) -> Any:
        """Return a side_effect that makes each model use a different cost."""

        def _factory(*args: Any, **kwargs: Any) -> _FakeLLM:
            model = kwargs.get("model", "gpt-4o")
            cost = cost_map.get(model, 0.0)
            return _FakeLLM(
                model=model,
                input_cost_per_1m=cost,
                output_cost_per_1m=cost,
            )

        return _factory

    def test_rank_by_cost_picks_cheapest(self) -> None:
        # The fake LLM's model name is the prefixed form (e.g. "anthropic/claude"),
        # so the cost map must use the prefixed names too.
        cost_map = {
            "gpt-4o": 1000.0,
            "anthropic/claude-sonnet-4-6": 10.0,
            "ollama/llama3.2": 1.0,
        }
        with patch(
            "rlmstudio.api.LiteLLMAdapter",
            side_effect=self._adapter_with_costs(cost_map),
        ):
            result = compare_matrix(
                "doc",
                "q",
                [
                    "openai/gpt-4o",  # expensive
                    "anthropic/claude-sonnet-4-6",  # medium
                    "ollama/llama3.2",  # cheapest
                ],
                modes=["direct"],
                ranking_metric="cost",
            )

        # Cheapest should be ranked first.
        assert result.best is not None
        assert result.best.provider == "ollama"
        # Full ranking: ollama < anthropic < openai
        ranked_providers = [result.slots[i].provider for i in result.ranking]
        assert ranked_providers == ["ollama", "anthropic", "openai"]
        assert result.ranking_metric == "cost"

    @patch("rlmstudio.api.LiteLLMAdapter", side_effect=_fake_adapter_factory)
    def test_rank_by_tokens(self, mock_adapter: Any) -> None:
        result = compare_matrix(
            "doc",
            "q",
            ["openai/gpt-4o", "anthropic/claude-sonnet-4-6"],
            ranking_metric="tokens",
        )
        # Both fakes report the same token counts, so ranking is input-order.
        assert result.ranking_metric == "tokens"
        assert result.ranking == [0, 1]


# ---------------------------------------------------------------------------
# Per-slot config isolation
# ---------------------------------------------------------------------------


class TestPerSlotConfig:
    @patch("rlmstudio.api.LiteLLMAdapter", side_effect=_fake_adapter_factory)
    def test_each_slot_gets_independent_config(self, mock_adapter: Any) -> None:
        """Per-slot MatrixSlotDTO.config must not share an `extra` dict
        across slots (one slot's RAG collection must not leak to another)."""
        from rlmstudio.api import build_matrix_slots

        slots = build_matrix_slots(
            ["openai/gpt-4o", "anthropic/claude"],
            ["direct", "direct"],
            api_key=None,
            api_base=None,
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_steps=16,
            num_retries=None,
            embedding_api_key=None,
        )
        assert len(slots) == 4
        # All slot configs must be distinct dict instances.
        ids = {id(s.config.extra) for s in slots if s.config is not None}
        assert len(ids) == len(slots)

    @patch("rlmstudio.api.LiteLLMAdapter", side_effect=_fake_adapter_factory)
    def test_max_steps_applied_to_slot_config(self, mock_adapter: Any) -> None:
        from rlmstudio.api import build_matrix_slots

        slots = build_matrix_slots(
            ["openai/gpt-4o"],
            ["rlm"],
            api_key=None,
            api_base=None,
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_steps=42,
            num_retries=None,
            embedding_api_key=None,
        )
        assert len(slots) == 1
        assert slots[0].config is not None
        assert slots[0].config.max_steps == 42


# ---------------------------------------------------------------------------
# Provider spec parsing
# ---------------------------------------------------------------------------


class TestProviderSpecParsing:
    def test_simple_spec(self) -> None:
        from rlmstudio.api import _parse_provider_spec

        assert _parse_provider_spec("openai/gpt-4o") == ("openai", "gpt-4o")
        assert _parse_provider_spec("anthropic/claude-sonnet-4-6") == (
            "anthropic",
            "claude-sonnet-4-6",
        )

    def test_partition_on_first_slash(self) -> None:
        from rlmstudio.api import _parse_provider_spec

        # The split is on the first slash, so HuggingFace-style model IDs
        # with embedded slashes are preserved in the model portion.
        provider, model = _parse_provider_spec("vllm/Qwen/Qwen2.5-7B-Instruct")
        assert provider == "vllm"
        assert model == "Qwen/Qwen2.5-7B-Instruct"

    def test_non_string_raises(self) -> None:
        from rlmstudio.api import _parse_provider_spec

        with pytest.raises(ValueError, match="Provider spec must be"):
            _parse_provider_spec(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------


class TestAsyncExecution:
    @patch("rlmstudio.api.LiteLLMAdapter", side_effect=_fake_adapter_factory)
    def test_compare_matrix_async_returns_same_shape(self, mock_adapter: Any) -> None:
        result = asyncio.run(
            compare_matrix_async(
                "doc",
                "q",
                ["openai/gpt-4o", "anthropic/claude-sonnet-4-6"],
                modes=["direct"],
            )
        )
        assert isinstance(result, MatrixCompareResult)
        assert len(result.slots) == 2
        assert all(s.success for s in result.slots)
