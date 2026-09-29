"""Tests for RunRLMOfficialUseCase (specs/interop-official-rlm FR-4).

The engine is a scripted fake; every scenario asserts the outcome category
the rest of Studio derives from the result, not just the raw fields.
"""

from __future__ import annotations

import asyncio
from typing import Any

from rlmstudio.application.dto import RunConfigDTO
from rlmstudio.application.sandbox_vars import MODE_RLM_OFFICIAL, RESULT_KEY_ENGINE_VERSION
from rlmstudio.application.services.outcome_classifier import (
    OutcomeCategory,
    classify_execution_outcome,
)
from rlmstudio.application.use_cases.run_rlm_official import RunRLMOfficialUseCase
from tests.fakes.fake_rlm_engine import FAKE_ENGINE_VERSION, FakeRLMEngine

# Long enough that a wall-clock budget of 0.05s reliably fires first, short
# enough not to slow the suite.
_SLOW = 0.5
_TIGHT_BUDGET = 0.05


def _category(result: Any) -> OutcomeCategory:
    return classify_execution_outcome(result.success, result.error, result.answer).category


class TestHappyPath:
    def test_result_passes_through_and_is_stamped(self) -> None:
        engine = FakeRLMEngine(answer="the answer")
        result = RunRLMOfficialUseCase(engine).execute("doc", "q", RunConfigDTO(max_steps=7))

        assert result.success
        assert result.answer == "the answer"
        assert result.mode_used == MODE_RLM_OFFICIAL
        assert (result.steps, result.input_tokens, result.output_tokens) == (3, 220, 15)
        assert result.total_cost == 0.0012
        assert len(result.trace) == 3
        assert result.metadata[RESULT_KEY_ENGINE_VERSION] == FAKE_ENGINE_VERSION
        assert _category(result) is OutcomeCategory.SUCCESS

    def test_engine_receives_content_query_and_config(self) -> None:
        engine = FakeRLMEngine()
        config = RunConfigDTO(mode=MODE_RLM_OFFICIAL, max_steps=7, max_recursion_depth=2)
        RunRLMOfficialUseCase(engine).execute("doc", "q", config)

        assert engine.calls == [("doc", "q", config)]

    def test_default_config_uses_the_official_mode(self) -> None:
        engine = FakeRLMEngine()
        RunRLMOfficialUseCase(engine).execute("doc", "q")

        assert engine.calls[0][2].mode == MODE_RLM_OFFICIAL

    def test_elapsed_is_measured_when_engine_reports_none(self) -> None:
        result = RunRLMOfficialUseCase(FakeRLMEngine()).execute("doc", "q")

        assert result.elapsed_time >= 0.0


class TestUnavailableEngine:
    def test_reports_reason_as_general_error(self) -> None:
        engine = FakeRLMEngine(available=False, reason="install rlm-studio[interop]", version=None)
        result = RunRLMOfficialUseCase(engine).execute("doc", "q")

        assert not result.success
        assert result.error is not None
        assert "install rlm-studio[interop]" in result.error
        assert result.answer.startswith("⚠️")
        assert result.metadata[RESULT_KEY_ENGINE_VERSION] is None
        assert _category(result) is OutcomeCategory.GENERAL_ERROR
        assert engine.calls == []


class TestEngineError:
    def test_exception_becomes_failed_result(self) -> None:
        engine = FakeRLMEngine(error=RuntimeError("backend exploded"))
        result = RunRLMOfficialUseCase(engine).execute("doc", "q")

        assert not result.success
        assert result.error == "backend exploded"
        assert result.answer.startswith("⚠️")
        assert _category(result) is OutcomeCategory.GENERAL_ERROR


class TestWallClockBudget:
    def test_slow_engine_is_classified_timeout(self) -> None:
        engine = FakeRLMEngine(delay=_SLOW)
        config = RunConfigDTO(max_time_seconds=_TIGHT_BUDGET)
        result = RunRLMOfficialUseCase(engine).execute("doc", "q", config)

        assert not result.success
        assert result.elapsed_time < _SLOW
        assert result.answer.startswith("⚠️")
        assert _category(result) is OutcomeCategory.TIMEOUT

    def test_no_budget_means_no_timeout(self) -> None:
        engine = FakeRLMEngine(delay=_TIGHT_BUDGET)
        result = RunRLMOfficialUseCase(engine).execute(
            "doc", "q", RunConfigDTO(max_time_seconds=None)
        )

        assert result.success

    def test_engine_error_inside_budget_still_surfaces(self) -> None:
        engine = FakeRLMEngine(error=ValueError("bad prompt"))
        result = RunRLMOfficialUseCase(engine).execute("doc", "q", RunConfigDTO(max_time_seconds=5))

        assert not result.success
        assert result.error == "bad prompt"


class TestPostRunCaps:
    def test_token_cap_breach_is_degraded_budget_outcome(self) -> None:
        engine = FakeRLMEngine(answer="real answer", input_tokens=900, output_tokens=200)
        result = RunRLMOfficialUseCase(engine).execute("doc", "q", RunConfigDTO(max_tokens=1000))

        assert result.success  # the engine finished; the answer is real
        assert result.answer.startswith("⚠️")
        assert "real answer" in result.answer
        assert _category(result) is OutcomeCategory.BUDGET_EXHAUSTED

    def test_cost_cap_breach_is_degraded_budget_outcome(self) -> None:
        engine = FakeRLMEngine(total_cost=0.5)
        result = RunRLMOfficialUseCase(engine).execute("doc", "q", RunConfigDTO(max_cost=0.1))

        assert result.success
        assert _category(result) is OutcomeCategory.BUDGET_EXHAUSTED

    def test_within_caps_leaves_answer_untouched(self) -> None:
        engine = FakeRLMEngine(answer="clean", input_tokens=10, output_tokens=5, total_cost=0.01)
        config = RunConfigDTO(max_tokens=1000, max_cost=1.0)
        result = RunRLMOfficialUseCase(engine).execute("doc", "q", config)

        assert result.answer == "clean"
        assert _category(result) is OutcomeCategory.SUCCESS


class _RecordingEmitter:
    def __init__(self) -> None:
        self.tokens: list[str] = []
        self.steps: list[dict[str, Any]] = []
        self.metrics: list[dict[str, Any]] = []

    async def on_token(self, token: str) -> None:
        self.tokens.append(token)

    async def on_step(self, step_data: dict[str, Any]) -> None:
        self.steps.append(step_data)

    async def on_metrics(self, metrics: dict[str, Any]) -> None:
        self.metrics.append(metrics)


class TestAsync:
    def test_happy_path_emits_steps_and_metrics(self) -> None:
        engine = FakeRLMEngine()
        emitter = _RecordingEmitter()
        result = asyncio.run(
            RunRLMOfficialUseCase(engine).execute_async("doc", "q", event_emitter=emitter)
        )

        assert result.success
        assert result.mode_used == MODE_RLM_OFFICIAL
        assert len(emitter.steps) == 3
        assert emitter.tokens == []  # the engine does not stream
        assert emitter.metrics == [
            {
                "input_tokens": 220,
                "output_tokens": 15,
                "total_tokens": 235,
                "cost_usd": 0.0012,
                "steps": 3,
                "elapsed_seconds": result.elapsed_time,
            }
        ]

    def test_slow_engine_is_classified_timeout(self) -> None:
        engine = FakeRLMEngine(delay=_SLOW)
        config = RunConfigDTO(max_time_seconds=_TIGHT_BUDGET)
        result = asyncio.run(RunRLMOfficialUseCase(engine).execute_async("doc", "q", config))

        assert not result.success
        assert result.elapsed_time < _SLOW
        assert _category(result) is OutcomeCategory.TIMEOUT

    def test_unavailable_engine_short_circuits(self) -> None:
        engine = FakeRLMEngine(available=False, reason="nope")
        result = asyncio.run(RunRLMOfficialUseCase(engine).execute_async("doc", "q"))

        assert not result.success
        assert engine.calls == []

    def test_engine_error_becomes_failed_result(self) -> None:
        engine = FakeRLMEngine(error=RuntimeError("async boom"))
        result = asyncio.run(RunRLMOfficialUseCase(engine).execute_async("doc", "q"))

        assert not result.success
        assert result.error == "async boom"
