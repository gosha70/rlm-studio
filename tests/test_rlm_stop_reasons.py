"""A stopped RLM run names the limit it hit and is not a success (M1).

Before: every ``BudgetExceededError`` was reported as "Step budget exhausted"
with a hint at a Chat Provider field that does not exist, and returned
``success=True``.
"""

from __future__ import annotations

import time

from rlmstudio.application.dto import LLMResponseDTO, RunConfigDTO
from rlmstudio.application.services.outcome_classifier import (
    OutcomeCategory,
    classify_execution_outcome,
)
from rlmstudio.application.use_cases.run_rlm import RunRLMUseCase
from tests.test_use_cases import FakeLLM, FakeSandbox

_CODE_STEP = '```python\nprint("step")\n```'


def _category(result) -> OutcomeCategory:
    return classify_execution_outcome(
        success=result.success, error=result.error, answer=result.answer
    ).category


class _SlowLLM(FakeLLM):
    """Each call takes a little wall-clock time, so a tiny time cap is hit."""

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        time.sleep(0.05)
        return super().complete(messages)


class TestBudgetStopReason:
    def test_token_cap_names_tokens_and_the_budget_setting(self) -> None:
        # 15 tokens per call; the cap is reached after the second call.
        llm = FakeLLM([_CODE_STEP])
        config = RunConfigDTO(mode="rlm", max_steps=32, max_tokens=20)
        result = RunRLMUseCase(llm, FakeSandbox()).execute("content", "q", config=config)

        assert result.success is False
        assert result.answer.startswith("⚠️ **Token budget exceeded** after 2 steps")
        assert "30 of 20 tokens" in result.answer
        assert "Settings → Budget" in result.answer
        assert "Step budget" not in result.answer
        assert _category(result) is OutcomeCategory.BUDGET_EXHAUSTED

    def test_time_cap_is_a_timeout(self) -> None:
        llm = _SlowLLM([_CODE_STEP])
        config = RunConfigDTO(mode="rlm", max_steps=32, max_time_seconds=0.01)
        result = RunRLMUseCase(llm, FakeSandbox()).execute("content", "q", config=config)

        assert result.success is False
        assert result.answer.startswith("⚠️ **Execution timed out**")
        assert "Max Execution Time (s)" in result.answer
        assert _category(result) is OutcomeCategory.TIMEOUT

    def test_step_hint_points_at_the_profile_not_chat_providers(self) -> None:
        llm = FakeLLM([_CODE_STEP])
        config = RunConfigDTO(mode="rlm", max_steps=2)
        result = RunRLMUseCase(llm, FakeSandbox()).execute("content", "q", config=config)

        assert "Step budget exhausted" in result.answer
        assert "Settings → Profiles" in result.answer
        assert "Chat Providers" not in result.answer
