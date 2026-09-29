"""Scripted :class:`RLMEnginePort` fake shared by use-case and API tests."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from rlmstudio.application.dto import RunConfigDTO, RunResultDTO
from rlmstudio.application.sandbox_vars import (
    MODE_RLM_OFFICIAL,
    TRACE_KEY_CODE,
    TRACE_KEY_CONTENT,
    TRACE_KEY_ELAPSED_SECONDS,
    TRACE_KEY_INPUT_TOKENS,
    TRACE_KEY_MODE,
    TRACE_KEY_MODEL,
    TRACE_KEY_OUTPUT_TOKENS,
    TRACE_KEY_ROLE,
    TRACE_KEY_STEP,
)

FAKE_ENGINE_VERSION = "0.1.3-fake"
FAKE_MODEL = "fake-model"


def scripted_trajectory(answer: str) -> list[dict[str, Any]]:
    """One code block, its execution output, and the final answer, in the raw trace shape."""
    code = "print(len(P))"
    return [
        {
            TRACE_KEY_STEP: 0,
            TRACE_KEY_ROLE: "assistant",
            TRACE_KEY_CONTENT: code,
            TRACE_KEY_CODE: code,
            TRACE_KEY_MODE: MODE_RLM_OFFICIAL,
            TRACE_KEY_INPUT_TOKENS: 100,
            TRACE_KEY_OUTPUT_TOKENS: 10,
            TRACE_KEY_MODEL: FAKE_MODEL,
            TRACE_KEY_ELAPSED_SECONDS: 0.1,
        },
        {
            TRACE_KEY_STEP: 1,
            TRACE_KEY_ROLE: "execution",
            TRACE_KEY_CONTENT: "1234",
            TRACE_KEY_MODE: MODE_RLM_OFFICIAL,
            TRACE_KEY_ELAPSED_SECONDS: 0.01,
        },
        {
            TRACE_KEY_STEP: 2,
            TRACE_KEY_ROLE: "assistant",
            TRACE_KEY_CONTENT: answer,
            TRACE_KEY_MODE: MODE_RLM_OFFICIAL,
            TRACE_KEY_INPUT_TOKENS: 120,
            TRACE_KEY_OUTPUT_TOKENS: 5,
            TRACE_KEY_MODEL: FAKE_MODEL,
            TRACE_KEY_ELAPSED_SECONDS: 0.1,
        },
    ]


class FakeRLMEngine:
    """Engine whose behaviour is fixed up front.

    Args:
        answer: Final answer returned on success.
        input_tokens / output_tokens / total_cost / steps: Reported totals.
        delay: Seconds to sleep inside ``run`` / ``run_async`` — exercises
            the use case's wall-clock guard.
        error: Exception raised instead of returning a result.
        available / reason: What ``is_available`` reports.
        version: What the ``version`` property reports.
    """

    def __init__(
        self,
        *,
        answer: str = "42",
        input_tokens: int = 220,
        output_tokens: int = 15,
        total_cost: float = 0.0012,
        steps: int = 3,
        delay: float = 0.0,
        error: Exception | None = None,
        available: bool = True,
        reason: str = "",
        version: str | None = FAKE_ENGINE_VERSION,
    ) -> None:
        self._answer = answer
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._total_cost = total_cost
        self._steps = steps
        self._delay = delay
        self._error = error
        self._available = available
        self._reason = reason
        self._version = version
        self.calls: list[tuple[str, str, RunConfigDTO]] = []

    @property
    def version(self) -> str | None:
        return self._version

    def is_available(self) -> tuple[bool, str]:
        return self._available, self._reason

    def run(self, content: str, query: str, config: RunConfigDTO) -> RunResultDTO:
        self.calls.append((content, query, config))
        if self._delay:
            time.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return self._result()

    async def run_async(self, content: str, query: str, config: RunConfigDTO) -> RunResultDTO:
        self.calls.append((content, query, config))
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return self._result()

    def _result(self) -> RunResultDTO:
        return RunResultDTO(
            answer=self._answer,
            mode_used=MODE_RLM_OFFICIAL,
            success=True,
            steps=self._steps,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            total_cost=self._total_cost,
            elapsed_time=self._delay,
            trace=scripted_trajectory(self._answer),
        )
