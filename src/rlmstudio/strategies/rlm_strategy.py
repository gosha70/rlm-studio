# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""RLM strategy — wraps the existing RLM controller."""

import time

from rlmstudio.application.sandbox_vars import MODE_RLM
from rlmstudio.config import RLMConfig
from rlmstudio.core.budget import TokenUsage, estimate_tokens
from rlmstudio.core.rlm import RLM, LLMClient

from .base import StrategyResult


class RLMStrategy:
    """Multi-step exploration via the RLM controller loop."""

    def __init__(
        self,
        client: LLMClient,
        config: RLMConfig | None = None,
    ):
        self.client = client
        self.config = config or RLMConfig()

    @property
    def name(self) -> str:
        return "rlm"

    def run(self, content: str, query: str) -> StrategyResult:
        rlm = RLM(client=self.client, config=self.config)

        start = time.time()
        try:
            result = rlm.run(prompt=content, query=query)
        except Exception as e:
            return StrategyResult(
                strategy=MODE_RLM,
                answer="",
                success=False,
                error=str(e),
                elapsed_time=time.time() - start,
            )

        elapsed = time.time() - start

        # Estimate tokens from the trace
        tokens = TokenUsage()
        tokens.add_input(estimate_tokens(content + query))
        for item in result.trace:
            if item.get("role") == "assistant":
                tokens.add_output(estimate_tokens(item["content"]))
            elif item.get("role") == "execution":
                tokens.add_input(estimate_tokens(item["content"]))

        return StrategyResult(
            strategy=MODE_RLM,
            answer=result.answer,
            success=result.success,
            error=result.error,
            steps=result.steps,
            tokens=tokens,
            elapsed_time=elapsed,
            trace=result.trace,
            metadata={
                "max_steps": self.config.execution.max_steps,
            },
        )
