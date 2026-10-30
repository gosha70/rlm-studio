"""Use case: run a direct LLM query (no RLM exploration).

Sends the full content and query to the LLM in a single request.
This is the simplest execution mode and serves as a baseline.
"""

from __future__ import annotations

import asyncio
import logging
import time

from rlmstudio.application.dto import LLMResponseDTO, RunConfigDTO, RunResultDTO
from rlmstudio.application.ports.event_port import ExecutionEventEmitter
from rlmstudio.application.ports.llm_port import LLMPort
from rlmstudio.application.sandbox_vars import (
    MODE_DIRECT,
    TRACE_KEY_CONTENT,
    TRACE_KEY_ELAPSED_SECONDS,
    TRACE_KEY_INPUT_TOKENS,
    TRACE_KEY_MODE,
    TRACE_KEY_MODEL,
    TRACE_KEY_OUTPUT_TOKENS,
    TRACE_KEY_ROLE,
    TRACE_KEY_STEP,
)
from rlmstudio.application.services.provider_failure import failure_kind_value
from rlmstudio.infrastructure.llm.litellm_adapter import TIMEOUT_ERROR_PREFIX
from rlmstudio.prompts import get_mode_system_prompt

logger = logging.getLogger(__name__)


def _format_error_answer(error_str: str) -> str:
    """Return a user-facing ⚠️ warning for the error, with actionable fix for known categories."""
    error_lower = error_str.lower()
    if (
        error_str.startswith(TIMEOUT_ERROR_PREFIX)
        or "timeout" in error_lower
        or "timed out" in error_lower
    ):
        return (
            "⚠️ **LLM request timed out**.\n\n"
            "The model did not respond within the configured timeout.\n\n"
            "**How to fix** — in Settings → Profile → Runtime Settings:\n"
            "- Increase **Timeout (s)** (try 300 or higher for large documents)."
        )
    if (
        "context window" in error_lower
        or "context_length" in error_lower
        or "contextwindowexceeded" in error_lower
        or "prompt is too long" in error_lower
        or "maximum context length" in error_lower
    ):
        return (
            "⚠️ **Context window exceeded**.\n\n"
            "The document is too large for this model's context limit.\n\n"
            "**How to fix:**\n"
            "- Use a model with a larger context window.\n"
            "- Or reduce the document size (fewer/smaller files).\n"
            "- Or use RAG mode, which retrieves only relevant chunks."
        )
    if "cannot connect" in error_lower or "connection" in error_lower:
        return (
            "⚠️ **Cannot connect to LLM server**.\n\n"
            f"{error_str}\n\n"
            "**How to fix:**\n"
            "- Check that the LLM server is running and reachable.\n"
            "- Verify the endpoint URL in Settings → LLM Providers."
        )
    # Fallback: show the raw error with a warning icon so the user
    # always sees what went wrong instead of "Execution failed".
    return f"⚠️ **Execution error**.\n\n{error_str}"


class RunDirectUseCase:
    """Orchestrates a direct (single-call) LLM query.

    Args:
        llm: LLM port adapter for generating completions.
    """

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    def _compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Compute cost using LiteLLM's completion_cost (handles prefix stripping)."""
        try:
            if hasattr(self._llm, "get_completion_cost"):
                return float(self._llm.get_completion_cost(input_tokens, output_tokens))
            pricing = self._llm.get_pricing()
            input_cost = input_tokens * pricing.get("input_cost_per_1m", 0) / 1_000_000
            output_cost = output_tokens * pricing.get("output_cost_per_1m", 0) / 1_000_000
            return float(input_cost + output_cost)
        except Exception:
            return 0.0

    def execute(
        self,
        content: str,
        query: str,
        config: RunConfigDTO | None = None,
    ) -> RunResultDTO:
        """Run a direct query against the LLM.

        Args:
            content: Document text to analyze.
            query: User question about the content.
            config: Optional run configuration.

        Returns:
            RunResultDTO with the LLM answer and metrics.
        """
        config = config or RunConfigDTO(mode="direct")
        start = time.time()

        from rlmstudio.application.sandbox_vars import EXTRA_KEY_HISTORY_MESSAGES

        system_prompt = get_mode_system_prompt("direct")
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        # Prepend prior turns as native chat messages so the model sees
        # proper user/assistant alternation (works with all model sizes).
        history_msgs = config.extra.get(EXTRA_KEY_HISTORY_MESSAGES)
        if history_msgs:
            messages.extend(history_msgs)
        messages.append(
            {"role": "user", "content": f"Content:\n{content}\n\nQuestion: {query}"},
        )

        try:
            response: LLMResponseDTO = self._llm.complete(messages)
            elapsed = time.time() - start

            total_cost = self._compute_cost(response.input_tokens, response.output_tokens)

            return RunResultDTO(
                answer=response.content,
                mode_used=MODE_DIRECT,
                success=True,
                steps=1,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_cost=total_cost,
                elapsed_time=elapsed,
                trace=[
                    {
                        TRACE_KEY_STEP: 0,
                        TRACE_KEY_ROLE: "assistant",
                        TRACE_KEY_CONTENT: response.content,
                        TRACE_KEY_MODE: "direct",
                        TRACE_KEY_INPUT_TOKENS: response.input_tokens,
                        TRACE_KEY_OUTPUT_TOKENS: response.output_tokens,
                        TRACE_KEY_MODEL: getattr(self._llm, "active_model", None) or response.model,
                        TRACE_KEY_ELAPSED_SECONDS: elapsed,
                        "ttft_ms": response.ttft_ms,
                        "decode_ms": response.decode_ms,
                        "cached_tokens": response.cached_tokens,
                        "cache_write_tokens": response.cache_write_tokens,
                    }
                ],
            )
        except Exception as exc:
            elapsed = time.time() - start
            error_str = str(exc)
            return RunResultDTO(
                answer=_format_error_answer(error_str),
                mode_used=MODE_DIRECT,
                success=False,
                error=error_str,
                failure_kind=failure_kind_value(exc),
                elapsed_time=elapsed,
            )

    async def execute_async(
        self,
        content: str,
        query: str,
        config: RunConfigDTO | None = None,
        event_emitter: ExecutionEventEmitter | None = None,
    ) -> RunResultDTO:
        """Async direct query with optional token streaming.

        If *event_emitter* is provided and the LLM adapter supports async
        streaming, tokens are emitted in real time.
        """
        config = config or RunConfigDTO(mode="direct")
        start = time.time()

        from rlmstudio.application.sandbox_vars import EXTRA_KEY_HISTORY_MESSAGES

        system_prompt = get_mode_system_prompt("direct")
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        history_msgs = config.extra.get(EXTRA_KEY_HISTORY_MESSAGES)
        if history_msgs:
            messages.extend(history_msgs)
        messages.append(
            {"role": "user", "content": f"Content:\n{content}\n\nQuestion: {query}"},
        )

        # Telemetry fields populated from whichever LLM path ran.
        ttft_ms: int | None = None
        decode_ms: int = 0
        cached_tokens: int = 0
        cache_write_tokens: int = 0

        try:
            if event_emitter and hasattr(self._llm, "complete_stream_async"):
                collected: list[str] = []
                final_dto: LLMResponseDTO | None = None
                async for chunk in self._llm.complete_stream_async(messages):
                    if chunk.is_final:
                        final_dto = chunk.response
                        continue
                    if chunk.delta:
                        collected.append(chunk.delta)
                        await event_emitter.on_token(chunk.delta)
                if final_dto is not None:
                    answer = final_dto.content
                    input_tokens = final_dto.input_tokens
                    output_tokens = final_dto.output_tokens
                    ttft_ms = final_dto.ttft_ms
                    decode_ms = final_dto.decode_ms
                    cached_tokens = final_dto.cached_tokens
                    cache_write_tokens = final_dto.cache_write_tokens
                else:
                    # Defensive fallback: adapter did not emit a terminal
                    # chunk. Log and fall back to the pre-Phase-1 heuristic.
                    logger.warning(
                        "complete_stream_async did not yield a terminal "
                        "StreamChunk; falling back to token-length heuristic"
                    )
                    answer = "".join(collected)
                    input_tokens = max(1, sum(len(m["content"]) for m in messages) // 4)
                    output_tokens = max(1, len(answer) // 4)
            elif hasattr(self._llm, "complete_async"):
                response: LLMResponseDTO = await self._llm.complete_async(messages)
                answer = response.content
                input_tokens = response.input_tokens
                output_tokens = response.output_tokens
                ttft_ms = response.ttft_ms
                decode_ms = response.decode_ms
                cached_tokens = response.cached_tokens
                cache_write_tokens = response.cache_write_tokens
            else:
                response = await asyncio.to_thread(self._llm.complete, messages)
                answer = response.content
                input_tokens = response.input_tokens
                output_tokens = response.output_tokens
                ttft_ms = response.ttft_ms
                decode_ms = response.decode_ms
                cached_tokens = response.cached_tokens
                cache_write_tokens = response.cache_write_tokens

            elapsed = time.time() - start

            step_entry = {
                TRACE_KEY_STEP: 0,
                TRACE_KEY_ROLE: "assistant",
                TRACE_KEY_CONTENT: answer,
                TRACE_KEY_MODE: "direct",
                TRACE_KEY_INPUT_TOKENS: input_tokens,
                TRACE_KEY_OUTPUT_TOKENS: output_tokens,
                TRACE_KEY_MODEL: getattr(self._llm, "active_model", None),
                TRACE_KEY_ELAPSED_SECONDS: elapsed,
                "ttft_ms": ttft_ms,
                "decode_ms": decode_ms,
                "cached_tokens": cached_tokens,
                "cache_write_tokens": cache_write_tokens,
            }

            total_cost = self._compute_cost(input_tokens, output_tokens)

            if event_emitter:
                await event_emitter.on_step(step_entry)
                await event_emitter.on_metrics(
                    {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                        "cost_usd": total_cost,
                        "steps": 1,
                        "elapsed_seconds": elapsed,
                    }
                )

            return RunResultDTO(
                answer=answer,
                mode_used=MODE_DIRECT,
                success=True,
                steps=1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_cost=total_cost,
                elapsed_time=elapsed,
                trace=[step_entry],
            )
        except Exception as exc:
            elapsed = time.time() - start
            error_str = str(exc)
            return RunResultDTO(
                answer=_format_error_answer(error_str),
                mode_used=MODE_DIRECT,
                success=False,
                error=error_str,
                failure_kind=failure_kind_value(exc),
                elapsed_time=elapsed,
            )
