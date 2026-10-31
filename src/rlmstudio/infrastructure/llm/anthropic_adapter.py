"""Anthropic adapter: wraps the existing ClaudeClient to implement LLMPort."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

from rlmstudio.application.dto import LLMResponseDTO, StreamChunk
from rlmstudio.domain.model_catalog import DEFAULT_ANTHROPIC_MODEL, price_per_1k

logger = logging.getLogger("rlmstudio.infrastructure.anthropic")


class AnthropicAdapter:
    """Adapter that wraps the existing ``rlmstudio.llm.ClaudeClient`` to
    satisfy the :class:`LLMPort` protocol.

    Args:
        client: An existing ``ClaudeClient`` instance.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        """Generate completion via the wrapped Anthropic client.

        Args:
            messages: Chat messages.

        Returns:
            LLMResponseDTO with content and token counts.
        """
        if hasattr(self._client, "complete_with_metadata"):
            resp = self._client.complete_with_metadata(messages)
            return LLMResponseDTO(
                content=resp.content,
                model=resp.model,
                input_tokens=resp.input_tokens or 0,
                output_tokens=resp.output_tokens or 0,
                finish_reason=resp.finish_reason,
            )
        text = self._client.complete(messages)
        model = getattr(self._client, "model", "")
        return LLMResponseDTO(content=text, model=model)

    def complete_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Streaming is not yet implemented for the legacy client."""
        result = self.complete(messages)
        yield result.content

    def count_tokens(self, text: str) -> int:
        """Estimate tokens using a heuristic (Anthropic has no public tokenizer)."""
        return max(1, len(text) // 4)

    async def complete_async(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        """Async completion delegating to the sync method."""
        return await asyncio.to_thread(self.complete, messages)

    async def complete_stream_async(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[StreamChunk]:
        """Non-blocking streaming: delegates to complete_async (asyncio.to_thread).

        The legacy Anthropic client does not support native streaming, so this
        method yields a single terminal ``StreamChunk`` carrying the full
        response DTO. ``ttft_ms`` is ``None`` (no streaming backend);
        token counts and content come from ``complete_with_metadata``.
        """
        response = await self.complete_async(messages)
        yield StreamChunk(delta=response.content, is_final=False)
        yield StreamChunk(delta="", is_final=True, response=response)

    def get_pricing(self) -> dict[str, float]:
        """Return pricing for the configured model, per million tokens.

        Priced from the shared catalog. The table this replaced listed only
        Claude 2 and 3 and fell back to Opus 3 for anything else, so the
        current default was billed at $15/$75 per MTok against a real
        $3/$15 — fivefold, and silently, because a plausible number came
        back. The adapter has no ``get_completion_cost``, so the use cases
        price runs through here; this was not a dormant copy.
        """
        model = getattr(self._client, "model", DEFAULT_ANTHROPIC_MODEL)
        prices = price_per_1k(model)
        if prices is None:
            # Zero rather than another model's rate: a confidently wrong
            # figure is what produced the overcharge above, and callers read
            # 0.0 as "not priced".
            logger.warning(
                "No price known for model %s; reporting 0.0 rather than "
                "guessing from another model's rate.",
                model,
            )
            return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}

        input_per_1k, output_per_1k = prices
        return {
            "input_cost_per_1m": input_per_1k * 1000,
            "output_cost_per_1m": output_per_1k * 1000,
        }
