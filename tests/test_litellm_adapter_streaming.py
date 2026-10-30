"""Phase 1 streaming-under-the-hood tests for LiteLLMAdapter.

Covers acceptance criteria AC-1 (ttft_ms populated on complete()),
AC-2 (TTFT tracks first non-empty content chunk), AC-11 (complete_async
mirrors complete on the same fake chunk sequence), AC-12 (terminal
StreamChunk carries a populated LLMResponseDTO with the four new
fields).

All litellm calls are mocked; no external services are contacted.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rlmstudio.application.dto import LLMResponseDTO, StreamChunk
from rlmstudio.infrastructure.llm.litellm_adapter import LiteLLMAdapter


def _chunk(content: str | None, *, finish_reason: str | None = None, usage=None, model="gpt-4o"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, role=None),
                finish_reason=finish_reason,
                index=0,
            )
        ],
        model=model,
        usage=usage,
    )


def _role_only_chunk():
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=None, role="assistant"),
                finish_reason=None,
                index=0,
            )
        ],
        model="gpt-4o",
        usage=None,
    )


def _empty_content_chunk():
    return _chunk("")


def _usage(prompt: int = 10, completion: int = 5):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


def _terminal_chunk(prompt: int = 10, completion: int = 5):
    return _chunk(None, finish_reason="stop", usage=_usage(prompt, completion))


class TestCompleteTTFT:
    """AC-1, AC-2 — complete() populates ttft_ms on the first non-empty chunk."""

    @patch("litellm.completion")
    def test_complete_populates_ttft_ms(self, mock_completion):
        """AC-1: ttft_ms is non-None after a successful call."""
        mock_completion.return_value = iter([_chunk("hi"), _terminal_chunk()])

        adapter = LiteLLMAdapter(model="gpt-4o")
        result = adapter.complete([{"role": "user", "content": "?"}])

        assert isinstance(result, LLMResponseDTO)
        assert result.ttft_ms is not None
        assert result.ttft_ms >= 0

    @patch("litellm.completion")
    def test_complete_ttft_tracks_first_non_empty_content(self, mock_completion):
        """AC-2: role-only and empty-content chunks do not count toward TTFT.

        The adapter's ``_observe_chunk`` only calls ``time.monotonic``
        when a chunk carries non-empty content, so we only need to
        script timestamps for (t_start, first real content, end).
        """
        # Add 0.0005 s to each timestamp to sidestep float-to-int
        # truncation (50.0 ms computes as 49 under `int()`).
        times = iter([100.0, 100.0505, 100.0805])
        mock_completion.return_value = iter(
            [
                _role_only_chunk(),
                _empty_content_chunk(),
                _chunk("hi"),
                _terminal_chunk(),
            ]
        )

        adapter = LiteLLMAdapter(model="gpt-4o")
        with patch(
            "rlmstudio.infrastructure.llm.litellm_adapter.time.monotonic",
            lambda: next(times),
        ):
            result = adapter.complete([{"role": "user", "content": "?"}])

        # First content chunk observed at t+50.5 ms → TTFT = 50 ms.
        # If the adapter had counted the role-only or empty-content
        # frames we would have measured ~0 ms, which is the
        # regression this test guards against.
        assert result.ttft_ms == 50
        # Total 80 ms, so decode is 80 - 50 = 30 ms.
        assert result.decode_ms == 30

    @patch("litellm.completion")
    def test_complete_single_chunk_provider_ttft_equals_total(self, mock_completion):
        """Fallback contract: when no content chunk is observed,
        ttft_ms == total_ms so decode_ms is 0."""
        # Only a terminal chunk, no content deltas.
        mock_completion.return_value = iter([_terminal_chunk()])

        adapter = LiteLLMAdapter(model="gpt-4o")
        result = adapter.complete([{"role": "user", "content": "?"}])

        assert result.ttft_ms is not None
        assert result.decode_ms == 0

    @patch("litellm.completion")
    def test_complete_flag_off_falls_back_to_non_streaming(self, mock_completion, monkeypatch):
        """Panic lever: RLM_STUDIO_STREAMED_COMPLETE=0 skips streaming; ttft_ms is None."""
        monkeypatch.setenv("RLM_STUDIO_STREAMED_COMPLETE", "0")

        # Non-streaming mock: plain object with .choices/.usage/.model.
        mock_completion.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="plain", role="assistant"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10),
            model="gpt-4o",
        )

        adapter = LiteLLMAdapter(model="gpt-4o")
        result = adapter.complete([{"role": "user", "content": "?"}])

        assert result.content == "plain"
        assert result.input_tokens == 7
        assert result.output_tokens == 3
        assert result.ttft_ms is None
        assert result.decode_ms == 0


class TestCompleteAsyncParity:
    """AC-11 — complete_async mirrors complete on the same fake sequence."""

    @pytest.mark.asyncio
    async def test_complete_async_populates_ttft_and_decode(self):
        async def mock_async_iter():
            yield _chunk("hi")
            yield _terminal_chunk(prompt=11, completion=4)

        adapter = LiteLLMAdapter(model="gpt-4o")
        with patch("litellm.acompletion", return_value=mock_async_iter()):
            result = await adapter.complete_async([{"role": "user", "content": "?"}])

        assert result.content == "hi"
        assert result.input_tokens == 11
        assert result.output_tokens == 4
        assert result.ttft_ms is not None
        assert result.decode_ms >= 0


class TestCompleteStreamLastTelemetry:
    """complete_stream() exposes telemetry via last_response_telemetry."""

    @patch("litellm.completion")
    def test_last_response_telemetry_populated_after_exhaustion(self, mock_completion):
        mock_completion.return_value = iter(
            [_chunk("Hello"), _chunk(" World"), _terminal_chunk(prompt=12, completion=6)]
        )

        adapter = LiteLLMAdapter(model="gpt-4o")
        collected = list(adapter.complete_stream([{"role": "user", "content": "?"}]))

        assert collected == ["Hello", " World"]
        telemetry = adapter.last_response_telemetry
        assert telemetry is not None
        assert telemetry.ttft_ms is not None
        assert telemetry.prompt_tokens == 12
        assert telemetry.completion_tokens == 6


class TestReasoningModelContentFallback:
    """Regression guard: reasoning-model content must not be dropped.

    Pre-Phase-1 ``complete()`` went through ``_extract_content(message)``
    which falls back to ``reasoning_content`` / ``thinking`` / ``thought``.
    The streaming path must honor the same fallback on chunk deltas
    AND on a terminal chunk's ``message`` when no delta ever arrived.
    """

    @patch("litellm.completion")
    def test_reasoning_content_on_delta_is_used(self, mock_completion):
        """DeepSeek-R1-style stream: content=None but reasoning_content on delta."""
        reasoning_chunk = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        role=None,
                        reasoning_content="The answer is 42.",
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
            model="deepseek-r1",
            usage=None,
        )
        mock_completion.return_value = iter(
            [reasoning_chunk, _terminal_chunk(prompt=10, completion=5)]
        )

        adapter = LiteLLMAdapter(model="deepseek-r1")
        result = adapter.complete([{"role": "user", "content": "?"}])

        assert result.content == "The answer is 42."

    @patch("litellm.completion")
    def test_terminal_message_content_fallback(self, mock_completion):
        """Provider packs full response on terminal chunk's message, no deltas.

        Some providers consolidate with ``include_usage=True`` and emit
        a single terminal chunk whose ``choices[0].message.content``
        carries the full response. The accumulator must pull it.
        """
        terminal_with_message = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, role=None),
                    message=SimpleNamespace(content="answer via terminal", role="assistant"),
                    finish_reason="stop",
                    index=0,
                )
            ],
            model="gpt-4o",
            usage=_usage(10, 5),
        )
        mock_completion.return_value = iter([terminal_with_message])

        adapter = LiteLLMAdapter(model="gpt-4o")
        result = adapter.complete([{"role": "user", "content": "?"}])

        assert result.content == "answer via terminal"


class TestMissingUsageFallback:
    """Regression guard: providers that omit usage must not silently zero tokens.

    Streaming with ``include_usage=True`` is not honored by every
    LiteLLM provider. When usage is absent, the adapter must fall
    back to ``litellm.token_counter`` rather than returning 0.
    """

    @patch("litellm.token_counter")
    @patch("litellm.completion")
    def test_token_counter_backfill_when_usage_absent(self, mock_completion, mock_token_counter):
        # Stream with content but no usage on any chunk.
        terminal_no_usage = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, role=None),
                    finish_reason="stop",
                    index=0,
                )
            ],
            model="some-local-model",
            usage=None,
        )
        mock_completion.return_value = iter([_chunk("hello world"), terminal_no_usage])

        # First call: prompt-token count; second: completion-token count.
        mock_token_counter.side_effect = [42, 3]

        adapter = LiteLLMAdapter(model="some-local-model")
        result = adapter.complete([{"role": "user", "content": "?"}])

        assert result.input_tokens == 42
        assert result.output_tokens == 3
        assert result.content == "hello world"


class TestCompleteStreamAsyncStreamChunk:
    """AC-12 — complete_stream_async terminal chunk carries LLMResponseDTO."""

    @pytest.mark.asyncio
    async def test_terminal_stream_chunk_carries_populated_dto(self):
        async def mock_async_iter():
            yield _chunk("Hello")
            yield _chunk(" World")
            yield _terminal_chunk(prompt=13, completion=7)

        adapter = LiteLLMAdapter(model="gpt-4o")
        with patch("litellm.acompletion", return_value=mock_async_iter()):
            chunks: list[StreamChunk] = []
            async for ch in adapter.complete_stream_async([{"role": "user", "content": "?"}]):
                chunks.append(ch)

        # Non-final chunks carry deltas
        deltas = [c.delta for c in chunks if not c.is_final]
        assert deltas == ["Hello", " World"]

        # Terminal chunk has is_final=True and a populated response DTO
        final = chunks[-1]
        assert final.is_final is True
        assert final.response is not None
        assert final.response.input_tokens == 13
        assert final.response.output_tokens == 7
        assert final.response.ttft_ms is not None
        assert final.response.decode_ms >= 0
        # Phase 1 leaves cache extraction for Phase 2
        assert final.response.cached_tokens == 0
        assert final.response.cache_write_tokens == 0

    @pytest.mark.asyncio
    async def test_single_frame_terminal_only_stream_populates_ttft(self):
        """Regression guard: single-frame (terminal-only) async stream.

        When a provider emits no content deltas and only a terminal
        chunk, `complete_async()` sets `ttft_ms = total_ms` so readers
        get a usable number. `complete_stream_async()` must mirror that
        fallback — the terminal StreamChunk's DTO is what WebSocket
        consumers in ``run_rlm.py`` / ``run_direct.py`` trust for
        telemetry, and a ``None`` TTFT there silently loses the signal.
        """

        async def mock_async_iter():
            # Only a terminal chunk, no content frames.
            yield _terminal_chunk(prompt=10, completion=5)

        adapter = LiteLLMAdapter(model="gpt-4o")
        with patch("litellm.acompletion", return_value=mock_async_iter()):
            chunks: list[StreamChunk] = []
            async for ch in adapter.complete_stream_async([{"role": "user", "content": "?"}]):
                chunks.append(ch)

        final = chunks[-1]
        assert final.is_final is True
        assert final.response is not None
        # The critical assertion: TTFT must not be None on a
        # single-frame stream — matches complete_async() behavior.
        assert final.response.ttft_ms is not None
        # decode_ms is 0 because there was no decode phase.
        assert final.response.decode_ms == 0


class TestTransportFallbackOnStreamFailure:
    """Phase 1 follow-up: transport-layer failure before any content
    must fall back to non-streaming ``acompletion`` instead of
    hard-failing the request. Guards the spec's "streaming-under-
    the-hood must not regress availability" contract."""

    def test_complete_sync_falls_back_when_stream_setup_fails(self):
        """Sync ``complete()`` mirrors ``complete_async``: a
        transport-layer failure on the streaming call retries once
        via non-streaming ``litellm.completion``."""

        import litellm as _litellm

        call_log: list[str] = []

        def _fail_then_succeed(*args, **kwargs):
            if kwargs.get("stream"):
                call_log.append("stream")
                raise _litellm.APIConnectionError(
                    message="AnthropicException - [Errno 54] Connection reset by peer",
                    llm_provider="anthropic",
                    model="claude-sonnet-4-6",
                )
            call_log.append("non_stream")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="sync fallback content", role="assistant"),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=30, completion_tokens=6, total_tokens=36),
                model="claude-sonnet-4-6",
            )

        adapter = LiteLLMAdapter(model="claude-sonnet-4-6")
        with patch("litellm.completion", side_effect=_fail_then_succeed):
            result = adapter.complete([{"role": "user", "content": "?"}])

        assert call_log == ["stream", "non_stream"]
        assert result.content == "sync fallback content"
        assert result.input_tokens == 30
        assert result.output_tokens == 6
        assert result.ttft_ms is None

    def test_complete_sync_falls_back_when_stream_walk_fails(self):
        """Transport failure during iteration (not setup) should also
        trigger the non-streaming fallback. The failing iterator's
        ``finally`` closes the orphan stream; the retry produces a
        clean DTO."""

        import litellm as _litellm

        call_log: list[str] = []

        def _failing_iter():
            # Emit one well-formed chunk, then raise mid-walk.
            yield _chunk("partial")
            raise _litellm.APIConnectionError(
                message="AnthropicException - [Errno 9] Bad file descriptor",
                llm_provider="anthropic",
                model="claude-sonnet-4-6",
            )

        def _fail_then_succeed(*args, **kwargs):
            if kwargs.get("stream"):
                call_log.append("stream")
                return _failing_iter()
            call_log.append("non_stream")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="recovered after mid-walk fail", role="assistant"
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=25, completion_tokens=7, total_tokens=32),
                model="claude-sonnet-4-6",
            )

        adapter = LiteLLMAdapter(model="claude-sonnet-4-6")
        with patch("litellm.completion", side_effect=_fail_then_succeed):
            result = adapter.complete([{"role": "user", "content": "?"}])

        # Verify the streaming attempt was made, then the non-streaming
        # retry succeeded. Partial content from the failed stream is
        # discarded — the returned DTO reflects the clean retry, not a
        # concatenation.
        assert call_log == ["stream", "non_stream"]
        assert result.content == "recovered after mid-walk fail"
        assert result.input_tokens == 25
        assert result.output_tokens == 7
        assert result.ttft_ms is None

    def test_complete_sync_non_transport_error_still_raises(self):
        """Auth/4xx errors on the sync streaming path still raise —
        no second attempt, no hiding real errors.

        The match was ``"LiteLLM completion failed"`` until #95 gave auth
        failures an actionable message instead of the generic prefix. The
        assertion is on the new wording because it is strictly more
        specific: it proves both that the error raised rather than being
        retried, and that it raised *as a classified auth failure*.
        """

        import litellm as _litellm

        def _auth_error(*args, **kwargs):
            raise _litellm.AuthenticationError(
                message="Invalid API key",
                llm_provider="openai",
                model="gpt-4o",
            )

        adapter = LiteLLMAdapter(model="gpt-4o")
        with patch("litellm.completion", side_effect=_auth_error):
            with pytest.raises(RuntimeError, match="API key was rejected"):
                adapter.complete([{"role": "user", "content": "?"}])

    @pytest.mark.asyncio
    async def test_complete_async_falls_back_when_stream_setup_fails(self):
        """``complete_async`` retries once through non-streaming when the
        streaming path hits a connection-class error."""

        import litellm as _litellm

        call_log: list[str] = []

        async def _fail_then_succeed(*args, **kwargs):
            # First call: streaming. Raise a connection error.
            # Second call: non-streaming. Return a complete DTO.
            if kwargs.get("stream"):
                call_log.append("stream")
                raise _litellm.APIConnectionError(
                    message="AnthropicException - [Errno 9] Bad file descriptor",
                    llm_provider="anthropic",
                    model="claude-sonnet-4-6",
                )
            call_log.append("non_stream")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="fallback content", role="assistant"),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=40, completion_tokens=8, total_tokens=48),
                model="claude-sonnet-4-6",
            )

        adapter = LiteLLMAdapter(model="claude-sonnet-4-6")
        with patch("litellm.acompletion", side_effect=_fail_then_succeed):
            result = await adapter.complete_async([{"role": "user", "content": "?"}])

        assert call_log == ["stream", "non_stream"]
        assert result.content == "fallback content"
        assert result.input_tokens == 40
        assert result.output_tokens == 8
        # Non-streaming fallback can't measure TTFT — honest None.
        assert result.ttft_ms is None

    @pytest.mark.asyncio
    async def test_complete_async_non_transport_error_still_raises(self):
        """Auth/4xx errors are NOT retried via non-streaming — those
        would just fail the same way and hide the real cause.

        Matched on the classified wording since #95; see the sync variant
        above for why.
        """

        import litellm as _litellm

        async def _auth_error(*args, **kwargs):
            raise _litellm.AuthenticationError(
                message="Invalid API key",
                llm_provider="openai",
                model="gpt-4o",
            )

        adapter = LiteLLMAdapter(model="gpt-4o")
        with patch("litellm.acompletion", side_effect=_auth_error):
            with pytest.raises(RuntimeError, match="API key was rejected"):
                await adapter.complete_async([{"role": "user", "content": "?"}])

    @pytest.mark.asyncio
    async def test_complete_stream_async_falls_back_before_first_content(self):
        """``complete_stream_async`` that hits a transport error before
        yielding any content falls back to a single-chunk non-streaming
        response."""

        import litellm as _litellm

        async def _fail_then_succeed(*args, **kwargs):
            if kwargs.get("stream"):
                raise _litellm.APIConnectionError(
                    message="Connection reset by peer",
                    llm_provider="anthropic",
                    model="claude-sonnet-4-6",
                )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="recovered", role="assistant"),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=20, completion_tokens=4, total_tokens=24),
                model="claude-sonnet-4-6",
            )

        adapter = LiteLLMAdapter(model="claude-sonnet-4-6")
        with patch("litellm.acompletion", side_effect=_fail_then_succeed):
            chunks: list[StreamChunk] = []
            async for ch in adapter.complete_stream_async([{"role": "user", "content": "?"}]):
                chunks.append(ch)

        # Final chunk carries a populated DTO. Content may arrive
        # either as one delta chunk + terminal, or only on terminal,
        # depending on fallback shape. Assert the invariant: the
        # request completed, the consumer got the full content, and
        # ttft_ms is None (honest "no streaming timing available").
        assert chunks[-1].is_final is True
        assert chunks[-1].response is not None
        assert chunks[-1].response.content == "recovered"
        assert chunks[-1].response.ttft_ms is None
        collected = "".join(c.delta for c in chunks if not c.is_final)
        # Either the delta chunk carried the content, or not — but
        # the terminal DTO always has it.
        assert collected == "recovered" or collected == ""
