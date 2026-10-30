"""LiteLLM adapter: unified provider supporting 100+ models via LiteLLM.

This is the primary LLM adapter for RLMKit, providing access to OpenAI,
Anthropic, Google, Cohere, Azure, Bedrock, Ollama, and many more providers
through a single interface.

Supports the paper's two-model optimization: a powerful root_model for
primary reasoning and a cheaper recursive_model for exploration subcalls.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

from rlmstudio.application.dto import LLMResponseDTO, StreamChunk
from rlmstudio.application.services.provider_failure import (
    ProviderFailureError,
    ProviderFailureKind,
    classify_provider_failure,
    describe_provider_failure,
    redact_secrets,
)
from rlmstudio.branding import env, env_name

logger = logging.getLogger(__name__)

# Panic-lever flag. Default on. When set to "0", `complete()` and
# `complete_async()` fall back to the pre-Phase-1 non-streaming path
# and return a DTO with `ttft_ms=None, decode_ms=0`. The flag does not
# gate `complete_stream_async()` (Protocol signature change is
# unconditional) or `complete_stream()` (always walks the shared
# helper).
_STREAMED_COMPLETE_ENV_VAR = env_name("STREAMED_COMPLETE")


def _streamed_complete_enabled() -> bool:
    return str(env("STREAMED_COMPLETE", "1")) != "0"


@dataclass
class _StreamingTelemetry:
    """Pure observation record built while walking a streaming iterator."""

    ttft_ms: int | None = None
    decode_ms: int = 0
    total_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0  # populated by Phase 2 cache extraction
    cache_write_tokens: int = 0  # populated by Phase 2 cache extraction
    finish_reason: str | None = None
    model: str | None = None
    chunks: list[str] = field(default_factory=list)

    @property
    def content(self) -> str:
        return "".join(self.chunks)


_CONNECTION_KEYWORDS = (
    "connection refused",
    "connection error",
    "cannot connect",
    "unreachable",
    "network is unreachable",
    "connect call failed",
    "failed to establish",
    "name or service not known",
    "no route to host",
    "connectionerror",
    "connecterror",
    "apiconnectionerror",
)
# Reserve tokens to absorb the gap between litellm's generic tokenizer
# and the model's actual tokenizer + chat template.  A fixed 32 was not
# enough: Qwen2.5-7B on vLLM showed a 33-token divergence in practice.
# Using 5% of context_window (floor 64) covers tokenizer differences,
# chat template special tokens, and BOS/EOS overhead safely.
CONTEXT_WINDOW_RESERVE_FRACTION = 0.05
CONTEXT_WINDOW_RESERVE_FLOOR = 64

# Minimum output budget the clamp will leave intact.  The model needs at
# least this many tokens to produce a useful response (a JSON action or
# short final answer) — anything less and the clamp refuses proactively.
MIN_OUTPUT_TOKENS = 128


def _is_connection_error(exc: BaseException) -> bool:
    """Return True when *exc* looks like a network-level connection failure."""
    msg = str(exc).lower()
    return any(kw in msg for kw in _CONNECTION_KEYWORDS)


def _read_usage_field(usage: Any, name: str) -> Any:
    """Fetch a field from a LiteLLM usage record tolerating both shapes.

    LiteLLM sometimes exposes ``usage`` as a Pydantic-like object
    (attribute access) and sometimes as a plain dict (when streaming
    with ``include_usage=True`` on some providers). Try both so cache
    extraction works uniformly.
    """
    if usage is None:
        return None
    val = getattr(usage, name, None)
    if val is not None:
        return val
    if isinstance(usage, dict):
        return usage.get(name)
    return None


def _extract_cache_tokens(usage: Any) -> tuple[int, int]:
    """Return ``(cached_read_tokens, cache_write_tokens)`` from a usage record.

    Handles Anthropic (``cache_read_input_tokens`` +
    ``cache_creation_input_tokens``), OpenAI
    (``prompt_tokens_details.cached_tokens``; no distinct write counter —
    returns 0 for cache_write), and any provider that surfaces neither
    (returns ``(0, 0)``).

    Tolerates both attribute-style and dict-style access, since LiteLLM
    can return either shape depending on provider + streaming mode.
    Never raises — missing fields are the common case (Ollama, vLLM
    without prefix caching, OpenAI's first request to a new prefix).
    """
    if usage is None:
        return 0, 0
    # Anthropic via LiteLLM.
    read = _read_usage_field(usage, "cache_read_input_tokens") or 0
    write = _read_usage_field(usage, "cache_creation_input_tokens") or 0
    if read or write:
        return int(read), int(write)
    # OpenAI via LiteLLM — nested under ``prompt_tokens_details``.
    details = _read_usage_field(usage, "prompt_tokens_details")
    if details is not None:
        cached_raw = _read_usage_field(details, "cached_tokens")
        cached = int(cached_raw) if cached_raw else 0
        if cached:
            logger.debug("OpenAI prompt_tokens_details.cached_tokens=%s", cached)
            return cached, 0
        # Details present but zero / missing cache hit. Log exact
        # shape so the operator can tell whether OpenAI literally
        # sent cached_tokens=0 (legit "no cache hit this request")
        # vs. a field-name drift we haven't taught the helper about.
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "usage.prompt_tokens_details present but no cache hit: "
                "cached_tokens=%r details_keys=%s",
                cached_raw,
                _usage_shape_keys(details),
            )
        return 0, 0
    # No details at all — legacy response shape or a provider that
    # doesn't report cache info. Log the top-level keys so we can
    # teach the helper about any new provider schema we encounter.
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "usage had no cache tokens: attrs=%s has_details=False",
            _usage_shape_keys(usage),
        )
    return 0, 0


def _close_stream_sync(stream: Any) -> None:
    """Best-effort sync cleanup for a LiteLLM streaming iterator.

    Same FD-hygiene concern as :func:`_close_stream` but for the
    synchronous paths (``complete``, ``complete_stream``).
    """
    close = getattr(stream, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:
            logger.debug("sync stream close() failed: %s", exc)


async def _close_stream(stream: Any) -> None:
    """Best-effort cleanup for a LiteLLM streaming iterator.

    ``litellm.acompletion(stream=True)`` returns a wrapper
    (``CustomStreamWrapper``) that owns an httpx response under the
    hood. Failing to close it leaves the socket orphaned; Python's
    async-generator GC may later close it from a different task
    after the FD was reused, producing
    ``[Errno 9] Bad file descriptor``.

    Tries ``aclose()`` (httpx-style async), then ``close()`` (sync
    fallback). Swallows everything — cleanup must never raise over
    the user's actual response.
    """
    aclose = getattr(stream, "aclose", None)
    if callable(aclose):
        try:
            await aclose()
            return
        except Exception as exc:
            logger.debug("stream aclose() failed: %s", exc)
    close = getattr(stream, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:
            logger.debug("stream close() failed: %s", exc)


def _usage_shape_keys(usage: Any) -> list[str]:
    """Return the visible attribute/key names on a usage record, for debug."""
    if isinstance(usage, dict):
        return sorted(str(k) for k in usage.keys())
    d = getattr(usage, "__dict__", None)
    if d is not None:
        return sorted(str(k) for k in d.keys())
    return []


_TIMEOUT_KEYWORDS = (
    "timed out",
    "timeout",
    "apitimeouterror",
    "request timed out",
)

# Prefix stamped on all timeout RuntimeErrors raised by this adapter.
# Other layers (use cases, routes) check for this prefix to detect
# timeouts without re-parsing the exception message.
TIMEOUT_ERROR_PREFIX = "LLM_TIMEOUT:"


def _is_timeout_error(exc: BaseException) -> bool:
    """Return True when *exc* looks like a request timeout."""
    msg = str(exc).lower()
    return any(kw in msg for kw in _TIMEOUT_KEYWORDS)


def _connection_error_message(api_base: str | None, exc: BaseException) -> str:
    """Build a human-readable message for connection failures."""
    location = f" at '{api_base}'" if api_base else ""
    hint = (
        " Make sure the server is running and reachable."
        if api_base
        else " Check your provider credentials and network connectivity."
    )
    return f"Cannot connect to the LLM server{location}.{hint} (Detail: {exc})"


def _timeout_error_message(timeout: float, exc: BaseException) -> str:
    """Build an actionable message for timeout failures.

    The message is prefixed with ``TIMEOUT_ERROR_PREFIX`` so callers
    can detect timeouts via ``str(exc).startswith(TIMEOUT_ERROR_PREFIX)``
    without re-parsing keywords.
    """
    return f"{TIMEOUT_ERROR_PREFIX} LLM request timed out after {timeout:.0f}s. (Detail: {exc})"


class LiteLLMAdapter:
    """Unified LLM adapter using LiteLLM for provider-agnostic access.

    This adapter implements :class:`LLMPort` and supports:
    - 100+ LLM providers through a single interface
    - Two-model configuration (root + recursive) per the RLM paper
    - Token counting and cost estimation for all supported models
    - Streaming completions
    - Provider health checking

    Args:
        model: Default model identifier (e.g. "gpt-4o", "claude-3-opus-20240229",
            "ollama/llama3"). LiteLLM uses provider prefixes for routing.
        root_model: Model for root-level reasoning (overrides model for root calls).
        recursive_model: Cheaper model for recursive exploration subcalls.
        api_key: API key override (usually set via environment variables).
        api_base: Custom API base URL.
        temperature: Sampling temperature (0.0-1.0).
        max_tokens: Maximum tokens to generate per call.
        timeout: Request timeout in seconds.
        extra_params: Additional parameters forwarded to litellm.completion().
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        root_model: str | None = None,
        recursive_model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        timeout: float = 120.0,
        num_retries: int = 2,
        extra_params: dict[str, Any] | None = None,
        context_window: int | None = None,
    ) -> None:
        self._model = model
        self._root_model = root_model or model
        self._recursive_model = recursive_model or model
        self._api_key = api_key
        self._api_base = api_base
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._num_retries = num_retries
        self._extra_params = extra_params or {}
        # Total context window (input + output) in tokens.  When set, the
        # adapter dynamically clamps max_tokens per call so that prompt +
        # output never exceeds the limit.
        self._context_window = context_window

        # Track which model to use for the next call (can be toggled)
        self._active_model = self._root_model

        # Diagnostics: updated on every _build_params() call so the RLM
        # trace can record exactly what the clamp decided.
        self._last_clamp_info: dict[str, Any] = {}

        # Telemetry from the most recent `complete_stream()` call; exposed
        # via the `last_response_telemetry` property.
        self._last_stream_telemetry: _StreamingTelemetry | None = None

    @property
    def last_clamp_info(self) -> dict[str, Any]:
        """Return the clamp diagnostics from the most recent _build_params call.

        Keys (all present when context_window is set):
            context_window, configured_max_tokens, estimated_prompt_tokens,
            reserve, effective_max_tokens, clamped (bool).
        Empty dict when context_window is not set or _build_params hasn't run.
        """
        return self._last_clamp_info

    @property
    def last_response_telemetry(self) -> _StreamingTelemetry | None:
        """Telemetry record for the most recent `complete_stream()` call.

        Populated after the iterator is exhausted. ``None`` before the
        first call. Readers consume this to pull TTFT / decode_ms off
        the sync streaming path without changing its yield contract.
        """
        return self._last_stream_telemetry

    # -- LLMPort protocol methods --

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        """Generate a completion using LiteLLM.

        When ``RLM_STUDIO_STREAMED_COMPLETE`` is set and not "0" (the
        default), the call is issued with ``stream=True`` under the
        hood so that TTFT and decode_ms can be measured; the iterator
        is accumulated into a single DTO. When the flag is "0", the
        original non-streaming path is used and ``ttft_ms`` remains
        ``None`` on the returned DTO.

        Args:
            messages: Chat messages with 'role' and 'content' keys.

        Returns:
            LLMResponseDTO with generated text and token counts.

        Raises:
            RuntimeError: If the LiteLLM call fails.
        """
        import litellm

        params = self._build_params(messages)

        if not _streamed_complete_enabled():
            try:
                response = litellm.completion(**params)
            except Exception as exc:
                raise self._translate_exception(exc, "LiteLLM completion failed") from exc

            choice = response.choices[0]
            usage = response.usage
            return LLMResponseDTO(
                content=self._extract_content(choice.message),
                model=response.model or self._active_model,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                finish_reason=choice.finish_reason,
            )

        params["stream"] = True
        stream_options = dict(params.get("stream_options") or {})
        stream_options["include_usage"] = True
        params["stream_options"] = stream_options

        try:
            response = litellm.completion(**params)
            telemetry = self._accumulate_stream_sync(response)
        except Exception as exc:
            # Symmetric with complete_async: retry once via
            # non-streaming on transport-layer failures so the
            # streaming-first default doesn't regress availability
            # for sync callers (run_rag, sync branches in
            # run_direct/run_rlm, public client paths).
            if _is_connection_error(exc):
                logger.warning(
                    "Streaming complete hit transport error (%s); "
                    "falling back to non-streaming once",
                    exc,
                )
                return self._non_streaming_fallback_sync(messages)
            raise self._translate_exception(exc, "LiteLLM completion failed") from exc

        self._backfill_token_counts(telemetry, messages)
        return self._dto_from_telemetry(telemetry)

    def _non_streaming_fallback_sync(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        """Sync twin of ``_non_streaming_fallback``. Retries a failed
        streaming ``complete()`` via plain non-streaming
        ``litellm.completion``. Returns a DTO with ``ttft_ms=None``.
        """
        import litellm

        params = self._build_params(messages)
        # Explicitly no stream/stream_options.
        try:
            response = litellm.completion(**params)
        except Exception as exc:
            raise self._translate_exception(
                exc, "LiteLLM completion failed (after streaming fallback)"
            ) from exc

        choice = response.choices[0]
        usage = response.usage
        return LLMResponseDTO(
            content=self._extract_content(choice.message),
            model=response.model or self._active_model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason,
        )

    def complete_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Generate a streaming completion, yielding text chunks.

        Records TTFT and decode_ms internally; readers can pull the
        resulting :class:`_StreamingTelemetry` from
        :pyattr:`last_response_telemetry` after the iterator is
        exhausted. The yielded chunks are unchanged — the WebSocket
        path sees no buffering regression.

        Args:
            messages: Chat messages.

        Yields:
            Text chunks as they are produced by the LLM.
        """
        import litellm

        params = self._build_params(messages)
        params["stream"] = True
        stream_options = dict(params.get("stream_options") or {})
        stream_options["include_usage"] = True
        params["stream_options"] = stream_options

        telemetry = _StreamingTelemetry()
        self._last_stream_telemetry = telemetry
        try:
            response = litellm.completion(**params)
            yield from self._walk_stream_sync(response, telemetry, forward_deltas=True)
        except Exception as exc:
            raise self._translate_exception(exc, "LiteLLM streaming failed") from exc
        self._backfill_token_counts(telemetry, messages)

    def count_tokens(
        self,
        text: str | None = None,
        *,
        messages: list[dict[str, str]] | None = None,
    ) -> int:
        """Count tokens using LiteLLM's model-aware tokenizer.

        Supports two input shapes (exactly one must be provided):

        - ``count_tokens("some text")`` — count tokens for a raw string.
        - ``count_tokens(messages=[{...}, ...])`` — count tokens for a full
          chat message list, including chat-template overhead.

        Args:
            text: Raw text to tokenize.
            messages: Chat messages (role/content dicts) to tokenize as a
                complete prompt.

        Returns:
            Token count.

        Raises:
            ValueError: If neither or both of ``text``/``messages`` are given.
        """
        if text is None and messages is None:
            raise ValueError("count_tokens requires either text or messages")
        if text is not None and messages is not None:
            raise ValueError("count_tokens accepts text OR messages, not both")

        import litellm

        if messages is not None:
            try:
                return litellm.token_counter(model=self._active_model, messages=messages)
            except Exception:
                # Fallback: sum content chars ÷ 3 + small chat template overhead
                chars = sum(len(m.get("content", "") or "") for m in messages)
                return max(1, chars // 3 + 10)

        assert text is not None  # narrowed by the checks above
        try:
            return litellm.token_counter(model=self._active_model, text=text)
        except Exception:
            # Fallback to heuristic if tokenizer unavailable
            return max(1, len(text) // 4)

    def get_pricing(self) -> dict[str, float]:
        """Return pricing info for the active model from LiteLLM's cost DB.

        Returns:
            Dictionary with 'input_cost_per_1m' and 'output_cost_per_1m'.
        """
        import litellm

        # Try with the full prefixed name first, then stripped base name
        models_to_try = [self._active_model]
        if "/" in self._active_model:
            models_to_try.append(self._active_model.split("/", 1)[1])

        for model_name in models_to_try:
            try:
                info = litellm.get_model_info(model=model_name)
                input_per_token = info.get("input_cost_per_token") or 0.0
                output_per_token = info.get("output_cost_per_token") or 0.0
                return {
                    "input_cost_per_1m": float(input_per_token) * 1_000_000,
                    "output_cost_per_1m": float(output_per_token) * 1_000_000,
                }
            except Exception:
                continue
        return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}

    # -- Async LLMPort methods --

    async def complete_async(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        """Async completion using ``litellm.acompletion``.

        When ``RLM_STUDIO_STREAMED_COMPLETE`` is set and not "0" (the
        default), the call is issued with ``stream=True`` so that TTFT
        and decode_ms can be measured; the async iterator is
        accumulated into a single DTO. When the flag is "0", the
        original non-streaming path is used and ``ttft_ms`` remains
        ``None`` on the returned DTO.

        Args:
            messages: Chat messages with 'role' and 'content' keys.

        Returns:
            LLMResponseDTO with generated text and token counts.

        Raises:
            RuntimeError: If the LiteLLM call fails.
        """
        import litellm

        params = self._build_params(messages)

        if not _streamed_complete_enabled():
            try:
                response = await litellm.acompletion(**params)
            except Exception as exc:
                raise self._translate_exception(exc, "LiteLLM async completion failed") from exc

            choice = response.choices[0]
            usage = response.usage
            return LLMResponseDTO(
                content=self._extract_content(choice.message),
                model=response.model or self._active_model,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                finish_reason=choice.finish_reason,
            )

        params["stream"] = True
        stream_options = dict(params.get("stream_options") or {})
        stream_options["include_usage"] = True
        params["stream_options"] = stream_options

        try:
            response = await litellm.acompletion(**params)
            telemetry = await self._accumulate_stream_async(response)
        except Exception as exc:
            # Transport-layer failure on the streaming path (Anthropic's
            # flaky MidStreamFallbackError, connection reset, etc.). The
            # pre-Phase-1 non-streaming path could have succeeded via
            # litellm's internal num_retries because the legacy
            # acompletion(stream=False) surface handles retryable
            # errors differently. Fall back to non-streaming once so
            # transient transport issues don't hard-fail the request —
            # we lose TTFT/decode visibility for that call but keep
            # the user's response.
            if _is_connection_error(exc):
                logger.warning(
                    "Streaming complete_async hit transport error (%s); "
                    "falling back to non-streaming once",
                    exc,
                )
                return await self._non_streaming_fallback(messages)
            raise self._translate_exception(exc, "LiteLLM async completion failed") from exc

        self._backfill_token_counts(telemetry, messages)
        return self._dto_from_telemetry(telemetry)

    async def _non_streaming_fallback(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        """Retry a failed streaming call via non-streaming ``acompletion``.

        Used when the streaming path hits a transport-layer failure
        before yielding any content. The non-streaming path has
        different retry semantics inside litellm, so a stream that
        failed mid-setup often succeeds on a plain retry. The returned
        DTO has ``ttft_ms=None`` — honest signal that streaming fell
        back and we couldn't measure per-token timing.
        """
        import litellm

        params = self._build_params(messages)
        # Explicitly no stream/stream_options — we're bypassing Phase 1
        # observability for this one call to save the request.
        try:
            response = await litellm.acompletion(**params)
        except Exception as exc:
            raise self._translate_exception(
                exc, "LiteLLM async completion failed (after streaming fallback)"
            ) from exc

        choice = response.choices[0]
        usage = response.usage
        return LLMResponseDTO(
            content=self._extract_content(choice.message),
            model=response.model or self._active_model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason,
        )

    async def complete_stream_async(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[StreamChunk]:
        """Async streaming completion, yielding :class:`StreamChunk` events.

        Non-final chunks carry text deltas. The terminal chunk has
        ``is_final=True`` and a populated ``response`` built from the
        provider's usage record — including TTFT, decode_ms, and
        token counts.

        Args:
            messages: Chat messages.

        Yields:
            :class:`StreamChunk` events; the final chunk carries the
            completed DTO.
        """
        import litellm

        params = self._build_params(messages)
        params["stream"] = True
        stream_options = dict(params.get("stream_options") or {})
        stream_options["include_usage"] = True
        params["stream_options"] = stream_options

        telemetry = _StreamingTelemetry()
        t_start = time.monotonic()
        response: Any = None
        observed_any_delta = False
        try:
            response = await litellm.acompletion(**params)
            async for chunk in response:
                self._observe_chunk(chunk, telemetry, t_start)
                delta_text = self._chunk_delta_text(chunk)
                if delta_text:
                    observed_any_delta = True
                    yield StreamChunk(delta=delta_text, is_final=False)
        except Exception as exc:
            # Transport-layer failure BEFORE any content reached the
            # consumer: safe to transparently fall back to a single
            # non-streaming call. The user gets one big final chunk
            # instead of streamed tokens, but the request completes
            # instead of hard-failing on transient Anthropic/OpenAI
            # connection resets. If content has already been yielded,
            # the UI is showing a partial response and a fallback
            # would produce a different full response — unsafe.
            if not observed_any_delta and _is_connection_error(exc):
                # Close the failed stream before retrying so its FD
                # doesn't survive into the fallback call.
                if response is not None:
                    await _close_stream(response)
                    response = None
                logger.warning(
                    "Streaming complete_stream_async hit transport error "
                    "before any content (%s); falling back to non-streaming",
                    exc,
                )
                try:
                    fb_dto = await self._non_streaming_fallback(messages)
                except Exception:
                    raise self._translate_exception(exc, "LiteLLM async streaming failed") from exc
                # Emit the full response as one content chunk + terminal
                # StreamChunk. Consumers (`run_rlm.py`, `run_direct.py`)
                # treat chunk.response on is_final as authoritative, so
                # this shape matches their expectations.
                if fb_dto.content:
                    yield StreamChunk(delta=fb_dto.content, is_final=False)
                yield StreamChunk(delta="", is_final=True, response=fb_dto)
                return
            raise self._translate_exception(exc, "LiteLLM async streaming failed") from exc
        finally:
            # Always close the httpx-backed stream. Without this the
            # socket gets orphaned when the caller stops consuming the
            # generator early (WebSocket disconnect, use-case budget
            # hit, cancellation). Python's later async-gen GC then
            # closes the FD after it may have been reused by a
            # different task — that's the "[Errno 9] Bad file
            # descriptor" MidStreamFallbackError observed on Anthropic.
            if response is not None:
                await _close_stream(response)

        telemetry.total_ms = int((time.monotonic() - t_start) * 1000)
        if telemetry.ttft_ms is not None:
            telemetry.decode_ms = max(0, telemetry.total_ms - telemetry.ttft_ms)
        else:
            # Single-frame provider (terminal-only chunk with no content
            # deltas). Mirror the sync/_accumulate_stream_async fallback:
            # TTFT == total_ms, decode_ms == 0. Without this branch the
            # terminal DTO reports ttft_ms=None while complete() /
            # complete_async() report a measured value for the same
            # stream — WebSocket consumers trust the terminal DTO, so
            # missing this fallback loses TTFT telemetry.
            telemetry.ttft_ms = telemetry.total_ms

        self._backfill_token_counts(telemetry, messages)
        yield StreamChunk(delta="", is_final=True, response=self._dto_from_telemetry(telemetry))

    # -- Two-model support --

    def use_root_model(self) -> None:
        """Switch to the root model for subsequent calls."""
        self._active_model = self._root_model

    def use_recursive_model(self) -> None:
        """Switch to the recursive model for subsequent calls."""
        self._active_model = self._recursive_model

    @property
    def active_model(self) -> str:
        """Currently active model identifier."""
        return self._active_model

    @property
    def context_length_chars(self) -> int | None:
        """Approximate context window size in characters for the active model.

        Uses ``litellm.get_model_info`` to look up ``max_input_tokens``,
        then converts tokens → chars with a 4-chars-per-token heuristic.
        Returns ``None`` when the lookup fails (unknown/local models).
        """
        import litellm

        models_to_try = [self._active_model]
        if "/" in self._active_model:
            models_to_try.append(self._active_model.split("/", 1)[1])

        for model_name in models_to_try:
            try:
                info = litellm.get_model_info(model=model_name)
                max_input = info.get("max_input_tokens") or info.get("max_tokens")
                if max_input and isinstance(max_input, int) and max_input > 0:
                    return max_input * 4  # ~4 chars per token heuristic
            except Exception:
                continue
        return None

    @property
    def context_window(self) -> int | None:
        """Total context window (input + output) in tokens for the active model.

        Returns the value configured via ``context_window=`` at construction
        time when set; otherwise falls back to ``litellm.get_model_info`` to
        detect ``max_input_tokens``.  Returns ``None`` when neither source
        produces a positive integer (unknown/local models without a configured
        override).
        """
        if self._context_window:
            return self._context_window

        import litellm

        models_to_try = [self._active_model]
        if "/" in self._active_model:
            models_to_try.append(self._active_model.split("/", 1)[1])

        for model_name in models_to_try:
            try:
                info = litellm.get_model_info(model=model_name)
                max_input = info.get("max_input_tokens") or info.get("max_tokens")
                if max_input and isinstance(max_input, int) and max_input > 0:
                    return max_input
            except Exception:
                continue
        return None

    @property
    def min_output_tokens(self) -> int:
        """Minimum output budget the clamp will leave intact.

        The model needs at least this many tokens to produce a useful
        response (a JSON action or a short final answer); anything less
        and the clamp refuses proactively with a ``ValueError``.
        """
        return MIN_OUTPUT_TOKENS

    @property
    def root_model(self) -> str:
        """Root model identifier."""
        return self._root_model

    @property
    def recursive_model(self) -> str:
        """Recursive model identifier."""
        return self._recursive_model

    @property
    def is_two_model(self) -> bool:
        """Whether root and recursive models are different."""
        return self._root_model != self._recursive_model

    # -- Health check --

    def check_health(self) -> bool:
        """Test connectivity by making a minimal completion call.

        Returns:
            True if the provider responds successfully, False otherwise.
        """
        import litellm

        try:
            response = litellm.completion(
                model=self._active_model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
                timeout=10,
                api_key=self._api_key,
                api_base=self._api_base,
                drop_params=True,
            )
            return bool(response.choices)
        except Exception as exc:
            logger.warning("Health check failed for %s: %s", self._active_model, exc)
            return False

    def get_completion_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost for a completion using LiteLLM.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Cost in USD.
        """
        import litellm

        try:
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=self._active_model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
            )
            return prompt_cost + completion_cost
        except Exception as exc:
            logger.debug("Cost lookup failed for model=%s: %s", self._active_model, exc)
            return 0.0

    # -- Private helpers --

    def _translate_exception(self, exc: BaseException, default_prefix: str) -> RuntimeError:
        """Map a raw litellm exception to a RuntimeError with adapter context.

        Timeout and connection keep their existing message shapes: three use
        cases (``run_direct``, ``run_rag``, ``run_rlm``) detect timeouts by
        ``str(exc).startswith(TIMEOUT_ERROR_PREFIX)``, so changing that
        prefix would silently break their classification.

        Everything else is classified through the shared classifier so the
        run path and the connection probe say the same thing (#95). The
        upstream detail is redacted before it is shown, because auth errors
        can echo the rejected key.
        """
        if _is_timeout_error(exc):
            return RuntimeError(_timeout_error_message(self._timeout, exc))
        if _is_connection_error(exc):
            return RuntimeError(_connection_error_message(self._api_base, exc))

        kind = classify_provider_failure(exc)
        if kind is not ProviderFailureKind.UNKNOWN:
            failure = describe_provider_failure(kind, model=self._model, api_base=self._api_base)
            detail = redact_secrets(str(exc))
            # ProviderFailureError subclasses RuntimeError, so every existing
            # `except RuntimeError` keeps working; the carried kind is what
            # lets the server attach a stable classification to the run
            # payload instead of making the UI match on wording.
            # Annotated because the package ships no `py.typed`, so mypy sees
            # cross-module `rlmstudio.*` symbols as Any and `warn_return_any`
            # would flag the return.
            classified: RuntimeError = ProviderFailureError(
                f"{failure.cause} {failure.fix} (Detail: {detail})", failure
            )
            return classified

        return RuntimeError(f"{default_prefix}: {redact_secrets(str(exc))}")

    @staticmethod
    def _reasoning_field_names() -> list[str]:
        """Return the ordered list of reasoning-side field names to fall back on.

        Mirrors :meth:`_extract_content` for the message path. Reading
        from the prompt-messages YAML keeps the fallback list in one
        place for reasoning/thinking models (DeepSeek-R1, Phi-4-reasoning,
        Ollama thinking models) that surface visible output on a side
        channel instead of ``content``.
        """
        from rlmstudio.prompts import get_rlm_message

        return [
            f.strip() for f in get_rlm_message("reasoning_content_fields").split(",") if f.strip()
        ]

    def _chunk_delta_text(self, chunk: Any) -> str:
        """Return the text delta from a streaming chunk, or ``""`` if none.

        Falls back through the reasoning-side field names (``reasoning_content``,
        ``thinking``, ``thought``) when ``delta.content`` is empty. Without
        this fallback, reasoning-model streams would arrive as empty
        content and the returned DTO would have ``content == ""`` even
        though the model produced visible output.
        """
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return ""
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            return ""
        content = getattr(delta, "content", None) or ""
        if content:
            return content
        for field_name in self._reasoning_field_names():
            val = getattr(delta, field_name, None)
            if val:
                return str(val)
        return ""

    def _terminal_message_text(self, chunk: Any) -> str:
        """Pull content from a terminal chunk's ``message`` if present.

        Some providers pack the full response on the terminal chunk's
        ``choices[0].message`` instead of streaming deltas (e.g. when
        ``include_usage=True`` is honored by consolidating). Reuse
        :meth:`_extract_content` which already handles
        ``content`` + reasoning-side fallback.
        """
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""
        return self._extract_content(message)

    def _observe_chunk(
        self,
        chunk: Any,
        telemetry: _StreamingTelemetry,
        t_start: float,
    ) -> None:
        """Fold one streaming chunk into the telemetry record.

        Records TTFT on the first non-empty content delta, accumulates
        content, and captures usage + model + finish_reason when the
        provider puts them on later chunks. Never raises on missing
        fields. Falls back to the terminal chunk's ``message.content``
        when no delta ever arrived (reasoning-model one-shot stream).
        """
        delta_text = self._chunk_delta_text(chunk)
        if delta_text:
            if telemetry.ttft_ms is None:
                telemetry.ttft_ms = int((time.monotonic() - t_start) * 1000)
            telemetry.chunks.append(delta_text)

        choices = getattr(chunk, "choices", None) or []
        if choices:
            finish = getattr(choices[0], "finish_reason", None)
            if finish:
                telemetry.finish_reason = finish
                # Terminal-chunk message fallback: if no deltas ever
                # arrived, pull content from choices[0].message now.
                # Reasoning models on some providers emit one terminal
                # frame with the full response on `message` and no
                # content on any `delta`.
                if not telemetry.chunks:
                    message_text = self._terminal_message_text(chunk)
                    if message_text:
                        if telemetry.ttft_ms is None:
                            telemetry.ttft_ms = int((time.monotonic() - t_start) * 1000)
                        telemetry.chunks.append(message_text)

        model = getattr(chunk, "model", None)
        if model:
            telemetry.model = model

        usage = getattr(chunk, "usage", None)
        if usage is not None:
            prompt = getattr(usage, "prompt_tokens", None)
            completion = getattr(usage, "completion_tokens", None)
            if prompt:
                telemetry.prompt_tokens = int(prompt)
            if completion:
                telemetry.completion_tokens = int(completion)
            cached, cache_write = _extract_cache_tokens(usage)
            # Providers stream partial usage frames during a call; keep the
            # highest observed value rather than overwriting with 0.
            if cached:
                telemetry.cached_tokens = cached
            if cache_write:
                telemetry.cache_write_tokens = cache_write

    def _backfill_token_counts(
        self,
        telemetry: _StreamingTelemetry,
        messages: list[dict[str, str]],
    ) -> None:
        """Fill in token counts when the provider omitted usage frames.

        Streaming with ``include_usage=True`` is not honored by every
        provider/model on the LiteLLM surface. When usage is missing,
        ``prompt_tokens`` / ``completion_tokens`` would otherwise be 0,
        silently undercounting budgets and trace metrics downstream.
        Falls back to ``litellm.token_counter`` for the prompt
        (model-aware) and for the accumulated content. Never raises —
        leaves 0 on tokenizer failure.
        """
        if telemetry.prompt_tokens == 0:
            try:
                import litellm

                telemetry.prompt_tokens = int(
                    litellm.token_counter(model=self._active_model, messages=messages)
                )
            except Exception as exc:
                logger.debug(
                    "Prompt-token backfill failed for model=%s: %s",
                    self._active_model,
                    exc,
                )
        if telemetry.completion_tokens == 0 and telemetry.chunks:
            try:
                import litellm

                telemetry.completion_tokens = int(
                    litellm.token_counter(model=self._active_model, text=telemetry.content)
                )
            except Exception as exc:
                logger.debug(
                    "Completion-token backfill failed for model=%s: %s",
                    self._active_model,
                    exc,
                )

    def _accumulate_stream_sync(self, response: Iterator[Any]) -> _StreamingTelemetry:
        """Walk a sync streaming iterator, return the accumulated telemetry.

        try/finally+close() mirrors the async path's FD-hygiene fix;
        stops the underlying socket from being abandoned if an
        exception fires mid-walk.
        """
        telemetry = _StreamingTelemetry()
        t_start = time.monotonic()
        try:
            for chunk in response:
                self._observe_chunk(chunk, telemetry, t_start)
        finally:
            _close_stream_sync(response)
        telemetry.total_ms = int((time.monotonic() - t_start) * 1000)
        if telemetry.ttft_ms is not None:
            telemetry.decode_ms = max(0, telemetry.total_ms - telemetry.ttft_ms)
        else:
            # No content chunk ever arrived (provider emitted only
            # role/usage frames). Treat TTFT as total so downstream
            # readers get a usable number rather than None.
            telemetry.ttft_ms = telemetry.total_ms
        return telemetry

    async def _accumulate_stream_async(self, response: AsyncIterator[Any]) -> _StreamingTelemetry:
        """Walk an async streaming iterator, return the accumulated telemetry.

        Wraps the walk in try/finally and closes the underlying stream
        explicitly on the way out. Without this, an exception mid-walk
        (or a later cancellation) leaves the httpx response socket
        orphaned — Python's async-generator GC then runs on a
        different task/loop, and the FD can be closed after it's been
        reused, producing the ``[Errno 9] Bad file descriptor``
        "MidStreamFallbackError" we saw on Anthropic.
        """
        telemetry = _StreamingTelemetry()
        t_start = time.monotonic()
        try:
            async for chunk in response:
                self._observe_chunk(chunk, telemetry, t_start)
        finally:
            await _close_stream(response)
        telemetry.total_ms = int((time.monotonic() - t_start) * 1000)
        if telemetry.ttft_ms is not None:
            telemetry.decode_ms = max(0, telemetry.total_ms - telemetry.ttft_ms)
        else:
            telemetry.ttft_ms = telemetry.total_ms
        return telemetry

    def _walk_stream_sync(
        self,
        response: Iterator[Any],
        telemetry: _StreamingTelemetry,
        *,
        forward_deltas: bool,
    ) -> Iterator[str]:
        """Walk a sync stream, mutate telemetry in place, yield deltas.

        Used by :meth:`complete_stream` to share observation logic
        with :meth:`complete` while preserving the chunk-yield
        contract of the streaming API. try/finally closes the
        underlying stream even if the caller abandons the generator.
        """
        t_start = time.monotonic()
        try:
            for chunk in response:
                self._observe_chunk(chunk, telemetry, t_start)
                if forward_deltas:
                    delta_text = self._chunk_delta_text(chunk)
                    if delta_text:
                        yield delta_text
        finally:
            _close_stream_sync(response)
        telemetry.total_ms = int((time.monotonic() - t_start) * 1000)
        if telemetry.ttft_ms is not None:
            telemetry.decode_ms = max(0, telemetry.total_ms - telemetry.ttft_ms)
        else:
            telemetry.ttft_ms = telemetry.total_ms

    def _dto_from_telemetry(self, telemetry: _StreamingTelemetry) -> LLMResponseDTO:
        """Build the adapter's returned DTO from an accumulated telemetry record."""
        return LLMResponseDTO(
            content=telemetry.content,
            model=telemetry.model or self._active_model,
            input_tokens=telemetry.prompt_tokens,
            output_tokens=telemetry.completion_tokens,
            finish_reason=telemetry.finish_reason,
            ttft_ms=telemetry.ttft_ms,
            decode_ms=telemetry.decode_ms,
            cached_tokens=telemetry.cached_tokens,
            cache_write_tokens=telemetry.cache_write_tokens,
        )

    def _extract_content(self, message: Any) -> str:
        """Extract text content from an LLM response message.

        Some reasoning/thinking models (DeepSeek-R1, Phi-4-reasoning, Ollama
        thinking models) place all output tokens in a side channel
        (``reasoning_content`` or ``thinking``) while leaving ``content``
        empty.  Fall back through known field names so callers always receive
        a non-empty string when the model did produce output.
        """
        content = getattr(message, "content", None) or ""
        if content:
            return content
        from rlmstudio.prompts import get_rlm_message

        fallback_fields = [
            f.strip() for f in get_rlm_message("reasoning_content_fields").split(",") if f.strip()
        ]
        for field_name in fallback_fields:
            val = getattr(message, field_name, None)
            if val:
                logger.debug(
                    "model=%s: content empty, using %s (%d chars)",
                    self._active_model,
                    field_name,
                    len(str(val)),
                )
                return str(val)
        return ""

    def _build_params(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Build the parameter dict for litellm.completion().

        When ``context_window`` is known, dynamically clamps ``max_tokens``
        so that (estimated_prompt_tokens + max_tokens) never exceeds the
        context limit.  This prevents hard rejections from vLLM / Ollama
        when conversation history grows mid-loop.

        Args:
            messages: Chat messages.

        Returns:
            Keyword arguments for litellm.completion().
        """
        # Anthropic Claude rejects requests with both temperature AND top_p.
        # When both are set, prefer top_p (the user's explicit sampling choice)
        # and drop temperature.  Detect Anthropic by model prefix — litellm
        # routes "anthropic/..." and bare "claude-..." to the Anthropic API.
        _is_anthropic = self._active_model.startswith(("anthropic/", "claude-"))
        _send_top_p = self._top_p != 1.0

        params: dict[str, Any] = {
            "model": self._active_model,
            "messages": messages,
            "timeout": self._timeout,
            "num_retries": self._num_retries,
            "drop_params": True,
        }
        if _send_top_p and _is_anthropic:
            # Anthropic: send only top_p, omit temperature
            params["top_p"] = self._top_p
        elif _send_top_p:
            # Other providers: send both
            params["temperature"] = self._temperature
            params["top_p"] = self._top_p
        else:
            # Default: just temperature
            params["temperature"] = self._temperature

        effective_max_tokens = self._max_tokens
        if effective_max_tokens is not None and self._context_window:
            # Use litellm's model-aware tokenizer for accurate counting;
            # fall back to a conservative chars÷3 heuristic if unavailable.
            try:
                import litellm as _litellm

                estimated_prompt_tokens = _litellm.token_counter(
                    model=self._active_model, messages=messages
                )
            except Exception:
                prompt_chars = sum(len(m.get("content", "")) for m in messages)
                # chars÷3 is safer than chars÷4 — multilingual models (Qwen,
                # Llama-3) and chat templates often use <3 chars per token.
                estimated_prompt_tokens = prompt_chars // 3 + 50  # +50 for chat template overhead
            reserve = max(
                int(self._context_window * CONTEXT_WINDOW_RESERVE_FRACTION),
                CONTEXT_WINDOW_RESERVE_FLOOR,
            )
            remaining = self._context_window - estimated_prompt_tokens - reserve
            configured_max = effective_max_tokens
            min_output_tokens = MIN_OUTPUT_TOKENS
            clamped = False
            if remaining < min_output_tokens:
                # Prompt + reserve exceeds context window.  Calculate hard
                # headroom (ignoring reserve) to decide whether to attempt.
                hard_headroom = self._context_window - estimated_prompt_tokens
                if hard_headroom < min_output_tokens:
                    # Even without any reserve the prompt leaves less than
                    # min_output_tokens — raise proactively so the RLM loop
                    # catches this as a context overflow and falls back to
                    # synthesis instead of sending a doomed request.
                    self._last_clamp_info = {
                        "context_window": self._context_window,
                        "configured_max_tokens": configured_max,
                        "estimated_prompt_tokens": estimated_prompt_tokens,
                        "reserve": reserve,
                        "effective_max_tokens": 0,
                        "clamped": True,
                        "proactive_overflow": True,
                    }
                    raise ValueError(
                        f"Context window exhausted: prompt ≈{estimated_prompt_tokens} tokens "
                        f"+ minimum output {min_output_tokens} > context window "
                        f"{self._context_window} tokens. "
                        f"Cannot generate a useful response."
                    )
                # There's *some* headroom but less than ideal — use it, skip
                # the safety reserve since we're already at the edge.
                logger.warning(
                    "Prompt ≈%d tokens nearly fills context_window=%d; "
                    "clamping max_tokens to hard headroom=%d (reserve skipped)",
                    estimated_prompt_tokens,
                    self._context_window,
                    hard_headroom,
                )
                effective_max_tokens = hard_headroom
                clamped = True
            elif remaining < effective_max_tokens:
                logger.debug(
                    "Dynamic max_tokens clamp: %d → %d (prompt≈%d, reserve=%d, context_window=%d)",
                    effective_max_tokens,
                    remaining,
                    estimated_prompt_tokens,
                    reserve,
                    self._context_window,
                )
                effective_max_tokens = remaining
                clamped = True

            self._last_clamp_info = {
                "context_window": self._context_window,
                "configured_max_tokens": configured_max,
                "estimated_prompt_tokens": estimated_prompt_tokens,
                "reserve": reserve,
                "effective_max_tokens": effective_max_tokens,
                "clamped": clamped,
            }
        else:
            self._last_clamp_info = {}

        if effective_max_tokens is not None:
            params["max_tokens"] = effective_max_tokens

        if self._api_key is not None:
            params["api_key"] = self._api_key

        if self._api_base is not None:
            params["api_base"] = self._api_base

        params.update(self._extra_params)
        return params

    def __repr__(self) -> str:
        if self.is_two_model:
            return f"LiteLLMAdapter(root={self._root_model!r}, recursive={self._recursive_model!r})"
        return f"LiteLLMAdapter(model={self._model!r})"
