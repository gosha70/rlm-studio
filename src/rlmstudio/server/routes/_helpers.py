"""Shared route-layer helpers.

Currently houses the raw-DTO-to-domain translator used by chat and
compare_matrix to materialize an ``ExecutionTrace`` from the raw
dict trace shape the use cases emit. The classifier (which lives in
``application/services/``) then consumes the materialized trace for
the PREFILL_TIMEOUT refinement — see spec v1.7 §8b.

This module exists because the raw-DTO shape uses different keys
than ``TraceStep.to_dict``/``from_dict``, so a dedicated translator
is needed at the route boundary. ``TraceStep.from_dict`` is the
inverse of ``to_dict`` and is NOT appropriate for the raw-DTO path.
"""

from __future__ import annotations

from typing import Any

from rlmstudio.application.sandbox_vars import (
    TRACE_KEY_CODE,
    TRACE_KEY_CONTENT,
    TRACE_KEY_ELAPSED_SECONDS,
    TRACE_KEY_INPUT_TOKENS,
    TRACE_KEY_MODEL,
    TRACE_KEY_OUTPUT_TOKENS,
    TRACE_KEY_ROLE,
    TRACE_KEY_STEP,
)
from rlmstudio.domain.entities import ExecutionTrace, TraceStep

# ---------------------------------------------------------------------------
# Role → action_type canonicalization
# ---------------------------------------------------------------------------
#
# Lifted here from ``chat.py`` so the route helpers don't depend on the
# chat module (which would otherwise be a circular import for every
# route that wants the translator or the materializer). The two
# pre-existing chat-side callers (``_save_trajectory`` and the JSONL
# export path) continue to work via a re-export on ``chat.py``.
#
# Contract is pinned: ``assistant → inspect``, ``execution → subcall``,
# ``compaction → compaction``, and the last step promotes to ``final``
# **only** when ``success=True`` and it is not a compaction row.
# Failed-terminal steps keep the role-mapped action_type (usually
# ``inspect``); no ``error`` branch exists and this spec does not add
# one (see v1.5 prose correction).

_ACTION_TYPE_MAP = {
    "assistant": "inspect",
    "execution": "subcall",
    "compaction": "compaction",
}


def _canonical_action_type(role: str | None, is_last: bool, success: bool) -> str:
    """Normalize a raw trace role into a canonical ExecutionTrace action type.

    Mirrors the normalization used by :func:`_save_trajectory` so telemetry
    rows, JSONL exports, and in-memory traces all agree.  A compaction row is
    never promoted to ``final``: it is a controller event, not the run's
    terminal answer.
    """
    action_type = _ACTION_TYPE_MAP.get(role or "", "inspect")
    if is_last and success and action_type != "compaction":
        action_type = "final"
    return action_type


def _translate_raw_trace_entry(
    d: dict[str, Any],
    *,
    is_last: bool,
    run_success: bool,
) -> TraceStep:
    """Translate one raw-DTO trace dict into a :class:`TraceStep`.

    - ``role`` → ``action_type`` via :func:`_canonical_action_type`.
      That helper maps ``assistant`` → ``inspect``, ``execution`` →
      ``subcall``, and promotes the last step to ``final`` **only**
      when ``run_success=True``. Failed-terminal steps keep the
      role-mapped action_type (usually ``inspect``); no ``error``
      promotion exists and this spec does not add one.
    - ``content`` → ``output`` for execution-like roles
      (``execution``, ``rag_retrieval``, ``compaction``); →
      ``raw_response`` for ``assistant``.
    - ``input_tokens`` + ``output_tokens`` → ``prompt_tokens`` +
      ``completion_tokens``; their sum feeds the legacy
      ``tokens_used``.
    - The four new telemetry keys (plain strings in the raw DTO)
      map to the equivalent ``TraceStep`` fields.
    """
    role = d.get(TRACE_KEY_ROLE)
    action_type = _canonical_action_type(role, is_last=is_last, success=run_success)
    input_tokens = int(d.get(TRACE_KEY_INPUT_TOKENS, 0) or 0)
    output_tokens = int(d.get(TRACE_KEY_OUTPUT_TOKENS, 0) or 0)
    content = d.get(TRACE_KEY_CONTENT)
    return TraceStep(
        index=int(d.get(TRACE_KEY_STEP, 0) or 0),
        action_type=action_type,  # type: ignore[arg-type]
        code=d.get(TRACE_KEY_CODE),
        output=content if role in ("execution", "rag_retrieval", "compaction") else None,
        raw_response=content if role == "assistant" else None,
        tokens_used=input_tokens + output_tokens,
        duration=float(d.get(TRACE_KEY_ELAPSED_SECONDS, 0.0) or 0.0),
        model=d.get(TRACE_KEY_MODEL),
        error=d.get("error"),
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        ttft_ms=d.get("ttft_ms"),
        decode_ms=int(d.get("decode_ms", 0) or 0),
        cached_tokens=int(d.get("cached_tokens", 0) or 0),
        cache_write_tokens=int(d.get("cache_write_tokens", 0) or 0),
    )


def _materialize_trace(
    serialized: list[dict[str, Any]] | None,
    *,
    run_success: bool,
) -> ExecutionTrace | None:
    """Rehydrate the route-held raw-DTO trace shape into an
    :class:`ExecutionTrace` the classifier can consume.

    Returns ``None`` when the input is ``None``, empty, or contains
    no dict entries. Tolerant of mixed entries (non-dicts are
    skipped).
    """
    if not serialized:
        return None
    dicts = [d for d in serialized if isinstance(d, dict)]
    if not dicts:
        return None
    last_i = len(dicts) - 1
    steps = [
        _translate_raw_trace_entry(
            d,
            is_last=(i == last_i),
            run_success=run_success,
        )
        for i, d in enumerate(dicts)
    ]
    return ExecutionTrace(steps=steps, start_time=0.0)
