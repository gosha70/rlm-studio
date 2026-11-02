"""Trace → LearnReplay converter service.

Owns the V2b conversion contract pinned in
``doc_internal/specs/learn-tab/NEXT.md`` §3b/§3f. Given a
canonical :class:`TraceResponse` (i.e. one whose ``action_type``
values are in the ``inspect | subcall | final | error`` enum —
see :func:`load_canonical_trace`), returns a :class:`LearnReplay`
ready to be served by ``GET /api/replays/{execution_id}``.

Pinned rules this module enforces:

* Synthetic ``question`` step at position 0, derived from
  ``TraceResponse.query``.
* ``inspect`` → ``code`` replay step carrying the generated code.
* ``subcall`` → ``result`` replay step carrying the sandbox output.
* ``final`` and ``error`` step payloads **fold into** the synthetic
  ``answer`` step rather than being emitted as separate replay
  steps; this guarantees exactly one ``answer`` per replay.
* ``plan`` / ``decision`` kinds are **not** emitted for
  trace-sourced replays (bundled replays keep them).
* Truncation operates on *groups* (adjacent ``inspect`` + ``subcall``
  is one atomic 2-step group; standalones are 1-step groups), never
  splitting a group mid-way. Final cap is
  ``REPLAY_STEP_CAP = 50`` including the 2 synthetic bookends.
* Run failure (``result.error`` or ``result.success == False``)
  sets ``metadata.failed = true`` and folds the error into the
  ``answer`` step's summary.

This is a pure function. No I/O, no dependency on the trace route
or the FastAPI request. The route layer (``replays.py``) is a thin
adapter that loads a canonical ``TraceResponse`` and calls
:func:`trace_to_replay`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rlmstudio.server.models import (
    LearnReplay,
    LearnReplayMetadata,
    LearnReplayStep,
    LearnReplayStepDetails,
    LearnReplayStepMetrics,
    TraceResponse,
    TraceStep,
)

# Bump when the conversion rules change so clients can detect stale
# bundled assets. Starts at 1 for the first V2b release.
CONVERTOR_VERSION = 1

# Ceiling on the final ``LearnReplay.steps.length``. Provisional
# (NEXT.md §3b note: revisit once real trace distributions are
# available). 2 slots are hard-reserved for the synthetic
# ``question`` + ``answer`` bookends; the remaining 48 are
# available for trace-derived steps.
REPLAY_STEP_CAP = 50
_RESERVED_BOOKENDS = 2
_TRACE_DERIVED_BUDGET = REPLAY_STEP_CAP - _RESERVED_BOOKENDS  # 48

_SUMMARY_MAX_CHARS = 120


def trace_to_replay(trace: TraceResponse) -> LearnReplay:
    """Convert a canonical :class:`TraceResponse` to a :class:`LearnReplay`.

    Caller contract: ``trace.steps[*].action_type`` MUST already be
    canonical (``inspect | subcall | compaction | final | error``). Raw
    roles like ``assistant`` / ``execution`` are a bug in the caller, not
    this function — see :func:`rlmstudio.server.routes.traces.load_canonical_trace`
    for the one source of canonicalization.
    """
    failed = _is_failed(trace)
    question_step = _synthesize_question(trace)
    answer_step = _synthesize_answer(trace, failed=failed)

    groups = _build_groups(trace.steps)
    expanded_count = sum(len(g.steps) for g in groups)
    # Original = bookends + every trace-derived step, pre-truncation.
    original_step_count = _RESERVED_BOOKENDS + expanded_count

    truncated = False
    if expanded_count > _TRACE_DERIVED_BUDGET:
        groups = _truncate_groups(groups, budget=_TRACE_DERIVED_BUDGET)
        truncated = True

    middle_steps: list[LearnReplayStep] = []
    for group in groups:
        middle_steps.extend(group.steps)

    title = (trace.query or "Execution replay").strip() or "Execution replay"
    if len(title) > _SUMMARY_MAX_CHARS:
        title = title[: _SUMMARY_MAX_CHARS - 1].rstrip() + "…"

    description_bits = []
    if trace.mode:
        description_bits.append(f"Mode: {trace.mode}.")
    if trace.chat_provider_name:
        description_bits.append(f"Provider: {trace.chat_provider_name}.")
    description_bits.append("This replay was reconstructed from a saved execution trace.")
    description = " ".join(description_bits)

    metadata = LearnReplayMetadata(
        source="trace",
        executionId=trace.execution_id,
        originalStepCount=original_step_count,
        truncated=truncated if truncated else None,
        failed=True if failed else None,
        convertorVersion=CONVERTOR_VERSION,
    )

    return LearnReplay(
        id=f"trace-{trace.execution_id}",
        title=title,
        description=description,
        steps=[question_step, *middle_steps, answer_step],
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Bookend synthesis
# ---------------------------------------------------------------------------


def _synthesize_question(trace: TraceResponse) -> LearnReplayStep:
    summary = (trace.query or "(empty query)").strip() or "(empty query)"
    return LearnReplayStep(
        id="question",
        kind="question",
        title="Question received",
        summary=summary,
    )


def _synthesize_answer(
    trace: TraceResponse,
    *,
    failed: bool,
) -> LearnReplayStep:
    """Exactly one ``answer`` step per replay.

    Folds in both the ``final`` step's payload (success path) and
    the ``error`` step's payload (failure path). If neither exists
    in ``trace.steps``, the answer still renders — using
    ``trace.result.answer`` / ``trace.result.error`` at the run
    level. See NEXT.md §3b "pinned decision: error folds into
    the synthetic answer".
    """
    terminal_step = _find_terminal_step(trace.steps)
    last_output = _last_nonempty_output(trace.steps)

    if failed:
        summary = (trace.result.error or "").strip() or "The run ended without a final answer."
        # details.output must hold the *trace-side* failing payload
        # (e.g. the error step's output). Falling back to
        # trace.result.error here would duplicate ``summary`` into
        # ``details`` and discard the real trace output when both are
        # present — see NEXT.md §3b "error folds into answer".
        details_output = last_output
        title = "Failed run"
    else:
        summary = (trace.result.answer or "").strip() or "(no answer)"
        details_output = terminal_step.output if terminal_step and terminal_step.output else None
        title = "Final answer"

    details: LearnReplayStepDetails | None = None
    if details_output:
        details = LearnReplayStepDetails(output=details_output)

    metrics = _answer_metrics(trace, terminal_step)

    if len(summary) > _SUMMARY_MAX_CHARS:
        summary = summary[: _SUMMARY_MAX_CHARS - 1].rstrip() + "…"

    return LearnReplayStep(
        id="answer",
        kind="answer",
        title=title,
        summary=summary,
        details=details,
        metrics=metrics,
    )


def _answer_metrics(
    trace: TraceResponse,
    terminal_step: TraceStep | None,
) -> LearnReplayStepMetrics | None:
    # Prefer the terminal step's own metrics; fall back to run-level
    # result totals when no terminal step was emitted.
    if terminal_step is not None:
        m = _metrics_from_step(terminal_step)
        if m is not None:
            return m
    tokens_in = trace.result.input_tokens or None
    tokens_out = trace.result.output_tokens or None
    cost = trace.result.total_cost or None
    if tokens_in is None and tokens_out is None and cost is None:
        return None
    return LearnReplayStepMetrics(
        tokensIn=tokens_in,
        tokensOut=tokens_out,
        latencyMs=None,
        costUsd=cost,
    )


def _find_terminal_step(steps: list[TraceStep]) -> TraceStep | None:
    for s in steps:
        if s.action_type in ("final", "error"):
            return s
    return None


def _last_nonempty_output(steps: list[TraceStep]) -> str | None:
    for s in reversed(steps):
        if s.action_type == "compaction":
            # A compaction row is controller bookkeeping, not model evidence.
            continue
        output: str | None = s.output
        if output:
            return output
    return None


def _is_failed(trace: TraceResponse) -> bool:
    if trace.result.error:
        return True
    if not trace.result.success:
        return True
    for s in trace.steps:
        if s.action_type == "error":
            return True
    return False


# ---------------------------------------------------------------------------
# Group construction — one pass over raw TraceSteps
# ---------------------------------------------------------------------------


@dataclass
class _Group:
    """A set of adjacent ``LearnReplayStep``s that truncate as a unit.

    For trace-sourced replays this is either a 2-step
    ``code + result`` pair (from an adjacent ``inspect`` + ``subcall``
    in the raw trace) or a 1-step group (standalone ``inspect`` /
    ``subcall`` / ``compaction``).
    """

    steps: list[LearnReplayStep] = field(default_factory=list)


def _build_groups(steps: list[TraceStep]) -> list[_Group]:
    groups: list[_Group] = []
    i = 0
    n = len(steps)
    while i < n:
        s = steps[i]
        if s.action_type in ("final", "error"):
            # Folds into the synthetic answer; never a group member.
            i += 1
            continue
        if s.action_type == "compaction":
            groups.append(_Group(steps=[_step_from_compaction(s)]))
            i += 1
            continue
        if s.action_type == "inspect":
            # V2b refinement (NEXT.md §3b): drop inspect steps whose
            # ``code`` is falsy or whitespace-only. These are almost
            # always controller/setup rows (e.g. the "Runtime
            # fingerprint" preamble every RLM run emits) rather than
            # model-authored actions; rendering them as empty
            # code-kind steps produces visible noise in the happy path
            # — a "Code generated" card with no code and the fallback
            # summary. A following ``subcall`` renders as a standalone
            # ``result`` rather than pairing with the dropped inspect.
            if not (s.code or "").strip():
                i += 1
                continue
            # Pair with an immediately-following subcall if present.
            if i + 1 < n and steps[i + 1].action_type == "subcall":
                groups.append(
                    _Group(
                        steps=[
                            _step_from_inspect(s),
                            _step_from_subcall(steps[i + 1]),
                        ]
                    )
                )
                i += 2
                continue
            # Standalone inspect with real code (no observed
            # execution result — valid per NEXT.md §3b).
            groups.append(_Group(steps=[_step_from_inspect(s)]))
            i += 1
            continue
        if s.action_type == "subcall":
            # Standalone subcall (no preceding inspect consumed us).
            groups.append(_Group(steps=[_step_from_subcall(s)]))
            i += 1
            continue
        # Defensive: unknown action_type. Skip rather than fail the
        # whole conversion — the replay is still useful without it.
        i += 1
    return groups


def _step_from_inspect(s: TraceStep) -> LearnReplayStep:
    return LearnReplayStep(
        id=f"code-{s.index}",
        kind="code",
        title="Code generated",
        summary=_summary_for_code(s.code),
        details=LearnReplayStepDetails(code=s.code) if s.code else None,
        metrics=_metrics_from_step(s),
    )


def _step_from_subcall(s: TraceStep) -> LearnReplayStep:
    return LearnReplayStep(
        id=f"result-{s.index}",
        kind="result",
        title="Sandbox executed",
        summary=_summary_for_output(s.output),
        details=LearnReplayStepDetails(output=s.output) if s.output else None,
        metrics=_metrics_from_step(s),
    )


def _step_from_compaction(s: TraceStep) -> LearnReplayStep:
    return LearnReplayStep(
        id=f"compaction-{s.index}",
        kind="decision",
        title="Context compacted",
        summary=_summary_for_output(s.output),
        details=LearnReplayStepDetails(output=s.output) if s.output else None,
        metrics=_metrics_from_step(s),
    )


def _metrics_from_step(s: TraceStep) -> LearnReplayStepMetrics | None:
    tokens_in = s.input_tokens or None
    tokens_out = s.output_tokens or None
    latency_ms = int(s.duration_seconds * 1000) if s.duration_seconds else None
    cost = s.cost_usd or None

    # Prefill/decode telemetry gating (spec v1.7 Phase 3).
    #
    # decode_ms / cached_tokens / cache_write_tokens are plain ints (0 is
    # a legitimate measurement — "single-chunk response", "cache miss",
    # "provider that doesn't distinguish writes"). Only ttft_ms is
    # genuinely None-able on the domain entity ("not populated by this
    # trace").
    #
    # Rule: a step has prefill/decode telemetry when ttft_ms is
    # populated OR any of the three ints is non-zero. When gated, we
    # emit all four as-is, preserving measured zeros. When ungated
    # (pre-Phase-3 legacy steps with all-default fields) we omit the
    # four new keys entirely rather than lie with zeros.
    has_prefill_decode = (
        s.ttft_ms is not None or s.decode_ms > 0 or s.cached_tokens > 0 or s.cache_write_tokens > 0
    )
    ttft_ms: int | None
    decode_ms: int | None
    cached_tokens: int | None
    cache_write_tokens: int | None
    if has_prefill_decode:
        ttft_ms = s.ttft_ms
        decode_ms = s.decode_ms
        cached_tokens = s.cached_tokens
        cache_write_tokens = s.cache_write_tokens
    else:
        ttft_ms = None
        decode_ms = None
        cached_tokens = None
        cache_write_tokens = None

    if (
        tokens_in is None
        and tokens_out is None
        and latency_ms is None
        and cost is None
        and not has_prefill_decode
    ):
        return None
    return LearnReplayStepMetrics(
        tokensIn=tokens_in,
        tokensOut=tokens_out,
        latencyMs=latency_ms,
        costUsd=cost,
        ttftMs=ttft_ms,
        decodeMs=decode_ms,
        cachedTokens=cached_tokens,
        cacheWriteTokens=cache_write_tokens,
    )


def _summary_for_code(code: str | None) -> str:
    if not code:
        return "The model generated an action."
    first_line = code.strip().split("\n", 1)[0].strip()
    if not first_line:
        return "The model generated an action."
    if len(first_line) > _SUMMARY_MAX_CHARS:
        return first_line[: _SUMMARY_MAX_CHARS - 1].rstrip() + "…"
    return first_line


def _summary_for_output(output: str | None) -> str:
    if not output:
        return "The sandbox produced an observation."
    first_line = output.strip().split("\n", 1)[0].strip()
    if not first_line:
        return "The sandbox produced an observation."
    if len(first_line) > _SUMMARY_MAX_CHARS:
        return first_line[: _SUMMARY_MAX_CHARS - 1].rstrip() + "…"
    return first_line


# ---------------------------------------------------------------------------
# Truncation — group-preserving, cap-respecting
# ---------------------------------------------------------------------------


def _truncate_groups(
    groups: list[_Group],
    *,
    budget: int,
) -> list[_Group]:
    """Keep as many groups as fit in ``budget`` expanded replay steps.

    Head and tail grow alternately inward from the first and last
    groups. Never splits a group — the pair-preservation invariant
    (NEXT.md §3b) falls out of that, because each adjacent
    ``inspect`` + ``subcall`` pair is already one atomic group
    produced by :func:`_build_groups`.
    """
    n = len(groups)
    if n == 0:
        return []
    expanded = [len(g.steps) for g in groups]
    if sum(expanded) <= budget:
        return list(groups)
    if n == 1:
        # Single group, atomic by definition. Group size is at most 2
        # and budget is 48, so this branch is unreachable in normal
        # runs; keep the group whole.
        return list(groups)

    # Always keep first + last. If they alone exceed budget the first
    # group wins — "at least one preserved group" in NEXT.md §3b.
    first_cost = expanded[0]
    last_cost = expanded[-1]
    if first_cost + last_cost > budget:
        return [groups[0]]

    left = 0  # last head index included (inclusive)
    right = n - 1  # first tail index included (inclusive)
    used = first_cost + last_cost

    # Alternate preference: head, tail, head, tail ...
    prefer_head = True
    while left + 1 < right:
        head_cost = expanded[left + 1]
        tail_cost = expanded[right - 1]
        if prefer_head:
            first_try, first_cost_try = ("head", head_cost)
            second_try, second_cost_try = ("tail", tail_cost)
        else:
            first_try, first_cost_try = ("tail", tail_cost)
            second_try, second_cost_try = ("head", head_cost)

        if used + first_cost_try <= budget:
            if first_try == "head":
                left += 1
                used += head_cost
            else:
                right -= 1
                used += tail_cost
            prefer_head = not prefer_head
            continue
        if used + second_cost_try <= budget:
            if second_try == "head":
                left += 1
                used += head_cost
            else:
                right -= 1
                used += tail_cost
            prefer_head = not prefer_head
            continue
        # Neither side fits — we're done.
        break

    return list(groups[: left + 1]) + list(groups[right:])
