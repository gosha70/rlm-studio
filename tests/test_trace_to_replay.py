"""Unit tests for the V2b trace → LearnReplay converter.

Tests the pure function :func:`trace_to_replay` in isolation
(no FastAPI, no telemetry store). The e2e tests in
``tests/e2e/test_replays.py`` cover the route + historical-id
round-trips.

Fixtures use the **live** action-type enum
(``inspect | subcall | final | error``) per
doc_internal/specs/learn-tab/NEXT.md §3f step 4. The earlier
``no-plan-step`` / ``no-decision-step`` fixtures referenced
emitters that do not exist in the live pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rlmstudio.application.services.trace_to_replay import (
    CONVERTOR_VERSION,
    REPLAY_STEP_CAP,
    trace_to_replay,
)
from rlmstudio.server.models import (
    TraceBudget,
    TraceResponse,
    TraceResult,
    TraceStep,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _trace(
    steps: list[TraceStep],
    *,
    execution_id: str = "exec-1",
    query: str = "What is the answer?",
    answer: str = "42",
    success: bool = True,
    error: str | None = None,
    mode: str = "rlm",
    total_cost: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> TraceResponse:
    """Build a TraceResponse fixture with sensible defaults."""
    return TraceResponse(
        execution_id=execution_id,
        session_id="sess-1",
        query=query,
        mode=mode,
        status="complete" if success else "error",
        started_at=datetime(2026, 4, 17, tzinfo=timezone.utc),
        completed_at=datetime(2026, 4, 17, 0, 0, 5, tzinfo=timezone.utc),
        result=TraceResult(
            answer=answer,
            success=success,
            error=error,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_cost=total_cost,
        ),
        budget=TraceBudget(),
        steps=steps,
    )


def _inspect(
    index: int,
    *,
    code: str | None = "x = 1",
    input_tokens: int = 100,
    output_tokens: int = 20,
    duration: float = 0.3,
) -> TraceStep:
    return TraceStep(
        index=index,
        action_type="inspect",
        code=code,
        output=None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_seconds=duration,
    )


def _subcall(
    index: int,
    *,
    output: str | None = "result-output",
    duration: float = 0.05,
) -> TraceStep:
    return TraceStep(
        index=index,
        action_type="subcall",
        code=None,
        output=output,
        duration_seconds=duration,
    )


def _final(index: int, *, output: str | None = "42") -> TraceStep:
    return TraceStep(
        index=index,
        action_type="final",
        code=None,
        output=output,
    )


def _error(index: int, *, output: str | None = "boom") -> TraceStep:
    return TraceStep(
        index=index,
        action_type="error",
        code=None,
        output=output,
    )


# ---------------------------------------------------------------------------
# Synthetic bookends
# ---------------------------------------------------------------------------


class TestBookends:
    def test_synthetic_question_from_query(self) -> None:
        replay = trace_to_replay(_trace([], query="Hello?"))
        assert replay.steps[0].kind == "question"
        assert replay.steps[0].summary == "Hello?"
        assert replay.steps[0].id == "question"

    def test_synthetic_answer_from_result(self) -> None:
        replay = trace_to_replay(_trace([], answer="Hi there."))
        assert replay.steps[-1].kind == "answer"
        assert replay.steps[-1].summary == "Hi there."
        assert replay.steps[-1].id == "answer"

    def test_exactly_one_answer_step_per_replay(self) -> None:
        steps = [
            _inspect(0),
            _subcall(1),
            _inspect(2),
            _subcall(3),
            _final(4),
        ]
        replay = trace_to_replay(_trace(steps))
        answer_steps = [s for s in replay.steps if s.kind == "answer"]
        assert len(answer_steps) == 1, (
            "final step must fold into the synthetic answer, not emit a second one"
        )

    def test_trivial_final_only_trace_emits_only_bookends(self) -> None:
        # A run that's entirely a 'final' step (no inspect/subcall). The
        # bookends still render; zero trace-derived steps.
        replay = trace_to_replay(_trace([_final(0)]))
        assert len(replay.steps) == 2
        assert replay.steps[0].kind == "question"
        assert replay.steps[1].kind == "answer"
        assert replay.metadata.originalStepCount == 2
        assert replay.metadata.truncated in (None, False)


# ---------------------------------------------------------------------------
# Kind-inference rules
# ---------------------------------------------------------------------------


class TestKindInference:
    def test_paired_inspect_subcall_produces_code_then_result(self) -> None:
        steps = [_inspect(0, code="x = 1"), _subcall(1, output="1\n")]
        replay = trace_to_replay(_trace(steps))
        middle = replay.steps[1:-1]
        assert [s.kind for s in middle] == ["code", "result"]
        assert middle[0].details is not None
        assert middle[0].details.code == "x = 1"
        assert middle[1].details is not None
        assert middle[1].details.output == "1\n"

    def test_standalone_inspect_emits_lone_code(self) -> None:
        # inspect without a following subcall — valid per NEXT.md §3b.
        steps = [_inspect(0, code="orphan = 1")]
        replay = trace_to_replay(_trace(steps))
        middle = replay.steps[1:-1]
        assert [s.kind for s in middle] == ["code"]

    def test_standalone_subcall_emits_lone_result(self) -> None:
        # subcall without a preceding inspect — valid per NEXT.md §3b.
        steps = [_subcall(0, output="orphan-output")]
        replay = trace_to_replay(_trace(steps))
        middle = replay.steps[1:-1]
        assert [s.kind for s in middle] == ["result"]

    def test_no_plan_or_decision_kinds_in_trace_sourced_replay(self) -> None:
        # Per NEXT.md §3b: trace-sourced replays use only question /
        # code / result / answer. Bundled replays keep plan/decision.
        steps = [_inspect(0), _subcall(1), _inspect(2), _subcall(3), _final(4)]
        replay = trace_to_replay(_trace(steps))
        kinds = {s.kind for s in replay.steps}
        assert kinds == {"question", "code", "result", "answer"}
        assert "plan" not in kinds
        assert "decision" not in kinds


class TestCodelessInspectSkipping:
    """V2b refinement (NEXT.md §3b): drop inspect steps with falsy
    or whitespace-only ``code``.

    Every real RLM run emits a leading ``inspect`` row for the
    controller's "Runtime fingerprint" preamble (role=assistant,
    no code). Without this rule the replay leads with a ghost
    ``code`` step that has no code payload and the generic fallback
    summary — visible noise in the happy path.
    """

    def test_real_rlm_fingerprint_trace_produces_no_ghost_code_step(self) -> None:
        # Fixture mirrors a real trace captured during manual QA:
        #   step 0: inspect, code="",  output="Runtime fingerprint"
        #   step 1: inspect, code=<grep>,   output=<assistant echo>
        #   step 2: subcall, output=<sandbox result>
        #   step 3: final,   output=<answer>
        # The first inspect must be dropped; step 1+2 pair; step 3
        # folds into the synthetic answer. Result: no empty-code
        # card at position 1, clean [question, code, result, answer].
        steps = [
            TraceStep(
                index=0,
                action_type="inspect",
                code="",
                output="Runtime fingerprint",
            ),
            _inspect(1, code="print(grep(pattern='MARKER-'))"),
            _subcall(2, output="Found 1 match"),
            _final(3, output="The secret password is: PINEAPPLE"),
        ]
        replay = trace_to_replay(_trace(steps))
        kinds = [s.kind for s in replay.steps]
        assert kinds == ["question", "code", "result", "answer"]
        # No ghost card: the single `code` step must carry real code.
        code_step = next(s for s in replay.steps if s.kind == "code")
        assert code_step.details is not None
        assert code_step.details.code == "print(grep(pattern='MARKER-'))"

    def test_whitespace_only_code_is_also_dropped(self) -> None:
        steps = [
            TraceStep(index=0, action_type="inspect", code="   \n\t", output=""),
            _final(1),
        ]
        replay = trace_to_replay(_trace(steps))
        # Only bookends survive.
        assert [s.kind for s in replay.steps] == ["question", "answer"]

    def test_null_code_inspect_is_dropped(self) -> None:
        steps = [
            TraceStep(index=0, action_type="inspect", code=None, output=""),
            _final(1),
        ]
        replay = trace_to_replay(_trace(steps))
        assert [s.kind for s in replay.steps] == ["question", "answer"]

    def test_subcall_after_dropped_inspect_renders_as_standalone_result(self) -> None:
        # Edge case: if a code-less inspect is immediately followed by
        # a subcall, the subcall must render as a lone `result`, not
        # pair with the dropped inspect (per NEXT.md §3b refinement).
        steps = [
            TraceStep(index=0, action_type="inspect", code="", output=""),
            _subcall(1, output="orphan-sandbox-output"),
            _final(2),
        ]
        replay = trace_to_replay(_trace(steps))
        kinds = [s.kind for s in replay.steps]
        assert kinds == ["question", "result", "answer"]

    def test_inspect_with_real_code_still_emits_code_step(self) -> None:
        # Regression guard: the skip rule must NOT fire on inspects
        # that carry model-authored code. Standalone inspects with
        # real code still render as lone `code` steps.
        steps = [_inspect(0, code="x = grep(prompt, pattern='y')"), _final(1)]
        replay = trace_to_replay(_trace(steps))
        assert [s.kind for s in replay.steps] == ["question", "code", "answer"]


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_source_and_execution_id(self) -> None:
        replay = trace_to_replay(_trace([_inspect(0), _subcall(1)], execution_id="e-42"))
        assert replay.metadata.source == "trace"
        assert replay.metadata.executionId == "e-42"

    def test_convertor_version_is_pinned(self) -> None:
        replay = trace_to_replay(_trace([]))
        assert replay.metadata.convertorVersion == CONVERTOR_VERSION

    def test_original_step_count_includes_bookends_plus_expanded(self) -> None:
        steps = [_inspect(0), _subcall(1), _inspect(2), _subcall(3)]
        replay = trace_to_replay(_trace(steps))
        # 2 bookends + 4 trace-derived (2 pairs × 2 steps each) = 6.
        assert replay.metadata.originalStepCount == 6

    def test_truncated_absent_when_trace_fits(self) -> None:
        steps = [_inspect(0), _subcall(1)]
        replay = trace_to_replay(_trace(steps))
        # Keep optional-default-None discipline: field is either True or absent.
        assert replay.metadata.truncated in (None, False)


# ---------------------------------------------------------------------------
# Failure handling (error folds into synthetic answer)
# ---------------------------------------------------------------------------


class TestFailureHandling:
    def test_explicit_error_sets_failed_true_and_folds_into_answer(self) -> None:
        steps = [_inspect(0), _subcall(1)]
        replay = trace_to_replay(
            _trace(
                steps,
                success=False,
                error="Provider returned 503",
                answer="",
            )
        )
        assert replay.metadata.failed is True
        answer = replay.steps[-1]
        assert answer.kind == "answer"
        assert "503" in answer.summary

    def test_error_step_in_trace_does_not_emit_second_answer(self) -> None:
        steps = [_inspect(0), _subcall(1), _error(2, output="stack trace here")]
        replay = trace_to_replay(
            _trace(
                steps,
                success=False,
                error="sandbox crashed",
                answer="",
            )
        )
        answer_count = sum(1 for s in replay.steps if s.kind == "answer")
        assert answer_count == 1
        # The error step's output should not leak in as a separate `result`.
        middle = replay.steps[1:-1]
        assert all(s.kind in ("code", "result") for s in middle)
        assert not any("stack trace here" in (s.details.output or "") for s in middle if s.details)

    def test_failed_run_preserves_trace_output_in_answer_details(self) -> None:
        # Contract (NEXT.md §3b): summary = run-level error label;
        # details.output = the failing *trace-side* payload (e.g. the
        # error step's output). Must not duplicate summary into details
        # when a real trace output exists.
        steps = [
            _inspect(0, code="x = 1 / 0"),
            _error(1, output="Traceback:\n  ZeroDivisionError"),
        ]
        replay = trace_to_replay(
            _trace(
                steps,
                success=False,
                error="Provider timeout after 30s",
                answer="",
            )
        )
        answer = replay.steps[-1]
        assert answer.kind == "answer"
        assert answer.summary == "Provider timeout after 30s"
        assert answer.details is not None
        assert answer.details.output == "Traceback:\n  ZeroDivisionError"

    def test_failed_run_without_trace_output_leaves_details_absent(self) -> None:
        # No trace-side output → details must be None (→ omitted on the
        # wire), not a duplicate of the summary string.
        replay = trace_to_replay(
            _trace(
                [_inspect(0, code="x = 1")],
                success=False,
                error="Provider timeout after 30s",
                answer="",
            )
        )
        answer = replay.steps[-1]
        assert answer.summary == "Provider timeout after 30s"
        assert answer.details is None

    def test_success_false_without_explicit_error_still_marked_failed(self) -> None:
        replay = trace_to_replay(_trace([_inspect(0)], success=False, error=None))
        assert replay.metadata.failed is True

    def test_successful_run_has_no_failed_flag(self) -> None:
        replay = trace_to_replay(_trace([_inspect(0), _subcall(1), _final(2)]))
        # Optional-default-None: the flag is either True (on failure)
        # or absent. Success does NOT set failed=False explicitly —
        # that would make the API noisier.
        assert replay.metadata.failed in (None, False)


# ---------------------------------------------------------------------------
# Truncation — cap math + pair-preservation invariant
# ---------------------------------------------------------------------------


def _paired_steps(num_pairs: int) -> list[TraceStep]:
    """N adjacent inspect+subcall pairs plus a final step."""
    steps: list[TraceStep] = []
    idx = 0
    for _ in range(num_pairs):
        steps.append(_inspect(idx, code=f"step_{idx}"))
        idx += 1
        steps.append(_subcall(idx, output=f"out_{idx}"))
        idx += 1
    steps.append(_final(idx))
    return steps


class TestTruncation:
    def test_no_truncation_for_short_trace(self) -> None:
        # 5 pairs = 10 trace-derived steps; well below the 48 budget.
        replay = trace_to_replay(_trace(_paired_steps(5)))
        assert replay.metadata.truncated in (None, False)
        # Bookends + 10 expanded = 12 total.
        assert len(replay.steps) == 12

    def test_truncation_fires_above_budget(self) -> None:
        # 40 pairs = 80 expanded trace-derived steps; must truncate.
        replay = trace_to_replay(_trace(_paired_steps(40)))
        assert replay.metadata.truncated is True
        assert len(replay.steps) <= REPLAY_STEP_CAP

    def test_truncated_replay_keeps_exactly_one_answer(self) -> None:
        replay = trace_to_replay(_trace(_paired_steps(40)))
        assert sum(1 for s in replay.steps if s.kind == "answer") == 1

    def test_truncated_replay_preserves_adjacent_inspect_subcall_pairs(self) -> None:
        # The pair-preservation invariant: every `code` step that was
        # originally followed by `result` in the raw trace is still
        # followed by a `result` in the output; every `result` that
        # was preceded by `code` is still preceded by one. With all
        # pairs being originally adjacent, no orphans are allowed.
        for num_pairs in (49, 50, 51, 100, 1000):
            replay = trace_to_replay(_trace(_paired_steps(num_pairs)))
            middle = replay.steps[1:-1]
            for i, step in enumerate(middle):
                if step.kind == "code":
                    assert i + 1 < len(middle), (
                        f"num_pairs={num_pairs}: code at end of middle has no following result"
                    )
                    assert middle[i + 1].kind == "result", (
                        f"num_pairs={num_pairs}: code at {i} not followed by result"
                    )
                if step.kind == "result":
                    assert i - 1 >= 0, (
                        f"num_pairs={num_pairs}: result at start of middle has no preceding code"
                    )
                    assert middle[i - 1].kind == "code", (
                        f"num_pairs={num_pairs}: result at {i} not preceded by code"
                    )

    def test_original_step_count_matches_pre_truncation_length(self) -> None:
        # originalStepCount = bookends + every trace-derived step had
        # truncation not fired. Must equal what steps.length WOULD
        # have been pre-cap, re-derived inside the test from the same
        # fixture.
        num_pairs = 100
        replay = trace_to_replay(_trace(_paired_steps(num_pairs)))
        # 2 bookends + 2 × num_pairs trace-derived (final folds in, so
        # not counted here).
        expected_original = 2 + 2 * num_pairs
        assert replay.metadata.originalStepCount == expected_original

    def test_final_step_never_consumes_truncation_budget(self) -> None:
        # Even with a trace sized to the budget, the final step folds
        # into the synthetic answer and does NOT push total above cap.
        replay = trace_to_replay(_trace(_paired_steps(24)))
        # 24 pairs = 48 trace-derived; +2 bookends = 50 exactly. No
        # truncation fires because `final` doesn't count as a
        # trace-derived step.
        assert replay.metadata.truncated in (None, False)
        assert len(replay.steps) == 50

    def test_standalone_groups_survive_in_truncated_output(self) -> None:
        # Interleave standalone inspects with paired ones. Truncation
        # must keep head and tail intact and may drop middle, but
        # standalones are valid outputs — no heuristic promotes them
        # to orphans.
        steps: list[TraceStep] = []
        idx = 0
        for _ in range(30):
            steps.append(_inspect(idx))
            idx += 1
        steps.append(_final(idx))
        replay = trace_to_replay(_trace(steps))
        middle = replay.steps[1:-1]
        # All middle steps should be `code` (standalone inspects).
        assert all(s.kind == "code" for s in middle)


# ---------------------------------------------------------------------------
# Step bounds (global replay length)
# ---------------------------------------------------------------------------


class TestStepBounds:
    def test_final_only_trace_yields_two_steps(self) -> None:
        replay = trace_to_replay(_trace([_final(0)]))
        assert len(replay.steps) == 2  # just question + answer

    def test_truncation_fired_has_at_least_three_steps(self) -> None:
        # When truncation fires, at least one preserved group survives
        # in addition to the bookends.
        replay = trace_to_replay(_trace(_paired_steps(100)))
        assert len(replay.steps) >= 3

    @pytest.mark.parametrize("num_pairs", [0, 1, 5, 24, 25, 100])
    def test_every_replay_length_respects_cap(self, num_pairs: int) -> None:
        replay = trace_to_replay(_trace(_paired_steps(num_pairs)))
        assert len(replay.steps) <= REPLAY_STEP_CAP


# ---------------------------------------------------------------------------
# Ids + titles
# ---------------------------------------------------------------------------


class TestIdsAndTitles:
    def test_replay_id_includes_execution_id(self) -> None:
        replay = trace_to_replay(_trace([], execution_id="exec-abc"))
        assert "exec-abc" in replay.id

    def test_step_ids_are_unique(self) -> None:
        steps = [_inspect(0), _subcall(1), _inspect(2), _subcall(3), _final(4)]
        replay = trace_to_replay(_trace(steps))
        ids = [s.id for s in replay.steps]
        assert len(set(ids)) == len(ids)


# ---------------------------------------------------------------------------
# Final step payload folds into answer
# ---------------------------------------------------------------------------


class TestFinalStepFolding:
    def test_final_step_output_surfaces_in_answer_details(self) -> None:
        steps = [_inspect(0), _subcall(1), _final(2, output="terminal output")]
        replay = trace_to_replay(_trace(steps, answer="The answer is X"))
        answer = replay.steps[-1]
        assert answer.details is not None
        assert answer.details.output == "terminal output"


# ---------------------------------------------------------------------------
# Replay metrics — P2 regression guard: preserve measured zeros, omit absence
# ---------------------------------------------------------------------------


class TestReplayMetricsTelemetryZeros:
    """A measured ``cached_tokens=0`` or ``decode_ms=0`` must not be
    collapsed into "field unavailable" by the replay converter — that
    would make cache-miss and single-chunk responses invisible in the
    Learn-tab replay payload, which defeats the purpose of the new
    telemetry."""

    def test_measured_zero_cached_tokens_is_preserved(self) -> None:
        from rlmstudio.application.services.trace_to_replay import _metrics_from_step

        step = TraceStep(
            index=0,
            action_type="final",
            input_tokens=100,
            output_tokens=20,
            duration_seconds=0.2,
            ttft_ms=150,  # measured — gates the prefill/decode block
            decode_ms=50,
            cached_tokens=0,  # measured cache MISS (not "unknown")
            cache_write_tokens=0,
        )
        metrics = _metrics_from_step(step)
        assert metrics is not None
        assert metrics.ttftMs == 150
        assert metrics.decodeMs == 50
        # The critical assertion: zero survives as zero, not None.
        assert metrics.cachedTokens == 0
        assert metrics.cacheWriteTokens == 0

    def test_measured_zero_decode_ms_is_preserved(self) -> None:
        """Single-chunk provider → TTFT == total_ms → decode_ms=0 by
        adapter contract. That zero is a real signal, not absence."""
        from rlmstudio.application.services.trace_to_replay import _metrics_from_step

        step = TraceStep(
            index=0,
            action_type="final",
            input_tokens=50,
            output_tokens=10,
            duration_seconds=0.1,
            ttft_ms=100,
            decode_ms=0,  # single-chunk response
            cached_tokens=20,
            cache_write_tokens=0,
        )
        metrics = _metrics_from_step(step)
        assert metrics is not None
        assert metrics.decodeMs == 0
        assert metrics.cachedTokens == 20

    def test_legacy_step_without_any_prefill_signal_omits_new_fields(self) -> None:
        """When a step has no prefill/decode signal at all (ttft_ms is
        None and all three ints are 0), the four new fields are
        omitted rather than lied about with zeros."""
        from rlmstudio.application.services.trace_to_replay import _metrics_from_step

        step = TraceStep(
            index=0,
            action_type="final",
            input_tokens=100,
            output_tokens=20,
            duration_seconds=0.2,
            # ttft_ms defaults to None; decode/cache default to 0.
        )
        metrics = _metrics_from_step(step)
        assert metrics is not None
        # Traditional fields survive.
        assert metrics.tokensIn == 100
        assert metrics.tokensOut == 20
        # No prefill/decode signal → omit the four new fields.
        assert metrics.ttftMs is None
        assert metrics.decodeMs is None
        assert metrics.cachedTokens is None
        assert metrics.cacheWriteTokens is None

    def test_all_absent_step_returns_none(self) -> None:
        """With zero traditional metrics AND no prefill/decode signal,
        the step has no metrics to show — return None."""
        from rlmstudio.application.services.trace_to_replay import _metrics_from_step

        step = TraceStep(index=0, action_type="inspect")
        assert _metrics_from_step(step) is None


class TestCompactionReplayStep:
    def test_compaction_renders_as_its_own_step(self) -> None:
        compaction = TraceStep(
            index=2,
            action_type="compaction",
            output="Compacted older context (threshold): 2 result(s) stubbed.",
        )
        replay = trace_to_replay(_trace([_inspect(1), compaction, _final(3)]))
        step = next(s for s in replay.steps if s.id == "compaction-2")
        assert step.kind == "decision"
        assert step.title == "Context compacted"
        assert "Compacted older context" in step.summary
        assert step.details is not None
        assert "2 result(s) stubbed" in (step.details.output or "")

    def test_failed_run_does_not_use_compaction_as_the_failure_payload(self) -> None:
        compaction = TraceStep(
            index=1,
            action_type="compaction",
            output="Compacted older context (threshold): 1 result(s) stubbed.",
        )
        replay = trace_to_replay(
            _trace([compaction], success=False, error="provider down", answer="")
        )
        answer = next(s for s in replay.steps if s.kind == "answer")
        assert answer.details is None
