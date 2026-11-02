"""Tests for deterministic, lossless in-run compaction (specs/rlm-working-state, AC-4)."""

from __future__ import annotations

import json

from rlmstudio.application.services.context_compaction import (
    COMPACTED_ACTION_PREFIX,
    COMPACTION_REASON_THRESHOLD,
    RESULT_STUB_PREFIX,
    compact_messages,
    estimate_tokens,
    should_compact,
)
from rlmstudio.application.services.result_registry import ResultRegistry


def _inspect(tool: str, **args: object) -> str:
    return json.dumps({"type": "inspect", "tool": tool, "args": args})


def _paired_messages(n_results: int) -> tuple[list[dict[str, str]], ResultRegistry]:
    """``n_results`` inspect/result pairs after the system + query head."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "QUESTION"},
    ]
    registry = ResultRegistry()
    for i in range(1, n_results + 1):
        messages.append({"role": "assistant", "content": _inspect("peek", start=i, end=i + 10)})
        messages.append({"role": "user", "content": f"Execution result:\npreview of result {i}"})
        ref = registry.register(f"payload {i}", kind="inspect", step=i)
        registry.note_message(len(messages) - 1, ref.id)
    return messages, registry


class TestEstimate:
    def test_uses_message_counter_capability(self) -> None:
        class Counter:
            def count_tokens(self, *, messages: list[dict[str, str]]) -> int:
                return 1234

        assert estimate_tokens(Counter(), [{"role": "user", "content": "x"}]) == 1234

    def test_falls_back_to_previous_estimate_plus_appended_chars(self) -> None:
        class TextOnly:
            def count_tokens(self, text: str) -> int:
                raise TypeError("text only")

        messages = [{"role": "user", "content": "x" * 400}]
        # 100 prior tokens + 100 appended chars / 4 = 125.
        assert (
            estimate_tokens(
                TextOnly(),
                messages,
                prior_prompt_tokens=100,
                prior_prompt_chars=300,
            )
            == 125
        )

    def test_chars_fallback_when_no_previous_prompt(self) -> None:
        assert estimate_tokens(object(), [{"role": "user", "content": "x" * 400}]) == 100


class TestShouldCompact:
    def test_crosses_threshold_only_when_above_reserve(self) -> None:
        assert should_compact(601, 1000, 400) is True
        assert should_compact(600, 1000, 400) is False
        assert should_compact(600, None, 400) is False


class TestCompactMessages:
    def test_stubs_old_results_and_shortens_old_actions(self) -> None:
        messages, registry = _paired_messages(5)
        compacted, record = compact_messages(
            messages, registry, keep_last=2, reason=COMPACTION_REASON_THRESHOLD
        )

        assert record.changed is True
        assert record.ids_stubbed == ("r1", "r2", "r3")
        assert record.actions_compacted == 3

        # Head and the two most recent result pairs are untouched.
        assert compacted[:2] == messages[:2]
        assert compacted[9] == messages[9]  # r4's execution preview
        assert compacted[10] == messages[10]  # action for r5
        assert compacted[11] == messages[11]  # r5's execution preview

        assert compacted[3]["content"] == (
            f"{RESULT_STUB_PREFIX}r1: 9 chars; read_result('r1', ...)]"
        )
        assert compacted[5]["content"].startswith(f"{RESULT_STUB_PREFIX}r2:")
        assert compacted[7]["content"].startswith(f"{RESULT_STUB_PREFIX}r3:")
        assert compacted[2]["content"] == (
            f"{COMPACTED_ACTION_PREFIX}print(peek(start=1, end=11, max_chars=10000))"
        )
        assert compacted[2]["role"] == "assistant"

    def test_is_idempotent(self) -> None:
        messages, registry = _paired_messages(5)
        first, _ = compact_messages(messages, registry, keep_last=2)
        second, record = compact_messages(first, registry, keep_last=2)
        assert second == first
        assert record.changed is False
        assert record.ids_stubbed == ()

    def test_keeps_head_and_short_history_verbatim(self) -> None:
        messages, registry = _paired_messages(2)
        compacted, record = compact_messages(messages, registry, keep_last=2)
        assert compacted == messages
        assert record.changed is False

    def test_unregistered_execution_output_is_left_intact(self) -> None:
        messages, registry = _paired_messages(3)
        unregistered = {"role": "user", "content": "Execution result:\narbitrary code output"}
        messages.insert(2, unregistered)
        # The inserted message shifts the registry index, so rebuild it.
        registry = ResultRegistry()
        for i in range(1, 4):
            ref = registry.register(f"payload {i}", kind="inspect", step=i)
            registry.note_message(3 + (i - 1) * 2 + 1, ref.id)
        compacted, _ = compact_messages(messages, registry, keep_last=1)
        assert unregistered["content"] in [m["content"] for m in compacted]
