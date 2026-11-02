"""Deterministic, lossless in-run compaction for the RLM loop.

When the pending prompt would exceed the model's context window, older
assistant actions and execution previews are replaced by compact
references.  Nothing is lost: every stubbed execution message names the
result id that still owns its full payload in
:class:`~rlmstudio.application.services.result_registry.ResultRegistry`,
and the model can read it back with ``read_result``.

Design (``specs/rlm-working-state`` G5/G6/D4a/D5):

* the trigger estimates the *exact* candidate ``call_messages`` (nudges and
  the last-step ``force_final_nudge`` included) through a
  ``count_tokens(messages=...)`` capability when the adapter has one, and
  falls back to the previous prompt's ``last_clamp_info`` estimate plus a
  chars/4 estimate of everything appended since;
* the last ``keep_last`` result messages and their assistant actions stay
  verbatim; older execution messages become
  ``[result r4: 31,842 chars; read_result('r4', ...)]`` stubs and older
  assistant messages become their one-line action;
* compaction is idempotent — a second pass only advances the window;
* execution messages that are not backed by a registry entry (the v1
  free-form code path) are left untouched, because dropping them would be
  lossy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from rlmstudio.application.sandbox_vars import (
    RLM_DEFAULT_COMPACTION_KEEP_LAST,
)
from rlmstudio.application.services.result_registry import ResultRegistry

COMPACTION_REASON_THRESHOLD = "threshold"
COMPACTION_REASON_OVERFLOW = "overflow_retry"

COMPACTED_ACTION_PREFIX = "[action] "
"""Prefix marking an assistant message that was already compacted."""
RESULT_STUB_PREFIX = "[result "
"""Prefix marking an execution message replaced by a registry reference."""

MAX_ACTION_CHARS = 500
"""Upper bound for the one-line action kept in a compacted assistant message."""


class MessageTokenCounter(Protocol):
    """The optional adapter capability used to count a full message list."""

    def __call__(self, *, messages: list[dict[str, str]]) -> int: ...


@dataclass(frozen=True)
class CompactionRecord:
    """What one ``compact_messages`` pass changed.

    Attributes:
        reason: ``threshold`` or ``overflow_retry``.
        ids_stubbed: Result ids whose execution messages became stubs.
        actions_compacted: Assistant messages reduced to their one-line action.
        tokens_before: Token estimate before the pass.
        tokens_after: Token estimate after the pass.
    """

    reason: str
    ids_stubbed: tuple[str, ...]
    actions_compacted: int
    tokens_before: int
    tokens_after: int

    @property
    def changed(self) -> bool:
        """Whether the pass replaced at least one message."""
        return bool(self.ids_stubbed) or self.actions_compacted > 0


def message_chars(messages: list[dict[str, str]]) -> int:
    """Total characters of message content."""
    return sum(len(m.get("content") or "") for m in messages)


def estimate_tokens(
    llm: Any,
    messages: list[dict[str, str]],
    *,
    prior_prompt_tokens: int | None = None,
    prior_prompt_chars: int = 0,
) -> int:
    """Estimate the prompt tokens of *messages*.

    Uses ``llm.count_tokens(messages=...)`` when the adapter supports it.
    Otherwise uses the previous prompt's reported estimate plus chars/4 for
    everything appended since it, or chars/4 for the whole payload when no
    previous estimate exists.
    """
    counter = getattr(llm, "count_tokens", None)
    if callable(counter):
        try:
            return max(1, int(counter(messages=messages)))
        except Exception:
            pass
    chars = message_chars(messages)
    if prior_prompt_tokens is not None and prior_prompt_chars > 0:
        appended = max(0, chars - prior_prompt_chars)
        return max(1, int(prior_prompt_tokens) + appended // 4)
    return max(1, chars // 4)


def should_compact(
    estimate: int,
    context_window: int | None,
    reserve_tokens: int,
) -> bool:
    """Whether *estimate* crosses ``context_window - reserve_tokens``."""
    if context_window is None or context_window <= 0:
        return False
    threshold = max(0, context_window - max(0, reserve_tokens))
    return estimate > threshold


def compact_messages(
    messages: list[dict[str, str]],
    registry: ResultRegistry,
    *,
    keep_last: int = RLM_DEFAULT_COMPACTION_KEEP_LAST,
    reason: str = COMPACTION_REASON_THRESHOLD,
    counter: MessageTokenCounter | None = None,
) -> tuple[list[dict[str, str]], CompactionRecord]:
    """Replace older actions and execution previews with compact references.

    Returns a new message list; the caller's list is not mutated.  The first
    two messages (system prompt and query), every ``system`` message, and
    the last ``keep_last`` result messages are preserved verbatim.
    """
    before_tokens = _safe_count(counter, messages)

    indexed = sorted(registry.message_index().items())
    if keep_last < 0:
        keep_last = 0
    if len(indexed) <= keep_last:
        return list(messages), _record(reason, (), 0, before_tokens, before_tokens)

    first_kept = indexed[-keep_last][0] if keep_last else len(messages)
    protected_prefix = min(2, len(messages))
    out = [dict(m) for m in messages]
    ids_stubbed: list[str] = []
    stubbed_positions: list[int] = []

    for position, result_id in indexed:
        if position >= first_kept or position < protected_prefix:
            continue
        content = out[position].get("content") or ""
        if content.startswith(RESULT_STUB_PREFIX):
            continue
        ref = registry.describe(result_id)
        if ref is None:
            # No ref means this message cannot be made losslessly reachable;
            # leave the original preview in place.
            continue
        out[position] = {
            "role": out[position].get("role", "user"),
            "content": (
                f"{RESULT_STUB_PREFIX}{ref.id}: {ref.length:,} chars; read_result('{ref.id}', ...)]"
            ),
        }
        stubbed_positions.append(position)
        if result_id not in ids_stubbed:
            ids_stubbed.append(result_id)

    actions_compacted = 0
    for position in stubbed_positions:
        action_position = _preceding_assistant(out, position, protected_prefix)
        if action_position is None:
            continue
        content = out[action_position].get("content") or ""
        if content.startswith(COMPACTED_ACTION_PREFIX):
            continue
        one_line = _one_line_action(content)
        if one_line is None:
            continue
        out[action_position] = {
            "role": "assistant",
            "content": COMPACTED_ACTION_PREFIX + one_line,
        }
        actions_compacted += 1

    after_tokens = _safe_count(counter, out)
    return out, _record(
        reason,
        tuple(ids_stubbed),
        actions_compacted,
        before_tokens,
        after_tokens,
    )


def _record(
    reason: str,
    ids_stubbed: tuple[str, ...],
    actions_compacted: int,
    tokens_before: int,
    tokens_after: int,
) -> CompactionRecord:
    return CompactionRecord(
        reason=reason,
        ids_stubbed=ids_stubbed,
        actions_compacted=actions_compacted,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
    )


def _preceding_assistant(
    messages: list[dict[str, str]], position: int, lower_bound: int
) -> int | None:
    """The assistant message that produced the result preview at ``position``."""
    for candidate in range(position - 1, lower_bound - 1, -1):
        if messages[candidate].get("role") == "assistant":
            return candidate
    return None


def _safe_count(counter: MessageTokenCounter | None, messages: list[dict[str, str]]) -> int:
    if counter is not None:
        try:
            return max(1, int(counter(messages=messages)))
        except Exception:
            pass
    return max(1, message_chars(messages) // 4)


_FENCED_CODE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _one_line_action(content: str) -> str | None:
    """The one-line action to keep for an older assistant message."""
    from rlmstudio.core.actions import parse_action

    try:
        action_type, action = parse_action(content)
    except Exception:
        action_type = None
        action = None

    if action_type == "inspect" and action is not None:
        from rlmstudio.application.services.inspect_dispatcher import display_code

        return _clip(display_code(action))
    if action_type == "subcall" and action is not None:
        query = str(getattr(action, "query", "") or "")
        return _clip(f"subcall(query={query!r})")

    match = _FENCED_CODE.search(content)
    if match:
        for line in match.group(1).splitlines():
            if line.strip():
                return _clip(line.strip())
    return None


def _clip(text: str) -> str:
    if len(text) <= MAX_ACTION_CHARS:
        return text
    return text[: MAX_ACTION_CHARS - 1].rstrip() + "…"
