"""Tests for _prepare_history_context dispatch helper."""

from __future__ import annotations

from typing import Any

import pytest

from rlmstudio.application.sandbox_vars import (
    EXTRA_KEY_HISTORY_MESSAGES,
    EXTRA_KEY_HISTORY_VARIABLE,
    HISTORY_PATH_DISABLED,
    HISTORY_PATH_EMPTY,
    HISTORY_PATH_INPROMPT,
    HISTORY_PATH_REPL_VARIABLE,
)
from rlmstudio.server.routes.chat import _prepare_history_context


class FakeAdapter:
    def __init__(self, context_window: int = 8192, min_output_tokens: int = 128):
        self._context_window = context_window
        self._min_output_tokens = min_output_tokens

    @property
    def context_window(self) -> int:
        return self._context_window

    @property
    def min_output_tokens(self) -> int:
        return self._min_output_tokens

    def count_tokens(
        self, text: str | None = None, *, messages: list[dict[str, str]] | None = None
    ) -> int:
        if messages:
            return sum(max(1, len(m.get("content", "")) // 4) + 3 for m in messages)
        return max(1, len(text or "") // 4)


class _FakeRAGConfig:
    chunk_size: int = 512
    top_k: int = 5


class _FakeModeConfig:
    rag_config = _FakeRAGConfig()


class _FakeConfig:
    mode_config = _FakeModeConfig()


class FakeState:
    config = _FakeConfig()

    def __init__(self, conversations: dict[str, list[dict[str, Any]]] | None = None):
        self._conversations = conversations or {}

    def get_conversation(self, session_id: str, chat_provider_id: str) -> list[dict[str, Any]]:
        return self._conversations.get(f"{session_id}:{chat_provider_id}", [])


class FakeChatProvider:
    def __init__(
        self,
        conversation_memory_enabled: bool = True,
        conversation_memory_fraction: float = 0.30,
    ):
        self.conversation_memory_enabled = conversation_memory_enabled
        self.conversation_memory_fraction = conversation_memory_fraction


def _make_conversation(
    turns: list[tuple[str, str]],
    trailing_user: str,
    *,
    double_assistant: bool = False,
) -> list[dict[str, Any]]:
    """Build a flat conversation list from (user, assistant) pairs + a trailing user msg.

    When *double_assistant* is True, each turn stores two assistant messages
    (matching compare mode's persist pattern: RLM answer then Direct answer).
    """
    msgs: list[dict[str, Any]] = []
    for user_text, assistant_text in turns:
        msgs.append({"role": "user", "content": user_text})
        msgs.append({"role": "assistant", "content": assistant_text})
        if double_assistant:
            msgs.append({"role": "assistant", "content": f"{assistant_text} (direct)"})
    msgs.append({"role": "user", "content": trailing_user})
    return msgs


class TestDisabledPaths:
    def test_disabled_provider_no_chat_provider_id(self) -> None:
        full_query, overlay, info = _prepare_history_context(
            state=FakeState(),
            session_id="s1",
            chat_provider_id=None,
            cp=FakeChatProvider(),
            mode="direct",
            adapter=FakeAdapter(),
            content="doc",
            current_query="what is this?",
        )
        assert full_query == "what is this?"
        assert overlay == {}
        assert info["path"] == HISTORY_PATH_DISABLED

    def test_disabled_when_conversation_memory_false(self) -> None:
        full_query, overlay, info = _prepare_history_context(
            state=FakeState(),
            session_id="s1",
            chat_provider_id="cp1",
            cp=FakeChatProvider(conversation_memory_enabled=False),
            mode="direct",
            adapter=FakeAdapter(),
            content="doc",
            current_query="hello",
        )
        assert full_query == "hello"
        assert overlay == {}
        assert info["path"] == HISTORY_PATH_DISABLED


class TestEmptyHistory:
    def test_empty_history(self) -> None:
        state = FakeState({"s1:cp1": [{"role": "user", "content": "current q"}]})
        full_query, overlay, info = _prepare_history_context(
            state=state,
            session_id="s1",
            chat_provider_id="cp1",
            cp=FakeChatProvider(),
            mode="direct",
            adapter=FakeAdapter(),
            content="doc",
            current_query="current q",
        )
        assert full_query == "current q"
        assert overlay == {}
        assert info["path"] == HISTORY_PATH_EMPTY
        assert info["turns_available"] == 0


class TestInpromptPath:
    def test_direct_mode_with_history(self) -> None:
        conv = _make_conversation(
            [("q1", "a1"), ("q2", "a2")],
            trailing_user="q3",
        )
        state = FakeState({"s1:cp1": conv})
        full_query, overlay, info = _prepare_history_context(
            state=state,
            session_id="s1",
            chat_provider_id="cp1",
            cp=FakeChatProvider(),
            mode="direct",
            adapter=FakeAdapter(),
            content="doc text",
            current_query="q3",
        )
        # Query is unchanged — history is in the overlay as native messages
        assert full_query == "q3"
        assert EXTRA_KEY_HISTORY_MESSAGES in overlay
        msgs = overlay[EXTRA_KEY_HISTORY_MESSAGES]
        assert len(msgs) == 4  # 2 turns × 2 messages
        assert msgs[0] == {"role": "user", "content": "q1"}
        assert msgs[1] == {"role": "assistant", "content": "a1"}
        assert info["path"] == HISTORY_PATH_INPROMPT
        assert info["mode"] == "direct"
        assert info["conversation_memory_enabled"] is True
        assert info["turns_available"] == 2
        assert info["history_turns_used"] == 2

    def test_compare_mode_uses_inprompt(self) -> None:
        conv = _make_conversation([("hi", "hello")], trailing_user="bye")
        state = FakeState({"s1:cp1": conv})
        full_query, overlay, info = _prepare_history_context(
            state=state,
            session_id="s1",
            chat_provider_id="cp1",
            cp=FakeChatProvider(),
            mode="compare",
            adapter=FakeAdapter(),
            content="doc",
            current_query="bye",
        )
        assert info["path"] == HISTORY_PATH_INPROMPT
        assert full_query == "bye"
        assert EXTRA_KEY_HISTORY_MESSAGES in overlay

    def test_compare_mode_populates_both_history_keys(self) -> None:
        """Compare runs Direct + RLM with the same config.extra.

        Direct reads EXTRA_KEY_HISTORY_MESSAGES, RLM reads
        EXTRA_KEY_HISTORY_VARIABLE.  Both must be present so
        neither half is blind to prior turns.
        """
        conv = _make_conversation([("q1", "a1")], trailing_user="q2")
        state = FakeState({"s1:cp1": conv})
        _, overlay, _ = _prepare_history_context(
            state=state,
            session_id="s1",
            chat_provider_id="cp1",
            cp=FakeChatProvider(),
            mode="compare",
            adapter=FakeAdapter(),
            content="doc",
            current_query="q2",
        )
        # Direct half
        assert EXTRA_KEY_HISTORY_MESSAGES in overlay
        assert len(overlay[EXTRA_KEY_HISTORY_MESSAGES]) == 2  # user + assistant
        # RLM half
        assert EXTRA_KEY_HISTORY_VARIABLE in overlay
        assert overlay[EXTRA_KEY_HISTORY_VARIABLE][0]["user"] == "q1"

    def test_rag_mode_uses_inprompt(self) -> None:
        conv = _make_conversation([("q1", "a1")], trailing_user="q2")
        state = FakeState({"s1:cp1": conv})
        full_query, overlay, info = _prepare_history_context(
            state=state,
            session_id="s1",
            chat_provider_id="cp1",
            cp=FakeChatProvider(),
            mode="rag",
            adapter=FakeAdapter(),
            content="doc",
            current_query="q2",
        )
        assert info["path"] == HISTORY_PATH_INPROMPT
        assert full_query == "q2"
        assert EXTRA_KEY_HISTORY_MESSAGES in overlay
        assert len(overlay[EXTRA_KEY_HISTORY_MESSAGES]) == 2

    def test_history_info_contains_expected_fields(self) -> None:
        conv = _make_conversation([("q1", "a1")], trailing_user="q2")
        state = FakeState({"s1:cp1": conv})
        _, _, info = _prepare_history_context(
            state=state,
            session_id="s1",
            chat_provider_id="cp1",
            cp=FakeChatProvider(conversation_memory_fraction=0.25),
            mode="direct",
            adapter=FakeAdapter(context_window=4096),
            content="doc",
            current_query="q2",
        )
        expected_keys = {
            "path",
            "mode",
            "conversation_memory_enabled",
            "turns_available",
            "history_turns_used",
            "history_turns_dropped",
            "history_tokens_used",
            "history_budget_tokens",
            "context_window",
            "conversation_memory_fraction",
        }
        assert expected_keys == set(info.keys())
        assert info["context_window"] == 4096
        assert info["conversation_memory_fraction"] == 0.25


class TestCompareFollowUp:
    """Pin the two compare-specific fixes from review findings."""

    def test_compare_follow_up_with_double_assistant(self) -> None:
        """Compare stores two assistant messages per turn (RLM + Direct).

        The helper must still produce a coherent in-prompt prefix on
        follow-up, keeping the first assistant answer per turn and
        not desyncing the extractor.
        """
        conv = _make_conversation(
            [("Q0", "A0-rlm"), ("Q1", "A1-rlm")],
            trailing_user="Q2",
            double_assistant=True,
        )
        state = FakeState({"s1:cp1": conv})
        full_query, _, info = _prepare_history_context(
            state=state,
            session_id="s1",
            chat_provider_id="cp1",
            cp=FakeChatProvider(),
            mode="compare",
            adapter=FakeAdapter(),
            content="doc",
            current_query="Q2",
        )
        assert info["path"] == HISTORY_PATH_INPROMPT
        assert info["turns_available"] == 2
        assert info["history_turns_used"] == 2
        # The overlay messages must contain the RLM answers, not the "(direct)" ones
        _, overlay, _ = _prepare_history_context(
            state=state,
            session_id="s1",
            chat_provider_id="cp1",
            cp=FakeChatProvider(),
            mode="compare",
            adapter=FakeAdapter(),
            content="doc",
            current_query="Q2",
        )
        msgs = overlay[EXTRA_KEY_HISTORY_MESSAGES]
        assistant_contents = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert "A0-rlm" in assistant_contents
        assert "A1-rlm" in assistant_contents
        assert all("(direct)" not in c for c in assistant_contents)

    def test_compare_budget_uses_rlm_system_prompt(self) -> None:
        """Compare budget must be computed against the RLM prompt, not Direct.

        The RLM system prompt is larger; using the Direct prompt would
        underestimate cost for the RLM half of compare, risking overflow
        on small-window providers.
        """
        conv = _make_conversation([("q1", "a1")], trailing_user="q2")
        state = FakeState({"s1:cp1": conv})

        # Run with a tight context window so the budget difference matters.
        _, _, info_compare = _prepare_history_context(
            state=state,
            session_id="s1",
            chat_provider_id="cp1",
            cp=FakeChatProvider(),
            mode="compare",
            adapter=FakeAdapter(context_window=512),
            content="some content",
            current_query="q2",
        )
        _, _, info_direct = _prepare_history_context(
            state=state,
            session_id="s1",
            chat_provider_id="cp1",
            cp=FakeChatProvider(),
            mode="direct",
            adapter=FakeAdapter(context_window=512),
            content="some content",
            current_query="q2",
        )
        # Compare should have a tighter (or equal) budget than direct
        # because it uses the larger RLM system prompt for budgeting.
        assert info_compare["history_budget_tokens"] <= info_direct["history_budget_tokens"]

    def test_compare_budget_includes_system_prompt_extra(self) -> None:
        """When a profile injects system_prompt_extra, the budget shrinks."""
        conv = _make_conversation([("q1", "a1")], trailing_user="q2")
        state = FakeState({"s1:cp1": conv})

        _, _, info_no_extra = _prepare_history_context(
            state=state,
            session_id="s1",
            chat_provider_id="cp1",
            cp=FakeChatProvider(),
            mode="compare",
            adapter=FakeAdapter(context_window=2048),
            content="doc",
            current_query="q2",
            system_prompt_extra="",
        )
        _, _, info_with_extra = _prepare_history_context(
            state=state,
            session_id="s1",
            chat_provider_id="cp1",
            cp=FakeChatProvider(),
            mode="compare",
            adapter=FakeAdapter(context_window=2048),
            content="doc",
            current_query="q2",
            system_prompt_extra="x" * 2000,  # large extra prompt
        )
        # With a large extra prompt, the budget should be tighter
        assert info_with_extra["history_budget_tokens"] <= info_no_extra["history_budget_tokens"]


class TestSystemPromptContainsHistoryParagraph:
    """Pin that the RLM system prompt tells the model about the `history` variable."""

    def test_rlm_system_prompt_mentions_history(self) -> None:
        from rlmstudio.prompts import format_system_prompt

        prompt = format_system_prompt(prompt_length=1000)
        assert "history" in prompt
        # The example uses v1 Python code (not a v2 JSON inspect action)
        assert "print(history[-1])" in prompt
        # Must NOT contain the invalid v2 JSON form
        assert '"code": "print(history' not in prompt

    def test_rlm_system_prompt_version_is_2_2(self) -> None:
        """2.2 added ``read_result`` (specs/rlm-working-state); 2.1 added history."""
        import importlib.resources as pkg_resources

        import yaml

        pkg = pkg_resources.files("rlmstudio.prompts")
        raw = (pkg / "system_prompt_v2_0.yaml").read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        assert data["version"] == "2.2"

    def test_history_paragraph_warns_against_tool_use(self) -> None:
        """The prompt must warn not to pass history to peek/grep/outline_file."""
        from rlmstudio.prompts import format_system_prompt

        prompt = format_system_prompt(prompt_length=1000)
        assert "Do NOT pass" in prompt
        assert "only work on" in prompt


class TestReplVariablePath:
    @pytest.mark.parametrize("mode", ["rlm", "auto"])
    def test_mode_returns_repl_variable(self, mode: str) -> None:
        conv = _make_conversation([("q1", "a1")], trailing_user="q2")
        state = FakeState({"s1:cp1": conv})
        full_query, overlay, info = _prepare_history_context(
            state=state,
            session_id="s1",
            chat_provider_id="cp1",
            cp=FakeChatProvider(),
            mode=mode,
            adapter=FakeAdapter(),
            content="doc",
            current_query="q2",
        )
        assert full_query == "q2"  # query unchanged — history is not in the prompt
        assert info["path"] == HISTORY_PATH_REPL_VARIABLE
        assert info["turns_available"] == 1

    def test_repl_overlay_contains_history_variable(self) -> None:
        """The extra_overlay must contain the history list for sandbox binding."""
        conv = _make_conversation(
            [("What is X?", "X is foo"), ("And Y?", "Y is bar")],
            trailing_user="current",
        )
        state = FakeState({"s1:cp1": conv})
        _, overlay, info = _prepare_history_context(
            state=state,
            session_id="s1",
            chat_provider_id="cp1",
            cp=FakeChatProvider(),
            mode="rlm",
            adapter=FakeAdapter(),
            content="doc",
            current_query="current",
        )
        # overlay carries the history_variable for the use case to bind
        assert EXTRA_KEY_HISTORY_VARIABLE in overlay
        hv = overlay[EXTRA_KEY_HISTORY_VARIABLE]
        assert isinstance(hv, list)
        assert len(hv) == 2
        assert hv[0] == {"turn": 0, "user": "What is X?", "assistant": "X is foo"}
        assert hv[1] == {"turn": 1, "user": "And Y?", "assistant": "Y is bar"}
        # info reports the REPL-variable diagnostics
        assert info["history_variable_turn_count"] == 2
        assert info["history_variable_byte_size"] > 0
        assert info["history_turns_evicted"] == 0

    def test_repl_path_query_unchanged(self) -> None:
        """The REPL path must NOT prepend 'Previous conversation:' to the query."""
        conv = _make_conversation([("q1", "a1")] * 10, trailing_user="current")
        state = FakeState({"s1:cp1": conv})
        full_query, _, _ = _prepare_history_context(
            state=state,
            session_id="s1",
            chat_provider_id="cp1",
            cp=FakeChatProvider(),
            mode="rlm",
            adapter=FakeAdapter(),
            content="doc",
            current_query="current",
        )
        assert full_query == "current"
        assert "Previous conversation" not in full_query
