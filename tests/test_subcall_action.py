"""Tests for single and batched subcall action parsing."""

from __future__ import annotations

import json
from typing import cast

import pytest

from rlmstudio.core.actions import ParseError, SubcallAction, parse_action


def _action(payload: dict) -> SubcallAction:
    kind, action = parse_action(json.dumps(payload))
    assert kind == "subcall"
    return cast(SubcallAction, action)


class TestSingleForm:
    def test_single_prompt_and_query(self) -> None:
        action = _action(
            {
                "type": "subcall",
                "prompt": "content",
                "query": "question",
                "note": "why",
            }
        )
        assert action.prompt == "content"
        assert action.query == "question"
        assert action.note == "why"
        assert action.calls is None
        assert len(action.batch) == 1
        assert action.batch[0].prompt == "content"
        assert action.batch[0].query == "question"

    def test_single_form_keeps_positional_construction(self) -> None:
        action = SubcallAction("content", "question", "note")
        action.validate()
        assert action.max_concurrency is None


class TestBatchForm:
    def test_batch_calls_and_concurrency(self) -> None:
        action = _action(
            {
                "type": "subcall",
                "calls": [
                    {"prompt": "one", "query": "q1"},
                    {"prompt": "two", "query": "q2"},
                ],
                "max_concurrency": 2,
            }
        )
        assert action.prompt is None
        assert action.query is None
        assert action.max_concurrency == 2
        assert [(c.prompt, c.query) for c in action.batch] == [
            ("one", "q1"),
            ("two", "q2"),
        ]

    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            (
                {"type": "subcall", "calls": [], "max_concurrency": 2},
                "non-empty list",
            ),
            (
                {"type": "subcall", "calls": [{"prompt": "one"}]},
                "missing required 'query'",
            ),
            (
                {"type": "subcall", "calls": [{"query": "q"}]},
                "missing required 'prompt'",
            ),
            (
                {
                    "type": "subcall",
                    "prompt": "one",
                    "query": "q1",
                    "calls": [{"prompt": "two", "query": "q2"}],
                },
                "not both",
            ),
            (
                {
                    "type": "subcall",
                    "calls": [{"prompt": "one", "query": "q1"}],
                    "max_concurrency": 0,
                },
                "positive integer",
            ),
        ],
    )
    def test_rejects_malformed_batches(self, payload: dict, message: str) -> None:
        with pytest.raises(ParseError, match=message):
            parse_action(json.dumps(payload))
