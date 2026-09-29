"""Benchmark scoring: match rules and the pointwise judge (specs/benchmarks-v1 G3, FR-3)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from rlmstudio.application.dto import LLMResponseDTO
from rlmstudio.benchmark.dataset import MATCH_CONTAINS, MATCH_EXACT
from rlmstudio.benchmark.scoring import (
    ANY_OF_SEPARATOR,
    JudgeScorer,
    matches,
    normalise,
    parse_judge_json,
)


class TestNormaliseAndMatch:
    def test_normalise(self) -> None:
        assert normalise("  The  Harbour\nPlatform team. ") == "the harbour platform team"
        assert normalise('"17"') == "17"

    def test_contains_is_case_and_whitespace_insensitive(self) -> None:
        assert matches(
            "It is owned by the HARBOUR   platform team.", "Harbour Platform", MATCH_CONTAINS
        )

    def test_exact_requires_the_whole_answer(self) -> None:
        assert matches("17", "17", MATCH_EXACT)
        assert not matches("17 times", "17", MATCH_EXACT)
        assert matches("17 times", "17", MATCH_CONTAINS)

    def test_any_of_alternatives(self) -> None:
        expected = ANY_OF_SEPARATOR.join(["not mention", "does not", "no information"])
        assert matches("The document does not name a CFO.", expected, MATCH_CONTAINS)
        assert not matches("The CFO is Jane Doe.", expected, MATCH_CONTAINS)

    def test_empty_alternative_never_matches_everything(self) -> None:
        assert not matches("anything", " || ", MATCH_CONTAINS)


class TestParseJudgeJson:
    def test_plain_and_fenced(self) -> None:
        assert parse_judge_json('{"a": 1}') == {"a": 1}
        assert parse_judge_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_garbage_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_judge_json("not json")


class _ScriptedLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        self.prompts.append(messages[-1]["content"])
        return LLMResponseDTO(content=self.reply, model="judge", input_tokens=1, output_tokens=1)

    def complete_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        yield self.reply

    def count_tokens(self, text: str) -> int:
        return 1

    def get_pricing(self) -> dict[str, float]:
        return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}


_GOOD = (
    '{"dimensions": {"relevance": 5, "correctness": 4, "completeness": 3, '
    '"coherence": 5, "conciseness": 4}, "reasoning": "solid"}'
)


class TestJudgeScorer:
    def test_scores_with_provenance(self) -> None:
        llm = _ScriptedLLM(_GOOD)
        judge = JudgeScorer(llm, model="openai/gpt-4o-mini")

        verdict = judge.score(query="q", response="a", source="doc")

        assert verdict.overall == 4.2
        assert verdict.dimensions["correctness"] == 4.0
        assert verdict.reasoning == "solid"
        assert verdict.model == "openai/gpt-4o-mini"
        assert verdict.prompt_version == "2.0"  # judge_pointwise.yaml
        assert verdict.parsed
        assert verdict.to_dict()["prompt_version"] == "2.0"

    def test_prompt_carries_query_response_source_and_rubric_hint(self) -> None:
        llm = _ScriptedLLM(_GOOD)
        JudgeScorer(llm, model="m").score(
            query="How many?", response="17", source="the doc", rubric_hint="expect 17"
        )

        prompt = llm.prompts[0]
        assert "How many?" in prompt
        assert "Grading guidance for the judge: expect 17" in prompt
        assert "17" in prompt
        assert "the doc" in prompt

    def test_long_source_is_truncated_like_the_studio_judge(self) -> None:
        llm = _ScriptedLLM(_GOOD)
        JudgeScorer(llm, model="m").score(query="q", response="a", source="x" * 20_000)

        assert "Source truncated at 8,000 characters" in llm.prompts[0]

    def test_clamps_out_of_range_dimensions(self) -> None:
        llm = _ScriptedLLM('{"dimensions": {"relevance": 9, "correctness": -2}, "reasoning": ""}')
        verdict = JudgeScorer(llm, model="m").score(query="q", response="a", source="s")

        assert verdict.dimensions == {"relevance": 5.0, "correctness": 1.0}
        assert verdict.overall == 3.0

    def test_unparseable_reply_falls_back_and_is_flagged(self) -> None:
        llm = _ScriptedLLM("I refuse to output JSON")
        verdict = JudgeScorer(llm, model="m").score(query="q", response="a", source="s")

        assert not verdict.parsed
        assert verdict.overall == 3.0
        assert "Failed to parse" in verdict.reasoning


class TestNumericExpectationsMatchOnBoundaries:
    """Every counting case expects a bare number under ``contains``.

    A plain substring test scores a wrong count as correct — "72" is inside
    "172" — which inflates accuracy on exactly the cases meant to be hardest.
    """

    @pytest.mark.parametrize(
        ("answer", "expected", "passes"),
        [
            ("It appears 72 times.", "72", True),
            ("The count is 72", "72", True),
            ("72", "72", True),
            ("It occurs seventy-two (72) times", "72", True),
            ("It appears 172 times.", "72", False),
            ("I found 1,072 matches", "72", False),
            ("appears 72,000 times", "72", False),
            ("the answer is 72.5", "72", False),
        ],
    )
    def test_a_count_must_appear_as_its_own_number(
        self, answer: str, expected: str, passes: bool
    ) -> None:
        assert matches(answer, expected, MATCH_CONTAINS) is passes

    def test_a_thousands_separated_expectation_is_matched_as_written(self) -> None:
        assert matches("There are 1,234 of them", "1,234", MATCH_CONTAINS)
        assert not matches("There are 11,234 of them", "1,234", MATCH_CONTAINS)

    def test_text_expectations_are_still_plain_substrings(self) -> None:
        assert matches("The answer is Netherfield Park.", "netherfield", MATCH_CONTAINS)


class TestJudgeRepliesThatAreNotObjects:
    def test_a_json_list_is_rejected_rather_than_returned(self) -> None:
        with pytest.raises(ValueError, match="not a JSON object"):
            parse_judge_json("[4, 4]")

    def test_a_bare_number_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a JSON object"):
            parse_judge_json("4")

    def test_an_object_in_a_fence_still_parses(self) -> None:
        assert parse_judge_json('```json\n{"dimensions": {"a": 4}}\n```') == {
            "dimensions": {"a": 4}
        }
