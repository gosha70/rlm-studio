# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Score benchmark answers: exact / contains matching and the pointwise LLM judge.

Two independent signals, reported side by side (specs/benchmarks-v1 G3):

- **match** — a cheap, deterministic accuracy proxy for cases that carry an
  ``expected_answer`` (needle, aggregation, refusal). Both sides are
  normalised (case, whitespace, edge punctuation); ``"a || b"`` lists
  alternatives, any of which passes.
- **judge** — the same ``judge_pointwise.yaml`` rubric the Studio judge uses,
  sent through an :class:`LLMPort`, with the case's ``rubric_hint`` appended
  to the query as grading guidance. The judge model and prompt version are
  recorded on every verdict.
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rlmstudio.application.ports.llm_port import LLMPort
from rlmstudio.prompts import templates as _prompt_templates

from .dataset import MATCH_CONTAINS, MATCH_EXACT

ANY_OF_SEPARATOR = " || "
JUDGE_POINTWISE_PROMPT = "judge_pointwise.yaml"
_PROMPTS_DIR = Path(_prompt_templates.__file__).parent
_SOURCE_CAP_CHARS = 8000  # same cap as the Studio judge
_FALLBACK_SCORE = 3.0
_DIMENSION_MIN = 1.0
_DIMENSION_MAX = 5.0
_PUNCTUATION = string.punctuation + "“”‘’"
# An expectation that is nothing but a number, with optional thousands
# separators or decimals — the shape every counting case uses.
_NUMERIC_EXPECTATION = re.compile(r"\d[\d.,]*")


def normalise(text: str) -> str:
    """Casefold, collapse whitespace, strip edge punctuation."""
    collapsed = re.sub(r"\s+", " ", text.casefold()).strip()
    return collapsed.strip(_PUNCTUATION + " ")


def matches(answer: str, expected: str, match: str) -> bool:
    """Whether *answer* satisfies *expected* under the ``match`` kind.

    A numeric expectation is matched on token boundaries, never as a bare
    substring: every counting case expects a bare number under ``contains``,
    and ``"72" in "it appears 172 times"`` would otherwise score a wrong count
    as correct and inflate the accuracy column on exactly those cases.
    """
    got = normalise(answer)
    for alternative in expected.split(ANY_OF_SEPARATOR):
        want = normalise(alternative)
        if not want:
            continue
        if match == MATCH_EXACT and got == want:
            return True
        if match == MATCH_CONTAINS and _contains(got, want):
            return True
    return False


def _contains(got: str, want: str) -> bool:
    """Substring test, tightened to number boundaries for a numeric expectation.

    The boundary has to separate "part of a longer number" from "followed by
    punctuation", which is why the trailing side rejects a digit or a
    separator *followed by a digit* rather than any separator: a count that
    ends a sentence or a clause — ``"77. This includes…"``, ``"17, by my
    count"`` — is a correct answer, and rejecting it would understate accuracy
    as surely as the plain substring test overstated it.  Punctuation is only
    stripped from the ends of the whole answer, so it is still there mid-text.
    """
    if _NUMERIC_EXPECTATION.fullmatch(want):
        pattern = rf"(?<![\d.,]){re.escape(want)}(?![\d]|[.,]\d)"
        return re.search(pattern, got) is not None
    return want in got


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


@dataclass
class JudgeVerdict:
    """Pointwise judge output, with provenance."""

    overall: float
    dimensions: dict[str, float]
    reasoning: str
    model: str
    prompt_version: str
    parsed: bool = True
    raw: str = field(default="", repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "dimensions": self.dimensions,
            "reasoning": self.reasoning,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "parsed": self.parsed,
        }


def parse_judge_json(text: str) -> dict[str, Any]:
    """Extract the JSON object from a judge reply, tolerating markdown fences.

    Raises ``ValueError`` when the reply parses to anything but an object — a
    bare list or number is as unusable as malformed JSON, and the caller's
    fallback path expects to hear about it rather than receive the wrong type.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    decoded = json.loads(text)
    if not isinstance(decoded, dict):
        raise ValueError(f"judge reply is {type(decoded).__name__}, not a JSON object")
    result: dict[str, Any] = decoded
    return result


def _load_prompt(path: Path) -> tuple[str, str]:
    """Return ``(template, version)`` from a prompt YAML."""
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return str(data["template"]), str(data.get("version", "unknown"))


class JudgeScorer:
    """Pointwise LLM judge over the Studio rubric.

    Args:
        llm: Adapter for the judge model (any :class:`LLMPort`).
        model: Identifier recorded on verdicts (e.g. ``"openai/gpt-4o-mini"``).
        prompt_path: Rubric YAML; defaults to Studio's ``judge_pointwise.yaml``.
    """

    def __init__(
        self,
        llm: LLMPort,
        *,
        model: str,
        prompt_path: Path | None = None,
    ) -> None:
        self._llm = llm
        self.model = model
        self._template, self.prompt_version = _load_prompt(
            prompt_path or _PROMPTS_DIR / JUDGE_POINTWISE_PROMPT
        )

    def score(
        self,
        *,
        query: str,
        response: str,
        source: str,
        rubric_hint: str | None = None,
    ) -> JudgeVerdict:
        if len(source) > _SOURCE_CAP_CHARS:
            source_block = (
                source[:_SOURCE_CAP_CHARS]
                + f"\n\n[Source truncated at {_SOURCE_CAP_CHARS:,} characters"
                f" — full document is {len(source):,} characters]"
            )
        else:
            source_block = source or "Not provided — evaluate based on the response alone."
        graded_query = query
        if rubric_hint:
            graded_query = f"{query}\n\nGrading guidance for the judge: {rubric_hint}"
        prompt = self._template.format(
            query=graded_query,
            response=response,
            source_document=source_block,
        )
        raw = self._llm.complete([{"role": "user", "content": prompt}]).content
        try:
            parsed = parse_judge_json(raw)
            parsed_ok = True
        except (json.JSONDecodeError, ValueError):
            parsed = {"dimensions": {}, "reasoning": f"Failed to parse judge response: {raw[:200]}"}
            parsed_ok = False

        dimensions = {
            key: max(_DIMENSION_MIN, min(_DIMENSION_MAX, float(value)))
            for key, value in (parsed.get("dimensions") or {}).items()
            if isinstance(value, (int, float))
        }
        if not dimensions:
            dimensions = {"overall": _FALLBACK_SCORE}
            parsed_ok = False
        overall = round(sum(dimensions.values()) / len(dimensions), 2)
        return JudgeVerdict(
            overall=overall,
            dimensions=dimensions,
            reasoning=str(parsed.get("reasoning", "")),
            model=self.model,
            prompt_version=self.prompt_version,
            parsed=parsed_ok,
            raw=raw,
        )
