# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Benchmark dataset format and loader for strategy evaluation."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Task types of the longdoc-v1 set (specs/benchmarks-v1 G2).
TASK_TYPE_GENERAL = "general"
TASK_TYPE_NEEDLE = "needle"
TASK_TYPE_SYNTHESIS = "synthesis"
TASK_TYPE_AGGREGATION = "aggregation"
TASK_TYPE_REFUSAL = "refusal"
TASK_TYPES: frozenset[str] = frozenset(
    {
        TASK_TYPE_GENERAL,
        TASK_TYPE_NEEDLE,
        TASK_TYPE_SYNTHESIS,
        TASK_TYPE_AGGREGATION,
        TASK_TYPE_REFUSAL,
    }
)

# How ``expected_answer`` is checked by the scorer.
MATCH_EXACT = "exact"
MATCH_CONTAINS = "contains"
MATCH_KINDS: frozenset[str] = frozenset({MATCH_EXACT, MATCH_CONTAINS})

# YAML keys that describe how to obtain a case's document (see corpus.py).
_CONTENT_SPEC_KEYS = ("content_file", "generator", "source_url", "sha256", "filename")
# Generator parameters live at the case level next to ``generator``.
_GENERATOR_KEYS = ("tokens", "seed", "title", "facts", "repeat")


@dataclass
class BenchmarkCase:
    """A single benchmark case: content + query + optional expected answer.

    ``content`` may be empty on load when the YAML gives a ``content_spec``
    instead; :func:`rlmstudio.benchmark.corpus.materialise` fills it in.
    """

    id: str
    content: str
    query: str
    expected_answer: str | None = None
    category: str = "general"
    difficulty: str = "medium"  # easy, medium, hard
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # --- longdoc-v1 additions (specs/benchmarks-v1 FR-2) ---
    task_type: str = TASK_TYPE_GENERAL
    min_tokens: int | None = None  # the size bucket the case is meant to occupy
    rubric_hint: str | None = None  # extra guidance handed to the judge
    match: str = MATCH_CONTAINS  # how expected_answer is checked
    budget: dict[str, Any] = field(default_factory=dict)  # RunConfigDTO overrides
    source: str | None = None  # id into BenchmarkDataset.sources
    content_spec: dict[str, Any] | None = None  # how to materialise content
    expected_spec: dict[str, Any] | None = None  # how to derive expected_answer

    @property
    def content_length(self) -> int:
        return len(self.content)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "content_length": self.content_length,
            "expected_answer": self.expected_answer,
            "category": self.category,
            "difficulty": self.difficulty,
            "tags": self.tags,
            "metadata": self.metadata,
            "task_type": self.task_type,
            "min_tokens": self.min_tokens,
            "rubric_hint": self.rubric_hint,
            "match": self.match,
            "budget": self.budget,
            "source": self.source,
        }


@dataclass
class BenchmarkDataset:
    """A collection of benchmark cases with metadata."""

    name: str
    description: str = ""
    cases: list[BenchmarkCase] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    sources: list[dict[str, Any]] = field(default_factory=list)  # {id, title, url, license}

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[BenchmarkCase]:
        return iter(self.cases)

    def __getitem__(self, index: int) -> BenchmarkCase:
        return self.cases[index]

    def filter_by_category(self, category: str) -> "BenchmarkDataset":
        """Return a new dataset with only cases from the given category."""
        filtered = [c for c in self.cases if c.category == category]
        return BenchmarkDataset(
            name=f"{self.name} [{category}]",
            description=self.description,
            cases=filtered,
            metadata=self.metadata,
        )

    def filter_by_difficulty(self, difficulty: str) -> "BenchmarkDataset":
        """Return a new dataset with only cases of the given difficulty."""
        filtered = [c for c in self.cases if c.difficulty == difficulty]
        return BenchmarkDataset(
            name=f"{self.name} [{difficulty}]",
            description=self.description,
            cases=filtered,
            metadata=self.metadata,
        )

    def filter_by_tag(self, tag: str) -> "BenchmarkDataset":
        """Return a new dataset with only cases containing the given tag."""
        filtered = [c for c in self.cases if tag in c.tags]
        return BenchmarkDataset(
            name=f"{self.name} [#{tag}]",
            description=self.description,
            cases=filtered,
            metadata=self.metadata,
        )

    @property
    def categories(self) -> list[str]:
        return sorted({c.category for c in self.cases})

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "case_count": len(self.cases),
            "categories": self.categories,
            "cases": [c.to_dict() for c in self.cases],
            "metadata": self.metadata,
            "sources": self.sources,
        }


def _case_from_dict(index: int, case_data: Any) -> BenchmarkCase:
    """Validate one YAML case mapping and build a :class:`BenchmarkCase`."""
    if not isinstance(case_data, dict):
        raise ValueError(f"Case {index} must be a mapping")
    if "query" not in case_data:
        raise ValueError(f"Case {index} must have a 'query' field")

    content_spec = {k: case_data[k] for k in _CONTENT_SPEC_KEYS if k in case_data}
    if "generator" in content_spec:
        content_spec.update({k: case_data[k] for k in _GENERATOR_KEYS if k in case_data})
    if "content" not in case_data and not content_spec:
        raise ValueError(
            f"Case {index} must have 'content' or one of {', '.join(_CONTENT_SPEC_KEYS[:3])}"
        )

    task_type = case_data.get("task_type", TASK_TYPE_GENERAL)
    if task_type not in TASK_TYPES:
        raise ValueError(f"Case {index}: unknown task_type {task_type!r}")
    match = case_data.get("match", MATCH_CONTAINS)
    if match not in MATCH_KINDS:
        raise ValueError(f"Case {index}: unknown match {match!r}")

    expected = case_data.get("expected")
    expected_spec = expected if isinstance(expected, dict) else None
    expected_answer = case_data.get("expected_answer")
    if expected_answer is None and isinstance(expected, str):
        expected_answer = expected

    return BenchmarkCase(
        id=case_data.get("id", f"case_{index}"),
        content=case_data.get("content", ""),
        query=case_data["query"],
        expected_answer=expected_answer,
        category=case_data.get("category", "general"),
        difficulty=case_data.get("difficulty", "medium"),
        tags=case_data.get("tags", []),
        metadata=case_data.get("metadata", {}),
        task_type=task_type,
        min_tokens=case_data.get("min_tokens"),
        rubric_hint=case_data.get("rubric_hint"),
        match=match,
        budget=case_data.get("budget", {}),
        source=case_data.get("source"),
        content_spec=content_spec or None,
        expected_spec=expected_spec,
    )


def load_dataset(path: str) -> BenchmarkDataset:
    """Load a benchmark dataset from a YAML file.

    Expected YAML format::

        name: "My Benchmark"
        description: "Testing strategy performance"
        cases:
          - id: "case_1"
            content: "..."
            query: "..."
            expected_answer: "..."  # optional
            category: "factual"     # optional
            difficulty: "easy"      # optional
            tags: ["short"]         # optional
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(filepath) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Dataset file must contain a YAML mapping, got {type(data).__name__}")

    dataset = load_dataset_from_dict(data)
    if dataset.name == "unnamed":
        dataset.name = filepath.stem
    return dataset


def load_dataset_from_dict(data: dict[str, Any]) -> BenchmarkDataset:
    """Load a benchmark dataset from an in-memory dictionary (same schema as YAML)."""
    cases = [_case_from_dict(i, case_data) for i, case_data in enumerate(data.get("cases", []))]
    return BenchmarkDataset(
        name=data.get("name", "unnamed"),
        description=data.get("description", ""),
        cases=cases,
        metadata=data.get("metadata", {}),
        sources=list(data.get("sources", []) or []),
    )
