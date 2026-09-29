"""Tests for benchmark dataset format and loader."""

from pathlib import Path

import pytest

from rlmstudio.benchmark.dataset import (
    BenchmarkCase,
    BenchmarkDataset,
    load_dataset,
    load_dataset_from_dict,
)

# ---------------------------------------------------------------------------
# BenchmarkCase
# ---------------------------------------------------------------------------


class TestBenchmarkCase:
    def test_defaults(self):
        case = BenchmarkCase(id="c1", content="hello", query="what?")
        assert case.id == "c1"
        assert case.content == "hello"
        assert case.query == "what?"
        assert case.expected_answer is None
        assert case.category == "general"
        assert case.difficulty == "medium"
        assert case.tags == []

    def test_content_length(self):
        case = BenchmarkCase(id="c1", content="abcdef", query="q")
        assert case.content_length == 6

    def test_to_dict(self):
        case = BenchmarkCase(
            id="c1",
            content="text",
            query="q",
            expected_answer="a",
            category="factual",
            difficulty="hard",
            tags=["t1"],
        )
        d = case.to_dict()
        assert d["id"] == "c1"
        assert d["content_length"] == 4
        assert d["expected_answer"] == "a"
        assert d["category"] == "factual"
        assert d["difficulty"] == "hard"
        assert d["tags"] == ["t1"]


# ---------------------------------------------------------------------------
# BenchmarkDataset
# ---------------------------------------------------------------------------


class TestBenchmarkDataset:
    def _make_dataset(self):
        cases = [
            BenchmarkCase(
                id="a",
                content="x",
                query="q1",
                category="factual",
                difficulty="easy",
                tags=["short"],
            ),
            BenchmarkCase(
                id="b",
                content="yy",
                query="q2",
                category="factual",
                difficulty="hard",
                tags=["long"],
            ),
            BenchmarkCase(
                id="c",
                content="zzz",
                query="q3",
                category="analytical",
                difficulty="easy",
                tags=["short"],
            ),
        ]
        return BenchmarkDataset(name="test", description="desc", cases=cases)

    def test_len(self):
        ds = self._make_dataset()
        assert len(ds) == 3

    def test_iter(self):
        ds = self._make_dataset()
        ids = [c.id for c in ds]
        assert ids == ["a", "b", "c"]

    def test_getitem(self):
        ds = self._make_dataset()
        assert ds[1].id == "b"

    def test_filter_by_category(self):
        ds = self._make_dataset().filter_by_category("factual")
        assert len(ds) == 2
        assert all(c.category == "factual" for c in ds)

    def test_filter_by_difficulty(self):
        ds = self._make_dataset().filter_by_difficulty("easy")
        assert len(ds) == 2
        assert all(c.difficulty == "easy" for c in ds)

    def test_filter_by_tag(self):
        ds = self._make_dataset().filter_by_tag("short")
        assert len(ds) == 2
        assert all("short" in c.tags for c in ds)

    def test_categories(self):
        ds = self._make_dataset()
        assert ds.categories == ["analytical", "factual"]

    def test_to_dict(self):
        ds = self._make_dataset()
        d = ds.to_dict()
        assert d["name"] == "test"
        assert d["case_count"] == 3
        assert len(d["cases"]) == 3


# ---------------------------------------------------------------------------
# YAML Loader
# ---------------------------------------------------------------------------


class TestLoadDataset:
    def test_load_valid_yaml(self, tmp_path):
        yaml_content = """
name: "Test Bench"
description: "A test"
cases:
  - id: "c1"
    content: "hello world"
    query: "what?"
    expected_answer: "hello"
    category: "factual"
    difficulty: "easy"
    tags: ["short"]
  - id: "c2"
    content: "foo bar"
    query: "who?"
"""
        f = tmp_path / "bench.yaml"
        f.write_text(yaml_content)

        ds = load_dataset(str(f))
        assert ds.name == "Test Bench"
        assert ds.description == "A test"
        assert len(ds) == 2
        assert ds[0].id == "c1"
        assert ds[0].expected_answer == "hello"
        assert ds[1].id == "c2"
        assert ds[1].category == "general"  # default

    def test_load_auto_ids(self, tmp_path):
        yaml_content = """
name: "auto"
cases:
  - content: "a"
    query: "q"
  - content: "b"
    query: "r"
"""
        f = tmp_path / "auto.yaml"
        f.write_text(yaml_content)

        ds = load_dataset(str(f))
        assert ds[0].id == "case_0"
        assert ds[1].id == "case_1"

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/file.yaml")

    def test_load_invalid_format(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("- just a list")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_dataset(str(f))

    def test_load_missing_fields(self, tmp_path):
        yaml_content = """
name: "bad"
cases:
  - id: "c1"
    content: "hello"
"""
        f = tmp_path / "bad2.yaml"
        f.write_text(yaml_content)
        with pytest.raises(ValueError, match="must have a 'query' field"):
            load_dataset(str(f))

    def test_load_sample_benchmark(self):
        """Load the actual sample benchmark file."""
        sample = Path(__file__).parent.parent / "benchmarks" / "sample_benchmark.yaml"
        if not sample.exists():
            pytest.skip("sample_benchmark.yaml not found")
        ds = load_dataset(str(sample))
        assert ds.name == "RLM Studio Sample Benchmark"
        assert len(ds) >= 7


# ---------------------------------------------------------------------------
# Dict Loader
# ---------------------------------------------------------------------------


class TestLoadDatasetFromDict:
    def test_basic(self):
        data = {
            "name": "dict_bench",
            "cases": [
                {"content": "hello", "query": "q1"},
                {"content": "world", "query": "q2", "category": "factual"},
            ],
        }
        ds = load_dataset_from_dict(data)
        assert ds.name == "dict_bench"
        assert len(ds) == 2
        assert ds[0].id == "case_0"
        assert ds[1].category == "factual"

    def test_empty_cases(self):
        ds = load_dataset_from_dict({"name": "empty"})
        assert len(ds) == 0
