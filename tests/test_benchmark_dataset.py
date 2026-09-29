"""Benchmark dataset schema, loader and corpus materialisation (specs/benchmarks-v1 Phase 1)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from rlmstudio.benchmark.corpus import (
    CorpusError,
    MaterialiseReport,
    count_word,
    derive_expected,
    estimate_tokens,
    generate_synthetic,
    materialise,
    materialise_case,
)
from rlmstudio.benchmark.dataset import (
    MATCH_CONTAINS,
    MATCH_EXACT,
    TASK_TYPE_AGGREGATION,
    TASK_TYPE_NEEDLE,
    TASK_TYPE_REFUSAL,
    TASK_TYPE_SYNTHESIS,
    BenchmarkCase,
    load_dataset,
    load_dataset_from_dict,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LONGDOC = REPO_ROOT / "benchmarks" / "longdoc-v1.yaml"

FACT = "The designated maintenance window for the Orion cluster is 03:15 UTC on the second Tuesday."


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class TestLoader:
    def test_legacy_inline_case_still_loads(self) -> None:
        ds = load_dataset_from_dict(
            {"name": "x", "cases": [{"id": "a", "content": "doc", "query": "q"}]}
        )
        case = ds.cases[0]
        assert case.content == "doc"
        assert case.task_type == "general"
        assert case.match == MATCH_CONTAINS
        assert case.content_spec is None

    def test_new_fields_are_read(self) -> None:
        ds = load_dataset_from_dict(
            {
                "name": "x",
                "sources": [{"id": "s", "title": "T", "url": "u", "license": "MIT"}],
                "cases": [
                    {
                        "id": "a",
                        "query": "q",
                        "generator": "synthetic",
                        "tokens": 5000,
                        "seed": 1,
                        "task_type": TASK_TYPE_NEEDLE,
                        "min_tokens": 5000,
                        "rubric_hint": "look for the window",
                        "match": MATCH_EXACT,
                        "budget": {"max_steps": 8},
                        "source": "s",
                        "expected": {"count_word": "Orion"},
                    }
                ],
            }
        )
        case = ds.cases[0]
        assert ds.sources[0]["id"] == "s"
        assert case.content == ""
        assert case.content_spec == {"generator": "synthetic", "tokens": 5000, "seed": 1}
        assert case.expected_spec == {"count_word": "Orion"}
        assert case.expected_answer is None
        assert (case.task_type, case.min_tokens, case.match) == (
            TASK_TYPE_NEEDLE,
            5000,
            MATCH_EXACT,
        )
        assert case.budget == {"max_steps": 8}
        assert case.to_dict()["task_type"] == TASK_TYPE_NEEDLE

    def test_expected_string_shorthand(self) -> None:
        ds = load_dataset_from_dict({"cases": [{"content": "d", "query": "q", "expected": "42"}]})
        assert ds.cases[0].expected_answer == "42"

    @pytest.mark.parametrize(
        ("case", "message"),
        [
            ({"content": "d"}, "must have a 'query'"),
            ({"query": "q"}, "must have 'content' or one of"),
            ({"content": "d", "query": "q", "task_type": "magic"}, "unknown task_type"),
            ({"content": "d", "query": "q", "match": "fuzzy"}, "unknown match"),
            ("not a mapping", "must be a mapping"),
        ],
    )
    def test_validation(self, case: object, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            load_dataset_from_dict({"cases": [case]})


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------


class TestSyntheticGenerator:
    def test_is_deterministic_and_sized(self) -> None:
        spec = {"generator": "synthetic", "tokens": 5000, "seed": 7}
        a = generate_synthetic(spec)
        b = generate_synthetic(spec)
        assert a == b
        assert hashlib.sha256(a.encode()).hexdigest() == hashlib.sha256(b.encode()).hexdigest()
        assert 5000 <= estimate_tokens(a) < 5000 + 2000  # grown section by section

    def test_seed_changes_the_body(self) -> None:
        assert generate_synthetic({"tokens": 3000, "seed": 1}) != generate_synthetic(
            {"tokens": 3000, "seed": 2}
        )

    def test_fact_lands_in_the_middle_as_its_own_paragraph(self) -> None:
        text = generate_synthetic(
            {"tokens": 20000, "seed": 3, "facts": [{"sentence": FACT, "position": "middle"}]}
        )
        assert text.count(FACT) == 1
        position = text.index(FACT) / len(text)
        assert 0.4 < position < 0.6
        assert f"\n\n{FACT}\n" in text

    def test_repeat_spreads_exactly_n_times(self) -> None:
        sentence = "Incident INC-4471 was escalated to the Kestrel on-call rotation."
        text = generate_synthetic(
            {"tokens": 20000, "seed": 4, "repeat": {"sentence": sentence, "times": 17}}
        )
        assert text.count(sentence) == 17
        assert count_word(text, "INC-4471") == 17

    def test_unknown_generator_rejected(self, tmp_path: Path) -> None:
        case = BenchmarkCase(id="c", content="", query="q", content_spec={"generator": "lorem"})
        with pytest.raises(CorpusError, match="unknown generator"):
            materialise_case(case, base_dir=tmp_path, corpus_dir=tmp_path)


# ---------------------------------------------------------------------------
# Expected answers derived from text
# ---------------------------------------------------------------------------


class TestDerivedExpected:
    def test_count_word_is_whole_word_and_case_insensitive(self) -> None:
        assert count_word("Netherfield netherfield Netherfields NETHERFIELD.", "Netherfield") == 3

    def test_derive_expected(self) -> None:
        assert derive_expected("a b a", {"count_word": "a"}) == "2"
        assert derive_expected("x. x. y.", {"count_sentence": "x."}) == "2"
        with pytest.raises(CorpusError):
            derive_expected("t", {"magic": 1})


# ---------------------------------------------------------------------------
# Materialise
# ---------------------------------------------------------------------------


class TestMaterialise:
    def test_content_file_is_relative_to_the_yaml(self, tmp_path: Path) -> None:
        (tmp_path / "doc.md").write_text("# Hello\n\nbody", encoding="utf-8")
        case = BenchmarkCase(id="c", content="", query="q", content_spec={"content_file": "doc.md"})
        assert materialise_case(case, base_dir=tmp_path, corpus_dir=tmp_path) is None
        assert case.content.startswith("# Hello")

    def test_public_text_is_skipped_without_fetch(self, tmp_path: Path) -> None:
        case = BenchmarkCase(
            id="c",
            content="",
            query="q",
            content_spec={"source_url": "https://example.invalid/x.txt", "sha256": "0" * 64},
        )
        reason = materialise_case(case, base_dir=tmp_path, corpus_dir=tmp_path / "corpus")
        assert reason is not None
        assert "--fetch" in reason
        assert case.content == ""

    def test_public_text_digest_is_verified(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "x.txt").write_text("tampered", encoding="utf-8")
        case = BenchmarkCase(
            id="c",
            content="",
            query="q",
            content_spec={"source_url": "https://example.invalid/x.txt", "sha256": "0" * 64},
        )
        with pytest.raises(CorpusError, match="sha256 mismatch"):
            materialise_case(case, base_dir=tmp_path, corpus_dir=corpus)

    def test_public_text_is_read_when_present_and_valid(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        body = "Netherfield Park is let at last."
        (corpus / "x.txt").write_text(body, encoding="utf-8")
        digest = hashlib.sha256(body.encode()).hexdigest()
        case = BenchmarkCase(
            id="c",
            content="",
            query="q",
            content_spec={"source_url": "https://example.invalid/x.txt", "sha256": digest},
            expected_spec={"count_word": "Netherfield"},
        )
        assert materialise_case(case, base_dir=tmp_path, corpus_dir=corpus) is None
        assert case.expected_answer == "1"

    def test_report_lists_materialised_and_skipped(self, tmp_path: Path) -> None:
        ds = load_dataset_from_dict(
            {
                "cases": [
                    {"id": "gen", "query": "q", "generator": "synthetic", "tokens": 1000},
                    {"id": "pub", "query": "q", "source_url": "https://example.invalid/y.txt"},
                ]
            }
        )
        report = materialise(ds, base_dir=tmp_path, corpus_dir=tmp_path / "corpus")
        assert isinstance(report, MaterialiseReport)
        assert report.materialised == ["gen"]
        assert set(report.skipped) == {"pub"}


# ---------------------------------------------------------------------------
# The tracked longdoc-v1 set
# ---------------------------------------------------------------------------


class TestLongdocV1:
    def test_shape(self) -> None:
        ds = load_dataset(str(LONGDOC))
        assert len(ds.cases) >= 12
        assert {c.task_type for c in ds.cases} >= {
            TASK_TYPE_NEEDLE,
            TASK_TYPE_SYNTHESIS,
            TASK_TYPE_AGGREGATION,
            TASK_TYPE_REFUSAL,
        }
        buckets = {c.min_tokens for c in ds.cases}
        assert any(b is not None and b <= 5_000 for b in buckets)
        assert any(b is not None and 40_000 <= b <= 60_000 for b in buckets)
        assert any(b is not None and b >= 150_000 for b in buckets)
        assert ds.sources, "sources block with licences is required"
        ids = [c.id for c in ds.cases]
        assert len(ids) == len(set(ids))
        for case in ds.cases:
            if case.source is not None:
                assert case.source in {s["id"] for s in ds.sources}, case.id

    def test_offline_cases_materialise_and_public_ones_skip_cleanly(self, tmp_path: Path) -> None:
        ds = load_dataset(str(LONGDOC))
        report = materialise(ds, base_dir=LONGDOC.parent, corpus_dir=tmp_path / "corpus")
        # Everything not needing the network is usable from a fresh clone.
        for case in ds.cases:
            spec = case.content_spec or {}
            if "source_url" in spec:
                assert case.id in report.skipped
            else:
                assert case.id in report.materialised, case.id
                assert case.content, case.id
                if case.min_tokens is not None:
                    assert estimate_tokens(case.content) >= case.min_tokens * 0.8, case.id
        # Derived aggregation targets exist for the synthetic aggregation cases.
        synthetic_aggregations = [
            c
            for c in ds.cases
            if c.task_type == TASK_TYPE_AGGREGATION and "generator" in (c.content_spec or {})
        ]
        assert synthetic_aggregations
        for case in synthetic_aggregations:
            assert case.expected_answer is not None, case.id
            assert case.expected_answer.isdigit(), case.id
            assert int(case.expected_answer) == int((case.content_spec or {})["repeat"]["times"])
