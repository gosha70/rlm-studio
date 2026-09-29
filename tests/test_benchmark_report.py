"""Matrix benchmark report: cell aggregates, Markdown, page regeneration (specs/benchmarks-v1 G4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlmstudio.application.sandbox_vars import MODE_DIRECT, MODE_RLM_OFFICIAL
from rlmstudio.application.services.outcome_classifier import OutcomeCategory
from rlmstudio.benchmark.matrix_runner import BenchmarkResults, SlotOutcome
from rlmstudio.benchmark.report import (
    BENCH_END_MARKER,
    BENCH_START_MARKER,
    MatrixBenchmarkReport,
    regenerate_page,
)
from rlmstudio.benchmark.scoring import JudgeVerdict

REPO_ROOT = Path(__file__).resolve().parents[1]


def _verdict(overall: float) -> JudgeVerdict:
    return JudgeVerdict(
        overall=overall, dimensions={"x": overall}, reasoning="", model="j", prompt_version="2.0"
    )


def _outcome(
    case_id: str,
    rep: int,
    provider: str,
    engine: str,
    *,
    outcome: str = OutcomeCategory.SUCCESS.value,
    match_pass: bool | None = True,
    judge: float | None = 4.0,
    cost: float = 0.01,
    cost_known: bool = True,
    ttft: int | None = 300,
    elapsed: float = 2.0,
) -> SlotOutcome:
    return SlotOutcome(
        case_id=case_id,
        rep=rep,
        provider=provider,
        model="m",
        engine=engine,
        success=outcome == OutcomeCategory.SUCCESS.value,
        outcome=outcome,
        answer="a",
        error=None if outcome == OutcomeCategory.SUCCESS.value else "boom",
        input_tokens=100,
        output_tokens=10,
        total_cost=cost,
        cost_known=cost_known,
        elapsed_seconds=elapsed,
        steps=1,
        median_ttft_ms=ttft,
        match_pass=match_pass,
        judge=_verdict(judge) if judge is not None else None,
    )


def _results(
    outcomes: list[SlotOutcome], *, reps: int = 1, skipped: dict[str, str] | None = None
) -> BenchmarkResults:
    return BenchmarkResults(
        dataset_name="longdoc-v1",
        providers=["openai/m", "ollama/q"],
        engines=[MODE_DIRECT, MODE_RLM_OFFICIAL],
        reps=reps,
        temperature=0.0,
        started_at="2026-09-29T10:00:00+00:00",
        finished_at="2026-09-29T10:30:00+00:00",
        judge_model="openai/gpt-4o-mini",
        judge_prompt_version="2.0",
        case_ids=["c1", "c2"],
        skipped=skipped or {},
        outcomes=outcomes,
    )


class TestCells:
    def test_aggregates_per_provider_engine_in_declared_order(self) -> None:
        outcomes = [
            _outcome("c1", 1, "openai", MODE_DIRECT, judge=4.0),
            _outcome("c2", 1, "openai", MODE_DIRECT, match_pass=False, judge=2.0),
            _outcome("c1", 1, "openai", MODE_RLM_OFFICIAL, ttft=None, cost=0.0, cost_known=False),
            _outcome(
                "c2",
                1,
                "openai",
                MODE_RLM_OFFICIAL,
                outcome=OutcomeCategory.TIMEOUT.value,
                match_pass=False,
                judge=None,
                ttft=None,
            ),
            _outcome("c1", 1, "ollama", MODE_DIRECT, match_pass=None, judge=5.0, cost=0.0),
        ]
        cells = MatrixBenchmarkReport(_results(outcomes)).cells()

        assert [(c.provider, c.engine) for c in cells] == [
            ("openai", MODE_DIRECT),
            ("openai", MODE_RLM_OFFICIAL),
            ("ollama", MODE_DIRECT),
        ]
        openai_direct, openai_official, ollama_direct = cells
        assert (openai_direct.cases, openai_direct.runs) == (2, 2)
        assert openai_direct.accuracy == 0.5
        assert openai_direct.judge_mean == 3.0
        assert openai_direct.judge_range == 0.0
        assert openai_direct.median_ttft_ms == 300
        assert openai_direct.total_cost == 0.02

        assert openai_official.failed == 1
        assert openai_official.timed_out == 1
        assert openai_official.cost_unknown_runs == 1
        assert openai_official.median_ttft_ms is None
        assert openai_official.outcomes == {"success": 1, "timeout": 1}

        assert ollama_direct.accuracy is None  # no expected answer
        assert ollama_direct.judge_mean == 5.0

    def test_judge_range_spans_repetitions(self) -> None:
        outcomes = [
            _outcome("c1", 1, "openai", MODE_DIRECT, judge=4.0),
            _outcome("c1", 2, "openai", MODE_DIRECT, judge=3.0),
            _outcome("c1", 3, "openai", MODE_DIRECT, judge=5.0),
        ]
        cell = MatrixBenchmarkReport(_results(outcomes, reps=3)).cells()[0]

        assert cell.runs == 3
        assert cell.cases == 1
        assert cell.judge_mean == 4.0
        assert cell.judge_range == 2.0


class TestMarkdown:
    def test_header_and_table(self) -> None:
        outcomes = [
            _outcome("c1", 1, "openai", MODE_DIRECT),
            _outcome("c1", 1, "openai", MODE_RLM_OFFICIAL, ttft=None, cost=0.0, cost_known=False),
        ]
        md = MatrixBenchmarkReport(_results(outcomes, skipped={"pub": "not fetched"})).to_markdown()

        assert "**Dataset:** `longdoc-v1`" in md
        assert "**Judge:** `openai/gpt-4o-mini` (prompt v2.0)" in md
        assert "**Skipped cases (not fetched):** `pub`" in md
        assert (
            "| openai / m | `direct` | 1 | 100% (1/1) | 4.00 | 0 / 0 | 100 / 10 | $0.0100 | 300 ms | 2.0 s |"
            in md
        )
        assert (
            "| openai / m | `rlm_official` | 1 | 100% (1/1) | 4.00 | 0 / 0 | 100 / 10 | $0.0000 (unknown ×1) | — | 2.0 s |"
            in md
        )

    def test_per_case_section_only_when_requested(self) -> None:
        report = MatrixBenchmarkReport(_results([_outcome("c1", 1, "openai", MODE_DIRECT)]))

        assert "### Per case" not in report.to_markdown()
        detailed = report.to_markdown(include_cases=True)
        assert "### Per case" in detailed
        assert (
            "| `c1` | openai | `direct` | success | pass | 4.00 | 110 | $0.0100 | 2.0 s |"
            in detailed
        )

    def test_save_json_and_markdown(self, tmp_path: Path) -> None:
        report = MatrixBenchmarkReport(_results([_outcome("c1", 1, "openai", MODE_DIRECT)]))
        report.save_json(tmp_path / "out" / "results.json")
        report.save_markdown(tmp_path / "out" / "results.md")

        data = json.loads((tmp_path / "out" / "results.json").read_text())
        assert data["cells"][0]["accuracy"] == 1.0
        assert data["results"]["dataset_name"] == "longdoc-v1"
        assert "### Per case" in (tmp_path / "out" / "results.md").read_text()


class TestRegeneratePage:
    def test_replaces_between_markers_idempotently(self, tmp_path: Path) -> None:
        page = tmp_path / "BENCHMARKS.md"
        page.write_text(
            f"# Title\n\nintro\n\n{BENCH_START_MARKER}\nold\n{BENCH_END_MARKER}\n\nfooter\n",
            encoding="utf-8",
        )

        first = regenerate_page(page, "| new |\n")
        second = regenerate_page(page, "| new |\n")

        assert first == second
        assert first.startswith("# Title\n\nintro\n\n")
        assert first.endswith("\n\nfooter\n")
        assert "old" not in first
        assert f"{BENCH_START_MARKER}\n| new |\n{BENCH_END_MARKER}" in first

    def test_missing_markers_raise(self, tmp_path: Path) -> None:
        page = tmp_path / "BENCHMARKS.md"
        page.write_text("no markers here", encoding="utf-8")

        with pytest.raises(ValueError, match="missing"):
            regenerate_page(page, "x")

    def test_tracked_page_has_markers(self) -> None:
        text = (REPO_ROOT / "BENCHMARKS.md").read_text(encoding="utf-8")
        assert text.index(BENCH_START_MARKER) < text.index(BENCH_END_MARKER)
        assert "not the paper's benchmark" in text
        assert "alexzhang13/rlm" in text
