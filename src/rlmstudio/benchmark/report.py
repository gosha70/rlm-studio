# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Benchmark report generation with aggregation and export."""

import csv
import io
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rlmstudio.application.services.outcome_classifier import OutcomeCategory

from .matrix_runner import BenchmarkResults, SlotOutcome
from .runner import BenchmarkRun

BENCH_START_MARKER = "<!-- bench:start -->"
BENCH_END_MARKER = "<!-- bench:end -->"
RESULTS_METADATA_NOTE = "note"  # BenchmarkResults.metadata key rendered as "Run notes"


class BenchmarkReport:
    """Generate reports from a benchmark run.

    Aggregates metrics across strategies and cases, and exports
    results to JSON or CSV.

    Usage::

        report = BenchmarkReport(run)
        report.save_json("results.json")
        report.save_csv("results.csv")
        summary = report.summary()
    """

    def __init__(self, run: BenchmarkRun):
        self.run = run

    def summary(self) -> dict[str, Any]:
        """High-level summary of the benchmark run."""
        per_strategy = {
            name: self.run.get_strategy_metrics(name) for name in self.run.strategy_names
        }

        # Find winners
        winners: dict[str, Any] = {}
        successful_strategies = {
            name: m for name, m in per_strategy.items() if m.get("cases", 0) > 0
        }

        if successful_strategies:
            winners["fastest"] = min(
                successful_strategies,
                key=lambda n: successful_strategies[n].get("avg_time", float("inf")),
            )
            winners["cheapest"] = min(
                successful_strategies,
                key=lambda n: successful_strategies[n].get("avg_cost", float("inf")),
            )
            winners["fewest_tokens"] = min(
                successful_strategies,
                key=lambda n: successful_strategies[n].get("avg_tokens", float("inf")),
            )
            winners["most_reliable"] = max(
                successful_strategies,
                key=lambda n: successful_strategies[n].get("success_rate", 0),
            )

        return {
            "dataset": self.run.dataset_name,
            "cases": self.run.case_count,
            "strategies": self.run.strategy_names,
            "total_time": self.run.total_elapsed_time,
            "success_rates": self.run.success_rate,
            "per_strategy": per_strategy,
            "winners": winners,
        }

    def pairwise_comparison(self, strategy_a: str, strategy_b: str) -> dict[str, Any]:
        """Compare two strategies across all cases."""
        a_metrics = self.run.get_strategy_metrics(strategy_a)
        b_metrics = self.run.get_strategy_metrics(strategy_b)

        if a_metrics.get("cases", 0) == 0 or b_metrics.get("cases", 0) == 0:
            return {"error": "One or both strategies have no results"}

        return {
            "strategies": (strategy_a, strategy_b),
            "tokens": {
                strategy_a: a_metrics["avg_tokens"],
                strategy_b: b_metrics["avg_tokens"],
                "delta": a_metrics["avg_tokens"] - b_metrics["avg_tokens"],
                "delta_pct": _pct_delta(a_metrics["avg_tokens"], b_metrics["avg_tokens"]),
            },
            "cost": {
                strategy_a: a_metrics["avg_cost"],
                strategy_b: b_metrics["avg_cost"],
                "delta": a_metrics["avg_cost"] - b_metrics["avg_cost"],
                "delta_pct": _pct_delta(a_metrics["avg_cost"], b_metrics["avg_cost"]),
            },
            "time": {
                strategy_a: a_metrics["avg_time"],
                strategy_b: b_metrics["avg_time"],
                "delta": a_metrics["avg_time"] - b_metrics["avg_time"],
                "delta_pct": _pct_delta(a_metrics["avg_time"], b_metrics["avg_time"]),
            },
            "success_rate": {
                strategy_a: a_metrics["success_rate"],
                strategy_b: b_metrics["success_rate"],
            },
        }

    def per_case_table(self) -> list[dict[str, Any]]:
        """Flat table of per-case, per-strategy metrics for CSV export."""
        rows: list[dict[str, Any]] = []
        for cr in self.run.case_results:
            for name in self.run.strategy_names:
                sr = cr.evaluation.results.get(name)
                if sr is None:
                    continue
                rows.append(
                    {
                        "case_id": cr.case.id,
                        "category": cr.case.category,
                        "difficulty": cr.case.difficulty,
                        "content_length": cr.case.content_length,
                        "strategy": name,
                        "success": sr.success,
                        "answer_length": len(sr.answer),
                        "steps": sr.steps,
                        "tokens_input": sr.tokens.input_tokens,
                        "tokens_output": sr.tokens.output_tokens,
                        "tokens_total": sr.tokens.total_tokens,
                        "cost": sr.cost,
                        "elapsed_time": sr.elapsed_time,
                        "error": sr.error or "",
                    }
                )
        return rows

    def save_json(self, path: str) -> None:
        """Export full benchmark results to JSON."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.summary(),
            "cases": self.per_case_table(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def save_csv(self, path: str) -> None:
        """Export per-case metrics to CSV."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        rows = self.per_case_table()
        if not rows:
            filepath.write_text("")
            return

        fieldnames = list(rows[0].keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def to_csv_string(self) -> str:
        """Return CSV content as a string."""
        rows = self.per_case_table()
        if not rows:
            return ""
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue()


def _pct_delta(a: float, b: float) -> float:
    """Percentage delta of a relative to b: (a - b) / b * 100."""
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0


# ---------------------------------------------------------------------------
# Matrix benchmark report (specs/benchmarks-v1 G4, FR-2)
# ---------------------------------------------------------------------------


@dataclass
class CellSummary:
    """Aggregates for one provider × engine cell across cases and repetitions."""

    provider: str
    model: str
    engine: str
    cases: int
    runs: int
    failed: int
    timed_out: int
    match_cases: int
    match_passed: int
    judge_mean: float | None
    judge_range: float | None  # max − min of per-repetition means (0 for a single rep)
    input_tokens: int
    output_tokens: int
    total_cost: float
    cost_unknown_runs: int
    median_ttft_ms: int | None
    mean_elapsed_seconds: float
    outcomes: dict[str, int] = field(default_factory=dict)

    @property
    def accuracy(self) -> float | None:
        return self.match_passed / self.match_cases if self.match_cases else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "engine": self.engine,
            "cases": self.cases,
            "runs": self.runs,
            "failed": self.failed,
            "timed_out": self.timed_out,
            "match_cases": self.match_cases,
            "match_passed": self.match_passed,
            "accuracy": self.accuracy,
            "judge_mean": self.judge_mean,
            "judge_range": self.judge_range,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_cost": self.total_cost,
            "cost_unknown_runs": self.cost_unknown_runs,
            "median_ttft_ms": self.median_ttft_ms,
            "mean_elapsed_seconds": self.mean_elapsed_seconds,
            "outcomes": self.outcomes,
        }


def _judge_mean_and_range(outcomes: list[SlotOutcome]) -> tuple[float | None, float | None]:
    """Mean of per-repetition mean judge scores, and their max − min."""
    per_rep: dict[int, list[float]] = {}
    for outcome in outcomes:
        if outcome.judge is not None:
            per_rep.setdefault(outcome.rep, []).append(outcome.judge.overall)
    if not per_rep:
        return None, None
    rep_means = [statistics.fmean(v) for v in per_rep.values()]
    return round(statistics.fmean(rep_means), 2), round(max(rep_means) - min(rep_means), 2)


class MatrixBenchmarkReport:
    """Aggregate :class:`BenchmarkResults` into per-cell summaries, Markdown and JSON.

    ``BENCHMARKS.md`` is regenerated between :data:`BENCH_START_MARKER` and
    :data:`BENCH_END_MARKER`, so the page never carries hand-edited numbers.
    """

    def __init__(self, results: BenchmarkResults) -> None:
        self.results = results

    def cells(self) -> list[CellSummary]:
        """One summary per provider/model × engine, in the run's declared order."""
        groups: dict[tuple[str, str, str], list[SlotOutcome]] = {}
        for outcome in self.results.outcomes:
            key = (outcome.provider, outcome.model, outcome.engine)
            groups.setdefault(key, []).append(outcome)

        declared: list[tuple[str, str, str]] = []
        for spec in self.results.providers:
            provider, _, model = spec.partition("/")
            declared.extend((provider, model, engine) for engine in self.results.engines)
        # Declared order first (deduplicated), then anything unexpected.
        order = list(dict.fromkeys(declared + list(groups)))

        summaries: list[CellSummary] = []
        for key in order:
            group = groups.get(key)
            if not group:
                continue
            judge_mean, judge_range = _judge_mean_and_range(group)
            ttfts = [o.median_ttft_ms for o in group if o.median_ttft_ms is not None]
            outcome_counts: dict[str, int] = {}
            for o in group:
                outcome_counts[o.outcome] = outcome_counts.get(o.outcome, 0) + 1
            summaries.append(
                CellSummary(
                    provider=key[0],
                    model=key[1],
                    engine=key[2],
                    cases=len({o.case_id for o in group}),
                    runs=len(group),
                    failed=sum(1 for o in group if o.outcome != OutcomeCategory.SUCCESS.value),
                    timed_out=sum(
                        1
                        for o in group
                        if o.outcome
                        in (OutcomeCategory.TIMEOUT.value, OutcomeCategory.PREFILL_TIMEOUT.value)
                    ),
                    match_cases=sum(1 for o in group if o.match_pass is not None),
                    match_passed=sum(1 for o in group if o.match_pass),
                    judge_mean=judge_mean,
                    judge_range=judge_range,
                    input_tokens=sum(o.input_tokens for o in group),
                    output_tokens=sum(o.output_tokens for o in group),
                    total_cost=round(sum(o.total_cost for o in group), 6),
                    cost_unknown_runs=sum(1 for o in group if not o.cost_known),
                    median_ttft_ms=int(statistics.median(ttfts)) if ttfts else None,
                    mean_elapsed_seconds=round(
                        statistics.fmean(o.elapsed_seconds for o in group), 2
                    ),
                    outcomes=outcome_counts,
                )
            )
        return summaries

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def header_markdown(self) -> str:
        r = self.results
        lines = [
            f"**Dataset:** `{r.dataset_name}` · **cases:** {len(r.case_ids)} · "
            f"**repetitions:** {r.reps} · **temperature:** {r.temperature}",
            f"**Providers:** {', '.join(f'`{p}`' for p in r.providers)} · "
            f"**Engines:** {', '.join(f'`{e}`' for e in r.engines)}",
            (
                f"**Judge:** `{r.judge_model}` (prompt v{r.judge_prompt_version})"
                if r.judge_model
                else "**Judge:** none — accuracy proxy only"
            ),
            f"**Run:** {r.started_at[:19]} → {r.finished_at[:19]} UTC · "
            f"**total cost of this run:** ${r.total_cost:.4f}",
        ]
        if r.skipped:
            skipped = ", ".join(f"`{cid}`" for cid in sorted(r.skipped))
            lines.append(f"**Skipped cases (not fetched):** {skipped}")
        if r.metadata.get(RESULTS_METADATA_NOTE):
            # Hardware / model-version note for local rows (spec AC-2).
            lines.append(f"**Run notes:** {r.metadata[RESULTS_METADATA_NOTE]}")
        return "\n".join(lines)

    def table_markdown(self) -> str:
        rows = [
            "| Provider / model | Engine | Cases | Accuracy | Judge (mean ± range) | "
            "Failed / timed out | Tokens in / out | Cost | TTFT p50 | Wall time (mean) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for c in self.cells():
            accuracy = (
                f"{c.accuracy:.0%} ({c.match_passed}/{c.match_cases})"
                if c.accuracy is not None
                else "—"
            )
            judge = "—"
            if c.judge_mean is not None:
                judge = f"{c.judge_mean:.2f}"
                if c.judge_range:
                    judge += f" ± {c.judge_range:.2f}"
            cost = f"${c.total_cost:.4f}"
            if c.cost_unknown_runs:
                cost += f" (unknown ×{c.cost_unknown_runs})"
            ttft = f"{c.median_ttft_ms} ms" if c.median_ttft_ms is not None else "—"
            rows.append(
                f"| {c.provider} / {c.model} | `{c.engine}` | {c.cases} | {accuracy} | "
                f"{judge} | {c.failed} / {c.timed_out} | {c.input_tokens:,} / {c.output_tokens:,} | "
                f"{cost} | {ttft} | {c.mean_elapsed_seconds:.1f} s |"
            )
        return "\n".join(rows)

    def per_case_markdown(self) -> str:
        """Per case × engine detail (first repetition), for ``results.md``."""
        rows = [
            "| Case | Provider | Engine | Outcome | Match | Judge | Tokens | Cost | Time |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for o in self.results.outcomes:
            if o.rep != 1:
                continue
            match = "—" if o.match_pass is None else ("pass" if o.match_pass else "fail")
            judge = f"{o.judge.overall:.2f}" if o.judge else "—"
            rows.append(
                f"| `{o.case_id}` | {o.provider} | `{o.engine}` | {o.outcome} | {match} | {judge} | "
                f"{o.input_tokens + o.output_tokens:,} | ${o.total_cost:.4f} | {o.elapsed_seconds:.1f} s |"
            )
        return "\n".join(rows)

    def to_markdown(self, *, include_cases: bool = False) -> str:
        parts = [self.header_markdown(), "", self.table_markdown()]
        if include_cases:
            parts += ["", "### Per case", "", self.per_case_markdown()]
        return "\n".join(parts) + "\n"

    def to_dict(self) -> dict[str, Any]:
        return {"cells": [c.to_dict() for c in self.cells()], "results": self.results.to_dict()}

    def save_json(self, path: str | Path) -> None:
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def save_markdown(self, path: str | Path) -> None:
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(self.to_markdown(include_cases=True), encoding="utf-8")


def regenerate_page(page: str | Path, block: str) -> str:
    """Replace the text between the bench markers in *page* with *block*; idempotent.

    Returns the new page text (also written back).  Raises ``ValueError``
    when the markers are missing so a run can never silently skip the page.
    """
    path = Path(page)
    text = path.read_text(encoding="utf-8")
    start = text.find(BENCH_START_MARKER)
    end = text.find(BENCH_END_MARKER)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"{path}: missing {BENCH_START_MARKER} / {BENCH_END_MARKER} markers")
    head = text[: start + len(BENCH_START_MARKER)]
    tail = text[end:]
    updated = f"{head}\n{block.strip()}\n{tail}"
    path.write_text(updated, encoding="utf-8")
    return updated
