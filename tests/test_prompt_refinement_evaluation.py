"""Evaluation summary for prompt refinement (#78, spec FR-7, FR-8, AC-6)."""

from __future__ import annotations

import pytest

from rlmstudio.application.services.refinement_evaluation import (
    ArmResult,
    CaseRuns,
    EvaluationRecord,
    EvaluationSetup,
    JudgeInfo,
    RepResult,
    ScoreRange,
    summarise_evaluation,
)


def _setup(reps: int = 2, **overrides: object) -> EvaluationSetup:
    values: dict[str, object] = {
        "chat_provider_id": "cp-1",
        "provider": "openai",
        "model": "gpt-4o",
        "temperature": 0.0,
        "reps": reps,
        "dataset_name": "longdoc-v1",
        "dataset_version": "1.0",
        "case_ids": ["a"],
        "excluded_cases": {},
        "skipped_cases": {},
        "judge": None,
        "started_at": 1.0,
    }
    values.update(overrides)
    return EvaluationSetup(**values)  # type: ignore[arg-type]


def _arm(
    passed: bool | None,
    *,
    judge: float | None = None,
    cost: float = 0.01,
    steps: int = 3,
    outcome: str = "success",
) -> ArmResult:
    return ArmResult(
        run_id="run",
        success=True,
        usable=True,
        outcome_category=outcome,
        passed=passed,
        judge_overall=judge,
        cost=cost,
        steps=steps,
    )


def _case(
    case_id: str, pairs: list[tuple[ArmResult | None, ArmResult | None]], *, det: bool = True
) -> CaseRuns:
    return CaseRuns(
        case_id=case_id,
        deterministic=det,
        content_sha256="h",
        reps=[RepResult(rep=i + 1, baseline=b, candidate=c) for i, (b, c) in enumerate(pairs)],
    )


def test_candidate_dropping_a_pass_rate_is_a_regression() -> None:
    cases = [
        _case("a", [(_arm(True), _arm(True)), (_arm(True), _arm(False))]),
        _case("b", [(_arm(False), _arm(True)), (_arm(False), _arm(True))]),
    ]
    record = summarise_evaluation(_setup(), cases, finished_at=2.0)
    assert record.complete
    assert record.incomplete_reasons == []
    assert record.deterministic_case_count == 2
    assert record.regression is True
    assert record.regression_case_ids == ["a"]
    a, b = record.cases
    assert (a.baseline_pass_rate, a.candidate_pass_rate) == (1.0, 0.5)
    assert (b.baseline_pass_rate, b.candidate_pass_rate) == (0.0, 1.0)
    assert record.deltas is not None
    assert record.deltas.accuracy_baseline == 0.5
    assert record.deltas.accuracy_candidate == 0.75


def test_equal_pass_rates_do_not_regress() -> None:
    cases = [_case("a", [(_arm(True), _arm(True)), (_arm(False), _arm(False))])]
    record = summarise_evaluation(_setup(), cases, finished_at=2.0)
    assert record.complete
    assert record.regression is False


def test_judge_only_case_never_regresses_but_reports_scores() -> None:
    cases = [
        _case(
            "j",
            [
                (_arm(None, judge=5.0), _arm(None, judge=1.0)),
                (_arm(None, judge=4.0), _arm(None, judge=2.0)),
            ],
            det=False,
        )
    ]
    record = summarise_evaluation(_setup(), cases, finished_at=2.0)
    assert record.complete
    assert record.regression is False
    assert record.deterministic_case_count == 0
    assert record.deltas is not None
    assert record.deltas.accuracy_baseline is None
    assert record.deltas.judge_baseline == ScoreRange(mean=4.5, min=4.0, max=5.0)
    assert record.deltas.judge_candidate == ScoreRange(mean=1.5, min=1.0, max=2.0)


def test_deltas_cover_cost_steps_and_outcomes() -> None:
    cases = [
        _case(
            "a",
            [
                (_arm(True, cost=0.1, steps=2), _arm(True, cost=0.3, steps=6, outcome="timeout")),
                (_arm(True, cost=0.1, steps=4), _arm(True, cost=0.1, steps=2)),
            ],
        )
    ]
    deltas = summarise_evaluation(_setup(), cases, finished_at=2.0).deltas
    assert deltas is not None
    assert deltas.cost_baseline == pytest.approx(0.2)
    assert deltas.cost_candidate == pytest.approx(0.4)
    assert (deltas.steps_mean_baseline, deltas.steps_mean_candidate) == (3.0, 4.0)
    assert deltas.outcomes_baseline == {"success": 2}
    assert deltas.outcomes_candidate == {"timeout": 1, "success": 1}


@pytest.mark.parametrize(
    ("setup", "cases", "reason"),
    [
        (_setup(), [], "no case was evaluated"),
        (
            _setup(skipped_cases={"pub": "public text not fetched"}),
            [_case("a", [(_arm(True), _arm(True)), (_arm(True), _arm(True))])],
            "could not be materialised",
        ),
        (
            _setup(),
            [_case("a", [(_arm(True), None), (_arm(True), _arm(True))])],
            "missing an arm",
        ),
        (
            _setup(),
            [_case("a", [(_arm(True), _arm(True))])],
            "1 of 2 repetitions ran",
        ),
    ],
)
def test_incomplete_evaluations_have_no_flag_or_deltas(
    setup: EvaluationSetup, cases: list[CaseRuns], reason: str
) -> None:
    record = summarise_evaluation(setup, cases, finished_at=2.0)
    assert record.complete is False
    assert any(reason in r for r in record.incomplete_reasons)
    assert record.deltas is None
    assert record.regression is False
    assert record.regression_case_ids == []


def test_extra_incomplete_reasons_are_kept() -> None:
    cases = [_case("a", [(_arm(True), _arm(True)), (_arm(True), _arm(True))])]
    record = summarise_evaluation(
        _setup(), cases, finished_at=2.0, extra_incomplete_reasons=["worker stopped"]
    )
    assert record.complete is False
    assert record.incomplete_reasons == ["worker stopped"]


def test_record_round_trips() -> None:
    cases = [
        _case("a", [(_arm(True, judge=4.0), _arm(False, judge=3.0)), (_arm(True), None)]),
        _case("b", [(_arm(True), _arm(True)), (_arm(True), _arm(True))]),
    ]
    setup = _setup(judge=JudgeInfo(chat_provider_id="judge", model="m", prompt_version="2.0"))
    for record in (
        summarise_evaluation(setup, cases, finished_at=2.0),
        summarise_evaluation(setup, cases[1:], finished_at=2.0),
    ):
        assert EvaluationRecord.from_dict(record.to_dict()) == record
