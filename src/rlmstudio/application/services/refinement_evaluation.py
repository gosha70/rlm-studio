"""Evaluation summary for evidence-gated prompt refinement (#78, spec FR-7/FR-8).

Pure computation over per-arm results; the server orchestrates the runs.
An evaluation is *complete* only when every evaluated case × repetition has
both arms, and deltas and the regression flag exist only for a complete
evaluation.  The flag is advisory: it never blocks an accept.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ArmResult:
    """One arm (baseline or candidate) of one case × repetition."""

    run_id: str
    success: bool
    usable: bool
    outcome_category: str
    passed: bool | None
    """Deterministic match result; ``None`` for a case without an expected answer."""
    judge_overall: float | None
    cost: float
    steps: int
    error: str | None = None


@dataclass(frozen=True)
class RepResult:
    """Both arms of one repetition; a missing arm is ``None``."""

    rep: int
    baseline: ArmResult | None
    candidate: ArmResult | None


@dataclass(frozen=True)
class CaseRuns:
    """Everything one evaluated case produced."""

    case_id: str
    deterministic: bool
    content_sha256: str
    reps: list[RepResult]


@dataclass(frozen=True)
class CaseSummary:
    """A case's runs plus its per-arm pass rates and regression verdict."""

    case_id: str
    deterministic: bool
    content_sha256: str
    reps: list[RepResult]
    baseline_pass_rate: float | None
    candidate_pass_rate: float | None
    regressed: bool


@dataclass(frozen=True)
class ScoreRange:
    """Mean, min and max of a set of judge scores."""

    mean: float
    min: float
    max: float


@dataclass(frozen=True)
class EvaluationDeltas:
    """Baseline-vs-candidate figures reported for a complete evaluation (scenario 2)."""

    accuracy_baseline: float | None
    accuracy_candidate: float | None
    judge_baseline: ScoreRange | None
    judge_candidate: ScoreRange | None
    cost_baseline: float
    cost_candidate: float
    steps_mean_baseline: float
    steps_mean_candidate: float
    outcomes_baseline: dict[str, int]
    outcomes_candidate: dict[str, int]


@dataclass(frozen=True)
class JudgeInfo:
    """Which judge scored the arms (AC-7)."""

    chat_provider_id: str
    model: str
    prompt_version: str


@dataclass(frozen=True)
class EvaluationSetup:
    """What was run: provider, dataset, selection and exclusions."""

    chat_provider_id: str
    provider: str
    model: str
    temperature: float
    reps: int
    dataset_name: str
    dataset_version: str
    case_ids: list[str]
    excluded_cases: dict[str, str]
    skipped_cases: dict[str, str]
    judge: JudgeInfo | None
    started_at: float


@dataclass(frozen=True)
class EvaluationRecord:
    """The stored evaluation (``refinements.evaluation_json``); phase D reads it back."""

    setup: EvaluationSetup
    cases: list[CaseSummary]
    complete: bool
    incomplete_reasons: list[str]
    deterministic_case_count: int
    regression: bool
    regression_case_ids: list[str]
    deltas: EvaluationDeltas | None
    finished_at: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationRecord:
        """Rebuild a record written by :meth:`to_dict`."""
        setup = dict(data["setup"])
        setup["judge"] = JudgeInfo(**setup["judge"]) if setup["judge"] else None
        cases = [
            CaseSummary(
                **{
                    **case,
                    "reps": [
                        RepResult(
                            rep=rep["rep"],
                            baseline=ArmResult(**rep["baseline"]) if rep["baseline"] else None,
                            candidate=ArmResult(**rep["candidate"]) if rep["candidate"] else None,
                        )
                        for rep in case["reps"]
                    ],
                }
            )
            for case in data["cases"]
        ]
        deltas = None
        if data["deltas"]:
            raw = dict(data["deltas"])
            for key in ("judge_baseline", "judge_candidate"):
                raw[key] = ScoreRange(**raw[key]) if raw[key] else None
            deltas = EvaluationDeltas(**raw)
        return cls(
            setup=EvaluationSetup(**setup),
            cases=cases,
            complete=data["complete"],
            incomplete_reasons=list(data["incomplete_reasons"]),
            deterministic_case_count=data["deterministic_case_count"],
            regression=data["regression"],
            regression_case_ids=list(data["regression_case_ids"]),
            deltas=deltas,
            finished_at=data["finished_at"],
        )


def _pass_rate(arms: Sequence[ArmResult]) -> float | None:
    scored = [a.passed for a in arms if a.passed is not None]
    if not scored:
        return None
    return sum(1 for p in scored if p) / len(scored)


def _score_range(values: Sequence[float]) -> ScoreRange | None:
    if not values:
        return None
    return ScoreRange(mean=sum(values) / len(values), min=min(values), max=max(values))


def _deltas(baseline: Sequence[ArmResult], candidate: Sequence[ArmResult]) -> EvaluationDeltas:
    def side(arms: Sequence[ArmResult]) -> tuple[float | None, ScoreRange | None, float, float]:
        scores = [a.judge_overall for a in arms if a.judge_overall is not None]
        steps = sum(a.steps for a in arms) / len(arms) if arms else 0.0
        return _pass_rate(arms), _score_range(scores), sum(a.cost for a in arms), steps

    b_acc, b_judge, b_cost, b_steps = side(baseline)
    c_acc, c_judge, c_cost, c_steps = side(candidate)
    return EvaluationDeltas(
        accuracy_baseline=b_acc,
        accuracy_candidate=c_acc,
        judge_baseline=b_judge,
        judge_candidate=c_judge,
        cost_baseline=b_cost,
        cost_candidate=c_cost,
        steps_mean_baseline=b_steps,
        steps_mean_candidate=c_steps,
        outcomes_baseline=dict(Counter(a.outcome_category for a in baseline)),
        outcomes_candidate=dict(Counter(a.outcome_category for a in candidate)),
    )


def summarise_evaluation(
    setup: EvaluationSetup,
    cases: Sequence[CaseRuns],
    *,
    finished_at: float,
    extra_incomplete_reasons: Sequence[str] = (),
) -> EvaluationRecord:
    """Build the stored record from the evaluated cases (FR-7, FR-8).

    A case regresses when it is deterministic and the candidate's pass rate
    across repetitions is lower than the baseline's.  Skipped cases, a case
    × repetition missing an arm, no evaluated case at all, or any
    ``extra_incomplete_reasons`` make the evaluation incomplete; an
    incomplete evaluation has no deltas and no regression flag.
    """
    reasons = list(extra_incomplete_reasons)
    if setup.skipped_cases:
        reasons.append(f"{len(setup.skipped_cases)} selected case(s) could not be materialised")
    if not cases:
        reasons.append("no case was evaluated")

    summaries: list[CaseSummary] = []
    all_baseline: list[ArmResult] = []
    all_candidate: list[ArmResult] = []
    for case in cases:
        if len(case.reps) != setup.reps:
            reasons.append(f"{case.case_id}: {len(case.reps)} of {setup.reps} repetitions ran")
        baseline = [r.baseline for r in case.reps if r.baseline is not None]
        candidate = [r.candidate for r in case.reps if r.candidate is not None]
        missing = sum(1 for r in case.reps if r.baseline is None or r.candidate is None)
        if missing:
            reasons.append(f"{case.case_id}: {missing} repetition(s) missing an arm")
        all_baseline += baseline
        all_candidate += candidate
        b_rate, c_rate = _pass_rate(baseline), _pass_rate(candidate)
        regressed = (
            case.deterministic and b_rate is not None and c_rate is not None and c_rate < b_rate
        )
        summaries.append(
            CaseSummary(
                case_id=case.case_id,
                deterministic=case.deterministic,
                content_sha256=case.content_sha256,
                reps=list(case.reps),
                baseline_pass_rate=b_rate,
                candidate_pass_rate=c_rate,
                regressed=regressed,
            )
        )

    complete = not reasons
    regression_ids = [s.case_id for s in summaries if s.regressed] if complete else []
    return EvaluationRecord(
        setup=setup,
        cases=summaries,
        complete=complete,
        incomplete_reasons=reasons,
        deterministic_case_count=sum(1 for s in summaries if s.deterministic),
        regression=bool(regression_ids),
        regression_case_ids=regression_ids,
        deltas=_deltas(all_baseline, all_candidate) if complete else None,
        finished_at=finished_at,
    )
