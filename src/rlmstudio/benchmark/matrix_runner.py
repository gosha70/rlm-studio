# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Run a benchmark dataset through the Compare matrix: providers × engines per case.

Uses exactly the code path the Compare page uses —
:class:`RunMatrixComparisonUseCase` over slots built by
:func:`rlmstudio.api.build_matrix_slots` — so BENCHMARKS.md numbers are the
numbers a user would see in the UI. Per-case budgets from the dataset seed
each slot's :class:`RunConfigDTO`; outcomes are classified with the same
:func:`classify_execution_outcome` the Dashboard uses, so failures and
timeouts are counted, never dropped (specs/benchmarks-v1 FR-1, FR-4, FR-5).
"""

from __future__ import annotations

import logging
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from rlmstudio.application.dto import RunConfigDTO
from rlmstudio.application.sandbox_vars import (
    MODE_RLM_OFFICIAL,
    RESULT_KEY_COST_KNOWN,
    RLM_DEFAULT_MAX_STEPS,
)
from rlmstudio.application.services.outcome_classifier import classify_execution_outcome
from rlmstudio.application.use_cases.run_matrix_comparison import (
    MatrixSlotDTO,
    MatrixSlotResultDTO,
    RunMatrixComparisonUseCase,
)
from rlmstudio.infrastructure.engines.rlms_adapter import SANDBOX_TYPE_DOCKER

from .dataset import BenchmarkCase, BenchmarkDataset
from .scoring import JudgeScorer, JudgeVerdict, matches

logger = logging.getLogger(__name__)

# ``budget`` keys a case may set (RunConfigDTO field names).
BUDGET_KEY_MAX_STEPS = "max_steps"
BUDGET_KEY_MAX_TIME_SECONDS = "max_time_seconds"
BUDGET_KEY_MAX_COST = "max_cost"
BUDGET_KEY_MAX_RECURSION_DEPTH = "max_recursion_depth"

SlotBuilder = Callable[[list[str], list[str], RunConfigDTO, BenchmarkCase], list[MatrixSlotDTO]]
"""``(providers, engines, base_config, case) -> slots``; swapped for fakes by ``--dry-run``."""


@dataclass
class SlotOutcome:
    """One (case, repetition, provider × engine) cell of the benchmark."""

    case_id: str
    rep: int
    provider: str
    model: str
    engine: str
    success: bool
    outcome: str  # OutcomeCategory value
    answer: str
    error: str | None
    input_tokens: int
    output_tokens: int
    total_cost: float
    cost_known: bool
    elapsed_seconds: float
    steps: int
    median_ttft_ms: int | None
    match_pass: bool | None = None  # None when the case has no expected answer
    judge: JudgeVerdict | None = None
    judge_error: str | None = None  # why this cell has no verdict, when it should have

    def to_dict(self) -> dict[str, Any]:
        data = {
            "case_id": self.case_id,
            "rep": self.rep,
            "provider": self.provider,
            "model": self.model,
            "engine": self.engine,
            "success": self.success,
            "outcome": self.outcome,
            "answer": self.answer,
            "error": self.error,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_cost": self.total_cost,
            "cost_known": self.cost_known,
            "elapsed_seconds": self.elapsed_seconds,
            "steps": self.steps,
            "median_ttft_ms": self.median_ttft_ms,
            "match_pass": self.match_pass,
            "judge": self.judge.to_dict() if self.judge else None,
            "judge_error": self.judge_error,
        }
        return data


@dataclass
class BenchmarkResults:
    """Everything a report needs: configuration, provenance, and every cell."""

    dataset_name: str
    providers: list[str]
    engines: list[str]
    reps: int
    temperature: float
    started_at: str
    finished_at: str = ""
    judge_model: str | None = None
    judge_prompt_version: str | None = None
    case_ids: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    outcomes: list[SlotOutcome] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        return sum(o.total_cost for o in self.outcomes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "providers": self.providers,
            "engines": self.engines,
            "reps": self.reps,
            "temperature": self.temperature,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "judge_model": self.judge_model,
            "judge_prompt_version": self.judge_prompt_version,
            "case_ids": self.case_ids,
            "skipped": self.skipped,
            "total_cost": self.total_cost,
            "metadata": self.metadata,
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


def budget_to_config(budget: dict[str, Any]) -> RunConfigDTO:
    """Translate a case's ``budget`` mapping into the base :class:`RunConfigDTO`."""
    return RunConfigDTO(
        max_steps=int(budget.get(BUDGET_KEY_MAX_STEPS, RLM_DEFAULT_MAX_STEPS)),
        max_time_seconds=(
            float(budget[BUDGET_KEY_MAX_TIME_SECONDS])
            if budget.get(BUDGET_KEY_MAX_TIME_SECONDS) is not None
            else None
        ),
        max_cost=(
            float(budget[BUDGET_KEY_MAX_COST])
            if budget.get(BUDGET_KEY_MAX_COST) is not None
            else None
        ),
        max_recursion_depth=int(budget.get(BUDGET_KEY_MAX_RECURSION_DEPTH, 1)),
    )


def median_ttft_ms(trace: list[dict[str, Any]] | None) -> int | None:
    """Median of the per-step ``ttft_ms`` values in a raw trace, or None if none."""
    values = [
        int(e["ttft_ms"])
        for e in trace or []
        if isinstance(e, dict) and e.get("ttft_ms") is not None
    ]
    return int(statistics.median(values)) if values else None


class MatrixBenchmarkRunner:
    """Drive a dataset through providers × engines, scoring every cell.

    Args:
        providers: ``"backend/model"`` specs, as for :func:`rlmstudio.api.compare_matrix`.
        engines: Slot modes to run (``direct`` / ``rag`` / ``rlm`` / ``rlm_official``).
        slot_builder: Builds the slots for one case; the default wraps
            :func:`rlmstudio.api.build_matrix_slots` with ``temperature`` and
            the connection settings below.
        judge: Optional pointwise judge; skipped for non-usable outcomes.
        reps: Repetitions per case (cloud judge scores are reported as mean ± range).
        temperature: Sampling temperature for every slot (0 for determinism, FR-5).
        api_key / api_base / timeout: Applied to every slot's adapter.
        on_slot_complete: Progress callback.
    """

    def __init__(
        self,
        *,
        providers: list[str],
        engines: list[str],
        slot_builder: SlotBuilder | None = None,
        judge: JudgeScorer | None = None,
        reps: int = 1,
        temperature: float = 0.0,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float | None = None,
        sandbox_type: str | None = None,
        on_slot_complete: Callable[[SlotOutcome], None] | None = None,
    ) -> None:
        if reps < 1:
            raise ValueError("reps must be >= 1")
        if not providers or not engines:
            raise ValueError("providers and engines must be non-empty")
        self._providers = providers
        self._engines = engines
        self._judge = judge
        self._reps = reps
        self._temperature = temperature
        self._api_key = api_key
        self._api_base = api_base
        self._timeout = timeout
        self._sandbox_type = sandbox_type
        self._on_slot_complete = on_slot_complete
        self._slot_builder: SlotBuilder = slot_builder or self._default_slot_builder

    def _default_slot_builder(
        self,
        providers: list[str],
        engines: list[str],
        base_config: RunConfigDTO,
        case: BenchmarkCase,
    ) -> list[MatrixSlotDTO]:
        # Imported here: the api facade imports every adapter, and the runner
        # must stay importable in a dry run without provider credentials.
        from rlmstudio.api import build_matrix_slots

        slots: list[MatrixSlotDTO] = build_matrix_slots(
            providers,
            engines,
            api_key=self._api_key,
            api_base=self._api_base,
            temperature=self._temperature,
            max_tokens=None,
            timeout=self._timeout,
            max_steps=base_config.max_steps,
            num_retries=None,
            embedding_api_key=None,
            base_config=base_config,
            sandbox_type=self._sandbox_type,
        )
        return slots

    # ------------------------------------------------------------------

    def run(
        self,
        dataset: BenchmarkDataset,
        *,
        case_ids: list[str] | None = None,
        limit: int | None = None,
        skipped: dict[str, str] | None = None,
    ) -> BenchmarkResults:
        """Run every materialised case; ``skipped`` is carried into the results verbatim."""
        cases = [c for c in dataset if c.content]
        if case_ids is not None:
            wanted = set(case_ids)
            cases = [c for c in cases if c.id in wanted]
        if limit is not None:
            cases = cases[:limit]

        results = self._empty_results(dataset, [c.id for c in cases], skipped)
        for case in cases:
            for rep in range(1, self._reps + 1):
                results.outcomes.extend(self._run_case(case, rep))
        results.finished_at = datetime.now(timezone.utc).isoformat()
        return results

    def partial_results(
        self,
        dataset: BenchmarkDataset,
        outcomes: list[SlotOutcome],
        *,
        skipped: dict[str, str] | None = None,
    ) -> BenchmarkResults:
        """Package the cells a stopped run had already produced.

        A real run is hours of paid provider calls, so whatever completed before
        an interruption is worth keeping even though the grid is incomplete.
        """
        results = self._empty_results(dataset, sorted({o.case_id for o in outcomes}), skipped)
        results.outcomes.extend(outcomes)
        results.finished_at = datetime.now(timezone.utc).isoformat()
        return results

    def _empty_results(
        self,
        dataset: BenchmarkDataset,
        case_ids: list[str],
        skipped: dict[str, str] | None,
    ) -> BenchmarkResults:
        return BenchmarkResults(
            dataset_name=dataset.name,
            providers=list(self._providers),
            engines=list(self._engines),
            reps=self._reps,
            temperature=self._temperature,
            started_at=datetime.now(timezone.utc).isoformat(),
            judge_model=self._judge.model if self._judge else None,
            judge_prompt_version=self._judge.prompt_version if self._judge else None,
            case_ids=case_ids,
            skipped=dict(skipped or {}),
        )

    def _run_case(self, case: BenchmarkCase, rep: int) -> list[SlotOutcome]:
        base_config = budget_to_config(case.budget)
        slots = self._slot_builder(self._providers, self._engines, base_config, case)
        started = time.time()
        slot_results = self._execute_slots(case, slots)
        logger.info(
            "bench case=%s rep=%d slots=%d elapsed=%.1fs",
            case.id,
            rep,
            len(slot_results),
            time.time() - started,
        )
        outcomes = [self._score(case, rep, slot) for slot in slot_results]
        if self._on_slot_complete is not None:
            for outcome in outcomes:
                self._on_slot_complete(outcome)
        return outcomes

    def _execute_slots(
        self, case: BenchmarkCase, slots: list[MatrixSlotDTO]
    ) -> list[MatrixSlotResultDTO]:
        """Run a case's slots, keeping in-process official slots out of each other's way.

        The official engine's in-process REPL admits one run at a time process
        wide, and a queued run gives up rather than waiting out its whole
        budget.  Left to the matrix's own thread pool, a grid with two providers
        would therefore lose every official cell but one to "engine busy" — a
        benchmark reporting failures it created itself.  So official slots run
        one after another while the rest still run in parallel.  With the Docker
        sandbox there is no shared REPL and no reason to hold them back.
        """
        official = [s for s in slots if s.mode == MODE_RLM_OFFICIAL]
        serialise = len(official) > 1 and self._sandbox_type != SANDBOX_TYPE_DOCKER
        if not serialise:
            return list(RunMatrixComparisonUseCase().execute(case.content, case.query, slots).slots)

        others = [s for s in slots if s.mode != MODE_RLM_OFFICIAL]
        results: list[MatrixSlotResultDTO] = []
        if others:
            results.extend(
                RunMatrixComparisonUseCase().execute(case.content, case.query, others).slots
            )
        logger.info("bench serialising %d official slots (in-process REPL)", len(official))
        results.extend(
            RunMatrixComparisonUseCase(max_workers=1)
            .execute(case.content, case.query, official)
            .slots
        )
        return results

    def _score(self, case: BenchmarkCase, rep: int, slot: MatrixSlotResultDTO) -> SlotOutcome:
        result = slot.result
        outcome = classify_execution_outcome(result.success, result.error, result.answer)
        usable = outcome.is_usable

        match_pass: bool | None = None
        if case.expected_answer is not None:
            match_pass = usable and matches(result.answer, case.expected_answer, case.match)

        verdict: JudgeVerdict | None = None
        judge_error: str | None = None
        if self._judge is not None and usable:
            try:
                verdict = self._judge.score(
                    query=case.query,
                    response=result.answer,
                    source=case.content,
                    rubric_hint=case.rubric_hint,
                )
            except Exception as exc:  # one judge call must not end a paid run
                # A rate limit or a dropped connection costs this cell its
                # verdict, which the report already renders as missing; it does
                # not cost the hours of provider calls already made.
                judge_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "bench judge failed case=%s rep=%d provider=%s engine=%s: %s",
                    case.id,
                    rep,
                    slot.provider,
                    slot.mode,
                    judge_error,
                )

        return SlotOutcome(
            case_id=case.id,
            rep=rep,
            provider=slot.provider,
            model=slot.model,
            engine=slot.mode,
            success=result.success,
            outcome=outcome.category.value,
            answer=result.answer,
            error=result.error,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_cost=result.total_cost,
            cost_known=bool(result.metadata.get(RESULT_KEY_COST_KNOWN, True)),
            elapsed_seconds=result.elapsed_time,
            steps=result.steps,
            median_ttft_ms=median_ttft_ms(result.trace),
            match_pass=match_pass,
            judge=verdict,
            judge_error=judge_error,
        )
