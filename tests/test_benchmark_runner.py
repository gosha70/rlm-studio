"""Matrix benchmark runner over fake slots (specs/benchmarks-v1 FR-1, FR-4, FR-5)."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from rlmstudio.application.dto import LLMResponseDTO, RunConfigDTO, RunResultDTO
from rlmstudio.application.sandbox_vars import (
    MODE_DIRECT,
    MODE_RLM_OFFICIAL,
    RLM_DEFAULT_MAX_STEPS,
    TRACE_KEY_ROLE,
)
from rlmstudio.application.services.outcome_classifier import OutcomeCategory
from rlmstudio.application.use_cases.run_matrix_comparison import (
    MatrixSlotDTO,
    MatrixSlotResultDTO,
)
from rlmstudio.benchmark import matrix_runner
from rlmstudio.benchmark.dataset import MATCH_CONTAINS, BenchmarkCase, load_dataset_from_dict
from rlmstudio.benchmark.matrix_runner import (
    BenchmarkResults,
    MatrixBenchmarkRunner,
    SlotOutcome,
    budget_to_config,
    median_ttft_ms,
)
from rlmstudio.benchmark.scoring import JudgeScorer
from tests.fakes.fake_rlm_engine import FakeRLMEngine


class _FakeLLM:
    """Direct-mode answers keyed by model name; can be told to fail."""

    def __init__(self, model: str, answer: str, *, fail: bool = False) -> None:
        self.model = model
        self.active_model = model
        self._answer = answer
        self._fail = fail

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        if self._fail:
            raise RuntimeError("provider timeout while waiting for tokens")
        return LLMResponseDTO(
            content=self._answer, model=self.model, input_tokens=100, output_tokens=10, ttft_ms=250
        )

    def complete_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        yield self._answer

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def get_pricing(self) -> dict[str, float]:
        return {"input_cost_per_1m": 1.0, "output_cost_per_1m": 2.0}

    def get_completion_cost(self, input_tokens: int, output_tokens: int) -> float:
        return input_tokens * 1e-6 + output_tokens * 2e-6


class _JudgeLLM:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        self.calls += 1
        return LLMResponseDTO(
            content='{"dimensions": {"relevance": 4, "correctness": 4}, "reasoning": "ok"}',
            model="judge",
            input_tokens=1,
            output_tokens=1,
        )

    def complete_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        yield ""

    def count_tokens(self, text: str) -> int:
        return 1

    def get_pricing(self) -> dict[str, float]:
        return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}


DATASET = load_dataset_from_dict(
    {
        "name": "mini",
        "cases": [
            {
                "id": "needle",
                "content": "The window is 03:15 UTC.",
                "query": "When?",
                "expected_answer": "03:15 UTC",
                "match": MATCH_CONTAINS,
                "budget": {"max_steps": 4, "max_time_seconds": 30, "max_cost": 0.1},
            },
            {"id": "open", "content": "Some text.", "query": "Summarise.", "rubric_hint": "short"},
        ],
    }
)


def _fake_slots(
    providers: list[str],
    engines: list[str],
    base_config: RunConfigDTO,
    case: BenchmarkCase | None = None,
) -> list[MatrixSlotDTO]:
    slots: list[MatrixSlotDTO] = []
    for spec in providers:
        provider, _, model = spec.partition("/")
        for engine in engines:
            fail = provider == "broken"
            slots.append(
                MatrixSlotDTO(
                    slot_id=f"{provider}-{engine}",
                    mode=engine,  # type: ignore[arg-type]
                    llm=_FakeLLM(model, "It is 03:15 UTC.", fail=fail),
                    provider=provider,
                    model=model,
                    config=RunConfigDTO(
                        mode=engine,
                        max_steps=base_config.max_steps,
                        max_time_seconds=base_config.max_time_seconds,
                        max_cost=base_config.max_cost,
                    ),
                    engine=FakeRLMEngine(answer="At 03:15 UTC.")
                    if engine == MODE_RLM_OFFICIAL
                    else None,
                )
            )
    return slots


class TestBudgetAndTrace:
    def test_budget_to_config(self) -> None:
        cfg = budget_to_config({"max_steps": 4, "max_time_seconds": 30, "max_cost": 0.1})
        assert (cfg.max_steps, cfg.max_time_seconds, cfg.max_cost, cfg.max_recursion_depth) == (
            4,
            30.0,
            0.1,
            1,
        )

    def test_budget_defaults(self) -> None:
        cfg = budget_to_config({})
        assert cfg.max_steps == RLM_DEFAULT_MAX_STEPS
        assert cfg.max_time_seconds is None
        assert cfg.max_cost is None

    def test_median_ttft(self) -> None:
        trace = [
            {"ttft_ms": 300},
            {TRACE_KEY_ROLE: "execution"},
            {"ttft_ms": 100},
            {"ttft_ms": 200},
        ]
        assert median_ttft_ms(trace) == 200
        assert median_ttft_ms([]) is None
        assert median_ttft_ms(None) is None


class TestRunner:
    def test_runs_every_case_across_providers_and_engines(self) -> None:
        runner = MatrixBenchmarkRunner(
            providers=["openai/gpt-4o-mini", "ollama/qwen3"],
            engines=[MODE_DIRECT, MODE_RLM_OFFICIAL],
            slot_builder=_fake_slots,
        )

        results = runner.run(DATASET)

        assert isinstance(results, BenchmarkResults)
        assert results.case_ids == ["needle", "open"]
        assert len(results.outcomes) == 2 * 2 * 2  # cases × providers × engines
        cells = {(o.case_id, o.provider, o.engine) for o in results.outcomes}
        assert ("needle", "ollama", MODE_RLM_OFFICIAL) in cells
        assert results.reps == 1
        assert results.temperature == 0.0
        assert results.started_at
        assert results.finished_at

    def test_match_pass_only_for_cases_with_expected_answers(self) -> None:
        runner = MatrixBenchmarkRunner(
            providers=["openai/m"],
            engines=[MODE_DIRECT, MODE_RLM_OFFICIAL],
            slot_builder=_fake_slots,
        )

        outcomes = {(o.case_id, o.engine): o for o in runner.run(DATASET).outcomes}

        assert outcomes[("needle", MODE_DIRECT)].match_pass is True
        assert outcomes[("needle", MODE_RLM_OFFICIAL)].match_pass is True
        assert outcomes[("open", MODE_DIRECT)].match_pass is None
        assert outcomes[("needle", MODE_DIRECT)].outcome == OutcomeCategory.SUCCESS.value
        assert outcomes[("needle", MODE_DIRECT)].median_ttft_ms == 250
        assert outcomes[("needle", MODE_RLM_OFFICIAL)].median_ttft_ms is None  # engine has no TTFT
        assert outcomes[("needle", MODE_DIRECT)].total_cost == pytest.approx(100e-6 + 10 * 2e-6)

    def test_failures_are_classified_and_counted_not_dropped(self) -> None:
        runner = MatrixBenchmarkRunner(
            providers=["broken/m"], engines=[MODE_DIRECT], slot_builder=_fake_slots
        )

        outcome = runner.run(DATASET, case_ids=["needle"]).outcomes[0]

        assert not outcome.success
        assert outcome.outcome == OutcomeCategory.TIMEOUT.value  # "timeout" keyword in the error
        assert outcome.match_pass is False
        assert outcome.error is not None

    def test_judge_runs_only_for_usable_outcomes_and_records_provenance(self) -> None:
        judge_llm = _JudgeLLM()
        judge = JudgeScorer(judge_llm, model="judge/model")
        runner = MatrixBenchmarkRunner(
            providers=["openai/m", "broken/m"],
            engines=[MODE_DIRECT],
            slot_builder=_fake_slots,
            judge=judge,
        )

        results = runner.run(DATASET)

        assert results.judge_model == "judge/model"
        assert results.judge_prompt_version == "2.0"
        judged = [o for o in results.outcomes if o.judge is not None]
        assert len(judged) == 2  # the two openai cells; the broken ones are not judged
        assert judge_llm.calls == 2
        assert judged[0].judge is not None
        assert judged[0].judge.overall == 4.0

    def test_reps_limit_and_case_ids(self) -> None:
        runner = MatrixBenchmarkRunner(
            providers=["openai/m"], engines=[MODE_DIRECT], slot_builder=_fake_slots, reps=3
        )

        results = runner.run(DATASET, limit=1)

        assert results.case_ids == ["needle"]
        assert [o.rep for o in results.outcomes] == [1, 2, 3]

    def test_skipped_cases_are_carried_and_unmaterialised_cases_excluded(self) -> None:
        dataset = load_dataset_from_dict(
            {
                "cases": [
                    {"id": "have", "content": "x", "query": "q"},
                    {"id": "missing", "query": "q", "source_url": "https://example.invalid/t"},
                ]
            }
        )
        runner = MatrixBenchmarkRunner(
            providers=["openai/m"], engines=[MODE_DIRECT], slot_builder=_fake_slots
        )

        results = runner.run(dataset, skipped={"missing": "public text not fetched"})

        assert results.case_ids == ["have"]
        assert results.skipped == {"missing": "public text not fetched"}
        assert results.to_dict()["skipped"] == {"missing": "public text not fetched"}

    def test_per_case_budget_reaches_the_slot_builder(self) -> None:
        seen: list[RunConfigDTO] = []

        def builder(
            providers: list[str], engines: list[str], base: RunConfigDTO, case: BenchmarkCase
        ) -> list[MatrixSlotDTO]:
            seen.append(base)
            assert case.id == "needle"
            return _fake_slots(providers, engines, base)

        MatrixBenchmarkRunner(
            providers=["openai/m"], engines=[MODE_DIRECT], slot_builder=builder
        ).run(DATASET, case_ids=["needle"])

        assert seen[0].max_steps == 4
        assert seen[0].max_time_seconds == 30.0
        assert seen[0].max_cost == 0.1

    def test_progress_callback(self) -> None:
        seen: list[SlotOutcome] = []
        MatrixBenchmarkRunner(
            providers=["openai/m"],
            engines=[MODE_DIRECT],
            slot_builder=_fake_slots,
            on_slot_complete=seen.append,
        ).run(DATASET)

        assert [o.case_id for o in seen] == ["needle", "open"]

    @pytest.mark.parametrize(
        "kwargs", [{"providers": [], "engines": ["direct"]}, {"providers": ["a/b"], "engines": []}]
    )
    def test_rejects_empty_axes(self, kwargs: dict[str, list[str]]) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            MatrixBenchmarkRunner(slot_builder=_fake_slots, **kwargs)

    def test_rejects_zero_reps(self) -> None:
        with pytest.raises(ValueError, match="reps"):
            MatrixBenchmarkRunner(
                providers=["a/b"], engines=["direct"], reps=0, slot_builder=_fake_slots
            )


class _RecordingUseCase:
    """Stands in for RunMatrixComparisonUseCase, recording how slots were batched."""

    batches: list[tuple[int | None, list[str]]] = []

    def __init__(self, max_workers: int | None = None) -> None:
        self._max_workers = max_workers

    def execute(self, content: str, query: str, slots: list[MatrixSlotDTO]) -> Any:
        type(self).batches.append((self._max_workers, [s.mode for s in slots]))
        return SimpleNamespace(
            slots=[
                MatrixSlotResultDTO(
                    slot_id=s.slot_id,
                    label=s.slot_id,
                    mode=s.mode,
                    provider=s.provider or "p",
                    model=s.model or "m",
                    result=RunResultDTO(answer="a", mode_used=s.mode, success=True),
                )
                for s in slots
            ]
        )


@pytest.fixture
def recording_use_case(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingUseCase]:
    _RecordingUseCase.batches = []
    monkeypatch.setattr(matrix_runner, "RunMatrixComparisonUseCase", _RecordingUseCase)
    return _RecordingUseCase


class TestOfficialSlotsShareOneInProcessREPL:
    """The official engine admits one in-process run at a time, process wide.

    Left in the matrix's own thread pool, a grid with two providers would lose
    every official cell but one to "engine busy" — a benchmark reporting
    failures it created itself.
    """

    def _run(self, sandbox_type: str | None, providers: list[str]) -> None:
        MatrixBenchmarkRunner(
            providers=providers,
            engines=[MODE_DIRECT, MODE_RLM_OFFICIAL],
            slot_builder=_fake_slots,
            sandbox_type=sandbox_type,
        ).run(DATASET, case_ids=["needle"])

    def test_two_official_slots_run_one_at_a_time(
        self, recording_use_case: type[_RecordingUseCase]
    ) -> None:
        self._run(None, ["openai/a", "anthropic/b"])

        batches = recording_use_case.batches
        assert len(batches) == 2
        parallel_workers, parallel_modes = batches[0]
        serial_workers, serial_modes = batches[1]
        assert parallel_workers is None
        assert set(parallel_modes) == {MODE_DIRECT}
        assert serial_workers == 1
        assert set(serial_modes) == {MODE_RLM_OFFICIAL}

    def test_every_slot_still_reaches_the_report(
        self, recording_use_case: type[_RecordingUseCase]
    ) -> None:
        seen: list[SlotOutcome] = []
        MatrixBenchmarkRunner(
            providers=["openai/a", "anthropic/b"],
            engines=[MODE_DIRECT, MODE_RLM_OFFICIAL],
            slot_builder=_fake_slots,
            on_slot_complete=seen.append,
        ).run(DATASET, case_ids=["needle"])

        assert len(seen) == 4
        assert sum(1 for o in seen if o.engine == MODE_RLM_OFFICIAL) == 2

    def test_a_single_official_slot_is_not_held_back(
        self, recording_use_case: type[_RecordingUseCase]
    ) -> None:
        self._run(None, ["openai/a"])

        assert len(recording_use_case.batches) == 1

    def test_the_docker_sandbox_needs_no_serialising(
        self, recording_use_case: type[_RecordingUseCase]
    ) -> None:
        """A container per run means no shared REPL to queue for."""
        self._run("docker", ["openai/a", "anthropic/b"])

        assert len(recording_use_case.batches) == 1


class TestJudgeFailureDoesNotEndTheRun:
    """A paid run is hours of provider calls; one judge hiccup must not void it."""

    class _AngryJudgeLLM(_JudgeLLM):
        def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
            raise RuntimeError("429 rate limit exceeded")

    def test_the_cell_loses_its_verdict_and_says_why(self) -> None:
        judge = JudgeScorer(self._AngryJudgeLLM(), model="judge/x")  # type: ignore[arg-type]
        results = MatrixBenchmarkRunner(
            providers=["openai/m"],
            engines=[MODE_DIRECT],
            slot_builder=_fake_slots,
            judge=judge,
        ).run(DATASET)

        assert [o.case_id for o in results.outcomes] == ["needle", "open"]
        assert all(o.judge is None for o in results.outcomes)
        assert all("429" in (o.judge_error or "") for o in results.outcomes)

    def test_a_judge_reply_that_is_not_an_object_scores_unparsed(self) -> None:
        class _ListJudgeLLM(_JudgeLLM):
            def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
                return LLMResponseDTO(
                    content="[4, 4]", model="judge", input_tokens=1, output_tokens=1
                )

        judge = JudgeScorer(_ListJudgeLLM(), model="judge/x")  # type: ignore[arg-type]
        results = MatrixBenchmarkRunner(
            providers=["openai/m"],
            engines=[MODE_DIRECT],
            slot_builder=_fake_slots,
            judge=judge,
        ).run(DATASET, case_ids=["needle"])

        verdict = results.outcomes[0].judge
        assert verdict is not None
        assert verdict.parsed is False
        assert results.outcomes[0].judge_error is None  # it answered, just not usefully


class TestPartialResults:
    """What a stopped run had already paid for is still worth keeping."""

    def test_completed_cells_are_packaged_with_the_dataset_provenance(self) -> None:
        runner = MatrixBenchmarkRunner(
            providers=["openai/m"], engines=[MODE_DIRECT], slot_builder=_fake_slots
        )
        done: list[SlotOutcome] = []
        runner_with_progress = MatrixBenchmarkRunner(
            providers=["openai/m"],
            engines=[MODE_DIRECT],
            slot_builder=_fake_slots,
            on_slot_complete=done.append,
        )
        runner_with_progress.run(DATASET, case_ids=["needle"])

        partial = runner.partial_results(DATASET, done, skipped={"other": "no corpus"})

        assert partial.dataset_name == DATASET.name
        assert partial.case_ids == ["needle"]
        assert partial.skipped == {"other": "no corpus"}
        assert len(partial.outcomes) == len(done)
        assert partial.finished_at
