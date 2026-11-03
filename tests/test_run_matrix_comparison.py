"""Tests for RunMatrixComparisonUseCase."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import pytest

from rlmstudio.application.dto import (
    ExecutionResultDTO,
    LLMResponseDTO,
    RunConfigDTO,
)
from rlmstudio.application.use_cases.run_matrix_comparison import (
    MatrixComparisonResultDTO,
    MatrixSlotDTO,
    RunMatrixComparisonUseCase,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def test_copy_config_for_slot_preserves_subcall_concurrency() -> None:
    from rlmstudio.application.use_cases.run_matrix_comparison import (
        _copy_config_for_slot,
    )

    base = RunConfigDTO(mode="compare", subcall_max_concurrency=2)
    base.extra["collection"] = "base"

    slot = _copy_config_for_slot(base, "rlm")

    assert slot.subcall_max_concurrency == 2
    assert slot.mode == "rlm"
    assert slot.extra == {"collection": "base"}
    assert slot.extra is not base.extra


class FakeLLM:
    """LLMPort-compliant fake with configurable cost and latency."""

    def __init__(
        self,
        response: str = "hello",
        *,
        input_cost_per_1m: float = 0.0,
        output_cost_per_1m: float = 0.0,
        input_tokens: int = 10,
        output_tokens: int = 5,
        latency: float = 0.0,
        name: str = "fake",
    ) -> None:
        self._response = response
        self._input_cost = input_cost_per_1m
        self._output_cost = output_cost_per_1m
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._latency = latency
        self._name = name

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        if self._latency:
            time.sleep(self._latency)
        return LLMResponseDTO(
            content=self._response,
            model=self._name,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )

    def complete_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        result = self.complete(messages)
        yield result.content

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def get_pricing(self) -> dict[str, float]:
        return {
            "input_cost_per_1m": self._input_cost,
            "output_cost_per_1m": self._output_cost,
        }


class RaisingLLM:
    """LLM fake that raises on every call — used to force slot failures."""

    def __init__(self, error: str = "boom") -> None:
        self._error = error

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        raise RuntimeError(self._error)

    def complete_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        raise RuntimeError(self._error)
        yield ""  # pragma: no cover — unreachable

    def count_tokens(self, text: str) -> int:
        return 1

    def get_pricing(self) -> dict[str, float]:
        return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}


class FakeSandbox:
    """Minimal SandboxPort-compliant fake (for rlm-mode slots)."""

    def __init__(self) -> None:
        self._namespace: dict[str, Any] = {}

    def execute(self, code: str) -> ExecutionResultDTO:
        return ExecutionResultDTO(stdout="ok")

    def reset(self) -> None:
        self._namespace.clear()

    def is_healthy(self) -> bool:
        return True

    def set_variable(self, name: str, value: Any) -> None:
        self._namespace[name] = value

    def get_variable(self, name: str) -> Any | None:
        return self._namespace.get(name)


# ---------------------------------------------------------------------------
# Happy-path execution
# ---------------------------------------------------------------------------


class TestBasicExecution:
    def test_runs_all_slots_and_preserves_order(self) -> None:
        uc = RunMatrixComparisonUseCase()
        slots = [
            MatrixSlotDTO(
                slot_id="s1",
                mode="direct",
                llm=FakeLLM(response="A", name="a"),
                provider="openai",
                model="gpt-4o",
            ),
            MatrixSlotDTO(
                slot_id="s2",
                mode="direct",
                llm=FakeLLM(response="B", name="b"),
                provider="anthropic",
                model="claude-sonnet",
            ),
            MatrixSlotDTO(
                slot_id="s3",
                mode="direct",
                llm=FakeLLM(response="C", name="c"),
                provider="ollama",
                model="llama3.2",
            ),
        ]

        out = uc.execute("doc", "q", slots)

        assert isinstance(out, MatrixComparisonResultDTO)
        assert len(out.slots) == 3
        assert [s.slot_id for s in out.slots] == ["s1", "s2", "s3"]
        assert [s.result.answer for s in out.slots] == ["A", "B", "C"]
        assert all(s.result.success for s in out.slots)
        assert out.comparison_group_id  # non-empty UUID
        assert out.total_elapsed >= 0.0

    def test_empty_slots_raises(self) -> None:
        uc = RunMatrixComparisonUseCase()
        with pytest.raises(ValueError, match="non-empty"):
            uc.execute("doc", "q", [])

    def test_duplicate_slot_ids_raise(self) -> None:
        uc = RunMatrixComparisonUseCase()
        slots = [
            MatrixSlotDTO(slot_id="dup", mode="direct", llm=FakeLLM()),
            MatrixSlotDTO(slot_id="dup", mode="direct", llm=FakeLLM()),
        ]
        with pytest.raises(ValueError, match="Duplicate slot_id"):
            uc.execute("doc", "q", slots)

    def test_exceeds_max_slots_raises(self) -> None:
        uc = RunMatrixComparisonUseCase()
        slots = [
            MatrixSlotDTO(slot_id=f"s{i}", mode="direct", llm=FakeLLM())
            for i in range(RunMatrixComparisonUseCase.MAX_SLOTS + 1)
        ]
        with pytest.raises(ValueError, match="Too many slots"):
            uc.execute("doc", "q", slots)

    def test_unsupported_mode_raises(self) -> None:
        uc = RunMatrixComparisonUseCase()
        slots = [
            MatrixSlotDTO(slot_id="s1", mode="magic", llm=FakeLLM()),  # type: ignore[arg-type]
        ]
        with pytest.raises(ValueError, match="unsupported mode"):
            uc.execute("doc", "q", slots)

    def test_rlm_without_sandbox_raises(self) -> None:
        uc = RunMatrixComparisonUseCase()
        slots = [
            MatrixSlotDTO(slot_id="s1", mode="rlm", llm=FakeLLM()),
        ]
        with pytest.raises(ValueError, match="requires a sandbox"):
            uc.execute("doc", "q", slots)

    def test_rag_without_embedder_raises(self) -> None:
        uc = RunMatrixComparisonUseCase()
        slots = [
            MatrixSlotDTO(slot_id="s1", mode="rag", llm=FakeLLM()),
        ]
        with pytest.raises(ValueError, match="requires embedder and storage"):
            uc.execute("doc", "q", slots)


# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------


class TestParallelism:
    def test_slots_run_in_parallel(self) -> None:
        """Wall-clock should be less than the sum of per-slot latencies."""
        uc = RunMatrixComparisonUseCase()
        slots = [
            MatrixSlotDTO(
                slot_id=f"s{i}",
                mode="direct",
                llm=FakeLLM(response=str(i), latency=0.2, name=f"slow{i}"),
            )
            for i in range(4)
        ]
        start = time.monotonic()
        out = uc.execute("doc", "q", slots)
        elapsed = time.monotonic() - start

        # Serial would take ~0.8s; parallel should be well under that.
        assert elapsed < 0.6, f"Expected parallel execution, got {elapsed:.2f}s"
        assert all(s.result.success for s in out.slots)

    def test_max_workers_override(self) -> None:
        uc = RunMatrixComparisonUseCase(max_workers=1)
        slots = [
            MatrixSlotDTO(
                slot_id=f"s{i}",
                mode="direct",
                llm=FakeLLM(response=str(i), latency=0.05),
            )
            for i in range(3)
        ]
        # max_workers=1 forces sequential execution — just make sure it still succeeds.
        out = uc.execute("doc", "q", slots)
        assert len(out.slots) == 3
        assert all(s.result.success for s in out.slots)


# ---------------------------------------------------------------------------
# Slot-level failure handling
# ---------------------------------------------------------------------------


class TestSlotFailures:
    def test_one_slot_failure_does_not_fail_others(self) -> None:
        uc = RunMatrixComparisonUseCase()
        slots = [
            MatrixSlotDTO(slot_id="ok1", mode="direct", llm=FakeLLM(response="hi")),
            MatrixSlotDTO(slot_id="bad", mode="direct", llm=RaisingLLM("nope")),
            MatrixSlotDTO(slot_id="ok2", mode="direct", llm=FakeLLM(response="bye")),
        ]
        out = uc.execute("doc", "q", slots)

        by_id = {s.slot_id: s for s in out.slots}
        assert by_id["ok1"].result.success
        assert by_id["ok1"].result.answer == "hi"
        assert by_id["ok2"].result.success
        assert by_id["ok2"].result.answer == "bye"
        # The failed slot returns through the use case as success=False
        # (RunDirectUseCase catches the exception internally).
        assert not by_id["bad"].result.success
        assert by_id["bad"].result.error is not None

    def test_failures_ranked_last(self) -> None:
        uc = RunMatrixComparisonUseCase()
        slots = [
            MatrixSlotDTO(slot_id="bad", mode="direct", llm=RaisingLLM()),
            MatrixSlotDTO(slot_id="ok", mode="direct", llm=FakeLLM(response="hi")),
        ]
        out = uc.execute("doc", "q", slots, ranking_metric="cost")

        # ok is at input index 1, bad at index 0.
        # Ranking: successful first → [1], then failures → [0].
        assert out.ranking == [1, 0]
        assert out.best is not None
        assert out.best.slot_id == "ok"


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------


class TestRanking:
    @staticmethod
    def _build_slots() -> list[MatrixSlotDTO]:
        # Three direct slots with different costs/tokens/answer lengths.
        return [
            MatrixSlotDTO(
                slot_id="expensive",
                mode="direct",
                llm=FakeLLM(
                    response="long answer here with more text",
                    input_cost_per_1m=1000.0,
                    output_cost_per_1m=1000.0,
                    input_tokens=1000,
                    output_tokens=1000,
                ),
            ),
            MatrixSlotDTO(
                slot_id="cheap",
                mode="direct",
                llm=FakeLLM(
                    response="short",
                    input_cost_per_1m=1.0,
                    output_cost_per_1m=1.0,
                    input_tokens=10,
                    output_tokens=5,
                ),
            ),
            MatrixSlotDTO(
                slot_id="medium",
                mode="direct",
                llm=FakeLLM(
                    response="medium reply",
                    input_cost_per_1m=100.0,
                    output_cost_per_1m=100.0,
                    input_tokens=100,
                    output_tokens=100,
                ),
            ),
        ]

    def test_rank_by_cost(self) -> None:
        uc = RunMatrixComparisonUseCase()
        slots = self._build_slots()
        out = uc.execute("doc", "q", slots, ranking_metric="cost")

        ranked_ids = [out.slots[i].slot_id for i in out.ranking]
        assert ranked_ids == ["cheap", "medium", "expensive"]
        assert out.best is not None
        assert out.best.slot_id == "cheap"
        assert out.ranking_metric == "cost"

    def test_rank_by_tokens(self) -> None:
        uc = RunMatrixComparisonUseCase()
        slots = self._build_slots()
        out = uc.execute("doc", "q", slots, ranking_metric="tokens")

        ranked_ids = [out.slots[i].slot_id for i in out.ranking]
        assert ranked_ids == ["cheap", "medium", "expensive"]

    def test_rank_by_latency(self) -> None:
        uc = RunMatrixComparisonUseCase()
        slots = [
            MatrixSlotDTO(slot_id="slow", mode="direct", llm=FakeLLM(response="x", latency=0.15)),
            MatrixSlotDTO(slot_id="fast", mode="direct", llm=FakeLLM(response="x", latency=0.01)),
        ]
        out = uc.execute("doc", "q", slots, ranking_metric="latency")

        ranked_ids = [out.slots[i].slot_id for i in out.ranking]
        assert ranked_ids == ["fast", "slow"]

    def test_rank_answer_per_cost(self) -> None:
        uc = RunMatrixComparisonUseCase()
        # Two slots with identical cost but different answer lengths.
        slots = [
            MatrixSlotDTO(
                slot_id="short",
                mode="direct",
                llm=FakeLLM(
                    response="hi",
                    input_cost_per_1m=100.0,
                    output_cost_per_1m=100.0,
                ),
            ),
            MatrixSlotDTO(
                slot_id="long",
                mode="direct",
                llm=FakeLLM(
                    response="hello world this is a much longer reply",
                    input_cost_per_1m=100.0,
                    output_cost_per_1m=100.0,
                ),
            ),
        ]
        out = uc.execute("doc", "q", slots, ranking_metric="answer_per_cost")
        ranked_ids = [out.slots[i].slot_id for i in out.ranking]
        # Long reply at same cost → more answer per dollar → ranks first.
        assert ranked_ids == ["long", "short"]


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


class TestResultHelpers:
    def test_get_slot_by_id(self) -> None:
        uc = RunMatrixComparisonUseCase()
        slots = [
            MatrixSlotDTO(slot_id="a", mode="direct", llm=FakeLLM(response="A")),
            MatrixSlotDTO(slot_id="b", mode="direct", llm=FakeLLM(response="B")),
        ]
        out = uc.execute("doc", "q", slots)

        assert out.get_slot("a") is not None
        assert out.get_slot("a").result.answer == "A"  # type: ignore[union-attr]
        assert out.get_slot("missing") is None

    def test_default_label_when_empty(self) -> None:
        uc = RunMatrixComparisonUseCase()
        slots = [
            MatrixSlotDTO(
                slot_id="s1",
                mode="direct",
                llm=FakeLLM(),
                provider="openai",
                model="gpt-4o",
                # label intentionally blank → use default
            ),
        ]
        out = uc.execute("doc", "q", slots)
        assert out.slots[0].label == "openai/gpt-4o · direct"

    def test_explicit_label_is_preserved(self) -> None:
        uc = RunMatrixComparisonUseCase()
        slots = [
            MatrixSlotDTO(
                slot_id="s1",
                mode="direct",
                llm=FakeLLM(),
                label="My Custom Label",
            ),
        ]
        out = uc.execute("doc", "q", slots)
        assert out.slots[0].label == "My Custom Label"


# ---------------------------------------------------------------------------
# Per-slot config isolation
# ---------------------------------------------------------------------------


class TestConfigIsolation:
    def test_extra_dict_not_shared_between_slots(self) -> None:
        """Each slot must get its own ``config.extra`` copy so mutations
        in one slot don't leak into another (e.g. RAG collection key)."""
        from rlmstudio.application.use_cases.run_matrix_comparison import (
            _copy_config_for_slot,
        )

        base = RunConfigDTO(mode="compare", extra={"k": "v"})
        a = _copy_config_for_slot(base, "direct")
        b = _copy_config_for_slot(base, "rlm")

        a.extra["k"] = "mutated_a"
        assert b.extra == {"k": "v"}
        assert base.extra == {"k": "v"}
        assert a.mode == "direct"
        assert b.mode == "rlm"
