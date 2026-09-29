"""Matrix-compare dispatch and ranking for ``rlm_official`` slots (interop FR-5, OQ-2)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from rlmstudio.application.dto import LLMResponseDTO, RunResultDTO
from rlmstudio.application.sandbox_vars import (
    MODE_DIRECT,
    MODE_RLM_OFFICIAL,
    RESULT_KEY_COST_KNOWN,
    RESULT_KEY_ENGINE_VERSION,
)
from rlmstudio.application.use_cases.run_matrix_comparison import (
    MatrixSlotDTO,
    MatrixSlotResultDTO,
    RunMatrixComparisonUseCase,
)
from tests.fakes.fake_rlm_engine import FAKE_ENGINE_VERSION, FakeRLMEngine


class _FakeLLM:
    """Just enough LLMPort for a direct slot next to an engine slot."""

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        return LLMResponseDTO(
            content="direct answer", model="fake", input_tokens=5, output_tokens=2
        )

    def complete_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        yield "direct answer"

    def count_tokens(self, text: str) -> int:
        return 1

    def get_pricing(self) -> dict[str, float]:
        return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}


class TestDispatch:
    def test_official_slot_runs_through_its_engine(self) -> None:
        engine = FakeRLMEngine(answer="official answer")
        slots = [
            MatrixSlotDTO(slot_id="d", mode=MODE_DIRECT, llm=_FakeLLM(), provider="openai"),
            MatrixSlotDTO(
                slot_id="o",
                mode=MODE_RLM_OFFICIAL,
                llm=_FakeLLM(),
                provider="openai",
                engine=engine,
            ),
        ]

        out = RunMatrixComparisonUseCase().execute("doc", "q", slots)

        official = out.get_slot("o")
        assert official is not None
        assert official.mode == MODE_RLM_OFFICIAL
        assert official.result.success
        assert official.result.answer == "official answer"
        assert official.result.mode_used == MODE_RLM_OFFICIAL
        assert official.result.metadata[RESULT_KEY_ENGINE_VERSION] == FAKE_ENGINE_VERSION
        assert engine.calls[0][:2] == ("doc", "q")
        assert engine.calls[0][2].mode == MODE_RLM_OFFICIAL

    def test_official_slot_without_engine_is_rejected_up_front(self) -> None:
        slots = [MatrixSlotDTO(slot_id="o", mode=MODE_RLM_OFFICIAL, llm=_FakeLLM())]

        with pytest.raises(ValueError, match="requires an engine adapter"):
            RunMatrixComparisonUseCase().execute("doc", "q", slots)

    def test_engine_failure_becomes_a_failed_slot_not_a_crash(self) -> None:
        engine = FakeRLMEngine(error=RuntimeError("backend down"))
        slots = [
            MatrixSlotDTO(slot_id="d", mode=MODE_DIRECT, llm=_FakeLLM()),
            MatrixSlotDTO(slot_id="o", mode=MODE_RLM_OFFICIAL, llm=_FakeLLM(), engine=engine),
        ]

        out = RunMatrixComparisonUseCase().execute("doc", "q", slots)

        assert out.get_slot("d").result.success  # type: ignore[union-attr]
        official = out.get_slot("o")
        assert official is not None
        assert not official.result.success
        assert official.result.error == "backend down"


def _slot_result(slot_id: str, *, cost: float, cost_known: bool = True) -> MatrixSlotResultDTO:
    return MatrixSlotResultDTO(
        slot_id=slot_id,
        label=slot_id,
        mode=MODE_RLM_OFFICIAL,
        provider="p",
        model="m",
        result=RunResultDTO(
            answer="a" * 10,
            mode_used=MODE_RLM_OFFICIAL,
            success=True,
            total_cost=cost,
            metadata={RESULT_KEY_COST_KNOWN: cost_known},
        ),
    )


class TestUnknownCostRanking:
    """OQ-2: a slot whose engine reported no price is not "free" — it ranks last."""

    def test_cost_metric_puts_unknown_cost_after_every_known_cost(self) -> None:
        slots = [
            _slot_result("unknown", cost=0.0, cost_known=False),
            _slot_result("expensive", cost=1.0),
            _slot_result("cheap", cost=0.1),
        ]

        ranking = RunMatrixComparisonUseCase._rank(slots, "cost")

        assert [slots[i].slot_id for i in ranking] == ["cheap", "expensive", "unknown"]

    def test_answer_per_cost_metric_does_the_same(self) -> None:
        slots = [
            _slot_result("unknown", cost=0.0, cost_known=False),
            _slot_result("known", cost=0.5),
        ]

        ranking = RunMatrixComparisonUseCase._rank(slots, "answer_per_cost")

        assert [slots[i].slot_id for i in ranking] == ["known", "unknown"]

    def test_an_unpriced_engine_slot_ranks_last_end_to_end(self) -> None:
        """The whole chain, not just ``_rank``: the flag the engine set is honoured.

        The hand-built cases above start from a DTO that already carries
        ``cost_known``; this one lets the engine report it and runs the real
        dispatch, so a slot that came back unpriced cannot quietly rank first
        because the key went missing on the way.
        """
        unpriced = FakeRLMEngine(
            answer="unpriced answer",
            total_cost=0.0,
            metadata={RESULT_KEY_COST_KNOWN: False},
        )
        priced = FakeRLMEngine(answer="priced answer", total_cost=0.25)
        slots = [
            MatrixSlotDTO(
                slot_id="unpriced", mode=MODE_RLM_OFFICIAL, llm=_FakeLLM(), engine=unpriced
            ),
            MatrixSlotDTO(slot_id="priced", mode=MODE_RLM_OFFICIAL, llm=_FakeLLM(), engine=priced),
        ]

        out = RunMatrixComparisonUseCase().execute("doc", "q", slots)

        assert out.get_slot("unpriced").result.metadata[RESULT_KEY_COST_KNOWN] is False  # type: ignore[union-attr]
        ranking = RunMatrixComparisonUseCase._rank(out.slots, "cost")
        assert [out.slots[i].slot_id for i in ranking] == ["priced", "unpriced"]

    def test_a_genuinely_free_slot_still_ranks_first(self) -> None:
        slots = [
            _slot_result("paid", cost=0.5),
            _slot_result("free", cost=0.0, cost_known=True),
        ]

        ranking = RunMatrixComparisonUseCase._rank(slots, "cost")

        assert [slots[i].slot_id for i in ranking] == ["free", "paid"]
