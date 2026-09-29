"""`rlmstudio.api.build_matrix_slots` — public, budget-aware, and `rlm_official`-capable."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest

from rlmstudio.api import build_matrix_slots, compare_matrix
from rlmstudio.application.dto import LLMResponseDTO, RunConfigDTO
from rlmstudio.application.sandbox_vars import MODE_DIRECT, MODE_RAG, MODE_RLM_OFFICIAL
from tests.fakes.fake_rlm_engine import FakeRLMEngine


class _FakeLLM:
    def __init__(self, model: str = "m", **kwargs: Any) -> None:
        self.model = model
        self.active_model = model

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        return LLMResponseDTO(
            content=f"answer from {self.model}", model=self.model, input_tokens=5, output_tokens=2
        )

    def complete_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        yield "x"

    def count_tokens(self, text: str) -> int:
        return 1

    def get_pricing(self) -> dict[str, float]:
        return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}

    def get_completion_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.001


class _EngineFactory:
    """Stands in for RlmsEngineAdapter: records constructor kwargs, returns a fake engine."""

    def __init__(self) -> None:
        self.kwargs: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> FakeRLMEngine:
        self.kwargs.append(kwargs)
        return FakeRLMEngine(answer="official answer")


_COMMON = {
    "api_key": None,
    "api_base": None,
    "temperature": 0.0,
    "max_tokens": None,
    "timeout": None,
    "max_steps": 5,
    "num_retries": None,
    "embedding_api_key": None,
}


class TestBuildMatrixSlots:
    def test_official_slot_gets_an_engine_priced_by_its_llm(self) -> None:
        factory = _EngineFactory()
        with (
            patch("rlmstudio.api.LiteLLMAdapter", side_effect=_FakeLLM),
            patch("rlmstudio.api.RlmsEngineAdapter", side_effect=factory),
        ):
            slots = build_matrix_slots(
                ["openai/gpt-4o-mini"], [MODE_DIRECT, MODE_RLM_OFFICIAL], **_COMMON
            )

        by_mode = {s.mode: s for s in slots}
        assert by_mode[MODE_DIRECT].engine is None
        assert by_mode[MODE_RLM_OFFICIAL].engine is not None
        kw = factory.kwargs[0]
        assert (kw["backend"], kw["model"]) == ("openai", "gpt-4o-mini")
        assert kw["base_url"] is None  # cloud backend: no default endpoint
        assert kw["cost_fn"] == by_mode[MODE_RLM_OFFICIAL].llm.get_completion_cost  # type: ignore[attr-defined]
        assert "sandbox_type" not in kw  # adapter default unless requested

    def test_local_backend_falls_back_to_the_catalog_endpoint(self) -> None:
        factory = _EngineFactory()
        with (
            patch("rlmstudio.api.LiteLLMAdapter", side_effect=_FakeLLM),
            patch("rlmstudio.api.RlmsEngineAdapter", side_effect=factory),
        ):
            build_matrix_slots(
                ["ollama/qwen3:8b"], [MODE_RLM_OFFICIAL], **_COMMON, sandbox_type="docker"
            )

        kw = factory.kwargs[0]
        assert kw["base_url"] == "http://localhost:11434"
        assert kw["sandbox_type"] == "docker"

    def test_explicit_api_base_wins_over_the_catalog(self) -> None:
        factory = _EngineFactory()
        with (
            patch("rlmstudio.api.LiteLLMAdapter", side_effect=_FakeLLM),
            patch("rlmstudio.api.RlmsEngineAdapter", side_effect=factory),
        ):
            build_matrix_slots(
                ["vllm/Qwen/Qwen2.5-7B"],
                [MODE_RLM_OFFICIAL],
                **{**_COMMON, "api_base": "http://spark:8000/v1"},
            )

        assert factory.kwargs[0]["base_url"] == "http://spark:8000/v1"
        assert factory.kwargs[0]["model"] == "Qwen/Qwen2.5-7B"

    def test_base_config_seeds_every_slot_and_its_max_steps_wins(self) -> None:
        base = RunConfigDTO(
            max_steps=24,
            max_time_seconds=900.0,
            max_cost=1.0,
            max_recursion_depth=2,
            extra={"k": "v"},
        )
        with patch("rlmstudio.api.LiteLLMAdapter", side_effect=_FakeLLM):
            slots = build_matrix_slots(
                ["openai/m"],
                [MODE_DIRECT, MODE_RAG],
                **{**_COMMON, "api_key": "sk"},
                base_config=base,
            )

        for slot in slots:
            cfg = slot.config
            assert cfg is not None
            assert (cfg.max_steps, cfg.max_time_seconds, cfg.max_cost, cfg.max_recursion_depth) == (
                24,
                900.0,
                1.0,
                2,
            )
            assert cfg.mode == slot.mode
            assert cfg.provider == "openai"
            assert cfg.api_key == "sk"
            assert cfg.extra["k"] == "v"
        rag_extra = next(s.config.extra for s in slots if s.mode == MODE_RAG)  # type: ignore[union-attr]
        assert rag_extra["collection"].startswith("rag_")
        assert base.extra == {"k": "v"}  # the base is not mutated

    def test_without_base_config_max_steps_argument_applies(self) -> None:
        with patch("rlmstudio.api.LiteLLMAdapter", side_effect=_FakeLLM):
            slots = build_matrix_slots(["openai/m"], [MODE_DIRECT], **_COMMON)

        assert slots[0].config is not None
        assert slots[0].config.max_steps == 5


class TestPublicCompareMatrixWithOfficialEngine:
    def test_runs_the_engine_as_a_slot(self) -> None:
        factory = _EngineFactory()
        with (
            patch("rlmstudio.api.LiteLLMAdapter", side_effect=_FakeLLM),
            patch("rlmstudio.api.RlmsEngineAdapter", side_effect=factory),
        ):
            result = compare_matrix(
                "doc", "q", ["openai/gpt-4o-mini"], modes=[MODE_DIRECT, MODE_RLM_OFFICIAL]
            )

        by_mode = {s.mode: s for s in result.slots}
        assert by_mode[MODE_RLM_OFFICIAL].success
        assert by_mode[MODE_RLM_OFFICIAL].answer == "official answer"
        assert by_mode[MODE_DIRECT].answer == "answer from gpt-4o-mini"

    def test_invalid_mode_message_lists_the_official_engine(self) -> None:
        with pytest.raises(ValueError, match="rlm_official"):
            compare_matrix("doc", "q", ["openai/m"], modes=["magic"])
