"""FR-1 (specs/interop-official-rlm): ``rlm_official`` is a first-class mode literal.

The constant lives in ``application/sandbox_vars.py`` next to ``MODE_*``; the
``ExecutionMode`` / ``ChatMode`` aliases defined there are what every request
model and the matrix use case import, so widening happens in exactly one place.
"""

from __future__ import annotations

from typing import get_args

import pytest

from rlmstudio.api import _determine_auto_mode
from rlmstudio.application.sandbox_vars import (
    MODE_DIRECT,
    MODE_RAG,
    MODE_RLM,
    MODE_RLM_OFFICIAL,
    ChatMode,
    ExecutionMode,
)
from rlmstudio.application.use_cases.run_matrix_comparison import SlotMode
from rlmstudio.server.models import (
    ChatProviderCreateRequest,
    ChatProviderUpdateRequest,
    ChatRequest,
)
from rlmstudio.server.routes.compare_matrix import (
    CompareMatrixRequest,
    CompareMatrixRequestV2,
    CompareMatrixUnifiedRequest,
)

ALL_FOUR = [MODE_DIRECT, MODE_RAG, MODE_RLM, MODE_RLM_OFFICIAL]


class TestModeLiteral:
    def test_constant_value(self) -> None:
        assert MODE_RLM_OFFICIAL == "rlm_official"

    def test_execution_mode_alias_includes_it(self) -> None:
        assert set(get_args(ExecutionMode)) == set(ALL_FOUR)

    def test_chat_mode_alias_includes_it(self) -> None:
        assert MODE_RLM_OFFICIAL in get_args(ChatMode)

    def test_slot_mode_is_the_shared_alias(self) -> None:
        assert get_args(SlotMode) == get_args(ExecutionMode)


class TestRequestModelsAcceptMode:
    def test_chat_request(self) -> None:
        req = ChatRequest(query="q", content="c", mode=MODE_RLM_OFFICIAL)
        assert req.mode == MODE_RLM_OFFICIAL

    def test_chat_provider_create(self) -> None:
        req = ChatProviderCreateRequest(
            name="n", llm_provider_id="lp", execution_mode=MODE_RLM_OFFICIAL
        )
        assert req.execution_mode == MODE_RLM_OFFICIAL

    def test_chat_provider_update(self) -> None:
        req = ChatProviderUpdateRequest(execution_mode=MODE_RLM_OFFICIAL)
        assert req.execution_mode == MODE_RLM_OFFICIAL

    def test_matrix_v1_accepts_all_four_modes(self) -> None:
        req = CompareMatrixRequest(query="q", chat_provider_ids=["cp"], modes=ALL_FOUR)
        assert req.modes == ALL_FOUR

    def test_matrix_v2_accepts_all_four_modes(self) -> None:
        req = CompareMatrixRequestV2(query="q", llm_provider_ids=["lp"], modes=ALL_FOUR)
        assert req.modes == ALL_FOUR

    def test_matrix_unified_accepts_all_four_modes(self) -> None:
        req = CompareMatrixUnifiedRequest(query="q", llm_provider_ids=["lp"], modes=ALL_FOUR)
        assert req.modes == ALL_FOUR

    def test_matrix_still_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValueError):
            CompareMatrixRequestV2(
                query="q",
                llm_provider_ids=["lp"],
                modes=["magic"],  # type: ignore[list-item]
            )


class TestAutoNeverSelectsOfficialEngine:
    @pytest.mark.parametrize("chars", [1, 4 * 20_000, 4 * 200_000])
    def test_auto_resolves_to_a_built_in_mode(self, chars: int) -> None:
        mode = _determine_auto_mode("x" * chars)
        assert mode != MODE_RLM_OFFICIAL
        assert mode in {MODE_DIRECT, MODE_RAG, MODE_RLM}
