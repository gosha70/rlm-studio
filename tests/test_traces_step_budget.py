"""Traces shows the run's own step count and step limit (M1, M2 display).

Before: the budget line counted trace rows (25 for a 13-step run, so the
chat footer and Traces disagreed with the run's own message) and always
showed the global limit (16) for a run whose profile allowed 32.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from rlmstudio.application.dto import RunResultDTO
from rlmstudio.server.dependencies import AppState, ExecutionRecord
from rlmstudio.server.models import ChatProviderConfig
from rlmstudio.server.routes.chat import _record_telemetry
from rlmstudio.server.routes.traces import (
    _trace_from_execution_record,
    _trace_from_telemetry,
)

_RUN_STEPS = 13
_TRACE_ROWS = 25
_PROFILE_STEPS = 32
_GLOBAL_STEPS = 16


def _state_with_cp() -> tuple[AppState, ChatProviderConfig]:
    state = AppState(load_from_disk=False)
    state.config.budget.max_steps = _GLOBAL_STEPS
    now = datetime.now(timezone.utc)
    cp = ChatProviderConfig(
        id=str(uuid.uuid4()),
        name="Qwen RLM",
        llm_provider="openai",
        llm_model="gpt-4o",
        execution_mode="rlm",
        rlm_max_steps=_PROFILE_STEPS,
        created_at=now,
        updated_at=now,
    )
    state.config.chat_providers.append(cp)
    return state, cp


def _execution(cp_id: str | None, result: dict) -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    return ExecutionRecord(
        execution_id="exec-steps-1",
        session_id="sess-1",
        query="q",
        mode="rlm",
        status="error",
        started_at=now,
        completed_at=now,
        steps=[{"role": "assistant", "content": "x"} for _ in range(_TRACE_ROWS)],
        result=result,
        chat_provider_id=cp_id,
    )


def test_live_trace_uses_the_runs_step_count_and_recorded_limit() -> None:
    state, cp = _state_with_cp()
    execution = _execution(
        cp.id, {"success": False, "steps_count": _RUN_STEPS, "steps_limit": _PROFILE_STEPS}
    )
    budget = _trace_from_execution_record(execution, state).budget

    assert budget.steps_used == _RUN_STEPS
    assert budget.steps_limit == _PROFILE_STEPS


def test_live_trace_without_a_recorded_limit_uses_the_chat_providers() -> None:
    state, cp = _state_with_cp()
    execution = _execution(cp.id, {"success": False})
    budget = _trace_from_execution_record(execution, state).budget

    assert budget.steps_used == _TRACE_ROWS  # older record: no step count either
    assert budget.steps_limit == _PROFILE_STEPS


def test_persisted_trace_uses_the_chat_providers_limit() -> None:
    state, cp = _state_with_cp()
    execution = _execution(cp.id, {"success": False})
    result = RunResultDTO(
        answer="stopped",
        mode_used="rlm",
        success=False,
        error="Budget exceeded",
        steps=_RUN_STEPS,
        trace=execution.steps,
    )
    _record_telemetry(state, execution, result, provider="openai", model="gpt-4o")
    detail = state.telemetry.get_run("exec-steps-1")
    assert detail is not None
    budget = _trace_from_telemetry(detail, state).budget

    assert budget.steps_used == _RUN_STEPS
    assert budget.steps_limit == _PROFILE_STEPS


def test_run_without_a_chat_provider_shows_the_global_limit() -> None:
    state = AppState(load_from_disk=False)
    state.config.budget.max_steps = _GLOBAL_STEPS
    execution = _execution(None, {"success": True, "steps_count": 2})
    budget = _trace_from_execution_record(execution, state).budget

    assert budget.steps_limit == _GLOBAL_STEPS
