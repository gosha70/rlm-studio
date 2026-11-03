"""Per-step cost for aggregate subcall rows (#94 + #77 item 3)."""

from __future__ import annotations

from datetime import datetime, timezone

from rlmstudio.application.dto import RunResultDTO
from rlmstudio.server.dependencies import AppState, ExecutionRecord
from rlmstudio.server.routes.chat import _record_telemetry
from rlmstudio.server.routes.traces import (
    _trace_from_execution_record,
    _trace_from_telemetry,
)


def _step(*, model: str) -> dict:
    return {
        "role": "assistant",
        "input_tokens": 10,
        "output_tokens": 5,
        "model": model,
    }


def _trace_steps(
    *,
    model: str,
    cost_known: bool | None,
    cost: float | None,
) -> list[dict]:
    subcall: dict = {
        "role": "execution",
        "content": '["answer child-0", "answer child-1"]',
        "input_tokens": 40,
        "output_tokens": 20,
        "model": model,
    }
    if cost_known is not None:
        subcall["cost_known"] = cost_known
    if cost is not None:
        subcall["cost"] = cost
    return [
        _step(model=model),
        subcall,
        {
            "role": "assistant",
            "content": '{"type": "final", "answer": "done"}',
            "input_tokens": 10,
            "output_tokens": 5,
            "model": model,
        },
    ]


def _execution(*, model: str, cost_known: bool | None, cost: float | None) -> ExecutionRecord:
    return ExecutionRecord(
        execution_id="exec-cost-1",
        session_id="sess-1",
        query="batch them",
        mode="rlm",
        status="complete",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        steps=_trace_steps(model=model, cost_known=cost_known, cost=cost),
        result={"success": True, "answer": "done"},
    )


def test_unpriced_subcall_tokens_render_as_unavailable() -> None:
    state = AppState(load_from_disk=False)
    trace = _trace_from_execution_record(
        _execution(
            model="hosted_vllm/some-unpriced-local-model",
            cost_known=False,
            cost=None,
        ),
        state,
    )
    subcall = next(step for step in trace.steps if step.action_type == "subcall")
    assert subcall.input_tokens == 40
    assert subcall.output_tokens == 20
    assert subcall.cost_usd is None


def test_known_aggregate_cost_is_used_instead_of_recomputed() -> None:
    state = AppState(load_from_disk=False)
    trace = _trace_from_execution_record(
        _execution(model="gpt-4o", cost_known=True, cost=0.000123),
        state,
    )
    subcall = next(step for step in trace.steps if step.action_type == "subcall")
    assert subcall.cost_usd == 0.000123


def test_known_zero_for_unlisted_model_remains_zero() -> None:
    state = AppState(load_from_disk=False)
    trace = _trace_from_execution_record(
        _execution(
            model="hosted_vllm/custom-provider-reported-zero",
            cost_known=True,
            cost=0.0,
        ),
        state,
    )
    subcall = next(step for step in trace.steps if step.action_type == "subcall")
    assert subcall.cost_usd == 0.0


def _record(model: str, *, cost_known: bool, cost: float) -> AppState:
    state = AppState(load_from_disk=False)
    execution = _execution(model=model, cost_known=cost_known, cost=cost)
    result = RunResultDTO(
        answer="done",
        mode_used="rlm",
        success=True,
        steps=3,
        input_tokens=60,
        output_tokens=30,
        total_cost=cost,
        elapsed_time=0.2,
        trace=execution.steps,
    )
    _record_telemetry(state, execution, result, provider="openai", model=model)
    return state


def test_telemetry_backed_known_zero_remains_zero() -> None:
    state = _record("hosted_vllm/custom-provider-reported-zero", cost_known=True, cost=0.0)
    detail = state.telemetry.get_run("exec-cost-1")
    assert detail is not None
    trace = _trace_from_telemetry(detail, state)
    subcall = next(step for step in trace.steps if step.action_type == "subcall")
    assert subcall.cost_usd == 0.0


def test_telemetry_backed_unknown_cost_stays_unavailable() -> None:
    state = _record("hosted_vllm/some-unpriced-local-model", cost_known=False, cost=0.0)
    detail = state.telemetry.get_run("exec-cost-1")
    assert detail is not None
    trace = _trace_from_telemetry(detail, state)
    subcall = next(step for step in trace.steps if step.action_type == "subcall")
    assert subcall.cost_usd is None
