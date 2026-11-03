"""Per-step cost for aggregate subcall rows (#94 + #77 item 3)."""

from __future__ import annotations

from datetime import datetime, timezone

from rlmstudio.server.dependencies import AppState, ExecutionRecord
from rlmstudio.server.routes.traces import _trace_from_execution_record


def _execution(*, model: str) -> ExecutionRecord:
    return ExecutionRecord(
        execution_id="exec-cost-1",
        session_id="sess-1",
        query="batch them",
        mode="rlm",
        status="complete",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        steps=[
            {
                "role": "assistant",
                "input_tokens": 10,
                "output_tokens": 5,
                "model": model,
            },
            {
                "role": "execution",
                "content": '["answer child-0", "answer child-1"]',
                "input_tokens": 40,
                "output_tokens": 20,
                "model": model,
                # Aggregate child cost, zero by ignorance for an unpriced model.
                "cost": 0.0,
            },
            {
                "role": "assistant",
                "content": '{"type": "final", "answer": "done"}',
                "input_tokens": 10,
                "output_tokens": 5,
                "model": model,
            },
        ],
        result={"success": True, "answer": "done"},
    )


def test_unpriced_subcall_tokens_render_as_unavailable() -> None:
    state = AppState(load_from_disk=False)
    trace = _trace_from_execution_record(
        _execution(model="hosted_vllm/some-unpriced-local-model"), state
    )
    subcall = next(step for step in trace.steps if step.action_type == "subcall")
    assert subcall.input_tokens == 40
    assert subcall.output_tokens == 20
    assert subcall.cost_usd is None


def test_priced_subcall_tokens_render_a_known_cost() -> None:
    state = AppState(load_from_disk=False)
    trace = _trace_from_execution_record(_execution(model="gpt-4o"), state)
    subcall = next(step for step in trace.steps if step.action_type == "subcall")
    assert subcall.cost_usd is not None
    assert subcall.cost_usd > 0.0
