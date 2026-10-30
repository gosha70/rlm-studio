"""Trace endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from rlmstudio.server.dependencies import AppState, ExecutionRecord, get_state
from rlmstudio.server.models import (
    TraceBudget,
    TraceResponse,
    TraceResult,
    TraceStep,
)
from rlmstudio.server.routes._helpers import _canonical_action_type
from rlmstudio.telemetry.store import RunDetail

router = APIRouter()


class ExecutionSummary(BaseModel):
    execution_id: str
    session_id: str
    query: str
    mode: str
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    total_tokens: int = 0
    total_cost: float = 0.0
    chat_provider_id: str | None = None
    chat_provider_name: str | None = None


@router.get("/api/executions")
async def list_executions(
    limit: int = Query(default=20, ge=1, le=100),
    chat_provider_id: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
    state: AppState = Depends(get_state),
) -> list[ExecutionSummary]:
    """List recent executions (newest first), optionally filtered."""
    from datetime import datetime, timezone

    # (started_at_epoch, summary) pairs so we can do one global sort below.
    candidates: list[tuple[float, ExecutionSummary]] = []
    seen_ids: set[str] = set()

    # In-memory executions (running or recently completed)
    for e in state.executions.values():
        if chat_provider_id is not None and e.chat_provider_id != chat_provider_id:
            continue
        if session_id is not None and e.session_id != session_id:
            continue
        rd = e.result or {}
        started_epoch = e.started_at.timestamp() if e.started_at else 0.0
        candidates.append(
            (
                started_epoch,
                ExecutionSummary(
                    execution_id=e.execution_id,
                    session_id=e.session_id,
                    query=e.query,
                    mode=e.mode,
                    status=e.status,
                    started_at=e.started_at.isoformat() if e.started_at else None,
                    completed_at=e.completed_at.isoformat() if e.completed_at else None,
                    total_tokens=rd.get("total_tokens", 0),
                    total_cost=rd.get("total_cost", 0.0),
                    chat_provider_id=e.chat_provider_id,
                    chat_provider_name=e.chat_provider_name,
                ),
            )
        )
        seen_ids.add(e.execution_id)

    # Completed executions from telemetry store (skip duplicates)
    # Fetch more than `limit` so the merged/sorted result still fills the page
    # when in-memory entries overlap.
    runs = state.telemetry.list_runs(
        limit=limit * 2,
        chat_provider_id=chat_provider_id,
        session_id=session_id,
    )
    for r in runs:
        if r.id in seen_ids:
            continue
        candidates.append(
            (
                r.created_at,
                ExecutionSummary(
                    execution_id=r.id,
                    session_id=r.session_id or "",
                    query=r.query,
                    mode=r.mode,
                    status="complete" if r.success else "error",
                    started_at=datetime.fromtimestamp(r.created_at, tz=timezone.utc).isoformat(),
                    total_tokens=r.total_tokens,
                    total_cost=r.total_cost,
                    chat_provider_id=r.chat_provider_id,
                    chat_provider_name=r.chat_provider_name,
                ),
            )
        )

    # Sort by started_at descending (newest first), then slice to limit.
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return [summary for _, summary in candidates[:limit]]


@router.get("/api/traces/{execution_id}")
async def get_trace(
    execution_id: str,
    state: AppState = Depends(get_state),
) -> TraceResponse:
    """Get execution trace with steps."""
    trace = load_canonical_trace(execution_id, state)
    if trace is None:
        raise HTTPException(status_code=404, detail="Execution not found")
    return trace


def load_canonical_trace(
    execution_id: str,
    state: AppState,
) -> TraceResponse | None:
    """Return a canonical TraceResponse for ``execution_id``, or ``None``.

    Shared loader used by ``/api/traces/{id}`` and
    ``/api/replays/{id}`` (V2b). Checks the in-memory live-execution
    dict first, then the persisted telemetry store. Both branches
    emit the canonical ``inspect | subcall | final | error``
    action-type enum so downstream consumers (the replay converter
    in particular) never have to guess which path the trace came
    from. See ``doc_internal/specs/learn-tab/NEXT.md`` §3e/§3f for
    the pinned canonicalization rule.
    """
    execution = state.executions.get(execution_id)
    if execution is not None:
        return _trace_from_execution_record(execution, state)
    detail = state.telemetry.get_run(execution_id)
    if detail is not None:
        return _trace_from_telemetry(detail, state)
    return None


@router.delete("/api/executions/{execution_id}", status_code=204)
async def delete_execution(
    execution_id: str,
    state: AppState = Depends(get_state),
) -> None:
    """Delete a single execution from telemetry and in-memory state."""
    state.executions.pop(execution_id, None)
    state.telemetry.delete_run(execution_id)


@router.delete("/api/executions", status_code=204)
async def delete_all_executions(
    state: AppState = Depends(get_state),
) -> None:
    """Delete all executions from telemetry and in-memory state."""
    state.executions.clear()
    state.telemetry.delete_all_runs()


def _trace_from_execution_record(execution: ExecutionRecord, state: AppState) -> TraceResponse:
    """Build TraceResponse from an in-memory ExecutionRecord.

    Raw step data stores the LLM-side "role" field
    (``"assistant"`` / ``"execution"``). The canonical
    ``ExecutionTrace`` action-type enum (``inspect | subcall |
    final | error``) is what telemetry / the replay converter /
    the JSONL export all expect, so this loader normalizes via
    :func:`_canonical_action_type` — the same helper the save
    path uses — before emitting TraceStep rows. Keeps the
    in-memory branch and the telemetry-backed branch on one
    action-type enum (pinned in NEXT.md §3e/§3f).
    """
    result_data = execution.result or {}
    success = bool(result_data.get("success", False))
    total = len(execution.steps)
    steps = []
    for i, step_data in enumerate(execution.steps):
        action_type = _canonical_action_type(
            step_data.get("role"),
            is_last=(i == total - 1),
            success=success,
        )
        input_tokens = step_data.get("input_tokens", 0)
        output_tokens = step_data.get("output_tokens", 0)
        steps.append(
            TraceStep(
                index=i,
                action_type=action_type,
                code=step_data.get("code"),
                output=step_data.get("content", ""),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_seconds=step_data.get("elapsed_seconds", 0.0),
                model=step_data.get("model"),
                prompt_tokens=step_data.get("prompt_tokens", input_tokens),
                completion_tokens=step_data.get("completion_tokens", output_tokens),
                ttft_ms=step_data.get("ttft_ms"),
                decode_ms=step_data.get("decode_ms", 0),
                cached_tokens=step_data.get("cached_tokens", 0),
                cache_write_tokens=step_data.get("cache_write_tokens", 0),
            )
        )
    return TraceResponse(
        execution_id=execution.execution_id,
        session_id=execution.session_id,
        query=execution.query,
        mode=execution.mode,
        status=execution.status,
        started_at=execution.started_at,
        completed_at=execution.completed_at,
        result=TraceResult(
            answer=result_data.get("answer", ""),
            success=result_data.get("success", False),
            error=result_data.get("error"),
            failure_kind=result_data.get("failure_kind"),
            input_tokens=result_data.get("input_tokens", 0),
            output_tokens=result_data.get("output_tokens", 0),
            total_cost=result_data.get("total_cost", 0.0),
        ),
        budget=TraceBudget(
            steps_used=len(execution.steps),
            steps_limit=state.config.budget.max_steps,
            tokens_used=result_data.get("total_tokens", 0),
            tokens_limit=state.config.budget.max_tokens,
            cost_used=result_data.get("total_cost", 0.0),
            cost_limit=state.config.budget.max_cost_usd,
        ),
        steps=steps,
        chat_provider_id=execution.chat_provider_id,
        chat_provider_name=execution.chat_provider_name,
    )


def _trace_from_telemetry(
    detail: RunDetail,
    state: AppState,
) -> TraceResponse:
    """Build TraceResponse from a TelemetryStore RunDetail."""
    from datetime import datetime, timezone

    steps = []
    for step in detail.steps:
        input_tokens = step.get("input_tokens", 0)
        output_tokens = step.get("output_tokens", 0)
        steps.append(
            TraceStep(
                index=step.get("step_index", 0),
                action_type=step.get("action_type", "inspect"),
                code=step.get("code"),
                output=step.get("output", ""),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_seconds=step.get("duration", 0.0),
                model=step.get("model"),
                prompt_tokens=step.get("prompt_tokens", input_tokens),
                completion_tokens=step.get("completion_tokens", output_tokens),
                ttft_ms=step.get("ttft_ms"),
                decode_ms=step.get("decode_ms", 0),
                cached_tokens=step.get("cached_tokens", 0),
                cache_write_tokens=step.get("cache_write_tokens", 0),
            )
        )

    started_at = datetime.fromtimestamp(detail.created_at, tz=timezone.utc)
    completed_at = datetime.fromtimestamp(
        detail.created_at + detail.elapsed_seconds, tz=timezone.utc
    )

    return TraceResponse(
        execution_id=detail.id,
        session_id=detail.session_id or "",
        query=detail.query,
        mode=detail.mode,
        status="complete" if detail.success else "error",
        started_at=started_at,
        completed_at=completed_at,
        result=TraceResult(
            answer=detail.answer,
            success=detail.success,
            error=detail.error,
            input_tokens=detail.input_tokens,
            output_tokens=detail.output_tokens,
            total_cost=detail.total_cost,
        ),
        budget=TraceBudget(
            steps_used=detail.steps_count,
            steps_limit=state.config.budget.max_steps,
            tokens_used=detail.total_tokens,
            tokens_limit=state.config.budget.max_tokens,
            cost_used=detail.total_cost,
            cost_limit=state.config.budget.max_cost_usd,
        ),
        steps=steps,
        chat_provider_id=detail.chat_provider_id,
        chat_provider_name=detail.chat_provider_name,
    )
