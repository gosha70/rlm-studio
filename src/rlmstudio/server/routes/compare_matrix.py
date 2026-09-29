"""POST /api/chat/compare-matrix — run a query across N chat providers × M modes.

Unlike the single-provider ``/api/chat`` endpoint, this route fans out
execution across an arbitrary list of ``(chat_provider_id × mode)`` slots,
runs them in parallel via :class:`RunMatrixComparisonUseCase`, records each
slot to the telemetry store under a shared ``comparison_group_id``, and
returns the full matrix so the client can render it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal, TypeAlias

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator

from rlmstudio.application.dto import RunConfigDTO
from rlmstudio.application.sandbox_vars import (
    EPHEMERAL_CP_PREFIX,
    MODE_COMPARE,
    MODE_RAG,
    MODE_RLM,
    RLM_DEFAULT_MAX_STEPS,
    RLM_DEFAULT_NUDGE_AT_FRACTION,
    RLM_DEFAULT_REPEAT_LIMIT,
    RLM_DEFAULT_WALL_CLOCK_BUDGET_SECONDS,
    ExecutionMode,
)
from rlmstudio.application.use_cases.run_matrix_comparison import (
    MatrixComparisonResultDTO,
    MatrixSlotDTO,
    MatrixSlotResultDTO,
    RunMatrixComparisonUseCase,
)
from rlmstudio.infrastructure.embedding.litellm_embedding_adapter import (
    LiteLLMEmbeddingAdapter,
)
from rlmstudio.infrastructure.storage.sqlite_adapter import SQLiteStorageAdapter
from rlmstudio.server.dependencies import AppState, ExecutionRecord, get_state
from rlmstudio.server.models import (
    BudgetConfig,
    ChatProviderConfig,
    RAGConfig,
    RuntimeSettings,
)

if TYPE_CHECKING:
    from rlmstudio.server.models import LLMProviderConfig

logger = logging.getLogger(__name__)

router = APIRouter()

SlotMode: TypeAlias = ExecutionMode

# Supported ranking metrics are the same set the use case exposes, restated
# here so the request schema can validate them at the HTTP boundary.
# Phase 5 widens the set with three prefill/decode-oriented metrics.
_RANKING_METRICS = (
    "cost",
    "tokens",
    "latency",
    "answer_per_cost",
    "judge_score",
    "ttft",
    "decode_tokens_per_sec",
    "cache_hit_rate",
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class CompareMatrixRequest(BaseModel):
    """Body for ``POST /api/chat/compare-matrix``."""

    content: str | None = None
    file_ids: list[str] | None = None
    query: str = Field(..., min_length=1)
    chat_provider_ids: list[str] = Field(..., min_length=1, max_length=10)
    modes: list[SlotMode] = Field(..., min_length=1, max_length=4)
    session_id: str | None = None
    ranking_metric: Literal[
        "cost",
        "tokens",
        "latency",
        "answer_per_cost",
        # ``judge_score`` is ranked client-side only (the frontend
        # re-ranks every response); the backend accepts it as a
        # passthrough so the request doesn't 422.
        "judge_score",
        "ttft",
        "decode_tokens_per_sec",
        "cache_hit_rate",
    ] = "cost"


class CompareMatrixRequestV2(BaseModel):
    """V2 request: selects LLM Providers + modes instead of Chat Providers."""

    content: str | None = None
    file_ids: list[str] | None = None
    query: str = Field(..., min_length=1)
    llm_provider_ids: list[str] = Field(..., min_length=1, max_length=10)
    modes: list[SlotMode] = Field(..., min_length=1, max_length=4)
    session_id: str | None = None
    ranking_metric: Literal[
        "cost",
        "tokens",
        "latency",
        "answer_per_cost",
        # ``judge_score`` is ranked client-side only (the frontend
        # re-ranks every response); the backend accepts it as a
        # passthrough so the request doesn't 422.
        "judge_score",
        "ttft",
        "decode_tokens_per_sec",
        "cache_hit_rate",
    ] = "cost"
    runtime_settings: RuntimeSettings = Field(default_factory=RuntimeSettings)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    rag_config: RAGConfig | None = None


class CompareMatrixUnifiedRequest(BaseModel):
    """Accepts either V1 (chat_provider_ids) or V2 (llm_provider_ids) shape."""

    content: str | None = None
    file_ids: list[str] | None = None
    query: str = Field(..., min_length=1)
    # V1 path
    chat_provider_ids: list[str] | None = None
    # V2 path
    llm_provider_ids: list[str] | None = None
    modes: list[SlotMode] = Field(default=["direct"], min_length=1, max_length=4)
    session_id: str | None = None
    ranking_metric: Literal[
        "cost",
        "tokens",
        "latency",
        "answer_per_cost",
        # ``judge_score`` is ranked client-side only (the frontend
        # re-ranks every response); the backend accepts it as a
        # passthrough so the request doesn't 422.
        "judge_score",
        "ttft",
        "decode_tokens_per_sec",
        "cache_hit_rate",
    ] = "cost"
    # V2-only fields
    runtime_settings: RuntimeSettings | None = None
    budget: BudgetConfig | None = None
    rag_config: RAGConfig | None = None

    @model_validator(mode="after")
    def _require_provider_ids(self) -> CompareMatrixUnifiedRequest:
        if not self.chat_provider_ids and not self.llm_provider_ids:
            msg = "Either chat_provider_ids (V1) or llm_provider_ids (V2) must be provided"
            raise ValueError(msg)
        return self


class CompareMatrixSlotResponse(BaseModel):
    slot_id: str
    label: str
    mode: str
    provider: str
    model: str
    chat_provider_id: str
    chat_provider_name: str | None = None
    execution_id: str
    success: bool
    answer: str
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    elapsed_seconds: float = 0.0
    steps: int = 0
    # Phase 5 — per-slot prefill/decode aggregates. Populated from the
    # slot's ExecutionTrace at write time so frontend ranking doesn't
    # need a second fetch. Defaults cover legacy slots and pre-Phase-3
    # traces without telemetry.
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0
    total_decode_ms: int = 0
    median_ttft_ms: int | None = None
    cache_hit_rate: float = 0.0


class CompareMatrixResponse(BaseModel):
    comparison_group_id: str
    session_id: str
    slots: list[CompareMatrixSlotResponse]
    ranking: list[int]
    ranking_metric: str
    total_elapsed: float


def _slot_perf_aggregates(raw_trace: list[dict] | None) -> dict:
    """Compute per-slot prefill/decode aggregates from the raw trace.

    Materializes the raw-DTO trace via the shared route helper and
    reuses the domain ExecutionTrace derived properties so this stays
    the single source of truth for totals / cache hit rate / median
    TTFT. Returns defaults when the trace is empty or carries no
    telemetry.
    """
    from rlmstudio.server.routes._helpers import _materialize_trace

    trace = _materialize_trace(raw_trace, run_success=True)
    if trace is None or not trace.steps:
        return {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cached_tokens": 0,
            "total_decode_ms": 0,
            "median_ttft_ms": None,
            "cache_hit_rate": 0.0,
        }
    return {
        "total_prompt_tokens": trace.total_prompt_tokens,
        "total_completion_tokens": trace.total_completion_tokens,
        "total_cached_tokens": sum(s.cached_tokens for s in trace.steps),
        "total_decode_ms": sum(s.decode_ms for s in trace.steps),
        "median_ttft_ms": trace.median_ttft_ms,
        "cache_hit_rate": trace.cache_hit_rate,
    }


# ---------------------------------------------------------------------------
# Shared metadata model + V2 ephemeral-CP helpers
# (all defined before the endpoint so mypy resolves them)
# ---------------------------------------------------------------------------


class _SlotMeta(BaseModel):
    """Per-slot metadata that is not part of the use-case layer."""

    chat_provider_id: str
    chat_provider_name: str | None = None
    execution_id: str


def _get_or_create_ephemeral_cp(
    state: AppState,
    lp: LLMProviderConfig,
    mode: str,
    runtime_settings: RuntimeSettings | None,
    budget: BudgetConfig | None,
    rag_config: RAGConfig | None,
) -> ChatProviderConfig:
    """Create an ephemeral CP for this (lp, mode) pair.

    Always returns a **fresh** ChatProviderConfig with a new ID so that
    concurrent compare-matrix requests never share a mutable CP object.
    To prevent unbounded growth, any previous ephemeral with the same
    name is removed from the provider list first.
    """
    now = datetime.now(timezone.utc)
    name = f"{EPHEMERAL_CP_PREFIX}{lp.name} \u00b7 {mode}"

    # Remove stale ephemeral with the same name (dedup / garbage-collect).
    state.config.chat_providers = [
        cp for cp in state.config.chat_providers if not (cp.ephemeral and cp.name == name)
    ]

    cp = ChatProviderConfig(
        id=str(uuid.uuid4()),
        name=name,
        llm_provider_id=lp.id,
        execution_mode=mode,  # type: ignore[arg-type]
        runtime_settings=runtime_settings or RuntimeSettings(),
        rlm_max_steps=budget.max_steps if budget else RLM_DEFAULT_MAX_STEPS,
        rlm_timeout_seconds=(
            budget.max_time_seconds if budget else RLM_DEFAULT_WALL_CLOCK_BUDGET_SECONDS
        ),
        rlm_repeat_limit=budget.repeat_limit if budget else RLM_DEFAULT_REPEAT_LIMIT,
        rlm_nudge_at_fraction=(
            budget.nudge_at_fraction if budget else RLM_DEFAULT_NUDGE_AT_FRACTION
        ),
        rag_config=rag_config,
        ephemeral=True,
        created_at=now,
        updated_at=now,
    )
    state.config.chat_providers.append(cp)
    return cp


async def _compare_matrix_v2(
    req: CompareMatrixUnifiedRequest,
    state: AppState,
) -> CompareMatrixResponse:
    """V2 execution path: LLM Providers × modes via auto-created ephemeral CPs."""
    assert req.llm_provider_ids is not None  # caller guarantees this
    logger.info(
        "compare-matrix V2: %d providers × %d modes, query=%r",
        len(req.llm_provider_ids),
        len(req.modes),
        req.query[:80],
    )

    # --- Validate all LLM providers are connected ---------------------
    lp_list: list[LLMProviderConfig] = []
    for lp_id in req.llm_provider_ids:
        lp = state.get_llm_provider(lp_id)
        if not lp:
            raise HTTPException(status_code=404, detail=f"LLM Provider {lp_id} not found")
        if lp.status not in ("connected", "configured"):
            raise HTTPException(
                status_code=400,
                detail=f"LLM Provider '{lp.name}' is not connected",
            )
        lp_list.append(lp)

    # --- Validate input content ---------------------------------------
    effective_file_ids: list[str] = req.file_ids or []
    if req.content is None and not effective_file_ids:
        raise HTTPException(
            status_code=400,
            detail="Either content or file_ids must be provided",
        )

    content = req.content or ""
    if effective_file_ids:
        from rlmstudio.server.routes.chat import _resolve_file_content

        file_content = _resolve_file_content(effective_file_ids, state)
        content = (content + "\n\n" + file_content).strip() if content else file_content

    # --- Resolve or create ephemeral CPs and collect their IDs -------
    cp_ids: list[str] = []
    for lp in lp_list:
        for mode in req.modes:
            ecp = _get_or_create_ephemeral_cp(
                state, lp, mode, req.runtime_settings, req.budget, req.rag_config
            )
            cp_ids.append(ecp.id)

    # --- Resolve session and append user message ----------------------
    session = state.get_or_create_session(req.session_id)
    started_at = datetime.now(timezone.utc)
    user_msg_id = str(uuid.uuid4())
    user_msg = {
        "id": user_msg_id,
        "role": "user",
        "content": req.query,
        "file_id": effective_file_ids[0] if effective_file_ids else None,
        "file_ids": effective_file_ids,
        "mode": MODE_COMPARE,
        "chat_provider_id": None,
        "timestamp": started_at.isoformat(),
    }
    session.messages.append(user_msg)
    for cp_id in dict.fromkeys(cp_ids):
        if cp_id not in session.conversations:
            session.conversations[cp_id] = []
        session.conversations[cp_id].append(user_msg)
    session.updated_at = started_at

    # --- Build slots (one per ephemeral CP; each CP already encodes one mode) ---
    total_slots = len(cp_ids)
    if total_slots > RunMatrixComparisonUseCase.MAX_SLOTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Too many slots: {total_slots} exceeds max of "
                f"{RunMatrixComparisonUseCase.MAX_SLOTS}"
            ),
        )

    slots: list[MatrixSlotDTO] = []
    slot_meta: dict[str, _SlotMeta] = {}

    # Iterate lp × mode in the same order as cp_ids above
    idx = 0
    for lp in lp_list:
        for mode in req.modes:
            cp_id = cp_ids[idx]
            idx += 1
            cp = state.get_chat_provider(cp_id)
            if cp is None:
                raise HTTPException(status_code=500, detail="Ephemeral CP disappeared")
            cp = state.resolve_chat_provider(cp)

            backend = lp.backend
            model_name = lp.model

            llm = state.create_llm_adapter_for_chat_provider(cp_id)
            sandbox = state.create_sandbox() if mode == MODE_RLM else None

            embedder = None
            storage = None
            slot_extra: dict[str, object] = {}
            if mode == MODE_RAG:
                embedding_api_key = _resolve_embedding_api_key(state, cp)
                rag_cfg_resolved = req.rag_config or state.config.mode_config.rag_config
                embedder = LiteLLMEmbeddingAdapter(
                    model=rag_cfg_resolved.embedding_model,
                    api_key=embedding_api_key,
                )
                storage = SQLiteStorageAdapter(":memory:")
                slot_extra = {
                    "collection": f"rag_{uuid.uuid4().hex}",
                    "chunk_size": rag_cfg_resolved.chunk_size,
                    "top_k": rag_cfg_resolved.top_k,
                }

            slot_config = _build_slot_run_config(state, cp, mode, slot_extra)

            slot_id = uuid.uuid4().hex[:12]
            execution_id = uuid.uuid4().hex
            slots.append(
                MatrixSlotDTO(
                    slot_id=slot_id,
                    mode=mode,
                    llm=llm,
                    sandbox=sandbox,
                    embedder=embedder,
                    storage=storage,
                    label=f"{lp.name} \u00b7 {mode}",
                    provider=backend,
                    model=model_name,
                    config=slot_config,
                )
            )
            execution = ExecutionRecord(
                execution_id=execution_id,
                session_id=session.id,
                query=req.query,
                mode=mode,
                status="running",
                started_at=started_at,
                chat_provider_id=cp_id,
                chat_provider_name=cp.name,
            )
            state.executions[execution_id] = execution
            slot_meta[slot_id] = _SlotMeta(
                chat_provider_id=cp_id,
                chat_provider_name=cp.name,
                execution_id=execution_id,
            )

    # --- Fan out -------------------------------------------------------
    uc = RunMatrixComparisonUseCase()
    result: MatrixComparisonResultDTO = await asyncio.to_thread(
        uc.execute,
        content,
        req.query,
        slots,
        None,
        req.ranking_metric,
    )

    # --- Finalize records and messages --------------------------------
    finish = datetime.now(timezone.utc)
    for slot_result in result.slots:
        meta = slot_meta[slot_result.slot_id]
        execution = state.executions[meta.execution_id]
        _update_execution_record(execution, slot_result, finish)
        _record_slot_telemetry(
            state=state,
            slot_result=slot_result,
            meta=meta,
            session_id=session.id,
            comparison_group_id=result.comparison_group_id,
            query=req.query,
            created_at=started_at,
        )
        answer_content = slot_result.result.answer
        if not slot_result.result.success and not answer_content:
            answer_content = f"Error: {slot_result.result.error or 'Execution failed'}"
        assistant_msg = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": answer_content,
            "mode_used": slot_result.mode,
            "provider": slot_result.provider,
            "execution_id": meta.execution_id,
            "chat_provider_id": meta.chat_provider_id,
            "chat_provider_name": meta.chat_provider_name,
            "metrics": {
                "input_tokens": slot_result.result.input_tokens,
                "output_tokens": slot_result.result.output_tokens,
                "total_tokens": slot_result.result.total_tokens,
                "cost_usd": slot_result.result.total_cost,
                "elapsed_seconds": slot_result.result.elapsed_time,
                "steps": slot_result.result.steps,
            },
            "timestamp": finish.isoformat(),
            "comparison_group_id": result.comparison_group_id,
        }
        state.add_message(session.id, assistant_msg, meta.chat_provider_id)

    session.updated_at = finish
    state.save_sessions()

    response_slots = [
        CompareMatrixSlotResponse(
            slot_id=s.slot_id,
            label=s.label,
            mode=s.mode,
            provider=s.provider,
            model=s.model,
            chat_provider_id=slot_meta[s.slot_id].chat_provider_id,
            chat_provider_name=slot_meta[s.slot_id].chat_provider_name,
            execution_id=slot_meta[s.slot_id].execution_id,
            success=s.result.success,
            answer=s.result.answer,
            error=s.result.error,
            input_tokens=s.result.input_tokens,
            output_tokens=s.result.output_tokens,
            total_tokens=s.result.total_tokens,
            total_cost=s.result.total_cost,
            elapsed_seconds=s.result.elapsed_time,
            steps=s.result.steps,
            **_slot_perf_aggregates(s.result.trace),
        )
        for s in result.slots
    ]

    return CompareMatrixResponse(
        comparison_group_id=result.comparison_group_id,
        session_id=session.id,
        slots=response_slots,
        ranking=result.ranking,
        ranking_metric=result.ranking_metric,
        total_elapsed=result.total_elapsed,
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/api/chat/compare-matrix")
async def compare_matrix(
    req: CompareMatrixUnifiedRequest,
    state: AppState = Depends(get_state),  # noqa: B008
) -> CompareMatrixResponse:
    """Run the same (content, query) across ``chat_provider_ids × modes``."""
    # --- Route to V2 or V1 based on which IDs are provided ------------
    if req.llm_provider_ids is not None:
        try:
            return await _compare_matrix_v2(req, state)
        except HTTPException:
            raise
        except Exception:
            logger.exception("compare-matrix V2 failed")
            raise
    if req.chat_provider_ids is None:
        raise HTTPException(
            status_code=400,
            detail="Either chat_provider_ids (V1) or llm_provider_ids (V2) must be provided",
        )

    # --- V1 path (unchanged) ------------------------------------------
    # --- Validate input content ---------------------------------------
    effective_file_ids: list[str] = req.file_ids or []
    if req.content is None and not effective_file_ids:
        raise HTTPException(
            status_code=400,
            detail="Either content or file_ids must be provided",
        )

    content = req.content or ""
    if effective_file_ids:
        from rlmstudio.server.routes.chat import _resolve_file_content

        file_content = _resolve_file_content(effective_file_ids, state)
        content = (content + "\n\n" + file_content).strip() if content else file_content

    # --- Resolve session and append the user message ------------------
    session = state.get_or_create_session(req.session_id)
    started_at = datetime.now(timezone.utc)
    user_msg_id = str(uuid.uuid4())
    user_msg = {
        "id": user_msg_id,
        "role": "user",
        "content": req.query,
        "file_id": effective_file_ids[0] if effective_file_ids else None,
        "file_ids": effective_file_ids,
        "mode": MODE_COMPARE,  # matrix mode marker for UI
        "chat_provider_id": None,  # fan-out across multiple providers
        "timestamp": started_at.isoformat(),
    }
    # Append once to the flat session.messages (legacy / compare views).
    session.messages.append(user_msg)
    # Also seed each affected chat provider's per-provider conversation so
    # that a follow-up /api/chat on any of those providers sees a balanced
    # user→assistant history.  Without this, get_conversation(cp_id) would
    # return only the matrix assistant answer with no preceding user turn.
    for cp_id in dict.fromkeys(req.chat_provider_ids):  # preserve order, dedupe
        if cp_id not in session.conversations:
            session.conversations[cp_id] = []
        session.conversations[cp_id].append(user_msg)
    session.updated_at = started_at

    # --- Cartesian product: chat_provider × mode ----------------------
    total_slots = len(req.chat_provider_ids) * len(req.modes)
    if total_slots > RunMatrixComparisonUseCase.MAX_SLOTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Too many slots: {total_slots} exceeds max of "
                f"{RunMatrixComparisonUseCase.MAX_SLOTS}"
            ),
        )

    slots: list[MatrixSlotDTO] = []
    slot_meta: dict[str, _SlotMeta] = {}

    for cp_id in req.chat_provider_ids:
        cp = state.get_chat_provider(cp_id)
        if cp is None:
            raise HTTPException(
                status_code=404,
                detail=f"Chat Provider {cp_id} not found",
            )
        cp = state.resolve_chat_provider(cp)

        lp = state.get_llm_provider(cp.llm_provider_id) if cp.llm_provider_id else None
        backend = (lp.backend if lp else None) or cp.llm_provider or "unknown"
        model_name = (lp.model if lp else None) or cp.llm_model or ""

        for mode in req.modes:
            # Each slot gets its own LLM adapter so one slot's streaming
            # state cannot bleed into another's.
            llm = state.create_llm_adapter_for_chat_provider(cp_id)
            sandbox = state.create_sandbox() if mode == MODE_RLM else None

            embedder = None
            storage = None
            slot_extra: dict[str, object] = {}
            if mode == MODE_RAG:
                embedding_api_key = _resolve_embedding_api_key(state, cp)
                rag_cfg = state.config.mode_config.rag_config
                embedder = LiteLLMEmbeddingAdapter(
                    model=rag_cfg.embedding_model,
                    api_key=embedding_api_key,
                )
                # Each slot gets its own in-memory store so collections
                # never collide across slots.
                storage = SQLiteStorageAdapter(":memory:")
                slot_extra = {
                    "collection": f"rag_{uuid.uuid4().hex}",
                    "chunk_size": rag_cfg.chunk_size,
                    "top_k": rag_cfg.top_k,
                }

            # Per-slot run config: start from the server default, then apply
            # the chat provider's profile-specific RLM knobs for rlm slots
            # (mirrors the single-chat /api/chat path).
            slot_config = _build_slot_run_config(state, cp, mode, slot_extra)

            slot_id = uuid.uuid4().hex[:12]
            execution_id = uuid.uuid4().hex
            slots.append(
                MatrixSlotDTO(
                    slot_id=slot_id,
                    mode=mode,
                    llm=llm,
                    sandbox=sandbox,
                    embedder=embedder,
                    storage=storage,
                    label=f"{cp.name} · {mode}",
                    provider=backend,
                    model=model_name,
                    config=slot_config,
                )
            )

            # Register an in-memory ExecutionRecord up front so the run is
            # visible to /api/executions and resolve_execution_context while
            # the matrix is in flight.  We'll fill in result/status after.
            execution = ExecutionRecord(
                execution_id=execution_id,
                session_id=session.id,
                query=req.query,
                mode=mode,
                status="running",
                started_at=started_at,
                chat_provider_id=cp_id,
                chat_provider_name=cp.name,
            )
            state.executions[execution_id] = execution

            slot_meta[slot_id] = _SlotMeta(
                chat_provider_id=cp_id,
                chat_provider_name=cp.name,
                execution_id=execution_id,
            )

    # --- Fan out via the use case (ThreadPoolExecutor under the hood) -
    uc = RunMatrixComparisonUseCase()
    result: MatrixComparisonResultDTO = await asyncio.to_thread(
        uc.execute,
        content,
        req.query,
        slots,
        None,  # each slot carries its own config
        req.ranking_metric,
    )

    # --- Finalize execution records, persist telemetry, add messages --
    finish = datetime.now(timezone.utc)
    for slot_result in result.slots:
        meta = slot_meta[slot_result.slot_id]
        execution = state.executions[meta.execution_id]

        # Update in-memory ExecutionRecord so judge/best-pick/quality see it.
        _update_execution_record(execution, slot_result, finish)

        # Persist to telemetry store.
        _record_slot_telemetry(
            state=state,
            slot_result=slot_result,
            meta=meta,
            session_id=session.id,
            comparison_group_id=result.comparison_group_id,
            query=req.query,
            created_at=started_at,
        )

        # Append assistant message per slot so session metrics / judge
        # flows can find them via state.add_message / session.messages.
        answer_content = slot_result.result.answer
        if not slot_result.result.success and not answer_content:
            answer_content = f"Error: {slot_result.result.error or 'Execution failed'}"
        assistant_msg = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": answer_content,
            "mode_used": slot_result.mode,
            "provider": slot_result.provider,
            "execution_id": meta.execution_id,
            "chat_provider_id": meta.chat_provider_id,
            "chat_provider_name": meta.chat_provider_name,
            "metrics": {
                "input_tokens": slot_result.result.input_tokens,
                "output_tokens": slot_result.result.output_tokens,
                "total_tokens": slot_result.result.total_tokens,
                "cost_usd": slot_result.result.total_cost,
                "elapsed_seconds": slot_result.result.elapsed_time,
                "steps": slot_result.result.steps,
            },
            "timestamp": finish.isoformat(),
            "comparison_group_id": result.comparison_group_id,
        }
        state.add_message(session.id, assistant_msg, meta.chat_provider_id)

    session.updated_at = finish
    state.save_sessions()

    # --- Build response -----------------------------------------------
    response_slots = [
        CompareMatrixSlotResponse(
            slot_id=s.slot_id,
            label=s.label,
            mode=s.mode,
            provider=s.provider,
            model=s.model,
            chat_provider_id=slot_meta[s.slot_id].chat_provider_id,
            chat_provider_name=slot_meta[s.slot_id].chat_provider_name,
            execution_id=slot_meta[s.slot_id].execution_id,
            success=s.result.success,
            answer=s.result.answer,
            error=s.result.error,
            input_tokens=s.result.input_tokens,
            output_tokens=s.result.output_tokens,
            total_tokens=s.result.total_tokens,
            total_cost=s.result.total_cost,
            elapsed_seconds=s.result.elapsed_time,
            steps=s.result.steps,
            **_slot_perf_aggregates(s.result.trace),
        )
        for s in result.slots
    ]

    return CompareMatrixResponse(
        comparison_group_id=result.comparison_group_id,
        session_id=session.id,
        slots=response_slots,
        ranking=result.ranking,
        ranking_metric=result.ranking_metric,
        total_elapsed=result.total_elapsed,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record_slot_telemetry(
    *,
    state: AppState,
    slot_result: MatrixSlotResultDTO,
    meta: _SlotMeta,
    session_id: str,
    comparison_group_id: str,
    query: str,
    created_at: datetime,
) -> None:
    """Persist one matrix slot as a telemetry ``runs`` row + steps."""
    try:
        result = slot_result.result

        # Classify per slot (not aggregate) so PREFILL_TIMEOUT stays
        # stable across dashboard reloads — see spec §8b.
        from rlmstudio.application.services.outcome_classifier import (
            classify_execution_outcome,
        )
        from rlmstudio.server.routes._helpers import _materialize_trace

        slot_trace = _materialize_trace(result.trace, run_success=result.success)
        outcome = classify_execution_outcome(
            success=result.success,
            error=result.error,
            answer=result.answer,
            trace=slot_trace,
        )

        state.telemetry.record_run(
            run_id=meta.execution_id,
            created_at=created_at.timestamp(),
            mode=slot_result.mode,
            provider=slot_result.provider,
            model=slot_result.model,
            query=query,
            answer=result.answer,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.total_tokens,
            total_cost=result.total_cost,
            elapsed_seconds=result.elapsed_time,
            success=result.success,
            error=result.error,
            session_id=session_id,
            chat_provider_id=meta.chat_provider_id,
            chat_provider_name=meta.chat_provider_name,
            comparison_group_id=comparison_group_id,
            steps_count=result.steps,
            outcome_category=outcome.category.value,
        )
        # Canonicalize action types to match in-memory traces / JSONL exports.
        from rlmstudio.server.routes._helpers import _canonical_action_type

        total = len(result.trace)
        for i, step_data in enumerate(result.trace):
            action_type = _canonical_action_type(
                step_data.get("role"),
                is_last=(i == total - 1),
                success=result.success,
            )
            input_tokens = step_data.get("input_tokens", 0)
            output_tokens = step_data.get("output_tokens", 0)
            state.telemetry.record_step(
                run_id=meta.execution_id,
                step_index=i,
                action_type=action_type,
                code=step_data.get("code"),
                output=step_data.get("content"),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration=step_data.get("elapsed_seconds", 0.0),
                model=step_data.get("model"),
                # Prefill/decode telemetry — mirrors the chat route.
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                ttft_ms=step_data.get("ttft_ms"),
                decode_ms=int(step_data.get("decode_ms", 0) or 0),
                cached_tokens=int(step_data.get("cached_tokens", 0) or 0),
                cache_write_tokens=int(step_data.get("cache_write_tokens", 0) or 0),
            )
    except Exception:
        # Telemetry failures must never break the user's matrix response.
        logger.exception("Failed to record telemetry for slot %s", slot_result.slot_id)


def _resolve_embedding_api_key(
    state: AppState,
    cp: ChatProviderConfig,
) -> str | None:
    """Resolve the embedding API key using the same fallback as ``/api/chat``.

    Preference order:
    1. ``OPENAI_API_KEY`` from the process environment.
    2. The stored secret for this chat provider's LLM provider instance,
       when that provider's backend is ``openai``.
    """
    api_key: str | None = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key

    lp = state.get_llm_provider(cp.llm_provider_id) if cp.llm_provider_id else None
    if lp and lp.backend == "openai":
        from rlmstudio.server.routes.llm_providers import _get_api_key

        api_key = _get_api_key(lp.id, lp.backend)

    return api_key


def _build_slot_run_config(
    state: AppState,
    cp: ChatProviderConfig,
    mode: str,
    extra: dict[str, object] | None = None,
) -> RunConfigDTO:
    """Build a per-slot :class:`RunConfigDTO` with the chat provider's profile
    settings applied (mirrors ``/api/chat`` for RLM) and optional RAG extras.
    """
    run_config = state.create_run_config(mode=mode)

    # Apply per-provider RLM knobs the same way /api/chat does.
    if mode == MODE_RLM:
        run_config.max_steps = cp.rlm_max_steps
        run_config.max_time_seconds = float(cp.rlm_timeout_seconds)
        run_config.repeat_limit = cp.rlm_repeat_limit
        run_config.nudge_at_fraction = cp.rlm_nudge_at_fraction
        if cp.profile_id:
            profile = state.find_profile(cp.profile_id)
            if profile:
                from rlmstudio.server.routes.chat import _resolve_profile_prompt

                prompt_extra = _resolve_profile_prompt(profile, MODE_RLM)
                if prompt_extra:
                    run_config.system_prompt_extra = prompt_extra

    if extra:
        run_config.extra.update(extra)

    return run_config


def _update_execution_record(
    execution: ExecutionRecord,
    slot_result: MatrixSlotResultDTO,
    completed_at: datetime,
) -> None:
    """Fill in the in-memory :class:`ExecutionRecord` from a slot result.

    Mirrors the shape of ``_run_execution`` in ``chat.py`` so downstream
    features (resolve_execution_context, get_execution_ids_for_turn,
    quality engine, judge flows) see a consistent record.
    """
    result = slot_result.result
    execution.status = "complete" if result.success else "error"
    execution.completed_at = completed_at
    execution.result = {
        "answer": result.answer,
        "success": result.success,
        "error": result.error,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "total_tokens": result.total_tokens,
        "total_cost": result.total_cost,
        "elapsed_time": result.elapsed_time,
        "steps_count": result.steps,
    }
    execution.steps = list(result.trace)
