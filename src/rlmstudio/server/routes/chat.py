"""Chat endpoints: POST /api/chat and WS /ws/chat/{session_id}."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from rlmstudio.application.dto import RunResultDTO
from rlmstudio.application.sandbox_vars import (
    EXTRA_KEY_HISTORY_MESSAGES,
    EXTRA_KEY_HISTORY_VARIABLE,
    HISTORY_PATH_DISABLED,
    HISTORY_PATH_EMPTY,
    HISTORY_PATH_INPROMPT,
    HISTORY_PATH_REPL_VARIABLE,
    MODE_AUTO,
    MODE_COMPARE,
    MODE_RAG,
    MODE_RLM,
    MODE_RLM_OFFICIAL,
    MODES_INPROMPT,
    MODES_REPL_VARIABLE,
    MODES_RLM_INTERNAL,
    TRACE_KEY_CODE,
    TRACE_KEY_CONTENT,
    TRACE_KEY_ELAPSED_SECONDS,
    TRACE_KEY_INPUT_TOKENS,
    TRACE_KEY_MODEL,
    TRACE_KEY_OUTPUT_TOKENS,
    TRACE_KEY_ROLE,
)
from rlmstudio.application.services.history_context import (
    assemble_inprompt_history_within_budget,
    build_history_variable,
    compute_history_cap_bytes,
    compute_inprompt_budget,
    extract_final_qa_pairs,
)
from rlmstudio.application.services.outcome_classifier import classify_execution_outcome
from rlmstudio.application.use_cases.run_direct import RunDirectUseCase
from rlmstudio.application.use_cases.run_rag import RunRAGUseCase
from rlmstudio.application.use_cases.run_rlm import RunRLMUseCase
from rlmstudio.application.use_cases.run_rlm_official import RunRLMOfficialUseCase
from rlmstudio.core.trace import ExecutionTrace
from rlmstudio.core.trace import TraceStep as CoreTraceStep
from rlmstudio.infrastructure.embedding.litellm_embedding_adapter import LiteLLMEmbeddingAdapter
from rlmstudio.infrastructure.storage.sqlite_adapter import SQLiteStorageAdapter
from rlmstudio.prompts import get_mode_system_prompt
from rlmstudio.server.dependencies import AppState, ExecutionRecord, get_state
from rlmstudio.server.models import (
    ChatRequest,
    ChatResponse,
)
from rlmstudio.server.models import (
    RunProfile as _RunProfile,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _resolve_profile_prompt(profile: _RunProfile, mode: str) -> str | None:
    """Resolve a profile's system prompt for the given mode.

    Prefer explicit per-mode text in ``system_prompts``.  If none is
    stored, fall back to the named ``prompt_template_name`` and look up
    the template's text for *mode* from the built-in registry.
    """
    # Explicit override takes priority
    custom: str | None = profile.system_prompts.get(mode)
    if custom:
        return custom

    # Named template reference — resolved at runtime so profiles follow
    # future edits to the template registry.
    if profile.prompt_template_name:
        from rlmstudio.ui.services.profile_store import SYSTEM_PROMPT_TEMPLATES

        tpl = SYSTEM_PROMPT_TEMPLATES.get(profile.prompt_template_name)
        if tpl:
            return tpl.get(mode) or None
    return None


def _prepare_history_context(
    *,
    state: AppState,
    session_id: str,
    chat_provider_id: str | None,
    cp: Any,  # ChatProviderConfig | None
    mode: str,
    adapter: Any,  # LiteLLMAdapter
    content: str,
    current_query: str,
    system_prompt_extra: str = "",
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Build conversation-history context for the current turn.

    Returns ``(full_query, extra_config_overlay, history_info)`` where:

    - ``full_query`` is the query string to pass to the use case (may
      have a "Previous conversation:" prefix prepended for in-prompt modes).
    - ``extra_config_overlay`` is a dict to merge into ``run_config.extra``
      (currently always empty; reserved for the REPL-variable path in Commit 5).
    - ``history_info`` is a diagnostics dict for telemetry / logging.
    """
    disabled_result: tuple[str, dict[str, Any], dict[str, Any]] = (
        current_query,
        {},
        {"path": HISTORY_PATH_DISABLED},
    )

    # The official engine receives only content + query — it has neither a
    # message list nor a ``history`` REPL variable to carry prior turns.
    if mode == MODE_RLM_OFFICIAL:
        return disabled_result

    if not chat_provider_id or cp is None:
        return disabled_result

    if not cp.conversation_memory_enabled:
        return disabled_result

    prev_msgs = state.get_conversation(session_id, chat_provider_id)
    turns = extract_final_qa_pairs(prev_msgs)

    if not turns:
        return (
            current_query,
            {},
            {
                "path": HISTORY_PATH_EMPTY,
                "conversation_memory_enabled": True,
                "turns_available": 0,
            },
        )

    if mode in MODES_INPROMPT:
        # --- In-prompt replay path ---
        context_window = getattr(adapter, "context_window", None) or 4096
        min_output = getattr(adapter, "min_output_tokens", None) or 128
        fraction = getattr(cp, "conversation_memory_fraction", 0.30)
        fraction_cap = int(context_window * fraction)

        # Estimate fixed token cost (system prompt + user message the use case
        # will construct) so the budget computation knows how much room is left.
        # For compare mode, both RunRLMUseCase and RunDirectUseCase receive the
        # same full_query, so the budget must be conservative against the *larger*
        # system prompt — the RLM one (which may also carry system_prompt_extra
        # from a profile).  Using the Direct prompt here would underestimate the
        # cost for the RLM half, risking context-window overflow on small-window
        # providers (the same Qwen/vLLM failure class the clamp was designed to
        # catch).
        budget_mode = MODE_RLM if mode == MODE_COMPARE else mode
        sys_prompt = get_mode_system_prompt(budget_mode)
        if system_prompt_extra:
            sys_prompt = sys_prompt + "\n\n" + system_prompt_extra

        # For RAG, the use case replaces the full document with retrieved
        # chunks (typically chunk_size * top_k chars).  Budget against
        # that estimate instead of the raw document, which would zero
        # out the history budget on large docs even though the actual
        # RAG prompt has plenty of room.
        if mode == MODE_RAG:
            rag_cfg = state.config.mode_config.rag_config
            estimated_context_chars = rag_cfg.chunk_size * rag_cfg.top_k + 200
            user_text = f"Context:\n{'x' * estimated_context_chars}\n\nQuestion: {current_query}"
        else:
            user_text = f"Content:\n{content}\n\nQuestion: {current_query}"
        try:
            fixed_tokens = adapter.count_tokens(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_text},
                ]
            )
        except Exception:
            # Fallback: rough char-based heuristic
            fixed_tokens = (len(sys_prompt) + len(user_text)) // 4 + 10

        budget = compute_inprompt_budget(
            system_prompt_tokens=fixed_tokens,
            current_query_tokens=0,
            reply_reserve=min_output,
            fraction_cap_tokens=fraction_cap,
            context_window=context_window,
        )

        def _token_counter(*, messages: list[dict[str, str]]) -> int:
            try:
                return int(adapter.count_tokens(messages=messages))
            except Exception:
                # Fallback: rough char-based heuristic
                return sum(len(m.get("content", "")) // 4 + 3 for m in messages)

        assembly = assemble_inprompt_history_within_budget(
            prev_turns=turns,
            budget_tokens=budget,
            token_counter=_token_counter,
        )

        # Pass prior turns as native chat messages via extra_overlay.
        # The use case prepends them before the current user message,
        # so the model sees proper user/assistant alternation instead
        # of a text prefix that small models tend to ignore.
        inprompt_overlay: dict[str, Any] = {}
        if assembly.messages:
            inprompt_overlay[EXTRA_KEY_HISTORY_MESSAGES] = assembly.messages

        # Compare mode runs BOTH RunDirectUseCase and RunRLMUseCase with
        # the same run_config.  Direct reads EXTRA_KEY_HISTORY_MESSAGES;
        # RLM reads EXTRA_KEY_HISTORY_VARIABLE (sandbox binding).  We
        # must populate both so neither half is blind to prior turns.
        # Use the same budgeted turn slice for both keys so the two
        # branches see identical history scope (apples-to-apples).
        if mode == MODE_COMPARE and assembly.turns_used > 0:
            budgeted_turns = turns[-assembly.turns_used :]
            history_var, _ = build_history_variable(
                prev_turns=budgeted_turns,
                cap_bytes=compute_history_cap_bytes(),
            )
            inprompt_overlay[EXTRA_KEY_HISTORY_VARIABLE] = history_var

        history_info: dict[str, Any] = {
            "path": HISTORY_PATH_INPROMPT,
            "mode": mode,
            "conversation_memory_enabled": True,
            "turns_available": assembly.turns_available,
            "history_turns_used": assembly.turns_used,
            "history_turns_dropped": assembly.turns_dropped,
            "history_tokens_used": assembly.tokens_used,
            "history_budget_tokens": budget,
            "context_window": context_window,
            "conversation_memory_fraction": fraction,
        }
        logger.info(
            "History [%s/%s]: path=%s, turns_used=%d/%d, budget=%d tok, messages=%d",
            mode,
            chat_provider_id,
            HISTORY_PATH_INPROMPT,
            assembly.turns_used,
            assembly.turns_available,
            budget,
            len(assembly.messages),
        )
        return (current_query, inprompt_overlay, history_info)

    if mode in MODES_REPL_VARIABLE:
        # REPL-variable path: bind a `history` Python list in the
        # sandbox rather than stuffing prior turns into the prompt.
        # Token cost is zero unless the model inspects `history`.
        history_var, var_info = build_history_variable(
            prev_turns=turns,
            cap_bytes=compute_history_cap_bytes(),
        )
        extra_overlay: dict[str, Any] = {
            EXTRA_KEY_HISTORY_VARIABLE: history_var,
        }
        history_info = {
            "path": HISTORY_PATH_REPL_VARIABLE,
            "mode": mode,
            "conversation_memory_enabled": True,
            "turns_available": len(turns),
            **var_info,
        }
        logger.info(
            "History [%s/%s]: path=%s, turns=%d, history_var_entries=%d",
            mode,
            chat_provider_id,
            HISTORY_PATH_REPL_VARIABLE,
            len(turns),
            len(history_var),
        )
        return (current_query, extra_overlay, history_info)

    # Unknown mode — pass through unchanged
    return disabled_result


# Per-message attachment limits
_MAX_FILES_PER_MESSAGE: int = 10
_MAX_TOTAL_CONTENT_BYTES: int = 50 * 1024 * 1024  # 50 MB

# RAG index cache: cache_key -> (SQLiteStorageAdapter, collection_name, embedding_model)
# cache_key = ":".join(sorted(file_ids))
# Avoids re-embedding the full document on every message in the same conversation.
_rag_index_cache: dict[str, tuple[SQLiteStorageAdapter, str, str]] = {}


def _render_multi_file_index(records: list[Any], offsets: list[tuple[int, int, int]]) -> str:
    """Render the multi-file document index with exact character offsets."""
    index_lines = [
        f"[DOCUMENT INDEX — {len(records)} files attached]",
        "Read this index first with peek(0, 1200).",
        "Each file entry includes exact character offsets within P:",
        "- Prefer outline_file(file_no=...) to get a document roadmap before reading large sections.",
        "- Use peek_file(file_no=..., start=..., end=...) for file-relative reading without offset math.",
        "- Use grep_file(file_no=..., pattern=...) for file-relative search; returned char_offset values work with peek_file().",
        "- content_start is still available for direct peek(start=content_start, end=...) jumps when needed.",
        "- If you cannot inspect every relevant file within budget, say which files you covered.",
    ]
    for i, (rec, (file_start, content_start, file_end_exclusive)) in enumerate(
        zip(records, offsets, strict=True), start=1
    ):
        index_lines.append(
            f'  {i}. "{rec.name}" '
            f"(file_start={file_start}, content_start={content_start}, "
            f"file_end_exclusive={file_end_exclusive})"
        )
    index_lines.append("[END DOCUMENT INDEX]")
    return "\n".join(index_lines)


def _resolve_file_content(file_ids: list[str], state: AppState) -> str:
    """Join text from multiple uploaded files into a single content string.

    Single file: returns plain content with a file header.
    Multiple files: prepends a Document Index with exact character offsets so
    RLM tools can jump directly to a file's content or use grep() follow-up
    peeks without guessing offsets.

    Enforces per-message file count and total byte limits.
    Raises HTTPException 400/404/413 on violations.
    """
    if len(file_ids) > _MAX_FILES_PER_MESSAGE:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files per message (max {_MAX_FILES_PER_MESSAGE})",
        )
    records = []
    cumulative_bytes = 0
    for fid in file_ids:
        rec = state.files.get(fid)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"File {fid} not found")
        cumulative_bytes += len(rec.text_content.encode())
        if cumulative_bytes > _MAX_TOTAL_CONTENT_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Combined file content exceeds {_MAX_TOTAL_CONTENT_BYTES // (1024 * 1024)} MB limit",
            )
        records.append(rec)

    separator = "\n\n---\n\n"

    if len(records) == 1:
        rec = records[0]
        result = f"[File: {rec.name}]\n\n{rec.text_content}"
    else:
        # Fixed-point calculation: offsets depend on index length, and the index
        # itself contains those offsets. Iterate until the rendered index is stable.
        section_headers = [f"[File {i + 1}: {rec.name}]\n\n" for i, rec in enumerate(records)]
        sections = [f"{section_headers[i]}{records[i].text_content}" for i in range(len(records))]
        offsets: list[tuple[int, int, int]] = [(0, 0, 0) for _ in records]

        for _ in range(8):
            index_block = _render_multi_file_index(records, offsets)
            cursor = len(index_block) + 2  # blank line between index and first section
            new_offsets: list[tuple[int, int, int]] = []
            for i in range(len(records)):
                section_text = sections[i]
                section_header = section_headers[i]
                file_start = cursor
                content_start = cursor + len(section_header)
                file_end_exclusive = cursor + len(section_text)
                new_offsets.append((file_start, content_start, file_end_exclusive))
                cursor = file_end_exclusive
                if i < len(records) - 1:
                    cursor += len(separator)

            if new_offsets == offsets:
                break
            offsets = new_offsets
        else:
            raise RuntimeError("Failed to stabilize multi-file document index offsets")

        result = index_block + "\n\n" + separator.join(sections)
    return result


# Canonical ExecutionTrace action types.
# Use-case traces store raw roles ("assistant"/"execution"); the rest of
# the system (JSONL export, ExecutionTrace schema, UI) expects the
# canonical set from rlmstudio.core.trace: inspect/subcall/final/error.
#
# `_canonical_action_type` lives in ``_helpers.py`` so the route helpers
# don't depend on this module (avoids the circular import that forced
# the earlier lazy import of ``_materialize_trace``). Re-exported here
# for the in-file callers (`_save_trajectory`, JSONL export path) and
# any external importers that already grep ``chat.py`` for this symbol.
from rlmstudio.server.routes._helpers import (  # noqa: E402
    _canonical_action_type,
    _materialize_trace,
)


def _save_trajectory(
    execution: ExecutionRecord,
    result: RunResultDTO,
    trace_dir: str,
) -> None:
    """Write execution trajectory to a JSONL file if trace_dir is configured."""
    try:
        os.makedirs(trace_dir, exist_ok=True)
        trace = ExecutionTrace()
        trace.metadata = {
            "execution_id": execution.execution_id,
            "session_id": execution.session_id,
            "query": execution.query,
            "mode": execution.mode,
        }
        total = len(result.trace)
        for i, step_data in enumerate(result.trace):
            action_type = _canonical_action_type(
                step_data.get(TRACE_KEY_ROLE),
                is_last=(i == total - 1),
                success=result.success,
            )
            trace.add_step(
                CoreTraceStep(
                    index=i,
                    action_type=action_type,  # type: ignore[arg-type]
                    code=step_data.get(TRACE_KEY_CODE),
                    output=step_data.get(TRACE_KEY_CONTENT, ""),
                    tokens_used=step_data.get(TRACE_KEY_INPUT_TOKENS, 0)
                    + step_data.get(TRACE_KEY_OUTPUT_TOKENS, 0),
                    duration=step_data.get(TRACE_KEY_ELAPSED_SECONDS, 0.0),
                    model=step_data.get(TRACE_KEY_MODEL),
                )
            )
        trace.finalize()
        filepath = os.path.join(trace_dir, f"{execution.execution_id}.jsonl")
        trace.to_jsonl(filepath)
        logger.info("Saved trajectory to %s", filepath)
    except Exception:
        logger.warning(
            "Failed to save trajectory for %s", execution.execution_id[:8], exc_info=True
        )


@router.post("/api/chat", status_code=202)
async def submit_chat(
    req: ChatRequest,
    state: AppState = Depends(get_state),  # noqa: B008
) -> ChatResponse:
    """Submit a chat query for execution."""
    # Canonicalise file IDs (model_validator already promotes file_id -> file_ids)
    effective_file_ids: list[str] = req.file_ids or []

    # Validate that either content or file(s) are provided
    if req.content is None and not effective_file_ids:
        raise HTTPException(status_code=400, detail="Either content or file_id(s) must be provided")

    # Resolve content
    content = req.content or ""
    if effective_file_ids:
        file_content = _resolve_file_content(effective_file_ids, state)
        content = (content + "\n\n" + file_content).strip() if content else file_content

    # Get or create session
    session = state.get_or_create_session(req.session_id)

    # Resolve execution mode from Chat Provider or legacy params
    chat_provider_id = req.chat_provider_id
    mode = req.mode
    if chat_provider_id:
        cp = state.get_chat_provider(chat_provider_id)
        if not cp:
            raise HTTPException(
                status_code=404, detail=f"Chat Provider {chat_provider_id} not found"
            )
        cp = state.resolve_chat_provider(cp)
        mode = cp.execution_mode

    if mode == MODE_RLM_OFFICIAL:
        engine_available, engine_reason, _ = state.rlm_engine_availability()
        if not engine_available:
            raise HTTPException(status_code=400, detail=engine_reason)

    # Create execution record
    exec_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    chat_provider_name = cp.name if (chat_provider_id and cp) else None
    execution = ExecutionRecord(
        execution_id=exec_id,
        session_id=session.id,
        query=req.query,
        mode=mode,
        started_at=now,
        chat_provider_id=chat_provider_id,
        chat_provider_name=chat_provider_name,
    )
    state.executions[exec_id] = execution

    # Add user message to session
    user_msg = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": req.query,
        "file_id": effective_file_ids[0] if effective_file_ids else None,  # compat alias
        "file_ids": effective_file_ids,
        "mode": mode,
        "chat_provider_id": chat_provider_id,
        "timestamp": now.isoformat(),
    }
    state.add_message(session.id, user_msg, chat_provider_id)
    session.updated_at = now
    # Save immediately so the session survives a server restart before execution completes
    state.save_sessions()

    # Execute in background
    asyncio.create_task(
        _run_execution(
            state,
            execution,
            content,
            req.query,
            mode,
            chat_provider_id,
            req.num_retries,
            file_ids=effective_file_ids,
        )
    )

    return ChatResponse(
        execution_id=exec_id,
        session_id=session.id,
        status="running",
        chat_provider_id=chat_provider_id,
    )


def _record_telemetry(
    state: AppState,
    execution: ExecutionRecord,
    result: RunResultDTO,
    *,
    provider: str,
    model: str,
) -> None:
    """Persist a completed execution to the telemetry store.

    Args:
        state: Application state providing the telemetry store.
        execution: The execution record being persisted.
        result: The final RunResultDTO from the use case.
        provider: LLM backend identifier (e.g. "openai", "anthropic", "vllm").
            This MUST be the structured backend key, not a user display name,
            so ``aggregate_by_provider()`` remains consistent across code paths.
        model: Model identifier (e.g. "gpt-4o", "claude-sonnet-4-6").
    """
    # Materialize the raw trace once and classify the outcome before
    # persisting. The classifier keys on TTFT/duration ratios, so it
    # needs the trace in memory here (dashboard reads are N+1 if we
    # defer classification to read time — see spec §8b).
    trace_obj = _materialize_trace(result.trace, run_success=result.success)
    outcome = classify_execution_outcome(
        success=result.success,
        error=result.error,
        answer=result.answer,
        trace=trace_obj,
    )

    run_id = state.telemetry.record_run(
        run_id=execution.execution_id,
        created_at=execution.started_at.timestamp() if execution.started_at else 0.0,
        mode=execution.mode,
        provider=provider,
        model=model,
        query=execution.query,
        content_length=0,
        answer=result.answer,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        total_tokens=result.total_tokens,
        total_cost=result.total_cost,
        elapsed_seconds=result.elapsed_time,
        success=result.success,
        error=result.error,
        session_id=execution.session_id,
        chat_provider_id=execution.chat_provider_id,
        chat_provider_name=execution.chat_provider_name,
        steps_count=result.steps,
        outcome_category=outcome.category.value,
    )
    total = len(result.trace)
    for i, step_data in enumerate(result.trace):
        action_type = _canonical_action_type(
            step_data.get(TRACE_KEY_ROLE),
            is_last=(i == total - 1),
            success=result.success,
        )
        input_tokens = step_data.get(TRACE_KEY_INPUT_TOKENS, 0)
        output_tokens = step_data.get(TRACE_KEY_OUTPUT_TOKENS, 0)
        state.telemetry.record_step(
            run_id=run_id,
            step_index=i,
            action_type=action_type,
            code=step_data.get(TRACE_KEY_CODE),
            output=step_data.get(TRACE_KEY_CONTENT),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration=step_data.get(TRACE_KEY_ELAPSED_SECONDS, 0.0),
            model=step_data.get(TRACE_KEY_MODEL),
            # Prefill/decode telemetry (spec v1.7). The four plain-string
            # keys come straight from the use-case raw-DTO writers; the
            # prompt/completion split is the raw token counts, mirroring
            # the route-layer translator in _helpers.
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            ttft_ms=step_data.get("ttft_ms"),
            decode_ms=int(step_data.get("decode_ms", 0) or 0),
            cached_tokens=int(step_data.get("cached_tokens", 0) or 0),
            cache_write_tokens=int(step_data.get("cache_write_tokens", 0) or 0),
        )


async def _run_execution(
    state: AppState,
    execution: ExecutionRecord,
    content: str,
    query: str,
    mode: str,
    chat_provider_id: str | None = None,
    num_retries: int | None = None,
    file_ids: list[str] | None = None,
) -> None:
    """Run the use case in the background and store results."""
    try:
        # Use Chat Provider-specific adapter if available, otherwise global.
        # telemetry_backend/telemetry_model are the structured backend key
        # and model id (preferred over the display label in telemetry records).
        if chat_provider_id:
            llm = state.create_llm_adapter_for_chat_provider(chat_provider_id, num_retries)
            cp = state.get_chat_provider(chat_provider_id)
            if cp:
                cp = state.resolve_chat_provider(cp)
            lp_http = (
                state.get_llm_provider(cp.llm_provider_id) if cp and cp.llm_provider_id else None
            )
            telemetry_backend = (
                (lp_http.backend if lp_http else None)
                or (cp.llm_provider if cp else None)
                or "unknown"
            )
            telemetry_model = (
                (lp_http.model if lp_http else None) or (cp.llm_model if cp else None) or ""
            )
            provider_label = f"{telemetry_backend}/{telemetry_model}"
        else:
            llm = state.create_llm_adapter(num_retries)
            cp = None
            telemetry_backend = state.config.active_provider
            telemetry_model = state.config.active_model
            provider_label = f"{telemetry_backend}/{telemetry_model}"
        logger.info(
            "Executing query [mode=%s, provider=%s, adapter=%s]: %.100s",
            mode,
            provider_label,
            type(llm).__name__,
            query,
        )

        # Build run config, using Chat Provider settings if available.
        # Apply RLM-specific knobs for rlm, auto, and compare (all run RLM internally).
        if cp and mode in MODES_RLM_INTERNAL:
            run_config = state.create_run_config(mode)
            run_config.max_steps = cp.rlm_max_steps
            run_config.max_time_seconds = float(cp.rlm_timeout_seconds)
            run_config.repeat_limit = cp.rlm_repeat_limit
            run_config.nudge_at_fraction = cp.rlm_nudge_at_fraction
            # Inject per-profile RLM system prompt (custom text or named template)
            if cp.profile_id:
                _prof = state.find_profile(cp.profile_id)
                if _prof:
                    _extra = _resolve_profile_prompt(_prof, MODE_RLM)
                    if _extra:
                        run_config.system_prompt_extra = _extra
                        logger.info(
                            "Profile '%s' (template=%s) → system_prompt_extra set (%d chars)",
                            _prof.name,
                            _prof.prompt_template_name or "(custom)",
                            len(_extra),
                        )
        else:
            run_config = state.create_run_config(mode)

        # Prepare conversation history via the budgeted helper
        full_query, extra_overlay, _history_info = _prepare_history_context(
            state=state,
            session_id=execution.session_id,
            chat_provider_id=chat_provider_id,
            cp=cp,
            mode=mode,
            adapter=llm,
            content=content,
            system_prompt_extra=getattr(run_config, "system_prompt_extra", "") or "",
            current_query=query,
        )
        if extra_overlay:
            run_config.extra.update(extra_overlay)

        if mode == MODE_COMPARE:
            # Run both RLM and Direct, store two assistant messages
            sandbox = state.create_sandbox()
            uc_rlm = RunRLMUseCase(llm, sandbox)
            uc_direct = RunDirectUseCase(llm)
            result_rlm = await asyncio.to_thread(uc_rlm.execute, content, full_query, run_config)
            result_direct = await asyncio.to_thread(
                uc_direct.execute, content, full_query, run_config
            )
            results = [result_rlm, result_direct]
        elif mode == MODE_RAG:
            rag_cfg = state.config.mode_config.rag_config
            # Resolve embedding API key: OpenAI key covers embeddings for any provider
            embedding_api_key: str | None = os.environ.get("OPENAI_API_KEY")
            if not embedding_api_key and cp:
                from rlmstudio.server.routes.llm_providers import _get_api_key

                lp = state.get_llm_provider(cp.llm_provider_id) if cp.llm_provider_id else None
                if lp and lp.backend == "openai":
                    embedding_api_key = _get_api_key(lp.id, lp.backend)

            # Re-use indexed storage for follow-up messages on the same file set.
            # This skips the expensive chunk+embed step (can take 30s+ for large docs).
            cache_key = ":".join(sorted(file_ids)) if file_ids else ""
            cached = _rag_index_cache.get(cache_key) if cache_key else None
            if cached and cached[2] == rag_cfg.embedding_model:
                storage, collection, _ = cached
                skip_indexing = True
                logger.info("RAG: reusing cached index for cache_key=%s", cache_key)
            else:
                storage = SQLiteStorageAdapter(":memory:")
                collection = f"rag_{uuid.uuid4().hex}"
                skip_indexing = False

            run_config.extra["collection"] = collection
            run_config.extra["chunk_size"] = rag_cfg.chunk_size
            run_config.extra["top_k"] = rag_cfg.top_k
            embedder = LiteLLMEmbeddingAdapter(
                model=rag_cfg.embedding_model,
                api_key=embedding_api_key,
            )
            uc_rag = RunRAGUseCase(llm, embedder, storage)
            results = [
                await asyncio.to_thread(
                    uc_rag.execute, content, full_query, run_config, skip_indexing
                )
            ]
            # Populate cache after successful first index
            if cache_key and not skip_indexing and results[0].success:
                _rag_index_cache[cache_key] = (storage, collection, rag_cfg.embedding_model)
        elif mode == MODE_RLM_OFFICIAL:
            engine = (
                state.create_rlm_engine_for_chat_provider(chat_provider_id, llm)
                if chat_provider_id
                else state.create_rlm_engine(llm)
            )
            uc_official = RunRLMOfficialUseCase(engine)
            results = [
                await asyncio.to_thread(uc_official.execute, content, full_query, run_config)
            ]
        elif mode in MODES_REPL_VARIABLE:
            sandbox = state.create_sandbox()
            uc = RunRLMUseCase(llm, sandbox)
            results = [await asyncio.to_thread(uc.execute, content, full_query, run_config)]
        else:
            uc_direct = RunDirectUseCase(llm)
            results = [await asyncio.to_thread(uc_direct.execute, content, full_query, run_config)]

        now = datetime.now(timezone.utc)
        # Use the first result for execution record status
        result = results[0]
        if result.success:
            logger.info(
                "Execution complete [exec=%s, tokens=%d]",
                execution.execution_id[:8],
                result.total_tokens,
            )
        else:
            logger.error("Execution failed [exec=%s]: %s", execution.execution_id[:8], result.error)
        execution.status = "complete" if result.success else "error"
        execution.completed_at = now
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
        execution.steps = result.trace

        # Persist to telemetry store
        _record_telemetry(
            state,
            execution,
            result,
            provider=telemetry_backend,
            model=telemetry_model,
        )

        # Save trajectory if configured
        if state.config.trajectory_dir:
            _save_trajectory(execution, result, state.config.trajectory_dir)

        # Add assistant message(s) to session
        session = state.sessions.get(execution.session_id)
        if session:
            for res in results:
                # Surface errors so the user can see what went wrong.
                # When the use case already produced a formatted ⚠️ warning
                # (timeout, budget, context overflow), use it as-is.  Only
                # prepend "Error:" when the answer is empty or raw text.
                answer_content = res.answer
                if not res.success and not answer_content:
                    error_detail = res.error or "Execution failed (no details available)"
                    answer_content = f"Error: {error_detail}"

                cp_name = cp.name if cp else None
                lp = (
                    state.get_llm_provider(cp.llm_provider_id)
                    if cp and cp.llm_provider_id
                    else None
                )
                provider_name = (
                    (lp.name if lp else None)
                    or (cp.llm_provider if cp else None)
                    or state.config.active_provider
                )

                assistant_msg = {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": answer_content,
                    "mode_used": res.mode_used,
                    "provider": provider_name,
                    "execution_id": execution.execution_id,
                    "chat_provider_id": chat_provider_id,
                    "chat_provider_name": cp_name,
                    "metrics": {
                        "input_tokens": res.input_tokens,
                        "output_tokens": res.output_tokens,
                        "total_tokens": res.total_tokens,
                        "cost_usd": res.total_cost,
                        "elapsed_seconds": res.elapsed_time,
                        "steps": res.steps,
                    },
                    "timestamp": now.isoformat(),
                }
                state.add_message(execution.session_id, assistant_msg, chat_provider_id)
            session.updated_at = now
            state.save_sessions()
            state.save_executions()

    except Exception as exc:
        logger.exception("Execution crashed [exec=%s]", execution.execution_id[:8])
        now = datetime.now(timezone.utc)
        execution.status = "error"
        execution.completed_at = now
        execution.result = {"answer": "", "success": False, "error": str(exc)}

        # Add error message to session so the user can see what went wrong
        session = state.sessions.get(execution.session_id)
        if session:
            elapsed = (now - execution.started_at).total_seconds() if execution.started_at else 0.0
            error_msg = {
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": f"Error: {exc}",
                "mode_used": execution.mode,
                "execution_id": execution.execution_id,
                "chat_provider_id": chat_provider_id,
                "timestamp": now.isoformat(),
                "metrics": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "elapsed_seconds": elapsed,
                    "steps": 0,
                },
            }
            state.add_message(execution.session_id, error_msg, chat_provider_id)
            session.updated_at = now
            state.save_sessions()
            state.save_executions()


class WebSocketEventEmitter:
    """Emits execution events to a WebSocket client."""

    def __init__(self, ws: WebSocket, msg_id: str) -> None:
        self._ws = ws
        self._id = msg_id

    async def on_token(self, token: str) -> None:
        await self._ws.send_json({"type": "token", "id": self._id, "data": token})

    async def on_step(self, step_data: dict[str, Any]) -> None:
        await self._ws.send_json({"type": "step", "id": self._id, "data": step_data})

    async def on_metrics(self, metrics: dict[str, Any]) -> None:
        await self._ws.send_json({"type": "metrics", "id": self._id, "data": metrics})


async def _ping_loop(ws: WebSocket) -> None:
    """Send periodic ping messages to detect stale connections."""
    while True:
        await asyncio.sleep(30)
        await ws.send_json({"type": "ping"})


@router.websocket("/ws/chat/{session_id}")
async def websocket_chat(
    websocket: WebSocket,
    session_id: str,
) -> None:
    """WebSocket endpoint for real-time chat streaming."""
    state = get_state()
    await websocket.accept()

    # Send connected message
    await websocket.send_json({"type": "connected", "session_id": session_id})

    # Track active query tasks for cancellation
    active_tasks: dict[str, asyncio.Task] = {}

    # Start heartbeat ping loop
    ping_task = asyncio.create_task(_ping_loop(websocket))
    try:
        while True:
            raw = await websocket.receive_text()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {
                        "type": "error",
                        "id": "",
                        "data": {
                            "code": "INVALID_JSON",
                            "message": "Malformed JSON message",
                            "recoverable": True,
                        },
                    }
                )
                continue

            msg_type = data.get("type")

            if msg_type == "pong":
                continue

            if msg_type == "cancel":
                task = active_tasks.pop(data.get("id", ""), None)
                if task:
                    task.cancel()
                continue

            if msg_type == "query":
                msg_id = data.get("id", str(uuid.uuid4()))
                query = data.get("query", "")
                content = data.get("content", "")
                ws_file_ids: list[str] = data.get("file_ids") or (
                    [data["file_id"]] if data.get("file_id") else []
                )
                mode = data.get("mode", MODE_AUTO)
                ws_chat_provider_id = data.get("chat_provider_id")

                if ws_file_ids:
                    try:
                        file_content = _resolve_file_content(ws_file_ids, state)
                        content = (
                            (content + "\n\n" + file_content).strip() if content else file_content
                        )
                    except HTTPException as exc:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "id": msg_id,
                                "data": {
                                    "code": str(exc.status_code),
                                    "message": exc.detail,
                                    "recoverable": True,
                                },
                            }
                        )
                        continue

                # Resolve mode from Chat Provider if provided
                ws_cp = None
                if ws_chat_provider_id:
                    ws_cp = state.get_chat_provider(ws_chat_provider_id)
                    if ws_cp:
                        ws_cp = state.resolve_chat_provider(ws_cp)
                    if not ws_cp:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "id": msg_id,
                                "data": {
                                    "code": "NOT_FOUND",
                                    "message": f"Chat Provider {ws_chat_provider_id} not found",
                                    "recoverable": True,
                                },
                            }
                        )
                        continue
                    mode = ws_cp.execution_mode

                if mode == MODE_RLM_OFFICIAL:
                    engine_available, engine_reason, _ = state.rlm_engine_availability()
                    if not engine_available:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "id": msg_id,
                                "data": {
                                    "code": "ENGINE_UNAVAILABLE",
                                    "message": engine_reason,
                                    "recoverable": True,
                                },
                            }
                        )
                        continue

                # Create execution record (mirrors REST path) so traces/dashboard work
                exec_id = str(uuid.uuid4())
                now = datetime.now(timezone.utc)
                session = state.get_or_create_session(session_id)
                ws_cp_name = ws_cp.name if (ws_chat_provider_id and ws_cp) else None
                execution = ExecutionRecord(
                    execution_id=exec_id,
                    session_id=session.id,
                    query=query,
                    mode=mode,
                    started_at=now,
                    chat_provider_id=ws_chat_provider_id,
                    chat_provider_name=ws_cp_name,
                )
                state.executions[exec_id] = execution

                # Add user message to session
                user_msg = {
                    "id": str(uuid.uuid4()),
                    "role": "user",
                    "content": query,
                    "file_id": ws_file_ids[0] if ws_file_ids else None,  # compat alias
                    "file_ids": ws_file_ids,
                    "mode": mode,
                    "chat_provider_id": ws_chat_provider_id,
                    "timestamp": now.isoformat(),
                }
                state.add_message(session.id, user_msg, ws_chat_provider_id)
                session.updated_at = now
                # Save immediately so the session survives a server restart before execution completes
                state.save_sessions()

                async def _ws_execute(
                    ws: WebSocket,
                    mid: str,
                    cnt: str,
                    q: str,
                    m: str,
                    exec_rec: ExecutionRecord,
                    sess: Any,
                    cp_id: str | None = None,
                ) -> None:
                    emitter = WebSocketEventEmitter(ws, mid)
                    try:
                        if cp_id:
                            llm = state.create_llm_adapter_for_chat_provider(cp_id)
                            ws_cp = state.get_chat_provider(cp_id)
                            if ws_cp:
                                ws_cp = state.resolve_chat_provider(ws_cp)
                            ws_lp = (
                                state.get_llm_provider(ws_cp.llm_provider_id)
                                if ws_cp and ws_cp.llm_provider_id
                                else None
                            )
                            provider_label = (
                                ws_lp.name
                                if ws_lp
                                else (ws_cp.llm_provider or "unknown")
                                if ws_cp
                                else "unknown"
                            )
                            # Structured backend/model for telemetry — must be
                            # the backend key (e.g. "openai"), not the display
                            # name, so aggregate_by_provider() stays consistent.
                            ws_telemetry_backend = (
                                (ws_lp.backend if ws_lp else None)
                                or (ws_cp.llm_provider if ws_cp else None)
                                or "unknown"
                            )
                            ws_telemetry_model = (
                                (ws_lp.model if ws_lp else None)
                                or (ws_cp.llm_model if ws_cp else None)
                                or ""
                            )
                        else:
                            llm = state.create_llm_adapter()
                            ws_cp = None
                            ws_lp = None
                            provider_label = (
                                f"{state.config.active_provider}/{state.config.active_model}"
                            )
                            ws_telemetry_backend = state.config.active_provider
                            ws_telemetry_model = state.config.active_model
                        logger.info(
                            "WS executing [mode=%s, provider=%s, adapter=%s]: %.100s",
                            m,
                            provider_label,
                            type(llm).__name__,
                            q,
                        )
                        cfg = state.create_run_config(m)
                        # Thread Chat Provider RLM settings into run config
                        if ws_cp and m in MODES_RLM_INTERNAL:
                            cfg.max_steps = ws_cp.rlm_max_steps
                            cfg.max_time_seconds = float(ws_cp.rlm_timeout_seconds)
                            cfg.repeat_limit = ws_cp.rlm_repeat_limit
                            cfg.nudge_at_fraction = ws_cp.rlm_nudge_at_fraction
                            # Inject per-profile RLM system prompt (custom or template)
                            if ws_cp.profile_id:
                                _prof = state.find_profile(ws_cp.profile_id)
                                if _prof:
                                    _extra = _resolve_profile_prompt(_prof, MODE_RLM)
                                    if _extra:
                                        cfg.system_prompt_extra = _extra
                                        logger.info(
                                            "WS profile '%s' (template=%s) → system_prompt_extra set (%d chars)",
                                            _prof.name,
                                            _prof.prompt_template_name or "(custom)",
                                            len(_extra),
                                        )

                        # Prepare conversation history
                        full_query, extra_overlay, _ws_history_info = _prepare_history_context(
                            state=state,
                            session_id=sess.id,
                            chat_provider_id=cp_id,
                            cp=ws_cp,
                            mode=m,
                            adapter=llm,
                            content=cnt,
                            current_query=q,
                            system_prompt_extra=getattr(cfg, "system_prompt_extra", "") or "",
                        )
                        if extra_overlay:
                            cfg.extra.update(extra_overlay)

                        if m == MODE_COMPARE:
                            sandbox = state.create_sandbox()
                            uc_rlm = RunRLMUseCase(llm, sandbox)
                            uc_direct = RunDirectUseCase(llm)
                            result_rlm = await uc_rlm.execute_async(
                                cnt, full_query, cfg, event_emitter=emitter
                            )
                            result_direct = await uc_direct.execute_async(
                                cnt, full_query, cfg, event_emitter=emitter
                            )
                            results = [result_rlm, result_direct]
                        elif m == MODE_RAG:
                            rag_cfg = state.config.mode_config.rag_config
                            emb_key: str | None = os.environ.get("OPENAI_API_KEY")
                            if not emb_key and ws_cp:
                                from rlmstudio.server.routes.llm_providers import _get_api_key

                                ws_lp_obj = (
                                    state.get_llm_provider(ws_cp.llm_provider_id)
                                    if ws_cp.llm_provider_id
                                    else None
                                )
                                if ws_lp_obj and ws_lp_obj.backend == "openai":
                                    emb_key = _get_api_key(ws_lp_obj.id, ws_lp_obj.backend)
                            ws_collection = f"rag_{uuid.uuid4().hex}"
                            cfg.extra["collection"] = ws_collection
                            cfg.extra["chunk_size"] = rag_cfg.chunk_size
                            cfg.extra["top_k"] = rag_cfg.top_k
                            ws_embedder = LiteLLMEmbeddingAdapter(
                                model=rag_cfg.embedding_model,
                                api_key=emb_key,
                            )
                            ws_storage = SQLiteStorageAdapter(":memory:")
                            uc_rag = RunRAGUseCase(llm, ws_embedder, ws_storage)
                            results = [
                                await asyncio.to_thread(uc_rag.execute, cnt, full_query, cfg)
                            ]
                        elif m == MODE_RLM_OFFICIAL:
                            engine = (
                                state.create_rlm_engine_for_chat_provider(cp_id, llm)
                                if cp_id
                                else state.create_rlm_engine(llm)
                            )
                            uc_official = RunRLMOfficialUseCase(engine)
                            results = [
                                await uc_official.execute_async(
                                    cnt, full_query, cfg, event_emitter=emitter
                                )
                            ]
                        elif m in MODES_REPL_VARIABLE:
                            sandbox = state.create_sandbox()
                            uc = RunRLMUseCase(llm, sandbox)
                            results = [
                                await uc.execute_async(cnt, full_query, cfg, event_emitter=emitter)
                            ]
                        else:
                            uc_d = RunDirectUseCase(llm)
                            results = [
                                await uc_d.execute_async(
                                    cnt, full_query, cfg, event_emitter=emitter
                                )
                            ]

                        finish = datetime.now(timezone.utc)
                        result = results[0]
                        exec_rec.status = "complete" if result.success else "error"
                        exec_rec.completed_at = finish
                        exec_rec.result = {
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
                        exec_rec.steps = result.trace

                        # Persist to telemetry store
                        _record_telemetry(
                            state,
                            exec_rec,
                            result,
                            provider=ws_telemetry_backend,
                            model=ws_telemetry_model,
                        )

                        for res in results:
                            answer = res.answer
                            if not res.success and not answer:
                                error_detail = (
                                    res.error or "Execution failed (no details available)"
                                )
                                answer = f"Error: {error_detail}"

                            cp_name = ws_cp.name if ws_cp else None
                            provider_name = (
                                (ws_lp.name if ws_lp else None)
                                or (ws_cp.llm_provider if ws_cp else None)
                                or state.config.active_provider
                            )

                            # Store in session for dashboard metrics
                            assistant_msg = {
                                "id": str(uuid.uuid4()),
                                "role": "assistant",
                                "content": answer,
                                "mode_used": res.mode_used,
                                "provider": provider_name,
                                "execution_id": exec_rec.execution_id,
                                "chat_provider_id": cp_id,
                                "chat_provider_name": cp_name,
                                "metrics": {
                                    "input_tokens": res.input_tokens,
                                    "output_tokens": res.output_tokens,
                                    "total_tokens": res.total_tokens,
                                    "cost_usd": res.total_cost,
                                    "elapsed_seconds": res.elapsed_time,
                                    "steps": res.steps,
                                },
                                "timestamp": finish.isoformat(),
                            }
                            state.add_message(sess.id, assistant_msg, cp_id)
                            sess.updated_at = finish

                            if not res.success:
                                await ws.send_json(
                                    {
                                        "type": "error",
                                        "id": mid,
                                        "data": {
                                            "code": "EXECUTION_ERROR",
                                            "message": res.error or "Execution failed",
                                            "mode": res.mode_used,
                                            "recoverable": True,
                                        },
                                    }
                                )
                            else:
                                await ws.send_json(
                                    {
                                        "type": "complete",
                                        "id": mid,
                                        "data": {
                                            "execution_id": exec_rec.execution_id,
                                            "mode": res.mode_used,
                                            "answer": res.answer,
                                            "success": res.success,
                                            "chat_provider_id": cp_id,
                                            "chat_provider_name": cp_name,
                                            "metrics": {
                                                "input_tokens": res.input_tokens,
                                                "output_tokens": res.output_tokens,
                                                "total_tokens": res.total_tokens,
                                                "cost_usd": res.total_cost,
                                                "elapsed_seconds": res.elapsed_time,
                                                "steps": res.steps,
                                            },
                                        },
                                    }
                                )
                        state.save_sessions()
                        state.save_executions()
                    except asyncio.CancelledError:
                        pass
                    except Exception as exc:
                        logger.exception("WebSocket execution error for %s", mid)
                        finish = datetime.now(timezone.utc)
                        exec_rec.status = "error"
                        exec_rec.completed_at = finish
                        exec_rec.result = {"answer": "", "success": False, "error": str(exc)}
                        # Persist error message to session (mirrors REST path)
                        if sess:
                            ws_elapsed = (
                                (finish - exec_rec.started_at).total_seconds()
                                if exec_rec.started_at
                                else 0.0
                            )
                            error_msg = {
                                "id": str(uuid.uuid4()),
                                "role": "assistant",
                                "content": f"Error: {exc}",
                                "mode_used": m,
                                "execution_id": exec_rec.execution_id,
                                "chat_provider_id": cp_id,
                                "timestamp": finish.isoformat(),
                                "metrics": {
                                    "input_tokens": 0,
                                    "output_tokens": 0,
                                    "total_tokens": 0,
                                    "cost_usd": 0.0,
                                    "elapsed_seconds": ws_elapsed,
                                    "steps": 0,
                                },
                            }
                            state.add_message(sess.id, error_msg, cp_id)
                            sess.updated_at = finish
                            state.save_sessions()
                            state.save_executions()
                        await ws.send_json(
                            {
                                "type": "error",
                                "id": mid,
                                "data": {
                                    "code": "INTERNAL_ERROR",
                                    "message": str(exc),
                                    "recoverable": False,
                                },
                            }
                        )
                    finally:
                        active_tasks.pop(mid, None)

                task = asyncio.create_task(
                    _ws_execute(
                        websocket,
                        msg_id,
                        content,
                        query,
                        mode,
                        execution,
                        session,
                        ws_chat_provider_id,
                    )
                )
                active_tasks[msg_id] = task

    except WebSocketDisconnect:
        pass
    finally:
        ping_task.cancel()
        for task in active_tasks.values():
            task.cancel()
