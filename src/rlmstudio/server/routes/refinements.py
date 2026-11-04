"""Prompt refinement routes (#78): source runs, proposals, evaluation and history.

Nothing here writes to a profile: a proposal is stored as a ``refinements``
row in ``proposal_ready``, an evaluation moves it through ``evaluating`` to
``evaluated`` (or ``failed``), and an explicit decision comes later (phase D).
See ``specs/prompt-refinement/spec.md``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from rlmstudio.application.refinement_constants import (
    REFINEMENT_STATUS_EVALUATING,
    REFINEMENT_STATUS_FAILED,
    REFINEMENT_STATUS_PROPOSAL_READY,
)
from rlmstudio.application.sandbox_vars import MODE_RLM
from rlmstudio.application.services.outcome_classifier import is_degraded_answer
from rlmstudio.application.services.provider_failure import redact_secrets
from rlmstudio.application.services.refinement import (
    BaseSupplementState,
    EvidenceRun,
    ProposalError,
    ProposalRecord,
    build_evidence_digest,
    propose_supplement,
    supplement_sha256,
)
from rlmstudio.prompts import (
    get_default_system_prompt,
    get_refine_supplement_prompt,
    get_system_prompt_version,
)
from rlmstudio.server import refinement_decisions
from rlmstudio.server.dependencies import AppState, get_state
from rlmstudio.server.models import (
    HoldoutCaseRow,
    RefinementAcceptRequest,
    RefinementCreateRequest,
    RefinementEvaluateRequest,
    RefinementEvaluateResponse,
    RefinementRejectRequest,
    RefinementResponse,
    RefinementSourceRun,
    RunProfile,
)
from rlmstudio.server.refinement_eval import (
    DatasetUnavailableError,
    EvaluationRequest,
    describe_cases,
    load_evaluation_dataset,
    select_cases,
    start_evaluation,
)
from rlmstudio.telemetry.store import (
    RefinementConflictError,
    RefinementRecord,
    RefinementStoreError,
    RunDetail,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_REFINEMENTS_PATH = "/api/profiles/{profile_id}/refinements"
SOURCE_RUNS_DEFAULT_LIMIT = 50
SOURCE_RUNS_MAX_LIMIT = 200
JSONL_MEDIA_TYPE = "application/x-ndjson"


def _get_profile(state: AppState, profile_id: str) -> RunProfile:
    profile = state.find_profile(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")
    return profile


def _to_response(state: AppState, record: RefinementRecord) -> RefinementResponse:
    return RefinementResponse(**asdict(record), progress=state.get_refinement_progress(record.id))


def _get_refinement(state: AppState, profile_id: str, refinement_id: str) -> RefinementRecord:
    record = state.telemetry.get_refinement(refinement_id)
    if record is None or record.profile_id != profile_id:
        raise HTTPException(status_code=404, detail=f"Refinement not found: {refinement_id}")
    return record


def _evidence_run(run: RunDetail) -> EvidenceRun:
    return EvidenceRun(
        id=run.id,
        created_at=run.created_at,
        mode=run.mode,
        success=run.success,
        outcome_category=run.outcome_category,
        query=run.query,
        answer=run.answer,
        error=run.error,
        steps_count=run.steps_count,
        action_types=tuple(str(step.get("action_type", "")) for step in run.steps),
        total_tokens=run.total_tokens,
        total_cost=run.total_cost,
        elapsed_seconds=run.elapsed_seconds,
    )


def _load_source_runs(state: AppState, profile_id: str, run_ids: list[str]) -> list[RunDetail]:
    """Return the named runs, or 400 listing every id that cannot be evidence."""
    runs: list[RunDetail] = []
    problems: list[str] = []
    for run_id in run_ids:
        run = state.telemetry.get_run(run_id)
        if run is None:
            problems.append(f"{run_id}: not found")
        elif run.profile_id != profile_id:
            problems.append(f"{run_id}: not a run of this profile")
        elif run.mode != MODE_RLM:
            problems.append(f"{run_id}: not an RLM run ({run.mode})")
        elif not run.content_sha256:
            problems.append(f"{run_id}: recorded before document hashing")
        else:
            runs.append(run)
    if problems:
        raise HTTPException(
            status_code=400, detail="Source runs cannot be used: " + "; ".join(problems)
        )
    return runs


def _base_state(profile: RunProfile) -> BaseSupplementState:
    # Same resolution the RLM loop uses: custom text wins, else the template's.
    from rlmstudio.server.routes.chat import _resolve_profile_prompt

    resolved = _resolve_profile_prompt(profile, MODE_RLM) or ""
    base_prompt = get_default_system_prompt()
    return BaseSupplementState(
        resolved=resolved,
        sha256=supplement_sha256(resolved),
        rlm_key_present=MODE_RLM in profile.system_prompts,
        rlm_key_value=profile.system_prompts.get(MODE_RLM),
        template_name=profile.prompt_template_name,
        base_prompt_version=get_system_prompt_version(),
        base_prompt_sha256=hashlib.sha256(base_prompt.encode("utf-8")).hexdigest(),
    )


@router.get(f"{_REFINEMENTS_PATH}/source-runs")
async def list_source_runs(
    profile_id: str,
    limit: int = Query(SOURCE_RUNS_DEFAULT_LIMIT, ge=1, le=SOURCE_RUNS_MAX_LIMIT),
    state: AppState = Depends(get_state),
) -> list[RefinementSourceRun]:
    """RLM runs of this profile that can be refinement evidence, newest first (FR-2).

    Only ``rlm`` runs are listed: the supplement changes nothing else, and
    an ``auto`` or ``compare`` run may not have run the RLM loop at all.
    Runs recorded before document hashing are never listed.
    """
    _get_profile(state, profile_id)
    runs = state.telemetry.list_runs(
        limit=limit, mode=MODE_RLM, profile_id=profile_id, require_content_hash=True
    )
    return [
        RefinementSourceRun(
            id=run.id,
            created_at=run.created_at,
            mode=run.mode,
            query=run.query,
            success=run.success,
            outcome_category=run.outcome_category,
            degraded=is_degraded_answer(run.success, run.answer),
            chat_provider_id=run.chat_provider_id,
            chat_provider_name=run.chat_provider_name,
            total_cost=run.total_cost,
            total_tokens=run.total_tokens,
            steps_count=run.steps_count,
            content_sha256=run.content_sha256,
        )
        for run in runs
        if run.content_sha256
    ]


@router.get(f"{_REFINEMENTS_PATH}/holdout-cases")
async def list_holdout_cases(
    profile_id: str,
    state: AppState = Depends(get_state),
) -> list[HoldoutCaseRow]:
    """Every evaluation dataset case, marked holdout / deterministic / runnable (FR-16)."""
    _get_profile(state, profile_id)
    try:
        dataset, path = load_evaluation_dataset()
    except DatasetUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    infos = await asyncio.to_thread(describe_cases, dataset, path)
    return [HoldoutCaseRow(**asdict(info)) for info in infos]


@router.get(f"{_REFINEMENTS_PATH}/export.jsonl")
async def export_refinements(
    profile_id: str,
    state: AppState = Depends(get_state),
) -> Response:
    """One JSON line per refinement of the profile, records verbatim (FR-15).

    Works for a deleted profile too: its refinement rows stay in the
    database and remain exportable (spec §6); 404 only when there is
    neither a profile nor any refinement for the id.
    """
    records = state.telemetry.list_refinements(profile_id)
    if not records:
        _get_profile(state, profile_id)
    lines = [json.dumps(asdict(record), sort_keys=True) for record in records]
    body = "".join(f"{line}\n" for line in lines)
    return Response(
        content=body,
        media_type=JSONL_MEDIA_TYPE,
        headers={"Content-Disposition": f'attachment; filename="refinements-{profile_id}.jsonl"'},
    )


@router.post(_REFINEMENTS_PATH, status_code=201)
async def create_refinement(
    profile_id: str,
    req: RefinementCreateRequest,
    state: AppState = Depends(get_state),
) -> RefinementResponse:
    """Propose a candidate RLM supplement from selected runs (FR-3, FR-4).

    One model call through the chosen Chat Provider.  The profile is not
    written.  A provider failure or an unusable reply returns 502 and
    stores nothing.
    """
    profile = _get_profile(state, profile_id)
    if profile.is_builtin:
        raise HTTPException(status_code=400, detail="Built-in profiles cannot be refined")

    runs = _load_source_runs(state, profile_id, list(dict.fromkeys(req.source_run_ids)))
    # Newest first, the same order the evidence digest uses.
    runs.sort(key=lambda r: r.created_at, reverse=True)
    newest = runs[0]
    cp_id = req.proposal_chat_provider_id or newest.chat_provider_id
    if not cp_id:
        raise HTTPException(
            status_code=400,
            detail="No proposal Chat Provider given and the newest source run has none",
        )
    cp = state.get_chat_provider(cp_id)
    if cp is None:
        raise HTTPException(status_code=400, detail=f"Chat Provider not found: {cp_id}")
    lp = state.get_llm_provider(cp.llm_provider_id) if cp.llm_provider_id else None
    provider = (lp.backend if lp else None) or cp.llm_provider or "unknown"
    configured_model = (lp.model if lp else None) or cp.llm_model or ""

    base = _base_state(profile)
    digest = build_evidence_digest([_evidence_run(run) for run in runs])
    template, prompt_version = get_refine_supplement_prompt()
    try:
        llm = state.create_llm_adapter_for_chat_provider(cp_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    started = time.monotonic()
    try:
        result = await asyncio.to_thread(
            propose_supplement,
            llm,
            template=template,
            prompt_version=prompt_version,
            digest=digest,
            current_supplement=base.resolved,
            # The same text whose sha256 the proposal record stores.
            base_prompt=get_default_system_prompt(),
            base_prompt_version=base.base_prompt_version,
        )
    except ProposalError as exc:
        raise HTTPException(
            status_code=502, detail=f"The proposal model returned an unusable reply: {exc}"
        ) from exc
    except Exception as exc:
        # Provider failures arrive as many litellm exception types; the
        # message is redacted because upstream errors can echo the key.
        logger.warning("Refinement proposal call failed for profile %s: %s", profile_id, exc)
        raise HTTPException(
            status_code=502, detail=f"Proposal call failed: {redact_secrets(str(exc))}"
        ) from exc
    logger.info(
        "Refinement proposal for profile %s via %s/%s: %d input / %d output tokens, %.1fs",
        profile_id,
        provider,
        result.model or configured_model,
        result.input_tokens,
        result.output_tokens,
        time.monotonic() - started,
    )

    record = ProposalRecord(
        source_run_ids=[run.id for run in runs],
        source_content_sha256={run.id: run.content_sha256 or "" for run in runs},
        evidence=digest,
        proposal_chat_provider_id=cp_id,
        provider=provider,
        model=result.model or configured_model,
        prompt_version=result.prompt_version,
        prompt_sha256=result.prompt_sha256,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        base=base,
        candidate=result.candidate,
        rationale=result.rationale,
        diff=result.diff,
    )
    try:
        stored = state.telemetry.create_refinement(
            profile_id=profile_id,
            status=REFINEMENT_STATUS_PROPOSAL_READY,
            proposal=record.to_dict(),
        )
    except RefinementStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _to_response(state, stored)


@router.get(_REFINEMENTS_PATH)
async def list_refinements(
    profile_id: str,
    state: AppState = Depends(get_state),
) -> list[RefinementResponse]:
    """A profile's refinements, newest first."""
    _get_profile(state, profile_id)
    return [_to_response(state, r) for r in state.telemetry.list_refinements(profile_id)]


@router.get(f"{_REFINEMENTS_PATH}/{{refinement_id}}")
async def get_refinement(
    profile_id: str,
    refinement_id: str,
    state: AppState = Depends(get_state),
) -> RefinementResponse:
    """One refinement of this profile, with progress while it is being evaluated."""
    return _to_response(state, _get_refinement(state, profile_id, refinement_id))


@router.post(f"{_REFINEMENTS_PATH}/{{refinement_id}}/evaluate", status_code=202)
async def evaluate_refinement(
    profile_id: str,
    refinement_id: str,
    req: RefinementEvaluateRequest,
    state: AppState = Depends(get_state),
) -> RefinementEvaluateResponse:
    """Start a held-out baseline-vs-candidate evaluation in the background (FR-5, FR-12).

    Only a ``proposal_ready`` refinement can be evaluated, and one
    evaluation runs per profile at a time.  The profile is not written.
    """
    _get_profile(state, profile_id)
    record = _get_refinement(state, profile_id, refinement_id)
    if record.status != REFINEMENT_STATUS_PROPOSAL_READY:
        raise HTTPException(
            status_code=409,
            detail=f"Only a proposal_ready refinement can be evaluated; this one is {record.status}",
        )
    proposal = ProposalRecord.from_dict(record.proposal)

    try:
        dataset, _path = load_evaluation_dataset()
        cases = select_cases(dataset, req.case_ids)
    except DatasetUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not cases:
        raise HTTPException(status_code=400, detail="No evaluation cases selected")

    cp_id = req.chat_provider_id or proposal.proposal_chat_provider_id
    cp = state.get_chat_provider(cp_id)
    if cp is None:
        raise HTTPException(status_code=400, detail=f"Chat Provider not found: {cp_id}")
    if cp.profile_id != profile_id:
        raise HTTPException(
            status_code=400,
            detail=f"Chat Provider {cp_id} does not use this profile; pass chat_provider_id"
            " of a Chat Provider that does",
        )
    judge_cp_id = req.judge_chat_provider_id or state.config.judge_chat_provider_id
    if judge_cp_id and state.get_chat_provider(judge_cp_id) is None:
        raise HTTPException(status_code=400, detail=f"Judge Chat Provider not found: {judge_cp_id}")

    if not state.try_start_refinement_job(profile_id):
        raise HTTPException(
            status_code=409, detail="An evaluation is already running for this profile"
        )
    try:
        state.telemetry.update_refinement(
            refinement_id,
            expected_status=REFINEMENT_STATUS_PROPOSAL_READY,
            status=REFINEMENT_STATUS_EVALUATING,
        )
    except RefinementConflictError as exc:
        state.finish_refinement_job(profile_id, refinement_id)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RefinementStoreError as exc:
        state.finish_refinement_job(profile_id, refinement_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    case_ids = [case.id for case in cases]
    try:
        start_evaluation(
            state,
            EvaluationRequest(
                refinement_id=refinement_id,
                profile_id=profile_id,
                proposal=proposal,
                case_ids=case_ids,
                reps=req.reps,
                chat_provider_id=cp_id,
                judge_chat_provider_id=judge_cp_id,
            ),
        )
    except RuntimeError as exc:
        # The worker never started: release the slot and record why, so the
        # row does not sit in "evaluating" with no thread behind it.
        state.finish_refinement_job(profile_id, refinement_id)
        try:
            state.telemetry.update_refinement(
                refinement_id,
                expected_status=REFINEMENT_STATUS_EVALUATING,
                status=REFINEMENT_STATUS_FAILED,
                error=f"evaluation worker did not start: {exc}",
            )
        except RefinementStoreError:
            logger.exception("Could not mark refinement %s failed", refinement_id)
        raise HTTPException(status_code=500, detail=f"Evaluation did not start: {exc}") from exc
    return RefinementEvaluateResponse(
        refinement_id=refinement_id,
        status=REFINEMENT_STATUS_EVALUATING,
        case_ids=case_ids,
        reps=req.reps,
        run_count=len(case_ids) * req.reps * 2,
        chat_provider_id=cp_id,
        judge_chat_provider_id=judge_cp_id,
    )


async def _decide(state: AppState, decide: Callable[[], RefinementRecord]) -> RefinementResponse:
    """Run a locked decision off the event loop and map its errors to HTTP."""
    try:
        record = await asyncio.to_thread(decide)
    except refinement_decisions.DecisionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return _to_response(state, record)


@router.post(f"{_REFINEMENTS_PATH}/{{refinement_id}}/accept")
async def accept_refinement(
    profile_id: str,
    refinement_id: str,
    req: RefinementAcceptRequest,
    state: AppState = Depends(get_state),
) -> RefinementResponse:
    """Write the candidate to the profile's RLM prompt (FR-9).

    Needs a complete evaluation with at least one deterministic case, an
    unchanged profile prompt, and ``acknowledge_regression`` when flagged.
    """
    return await _decide(
        state,
        lambda: refinement_decisions.accept(
            state,
            profile_id,
            refinement_id,
            acknowledge_regression=req.acknowledge_regression,
            reason=req.reason,
        ),
    )


@router.post(f"{_REFINEMENTS_PATH}/{{refinement_id}}/reject")
async def reject_refinement(
    profile_id: str,
    refinement_id: str,
    req: RefinementRejectRequest,
    state: AppState = Depends(get_state),
) -> RefinementResponse:
    """Record a reject; the profile is not written (FR-10)."""
    return await _decide(
        state,
        lambda: refinement_decisions.reject(state, profile_id, refinement_id, reason=req.reason),
    )


@router.post(f"{_REFINEMENTS_PATH}/{{refinement_id}}/rollback")
async def rollback_refinement(
    profile_id: str,
    refinement_id: str,
    state: AppState = Depends(get_state),
) -> RefinementResponse:
    """Restore the profile's RLM prompt to its state at proposal time (FR-11)."""
    return await _decide(
        state, lambda: refinement_decisions.rollback(state, profile_id, refinement_id)
    )
