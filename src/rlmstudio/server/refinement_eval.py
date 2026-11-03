"""Held-out evaluation for prompt refinement (#78, spec FR-5..FR-8, FR-16).

Runs a baseline arm (the supplement the proposal was built on) and a
candidate arm (the proposed supplement) on the same held-out benchmark case,
same Chat Provider, separate adapters, temperature 0, through
:class:`RunMatrixComparisonUseCase`.  Every arm is recorded as an ordinary
telemetry run and its id kept in the evaluation record.

Evaluation runs are recorded **without** a profile id: they are not runs of
the profile's real prompt (the candidate arm never was), so they must never
come back as refinement source runs.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import rlmstudio
from rlmstudio.application.dto import RunConfigDTO
from rlmstudio.application.refinement_constants import (
    REFINEMENT_ARM_BASELINE,
    REFINEMENT_ARM_CANDIDATE,
    REFINEMENT_DATASET_ENV_SUFFIX,
    REFINEMENT_DEFAULT_DATASET,
    REFINEMENT_EVALUATION_TEMPERATURE,
    REFINEMENT_HOLDOUT_TAG,
    REFINEMENT_STATUS_EVALUATED,
    REFINEMENT_STATUS_EVALUATING,
    REFINEMENT_STATUS_FAILED,
)
from rlmstudio.application.sandbox_vars import MODE_RLM
from rlmstudio.application.services.outcome_classifier import classify_execution_outcome
from rlmstudio.application.services.provider_failure import redact_secrets
from rlmstudio.application.services.refinement import ProposalRecord
from rlmstudio.application.services.refinement_evaluation import (
    ArmResult,
    CaseRuns,
    EvaluationRecord,
    EvaluationSetup,
    JudgeInfo,
    RepResult,
    summarise_evaluation,
)
from rlmstudio.application.use_cases.run_matrix_comparison import (
    MatrixSlotDTO,
    MatrixSlotResultDTO,
    RunMatrixComparisonUseCase,
)
from rlmstudio.benchmark.corpus import CorpusError, materialise_case
from rlmstudio.benchmark.dataset import BenchmarkCase, BenchmarkDataset, load_dataset
from rlmstudio.benchmark.matrix_runner import (
    BUDGET_KEY_MAX_COST,
    BUDGET_KEY_MAX_RECURSION_DEPTH,
    BUDGET_KEY_MAX_STEPS,
    BUDGET_KEY_MAX_TIME_SECONDS,
)
from rlmstudio.benchmark.scoring import JudgeScorer, matches
from rlmstudio.branding import env, env_name
from rlmstudio.server.dependencies import AppState
from rlmstudio.server.models import ChatProviderConfig
from rlmstudio.server.routes._helpers import _record_slot_telemetry, _SlotMeta
from rlmstudio.telemetry.store import RefinementStoreError, content_sha256

logger = logging.getLogger(__name__)

CORPUS_DIR_NAME = "corpus"
"""Public-text cache beside the dataset, as ``rlm-studio bench`` uses it."""

EXCLUDED_SAME_DOCUMENT = "same document as a source run"
"""Reason recorded when a case shares its content hash with a source run (FR-6)."""


class DatasetUnavailableError(RuntimeError):
    """The evaluation dataset YAML cannot be found or read."""


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def evaluation_dataset_path() -> Path:
    """The evaluation dataset: the env override, else the source checkout's copy.

    The benchmark set is not packaged in the wheel, so an installed Studio
    without a checkout needs ``RLM_STUDIO_REFINEMENT_DATASET``.
    """
    override = env(REFINEMENT_DATASET_ENV_SUFFIX)
    if override:
        return Path(override).expanduser()
    checkout_root = Path(rlmstudio.__file__).resolve().parents[2]
    return checkout_root / REFINEMENT_DEFAULT_DATASET


def load_evaluation_dataset() -> tuple[BenchmarkDataset, Path]:
    """Load the evaluation dataset or raise :class:`DatasetUnavailableError`."""
    path = evaluation_dataset_path()
    if not path.is_file():
        raise DatasetUnavailableError(
            f"Evaluation dataset not found at {path}. Run Studio from a source checkout"
            f" or set {env_name(REFINEMENT_DATASET_ENV_SUFFIX)} to a benchmark YAML."
        )
    try:
        return load_dataset(str(path)), path
    except (OSError, ValueError) as exc:
        raise DatasetUnavailableError(f"Evaluation dataset at {path} is unreadable: {exc}") from exc


def materialise_offline(case: BenchmarkCase, dataset_path: Path) -> str | None:
    """Fill the case's content without network access; return a skip reason or None."""
    try:
        return materialise_case(
            case,
            base_dir=dataset_path.parent,
            corpus_dir=dataset_path.parent / CORPUS_DIR_NAME,
            fetch=False,
        )
    except (CorpusError, OSError) as exc:
        return str(exc)


@dataclass(frozen=True)
class HoldoutCaseInfo:
    """What the UI needs to choose evaluation cases (FR-16)."""

    id: str
    query: str
    tags: list[str]
    size: str | None
    holdout: bool
    deterministic: bool
    materialisable: bool
    skip_reason: str | None


def describe_cases(dataset: BenchmarkDataset, dataset_path: Path) -> list[HoldoutCaseInfo]:
    """Describe every dataset case, materialising offline to learn what can run."""
    size_names = set(dataset.metadata.get("size_buckets", {}))
    infos: list[HoldoutCaseInfo] = []
    for case in dataset:
        reason = materialise_offline(case, dataset_path)
        infos.append(
            HoldoutCaseInfo(
                id=case.id,
                query=case.query,
                tags=list(case.tags),
                size=next((t for t in case.tags if t in size_names), None),
                holdout=REFINEMENT_HOLDOUT_TAG in case.tags,
                deterministic=case.expected_answer is not None,
                materialisable=reason is None,
                skip_reason=reason,
            )
        )
    return infos


def select_cases(dataset: BenchmarkDataset, case_ids: list[str] | None) -> list[BenchmarkCase]:
    """The named cases in dataset order, or the ``holdout``-tagged ones by default.

    Raises:
        ValueError: A named case is not in the dataset.
    """
    if case_ids is None:
        return list(dataset.filter_by_tag(REFINEMENT_HOLDOUT_TAG))
    by_id = {case.id: case for case in dataset}
    unknown = [cid for cid in case_ids if cid not in by_id]
    if unknown:
        raise ValueError(f"Unknown case id(s): {', '.join(unknown)}")
    wanted = set(case_ids)
    return [case for case in dataset if case.id in wanted]


# ---------------------------------------------------------------------------
# Running one evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationRequest:
    """Everything a worker needs; resolved and validated by the route."""

    refinement_id: str
    profile_id: str
    proposal: ProposalRecord
    case_ids: list[str]
    reps: int
    chat_provider_id: str
    judge_chat_provider_id: str | None


def _provider_and_model(state: AppState, cp: ChatProviderConfig) -> tuple[str, str]:
    lp = state.get_llm_provider(cp.llm_provider_id) if cp.llm_provider_id else None
    provider = (lp.backend if lp else None) or cp.llm_provider or "unknown"
    model = (lp.model if lp else None) or cp.llm_model or ""
    return provider, model


def _apply_case_budget(config: RunConfigDTO, budget: dict[str, object]) -> None:
    """Apply the case's own budget keys; keys the case omits keep the CP's values."""
    if budget.get(BUDGET_KEY_MAX_STEPS) is not None:
        config.max_steps = int(str(budget[BUDGET_KEY_MAX_STEPS]))
    if budget.get(BUDGET_KEY_MAX_TIME_SECONDS) is not None:
        config.max_time_seconds = float(str(budget[BUDGET_KEY_MAX_TIME_SECONDS]))
    if budget.get(BUDGET_KEY_MAX_COST) is not None:
        config.max_cost = float(str(budget[BUDGET_KEY_MAX_COST]))
    if budget.get(BUDGET_KEY_MAX_RECURSION_DEPTH) is not None:
        config.max_recursion_depth = int(str(budget[BUDGET_KEY_MAX_RECURSION_DEPTH]))


def _arm_slot(
    state: AppState,
    cp: ChatProviderConfig,
    cp_id: str,
    *,
    arm: str,
    supplement: str,
    case: BenchmarkCase,
    provider: str,
    model: str,
) -> MatrixSlotDTO:
    # Same per-slot knobs the Compare matrix applies (profile RLM settings),
    # then the arm's supplement replaces the profile's.
    from rlmstudio.server.routes.compare_matrix import _build_slot_run_config

    config = _build_slot_run_config(state, cp, MODE_RLM)
    config.system_prompt_extra = supplement or None
    _apply_case_budget(config, case.budget)
    return MatrixSlotDTO(
        slot_id=arm,
        mode=MODE_RLM,
        llm=state.create_llm_adapter_for_chat_provider(
            cp_id, temperature=REFINEMENT_EVALUATION_TEMPERATURE
        ),
        sandbox=state.create_sandbox(),
        label=f"{case.id} · {arm}",
        provider=provider,
        model=model,
        config=config,
    )


def _score_arm(
    slot: MatrixSlotResultDTO,
    run_id: str,
    case: BenchmarkCase,
    judge: JudgeScorer | None,
) -> ArmResult:
    result = slot.result
    outcome = classify_execution_outcome(result.success, result.error, result.answer)
    passed: bool | None = None
    if case.expected_answer is not None:
        passed = outcome.is_usable and matches(result.answer, case.expected_answer, case.match)
    judge_overall: float | None = None
    if judge is not None and outcome.is_usable:
        try:
            judge_overall = judge.score(
                query=case.query,
                response=result.answer,
                source=case.content,
                rubric_hint=case.rubric_hint,
            ).overall
        except Exception:
            # A judge failure leaves the score empty; it never gates (FR-8).
            logger.exception("Judge failed for refinement arm %s", run_id)
    return ArmResult(
        run_id=run_id,
        success=result.success,
        usable=outcome.is_usable,
        outcome_category=outcome.category.value,
        passed=passed,
        judge_overall=judge_overall,
        cost=result.total_cost,
        steps=result.steps,
        error=redact_secrets(result.error) if result.error else None,
    )


def _run_rep(
    state: AppState,
    request: EvaluationRequest,
    cp: ChatProviderConfig,
    case: BenchmarkCase,
    case_hash: str,
    rep: int,
    provider: str,
    model: str,
    judge: JudgeScorer | None,
) -> RepResult:
    supplements = {
        REFINEMENT_ARM_BASELINE: request.proposal.base.resolved,
        REFINEMENT_ARM_CANDIDATE: request.proposal.candidate,
    }
    try:
        slots = [
            _arm_slot(
                state,
                cp,
                request.chat_provider_id,
                arm=arm,
                supplement=text,
                case=case,
                provider=provider,
                model=model,
            )
            for arm, text in supplements.items()
        ]
        result = RunMatrixComparisonUseCase().execute(case.content, case.query, slots)
    except Exception:
        # Both arms are missing for this repetition; the evaluation is then
        # incomplete rather than silently smaller (FR-7).
        logger.exception(
            "Refinement %s: %s rep %d did not run", request.refinement_id, case.id, rep
        )
        return RepResult(rep=rep, baseline=None, candidate=None)

    arms: dict[str, ArmResult] = {}
    created_at = datetime.now(timezone.utc)
    for slot_result in result.slots:
        run_id = uuid.uuid4().hex
        _record_slot_telemetry(
            state=state,
            slot_result=slot_result,
            meta=_SlotMeta(
                chat_provider_id=request.chat_provider_id,
                chat_provider_name=cp.name,
                execution_id=run_id,
                profile_id=None,
            ),
            session_id=f"refinement-{request.refinement_id}",
            comparison_group_id=result.comparison_group_id,
            query=case.query,
            created_at=created_at,
            content_sha256=case_hash,
        )
        arms[slot_result.slot_id] = _score_arm(slot_result, run_id, case, judge)
    return RepResult(
        rep=rep,
        baseline=arms.get(REFINEMENT_ARM_BASELINE),
        candidate=arms.get(REFINEMENT_ARM_CANDIDATE),
    )


def _judge_for(
    state: AppState, judge_cp_id: str | None
) -> tuple[JudgeScorer | None, JudgeInfo | None]:
    if not judge_cp_id:
        return None, None
    judge_cp = state.get_chat_provider(judge_cp_id)
    if judge_cp is None:
        return None, None
    _provider, judge_model = _provider_and_model(state, state.resolve_chat_provider(judge_cp))
    scorer = JudgeScorer(state.create_llm_adapter_for_chat_provider(judge_cp_id), model=judge_model)
    return scorer, JudgeInfo(
        chat_provider_id=judge_cp_id, model=judge_model, prompt_version=scorer.prompt_version
    )


def run_evaluation(state: AppState, request: EvaluationRequest) -> EvaluationRecord:
    """Run every selected case × repetition through both arms and summarise (FR-5..FR-8)."""
    started_at = time.time()
    dataset, dataset_path = load_evaluation_dataset()
    cases = select_cases(dataset, request.case_ids)
    raw_cp = state.get_chat_provider(request.chat_provider_id)
    if raw_cp is None:
        raise ValueError(f"Chat Provider not found: {request.chat_provider_id}")
    cp = state.resolve_chat_provider(raw_cp)
    provider, model = _provider_and_model(state, cp)
    judge, judge_info = _judge_for(state, request.judge_chat_provider_id)

    source_hashes = set(request.proposal.source_content_sha256.values())
    excluded: dict[str, str] = {}
    skipped: dict[str, str] = {}
    runnable: list[tuple[BenchmarkCase, str]] = []
    for case in cases:
        reason = materialise_offline(case, dataset_path)
        if reason is not None:
            skipped[case.id] = reason
            continue
        case_hash = content_sha256(case.content)
        if case_hash in source_hashes:
            excluded[case.id] = EXCLUDED_SAME_DOCUMENT
            continue
        runnable.append((case, case_hash))

    setup = EvaluationSetup(
        chat_provider_id=request.chat_provider_id,
        provider=provider,
        model=model,
        temperature=REFINEMENT_EVALUATION_TEMPERATURE,
        reps=request.reps,
        dataset_name=dataset.name,
        dataset_version=str(dataset.metadata.get("version", "")),
        case_ids=[case.id for case in cases],
        excluded_cases=excluded,
        skipped_cases=skipped,
        judge=judge_info,
        started_at=started_at,
    )

    total = len(runnable) * request.reps
    done = 0
    state.set_refinement_progress(request.refinement_id, done, total)
    case_runs: list[CaseRuns] = []
    for case, case_hash in runnable:
        reps: list[RepResult] = []
        for rep in range(1, request.reps + 1):
            reps.append(_run_rep(state, request, cp, case, case_hash, rep, provider, model, judge))
            done += 1
            state.set_refinement_progress(request.refinement_id, done, total)
        case_runs.append(
            CaseRuns(
                case_id=case.id,
                deterministic=case.expected_answer is not None,
                content_sha256=case_hash,
                reps=reps,
            )
        )
    return summarise_evaluation(setup, case_runs, finished_at=time.time())


def start_evaluation(state: AppState, request: EvaluationRequest) -> threading.Thread:
    """Run :func:`run_evaluation` on a daemon thread and store the outcome.

    The caller has already claimed the profile's job slot and moved the row
    to ``evaluating``; the worker always releases the slot.
    """

    def worker() -> None:
        try:
            record = run_evaluation(state, request)
            state.telemetry.update_refinement(
                request.refinement_id,
                expected_status=REFINEMENT_STATUS_EVALUATING,
                status=REFINEMENT_STATUS_EVALUATED,
                evaluation=record.to_dict(),
            )
        except Exception as exc:
            logger.exception("Refinement %s evaluation failed", request.refinement_id)
            try:
                state.telemetry.update_refinement(
                    request.refinement_id,
                    expected_status=REFINEMENT_STATUS_EVALUATING,
                    status=REFINEMENT_STATUS_FAILED,
                    error=redact_secrets(str(exc)) or type(exc).__name__,
                )
            except RefinementStoreError:
                logger.exception("Could not mark refinement %s failed", request.refinement_id)
        finally:
            state.finish_refinement_job(request.profile_id, request.refinement_id)

    thread = threading.Thread(
        target=worker, name=f"refinement-eval-{request.refinement_id[:8]}", daemon=True
    )
    thread.start()
    return thread
