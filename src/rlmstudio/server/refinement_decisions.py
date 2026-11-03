"""Accept, reject and rollback for prompt refinement (#78, spec FR-9..FR-11a).

Every decision holds ``AppState._config_lock`` from its status check
through its decision write (FR-10a), so two decisions on one refinement, or
a decision and a profile edit, cannot interleave in this process.  Accept
and rollback re-check the profile's RLM prompt, change only
``system_prompts["rlm"]``, use the checked config save, and restore the
in-memory profile when that save fails.

A successful save followed by a failed status write is the accepted
split-brain window (spec §6): the config is authoritative and the error says
so; nothing is retried or reverted.
"""

from __future__ import annotations

import time

from rlmstudio.application.refinement_constants import (
    REFINEMENT_STATUS_ACCEPTED,
    REFINEMENT_STATUS_EVALUATED,
    REFINEMENT_STATUS_FAILED,
    REFINEMENT_STATUS_INTERRUPTED,
    REFINEMENT_STATUS_PROPOSAL_READY,
    REFINEMENT_STATUS_REJECTED,
    REFINEMENT_STATUS_ROLLED_BACK,
)
from rlmstudio.application.sandbox_vars import MODE_RLM
from rlmstudio.application.services.refinement import (
    DecisionRecord,
    ProposalRecord,
    RollbackRecord,
    supplement_sha256,
)
from rlmstudio.application.services.refinement_evaluation import EvaluationRecord
from rlmstudio.server.dependencies import AppState
from rlmstudio.server.models import RunProfile
from rlmstudio.telemetry.store import RefinementRecord, RefinementStoreError

REJECTABLE_STATUSES = (
    REFINEMENT_STATUS_PROPOSAL_READY,
    REFINEMENT_STATUS_EVALUATED,
    REFINEMENT_STATUS_INTERRUPTED,
    REFINEMENT_STATUS_FAILED,
)
"""Statuses a reject is allowed from (FR-10)."""


class DecisionError(Exception):
    """A decision could not be made; ``status_code`` is the HTTP answer."""

    status_code = 400


class DecisionNotFoundError(DecisionError):
    """The profile or the refinement does not exist."""

    status_code = 404


class DecisionConflictError(DecisionError):
    """The refinement or the profile is not in the state the decision needs."""

    status_code = 409


class DecisionSaveError(DecisionError):
    """The checked config save failed; the in-memory profile was restored."""

    status_code = 500


class DecisionSplitBrainError(DecisionError):
    """The profile was saved but the refinement status write failed (spec §6)."""

    status_code = 500


def _user_profile(state: AppState, profile_id: str) -> RunProfile:
    for profile in state.user_profiles:
        if profile.id == profile_id:
            return profile
    if state.find_profile(profile_id) is not None:
        raise DecisionError("Built-in profiles cannot be refined")
    raise DecisionNotFoundError(f"Profile not found: {profile_id}")


def _refinement(state: AppState, profile_id: str, refinement_id: str) -> RefinementRecord:
    record = state.telemetry.get_refinement(refinement_id)
    if record is None or record.profile_id != profile_id:
        raise DecisionNotFoundError(f"Refinement not found: {refinement_id}")
    return record


def _resolved_rlm_prompt(profile: RunProfile) -> str:
    # The text the RLM loop receives: custom text wins, else the template's.
    from rlmstudio.server.routes.chat import _resolve_profile_prompt

    return _resolve_profile_prompt(profile, MODE_RLM) or ""


def _save_or_restore(state: AppState, profile: RunProfile, snapshot: dict[str, str]) -> None:
    """Checked save; on any failure put ``snapshot`` back and raise."""
    try:
        state.save_config_checked()
    except Exception as exc:
        profile.system_prompts = snapshot
        raise DecisionSaveError(
            f"The profile could not be saved ({exc}); nothing was changed"
        ) from exc


def accept(
    state: AppState,
    profile_id: str,
    refinement_id: str,
    *,
    acknowledge_regression: bool,
    reason: str | None,
) -> RefinementRecord:
    """Write the candidate to the profile's RLM prompt and record the decision (FR-9)."""
    with state._config_lock:
        profile = _user_profile(state, profile_id)
        record = _refinement(state, profile_id, refinement_id)
        if record.status != REFINEMENT_STATUS_EVALUATED:
            raise DecisionConflictError(
                f"Only an evaluated refinement can be accepted; this one is {record.status}"
            )
        if record.evaluation is None:
            raise DecisionConflictError("The refinement has no evaluation record")
        evaluation = EvaluationRecord.from_dict(record.evaluation)
        if not evaluation.complete:
            raise DecisionConflictError(
                "The evaluation is incomplete and cannot be accepted: "
                + "; ".join(evaluation.incomplete_reasons)
            )
        if evaluation.deterministic_case_count == 0:
            raise DecisionConflictError(
                "The evaluation has no case with an expected answer, so it cannot gate an accept"
            )
        if evaluation.regression and not acknowledge_regression:
            raise DecisionError(
                "The candidate regressed on "
                + ", ".join(evaluation.regression_case_ids)
                + "; set acknowledge_regression to accept it anyway"
            )
        proposal = ProposalRecord.from_dict(record.proposal)
        if supplement_sha256(_resolved_rlm_prompt(profile)) != proposal.base.sha256:
            raise DecisionConflictError(
                "The profile's RLM prompt changed after this proposal; propose again"
            )

        snapshot = dict(profile.system_prompts)
        profile.system_prompts = {**snapshot, MODE_RLM: proposal.candidate}
        _save_or_restore(state, profile, snapshot)

        decision = DecisionRecord(
            decision=REFINEMENT_STATUS_ACCEPTED,
            decided_at=time.time(),
            reason=reason,
            regression=evaluation.regression,
            acknowledge_regression=acknowledge_regression,
        )
        try:
            return state.telemetry.update_refinement(
                refinement_id,
                expected_status=REFINEMENT_STATUS_EVALUATED,
                status=REFINEMENT_STATUS_ACCEPTED,
                decision=decision.to_dict(),
            )
        except RefinementStoreError as exc:
            raise DecisionSplitBrainError(
                "The candidate was saved to the profile, but recording the accept failed"
                f" ({exc}). The profile now uses the candidate while this refinement still"
                " shows 'evaluated'."
            ) from exc


def reject(
    state: AppState,
    profile_id: str,
    refinement_id: str,
    *,
    reason: str | None,
) -> RefinementRecord:
    """Record a reject; the profile is never written (FR-10)."""
    with state._config_lock:
        record = _refinement(state, profile_id, refinement_id)
        if record.status not in REJECTABLE_STATUSES:
            raise DecisionConflictError(f"A refinement that is {record.status} cannot be rejected")
        regression = False
        if record.evaluation is not None:
            regression = EvaluationRecord.from_dict(record.evaluation).regression
        decision = DecisionRecord(
            decision=REFINEMENT_STATUS_REJECTED,
            decided_at=time.time(),
            reason=reason,
            regression=regression,
            acknowledge_regression=False,
        )
        try:
            return state.telemetry.update_refinement(
                refinement_id,
                expected_status=record.status,
                status=REFINEMENT_STATUS_REJECTED,
                decision=decision.to_dict(),
            )
        except RefinementStoreError as exc:
            raise DecisionSaveError(f"Recording the reject failed: {exc}") from exc


def rollback(state: AppState, profile_id: str, refinement_id: str) -> RefinementRecord:
    """Restore the RLM prompt state recorded at proposal time (FR-11)."""
    with state._config_lock:
        profile = _user_profile(state, profile_id)
        record = _refinement(state, profile_id, refinement_id)
        if record.status != REFINEMENT_STATUS_ACCEPTED:
            raise DecisionConflictError(
                f"Only an accepted refinement can be rolled back; this one is {record.status}"
            )
        if record.decision is None:
            raise DecisionConflictError("The accepted refinement has no decision record")
        accepted = DecisionRecord.from_dict(record.decision)
        proposal = ProposalRecord.from_dict(record.proposal)
        current = profile.system_prompts.get(MODE_RLM)
        if current is None or supplement_sha256(current) != supplement_sha256(proposal.candidate):
            raise DecisionConflictError(
                "The profile's RLM prompt changed after the accept; edit the profile instead"
            )

        snapshot = dict(profile.system_prompts)
        restored = dict(snapshot)
        if proposal.base.rlm_key_present and proposal.base.rlm_key_value is not None:
            restored[MODE_RLM] = proposal.base.rlm_key_value
        else:
            restored.pop(MODE_RLM, None)
        profile.system_prompts = restored
        _save_or_restore(state, profile, snapshot)

        decision = DecisionRecord(
            decision=accepted.decision,
            decided_at=accepted.decided_at,
            reason=accepted.reason,
            regression=accepted.regression,
            acknowledge_regression=accepted.acknowledge_regression,
            rollback=RollbackRecord(
                rolled_back_at=time.time(),
                restored_rlm_key_present=MODE_RLM in restored,
                restored_rlm_key_value=restored.get(MODE_RLM),
            ),
        )
        try:
            return state.telemetry.update_refinement(
                refinement_id,
                expected_status=REFINEMENT_STATUS_ACCEPTED,
                status=REFINEMENT_STATUS_ROLLED_BACK,
                decision=decision.to_dict(),
            )
        except RefinementStoreError as exc:
            raise DecisionSplitBrainError(
                "The profile's prior RLM prompt was restored, but recording the rollback failed"
                f" ({exc}). The profile no longer uses the candidate while this refinement still"
                " shows 'accepted'."
            ) from exc
