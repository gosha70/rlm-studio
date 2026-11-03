"""Named constants for evidence-gated prompt refinement (#78).

Refinement statuses cross the telemetry store, the evaluation service, the
REST routes and the frontend, so they are spelled once here.  See
``specs/prompt-refinement/spec.md`` for the lifecycle.
"""

from typing import Final, Literal, TypeAlias

REFINEMENT_STATUS_PROPOSAL_READY: Final[Literal["proposal_ready"]] = "proposal_ready"
"""A candidate supplement exists; nothing has been evaluated yet."""

REFINEMENT_STATUS_EVALUATING: Final[Literal["evaluating"]] = "evaluating"
"""A held-out baseline-vs-candidate evaluation is running in this process."""

REFINEMENT_STATUS_EVALUATED: Final[Literal["evaluated"]] = "evaluated"
"""The evaluation finished; the record says whether it is complete."""

REFINEMENT_STATUS_INTERRUPTED: Final[Literal["interrupted"]] = "interrupted"
"""The server stopped while an evaluation was running."""

REFINEMENT_STATUS_FAILED: Final[Literal["failed"]] = "failed"
"""The evaluation raised; the row's ``error`` carries the reason."""

REFINEMENT_STATUS_ACCEPTED: Final[Literal["accepted"]] = "accepted"
"""The user accepted; the candidate was written to the profile."""

REFINEMENT_STATUS_REJECTED: Final[Literal["rejected"]] = "rejected"
"""The user rejected; the profile was not written."""

REFINEMENT_STATUS_ROLLED_BACK: Final[Literal["rolled_back"]] = "rolled_back"
"""An accepted candidate was reverted to the recorded prior state."""

RefinementStatus: TypeAlias = Literal[
    "proposal_ready",
    "evaluating",
    "evaluated",
    "interrupted",
    "failed",
    "accepted",
    "rejected",
    "rolled_back",
]
"""Every value the ``refinements.status`` column can hold."""

REFINEMENT_INTERRUPTED_ON_STARTUP_ERROR = "server restarted during evaluation"
"""``error`` text written when startup marks a running evaluation interrupted."""

# ---------------------------------------------------------------------------
# Proposal limits (spec FR-3, FR-17)
# ---------------------------------------------------------------------------

REFINEMENT_MAX_SOURCE_RUNS = 50
"""Most source runs one proposal request may name."""

REFINEMENT_EVIDENCE_MAX_TRACES = 12
"""Most source runs rendered into the evidence digest; newer runs win."""

REFINEMENT_EVIDENCE_MAX_CHARS = 24_000
"""Character budget of the whole evidence digest."""

REFINEMENT_EVIDENCE_QUERY_MAX_CHARS = 2_000
"""Per-run cap on the query text in the digest."""

REFINEMENT_EVIDENCE_ANSWER_MAX_CHARS = 3_000
"""Per-run cap on the answer text in the digest."""

REFINEMENT_EVIDENCE_ERROR_MAX_CHARS = 1_000
"""Per-run cap on the error text in the digest."""

REFINEMENT_CANDIDATE_MAX_CHARS = 8_000
"""Longest candidate supplement a proposal may return."""
