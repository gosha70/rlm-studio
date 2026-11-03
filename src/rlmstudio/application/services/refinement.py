"""Proposal half of evidence-gated prompt refinement (#78, spec FR-3/FR-4/FR-17).

Turns selected runs of a profile into a redacted, capped evidence digest,
asks one model for a revised RLM supplement, validates the reply, and
computes the diff server-side.  Nothing here writes to a profile.

The module is layer-clean: it takes plain :class:`EvidenceRun` values (the
server converts telemetry rows into them) and calls the model through
:class:`LLMPort`.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from rlmstudio.application.ports.llm_port import LLMPort
from rlmstudio.application.refinement_constants import (
    REFINEMENT_CANDIDATE_MAX_CHARS,
    REFINEMENT_EVIDENCE_ANSWER_MAX_CHARS,
    REFINEMENT_EVIDENCE_ERROR_MAX_CHARS,
    REFINEMENT_EVIDENCE_MAX_CHARS,
    REFINEMENT_EVIDENCE_MAX_TRACES,
    REFINEMENT_EVIDENCE_QUERY_MAX_CHARS,
)
from rlmstudio.application.services.outcome_classifier import is_degraded_answer
from rlmstudio.application.services.provider_failure import redact_secrets

EMPTY_SUPPLEMENT_PLACEHOLDER = "(empty: the profile has no RLM supplement yet)"
"""Rendered in place of an empty current supplement so the model sees it explicitly."""

REDACTED_LINE_PLACEHOLDER = "[line removed: possible secret]"
"""Replaces a line dropped by the key-value redaction rule (FR-17)."""

# FR-17 line rule: a line is dropped when an identifier ending in one of these
# names is followed (after an optional closing quote and whitespace) by ``=``
# or ``:``.  That covers ``key=``, ``API_KEY = x``, ``"token": "x"`` and
# ``Authorization: Bearer x``.  Identifiers that merely *end* in a name
# (``aws_secret_key=``) are dropped too; failing closed is the intent.
_SECRET_NAMES = (
    "api_key",
    "apikey",
    "key",
    "token",
    "password",
    "passwd",
    "secret",
    "authorization",
)
_SECRET_LINE = re.compile(
    r"(?i)(?<![A-Za-z0-9])[A-Za-z0-9_-]*?(?:" + "|".join(_SECRET_NAMES) + r")[\"']?\s*[:=]"
)

_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


class ProposalError(ValueError):
    """The model's reply is not a usable proposal."""


# ---------------------------------------------------------------------------
# Evidence digest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceRun:
    """One source run, as the digest needs it (no telemetry types)."""

    id: str
    created_at: float
    mode: str
    success: bool
    outcome_category: str | None
    query: str
    answer: str
    error: str | None
    steps_count: int
    action_types: tuple[str, ...]
    total_tokens: int
    total_cost: float
    elapsed_seconds: float


@dataclass(frozen=True)
class EvidenceDigest:
    """The exact text sent to the proposer, plus how it was cut (FR-4)."""

    text: str
    sha256: str
    included_run_ids: list[str]
    excluded_run_ids: list[str]
    truncated_chars: int
    redacted_lines: int


def redact_evidence_text(text: str) -> tuple[str, int]:
    """Apply the FR-17 policy; return the redacted text and the dropped-line count.

    Known key shapes are replaced first (:func:`redact_secrets`), then every
    line matching the key-value rule is replaced with a placeholder.  This
    does not catch secrets in free text (spec §6, residual evidence leakage).
    """
    shaped = redact_secrets(text)
    kept: list[str] = []
    dropped = 0
    for line in shaped.split("\n"):
        if _SECRET_LINE.search(line):
            kept.append(REDACTED_LINE_PLACEHOLDER)
            dropped += 1
        else:
            kept.append(line)
    return "\n".join(kept), dropped


def _clip(text: str, limit: int) -> tuple[str, int]:
    if len(text) <= limit:
        return text, 0
    cut = len(text) - limit
    return f"{text[:limit]}\n[truncated {cut:,} chars]", cut


def _render_run(run: EvidenceRun) -> tuple[str, int, int]:
    """Render one run; return ``(block, truncated_chars, redacted_lines)``.

    Each field is redacted whole and clipped afterwards.  Clipping first
    would cut a key at the limit into a fragment too short for the key-shape
    patterns, and that fragment would reach the proposer.
    """
    query, q_red = redact_evidence_text(run.query)
    answer, a_red = redact_evidence_text(run.answer)
    error, e_red = redact_evidence_text(run.error or "")
    query, q_cut = _clip(query, REFINEMENT_EVIDENCE_QUERY_MAX_CHARS)
    answer, a_cut = _clip(answer, REFINEMENT_EVIDENCE_ANSWER_MAX_CHARS)
    error, e_cut = _clip(error, REFINEMENT_EVIDENCE_ERROR_MAX_CHARS)

    outcome = run.outcome_category or "unknown"
    if is_degraded_answer(run.success, run.answer):
        outcome += " (degraded answer)"
    actions = Counter(run.action_types)
    action_text = ", ".join(f"{name} x{count}" for name, count in sorted(actions.items()))
    created = datetime.fromtimestamp(run.created_at, tz=timezone.utc).isoformat()
    lines = [
        f"### Run {run.id}",
        f"- created: {created}",
        f"- mode: {run.mode}",
        f"- outcome: {outcome}",
        f"- steps: {run.steps_count}" + (f" ({action_text})" if action_text else ""),
        f"- tokens: {run.total_tokens}; cost_usd: {run.total_cost:.6f};"
        f" elapsed_s: {run.elapsed_seconds:.1f}",
        "Query:",
        query,
        "Answer:",
        answer or "(none)",
    ]
    if error:
        lines += ["Error:", error]
    return "\n".join(lines), q_cut + a_cut + e_cut, q_red + a_red + e_red


def build_evidence_digest(runs: Sequence[EvidenceRun]) -> EvidenceDigest:
    """Render up to the trace cap of the newest runs within the character budget.

    Runs beyond :data:`REFINEMENT_EVIDENCE_MAX_TRACES`, or whose block would
    push the digest past :data:`REFINEMENT_EVIDENCE_MAX_CHARS`, are listed
    in ``excluded_run_ids`` rather than silently dropped.
    """
    ordered = sorted(runs, key=lambda r: r.created_at, reverse=True)
    blocks: list[str] = []
    included: list[str] = []
    excluded: list[str] = []
    truncated = 0
    redacted = 0
    used = 0
    for run in ordered:
        if len(included) >= REFINEMENT_EVIDENCE_MAX_TRACES:
            excluded.append(run.id)
            continue
        block, cut, dropped = _render_run(run)
        separator = 2 if blocks else 0
        if used + separator + len(block) > REFINEMENT_EVIDENCE_MAX_CHARS:
            excluded.append(run.id)
            continue
        blocks.append(block)
        included.append(run.id)
        used += separator + len(block)
        truncated += cut
        redacted += dropped
    text = "\n\n".join(blocks)
    return EvidenceDigest(
        text=text,
        sha256=_sha256(text),
        included_run_ids=included,
        excluded_run_ids=excluded,
        truncated_chars=truncated,
        redacted_lines=redacted,
    )


# ---------------------------------------------------------------------------
# Proposal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProposalResult:
    """A validated candidate supplement from one model call."""

    candidate: str
    rationale: str
    diff: str
    prompt_version: str
    prompt_sha256: str
    model: str
    input_tokens: int
    output_tokens: int


def render_proposal_prompt(
    template: str, *, digest: EvidenceDigest, current_supplement: str, base_prompt_version: str
) -> str:
    """Fill the proposal template; values are inserted verbatim, never re-formatted."""
    return template.format(
        base_prompt_version=base_prompt_version,
        current_supplement=current_supplement or EMPTY_SUPPLEMENT_PLACEHOLDER,
        evidence=digest.text,
        max_candidate_chars=f"{REFINEMENT_CANDIDATE_MAX_CHARS:,}",
    )


def parse_proposal_reply(raw: str) -> tuple[str, str]:
    """Extract ``(candidate, rationale)`` from a bare or fenced JSON reply.

    Raises:
        ProposalError: No JSON object, wrong shape, or empty fields.
    """
    text = raw.strip()
    fenced = _FENCED_JSON.search(text)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ProposalError("reply contains no JSON object")
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProposalError(f"reply is not valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ProposalError("reply JSON is not an object")
    candidate, rationale = data.get("candidate"), data.get("rationale")
    if not isinstance(candidate, str) or not isinstance(rationale, str):
        raise ProposalError("reply must have string 'candidate' and 'rationale' fields")
    candidate, rationale = candidate.strip(), rationale.strip()
    if not candidate:
        raise ProposalError("candidate is empty")
    if not rationale:
        raise ProposalError("rationale is empty")
    return candidate, rationale


def supplement_diff(current: str, candidate: str) -> str:
    """Unified diff from the current supplement to the candidate (server-computed)."""
    return "\n".join(
        difflib.unified_diff(
            current.splitlines(),
            candidate.splitlines(),
            fromfile="current",
            tofile="candidate",
            lineterm="",
        )
    )


def propose_supplement(
    llm: LLMPort,
    *,
    template: str,
    prompt_version: str,
    digest: EvidenceDigest,
    current_supplement: str,
    base_prompt_version: str,
) -> ProposalResult:
    """Make the single proposal call and validate the reply (FR-3).

    Raises:
        ProposalError: The reply is unusable, identical to the current
            supplement, or longer than :data:`REFINEMENT_CANDIDATE_MAX_CHARS`.
        Exception: Whatever the adapter raises for a provider failure.
    """
    prompt = render_proposal_prompt(
        template,
        digest=digest,
        current_supplement=current_supplement,
        base_prompt_version=base_prompt_version,
    )
    response = llm.complete([{"role": "user", "content": prompt}])
    candidate, rationale = parse_proposal_reply(response.content)
    if candidate == current_supplement.strip():
        raise ProposalError("candidate is identical to the current supplement")
    if len(candidate) > REFINEMENT_CANDIDATE_MAX_CHARS:
        raise ProposalError(
            f"candidate is {len(candidate):,} chars; the limit is"
            f" {REFINEMENT_CANDIDATE_MAX_CHARS:,}"
        )
    return ProposalResult(
        candidate=candidate,
        rationale=rationale,
        diff=supplement_diff(current_supplement, candidate),
        prompt_version=prompt_version,
        prompt_sha256=_sha256(prompt),
        model=response.model,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


# ---------------------------------------------------------------------------
# Stored record (refinements.proposal_json)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseSupplementState:
    """The profile's RLM prompt state the proposal was built on (FR-4).

    ``rlm_key_present`` / ``rlm_key_value`` are the raw
    ``system_prompts["rlm"]`` state rollback restores; ``resolved`` is the
    text the RLM loop actually received (custom text or the template's).
    """

    resolved: str
    sha256: str
    rlm_key_present: bool
    rlm_key_value: str | None
    template_name: str | None
    base_prompt_version: str
    base_prompt_sha256: str


@dataclass(frozen=True)
class ProposalRecord:
    """Everything a proposal stores; phases C and D read it back by attribute."""

    source_run_ids: list[str]
    source_content_sha256: dict[str, str]
    evidence: EvidenceDigest
    proposal_chat_provider_id: str
    provider: str
    model: str
    prompt_version: str
    prompt_sha256: str
    input_tokens: int
    output_tokens: int
    base: BaseSupplementState
    candidate: str
    rationale: str
    diff: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form stored in ``refinements.proposal_json``."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProposalRecord:
        """Rebuild a record written by :meth:`to_dict`."""
        values = dict(data)
        values["evidence"] = EvidenceDigest(**values["evidence"])
        values["base"] = BaseSupplementState(**values["base"])
        return cls(**values)


def supplement_sha256(text: str) -> str:
    """Hash used to detect that a profile's supplement changed since proposal."""
    return _sha256(text)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
