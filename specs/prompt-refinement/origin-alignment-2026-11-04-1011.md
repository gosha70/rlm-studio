# Origin alignment check — prompt-refinement

Gate: phase-complete preview, Phase F entry (after PR #119 merged at `e50ca51`).

Origin: https://github.com/gosha70/rlm-studio/issues/78 (read in full via `gh issue view 78` this session; parent epic #77 read too). Owner transcripts as quoted in `spec.md` frontmatter.

Origin claim:
> Studio records every run's trace, tokens, cost and outcome, and has a judge
> and a Compare matrix, but nothing turns that evidence into a proposed
> profile supplement and nothing checks a proposal against an unchanged
> baseline before it is saved. Build proposal -> held-out baseline/candidate
> evaluation -> explicit accept/reject/rollback with the evidence recorded.
> Prompt supplements only. Minimal variant for the 1.x release evaluation;
> the hardened variant is deferred to a follow-up issue.

Working claim:
> Settings → Profiles opens a refinement panel for a user profile. The user
> picks that profile's RLM runs (only runs with `profile_id` and a document
> hash), a proposal Chat Provider, and gets one model call through the
> versioned `prompts/refine_supplement.yaml` returning a candidate RLM
> supplement and rationale; the server computes the diff and stores the
> redacted evidence digest. Evaluation runs baseline and candidate through
> `RunMatrixComparisonUseCase` on held-out `longdoc-v1` cases (content-hash
> exclusion against the source runs), reps 2–5, temperature 0, optional
> judge, and reports per-case pass rates, a deterministic regression flag
> (advisory; Reject is the default and Accept needs an acknowledgement),
> accuracy, judge, cost, steps and outcome deltas. Accept and rollback are the
> only profile writes, under `_config_lock` with a hash re-check and a
> checked save; reject writes nothing; every decision and rollback is
> recorded on one `refinements` row, listed in the panel and exported as
> JSONL per profile. The base prompt YAML is never written. Hardening
> (events table, leases, tokens, revision counter, raw run JSONL export,
> judge-only gate) is deferred to #114.

Mismatches:
  - Proposal input: #78 says the proposal call runs "over those traces, the current supplement and the base prompt". FR-3 and `refine_supplement.yaml` send the base prompt's version string and a one-sentence protocol description, not the base prompt text. The owner approved the minimal plan containing FR-3 (O3), but this difference from the issue was never called out to the owner.

Owner-approved reductions (not mismatches; recorded for the reviewer):
  - Hardening dropped to #114 per `spec.md` §2, including the raw telemetry-run JSONL export ("exportable with the run's JSONL"); a per-profile refinement export (FR-15) ships instead. Owner chose the minimal plan 2026-11-03.
  - "Base profile version" is recorded as the prior `system_prompts["rlm"]` value plus sha256 instead of a revision counter (§2).
  - Source runs are RLM-only and need `runs.profile_id` (owner, 2026-11-02 and 2026-11-03).
  - Documentation (issue component area) is Phase F, in progress.

Verdict: partial
Confidence: high
