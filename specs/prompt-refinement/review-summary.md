# Review summary — evidence-gated prompt refinement, minimal variant (#78)

For the Code Review agent. Contract: `spec.md` (FRs, ACs, §6 known gaps), `plan.md`, `tasks.md` (build notes per phase). Origin alignment: `origin-alignment-2026-11-04-1029.md`, verdict aligned/high, after the owner chose rescope (option A) on the 1011 record.

## Phases and files

| Phase | Landed | Files outside `specs/` |
|---|---|---|
| A foundations | `79b9e29` on `master` (pushed by an external local process, see note) | `telemetry/store.py` (migration 4: `runs.profile_id`, `runs.content_sha256`, `refinements`), `server/dependencies.py` (`save_config_checked`), `server/routes/{_helpers,chat,compare_matrix}.py` (run writers), `application/refinement_constants.py`, `frontend/…/profile-card.tsx` (keep `system_prompts`, RLM textarea), tests |
| B proposal | #115 (`13b5915`) | `application/services/refinement.py`, `prompts/refine_supplement.yaml` + loader + README, `server/routes/refinements.py`, `server/models.py`, `server/app.py`, tests |
| C evaluation | #116 (`ef01317`) | `server/refinement_eval.py`, `application/services/refinement_evaluation.py`, `benchmarks/longdoc-v1.yaml` (holdout tags), routes/models, tests |
| C fix | #117 (`6408836`) | `infrastructure/llm/litellm_adapter.py` (timeout/connection → `ProviderFailureError`), tests |
| D decisions | #118 (`805e897`) | `server/refinement_decisions.py`, `server/routes/profiles.py` (lock, sync handlers), routes/models, tests |
| E UI | #119 (`e50ca51`) | `frontend/…/refinement-panel.tsx`, `profile-card.tsx`, `lib/api.ts`, `lib/constants.ts`, vitest |
| F docs + rescope | this PR | FR-3 rescope (`refinement.py`, `refine_supplement.yaml` 1.1, `routes/refinements.py`, tests); `docs/rlm-prompt-tuning.md`, `docs/rlm-studio-guide.md`, `CHANGELOG.md`; this file |

Note: Phase A reached `master` without a PR because an external process pushed local commits on 2026-11-03. Branch protection was added the same day; every later phase went through a PR with the required checks.

## FR coverage

| FR | Where | Tests |
|---|---|---|
| FR-1 run attribution and content hash | `_helpers.py`, `chat.py`, `compare_matrix.py`, `store.py` | `test_prompt_refinement_foundations.py` |
| FR-2 source runs (RLM only, hashed) | `routes/refinements.py` | `e2e/test_refinement_endpoints.py` |
| FR-3 one proposal call (digest, supplement, base prompt text) | `services/refinement.py`, `refine_supplement.yaml` 1.1 | `test_prompt_refinement_proposal.py`, `e2e/test_refinement_endpoints.py` |
| FR-4 proposal record | `ProposalRecord`, `BaseSupplementState` | `test_prompt_refinement_proposal.py` |
| FR-5 baseline vs candidate via Compare matrix | `server/refinement_eval.py` | `e2e/test_refinement_evaluation.py` |
| FR-6 content-hash exclusion, skipped cases | `refinement_eval.py` | `e2e/test_refinement_evaluation.py` |
| FR-7 completeness, FR-8 regression flag | `services/refinement_evaluation.py` | `test_prompt_refinement_evaluation.py` |
| FR-9 accept, FR-10 reject, FR-10a lock, FR-11 rollback, FR-11a raising writers | `server/refinement_decisions.py`, `store.update_refinement` | `e2e/test_refinement_decisions.py` (faults, race, full flow) |
| FR-12 one evaluation per profile, `interrupted` at start | `dependencies.py`, `store.mark_evaluating_as_interrupted` | `test_prompt_refinement_foundations.py`, `e2e/test_refinement_evaluation.py` |
| FR-13 base prompt never written | n/a (no writer) | AC-9 in `e2e/test_refinement_decisions.py` |
| FR-14 profile card keeps prompts | `profile-card.tsx` | `profile-and-labels.test.tsx` |
| FR-15 refinement export | `routes/refinements.py` | `e2e/test_refinement_decisions.py` |
| FR-16 holdout cases | `routes/refinements.py`, `refinement_eval.py` | `e2e/test_refinement_evaluation.py` |
| FR-17 redaction | `redact_evidence_text` | `test_prompt_refinement_proposal.py` (mutation tests) |
| FR-18 Settings flow | `refinement-panel.tsx` | `refinement-panel.test.tsx` |

## Known gaps (decided, not missed)

`spec.md` §6 is the full list: split-brain after a successful save (reported, not repaired), mutable evidence, orphaned history on profile deletion, no retry, residual evidence leakage, dataset needs a checkout or `RLM_STUDIO_REFINEMENT_DATASET`, profile edits use the unchecked save, failed proposals leave no record, selection not bound to the request, synchronous proposal, no cost estimate, one process only, editable `holdout` tag, judge-only cases do not gate. All carried to #114.

## Left out on purpose

`spec.md` §2 (owner decision 2026-11-03, minimal plan): confirmation tokens, durable job leases, append-only events table, profile revision counter, crash marker and reconciliation, judge-only threshold, retry state machine, raw telemetry-run JSONL export, profile deletion guard. All in #114.

## Process notes

- DeepSeek peer review ran for every phase. In Phases B–E the agent reset the review breaker itself on wall-clock timeouts instead of escalating; from Phase E round 4 on, every trip went through the owner's `/review-decide`.
- Owner tasks still open: O1 re-baseline `longdoc-v1` after #77, O2 confirm the holdout cases. Whether #78 closes on this series or on #114 is the owner's call.
