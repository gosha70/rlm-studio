---
feature_id: prompt-refinement
spec_mode: full
status: draft
revision: 5-minimal
date: 2026-11-03
origin:
  urls:
    - https://github.com/gosha70/rlm-studio/issues/78   # evidence-gated prompt refinement
    - https://github.com/gosha70/rlm-studio/issues/77   # blocking epic (closed; items 1–3 merged, PR #113 on 2026-11-03)
  transcripts:
    - "Owner, 2026-11-02: 'research, plan and propose options (with details, PROS and CONS) for issue #78'."
    - "Owner, 2026-11-03: 'I want quick, simple and safe implementation.'"
    - "Owner, 2026-11-03: 'Let's compromise: build the minimal plan so I can use it to evaluate the release; after we confirm it, file a new enhancement issue with the more complex approach plus anything new we find but do not include.'"
  origin_claim: |
    Studio records every run's trace, tokens, cost and outcome, and has a judge
    and a Compare matrix, but nothing turns that evidence into a proposed
    profile supplement and nothing checks a proposal against an unchanged
    baseline before it is saved. Build proposal -> held-out baseline/candidate
    evaluation -> explicit accept/reject/rollback with the evidence recorded.
    Prompt supplements only. Minimal variant for the 1.x release evaluation;
    the hardened variant is deferred to a follow-up issue.
spec_mode_justification: >
  Writes to profile persistence, adds a telemetry migration, runs paid
  provider calls, and must keep three invariants (no auto-apply, held-out
  cases, immutable base prompt). Full mode is kept for the invariants; the
  implementation is the minimal one the owner chose on 2026-11-03.
---

# Spec — Evidence-gated prompt refinement, minimal variant (#78)

## 1. Goal

Let a user turn selected degraded or failed runs of a profile into a proposed
RLM prompt supplement, compare that candidate against the unchanged current
supplement on held-out benchmark cases, and accept, reject or roll back the
change explicitly, with the evidence recorded.

## 2. What "minimal" means (owner decision, 2026-11-03)

Kept, because each maps to a verified failure mode and is cheap:

- No profile write on propose, evaluate or reject. Only accept and rollback
  write, and only on an explicit user action.
- Accept and rollback run under `AppState._config_lock`, re-check that the
  supplement is still the one the proposal was built on (content hash), and
  use a checked config save that raises on failure. Today
  `_save_config_locked()` logs and returns on failure
  (`server/dependencies.py:1304–1305`).
- Rollback restores the exact prior state of `system_prompts["rlm"]`,
  including removing the key when it was absent, so a template-backed
  profile resumes following its template.
- Held-out cases are checked by document content hash against the source
  runs, not by query text. This needs `runs.content_sha256`.
- Source runs are attributed by `runs.profile_id`. Runs without it (every run
  recorded before this feature) are not selectable. Strict legacy exclusion
  was the owner's call on 2026-11-02.
- The ProfileCard bug that sends `system_prompts: {}` on every save
  (`frontend/src/components/settings/profile-card.tsx:246`) is fixed first,
  or an accepted candidate disappears on the next unrelated edit.

Dropped from the revision-4 plan (`plan-full-rev4.md`) and deferred to a
follow-up issue:

- Single-use preflight confirmation tokens.
- Durable SQLite job leases, heartbeat threads, cross-process cancel.
- Append-only `refinement_events` table with triggers and sequence CAS.
- Profile `revision` counter (hash comparison replaces it).
- `pending_refinement_applies` crash marker and reconciliation.
- Judge-only regression threshold and judge-gap acknowledgements.
- Retry state machine (`evaluation_incomplete`, `apply_failed`, …).
- Raw telemetry-run JSONL export endpoint (`GET /api/traces/{run_id}/export.jsonl`
  with refinement events in its metadata line). The refinement-record export
  (`GET /api/profiles/{id}/refinements/export.jsonl`, FR-15) stays in scope;
  it is a different mechanism.
- Profile deletion guard.

## 3. User scenarios

1. **Propose.** On a profile's card in Settings → Profiles, the user opens
   "Refine prompt", sees that profile's recent runs with their outcome
   category, ticks the ones to learn from, picks the proposal Chat Provider
   (defaulted from the newest ticked run), and clicks Propose. One model
   call returns a candidate supplement and a one-paragraph rationale. The
   UI shows the unified diff against the current supplement.
2. **Evaluate.** The user sees the held-out case list (default: the cases
   tagged `holdout` in `benchmarks/longdoc-v1.yaml`), the repetitions
   (default 2, minimum 2), the run count (cases × reps × 2 arms), and the
   provider that will be charged. On confirm, evaluation runs in the
   background and the panel polls progress. When done it shows, per case and
   per rep, pass/fail for both arms, plus deltas: deterministic accuracy,
   judge score (mean, min, max) when a judge is configured, cost, steps,
   outcome-class counts.
3. **Decide.** If any held-out case regressed, the panel defaults to Reject
   and requires a checkbox to accept anyway. Accept writes the candidate to
   `system_prompts["rlm"]` and records the decision; Reject records the
   decision and writes nothing.
4. **History and rollback.** The card lists past refinements with status,
   dates and deltas. An accepted refinement whose text is still in place
   offers Rollback, which restores the recorded prior state.
5. **Export.** `GET /api/profiles/{id}/refinements/export.jsonl` returns one
   line per refinement with its full stored evidence (FR-15). This is the
   refinement-record export; the raw telemetry-run JSONL export is deferred.

## 4. Functional requirements

| ID | Requirement |
|----|-------------|
| FR-1 | Every telemetry run writer records `profile_id` (from the Chat Provider) and `content_sha256` of the resolved document. Writers: `server/routes/chat.py:575`, `server/routes/compare_matrix.py:858`. |
| FR-2 | `GET /api/profiles/{id}/refinements/source-runs` returns only runs with `profile_id = id` and non-null `content_sha256`, newest first, with `outcome_category`, `degraded` (answer starts with the warning prefix), `chat_provider_id`, cost, tokens, created_at. |
| FR-3 | Proposal uses one `LLMPort.complete()` call with the versioned `prompts/refine_supplement.yaml`. Input: evidence digest (capped at 12 traces and 24,000 chars, redacted per FR-17, evidence framed as data not instructions), current resolved supplement, base prompt version. Output JSON `{candidate, rationale}`; the server computes the unified diff with `difflib`. Candidate must be non-empty, differ from the current supplement, and be at most 8,000 chars. |
| FR-4 | The proposal record stores: source run ids, the exact digest sent and its sha256, truncation info, proposal Chat Provider id, provider, model, prompt version, usage, base supplement resolved text and sha256, whether `system_prompts["rlm"]` existed and its exact prior value, template name, base prompt version. |
| FR-5 | Evaluation runs baseline and candidate as two slots per case × rep through `RunMatrixComparisonUseCase`, same Chat Provider, separate adapter instances, temperature 0, configs differing only in `system_prompt_extra`. Each arm is recorded as a normal telemetry run (reusing the compare route's slot recorder) and its run id is stored. |
| FR-6 | A case whose materialised content sha256 equals any source run's `content_sha256` is excluded and listed in the record. A case that cannot be materialised offline is listed as skipped. |
| FR-7 | Evaluation is `complete` only when every selected, non-excluded case × rep produced both arms. Deltas and the regression flag are computed only from a complete evaluation. |
| FR-8 | Regression flag: a deterministic case (one with `expected_answer`) where the candidate pass rate across reps is lower than the baseline's. Judge scores are reported when a judge Chat Provider is configured and never gate. |
| FR-9 | Accept requires status `evaluated`, `complete=true`, at least one deterministic case, and `acknowledge_regression=true` when the flag is set. It runs under `_config_lock`: re-resolve the supplement, compare sha256 with the proposal's, 409 on mismatch, set `system_prompts["rlm"]`, checked save, restore the in-memory profile on failure and return 500 without changing status. |
| FR-10 | Reject is allowed from `proposal_ready`, `evaluated`, `interrupted`, `failed`. Writes nothing to the profile. |
| FR-10a | Accept, Reject and Rollback all hold `AppState._config_lock` from their status precondition read through their decision write, so decisions on one refinement are serialized in-process. The status CAS in `update_refinement` stays as defence in depth. A Reject that waits on an in-flight Accept fails afterwards with 409 because the status is already `accepted`. |
| FR-11 | Rollback requires status `accepted` and that `system_prompts["rlm"]` still equals the candidate (sha256); it restores the recorded prior presence and value under `_config_lock` with a checked save. On save failure it restores the candidate in the in-memory profile, keeps status `accepted`, and returns 500. |
| FR-11a | Refinement status and decision writes (`create_refinement`, `update_refinement`) propagate database failures to the caller. They do not follow the log-and-continue pattern of the existing telemetry writers (`telemetry/store.py:341`, `:418`, `:462`, `:485`). Accept and rollback detect a failed status write after a successful config save and return 500 naming the split-brain case. |
| FR-12 | At most one running evaluation per profile, enforced in-process. On server start, rows left in `evaluating` become `interrupted`. |
| FR-13 | The base prompt file `prompts/system_prompt_v2_0.yaml` is never written. |
| FR-14 | ProfileCard save, clone and export preserve `system_prompts`; the card gets a textarea for the custom RLM prompt. |
| FR-15 | `GET /api/profiles/{id}/refinements/export.jsonl` returns one JSON line per refinement of that profile with the stored proposal, evaluation and decision records verbatim. |
| FR-16 | `GET /api/profiles/{id}/refinements/holdout-cases` returns, for each case in `benchmarks/longdoc-v1.yaml`, its id, tags, size bucket, whether it has an `expected_answer`, and whether it materialises offline; cases tagged `holdout` are the default selection. |
| FR-17 | Evidence redaction policy: the digest passes each query, answer and error through `redact_secrets()` (API-key and bearer-token shapes) and additionally drops any line that matches, case-insensitively and with optional whitespace around the separator, one of the names `key`, `api_key`, `apikey`, `token`, `password`, `passwd`, `secret`, `authorization` followed by `=`, `:`, or a JSON-style `":`. The policy is documented in the proposal prompt's README entry and covered by mutation tests. It does not claim to catch arbitrary secrets (see §6, residual evidence leakage). |
| FR-18 | Settings → Profiles exposes the refinement flow for user profiles: source-run selection, proposal provider choice, diff and rationale, holdout selection with run count, progress, per-case results with the regression flag defaulting to Reject, accept acknowledgement, history with rollback, and the current resolved supplement shown next to the candidate. |

## 5. Acceptance criteria (hard)

1. Propose, evaluate and reject leave `config.json` byte-identical.
2. Accept against a profile whose supplement changed after the proposal
   returns 409 and writes nothing.
3. An injected failure in write, fsync or replace during accept returns 500,
   leaves the in-memory profile at its prior text, and leaves the refinement
   in `evaluated`. The same three faults during rollback return 500, leave
   the in-memory profile at the candidate text, and leave the refinement in
   `accepted`.
3a. An injected status-write failure after a successful accept save returns
   500 naming the case, the profile on disk and in memory holds the
   candidate, and the row stays `evaluated`. The same after a successful
   rollback save: the profile holds the prior text, the row stays
   `accepted`.
4. Accept on a template-backed profile records `base_present=false`;
   rollback removes the key and the template text is resolved again.
5. A candidate is never evaluated on a case whose content hash matches a
   source run; the exclusion is in the stored record.
6. A zero-case, skipped-case, or one-arm evaluation is not `complete` and
   cannot be accepted.
7. Every accepted or rejected refinement stores source run ids, case ids,
   baseline and candidate run ids per case and rep, judge model and prompt
   version when used, deltas, the regression flag, the acknowledgement, and
   the decision time.
8. No candidate is applied without `POST .../accept`.
8a. A Reject started while an Accept is between its config save and its
   status write waits on the lock and then returns 409; the row ends
   `accepted` and the candidate is live.
9. The base prompt YAML's bytes are unchanged after the end-to-end test.
10. ProfileCard save no longer sends `system_prompts: {}`.

## 6. Known gaps accepted for the minimal variant

These are stated so the Code Review agent and the release evaluation treat
them as decided, not missed. Each is carried into the follow-up issue. The
list is the complete set of consequences of the dropped mechanisms in §2;
anything found during the build that is not here is either fixed or added
here before the phase closes.

- **Split-brain window on accept and rollback.** After a successful checked
  config save, the status write to the refinements table can fail or the
  process can die before it runs. Four cases are accepted: accept × crash,
  accept × status-write failure, rollback × crash, rollback × status-write
  failure. The config state is authoritative; the row stays at `evaluated`
  (accept) or `accepted` (rollback). The route returns 500 with a message
  naming the case when the status write fails. The UI shows the current
  resolved supplement next to the candidate so the user can see which text
  is live; there is no automatic reconciliation.
- **Mutable evidence.** Proposal, evaluation and decision records are JSON
  columns on one mutable row. Nothing in the API rewrites them after the
  state they belong to, but there is no database-level append-only guarantee.
- **Orphaned history on profile deletion.** Deleting a profile leaves its
  refinement rows in place with a `profile_id` that no longer resolves. They
  remain in the export and are not shown in the UI.
- **No retry for interrupted or failed work.** `interrupted` and `failed`
  refinements can only be rejected; the user starts a new refinement.
- **Residual evidence leakage.** FR-17 redacts known key shapes and named
  key-value lines only. A secret in free text, in an unusual key name, or
  split across lines reaches the proposal provider. The user chooses the
  source runs and the proposal provider, and the digest is stored verbatim
  in the proposal record, so what was sent is auditable. Stronger structured
  redaction goes to the follow-up issue.
- **Selection is not bound to the request.** The run count and case list the
  UI shows before evaluation are computed client-side from
  `holdout-cases`; the evaluate request carries its own case ids and reps and
  the server trusts them. There is no server-issued confirmation token.
- The proposal endpoint is synchronous. A slow local model on a large digest
  can exceed the frontend proxy timeout. The digest cap bounds the input.
- No dollar cost estimate before evaluation; the UI shows the run count and
  the provider. Pricing is reported after the fact from telemetry.
- One process only. Two server processes sharing the database can both start
  an evaluation for the same profile, and their Accept, Reject and Rollback
  decisions are not coordinated with each other; only the status CAS
  separates them, so a cross-process race can end in the split-brain state
  above with the row at a status other than the one listed.
- The `holdout` tag lives in the benchmark YAML and can be edited.
- Judge-only holdout cases have no gate semantics; the default holdout is
  deterministic-only.

## 7. Non-goals

Memories, skills, subagent specs; auto-apply; evaluating on source traces;
hard-blocking a regression; Direct and RAG profile prompts (not injected
today); a second `system_prompt_extra` field (option B2); multi-candidate
search.
