---
feature_id: prompt-refinement
status: draft
revision: 5-minimal
date: 2026-11-03
---

# Tasks — prompt refinement, minimal variant (#78)

Owners follow the CLAUDE.md agent roles. One PR per phase, six PRs, same
sequence as `plan.md` §5. Tick a task only when its tests pass. Every task
carries one of: an FR from `spec.md` §4, an acceptance criterion (AC-n from
§5), a known gap (KG from §6), or the marker PROCESS for work that exists
only to keep the codebase conventions or ship the feature.

## Phase A — foundations (Runtime + API + Frontend)

- [x] A1 Telemetry migration 4: `runs.profile_id`, `runs.content_sha256`, index, `refinements` table (`telemetry/store.py` `_MIGRATIONS[4]`, bump `_CURRENT_SCHEMA_VERSION`). FR-1, FR-4. Tests: fresh, v3→v4, idempotent.
- [x] A2 Store API: `record_run` new kwargs; `list_runs(profile_id=, require_content_hash=)`; `RunSummary.profile_id`, `.content_sha256`; refinement CRUD with raising writers and status CAS; `mark_evaluating_as_interrupted()`. FR-2, FR-11a, FR-12. Tests: injected `sqlite3.Error` and CAS miss raise.
- [x] A3 Status constants module; no literals elsewhere. PROCESS (CLAUDE.md "no magic strings").
- [x] A4 Writers pass `profile_id` and `content_sha256` in `chat.py` `_record_telemetry` (HTTP and WebSocket share it) and `compare_matrix.py:858`. FR-1.
- [x] A5 Move `_record_slot_telemetry` and `_SlotMeta` to `routes/_helpers.py`; compare route imports them. FR-5 (the evaluation service needs the recorder without importing a route module). Existing compare tests stay green.
- [x] A6 `AppState.save_config_checked()` (`dependencies.py`); fault-injection tests for write, fsync, replace. FR-9, FR-11, AC-3.
- [x] A7 ProfileCard: save preserves and edits `system_prompts["rlm"]`; clone and export include `system_prompts`; vitest for the save payload. FR-14, AC-10.

## Phase B — proposal (Runtime + API)

- [x] B1 `prompts/refine_supplement.yaml` v1.0 with version read explicitly (pattern: `benchmark/scoring.py:138`); README entry documents the FR-17 redaction policy. FR-3, FR-17.
- [x] B2 `application/services/refinement.py`: `build_evidence_digest` with `redact_secrets()` plus the FR-17 name × separator line-drop rule, mutation tests per plan §3.4; `propose_supplement`, JSON parsing, validation, `difflib` diff. FR-3, FR-4, FR-17, KG residual leakage. Unit tests with a fake `LLMPort`.
- [x] B3 Pydantic models in `server/models.py`: source run row, create request, refinement response, evaluate/accept/reject requests, holdout case row. FR-2, FR-3, FR-16 (CLAUDE.md: DTOs validated at the boundary).
- [x] B4 Routes: `source-runs`, `POST /refinements` (sync, `asyncio.to_thread`), list, get. Built-in profile → 400. Register in `app.py`. FR-2, FR-3, FR-4, AC-1. `TestClient` tests; `tests/e2e/test_api_endpoints.py` mirror.

## Phase C — evaluation (Runtime + API + QA)

- [x] C1 Tag the six holdout cases in `benchmarks/longdoc-v1.yaml`; `test_benchmark_dataset.py` asserts the tag set. FR-16 (default selection), plan §2.
- [x] C2 `GET .../holdout-cases`. FR-16.
- [x] C3 `server/services/refinement_eval.py`: materialise, source-hash exclusion, two-slot matrix per case × rep with separate adapters, temperature 0, case budgets, slot recording with run ids, `matches` scoring, optional `JudgeScorer`, aggregation, `complete` flag, regression flag. FR-5, FR-6, FR-7, FR-8, AC-5, AC-6. Tests with `ScriptedLLM`.
- [x] C4 `POST .../evaluate` → 202; background thread; per-profile single-job guard; progress dict in `AppState`; `reset_state()` clears it. FR-12, AC-1.
- [x] C5 Startup marks `evaluating` rows `interrupted`. FR-12.

## Phase D — decisions (API)

- [x] D1 Accept under `_config_lock` with sha256 check, checked save, in-memory restore on failure, decision record; 500 naming the split-brain case when the status write fails; lock held from precondition read through status write. FR-9, FR-10a, FR-11a, AC-2, AC-3, AC-3a, AC-4, AC-8, KG split-brain. Tests: 409 on drift; write, fsync, replace faults → 500 with memory restored and status unchanged; status-write fault after save → 500 with config kept; template profile records `base_present=false`.
- [x] D2 Reject from allowed statuses under `_config_lock`; writes nothing to the profile. FR-10, FR-10a, AC-1, AC-8a. Race test: Accept paused between save and status write, Reject waits then returns 409 with the row `accepted`.
- [x] D3 Rollback: candidate still in place check, restore prior presence/value, checked save, restore the candidate in memory on save failure with status kept `accepted`; 500 naming the split-brain case when the status write fails; lock held from precondition read through status write. FR-10a, FR-11, FR-11a, AC-3, AC-3a, AC-4, KG split-brain. Tests: write, fsync, replace faults → 500 with candidate still in memory and status `accepted`; status-write fault after save → 500 with prior text on disk; key removed and template text resolved again on success.
- [x] D4 `GET .../export.jsonl`. FR-15, AC-7.
- [x] D5 End-to-end fake-provider test: propose → evaluate → accept → chat run carries candidate → rollback; base prompt YAML bytes unchanged. FR-13, AC-1 to AC-10.

## Phase E — UI (Frontend)

- [ ] E1 `lib/api.ts` functions and types for every route. FR-18 (CLAUDE.md: use the shared API client).
- [ ] E2 `refinement-panel.tsx`: source-run table, proposal CP select defaulted from newest selected run, Propose, diff + rationale, holdout list + reps + run-count confirm, progress polling, results table (per case/rep/arm, deltas, regression flag, default Reject), accept checkbox when flagged, history with Rollback, current-supplement display next to the candidate. FR-18, KG split-brain, KG selection-not-bound.
- [ ] E3 "Refine prompt" button on `profile-card.tsx` (user profiles only). FR-18.
- [ ] E4 Vitest: panel states (empty sources, proposal ready, evaluating, evaluated with and without regression, accepted, rolled back). FR-18 (CLAUDE.md: vitest for new components).

## Phase F — docs and closure (Team Lead + QA)

- [ ] F1 `docs/rlm-prompt-tuning.md` section; `docs/rlm-studio-guide.md` Profiles paragraph; `CHANGELOG.md`. PROCESS (issue #78 lists documentation as a component area).
- [ ] F2 Full backend suite and frontend suite green; lint 0. PROCESS (coding standards quality gates).
- [ ] F3 Review summary written for the Code Review agent: files changed per phase, FR coverage table, known gaps from `spec.md` §6, and what was left out on purpose. PROCESS.
- [x] F4 File the follow-up enhancement issue (plan §6) and link it from #78. PROCESS. Done 2026-11-03: #114 filed, linked from #78. Whether #78 closes on this PR series or on #114 is still the owner's call.

## Build notes

- A1: each migration version runs in one explicit transaction with its `user_version` bump, because Python's sqlite3 does not open a transaction before DDL. Regression test fails migration 4 after its first statement and checks rollback plus a clean retry.
- A3: statuses live in `application/refinement_constants.py`; the telemetry store imports them (outer → inner, allowed).
- A4: `_record_telemetry` takes `profile_id`/`content` as optional keywords so the seven existing direct test calls stay valid; both production callers pass them.
- A6 (found in Phase B): `reset_state()` stubbed `save_config` but not `save_config_checked`, so a test reaching the checked save through the test state would write the user's real config. Stubbed in Phase B.
- B2: the FR-17 line rule matches any identifier *ending* in a listed name (`aws_secret_key=`), which is stricter than the literal names; FR-17 wording updated.
- B3: only the models Phase B routes use were added (source run row, create request, refinement response). The evaluate, accept, reject and holdout-case models land with their routes in C and D, so nothing unused ships.
- B2 (review fix): each evidence field is redacted whole and then clipped; clipping first cut a key straddling the limit into a fragment the patterns could not match.
- B4 (spec change, owner 2026-11-03): source runs are `mode = rlm` only, in both the list and proposal creation; FR-2 updated.
- B4: a failed or unusable proposal returns 502 and stores nothing; recorded as a known gap in spec §6.
- C3: the service lives at `server/refinement_eval.py`, beside `judge.py` and `quality.py`; there is no `server/services/` package. Pure summary logic (completeness, regression, deltas) is in `application/services/refinement_evaluation.py`.
- C3: temperature 0 is applied through a new optional `temperature` argument on `AppState.create_llm_adapter_for_chat_provider`; `RunConfigDTO.temperature` does not reach the RLM loop.
- C3: evaluation arm runs are recorded with `profile_id = None`. They are not runs of the profile's real prompt, so they must never come back as refinement sources (FR-1 nuance).
- C3: an arm passes only if its outcome is usable and its answer matches, and the judge scores only usable answers, the same rules as `rlm-studio bench`.
- C4: the evaluation Chat Provider defaults to the proposal's and must use this profile (400 otherwise); the judge defaults to the configured judge Chat Provider, and none runs when neither is set.
- C4: the dataset path is `RLM_STUDIO_REFINEMENT_DATASET`, else the checkout's `benchmarks/longdoc-v1.yaml`; known gap added to spec §6.
- C1: the six holdout cases are provisional until owner task O2.
- C3 (review fix): an arm whose run has `failure_kind` set (auth, model unavailable, connection, request timeout, rate limit) is a missing arm with a reason naming the kind and run id, so the evaluation is incomplete instead of scoring a provider failure as a wrong answer. A provider request timeout counts too, so a slow local model on the large case can leave an evaluation incomplete.
- C3 (review fix 2): before this, `LiteLLMAdapter._translate_exception` returned request timeouts and connection errors as a plain `RuntimeError`, so their runs had `failure_kind = None` and were still scored as wrong answers. The adapter now raises `ProviderFailureError` with the `timeout` / `connection` kind and the exact same message, so the use cases' `TIMEOUT_ERROR_PREFIX` checks are unchanged. Side effect outside #78: Chat results for these failures now carry `failure_kind` too (the frontend already types both values; no component reads them yet). An RLM run that exhausts its own `max_time_seconds` raises nothing at the adapter, keeps `failure_kind = None`, and stays a scored outcome (tested).
- D1-D3: decisions live in `server/refinement_decisions.py` as synchronous functions holding `_config_lock` from the status check through the decision write; the routes run them with `asyncio.to_thread` so a held lock never blocks the event loop. Errors carry their HTTP status (400 / 404 / 409 / 500).
- D1-D3 (found in Phase D): `PUT` and `DELETE /api/profiles/{id}` changed the profile without `_config_lock`, so the lock did not serialize an edit against an accept or rollback; both now mutate and save under it.
- D1-D3: the decision record is one typed `DecisionRecord` (decision, time, reason, regression, acknowledgement); a rollback adds a `RollbackRecord` to the accepted decision instead of replacing it.
- D1-D3 (review fix): `update_profile` and `delete_profile` are plain `def` handlers, so their wait on `_config_lock` runs in FastAPI's thread pool instead of blocking the event loop while an accept or rollback holds the lock across a disk write. Tested with an unrelated `/health` request during a blocked edit.
- D4 (review fix): export serves a deleted profile's refinement rows (spec §6) and returns 404 only when there is neither a profile nor any refinement.
- D5 (review fix): the full-flow test runs real `/api/chat` RLM requests after accept and after rollback and checks the system prompt the model received.
- D2 (review fix): reject returns 409 when the profile's RLM prompt already equals the candidate. After a split-brain accept the row stays `evaluated` while the candidate is live, and recording `rejected` would contradict the profile with no API path back.
- E2 (carried from review): (a) progress can be `{done: 0, total: 0}` when every case is excluded, so never divide by `total`; (b) after an accept or rollback the panel must refresh or close the card's edit form, because a ProfileCard save sends the RLM prompt captured when editing started and would overwrite the decision's result.
- D4: `GET …/refinements/export.jsonl` serves `application/x-ndjson` with one `RefinementRecord` per line, newest first.
- A7: `afd2db3` (2026-04-07) removed per-mode prompt editors in favour of prompt groups. FR-14 re-adds one RLM textarea; the old "no per-mode textareas" test now asserts only RLM is editable. Owner decision 2026-11-03: keep the single RLM textarea.

## Owner tasks (human)

- [ ] O1 Re-baseline `longdoc-v1` on the post-#77 loop before trusting evaluation numbers.
- [ ] O2 Confirm or swap the six holdout cases after O1.
- [x] O3 Confirm this minimal plan before Phase A starts. Confirmed 2026-11-03, recorded in #78.
