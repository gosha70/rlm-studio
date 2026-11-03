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

- [ ] B1 `prompts/refine_supplement.yaml` v1.0 with version read explicitly (pattern: `benchmark/scoring.py:138`); README entry documents the FR-17 redaction policy. FR-3, FR-17.
- [ ] B2 `application/services/refinement.py`: `build_evidence_digest` with `redact_secrets()` plus the FR-17 name × separator line-drop rule, mutation tests per plan §3.4; `propose_supplement`, JSON parsing, validation, `difflib` diff. FR-3, FR-4, FR-17, KG residual leakage. Unit tests with a fake `LLMPort`.
- [ ] B3 Pydantic models in `server/models.py`: source run row, create request, refinement response, evaluate/accept/reject requests, holdout case row. FR-2, FR-3, FR-16 (CLAUDE.md: DTOs validated at the boundary).
- [ ] B4 Routes: `source-runs`, `POST /refinements` (sync, `asyncio.to_thread`), list, get. Built-in profile → 400. Register in `app.py`. FR-2, FR-3, FR-4, AC-1. `TestClient` tests; `tests/e2e/test_api_endpoints.py` mirror.

## Phase C — evaluation (Runtime + API + QA)

- [ ] C1 Tag the six holdout cases in `benchmarks/longdoc-v1.yaml`; `test_benchmark_dataset.py` asserts the tag set. FR-16 (default selection), plan §2.
- [ ] C2 `GET .../holdout-cases`. FR-16.
- [ ] C3 `server/services/refinement_eval.py`: materialise, source-hash exclusion, two-slot matrix per case × rep with separate adapters, temperature 0, case budgets, slot recording with run ids, `matches` scoring, optional `JudgeScorer`, aggregation, `complete` flag, regression flag. FR-5, FR-6, FR-7, FR-8, AC-5, AC-6. Tests with `ScriptedLLM`.
- [ ] C4 `POST .../evaluate` → 202; background thread; per-profile single-job guard; progress dict in `AppState`; `reset_state()` clears it. FR-12, AC-1.
- [ ] C5 Startup marks `evaluating` rows `interrupted`. FR-12.

## Phase D — decisions (API)

- [ ] D1 Accept under `_config_lock` with sha256 check, checked save, in-memory restore on failure, decision record; 500 naming the split-brain case when the status write fails; lock held from precondition read through status write. FR-9, FR-10a, FR-11a, AC-2, AC-3, AC-3a, AC-4, AC-8, KG split-brain. Tests: 409 on drift; write, fsync, replace faults → 500 with memory restored and status unchanged; status-write fault after save → 500 with config kept; template profile records `base_present=false`.
- [ ] D2 Reject from allowed statuses under `_config_lock`; writes nothing to the profile. FR-10, FR-10a, AC-1, AC-8a. Race test: Accept paused between save and status write, Reject waits then returns 409 with the row `accepted`.
- [ ] D3 Rollback: candidate still in place check, restore prior presence/value, checked save, restore the candidate in memory on save failure with status kept `accepted`; 500 naming the split-brain case when the status write fails; lock held from precondition read through status write. FR-10a, FR-11, FR-11a, AC-3, AC-3a, AC-4, KG split-brain. Tests: write, fsync, replace faults → 500 with candidate still in memory and status `accepted`; status-write fault after save → 500 with prior text on disk; key removed and template text resolved again on success.
- [ ] D4 `GET .../export.jsonl`. FR-15, AC-7.
- [ ] D5 End-to-end fake-provider test: propose → evaluate → accept → chat run carries candidate → rollback; base prompt YAML bytes unchanged. FR-13, AC-1 to AC-10.

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
- A7: `afd2db3` (2026-04-07) removed per-mode prompt editors in favour of prompt groups. FR-14 re-adds one RLM textarea; the old "no per-mode textareas" test now asserts only RLM is editable. Owner decision 2026-11-03: keep the single RLM textarea.

## Owner tasks (human)

- [ ] O1 Re-baseline `longdoc-v1` on the post-#77 loop before trusting evaluation numbers.
- [ ] O2 Confirm or swap the six holdout cases after O1.
- [x] O3 Confirm this minimal plan before Phase A starts. Confirmed 2026-11-03, recorded in #78.
