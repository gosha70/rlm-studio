---
feature_id: prompt-refinement
spec_mode: full
status: blocked
revision: 4
date: 2026-11-02
origin:
  urls:
    - https://github.com/gosha70/rlm-studio/issues/78   # evidence-gated prompt refinement
    - https://github.com/gosha70/rlm-studio/issues/77   # blocking epic: working state
    - https://github.com/PrimeIntellect-ai/prime-agent  # refinement design being borrowed from
    - https://arxiv.org/abs/2608.23552                   # Prime Agent paper
  transcripts:
    - "Owner, 2026-11-02: 'research, plan and propose options (with details, PROS and CONS) for issue #78'."
  origin_claim: |
    Studio records every run's trace, tokens, cost and outcome, and has a judge
    and a Compare matrix, but nothing turns that evidence into a proposed
    profile supplement (`system_prompt_extra`) and nothing checks a proposal
    against an unchanged baseline before it is saved. Build proposal ->
    held-out baseline/candidate evaluation -> explicit accept/reject/rollback,
    with the full evidence record persisted and exported. Prompt supplements
    only; memories, skills and subagent specs are non-goals.
spec_mode_justification: >
  Touches profile persistence and concurrency, a new telemetry table and
  run-attribution columns, a new proposal prompt, an evaluation orchestrator
  that runs paid provider calls, the Compare-matrix and benchmark scoring
  paths, a new async job lifecycle, REST endpoints, and a Settings workflow.
  Needs explicit invariants (no auto-apply, held-out cases, immutable base
  prompt) and deterministic tests before implementation.
---

# Plan — Evidence-gated prompt refinement for profile supplements (#78)

## 0. Status and sequencing gate

**Blocked, as the issue says.** `#77` item 1 (result registry + `read_result`)
landed in `3de7777`; items 2 (lossless compaction) and 3 (batched subcalls)
are still unchecked in `specs/rlm-working-state/tasks.md`, with no branches.
Do not start #78 implementation until both land.

Why the gate matters for *this* feature, not just project bookkeeping:

- The evaluation baseline must be measured on the post-#77 loop. Compaction
  changes when long documents trigger overflow and adds a `compaction` trace
  row; a baseline measured before it is not comparable to a candidate measured
  after it.
- Source-trace evidence extraction must understand the `compaction` action and
  the registry fields; building that parser twice is waste.

**Recommendation:** keep #78 blocked; treat this document as the design input.
Start Phase 0 the day #77 items 2–3 merge, and run one fresh
`longdoc-v1` baseline on the merged code before any refinement evaluation is
accepted.

## 0.1 Review findings resolved (revision 2)

| # | Finding | Resolution in this revision |
|---|---|---|
| 1 | `_save_config_locked()` swallows write/fsync/replace failures, so accept can report success without persisting | A checked profile-persistence operation that raises; accept restores the in-memory profile on failure and records `apply_failed`. Tests inject write/fsync/replace faults. §3 D1/D4, §4.2. |
| 2 | CAS has no atomic lock boundary | One `_config_lock` transaction covers re-read, revision/hash comparison, mutation, checked save, and rollback of the in-memory mutation. A race test edits the profile after accept's first read. §3 D1, §4.2. |
| 3 | Query-string leakage guard cannot prove held-outness | Runs carry `content_sha256` plus benchmark dataset/version/case-id provenance; evaluation cases are checked by content hash and provenance, not query text. Legacy runs without provenance are explicitly unverifiable; verified holdout is required for acceptance. §3 G1, §4.1, §4.2. |
| 4 | Rollback leaves a template-backed profile detached | The proposal records whether `system_prompts["rlm"]` existed and its exact prior value; rollback removes the key when it was absent, restoring template following. §3 D1, §4.1. |
| 5 | Partial/empty evaluation can reach `evaluated` | Evaluation is `evaluation_incomplete` unless every selected case × repetition yields both arms. Zero comparable cases, a missing arm, missing corpus material, or a failed repetition block acceptance; judge coverage is recorded separately. §3 J5, §4.2, §4.3. |
| 6 | The evidence digest sent to the proposer is not persisted | The proposal event stores the exact redacted digest, its SHA-256, truncation metadata, and the rendered proposal-prompt hash/version. §3 F-a, §4.1. |
| 7 | A shared Chat Provider cannot attribute a legacy run to one profile | Legacy runs remain explicitly unattributed; they cannot be used as verified sources, and no shared-CP inference is treated as ownership. §4.2. |
| 8 | “Immutable evidence” conflicts with one mutable row | Refinement evidence is an append-only `refinement_events` log; the `refinements` row is a status/lease projection. Terminal event payloads are never updated, and exports replay the decision-time event. The state machine includes `applying`, `apply_failed`, and `evaluation_incomplete`. §3 C1, §4.1, §4.3. |
| 9 | In-memory jobs and missing preflight confirmation | A durable SQLite lease (owner token, expiry) gates job ownership and every result write; preflight returns a cost/time estimate and a one-time confirmation token required by `evaluate`; shutdown/restart marks interrupted jobs. §3 I1, §4.2. |

## 0.2 Revision 3 review findings resolved

| # | Finding | Resolution in this revision |
|---|---|---|
| 1 | Confirmation tokens have no durable storage | New `refinement_preflights` table stores `token_hash`, canonical selection, estimate, expiry, and consumption; token consumption + refinement creation/lease acquisition happen in one `BEGIN IMMEDIATE` transaction. Tests cover concurrent use, replay, restart, expiry, and request mismatch. §4.1, §4.2. |
| 2 | Save-success/event-append-failure is ambiguous | `config.json` carries a `pending_refinement_applies[profile_id]` marker written atomically with the profile. Profile writes for that profile are blocked while the marker/`applying` state exists; reconciliation runs first under `_config_lock`, compares the persisted revision and target key presence/hash, appends `accepted` or `apply_failed`, and clears the marker. A save-success/event-failure/concurrent-edit test is required. §3 D4, §4.1, §4.2. |
| 3 | Proposal calls bypass paid-work confirmation | A proposal preflight returns the proposal provider/model, the exact redacted evidence digest and its hash, token estimates, and known/unknown maximum cost; it issues the single-use token consumed by `POST /refinements`. Evaluation preflight keeps its own token. Unknown price is an explicit `cost_known=false` confirmation state. §3 I1, §4.2. |
| 4 | Judge-only holdout regressions have no gate semantics | Judge-only cases get a documented `judge_regression_threshold` (default 0.5 on the 1–5 scale) and require `acknowledge_judge_regression=true`; a holdout containing judge-only cases requires a configured judge and `judge_complete=true`. The final six-case split is chosen after the #77 re-baseline. §3 J6, §4.2, §6. |
| 5 | Lease renewal during provider calls is unspecified | The job holds a heartbeat thread renewing the lease with owner-token CAS at `ttl/3`; the TTL is derived from the maximum provider request timeout plus margin. Renewal failure stops scheduling and result writes; stale workers cannot write. Expiry during an in-flight call is tested. §3 I1, §4.1. |
| 6 | Event replay lacks deterministic ordering | `refinement_events.sequence` is allocated inside the same transaction as the status CAS, with `UNIQUE(refinement_id, sequence)`; projection and export order by sequence, never `created_at`. The head row keeps `last_sequence`. §4.1, §4.3. |
| 7 | Retry transitions contradict the API | A complete transition table now covers proposal, evaluation retry from `evaluation_incomplete`/`interrupted`, cancel, reject, accept retry from `apply_failed`, reconciliation, rollback, and rollback failure; every transition is status-CAS enforced in the event-append transaction. §4.3. |

### Owner decisions (2026-11-02)

- **B1 chosen:** reuse `system_prompts["rlm"]`; record and restore the exact
  raw key state, including removing the key when it was absent.
- **Explicit proposal Chat Provider**, defaulted from the newest source run;
  proposal preflight shows provider, evidence, token limits, and cost/privacy
  implications.
- **Strict legacy exclusion:** source runs without `profile_id`/
  `content_sha256` are not selectable and do not make a verified holdout.
- **Holdout composition deferred:** select the six `longdoc-v1` cases after
  the #77 re-baseline and after the judge-only behavior in §3 J6 is accepted.
- **Full Settings wizard and history** are part of #78’s closure; source
  selection, proposal/evaluation cost confirmation, diff review, and
  acknowledgements are safety boundaries, not optional polish.

## 0.3 Revision 4 review findings resolved

| # | Finding | Resolution in this revision |
|---|---|---|
| 1 | Successful jobs keep their durable leases | Every transition that leaves an active job state (`proposal_ready`, `evaluated`, `evaluation_incomplete`, `failed`, `interrupted`, `cancelled`, and terminal decisions) clears `lease_owner`, `lease_token`, `lease_expires_at`, and `job_kind` in the same CAS/update transaction and stops the heartbeat. Tests start evaluation immediately after proposal completion and start another refinement without TTL delay. §4.3. |
| 2 | Cancellation is not durable across processes | `refinements.cancel_requested_at` is persisted; `POST /cancel` returns 202 “requested” for an active job. Heartbeat, scheduling, and result-write transactions observe it; the owner transitions to `cancelled`, while completion-versus-cancel is resolved by the same `BEGIN IMMEDIATE`/status CAS. A cancel from another process reaches the owner through the next heartbeat. §4.1, §4.2, §4.3. |
| 3 | Proposal preflight does not store confirmed evidence | `refinement_preflights.snapshot_json`/`snapshot_sha256` stores the complete immutable snapshot: rendered redacted evidence digest, digest hash, truncation metadata, provider/model fingerprint, profile revision/raw key/base-prompt state, and proposal prompt version/hash. `POST /refinements` copies it into `proposal_requested`; the worker consumes the snapshot and never re-reads source runs. Deleting source telemetry after preflight cannot change the confirmed payload. §4.1, §4.2. |
| 4 | One top-level apply marker is unsafe across profiles | Replace it with `pending_refinement_applies`, a map keyed by profile id. Reconciliation blocks only the affected profile; accepting profile B cannot overwrite A’s recovery record. A two-profile crash/concurrency test is required. §4.1, §4.2. |
| 5 | Accepted event omits the judge-regression acknowledgement | Add `acknowledge_judge_regression` to the immutable `accepted` payload minimums and the API decision record. §4.1, §4.2. |
| 6 | Evaluation preflight freezes too little execution state | The preflight snapshot binds materialized case content hashes, dataset/version/case provenance, baseline/candidate prompt fingerprints, provider/model/config fingerprint, judge configuration and threshold, budgets, and the estimate. Consumption revalidates them atomically and rejects on drift; each case is rehashed before execution, so a changed corpus makes the evaluation incomplete rather than silently different. §4.1, §4.2. |
| 7 | Marker-clearing failure is unspecified | The terminal event is authoritative. Marker removal is an idempotent post-step; if its checked save fails, the marker remains, profile writes stay blocked, and reconciliation retries only the clear (never appends a second terminal event). Tests cover accept and rollback marker-clear failure and restart. §4.2, §4.3. |
| 8 | The recovery guarantee needs an operational boundary | Stated explicitly: the “never confuse applied-then-edited with never-applied” guarantee covers writers that use the current checked API and respect `pending_refinement_applies`. Older binaries or external `config.json` replacement can erase the marker; in that case startup records `recovery_unverifiable`/`apply_failed`, never `accepted`, and the UI says the recovery evidence was lost. §4.2, §7. |

## 1. What the code actually provides (and where the issue is imprecise)

These facts change the design, so they come before the options.

| Issue assumption | Current code |
|---|---|
| A profile has `system_prompt_extra` | **No.** `RunConfigDTO.system_prompt_extra` (`application/dto.py:155`) is populated at execution time from `profile.system_prompts[mode]`, else from `SYSTEM_PROMPT_TEMPLATES[profile.prompt_template_name][mode]` (`server/routes/chat.py:71–92`, `:683–693`; `server/routes/compare_matrix.py:945–963`). The server `RunProfile` has `system_prompts: dict[str,str]` and `prompt_template_name` only (`server/models.py:858–871`). |
| The supplement applies to every mode | **No.** Only `RunRLMUseCase` appends it (`application/use_cases/run_rlm.py:151–152`, `:981–982`). Direct and RAG ignore profile prompts. #78 is therefore RLM-only unless the scope is widened. |
| The base prompt is immutable | It is a packaged YAML (`prompts/system_prompt_v2_0.yaml`, internal `version: "2.2"`), never written by the app. Nothing currently records its version on a run except the RLM fingerprint's `prompt_version`. |
| A profile has a version | **No.** `RunProfile` has no revision or timestamps; `PUT /api/profiles/{id}` mutates in place and saves the whole config (`server/routes/profiles.py:134–180`). Profiles live in `<state>/config.json` under `user_profiles`, written atomically under `_config_lock` (`server/dependencies.py:1207–1300`). |
| Telemetry can select degraded/failed runs for a profile | **Only partly.** `runs.outcome_category` exists (schema v2), but `TelemetryStore.get_run()` drops it (`telemetry/store.py:540–563`), `list_runs()` has no outcome filter (`:479–522`), and `runs` has no profile column. “Degraded” is `success=True` with a `⚠️`-prefixed answer (`outcome_classifier.py:80–90`), so `/api/executions` reports it as `complete` (`routes/traces.py:97`). |
| The bench path can supply baseline/candidate run ids | **No for the CLI.** `rlmstudio bench` / `MatrixBenchmarkRunner` never writes `TelemetryStore`; the Compare matrix route does (`routes/compare_matrix.py:827–910`). The CLI can therefore produce scores but not the run ids #78 requires. |
| The judge is fully attributable | **No.** `JudgeScore` stores only `judge_provider_id`, no model string and no judge prompt version (`server/models.py:951–971`); the benchmark `JudgeScorer` does record model + prompt version (`benchmark/scoring.py:95–115`). |
| The UI can edit a per-profile custom RLM prompt | **No.** `ProfileCard` only chooses a template, and its save sends `system_prompts: {}` (`frontend/src/components/settings/profile-card.tsx:246`), wiping any custom text. Clone and export also omit `system_prompts` (`:99–113`, `:143–150`). This is a prerequisite bug for any refinement that stores text there. |
| Run evidence can be exported with the run | `TelemetryStore.export_jsonl()` exists but has **no HTTP endpoint**, and its metadata line has no run id or outcome (`telemetry/store.py:619–691`). |

Two persistence facts are load-bearing for the accept path:

- `AppState._save_config_locked()` catches `Exception`, logs
  `"Failed to save config"`, and **returns normally** (`server/dependencies.py:1304–1305`).
  The temp-write/fsync/replace failures do re-raise internally, but the outer
  handler swallows them. Accept cannot distinguish a persisted candidate from
  an in-memory-only one unless a checked save is added.
- `runs` has no `content_sha256` and no benchmark dataset/version/case-id
  provenance (`telemetry/store.py:28–50`). A held-out guarantee cannot be
  proven from query strings; the run writers need to persist content identity
  and benchmark provenance.

Useful extension points that do exist:

- `MatrixSlotDTO.config` is a per-slot `RunConfigDTO` override
  (`application/use_cases/run_matrix_comparison.py:61–96`), and
  `RunConfigDTO.system_prompt_extra` is exactly the field a candidate needs.
- `rlmstudio.api.build_matrix_slots(..., temperature=...)` builds independent
  adapter + sandbox instances per slot (`api.py:744–805`), so baseline and
  candidate can be two slots in one matrix run with temperature 0.
- `BenchmarkCase.tags` allows a `holdout` marker without a dataset schema
  change (`benchmark/dataset.py:40–78`); `load_dataset` +
  `corpus.materialise` provide cases and derived expected answers.
- The telemetry schema has a real migration framework: add `_MIGRATIONS[3]`
  and bump `_CURRENT_SCHEMA_VERSION`; never edit `_SCHEMA_SQL`
  (`telemetry/store.py:98–144`).

Further facts that constrain the runner/UI choice:

- The Compare matrix route is `POST /api/chat/compare-matrix`, not
  `/api/compare/matrix`; it is **synchronous** (the request stays open until
  every slot finishes), has no job id, and its V2 ephemeral Chat Providers do
  not set `profile_id`. A new job service cannot simply call that route.
- `MatrixBenchmarkRunner` has a custom `slot_builder` hook
  (`benchmark/matrix_runner.py:50–51`, `:188–217`), so a Python caller can
  inject per-slot prompts, but `SlotOutcome` has no trace/label/prompt field
  and the report groups cells by `(provider, model, engine)`
  (`benchmark/report.py:268–270`): baseline and candidate with identical
  provider/model/engine would merge into one cell. The CLI exposes no prompt
  flag.
- `RunMatrixComparisonUseCase._copy_config_for_slot` rebuilds
  `RunConfigDTO` field-by-field and currently omits newer fields such as
  `result_preview_chars` and the registry caps
  (`run_matrix_comparison.py:422–445`). Any new run-config field added for
  refinement must be added there or it is silently dropped.
- `MAX_SLOTS = 10` (`run_matrix_comparison.py:159–160`); two arms per case fit,
  but provider × mode × variant expansion does not.
- The benchmark loader drops unknown case-level YAML keys
  (`benchmark/dataset.py:154–199`). A holdout marker must use the preserved
  `tags` list or `metadata` mapping, which is why G1 uses a tag rather than a
  new `holdout:` key.

## 2. Recommended shape

The recommended end-to-end flow (alternatives and their trade-offs follow):

1. **Define the supplement** as the effective RLM profile prompt: the current
   `system_prompts["rlm"]` if set, otherwise the named template's `rlm` text.
   The proposal records the *storage state*, not just the resolved text:
   whether `system_prompts["rlm"]` existed and its exact prior value. Accept
   materializes the candidate into that key; rollback restores the exact
   presence/value, so a profile that used a template resumes following it.
2. **Version the profile.** Add `revision` (monotonic integer) and
   `updated_at`; every profile write increments it. The proposal records the
   base revision, resolved-supplement hash, raw key presence/value, and base
   prompt version/hash.
3. **Persist an append-only event log** in telemetry schema v3:
   `refinement_events` holds immutable proposal, evaluation, decision, apply,
   and rollback payloads with a transactionally allocated per-refinement
   `sequence`; the `refinements` row is a status/lease projection, not
   evidence. Projection and export order by sequence, never timestamp.
4. **Propose** through one LLMPort call using a new versioned
   `prompts/refine_supplement.yaml`; the model returns JSON
   `{candidate, rationale}` and the server computes the diff. Proposal
   preflight shows the exact redacted evidence digest, its hash, truncation
   metadata, provider model, token limits, and known/unknown cost; it stores
   the complete immutable snapshot (including the rendered prompt and profile
   base state) in `refinement_preflights`. Its single-use token is consumed in
   the same transaction that creates the refinement and lease; the
   `proposal_requested` event copies the snapshot, and the worker uses that
   snapshot rather than re-reading source runs. The proposal event stores all
   of it (including the rendered prompt hash/version). Note: the shared
   `load_prompt_from_file` returns only the template and discards the YAML
   `version` (`prompts/templates.py:201–243`); the proposal loader must read
   the version explicitly (the `benchmark/scoring.py::_load_prompt` pattern).
5. **Evaluate** on held-out benchmark cases, baseline vs candidate, through
   `RunMatrixComparisonUseCase` with two slots and per-slot configs. Persist
   each arm as a normal telemetry run with content hash + benchmark provenance
   and record the run ids. Every selected case × repetition must produce both
   arms; otherwise the evaluation is `evaluation_incomplete` and cannot be
   accepted. Judge-only cases use the §3 J6 threshold; a holdout containing
   one requires a configured judge and complete verdicts.
6. **Gate** is advisory: a per-case deterministic regression marks the
   candidate and defaults the UI to reject; a judge-only regression beyond the
   threshold requires its own acknowledgement; explicit accept is allowed and
   recorded. The gate is only computed from a complete evaluation.
7. **Accept/reject/rollback** are explicit actions. Nothing writes to the
   profile before accept. Accept is one `_config_lock` transaction:
   re-read + revision/hash/raw-key/base-prompt comparison + mutate + **checked
   save** (which raises on write/fsync/replace failure) + restore in-memory
   state on failure. The save writes an entry in the
   `pending_refinement_applies` map (keyed by profile id) in `config.json`;
   writes for that profile are blocked while its marker/`applying` state
   exists, and reconciliation under the lock decides applied vs failed before
   any edit proceeds. A terminal event is authoritative even if the later
   marker-clear save fails; reconciliation retries only the clear. Rollback
   restores the raw key state and appends history. Proposal/evaluation jobs
   hold a durable SQLite lease with heartbeat renewal and a consumed preflight
   token; every transition that leaves an active job clears all lease fields
   atomically, and cancellation is persisted in `cancel_requested_at` so it
   reaches the lease owner across processes.
8. **Expose** it in a Refinements area of Settings (wizard + history), and
   include refinement events in the run JSONL export and a profile-level
   JSONL export.

## 3. Options

### A. Sequencing and scope

**A1 — Keep #78 blocked; implement after #77 items 2–3 (recommended).**
- Details: design now; no code until both merge; re-baseline `longdoc-v1`.
- PROS: evaluation is comparable; no duplicate evidence parser; matches the
  issue's explicit gate.
- CONS: delays user-facing value; the proposal/persistence layer could be
  built earlier in isolation.
- Verdict: recommended.

**A2 — Split: build proposal + persistence now, gate/evaluation later.**
- Details: deliver propose/review/reject/rollback without the A/B evaluation;
  add evaluation after #77.
- PROS: earlier UX; data model can settle; less waiting.
- CONS: violates the issue's “do not start” note; ships a feature whose core
  safety gate is absent; prompt-storage decisions may be invalidated by #77
  trace changes.
- Verdict: not recommended unless the owner explicitly relaxes the block.

**A3 — Full closed loop only (no intermediate release).**
- Details: one feature branch/PR series ending with proposal + evaluation +
  gate + history.
- PROS: coherent acceptance; no half-feature in the UI.
- CONS: long feedback cycle; most code is unreachable until the end.
- Verdict: acceptable if PRs are still staged internally, but A1 is safer.

### B. Which persisted text does refinement edit?

**B1 — Reuse `profile.system_prompts["rlm"]`; materialise template text on
accept (recommended).**
- Details: current supplement = `system_prompts["rlm"]` or template `rlm`.
  Accept writes the candidate to `system_prompts["rlm"]`; if the profile was
  template-based, the proposal records `base_supplement_present=false` and
  `base_supplement_value=null`, so the event shows the detach and rollback can
  remove the key. Fix `ProfileCard` to preserve/edit the dict. Direct/RAG
  prompts are untouched.
- PROS: the candidate is exactly the text the RLM loop receives; no second
  supplement layer; no prompt-assembly change; only RLM, which is already the
  only mode that uses it.
- CONS: accepting a template-based profile silently stops following future
  template edits for RLM; the UI must explain the detach; requires fixing the
  `ProfileCard` wipe/clone/export bugs first.

**B2 — Add a new `profile.system_prompt_extra` field, appended after the
resolved template/custom prompt.**
- Details: base prompt + existing resolved prompt + refined extra; candidate
  lives in the new field; history never touches `system_prompts`.
- PROS: clean separation of manual prompt text and refined note; template
  hot-reloads still apply; rollback only touches the new field; matches the
  issue's vocabulary literally.
- CONS: two supplements can stack and conflict; requires changes to the
  profile DTO, chat/compare prompt assembly, history budgeting, clone/export,
  and the fingerprint; the issue's “current supplement” becomes ambiguous.

**B3 — Refine only custom profiles; require the user to detach first.**
- Details: if `prompt_template_name` is set and no custom RLM text exists,
  disable refinement with “create a custom RLM prompt first”.
- PROS: no silent detach; smallest semantic change.
- CONS: worse UX; template-based profiles (likely the default) cannot be
  refined; user must create a custom prompt manually before the feature works.

**Recommendation (owner-selected):** B1 for 1.x. It matches the code path and
avoids a second supplement layer; record the detach and restore the raw key
state explicitly. B2 is not in 1.x.

### C. Refinement event storage and shape

**C1 — Append-only `refinement_events` log plus a `refinements` head/lease
row and a `refinement_runs` link table (recommended).**
- Details: schema v3 migration. `refinement_events` stores one immutable
  payload per transition: proposal (`base` storage state, candidate, diff,
  rationale, proposal provenance, exact evidence digest/hash/truncation),
  evaluation (cases, reps, run links, deltas, flags, completeness, judge
  provenance, holdout verification), decision (`accepted`/`rejected` with
  acknowledgement/reason), apply attempts (`applying`/`apply_failed`), and
  rollback. The `refinements` row is a projection of status and the durable
  job lease; it is not evidence. The link table holds
  `(refinement_id, run_id, role, case_id, rep, arm)` for source/baseline/
  candidate runs. No FK cascade: audit must survive run deletion.
- PROS: immutable decision-time evidence; export reproduces exactly what was
  decided; status is rebuildable; queryable with telemetry; one migration
  path; link table gives run-id lookups without normalising every payload.
- CONS: more rows and a projection/replay layer; profiles are in
  `config.json`, so accept is still a two-store write (see D); payloads need
  app-level validation; head status can drift from the log and must be
  repaired/rebuilt.

**C2 — Store events inside `config.json` under the profile.**
- Details: atomic with the profile save; no SQLite migration.
- PROS: one atomic write for accept/rollback; simple local persistence.
- CONS: config bloat; mixes settings with evidence; no SQL filters; every
  profile save rewrites history; run-export joins still need the file.

**C3 — Append-only `refinements.jsonl` in the state directory.**
- Details: one JSON line per state transition/event, like Prime Agent's
  `refinement_history.jsonl`; profile remains in `config.json`.
- PROS: simple audit trail; append is cheap; survives config rewrites.
- CONS: queries and status transitions require scanning/compacting; UI needs a
  parser; no referential integrity; two stores for export.

**C4 — Fully normalised tables (`refinements`, source traces, evaluation
cases, evaluation runs).**
- PROS: strongest integrity and querying.
- CONS: most migration/code effort for a low-volume audit feature; premature.

**C5 — One mutable `refinements` row updated through the lifecycle.**
- PROS: simplest read path; one place for status and evidence.
- CONS: overwrites proposal/evaluation/decision detail as status advances;
  cannot reproduce the decision-time record; terminal evidence is not
  immutable; concurrent updates lose history. This is the design finding 8
  rejects.

**Recommendation:** C1. Keep evidence in immutable event payloads, keep only
status/lease in the head row, and use the link table for run joins.

### D. Profile versioning, accept atomicity, rollback

**D1 — Add `revision` + `updated_at`, CAS inside one `_config_lock`
transaction, with a checked save (recommended).**
- Details: `revision` starts at 1 for user profiles; every create/update/
  accept/rollback increments it. The proposal records `base_revision`,
  `base_supplement_sha256`, raw `system_prompts["rlm"]` presence/value, and
  base-prompt version/hash. Accept takes `_config_lock` **before** re-reading
  and holds it through comparison, mutation, checked persistence, and the
  in-memory rollback on failure; otherwise 409. Rollback requires
  `applied_revision` to match and restores the raw key state (removing the
  key when it was absent). Lock order is `_config_lock` → telemetry store
  lock; no path may take them in reverse.
- PROS: closes the check-then-write race; precise optimistic concurrency;
  robust audit; cheap; also useful to future runs (`runs.profile_revision`).
- CONS: profile schema/API change; existing profiles need a default (0/1);
  every profile write site must bump it; accept holds the config lock across
  a disk write (short, but must not call back into unrelated locked state).
- Required persistence primitive: a `save_config_checked()`-style operation
  whose write/fsync/replace failures propagate. `_save_config_locked()`
  currently catches and logs `Exception`, so it cannot signal failure
  (`server/dependencies.py:1304–1305`). Use it for profile create/update as
  well as accept/rollback, so a profile CAS never runs against an in-memory
  revision that failed to persist. Accept snapshots the profile object,
  mutates, calls the checked save, and restores the snapshot + revision on any
  exception before recording `apply_failed`.

**D2 — Content hash only, no revision counter.**
- PROS: no schema/API field; hashes detect edits.
- CONS: cannot order edits; rollback/attribution wording (“base version”)
  becomes a hash; collision/concurrency edge cases are subtler.

**D3 — Full profile snapshots in a history table and rollback by snapshot.**
- PROS: exact restoration of all profile fields, not just the supplement.
- CONS: larger storage; profile fields are partly unrelated to refinement;
  still need CAS; the issue only requires supplement rollback.

**D4 — Accept appends `applying`, then performs a checked profile save carrying
`pending_refinement_applies[profile_id]`, then appends `accepted`; on failure
appends `apply_failed` and restores memory; reconcile before any later profile
write or on startup (recommended with D1).**
- Details: the `applying` event carries the intended candidate and both the
  base and target revisions. If that append fails, accept aborts before
  mutating. The checked save writes the new profile **and** an entry
  `{refinement_id, kind, base_revision, target_revision,
  target_rlm_key_present, target_rlm_key_sha256}` keyed by profile id. After a
  successful save, `accepted` (or `rolled_back`) is appended. If that append
  fails, the entry remains. Profile create/update routes for that profile
  first call reconciliation under `_config_lock`: if the expected terminal
  event already exists, only clear the entry; if the persisted revision equals
  the target and the RLM key presence/hash equal the entry, append the
  terminal event (the save succeeded, event was lost); otherwise append
  `apply_failed`/`rollback_failed` with the observed revision/hash. No profile
  edit is allowed while its entry or `applying` state remains unresolved.
  Terminal events are authoritative: if clearing the entry fails after the
  event exists, the profile stays blocked and reconciliation retries only the
  clear; it never appends a second terminal event. Rollback's raw key state is
  applied inside the same locked/checked transaction.
- **Operational boundary:** the “applied then edited” vs “never applied”
  distinction holds for writers that use the current checked API and respect
  the map. An older binary or external `config.json` replacement can erase the
  entry; if an `applying` event survives without a marker, startup records
  `apply_failed` with `recovery_unverifiable=true` and never `accepted`.
- PROS: resolves the post-save ambiguity with evidence in both stores; no
  silent loss; per-profile isolation; profile writes cannot advance past an
  unresolved apply.
- CONS: a config-schema map, a reconciliation path on every write for an
  affected profile, and idempotent tests; event and config writes remain
  separately committed, so reconciliation is still required.

**D5 — Write the profile then insert the event, no recovery state.**
- PROS: simplest.
- CONS: a failed event insert leaves a modified profile with no evidence; the
  feature's own audit guarantee is broken.

**Recommendation:** D1 + D4. Rollback appends a new event with `rollback_of`
rather than deleting/erasing the accepted event (Prime Agent records rollback
as history; audit should be append-oriented).

### E. Candidate representation and proposal output

**E1 — Ask the model for full candidate text + rationale; compute the diff
server-side with `difflib` (recommended).**
- Details: JSON `{candidate, rationale}`; validate non-empty, not identical,
  length cap (default 8,000 chars); store full text + unified diff.
- PROS: deterministic diff; apply is exact; tolerant of model formatting;
  no fragile patch application.
- CONS: the model can restate the whole supplement (slightly more output
  tokens) and can propose an overlong text (caught by the cap).

**E2 — Ask the model for structured edit operations (add/replace/remove
lines) and apply them.**
- PROS: smaller output; explicit intent; Prime Agent-style JSON edits.
- CONS: local-model JSON reliability is lower; merge/offset bugs; harder to
  review a patch than a full text diff; needs validation and conflict rules.

**E3 — Ask only for a rationale/suggestions; no candidate text.**
- PROS: safest against prompt injection; cheap.
- CONS: not automatable; the user must write the candidate; fails the issue's
  proposal requirement.

**Recommendation:** E1. Compute and store the diff; never trust a
model-produced patch.

### F. Proposal model and evidence bundle

**F1 — User selects a proposal Chat Provider; default to the Chat Provider on
the newest source run; record provider/model/prompt version/usage
(recommended).**
- PROS: explicit cost/quality control; local users can stay local; provenance
  is exact; avoids silently using a judge or a different model family.
- CONS: one more choice; source runs may span providers; the user must have a
  capable context window.

**F2 — Always use the app's judge Chat Provider.**
- PROS: already configured; often a strong cloud model; fewer choices.
- CONS: conflates grading and proposing; the judge may be expensive or absent;
  evidence privacy goes to the judge provider; no per-proposal control.

**F3 — Automatically use the provider/model of the newest source run.**
- PROS: zero configuration; evidence and proposer share the same model.
- CONS: surprise provider spend; the newest run may use a weak/small-context
  model; no way to choose a stronger proposer.

**F4 — Add a dedicated “refinement model” setting.**
- PROS: stable default; good for repeated use.
- CONS: another global setting; still needs overrides.

**Evidence bundle options (orthogonal to F1–F4):**

**F-a — Deterministic capped digest (recommended).** Per selected trace:
id, outcome category/degraded, query (bounded), answer/error (bounded), step
counts/action types, token/cost/duration, timestamps. Cap trace count
(e.g. 12) and total chars (e.g. 24k); apply secret redaction; record what was
truncated. Persist the **exact redacted digest text**, its SHA-256, the
truncation metadata (included/excluded trace ids and chars), and the rendered
proposal-prompt hash/version in the proposal event, so deleting source runs
does not destroy the audit record of what the proposer saw. PROS: bounded
context/cost; reproducible; no extra LLM calls; auditable after source-run
deletion. CONS: may omit the decisive detail from a long trace; the stored
digest duplicates some trace content (intended evidence, not a cache).
**F-b — Full trace JSON, token-aware truncation.** PROS: maximum fidelity.
CONS: often exceeds local-model context; expensive; truncation is hard to make
fair.
**F-c — Map-reduce summaries per trace, then propose.** PROS: handles many
long traces. CONS: multiple model calls, extra cost/latency, summariser bias,
violates the issue's “one model call”.

**Recommendation (owner-selected):** F1 + F-a, with the proposal preflight
showing provider/privacy/evidence/cost. Include the current supplement in
full, the base prompt version/hash, and a stable instruction that evidence is
data, not instructions. Store the exact rendered prompt (or its hash + template
version) with the event.

### G. Held-out case selection and leakage

**G1 — Mark a fixed holdout subset in `benchmarks/longdoc-v1.yaml` with a
`holdout` tag, and persist content + benchmark provenance on every run
(recommended).**
- Details: tag 6 candidate cases (2 small, 2 medium, 2 large; include needle,
  aggregation, refusal, and optionally one judge-only synthesis). The final
  composition is confirmed in Phase 0 after the #77 re-baseline; if a
  judge-only case is included, §3 J6 requires a configured judge and complete
  verdicts. Use `filter_by_tag("holdout")`. Every run records `content_sha256` of the
  resolved document; evaluation runs additionally record
  `benchmark_dataset`, `benchmark_dataset_version`, and `benchmark_case_id`.
  A case is eligible only if its materialised-content hash is disjoint from
  every selected source run's `content_sha256`, and its case id/dataset does
  not match a source run's benchmark provenance. The evaluation event records
  `holdout_verified=true` only when every source run has a content hash and
  the check passes; otherwise it is false and acceptance is blocked. Legacy
  runs without a content hash are **unverifiable**: they can be inspected but
  cannot make a gate-eligible proposal. User-chosen cases from the holdout
  split are allowed; non-holdout cases are recorded as `holdout=false` and
  block verified acceptance.
- PROS: proves disjointness by document identity, not query text; catches
  paraphrases and different questions over the same case document; no dataset
  schema change; clear default; reproducible.
- CONS: requires run-writer changes and a migration; the holdout tag is
  mutable in the YAML; public-text cases may be un-materialisable offline;
  legacy runs cannot be used for a verified gate.

**G2 — Separate `benchmarks/longdoc-holdout-v1.yaml`.**
- PROS: hard separation; cannot accidentally train/refine on it by default;
  explicit provenance.
- CONS: duplicates sources/cases; maintenance drift; more files.

**G3 — Purely user-chosen case ids, no holdout concept.**
- PROS: maximum flexibility.
- CON: the gate can be run on cases the supplement was tuned against; weakens
  the issue's central invariant.

**Leakage guard (required with G1):** compare the evaluation case's
materialised-content SHA-256 with every selected source run's
`content_sha256`, and compare benchmark provenance (dataset/version/case id)
where present. Any match excludes the case and sets
`holdout_verified=false`; there is **no `allow_source_overlap` escape in
1.x**. A source run without `content_sha256` is not selectable for a proposal
at all; legacy/unattributed runs are view-only and cannot enter this workflow.
The event records `holdout_verified` plus the excluded case/run ids so the
limitation is auditable. Query-string comparison is not a substitute; it
misses paraphrases and same-document/different-question cases.

**Recommendation (owner-selected strict exclusion):** G1 + content-hash/
provenance disjointness. Default holdout, user-visible non-holdout selection,
no query-string substitute, no verified acceptance from unattributed legacy
runs, never silent.

### H. Evaluation runner and run-id persistence

**H1 — Dedicated server-side evaluation service over
`RunMatrixComparisonUseCase`, with two slots per case (baseline/candidate) and
telemetry writes reusing the Compare route's slot-recording helper
(recommended).**
- Details: for each case/repetition, build two slots with the same provider/
  model adapter (temperature 0) and per-slot `RunConfigDTO`s differing only in
  `system_prompt_extra`; run one matrix call; persist both results as ordinary
  telemetry runs with a refinement/evaluation tag; record run ids in
  `refinement_runs`. The use case returns only `comparison_group_id` and slot
  ids — no execution ids and no telemetry writes — so the evaluation service
  assigns execution ids and calls the extracted recording helper itself
  (`run_matrix_comparison.py:99–108`, `compare_matrix.py:827–910`). Reuse
  `benchmark.scoring.matches` and `JudgeScorer`.
- PROS: baseline and candidate experience the same process, provider state and
  wall-clock window; run ids are real telemetry rows; reuses the Compare path
  the issue names; per-slot failure isolation already exists.
- CONS: a new orchestration loop (case × rep) and a small extraction of the
  compare route's telemetry writer; the server must own sandbox/adapter
  construction.

**H2 — Extend `MatrixBenchmarkRunner` with prompt variants, a trace field on
`SlotOutcome`, and a telemetry sink.**
- Details: custom slot builder already exists; add a callback/trace and a
  storage sink, then use `reps`/judge/report as-is.
- PROS: maximal reuse of the benchmark harness (reps, judge mean ± range,
  progress, dry-run fakes); one runner for both.
- CONS: changes a benchmark API used by the CLI; `SlotOutcome` lacks traces
  and a variant field; the report groups by provider/model/engine, so
  baseline and candidate cells merge unless a variant dimension is added;
  the bench path is explicitly report-oriented; telemetry semantics become a
  benchmark concern.

**H3 — Instrument `rlm-studio bench` to write telemetry and shell out.**
- PROS: reuses the exact CLI; easy to reproduce manually.
- CONS: CLI writes become a side effect; no job progress; process management;
  server state/correlation is hard; run ids are a new CLI concern.

**H4 — CLI/report-only evaluation; store scores, not run ids.**
- PROS: least server code.
- CONS: violates #78's requirement for baseline/candidate run ids and
  trace-linked evidence; no in-app gate.

**Recommendation:** H1. Use `build_matrix_slots` for adapter/sandbox
construction if practical (it already supports temperature 0), then override
each slot's `config.system_prompt_extra`; extract the compare route's
`_record_slot_telemetry` into a shared server helper so both paths record the
same columns/steps.

### I. Async execution and job lifecycle

**I1 — Durable background jobs (proposal and evaluation), with a SQLite lease,
heartbeat renewal, durable preflight tokens, a transactionally sequenced event
log, progress projection, and polling (recommended).**
- Details: `refinements` carries `lease_owner`, `lease_token`,
  `lease_expires_at`, `last_sequence`, `job_kind`, `cancel_requested_at`, and
  progress. A `refinement_preflights` table stores the SHA-256 of the one-time
  token, its purpose (`proposal`/`evaluation`), canonical selection, estimate,
  **immutable snapshot** (`snapshot_json`/`snapshot_sha256`), expiry, and
  consumption. Starting either job runs one `BEGIN IMMEDIATE` transaction:
  validate token hash/expiry/selection, revalidate the snapshot's execution
  state, mark it consumed, acquire the profile lease (and create the
  refinement head for a proposal), allocate the next event sequence, append
  the start event with the snapshot, and commit. A replay, mismatch, or
  concurrent consumer fails in that transaction.
- **Proposal preflight** returns the provider/model, exact redacted evidence
  digest/hash/truncation, token limits, and known/unknown cost, and stores the
  complete rendered snapshot. The worker uses `proposal_requested`'s copied
  snapshot, never re-reading source runs; deleting source telemetry after
  preflight cannot change the confirmed payload.
- **Evaluation preflight** returns selected/missing cases, run count, maximum
  cost/time, and holdout verification, and freezes the case content hashes,
  dataset/version/case provenance, baseline/candidate prompt fingerprints,
  provider/model/config fingerprint, judge configuration/threshold, budgets,
  and estimate. Consumption revalidates all of them; a case is rehashed before
  execution and a mismatch makes the evaluation incomplete.
- **Heartbeat:** a job-owned thread renews with
  `UPDATE refinements SET lease_expires_at=? WHERE id=? AND lease_token=? AND
  status=...`; the TTL is derived from the maximum provider request timeout
  plus margin (e.g. `max(90s, 2 × provider_timeout)`) and renewal runs at
  `ttl/3`. It also reads `cancel_requested_at` and signals the worker. A
  rowcount of 0 means ownership was lost: stop scheduling, stop writing
  results, and log/record `interrupted` only if the lease can be reacquired.
  Every event/result write re-checks ownership.
- **Lease release:** every transition that leaves an active job
  (`proposal_ready`, `evaluated`, `evaluation_incomplete`, `rejected`,
  `failed`, `interrupted`, `cancelled`, and terminal decisions) clears
  `lease_owner`, `lease_token`, `lease_expires_at`, and `job_kind` in the same
  status-CAS/update transaction, then the heartbeat stops. The profile’s
  partial-unique lease slot is free immediately; no TTL wait.
- Every transition event is inserted in the same transaction as its status
  CAS with `last_sequence`; event order is `(refinement_id, sequence)`, never
  `created_at`.
- **Cancellation is durable:** `POST /cancel` sets `cancel_requested_at` in
  the head row (202 `requested` for an active job; if no live lease remains it
  may terminate directly). The owner’s heartbeat sees it, stops scheduling,
  and the next result-write transaction refuses further work; the owner then
  transitions to `cancelled`. If completion and cancel race, the first
  `BEGIN IMMEDIATE` + status CAS wins: `proposal_created` and
  `evaluation_completed` transition only when `cancel_requested_at IS NULL`;
  otherwise the owner records `cancelled`. An in-flight provider call is not
  killed; its result is not persisted after a cancel request.
- On startup, expired leases with `cancel_requested_at` become `cancelled`;
  other expired leases become `interrupted`. An `applying` marker is
  reconciled before any profile write. Graceful shutdown clears the lease and
  records `interrupted` (or `cancelled` when requested).
- PROS: no HTTP timeout; progress polling; one-active-job and single-use
  confirmations enforced durably; lease slots free immediately on completion;
  paid work cannot start without an explicit provider/evidence/cost
  acknowledgement; cancellation reaches the owner across processes; restart
  behavior is defined.
- CONS: more schema/state than an in-memory registry; heartbeat/release/
  transaction tests are required; cancellation is not immediate; a
  multi-process deployment still shares one SQLite file, acceptable for this
  single-node workbench but not a general queue.

**I2 — Synchronous proposal, async evaluation.**
- PROS: simpler proposal endpoint; one job type.
- CONS: a slow proposal (large evidence, local model) can hit the frontend
  proxy/HTTP timeout; inconsistent UX; a crash loses the proposal row.

**I3 — WebSocket progress for both.**
- PROS: live progress; consistent with chat streaming.
- CONS: more client/server machinery; reconnect semantics; overkill for a
  settings workflow that can poll.

**I4 — External queue (Celery/RQ).**
- PROS: restart-safe, scalable.
- CONS: a new dependency/service for a single-user workbench; deployment cost.

**Recommendation:** I1 with polling. Proposal preflight stores and shows the
complete provider/evidence/prompt snapshot and known/unknown cost; evaluation
preflight freezes and shows the case/config/judge/threshold/budget state. Each
issues a durable single-use token consumed in the job-start transaction; the
lease heartbeat prevents two workers from owning the same job across reloads,
lease fields clear atomically on completion, and `cancel_requested_at` carries
cancellation to the owner across processes.

### J. Metrics and the regression flag

**J1 — Per-case majority across repetitions; flag a candidate only when its
per-case success rate is lower than the baseline's on a deterministic case
(recommended).**
- Details: deterministic accuracy over cases with `expected_answer`; per-case
  success rate = passes/reps; regression case if candidate rate < baseline
  rate. Judge score per arm = mean and min/max range; cost total/mean; steps
  mean; outcome-class counts. Report missing cases/skips separately.
- PROS: robust to a single noisy repetition; still catches a real per-case
  drop; matches the issue's “any held-out case” intent without treating one
  flaky run as decisive.
- CONS: a 1-of-2 vs 2-of-2 difference is noise-sensitive; needs clear display
  of per-rep results.

**J2 — Strict per-repetition regression: any baseline pass followed by a
candidate fail flags the candidate.**
- PROS: maximally conservative; simple to explain.
- CONS: temperature 0 is not fully deterministic across providers; noisy
  flags; users learn to ignore them.

**J3 — Aggregate accuracy drop only.**
- PROS: stable.
- CONS: misses a narrow but real regression on one case, which the issue
  calls out.

**J4 — Hard block on any regression.**
- Rejected by the issue; the human accept is the boundary. Keep the flag
  advisory and require an explicit acknowledgement.

**J5 — Completeness precondition (required with J1).**
- Details: an evaluation is `evaluation_incomplete` unless every selected case
  materialised, every selected repetition executed, and both baseline and
  candidate result rows exist for every case × repetition. A failed arm is a
  result (it counts as a non-match/outcome), but a missing arm, a skipped
  case, an interrupted job, or zero deterministic cases is not complete.
  `evaluation_complete=false` cannot be accepted. Judge coverage is a second
  flag: when a judge is configured, `judge_complete` is false if any usable
  cell lacks a verdict; it is displayed and requires
  `acknowledge_judge_gaps=true` on accept, but it does not block the
  deterministic gate. A complete evaluation with zero deterministic cases is
  not gate-eligible.
- PROS: prevents an empty/partial run from looking like “no regression”;
  makes skipped corpus material explicit; separates deterministic completeness
  from judge coverage.
- CONS: partial results are still useful for diagnosis, so they need a
  distinct status and a retry path; users may be blocked when a public-text
  corpus is not fetched.

**J6 — Judge-only regression threshold (required with J1/J5).**
- Details: for cases without an expected answer, compute each arm's mean judge
  score across repetitions. Flag the case when
  `candidate_mean < baseline_mean - judge_regression_threshold`; the threshold
  defaults to 0.5 on the 1–5 scale, is configurable per evaluation, and is
  recorded in the evaluation event. A flagged judge-only case requires
  `acknowledge_judge_regression=true` on accept. A holdout containing
  judge-only cases requires a configured judge and `judge_complete=true`;
  missing verdicts are a separate `judge_complete=false` acknowledgement and
  do not silently satisfy the threshold. The final six-case split is selected
  after the #77 re-baseline, so the judge-only case can be included only once
  this rule is accepted.
- PROS: gives synthesis cases gate semantics; avoids silently ignoring a
  quality drop that deterministic accuracy cannot see; keeps the threshold
  explicit and auditable.
- CONS: judge scores are model-dependent and noisier than exact matches; the
  threshold is a policy choice; a missing judge excludes judge-only cases.

**Recommendation:** J1 + J5 + J6. Display per-case/per-rep detail and mark the
decision as advisory. Judge scoring uses `JudgeScorer` (records model + prompt
version); if no judge Chat Provider is configured, deterministic-only
evaluation is allowed but `judge_complete=false`, judge-only cases are
ineligible, and the UI states it. A complete evaluation with a judge
configured but missing verdicts still requires an explicit judge-gap
acknowledgement.

### K. UI placement

**K1 — New “Refinements” tab in Settings, with a profile-scoped wizard and
history; launch from a “Refine prompt” action on `ProfileCard`
(recommended).**
- PROS: long-running jobs and history need a stable page; the profile context
  is visible; no separate top-level navigation for a settings workflow.
- CONS: Settings becomes larger; the wizard has several steps.

**K2 — Modal/drawer from `ProfileCard` only.**
- PROS: minimal navigation; natural context.
- CONS: long evaluation progress and a case table are cramped; history/rollback
  has no home; closing the modal loses visibility.

**K3 — Standalone `/refinements` top-level page.**
- PROS: room for evidence and history; can deep-link.
- CONS: splits profile-related work across two areas; another route.

**K4 — CLI/API only for 1.x; defer UI.**
- PROS: smallest scope; backend correctness first.
- CONS: the issue explicitly lists UI; source selection and diff review are
  poor via curl.

**Recommendation (owner-selected for 1.x):** K1. The full wizard and history
ship before #78 closes. As a prerequisite, fix `ProfileCard` so saving does not
send `system_prompts: {}` and so clone/export include custom prompts; add a
textarea for the RLM custom prompt. Without that, any accepted candidate in
`system_prompts["rlm"]` is one unrelated profile edit away from deletion.

### L. Export and visibility

**L1 — Add an HTTP run-JSONL endpoint; include refinement events in the
metadata line; add a profile-level refinements JSONL export
(recommended).**
- Details: `GET /api/traces/{run_id}/export.jsonl` uses
  `TelemetryStore.export_jsonl`, adds `rlm_studio_execution_id`,
  `rlm_studio_outcome_category`, and a `rlm_studio_refinements` array (events
  where the run is a source or evaluation arm). `GET
  /api/profiles/{profile_id}/refinements/export.jsonl` replays the append-only
  event log (decision-time snapshots), not the mutable head row or current
  profile. Unknown metadata keys are ignored by the upstream visualizer.
- PROS: satisfies “exportable with the run's JSONL”; no step-action enum
  change; one canonical event representation.
- CONS: new endpoint (none exists today); the run export currently cannot
  include outcome until `get_run()` is fixed; deleted runs lose run-level
  context in the profile export (event snapshots preserve the evidence).

**L2 — Separate refinement export only.**
- PROS: least change.
- CONS: does not satisfy the issue's “with the run's JSONL”.

**Recommendation:** L1, with the `get_run()` outcome fix as a prerequisite.

## 4. Proposed data model and API

### 4.1 Telemetry schema v3

Add a migration block (keep `_SCHEMA_SQL` frozen). Everything below is a
migration; do not add these columns/tables to the v1 bootstrap.

```sql
-- Run attribution and provenance; legacy rows stay NULL and are unverified.
ALTER TABLE runs ADD COLUMN profile_id TEXT;
ALTER TABLE runs ADD COLUMN profile_revision INTEGER;
ALTER TABLE runs ADD COLUMN prompt_fingerprint TEXT;
ALTER TABLE runs ADD COLUMN content_sha256 TEXT;
ALTER TABLE runs ADD COLUMN benchmark_dataset TEXT;
ALTER TABLE runs ADD COLUMN benchmark_dataset_version TEXT;
ALTER TABLE runs ADD COLUMN benchmark_case_id TEXT;
CREATE INDEX IF NOT EXISTS idx_runs_profile ON runs(profile_id, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_content_hash ON runs(content_sha256);

-- Head/lease projection. Evidence is never stored here.
CREATE TABLE IF NOT EXISTS refinements (
    id               TEXT PRIMARY KEY,
    profile_id       TEXT NOT NULL,
    status           TEXT NOT NULL,   -- proposing|proposal_ready|evaluating|
                                      -- evaluation_incomplete|evaluated|applying|
                                      -- accepted|apply_failed|rejected|rolled_back|
                                      -- failed|interrupted|cancelled
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL,
    error            TEXT,
    last_sequence    INTEGER NOT NULL DEFAULT 0,
    job_kind         TEXT,            -- proposal|evaluation
    lease_owner      TEXT,
    lease_token      TEXT,
    lease_expires_at REAL,
    cancel_requested_at REAL,          -- durable cooperative cancel
    progress         TEXT             -- JSON: phase, cases done/total, current case, rep
);
CREATE INDEX IF NOT EXISTS idx_refinements_profile ON refinements(profile_id, created_at);
CREATE INDEX IF NOT EXISTS idx_refinements_status ON refinements(status);
-- At most one live job per profile; expired leases are cleared/reclaimed in
-- the acquisition transaction before this partial unique index is evaluated.
CREATE UNIQUE INDEX IF NOT EXISTS idx_refinements_active_lease
ON refinements(profile_id) WHERE lease_token IS NOT NULL;

-- Immutable evidence. Append-only: no UPDATE/DELETE API; optional triggers
-- below enforce it at the database level.
CREATE TABLE IF NOT EXISTS refinement_events (
    id             TEXT PRIMARY KEY,
    refinement_id  TEXT NOT NULL,
    sequence       INTEGER NOT NULL,  -- allocated with the status CAS
    event_type     TEXT NOT NULL,     -- proposal_requested|proposal_created|
                                      -- evaluation_started|evaluation_completed|
                                      -- evaluation_incomplete|applying|accepted|
                                      -- apply_failed|rejected|rolled_back|
                                      -- rollback_failed|failed|interrupted|cancelled
    payload_json   TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    created_at     REAL NOT NULL,
    actor          TEXT,
    UNIQUE (refinement_id, sequence)
);
CREATE INDEX IF NOT EXISTS idx_refinement_events_ref ON refinement_events(refinement_id, sequence);
CREATE TRIGGER IF NOT EXISTS refinement_events_no_update
BEFORE UPDATE ON refinement_events
BEGIN SELECT RAISE(ABORT, 'refinement_events is append-only'); END;
CREATE TRIGGER IF NOT EXISTS refinement_events_no_delete
BEFORE DELETE ON refinement_events
BEGIN SELECT RAISE(ABORT, 'refinement_events is append-only'); END;

-- Single-use confirmation tokens for proposal and evaluation paid work.
CREATE TABLE IF NOT EXISTS refinement_preflights (
    id              TEXT PRIMARY KEY,
    profile_id      TEXT NOT NULL,
    refinement_id   TEXT,             -- NULL until a proposal token is consumed
    purpose         TEXT NOT NULL,    -- proposal|evaluation
    token_hash      TEXT NOT NULL UNIQUE,  -- SHA-256 of the raw token
    selection_json  TEXT NOT NULL,    -- canonical source/case/reps/judge/provider
    snapshot_json   TEXT NOT NULL,    -- immutable confirmed payload and state
    snapshot_sha256 TEXT NOT NULL,
    estimate_json   TEXT NOT NULL,    -- canonical estimate/cost/knownness
    created_at      REAL NOT NULL,
    expires_at      REAL NOT NULL,
    consumed_at     REAL,
    consumed_by     TEXT              -- lease token
);
CREATE INDEX IF NOT EXISTS idx_refinement_preflights_profile
ON refinement_preflights(profile_id, created_at);
-- Only consumed_at/consumed_by may be updated; selection/snapshot/token_hash
-- are immutable after creation (enforced by the store API, trigger optional).

-- Audit links; deliberately no FK cascade so evidence survives run cleanup.
CREATE TABLE IF NOT EXISTS refinement_runs (
    refinement_id TEXT NOT NULL,
    run_id        TEXT NOT NULL,
    role          TEXT NOT NULL,      -- source|baseline|candidate
    case_id       TEXT,
    rep           INTEGER,
    arm           TEXT,               -- baseline|candidate
    PRIMARY KEY (refinement_id, run_id, role)
);
CREATE INDEX IF NOT EXISTS idx_refinement_runs_run ON refinement_runs(run_id);
```

Payload minimums (stored verbatim; `payload_sha256` is SHA-256 over canonical
JSON — sorted keys, UTF-8, compact separators).

- `proposal_requested`: `preflight_id`, `token_hash`, `snapshot_sha256`, the
  copied `snapshot_json` (rendered evidence/prompt and base state), and the
  estimate.
- `proposal_created`: `preflight_id`, `base_revision`, `base_supplement_present` (0/1),
  `base_supplement_value` (exact prior `system_prompts["rlm"]` value or null),
  `base_supplement_resolved`, `base_supplement_sha256`, `base_template_name`,
  `base_prompt_version`, `base_prompt_sha256`, `candidate_supplement`,
  `candidate_diff`, `rationale`, `proposal_chat_provider_id`,
  `proposal_provider`, `proposal_model`, `proposal_prompt_version`,
  `proposal_prompt_sha256`, `proposal_usage`, `cost_known`, `source_trace_ids`,
  `evidence_digest`, `evidence_sha256`, `evidence_truncation`,
  `source_provenance_verified`.
- `evaluation_started`: `preflight_id`, `token_hash`, `snapshot_sha256`, the
  copied `snapshot_json` (frozen cases/provenance/prompt/provider/judge/
  threshold/budgets), and `holdout_verified`.
- `evaluation_completed`: `evaluation_complete`, `judge_complete`,
  `case_ids`, `case_content_hashes`, `dataset`, `dataset_version`, `reps`,
  `provider`, `model`, `temperature`, `provider_config_fingerprint`,
  `baseline_prompt_fingerprint`, `candidate_prompt_fingerprint`,
  `judge_chat_provider_id`, `judge_model`, `judge_prompt_version`,
  `judge_config_fingerprint`, `judge_regression_threshold`,
  `judge_regression_cases`, per-case/per-rep/arm `run_ids`, `deltas`,
  `regression_cases`, `skipped`, `holdout_verified`,
  `source_overlap_exclusions`.
- `evaluation_incomplete`: the partial payload plus `incomplete_reasons`.
- `applying`: base/target revision, candidate hash, `evaluation_event_id`.
- `accepted`: `proposal_event_id`, `evaluation_event_id`, `applied_revision`,
  `decision_actor`, `reason`, `acknowledge_regression`,
  `acknowledge_judge_gaps`, `acknowledge_judge_regression`, decision time.
- `apply_failed`: error, restored revision, whether persistence completed.
- `rollback_failed`: error, restored revision; the head remains `accepted`
  because the accepted prompt is still the persisted state.
- `rejected`: `proposal_event_id`, `evaluation_event_id`, reason, actor,
  decision time.
- `rolled_back`: `rollback_of` (accepted event id), restored revision, raw key
  presence/value after rollback.
- `failed` / `interrupted` / `cancelled`: phase, reason, and — when the
  proposal phase reached rendering — `source_trace_ids`, `evidence_digest`,
  `evidence_sha256`, `evidence_truncation`, and the proposal prompt hash.

`runs.content_sha256` is SHA-256 of the exact resolved document content the
run saw. `runs.prompt_fingerprint` is SHA-256 over
`(base_prompt_version, template_or_custom marker, effective_supplement,
profile_id, profile_revision)`; evaluation runs also carry
`benchmark_dataset`, `benchmark_dataset_version`, and `benchmark_case_id`.
Writers compute these best-effort but the evaluation and accept paths require
them; a run without provenance is treated as unverifiable.

`config.json` also gains a top-level `pending_refinement_applies` **map keyed
by profile id**, written atomically with the profile:

```json
{
  "<profile_id>": {
    "refinement_id": "…",
    "kind": "accept",
    "base_revision": 3,
    "target_revision": 4,
    "target_rlm_key_present": true,
    "target_rlm_key_sha256": "…",
    "started_at": "…"
  }
}
```

An entry is absent when no accept/rollback is in flight for that profile. The
map (rather than one global object) prevents an unresolved apply on profile A
from being overwritten by an accept on profile B. Accept writes the entry in
the same checked save as the mutated profile. Profile update/delete for that
profile must reconcile its entry first; a marker whose target revision and key
hash match the persisted profile proves the save succeeded, while a mismatch
or a different revision proves it did not (or was overwritten outside the
API).

### 4.2 REST surface

All proposal/evaluation/decision routes are profile-scoped:

```
GET  /api/profiles/{profile_id}/refinements/source-runs
        ?outcome=timeout,context_overflow,...&degraded=true&limit=50
POST /api/profiles/{profile_id}/refinements/proposal/preflight
        {source_trace_ids: [...], proposal_chat_provider_id?: str}
        -> {preflight_id, confirmation_token, expires_at,
            proposal_provider, proposal_model, snapshot_sha256,
            profile_revision, base_prompt_version, proposal_prompt_version,
            evidence_digest, evidence_sha256, evidence_truncation,
            max_input_tokens, max_output_tokens,
            max_cost_usd, cost_known}
POST /api/profiles/{profile_id}/refinements
        {confirmation_token: str, source_trace_ids: [...],
         proposal_chat_provider_id?: str}
        -> 202 {refinement_id, status: "proposing", ...}
GET  /api/profiles/{profile_id}/refinements
GET  /api/profiles/{profile_id}/refinements/{refinement_id}
POST /api/profiles/{profile_id}/refinements/{refinement_id}/evaluation/preflight
        {case_ids?: [...], reps?: int=2, judge_chat_provider_id?: str}
        -> {preflight_id, snapshot_sha256, case_count, run_count,
            max_cost_usd, cost_known, max_time_seconds, skipped,
            holdout_verified, confirmation_token, expires_at}
POST /api/profiles/{profile_id}/refinements/{refinement_id}/evaluate
        {confirmation_token: str, case_ids?: [...], reps?: int=2,
         judge_chat_provider_id?: str}
        -> 202 {status: "evaluating"}
POST /api/profiles/{profile_id}/refinements/{refinement_id}/cancel
        -> 202 {status: "cancel_requested" | "cancelled"}
POST /api/profiles/{profile_id}/refinements/{refinement_id}/accept
        {acknowledge_regression?: bool, acknowledge_judge_gaps?: bool,
         acknowledge_judge_regression?: bool, reason?: str}
POST /api/profiles/{profile_id}/refinements/{refinement_id}/reject
        {reason?: str}
POST /api/profiles/{profile_id}/refinements/{refinement_id}/rollback
GET  /api/profiles/{profile_id}/refinements/export.jsonl
GET  /api/traces/{run_id}/export.jsonl
```

Rules:

- `source-runs` reads only `runs.profile_id = {profile_id}`. It does **not**
  infer ownership from a shared Chat Provider; legacy rows without a profile
  are explicitly unattributed and never appear here. Manual inclusion of
  unattributed legacy runs is not offered in 1.x; if added later it must carry
  an explicit acknowledgement and `holdout_verified=false`, and cannot lead to
  acceptance. The endpoint returns `outcome_category`, computed `degraded`
  (`success=True` and the answer starts with `⚠️`), `content_sha256`, benchmark
  provenance, and `provenance_verified`.
- `proposal/preflight` renders the exact redacted evidence digest, validates
  every source id exists, belongs to the profile, and has `content_sha256`,
  resolves the proposal provider/model and token limits, and returns its hash,
  `max_cost_usd`, and `cost_known` (false when no price is known, so the UI
  must ask for an unknown-cost acknowledgement). It stores a durable
  single-use token bound to the canonical selection and a **complete immutable
  snapshot**: rendered digest/prompt, digest hash, truncation metadata,
  provider/model fingerprint, profile revision/raw key/base-prompt state, and
  prompt version/hash. `POST /refinements` requires that token and matching
  selection; in one `BEGIN IMMEDIATE` transaction it revalidates the snapshot
  against the current profile/base prompt, consumes the token, creates the
  refinement head, acquires the profile lease, allocates the first sequence,
  and appends `proposal_requested` with the copied snapshot. The worker uses
  that snapshot and never re-reads source runs. A proposal cannot be created
  from unverified/legacy sources.
- `evaluation/preflight` materialises the selected cases, computes
  `holdout_verified` by comparing each case's content SHA-256 and benchmark
  provenance against every source run, rejects any overlap (no override), and
  returns a durable single-use confirmation token plus the maximum case/run
  count and cost/time estimate. It stores the frozen execution snapshot:
  per-case content hashes and dataset/version/case provenance; baseline and
  candidate prompt fingerprints; provider/model/config fingerprint; judge
  Chat Provider/model/prompt version/configuration and
  `judge_regression_threshold` (default 0.5); case budgets; reps; the pricing/
  cost estimate and `cost_known`; and the estimate fingerprint. Corpus-missing
  cases are listed as skipped and make the preflight
  ineligible unless the selected set is adjusted. A judge-only case requires a
  configured judge.
- `evaluate` requires the token; the request's case ids, reps, and judge must
  match the token's canonical selection; it also requires `proposal_ready`,
  `evaluation_incomplete`, or `interrupted` after an `evaluation_started`
  event (not a proposal interruption), `reps >= 2`, and a holdout-verified,
  fully materialisable case set. In one `BEGIN IMMEDIATE` transaction it
  revalidates every frozen snapshot field (profile revision/raw key,
  candidate event hash, base prompt, provider/model/config, judge config,
  threshold, dataset/version/case ids, materialised content hashes, budgets,
  pricing/cost estimate),
  consumes the token, acquires the durable job lease, allocates the next
  sequence, clears any stale `cancel_requested_at`, and appends
  `evaluation_started` with the snapshot. A replay, expired token, any drift,
  or second unexpired lease fails. The worker rehashes each case before
  executing it; a changed/missing corpus concludes `evaluation_incomplete`
  rather than evaluating a different document.
- `evaluation_completed` is written only when every selected case × repetition
  produced both arms. Otherwise `evaluation_incomplete` is written and the
  refinement cannot be accepted; retry is allowed.
- `accept` requires status `evaluated` or `apply_failed` (retry),
  `evaluation_complete=true`, at least one deterministic (expected-answer)
  case, `holdout_verified=true`, and a still-matching profile revision,
  supplement hash, raw key state, and base-prompt version/hash. It takes
  `_config_lock` across re-read → compare → mutate → **checked save**
  (failures propagate) → restore in-memory state on failure. The checked save
  writes `pending_refinement_applies[profile_id]` with `kind=accept`. A
  flagged deterministic regression requires `acknowledge_regression=true`;
  missing judge verdicts require `acknowledge_judge_gaps=true`; a judge-only
  regression requires `acknowledge_judge_regression=true`. On success it
  appends `accepted` with `applied_revision` and `acknowledge_judge_regression`,
  then clears the entry with another checked save. If the accepted append
  fails, the entry remains and profile writes for that profile are blocked
  until reconciliation classifies the save as applied or failed. If only the
  marker-clear save fails, `accepted` remains authoritative; the profile stays
  blocked and reconciliation retries only the clear (never a second
  `accepted`). On save failure it restores the profile object, clears the
  entry in memory, and appends `apply_failed`.
- `cancel` is allowed only while `proposing` or `evaluating`. It writes
  `cancel_requested_at` and returns 202 `requested` for an active lease; if no
  live lease remains it may transition directly to `cancelled`. The owner’s
  heartbeat/scheduling/result-write transactions observe the flag, refuse
  further work, and then append `cancelled` while clearing the lease. An
  in-flight provider call is not killed and its result is not persisted after
  the request. If completion and cancel race, the first `BEGIN IMMEDIATE` +
  status CAS wins: `evaluation_completed` is accepted only with
  `cancel_requested_at IS NULL`; otherwise the worker records `cancelled`.
  Cancellation is not a decision on the candidate.
- `reject` is allowed from `proposal_ready`, `evaluated`,
  `evaluation_incomplete`, or `apply_failed`; it records the decision/reason
  and writes nothing to the profile.
- `rollback` requires the accepted event and current revision ==
  `applied_revision`; inside the same checked `_config_lock` transaction it
  restores the raw key state (removing `system_prompts["rlm"]` when the
  proposal recorded it absent), bumps the revision, writes
  `pending_refinement_applies[profile_id]` with `kind=rollback`, and saves.
  After the save it appends `rolled_back` with `rollback_of` and clears the
  entry. If the append fails, profile writes stay blocked until reconciliation
  classifies the persisted state. If only the marker-clear save fails,
  `rolled_back` remains authoritative; reconciliation retries only the clear.
  A checked-save failure restores the in-memory profile, appends
  `rollback_failed`, and leaves the head `accepted` because the accepted prompt
  is still what is persisted.
- Every transition that leaves `proposing`/`evaluating` (`proposal_ready`,
  `evaluated`, `evaluation_incomplete`, `rejected`, `failed`, `interrupted`,
  `cancelled`) clears `lease_owner`, `lease_token`, `lease_expires_at`, and
  `job_kind` in the same status-CAS transaction and stops the heartbeat. The
  partial unique lease slot is free immediately; the next job does not wait
  for TTL expiry.
- Profile update/delete, or starting a new proposal/evaluation job, for a
  profile with a `pending_refinement_applies` entry or an unresolved
  `applying` status first runs reconciliation under `_config_lock`. If
  reconciliation cannot classify the persisted state, the operation returns
  409 and nothing changes; the entry must be resolved first.
- One live job per profile, enforced atomically in SQLite by the partial unique
  lease index; every job result write re-checks the lease token and the
  transactionally allocated event sequence. Profile deletion is blocked while
  refinement history exists unless explicitly forced, and the forced path
  records the deletion in history.

### 4.3 Event state machine and transition table

Every transition is one SQLite transaction: validate the preflight token and
lease (where relevant), `UPDATE refinements SET status = :to,
last_sequence = last_sequence + 1 [, lease_owner = NULL, lease_token = NULL,
lease_expires_at = NULL, job_kind = NULL when leaving an active job]
WHERE id = :id AND status IN (:allowed) [AND cancel_requested_at IS NULL
for a successful completion]` and require rowcount 1, insert the event with
the returned sequence, then commit. `UNIQUE(refinement_id, sequence)` is the
backstop against a concurrent append. The same transaction stops/closes the
job lease. A failed CAS/insert rolls the whole transition back. Cancellation
itself is a separate CAS that sets `cancel_requested_at` while the job remains
active; the owner performs the final `cancelled` transition and lease clear.

| From | Trigger / event | To |
|---|---|---|
| (none) | `POST /refinements` with a consumed proposal token; appends `proposal_requested` | `proposing` |
| `proposing` | proposal LLM succeeds; `proposal_created` | `proposal_ready` |
| `proposing` | LLM error / shutdown / cancel; `failed` / `interrupted` / `cancelled` | terminal |
| `proposing` or `evaluating` | `POST /cancel`; sets `cancel_requested_at` (202 requested) | status unchanged (active) |
| `proposing` | owner observes cancel before starting; `cancelled` | terminal |
| `proposal_ready` | `POST /evaluate` with a consumed evaluation token; `evaluation_started` | `evaluating` |
| `proposal_ready` | reject | `rejected` |
| `evaluating` | every case/rep produced both arms and no cancel flag; `evaluation_completed` | `evaluated` |
| `evaluating` | any missing case/rep/arm; `evaluation_incomplete` | `evaluation_incomplete` |
| `evaluating` | owner observes cancel before a completion CAS; `cancelled` | terminal |
| `evaluating` | completion CAS wins before the cancel flag; `evaluation_completed` | `evaluated` |
| `evaluating` | error / shutdown; `failed` / `interrupted` | terminal |
| `evaluation_incomplete` | retry with a new token; `evaluation_started` | `evaluating` |
| `evaluation_incomplete` | reject | `rejected` |
| `interrupted` (evaluation) | retry with a new token; `evaluation_started` | `evaluating` |
| `interrupted` (proposal) | no retry; start a new refinement | terminal |
| `evaluated` | accept; `applying` | `applying` |
| `evaluated` | reject | `rejected` |
| `applying` | checked save succeeded; `accepted` | `accepted` |
| `applying` | checked save failed or reconciliation failed; `apply_failed` | `apply_failed` |
| `apply_failed` | accept retry; `applying` | `applying` |
| `apply_failed` | reject | `rejected` |
| `accepted` | rollback checked save succeeded; `rolled_back` | `rolled_back` |
| `accepted` | rollback save failed; `rollback_failed` (head stays accepted) | `accepted` |
| `accepted` | forward accept/reject/evaluate | illegal (409; create a new refinement) |
| terminal (`rejected`, `rolled_back`, `failed`, `cancelled`) | any | illegal (409; create a new refinement) |

- Every transition leaving `proposing` or `evaluating` clears all lease fields
  in the same CAS transaction. `proposal_ready` therefore frees the profile
  immediately for evaluation; a finished evaluation frees it for the next
  refinement without waiting for TTL.
- `applying` is written before the checked save; `accepted` only after the
  persisted revision and target key hash are read back and match.
  Reconciliation moves `applying` to `accepted` or `apply_failed` using
  `pending_refinement_applies[profile_id]`.
- If the terminal event is appended but clearing the marker fails, the terminal
  event is authoritative; the profile remains blocked and reconciliation
  retries only the clear, never a duplicate terminal event.
- `apply_failed` leaves the profile at the base revision and is retryable;
  `evaluation_incomplete`/`interrupted` evaluation is retryable with a new
  preflight token. A proposal interruption/failure is not retried in place
  (the evidence/baseline may have moved); create a new refinement.
- Reject is allowed from `proposal_ready`, `evaluated`, `evaluation_incomplete`,
  and `apply_failed`; cancel is allowed only while a job is `proposing` or
  `evaluating` and is persisted in `cancel_requested_at`.
- If the owner dies with a cancel request pending, the expired lease is
  reclaimed as `interrupted`/`cancelled` on startup; no result writes from the
  old owner are accepted.
- `refinements.status` is a projection of the highest-sequence event plus job
  state; it can be rebuilt from `refinement_events` and must never overwrite
  evidence.
- Exports of a decision use the decision-time event payload ordered by
  `sequence`, not the current head row or mutable profile text.

## 5. Phasing and estimates

Estimates are single-developer effort after #77 items 2–3; they exclude the
blocked period.

| Phase | Deliverable | Estimate |
|---|---|---|
| 0 | Re-baseline `longdoc-v1` post-#77; confirm the six-case holdout and judge-only threshold; confirm the recorded B1/proposal-CP/strict-legacy/full-UI decisions | 1–2 days |
| 1 | Schema v3 (append-only sequenced events, head/lease/cancel, preflights with immutable snapshots, run content+benchmark provenance), profile revisions/raw key state, checked save + lock-boundary + per-profile apply map, `get_run()` outcome fix, source-runs query, ProfileCard preservation fix | 7–9 days |
| 2 | Proposal prompt/service, evidence digest + persisted hash/truncation, proposal preflight snapshot/token, durable proposal job, review API | 5–7 days |
| 3 | Evaluation service (two-slot matrix, reps, telemetry links, completeness gate, matcher/judge/judge-only threshold), evaluation preflight snapshot/token, lease heartbeat/release, durable cancel, transition CAS | 9–11 days |
| 4 | Accept/reject/rollback (checked transaction, raw-key restore, per-profile marker map/clear reconciliation, recovery boundary, apply/rollback failure), exports, history API | 5–7 days |
| 5 | Refinements UI (source wizard, proposal/evaluation cost confirmation, diff/reasoning, evaluation/regression table, history, rollback) | 6–8 days |
| 6 | Docs, fake-provider e2e, fault-injection/concurrency tests, manual gate, changelog | 4–5 days |

Suggested PR split (one logical change each): schema/event log/run provenance/
profile revisions/checked-save primitive/source query; proposal; evaluation +
preflight/lease/transitions; decision/rollback/export; UI; docs/e2e/fault
injection. Rough total: **5–7 weeks** with the review hardening.

## 6. Test strategy and acceptance

Fast, hermetic tests; no network or paid calls.

- **Migration:** fresh v3; v2→v3 with seeded runs/steps; idempotent reopen;
  legacy rows have null profile/content provenance; `refinements`,
  `refinement_events`, `refinement_preflights`, `refinement_runs`, the run
  provenance columns, and the append-only triggers exist; event UPDATE/DELETE
  is rejected; `(refinement_id, sequence)` is unique and `last_sequence`
  matches the highest event after append.
- **Profile revisions/CAS/persistence:** create/update bump; accept refuses a
  changed revision/hash, raw key state, or base-prompt version/hash; a
  profile edit after accept's first read is serialized by `_config_lock` and
  produces a 409 rather than being overwritten; injected write, fsync, and
  os.replace failures propagate on both profile update and accept, restore
  the in-memory profile/revision, and record `apply_failed` (never
  `accepted`); rollback refuses a changed profile; base prompt file bytes
  unchanged after accept/rollback.
- **Apply marker:** a save-success/event-append-failure leaves
  `pending_refinement_applies[profile_id]`; a profile edit is blocked until
  reconciliation; a persisted target revision/key hash produces `accepted`,
  while a mismatch or an out-of-band revision produces `apply_failed`; a
  concurrent edit after accept's initial read cannot be mistaken for “never
  applied”; the entry is cleared after either classification. Two profiles
  with simultaneous markers remain isolated (accepting B does not overwrite
  A). If the terminal event is written but clearing the entry fails, the
  event remains authoritative, the profile stays blocked, and restart/
  reconciliation retries only the clear without a duplicate terminal event;
  both accept and rollback are tested. Starting a new proposal/evaluation
  while the entry is unresolved returns 409.
- **Template rollback:** a proposal from a template-backed profile records
  `base_supplement_present=false`; accept creates the custom RLM key; rollback
  removes the key, and a later template edit becomes visible again.
- **Proposal:** fake LLM returns JSON (and fenced JSON; malformed reply → no
  candidate + failed event); diff is server-computed; candidate length cap;
  evidence digest caps traces/chars and redacts secrets; the exact digest,
  digest hash, truncation metadata, and proposal-prompt hash/version are
  persisted; source ids are validated against profile/content provenance;
  proposal preflight returns provider/model, digest/hash, token limits, and
  `cost_known` (with `max_cost_usd=null` for an unknown price), and stores the
  complete rendered snapshot; deleting source telemetry after preflight does
  not change the proposal request; a changed profile/base prompt at
  consumption is rejected; the token is consumed exactly once with a matching
  canonical selection; missing, expired, replayed, or mismatched tokens are
  rejected.
- **Evaluation:** fake adapters produce per-case baseline/candidate results;
  two-slot configs differ only in `system_prompt_extra`; temperature is 0;
  every arm produces a telemetry run and a `refinement_runs` link; content
  hashes and benchmark provenance are persisted; skipped public-text cases and
  zero deterministic cases make the evaluation `evaluation_incomplete` and
  non-acceptable; one-arm/missing-repetition faults do the same; judge
  failures set `judge_complete=false` without masking deterministic results;
  regression flags use the per-case majority rule; a judge-only case whose
  candidate mean falls below `baseline_mean - judge_regression_threshold`
  sets `judge_regression_cases` and requires the matching acknowledgement;
  a judge-only holdout without a configured judge is rejected; content-hash/
  provenance overlap blocks a case with no override; changing the provider,
  model, profile/base prompt, judge config, threshold, dataset/version, case
  ids, case content hash, budgets, or pricing/cost estimate between preflight
  and consumption is rejected; changing corpus content during evaluation makes
  it incomplete.
- **Leases/preflight:** proposal and evaluation preflights return estimates
  and durable single-use tokens; concurrent consumption, replay after
  completion, restart, expiry, and request/selection mismatch are all
  rejected; two concurrent starts produce one lease and one 409; a stale
  worker's write is rejected; a heartbeat renews before `lease_expires_at`;
  an expiry during an in-flight provider call lets another worker reclaim
  without the stale worker writing or starting another paid call;
  restart-expired leases become `interrupted`; successful proposal/evaluation
  transitions clear all lease fields atomically and the next job starts
  immediately; a cancel written by another connection is observed by the
  owner, result writes stop, and completion-versus-cancel races resolve by the
  first status CAS.
- **Decision flow/transitions:** accept writes only after a complete,
  holdout-verified evaluation and only on explicit action; reject leaves the
  profile revision and text untouched; rollback restores the exact raw key
  state and appends history; an `applying` append failure aborts before
  mutating the profile; a save-success/event-append failure is reconciled
  before any profile edit; retry from `evaluation_incomplete`/`interrupted`
  and accept retry from `apply_failed` follow the transition table; every
  illegal transition is 409; concurrent accept/rollback returns 409; terminal
  event payloads cannot be updated, event replay is `sequence`-ordered with
  equal timestamps, and an export reproduces the decision-time snapshot;
  the accepted payload records `acknowledge_regression`,
  `acknowledge_judge_gaps`, and `acknowledge_judge_regression`; a marker-clear
  failure leaves the terminal event authoritative and only retries the clear;
  a lost marker with a surviving `applying` event records
  `apply_failed` + `recovery_unverifiable`, never `accepted`.
- **Exports:** run JSONL has the run id/outcome/content hash and linked
  refinement events; deleted evaluation runs do not delete events or links.
- **API/frontend:** profile-scoped routes, one-active-job 409, source filter
  degraded/outcome with unverified legacy rows excluded, proposal and
  evaluation preflight cost/provider/snapshot confirmation (including
  unknown-price state), durable cancel returns 202 and shows requested/cancelled,
  diff rendering, default reject on deterministic/judge-only regression,
  judge-gap acknowledgement, history and rollback states, `ProfileCard`
  save/clone/export preserving custom prompts.
- **End-to-end (fake provider):** select sources → proposal preflight →
  propose (source telemetry deleted afterward; snapshot still used) →
  evaluation preflight → evaluate held-out baseline vs candidate → accept
  (marker written/cleared) → new run carries the accepted fingerprint →
  rollback (raw key restored) → history shows all events in sequence order.
  Separate cases exercise immediate proposal-to-evaluation start after lease
  release, cross-connection cancel during a call, two-profile marker
  isolation, `evaluation_incomplete` retry, apply-save/event-append failure
  reconciliation, marker-clear failure, rollback failure, proposal-token
  replay, lease expiry during a call, and illegal transitions. No real
  provider calls.

Hard acceptance criteria:

1. No profile write occurs on propose, evaluate, or reject.
2. Accept is possible only after a complete, holdout-verified evaluation and
   only with a matching profile revision, raw key state, supplement hash, and
   base-prompt version/hash; a deterministic regression, judge-only
   regression beyond threshold, or judge gap requires the matching explicit
   acknowledgement.
3. The base v2 prompt file is never modified by this feature.
4. Every accepted/rejected refinement has an immutable, `sequence`-ordered
   event record with source trace ids, evaluation case ids, baseline/candidate
   run ids, judge model/prompt version, deltas, decision, and the exact
   evidence digest sent to the proposer.
5. A candidate is never evaluated on a case that shares a document hash or
   benchmark dataset/version/case id with a source run; source runs without a
   content hash cannot make a gate-eligible proposal.
6. No candidate is auto-applied, even when every metric improves.
7. A failed config write/fsync/replace is never reported as accepted, and the
   in-memory profile is restored to the base revision.
8. The revision/hash check, mutation, checked save, and restore are one
   `_config_lock` transaction; a concurrent profile edit cannot be overwritten.
9. Rollback restores the raw `system_prompts["rlm"]` presence/value, including
   removing the key when it was absent, so template following resumes.
10. A zero-case, skipped-case, one-arm, or interrupted evaluation is
    `evaluation_incomplete` and cannot be accepted.
11. Proposal and evaluation paid work cannot start without a durable,
    single-use token bound to the provider/selection/estimate; proposal
    preflight freezes the exact evidence/prompt snapshot, evaluation preflight
    freezes case/config/judge/threshold state, and concurrent consumption,
    replay, restart, expiry, mismatch, or state drift are rejected.
12. Within the supported checked-API boundary, a save-success/event-append
    failure is reconciled through the per-profile config marker before any
    profile write, so “applied then edited” and “never applied” are never
    conflated. External replacement or an older binary that erases the marker
    is outside that boundary; a surviving `applying` event is then recorded as
    `apply_failed` + `recovery_unverifiable`, never `accepted`.
13. Event replay is deterministic (`sequence`), illegal transitions are 409,
    every completion/failure/cancel transition clears the lease atomically,
    durable `cancel_requested_at` reaches the owner across processes, and
    lease heartbeat renewal prevents a second worker from starting duplicate
    paid work while a provider call is in flight.

## 7. Risks and open questions

1. **B1 detaches template profiles.** Owner-selected B1 reuses
   `system_prompts["rlm"]`, so accept deliberately stops following a named
   template for RLM. The UI must state this before confirmation, and rollback
   must remove the key when it was absent. B2 (a second supplement field) is
   not in 1.x.
2. **Profile lifecycle bug.** `ProfileCard` currently wipes custom prompts on
   save and omits them from clone/export. This must be fixed in Phase 1 or
   accepted candidates will silently disappear.
3. **Cross-store accept.** Profiles live in `config.json`; events in SQLite.
   The per-profile `pending_refinement_applies` map, the `applying` state
   machine, the checked-save primitive, and the single `_config_lock` boundary
   are all required. `_save_config_locked()` currently swallows failures, so
   the checked variant is not optional. Terminal events are authoritative if
   clearing an entry fails. The guarantee is bounded to writers that use the
   current checked API; an older binary or external config replacement can
   erase the entry, in which case reconciliation records
   `apply_failed`/`recovery_unverifiable`, never `accepted`.
4. **Paid work and duration.** Proposal calls are also paid and must use the
   proposal preflight (provider, exact evidence, token limits, known/unknown
   cost). Evaluation preflight returns the maximum case/run count and cost/
   time estimate; each token is durable and single-use. A global maximum
   (e.g. 12 runs or a cost cap) should be configurable.
5. **Provider nondeterminism.** Temperature 0 does not guarantee identical
   answers; per-case majority and per-rep display are the mitigation. The gate
   is advisory, as the issue requires.
6. **Benchmark corpus availability.** Public-text cases need `--fetch` and a
   corpus directory; a skipped case makes the evaluation `incomplete`, not a
   quietly smaller holdout. The preflight lists missing cases so the user can
   fetch or adjust the selection before spending.
7. **Prompt injection/privacy.** Source traces are untrusted content and may
   contain secrets. Delimit evidence, redact, show exactly what is sent, and
   keep the human diff review. Record the proposal provider.
8. **Judge provenance.** `JudgeService` cannot supply the required judge model
   and prompt version; use `JudgeScorer`. Judge-less evaluation is
   deterministic-only and clearly marked; judge-only holdout cases are
   ineligible without a configured judge and complete verdicts.
9. **Profile deletion.** Accepted refinement history must not be orphaned
   silently; block deletion of profiles with active evidence or define a
   retention rule.
10. **#77 re-baseline.** Compaction and batched subcalls change long-document
    behavior; do not reuse pre-#77 benchmark results as baselines.
11. **Concurrency and leases.** One live job per profile, enforced atomically
    in SQLite by the partial unique lease index. The lease TTL must exceed the
    maximum provider request timeout and be renewed by a heartbeat with an
    owner-token CAS; on renewal failure the worker stops scheduling/writing.
    Every completion/failure/cancel transition clears all lease fields in the
    same transaction, and `cancel_requested_at` is the durable cross-process
    cancel signal. Provider rate limits and the two-arm parallelism need a
    small scheduler/limit. Server restart marks expired leases `interrupted`;
    overlapping reload processes cannot both own a job.
12. **Issue scope creep.** Memories/skills/subagent specs, auto-apply,
    multi-candidate search, and Direct/RAG supplements remain out of scope.
13. **Config-copy trap.** `_copy_config_for_slot()` rebuilds `RunConfigDTO`
    field-by-field and already omits newer fields. Do not add a profile id or
    prompt hash to `RunConfigDTO` for this feature without updating that
    function and its tests; prefer carrying attribution alongside the slot in
    the evaluation service and telemetry fields.
14. **Slot cap.** `MAX_SLOTS = 10`. Two variants fit, but two providers × four
    modes × two variants does not; the UI must cap arms or run cases
    separately. The issue's baseline-vs-candidate scope is two arms.
15. **Bench is not an evidence ledger.** `MatrixBenchmarkRunner` discards slot
    ids, has no variant dimension, and writes no telemetry. Do not build the
    gate on its report; use it only for the dataset/materialisation/scoring
    patterns.
16. **Provenance migration is forward-only.** `content_sha256` and benchmark
    provenance are written for new runs; legacy rows remain unverifiable and
    cannot be source traces for a gate-eligible proposal. The UI must say this
    rather than implying the old rows were checked.
17. **Event log projection drift.** `refinements.status` is a cache of the
    latest event plus lease state. Add a repair/rebuild path from
    `refinement_events` and test a head row that disagrees with the log; do
    not let the head row overwrite evidence.
18. **Raw key semantics.** Reusing `system_prompts["rlm"]` means accept
    deliberately detaches a template-backed profile. The UI must show the
    detach before confirmation, and rollback must remove the key when it was
    absent; both need explicit copy and tests, or users will not understand
    why the template stopped applying.
19. **Preflight tokens are bearer credentials.** Store only their SHA-256,
    never log the raw token, bind every request field into the canonical
    selection, expire them, and retain consumed rows only for audit. Purge
    policy must not break replay detection during the retention window.
20. **Config marker evolution.** `pending_refinement_applies` is part of the
    persisted config schema; older configs must load with it empty, and a
    malformed entry must fail closed (block that profile’s edit) rather than be
    ignored. Reconciliation needs tests for both accept and rollback entries
    and for two profiles with simultaneous markers.

## 8. Explicit non-goals for this feature

- Memories, executable skills, subagent specs, or other Prime Agent harness
  state.
- Auto-applying a candidate on a passing gate.
- Optimising on the source traces or evaluating them as a held-out set.
- Hard-blocking a human accept on a single-case regression.
- Direct/RAG profile prompts (they are not injected today).
- A general prompt-versioning system for every prompt file; only the
  refinement event and the proposal prompt carry versions.
