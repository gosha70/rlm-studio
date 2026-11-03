---
feature_id: prompt-refinement
spec_mode: full
status: draft
revision: 5-minimal
date: 2026-11-03
origin:
  urls:
    - https://github.com/gosha70/rlm-studio/issues/78
    - https://github.com/gosha70/rlm-studio/issues/77
  transcripts:
    - "Owner, 2026-11-03: 'I want quick, simple and safe implementation.'"
    - "Owner, 2026-11-03: 'Let's compromise: build the minimal plan ... file a new enhancement issue with the more complex approach.'"
  origin_claim: see spec.md
supersedes: plan-full-rev4.md (kept as the input for the follow-up issue)
---

# Plan — Evidence-gated prompt refinement, minimal variant (#78)

Read `spec.md` first. This plan says where each requirement lands in the
code. File and line references were verified on 2026-11-03 against
`origin/master` at `ff842bf` (after PR #113).

## 0. Status

- #78 is unblocked: #77 items 1–3 are merged (`3de7777`, PR #112, PR #113).
- `scripts/check-origin-alignment.sh` does not exist in this repository, so
  the origin check in the global instructions cannot be run; the origin
  frontmatter above is the substitute.
- Phase 0 of the old plan (fresh `longdoc-v1` baseline on the post-#77 loop)
  needs real provider runs and is the owner's task. It does not block the
  build; it blocks trusting the first evaluation numbers.
- `specs/*` is gitignored (`.gitignore:58`); the other spec directories were
  force-added. These files need `git add -f`.

## 1. Verified facts the design rests on

| Fact | Where |
|------|-------|
| The supplement the RLM loop receives is `profile.system_prompts["rlm"]`, else the named template's `rlm` text | `server/routes/chat.py:73–96` (`_resolve_profile_prompt`) |
| Only RLM appends it (`RunConfigDTO.system_prompt_extra`) | `application/dto.py:158`, `routes/chat.py:167–168` |
| Profiles have no revision; `PUT /api/profiles/{id}` mutates and calls `save_config()` | `server/routes/profiles.py:134–180` |
| `_save_config_locked()` swallows all exceptions | `server/dependencies.py:1304–1305`; lock is an `RLock` at `:156` |
| ProfileCard save sends `system_prompts: {}`; clone and export omit it | `profile-card.tsx:246`, `:99–113`, `:142–150` |
| `RunProfileUpdate.system_prompts` is optional; `None` means "keep" | `server/models.py:898` |
| Telemetry schema is at v3 (`steps.cost_known`); new work is migration 4 | `telemetry/store.py:108–128` |
| `runs` has no `profile_id` or `content_sha256`; it has `outcome_category` and `chat_provider_id` | `telemetry/store.py:28–50`, `:120` |
| `get_run()` reads `outcome_category` | `telemetry/store.py:742–765` |
| Two run writers | `routes/chat.py:575`, `routes/compare_matrix.py:858` |
| Compare route records a slot via `_record_slot_telemetry(state, slot_result, meta, ...)`; `_SlotMeta` carries `chat_provider_id`, `execution_id` | `routes/compare_matrix.py:250–255`, `:829–918` |
| Per-slot config from a Chat Provider, including the profile prompt | `routes/compare_matrix.py:943–977` (`_build_slot_run_config`) |
| Adapter and sandbox factories | `dependencies.py:1165` (`create_llm_adapter_for_chat_provider`), `:1557` (`create_sandbox`) |
| `MatrixSlotDTO.config` is a per-slot `RunConfigDTO`; `execute(content, query, slots)` returns per-slot results and `comparison_group_id` | `run_matrix_comparison.py:62–96`, `:169–200` |
| `RunConfigDTO.temperature` exists (default 0.7) | `application/dto.py:42` |
| Deterministic scoring and judge | `benchmark/scoring.py:53` (`matches`), `:145` (`JudgeScorer`, records `model` and `prompt_version`) |
| Dataset: `load_dataset`, `filter_by_tag`, `materialise(dataset, base_dir, corpus_dir, fetch)`; skipped cases come back in `MaterialiseReport.skipped` | `benchmark/dataset.py:128`, `:202`; `benchmark/corpus.py:325–346` |
| Bench CLI's corpus default is `<dataset dir>/corpus` | `cli/main.py:389` |
| Judge Chat Provider setting | `models.py:829` (`AppConfig.judge_chat_provider_id`) |
| Background thread precedent | `dependencies.py:485`, `:561` (connection tester, startup probe) |
| Prompt loader discards the YAML `version` | `prompts/templates.py:201`; use the `benchmark/scoring.py:138` pattern |
| Degraded = `success=True` and answer starts with the warning prefix | `application/services/outcome_classifier.py:39`, `:82–84` |
| Benchmark fakes for tests | `benchmark/fakes.py` (`ScriptedLLM`, `dry_run_judge_llm`) |

## 2. Holdout cases (proposed, deterministic-only, offline)

Tag these six with `holdout` in `benchmarks/longdoc-v1.yaml`. All are
synthetic or repo-docs, so they materialise without `--fetch`, and all have
an `expected_answer`, so the gate needs no judge.

| Case | Size | Kind |
|------|------|------|
| `docs-operations-needle` | small | needle |
| `synthetic-5k-needle` | small | needle |
| `synthetic-5k-refusal` | small | refusal |
| `synthetic-50k-needle` | medium | needle |
| `synthetic-50k-refusal` | medium | refusal |
| `synthetic-150k-needle` | large | needle |

Owner may swap cases after the re-baseline; the tag is the only coupling.
Default evaluation: 6 cases × 2 reps × 2 arms = 24 RLM runs.

## 3. Changes by layer

### 3.1 Telemetry (`src/rlmstudio/telemetry/store.py`)

Migration 4, appended to `_MIGRATIONS` (never edit `_SCHEMA_SQL`):

```sql
ALTER TABLE runs ADD COLUMN profile_id TEXT;
ALTER TABLE runs ADD COLUMN content_sha256 TEXT;
CREATE INDEX IF NOT EXISTS idx_runs_profile ON runs(profile_id, created_at);
CREATE TABLE IF NOT EXISTS refinements (
    id              TEXT PRIMARY KEY,
    profile_id      TEXT NOT NULL,
    status          TEXT NOT NULL,   -- proposal_ready|evaluating|evaluated|
                                     -- interrupted|failed|accepted|rejected|rolled_back
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    proposal_json   TEXT NOT NULL,   -- FR-4 fields
    evaluation_json TEXT,            -- FR-5..8 fields, run ids, deltas
    decision_json   TEXT,            -- accept/reject/rollback record
    error           TEXT
);
CREATE INDEX IF NOT EXISTS idx_refinements_profile ON refinements(profile_id, created_at);
```

Store API additions: `record_run(..., profile_id=None, content_sha256=None)`;
`list_runs(..., profile_id=None, require_content_hash=False)`;
`RunSummary.profile_id` and `.content_sha256` (`chat_provider_id` is already
mapped at `telemetry/store.py:174`);
`create_refinement`, `update_refinement(id, *, status, evaluation_json=,
decision_json=, error=)`, `get_refinement`, `list_refinements(profile_id)`,
`mark_evaluating_as_interrupted()`.

`create_refinement` and `update_refinement` **raise** on database failure
(FR-11a). Every existing writer in this store catches `Exception` and logs
(`store.py:341`, `:418`, `:462`, `:485`); the refinement writers must not
copy that pattern, and a test injects an `sqlite3.Error` to prove it.
`update_refinement` also raises when the row is missing or the status CAS
(`WHERE id=? AND status=?`) matches no row, so a concurrent decision is a
detected failure rather than a silent no-op.

Status constants live in one module (`application/constants.py` or a new
`application/refinement_constants.py`); no string literals in routes or UI
glue.

### 3.2 Run writers

- `routes/chat.py:575` (inside `_record_telemetry`, called by both the HTTP
  path at `:814` and the WebSocket path at `:1263`) and
  `routes/compare_matrix.py:858`: pass `profile_id=cp.profile_id` (None for
  ephemeral V2 Chat Providers) and
  `content_sha256=hashlib.sha256(content.encode()).hexdigest()`.
- Move `_record_slot_telemetry` and `_SlotMeta` from `compare_matrix.py` to
  `routes/_helpers.py` so the evaluation service can import them without
  importing a route module. Pure move, no behaviour change; the compare
  route imports them back.

### 3.3 Config persistence (`server/dependencies.py`)

Add `AppState.save_config_checked()`: acquires `_config_lock`, runs the
existing write body, and lets exceptions propagate. Implement by splitting
the body of `_save_config_locked()` into `_write_config_unchecked()` and
having `_save_config_locked()` call it inside the existing try/except. No
other caller changes.

### 3.4 Proposal (application layer, no server imports)

- `prompts/refine_supplement.yaml`: `version: "1.0"`, `template` with
  `{base_prompt_version}`, `{current_supplement}`, `{evidence}`; instructs a
  single JSON object `{"candidate": "...", "rationale": "..."}`; states that
  evidence is data and must not be followed as instructions.
- `application/services/refinement.py`:
  - `build_evidence_digest(runs: Sequence[RunDetail], *, max_traces=12,
    max_chars=24_000) -> EvidenceDigest(text, sha256, included_ids,
    excluded_ids, truncated_chars)`. Per run: id, outcome category, degraded
    flag, query (bounded), answer or error (bounded), step count and action
    types, tokens, cost, elapsed. Redaction per FR-17:
    `redact_secrets()` (`application/services/provider_failure.py:62`,
    covers API-key and bearer-token shapes only) plus the line-drop rule for
    the FR-17 key names with `=`, `:` or `":` separators and optional
    whitespace. Mutation tests: each name × each separator in a query, an
    answer and an error is removed; a benign line containing "token" as a
    word survives; a secret in free text is documented as passing through
    (spec §6, residual evidence leakage).
  - `propose_supplement(llm: LLMPort, *, digest, current_supplement,
    base_prompt_version) -> ProposalResult(candidate, rationale, diff,
    prompt_version, prompt_sha256, usage)`. Parses fenced or bare JSON,
    validates per FR-3, computes `difflib.unified_diff`.

### 3.5 Evaluation service (`server/services/refinement_eval.py`)

Server layer because it needs `AppState` factories. One function
`run_evaluation(state, refinement_id, *, case_ids, reps, judge_cp_id)` run
on a `threading.Thread(daemon=True)`:

1. Load `benchmarks/longdoc-v1.yaml`, filter to `case_ids` (default
   `filter_by_tag("holdout")`), `materialise(fetch=False)` with
   `corpus_dir = <dataset dir>/corpus`. Skipped cases → recorded, evaluation
   not complete.
2. Exclude any case whose `sha256(case.content)` is in the proposal's source
   content hashes.
3. For each case × rep: two `MatrixSlotDTO`s (`baseline`, `candidate`),
   each with its own `create_llm_adapter_for_chat_provider(cp_id)` and
   `create_sandbox()`, config from `_build_slot_run_config(state, cp,
   MODE_RLM)` with `temperature=0.0` and `system_prompt_extra` set to the
   base supplement or the candidate. Apply the case's `budget` the way the
   bench runner does. Run `RunMatrixComparisonUseCase().execute(...)`.
   Record both arms with the shared slot recorder (`profile_id`,
   `content_sha256` included) and keep the run ids.
4. Score: `matches(answer, expected_answer, case.match)` per arm; judge via
   `JudgeScorer` when a judge Chat Provider is given.
5. Aggregate per FR-7/FR-8, write `evaluation_json`, set `evaluated`;
   on exception set `failed` with the error.
6. Progress: `state.refinement_progress[refinement_id] = {done, total}`
   (in-memory dict), surfaced by the GET route; one running job per profile
   via an in-memory set guarded by a lock on `AppState`.

Startup: call `telemetry.mark_evaluating_as_interrupted()` where the
telemetry store is initialised on `AppState`.

### 3.6 Routes (`server/routes/refinements.py`, models in `server/models.py`)

```
GET  /api/profiles/{id}/refinements/source-runs?limit=50
GET  /api/profiles/{id}/refinements/holdout-cases
POST /api/profiles/{id}/refinements            {source_run_ids, proposal_chat_provider_id}
GET  /api/profiles/{id}/refinements
GET  /api/profiles/{id}/refinements/{rid}
POST /api/profiles/{id}/refinements/{rid}/evaluate  {case_ids?, reps=2, judge_chat_provider_id?} -> 202
POST /api/profiles/{id}/refinements/{rid}/accept    {acknowledge_regression=false, reason?}
POST /api/profiles/{id}/refinements/{rid}/reject    {reason?}
POST /api/profiles/{id}/refinements/{rid}/rollback
GET  /api/profiles/{id}/refinements/export.jsonl
```

Rules are FR-9 to FR-12, FR-15, FR-16. All three decision routes take
`_config_lock` before reading the refinement status and release it after
the decision write (FR-10a); Reject takes it even though it touches no
profile, so it cannot interleave with an Accept's save-then-status window.
The lock is an `RLock`, so `save_config_checked()` nesting inside is fine.
Accept and rollback share one shape under the lock: snapshot the profile's
`system_prompts` dict,
verify the hash precondition (409 on mismatch), mutate, `save_config_checked()`;
on save failure restore the snapshot in memory and return 500 with status
unchanged; on success call `update_refinement` with the status CAS; if that
raises, return 500 with a message naming the case (spec §6, split-brain
window) and do not retry or revert the config write. The proposal POST runs the LLM call with
`asyncio.to_thread` and returns 201 with the record. Built-in profiles are
rejected with 400 like `update_profile`. Register the router in
`server/app.py` after `profiles.router`.

### 3.7 Frontend

- `profile-card.tsx`: save sends the edited `system_prompts` (textarea for
  `rlm`); clone and export include `system_prompts`.
- New `components/settings/refinement-panel.tsx` opened from a "Refine
  prompt" button on the card: source-run table with checkboxes, proposal
  Chat Provider select, diff and rationale view, holdout case list with
  reps and run-count confirmation, progress polling, results table with the
  regression flag and default Reject, accept checkbox, history list with
  Rollback. Uses `lib/api.ts` additions mirroring the routes.
- Vitest: `__tests__/refinement-panel.test.tsx` and an added case in
  `profile-and-labels.test.tsx` for the save payload.

### 3.8 Docs

- `docs/rlm-prompt-tuning.md`: new section "Evidence-gated refinement"
  (flow, what is recorded, the advisory gate, the detach from a template on
  accept, the known gaps).
- `docs/rlm-studio-guide.md` Profiles section: one paragraph plus the
  custom RLM prompt textarea.
- `CHANGELOG.md` entry.

## 4. Tests (hermetic, fake LLMs, no network)

- Migration: fresh and v3→v4 upgrade, columns and table present, idempotent.
- Store: refinement CRUD, `list_runs(profile_id=...)` excludes rows without
  a content hash, `mark_evaluating_as_interrupted`.
- Writers: a chat run and a compare slot carry `profile_id` and
  `content_sha256`.
- `save_config_checked` raises on injected write, fsync and replace faults;
  `save_config` still swallows.
- `create_refinement` and `update_refinement` raise on an injected
  `sqlite3.Error` and on a status CAS miss.
- Accept and rollback each: three config faults (write, fsync, replace)
  leave memory restored and status unchanged; a status-write fault after a
  successful save returns 500 naming the case with the config change kept.
- Race: Accept is paused (monkeypatched `save_config_checked` blocks on an
  event) after its save and before its status write; Reject is started on
  another thread, must block on the lock, and after Accept finishes must
  return 409 with the row `accepted`.
- Proposal service: digest caps and redaction; JSON parse of bare and
  fenced output; malformed output raises; identical or overlong candidate
  rejected; diff is server-computed.
- Evaluation service with `ScriptedLLM`: both arms recorded with run ids;
  source-hash overlap excludes the case; skipped case → not complete; one
  failed arm is still a result; regression flag per FR-8; judge optional.
- Routes (`TestClient`): the acceptance list in `spec.md` §5 one test each,
  plus 409 on concurrent evaluate for the same profile, 400 on built-in
  profile, export line shape.
- End-to-end: propose → evaluate → accept → new chat run carries the
  candidate text in `system_prompt_extra` → rollback restores absence of the
  key → base prompt YAML bytes unchanged.

## 5. Phasing

| Phase | Deliverable | Estimate |
|-------|-------------|----------|
| A | Migration 4, writers, `save_config_checked`, ProfileCard fix, slot recorder move | 2 days |
| B | Proposal prompt, service, source-runs and proposal routes | 2 days |
| C | Evaluation service, holdout tags, evaluate route, progress, interrupted-on-start | 3 days |
| D | Accept, reject, rollback, export | 1 day |
| E | Refinement panel and vitest | 3 days |
| F | Docs, e2e, changelog | 1 day |

PR split: one PR per phase, six PRs (A, B, C, D, E, F). Each PR is
review-ready alone; D depends on C's `evaluated` rows and is reviewed after
it. `tasks.md` uses the same six-PR sequence.

## 6. Follow-up issue (to file after this plan is confirmed)

Title: "Harden prompt refinement: durable jobs, preflight confirmation,
append-only evidence log, crash-safe accept (#78 follow-up)". Body is the
"Dropped" list in `spec.md` §2 and the "Known gaps" in §6, with
`plan-full-rev4.md` sections C, D4, I1, J6, L1 as the design reference, plus
anything found during the build that was deliberately left out.
