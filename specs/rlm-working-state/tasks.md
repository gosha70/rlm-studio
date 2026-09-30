---
feature_id: rlm-working-state
spec: ./spec.md
plan: ./plan.md
status: draft
date: 2026-09-30
---

# Tasks — RLM working state

Branch `feat/rlm-working-state` from `master`. One PR per item (three PRs
against this spec), each with CI green; item 2 builds on item 1's registry
API, item 3 on both. Allowlist in `plan.md` §3 is binding. Apply defaults for
OQ-1..3 in `spec.md` §5 unless the owner answers otherwise in Phase 0.

## Phase 0 — confirmations (no edits)
- [x] T0.1 D1 confirmed (controller-side registry, no `r<K>` sandbox binding); plan-r1 D2 (stdout-cap raise) **rejected** and replaced by controller-dispatched deterministic actions. *(2026-09-30 review: both in-process sandboxes buffer the whole stdout before truncating, a sandbox-side cut leaves nothing to spill, docker has no cap plumbing; the subprocess child re-injects only the content tools so the `subcall` closure never survives the JSON namespace transfer. Default sandbox is `restricted`; `local` maps to subprocess at `server/dependencies.py:1431`.)*
- [x] T0.2 Accepted. Full candidate-message counting once per controller step is acceptable for 1.x. It duplicates `_build_params()` tokenization when `context_window` is configured, but this is CPU-local bounded work and preferable to weakening the pre-compaction estimate. Do not add per-message caching in this feature; profile after AC-9 and optimise only if token counting becomes material. If the duplicate is removed later, prefer making `_build_params()` consume a precomputed estimate over a content-sensitive cache. *(Owner, 2026-09-30.)*
- [x] T0.3 Frontend has no trace-role union to widen: `frontend/src/lib/api.ts::TraceStep.action_type` is `string`. The canonical layer is where the work is: `domain/entities.py::TraceStep.action_type` Literal, `server/routes/_helpers.py::_canonical_action_type` (unknown → `inspect`, last row of a success → `final`), `trace_to_replay.py` (four actions). *(2026-09-30 review; see plan D7.)*
- [x] T0.4 OQ-1 no sandbox bindings (`r<K>` controller-only); OQ-2 `read_result` consumes a step; OQ-3 prompt version 2.2. *(Owner, 2026-09-30; rationale recorded in `spec.md` §5.)* Phase 0 closed; the wall-clock guarantee in PR 3 softened to a between-steps rule (plan D6).

## Phase 1 — result registry + `read_result` (PR 1)
- [ ] T1.1 `application/services/result_registry.py` + `tests/test_result_registry.py` (ids, `last`, canonical JSON for list/dict, character slice/clamp, spill with character→byte checkpoints on CJK/emoji, eviction order, accounting hook, `close()` on exception).
- [ ] T1.2 `application/services/inspect_dispatcher.py` + tests (each content tool with validated args; bad-arg error path); a parametrised test runs each tool through the dispatcher and through the restricted sandbox and compares.
- [ ] T1.3 `application/services/execution_preview.py`; `_format_execution` delegates for registered results; marker text per spec G3; tests.
- [ ] T1.4 `core/actions.py`: `read_result` in the tool list + arg validation; parser tests.
- [ ] T1.5 `run_rlm.py` sync loop: registry lifecycle; v2 `inspect` and `_auto_inspect_missing_file` through the dispatcher (register full value, then preview); `read_result` answered from the registry (D4); v1 code path untouched.
- [ ] T1.6 `run_rlm.py` async loop: same call sites; a test asserts normalised-semantic-trace equality for the same scripted run on both paths (timing/stream fields excluded).
- [ ] T1.7 `RunConfigDTO` (`spill_result_above_bytes`, `max_registry_bytes`, `max_spill_bytes`) + `rlm_studio_config.default.yaml` + `server/dependencies.py` / `api.py` plumbing.
- [ ] T1.8 Prompt 2.2 (tool row, example, marker note); fingerprint records version; `docs/rlm-prompt-tuning.md` note.
- [ ] T1.9 AC-1, AC-2, AC-3, AC-3a loop tests on both paths; existing v1 sandbox tests still green on restricted + subprocess (docker as `integration`).

## Phase 2 — lossless compaction (PR 2)
- [ ] T2.1 `application/services/context_compaction.py` (`MessageTokenCounter` protocol per D4a, `estimate_tokens`, `should_compact`, `compact_messages`) + tests (threshold on candidate `call_messages`, capability vs fallback, idempotency, stub text, index sync).
- [ ] T2.2 Extend `benchmark/fakes.py::ScriptedLLM` (or a test-local fake) with `context_window`, an *optional* `count_tokens(messages=)`, `last_clamp_info`, scripted overflow on step N.
- [ ] T2.3 Both loops: build candidate `call_messages` (incl. last-step `force_final_nudge`) → estimate → compact → rebuild; overflow → compact → retry once; raw `compaction` trace row; registry index sync.
- [ ] T2.4 `RunConfigDTO.compaction_reserve_tokens` / `compaction_keep_last` + config plumbing.
- [ ] T2.5 Canonical `compaction` action (D7): `domain/entities.py::TraceStep.action_type` Literal; `server/routes/_helpers.py` role map with no `final` promotion; telemetry `record_step(action_type="compaction")`; `trace_to_replay.py` compaction replay step; `timeline.tsx` display treatment; outcome classifier counts it; tests for each.
- [ ] T2.6 AC-4, AC-4a, AC-5 loop tests on both paths; one e2e WebSocket case for the compaction `on_step` event; a `_helpers` test that a terminal `compaction` row on a successful run is not promoted to `final`.

## Phase 3 — batched subcalls + ledger (PR 3)
- [ ] T3.1 `core/actions.py`: `calls` + `max_concurrency` on `SubcallAction`; single form unchanged; parser tests for both.
- [ ] T3.2 `application/services/subcall_ledger.py` + tests (equal split of cost/tokens/steps, release/re-reserve, common absolute deadline carried unchanged into every allowance, lock under a thread pool, parent fold-in).
- [ ] T3.3 `run_rlm.py`: controller-dispatched subcalls (D2/D6): delete the `subcall` sandbox binding, `_make_subcall`, `parent_budget_snapshot`, `subcall_usage`; `_run_subcalls` with `copy.copy(llm)` + sandbox copy per child, each child's between-steps time limit = parent deadline (no hard pool bound, plan D6); ordered results → one registered list result; per-child error isolation; single call = one-element batch.
- [ ] T3.4 `RunConfigDTO.subcall_max_concurrency`; provider-kind defaults (2 local / 4 cloud) in `server/dependencies.py` and `api.py`; action value `min`'d.
- [ ] T3.5 Prompt 2.2 batched example + `chunk()` map-reduce workflow.
- [ ] T3.6 AC-6, AC-6a, AC-7 loop tests on both paths (AC-6a with the subprocess sandbox configured); `tests/test_json_subcall.py` and `tests/integration/test_budget_enforcement.py` still green or updated for controller dispatch.

## Phase 4 — docs
- [ ] T4.1 Design doc §3 / §4 / §5 / §6 / §9.6 / §9.9; `rlm-concepts.md` §4 and §9; CHANGELOG `[Unreleased]` entries (one per item, the subprocess-subcall fix called out under item 3).

## Phase 5 — regression run (owner)
- [ ] T5.1 AC-9: `rlm-studio bench --config benchmarks/longdoc-v1.yaml --providers ollama/<8k-context model> --engines rlm --reps 2 --out benchmarks/results/<date>-working-state/` before (baseline tag) and after; compare overflow-induced limit warnings, accuracy, judge; commit results.

## Phase 6 — acceptance
- [ ] T6.1 AC-1..AC-8 green in CI on the final PR; AC-9 recorded; #77 items 1–3 closed with links to the three PRs; #78 unblocked.
