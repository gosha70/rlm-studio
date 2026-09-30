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
- [ ] T0.1 Confirm D1 (controller-side registry, no `r<K>` sandbox binding) and D2 (stdout cap as the per-result enforcement point) against `infrastructure/sandbox/subprocess_sandbox.py` and `server/dependencies.py:1427-1433`.
- [ ] T0.2 Confirm `LiteLLMAdapter.count_tokens(messages=...)` cost is acceptable per step for the configured providers (it calls `litellm.token_counter`); if it is slow for local models, cache per message and count only appended messages.
- [ ] T0.3 Locate the frontend trace `role` union and the Traces timeline/tree renderers; confirm adding a `compaction` role is one type change plus one row component.
- [ ] T0.4 Owner answers OQ-1..3 or accepts defaults.

## Phase 1 — result registry + `read_result` (PR 1)
- [ ] T1.1 `application/services/result_registry.py` + `tests/test_result_registry.py` (ids, `last`, slice/clamp, spill, eviction order, accounting hook, `close()` on exception).
- [ ] T1.2 `application/services/execution_preview.py`; `_format_execution` delegates; marker text per spec G3; tests.
- [ ] T1.3 `core/actions.py`: `read_result` in the tool list + arg validation; parser tests.
- [ ] T1.4 `run_rlm.py` sync loop: registry lifecycle, registration after inspect / subcall / auto-inspect, `read_result` interception (D4), stdout cap raise (D2).
- [ ] T1.5 `run_rlm.py` async loop: same call sites; a test asserts identical traces for the same scripted run on both paths.
- [ ] T1.6 `RunConfigDTO` caps + `rlm_studio_config.default.yaml` + `server/dependencies.py` / `api.py` plumbing.
- [ ] T1.7 Prompt 2.2 (tool row, example, marker note); fingerprint records version; `docs/rlm-prompt-tuning.md` note.
- [ ] T1.8 AC-1, AC-2, AC-3 loop tests, parametrised over restricted + subprocess (docker as `integration`).

## Phase 2 — lossless compaction (PR 2)
- [ ] T2.1 `application/services/context_compaction.py` (`estimate_pending_tokens`, `should_compact`, `compact_messages`) + tests (threshold, idempotency, stub text, index sync).
- [ ] T2.2 Extend `benchmark/fakes.py::ScriptedLLM` (or a test-local fake) with `context_window`, `count_tokens(messages=)`, `last_clamp_info`, scripted overflow on step N.
- [ ] T2.3 Both loops: pre-call `should_compact`; overflow → compact → retry once; compaction trace row; registry index sync.
- [ ] T2.4 `RunConfigDTO.compaction_reserve_tokens` / `compaction_keep_last` + config plumbing.
- [ ] T2.5 Telemetry `action_type="compaction"`; frontend role union + row in Traces and Replay; outcome classifier counts it.
- [ ] T2.6 AC-4, AC-5 loop tests on both paths; one e2e WebSocket case for the compaction `on_step` event.

## Phase 3 — batched subcalls + ledger (PR 3)
- [ ] T3.1 `core/actions.py`: `calls` + `max_concurrency` on `SubcallAction`; single form unchanged; parser tests for both.
- [ ] T3.2 `application/services/subcall_ledger.py` + tests (equal split, release/re-reserve, lock under a thread pool, parent fold-in).
- [ ] T3.3 `run_rlm.py`: `_make_subcall` over the ledger (delete `parent_budget_snapshot` / `subcall_usage`); `_run_subcall_batch` with `copy.copy(llm)` + sandbox copy per child; ordered results → one registered list result; per-child error isolation.
- [ ] T3.4 `RunConfigDTO.subcall_max_concurrency`; provider-kind defaults (2 local / 4 cloud) in `server/dependencies.py` and `api.py`; action value `min`'d.
- [ ] T3.5 Prompt 2.2 batched example + `chunk()` map-reduce workflow.
- [ ] T3.6 AC-6, AC-7 loop tests on both paths; `tests/test_json_subcall.py` and `tests/integration/test_budget_enforcement.py` still green.

## Phase 4 — docs
- [ ] T4.1 Design doc §4 / §5 / §9.6 / §9.9; `rlm-concepts.md` §4 and §9; CHANGELOG `[Unreleased]` entries (one per item).

## Phase 5 — regression run (owner)
- [ ] T5.1 AC-9: `rlm-studio bench --config benchmarks/longdoc-v1.yaml --providers ollama/<8k-context model> --engines rlm --reps 2 --out benchmarks/results/<date>-working-state/` before (baseline tag) and after; compare overflow-induced limit warnings, accuracy, judge; commit results.

## Phase 6 — acceptance
- [ ] T6.1 AC-1..AC-8 green in CI on the final PR; AC-9 recorded; #77 items 1–3 closed with links to the three PRs; #78 unblocked.
