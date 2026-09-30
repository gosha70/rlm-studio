---
feature_id: rlm-working-state
spec_mode: full
spec: ./spec.md
status: draft
date: 2026-09-30
origin:
  urls:
    - https://github.com/gosha70/rlm-studio/issues/77
  transcripts:
    - "Owner + external review, 2026-09-30: order 1 → 2 → 3; item 1 establishes the registry API that compaction and batching both consume."
  origin_claim: |
    Inherited from spec.md — information can leave the model context
    without leaving the RLM working state.
---

# Plan — RLM working state

Depends on nothing unmerged (`master` at `aa385f5`). Size: **L** (≈3 weeks:
registry ≈4 days, compaction ≈5 days, batching ≈5 days, docs/bench ≈2 days).
Circuit breaker: registry + compaction ship on their own if batching is not
green by end of week 3; batching becomes a follow-up PR against the same spec.

## 1. Design decisions and deviations from #77
- **D1 — registry is controller-side, not sandbox variables.** The server
  maps `sandbox.type: local` to the subprocess sandbox
  (`server/dependencies.py:1431`), which starts a fresh process per
  `execute` and round-trips the namespace as JSON both ways. Holding results
  as sandbox variables would cost O(total payload) per step in the production
  default. So the registry (`application/services/result_registry.py`) stores
  the *execution text* of each primary action in the parent process, and
  `read_result` is answered by the controller without a sandbox round-trip.
  Consequence: `r<K>` is **not** bound into the sandbox namespace (spec OQ-1);
  the v1 code path keeps its current behaviour. If a later need appears, bind
  for the in-process sandboxes only and add the names to
  `SANDBOX_SKIP_RETURN_VARS`.
- **D2 — per-result cap is enforced by the sandbox stdout cap.** For
  inspect/subcall executions the loop asks the sandbox for up to
  `max_result_bytes` of stdout (the existing `max_stdout_chars` mechanism,
  raised for the duration of that call or configured at construction), and
  the *preview* cap (10,000 chars) moves into the controller
  (`_format_execution` → `format_preview`). One truncation point per layer.
- **D3 — one implementation, two loops.** `execute` and `execute_async` are
  near-identical ~800-line copies. All new behaviour lives in pure helpers
  under `application/services/` (registry, compaction, ledger, preview) with
  their own unit tests; each loop gets the same few call sites. No new logic
  is written inline in either loop.
- **D4 — `read_result` is a step** (spec OQ-2). It goes through the normal
  parse → dispatch path, records an `execution` trace row with `code` set to
  the display string `read_result('r4', 10000, 20000)`, and never touches the
  sandbox.
- **D5 — compaction is a message transform, not a summarizer.**
  `compact_messages(messages, registry, keep_last)` is a pure function over
  the message list plus the registry's index of message-position → result id.
  The loop maintains that index as it appends execution messages.
- **D6 — children are independent `RunRLMUseCase` runs** (as today) with a
  `copy.copy(llm)` and their own sandbox copy; the ledger replaces the
  `parent_budget_snapshot` / `subcall_usage` pair, which is deleted.

## 2. Reuse map
| Need | Existing |
|---|---|
| Action schema + validation | `core/actions.py` (`InspectAction.validate` tool list, `SubcallAction`) |
| JSON → sandbox code | `run_rlm.py::_inspect_to_code`, `_parse_rlm_response` |
| Execution formatting | `run_rlm.py::_format_execution` (becomes preview formatter) |
| Coverage auto-inspect | `run_rlm.py::_auto_inspect_missing_file` (registers like any inspect) |
| Token estimation | `LiteLLMAdapter.count_tokens(messages=...)`, `context_window`, `last_clamp_info`; `LLMPort.count_tokens(text)` for fakes |
| Overflow detection | `run_rlm.py::_is_context_overflow` |
| Budget types | `domain/entities.py` (`BudgetConfig`, `BudgetState.is_within`) |
| Subcall closure | `run_rlm.py::_make_subcall` (rewritten over the ledger) |
| Thread-pool fan-out | `run_matrix_comparison.py` (`ThreadPoolExecutor`, per-slot failure isolation) |
| Adapter isolation precedent | `run_comparison.py` (`copy.copy(self._llm)` per branch) |
| Trace rows / telemetry | trace dict keys in `application/sandbox_vars.py`; `TelemetryStore.record_step(action_type=...)` |
| Outcome classification | `application/services/outcome_classifier.py` |
| Fakes | `benchmark/fakes.py::ScriptedLLM` (extend with `context_window`, scripted overflow); `tests/integration/test_budget_enforcement.py` patterns |
| Prompt | `prompts/system_prompt_v2_0.yaml` (version 2.1 → 2.2), `prompts/rlm_messages.yaml` |
| Provider kind (local vs cloud) | provider catalog in `ui/data/providers_catalog.py`; the slot builder in `rlmstudio/api.py` |

## 3. Allowlist
`src/rlmstudio/application/services/{result_registry,context_compaction,subcall_ledger,execution_preview}.py` (new),
`src/rlmstudio/application/use_cases/run_rlm.py`,
`src/rlmstudio/application/dto.py` (RunConfigDTO fields),
`src/rlmstudio/application/sandbox_vars.py` (trace keys/roles),
`src/rlmstudio/core/actions.py`,
`src/rlmstudio/infrastructure/sandbox/*.py` (stdout cap plumbing only),
`src/rlmstudio/prompts/system_prompt_v2_0.yaml`, `src/rlmstudio/prompts/rlm_messages.yaml`,
`src/rlmstudio/rlm_studio_config.default.yaml`, `src/rlmstudio/server/dependencies.py` (config → RunConfigDTO plumbing),
`src/rlmstudio/telemetry/store.py` (action_type value only; no schema migration expected),
`frontend/src/**` (trace role union + one compaction row renderer),
`tests/test_result_registry.py`, `tests/test_context_compaction.py`, `tests/test_subcall_ledger.py`, `tests/test_rlm_working_state_loop.py` (new), existing loop tests as needed,
`docs/RLM_Studio_Design_Document.md` (§4, §5, §9.6, §9.9), `docs/rlm-concepts.md` (§4, §9), `docs/rlm-prompt-tuning.md` (one note), `CHANGELOG.md`.

## 4. Steps
### Item 1 — registry + `read_result`
1. `result_registry.py`: `ResultRegistry(max_result_bytes, max_registry_bytes, max_spill_bytes, scratch_dir)` with `register(text, *, kind, step) -> ResultRef(id, length, spilled)`, `read(name, start, end, max_chars) -> str | ErrorText`, `last`, `index` (message position → id), `close()`; spill via `tempfile.mkdtemp(prefix="rlm-run-")`, atomic writes, accounting hook for tests. Pure Python, no ports.
2. `execution_preview.py`: `format_preview(text, cap, ref) -> str` producing the marker from spec G3; `_format_execution` delegates to it.
3. `core/actions.py`: add `read_result` to the tool list with arg validation (`name` str, `start`/`end` int|None, `max_chars` int).
4. `run_rlm.py` (both loops): construct the registry per run (config-driven caps; `try/finally close()`); register after every inspect execution, subcall execution and `_auto_inspect_missing_file`; intercept `tool == "read_result"` before sandbox dispatch (D4); raise the sandbox stdout cap for primary executions (D2).
5. `RunConfigDTO`: `max_result_bytes`, `max_registry_bytes`, `max_spill_bytes`; defaults in `rlm_studio_config.default.yaml`; plumb through `server/dependencies.py` and `api.py`.
6. Prompt 2.2: tool table row, one example workflow ("grep returned 200 matches"), marker explanation; fingerprint records the version.

### Item 2 — compaction
7. `context_compaction.py`: `estimate_pending_tokens(llm, messages, appended_since_last_call) -> int` (adapter `count_tokens(messages=...)` if present, else `last_clamp_info` + chars/4); `should_compact(estimate, context_window, reserve) -> bool`; `compact_messages(messages, registry, keep_last) -> (messages, CompactionRecord)`; idempotent.
8. Both loops: call `should_compact` right before building `call_messages` (after nudges are appended); on overflow exception, compact and retry the same step once (`retry_reason="overflow_retry"`), else existing fallback; append the compaction trace row (`role: compaction`, new constants in `sandbox_vars.py`); keep the registry's message index in sync when messages are replaced.
9. `RunConfigDTO`: `compaction_reserve_tokens` (4096), `compaction_keep_last` (4); config + plumbing as in step 5.
10. Telemetry/UI: `record_step(action_type="compaction")`; trace role union + a one-line row in the Traces timeline/tree and Replay; outcome classifier counts compactions (no category change).

### Item 3 — batched subcalls
11. `core/actions.py`: `SubcallAction` accepts either `prompt`+`query` or `calls: list[{prompt, query}]` (≥1, each validated) and optional `max_concurrency`; single form unchanged.
12. `subcall_ledger.py`: `SubcallLedger(config, parent_state, lock)` with `reserve(n_children) -> list[ChildAllowance]`, `release(allowance, actual_usage)`; equal split of remaining cost/tokens/steps/time; folds actual spend into parent totals under the lock.
13. `run_rlm.py`: rewrite `_make_subcall` over the ledger (delete `parent_budget_snapshot`/`subcall_usage`); add `_run_subcall_batch(calls, max_concurrency)` using `ThreadPoolExecutor` with `copy.copy(llm)` and a sandbox copy per child; results in order → one registered list result; per-child failure becomes an error string in that slot, never fails the batch.
14. Concurrency cap: `RunConfigDTO.subcall_max_concurrency`; `server/dependencies.py` / `api.py` set 2 for local providers, 4 for cloud from the provider catalog; the action's `max_concurrency` is `min`'d against it.
15. Prompt 2.2: batched example after the single-call example; `chunk()` → batched subcall map-reduce workflow.

### Docs / bench
16. Design doc §4 (new tool), §5 (compaction as a safety valve), retire §9.6/§9.9 text; `rlm-concepts.md` §4 ("batched subcall is the map-reduce step") and §9; prompt-tuning note on `read_result`; CHANGELOG `[Unreleased]` entry per item.
17. Owner: AC-9 regression run (`rlm-studio bench` with an 8K-context local model, before/after), results under `benchmarks/results/<date>/`.

## 5. Test strategy
- Unit (fast, no network): registry (ids, `last`, slicing/clamping, spill, eviction, accounting, cleanup on exception); preview marker; compaction (threshold, idempotency, stub text, index sync); ledger (reservation math, release, lock under a thread pool); action parsing (both subcall forms, `read_result` args).
- Loop-level (ScriptedLLM extended with `context_window`, `count_tokens(messages=)`, and a scripted overflow-on-step-N): AC-1..7 on `execute` and `execute_async`; parametrised over restricted/subprocess sandboxes (docker marked `integration`, skipped without a daemon).
- e2e: one WebSocket streaming case asserting the compaction row arrives as an `on_step` event.
- Bench: AC-9 by the owner; command recorded in tasks.md.

## 6. Risks
| Risk | Mitigation |
|---|---|
| Two loop copies drift | D3: helpers own the logic; a test asserts both paths produce identical traces for the same scripted run |
| Subprocess sandbox stdout cap raise costs memory per step | Cap = `max_result_bytes` (16 MiB); registry spills above `max_registry_bytes`; measured in AC-3 |
| Token estimate undercounts → overflow anyway | Overflow retry path (G6) is the backstop; reserve default 4,096 leaves margin |
| Ledger equal split starves a heavy child | Unused headroom is released and re-reservable; a child that runs out stops with a classified error, batch continues |
| Local backends flooded by parallel children | Provider-kind cap (2) is authoritative over the model's request |
| Prompt growth on small models | 2.2 adds one table row and two short examples; measured by the same bench run |
