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
    - "External review of plan r1, 2026-09-30: D1 kept with corrected rationale; D2 replaced by controller-dispatched deterministic actions; registry representation and spill offsets specified; compaction API boundary tightened; canonical trace types and wall-clock ledger semantics fixed."
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
- **D1 — registry is controller-side, not sandbox variables.** Required for
  consistent semantics across the supported sandboxes and to avoid
  O(working-set) namespace transfer on the subprocess sandbox, which starts a
  fresh process per `execute` and round-trips its namespace as JSON both
  ways (only JSON-serialisable values survive). Fact-check: the configured
  default sandbox is `restricted`; the server maps an explicitly configured
  `local` to `subprocess` (`server/dependencies.py:1431`), so subprocess is
  a supported path, not the default. The registry
  (`application/services/result_registry.py`) lives in the parent process
  and `read_result` is answered by the controller. `r<K>` is **not** bound
  into any sandbox namespace (spec OQ-1); the v1 code path is unchanged.
- **D2 — deterministic v2 actions are controller-dispatched.** Replaces
  plan-r1's "raise the sandbox stdout cap", which could not deliver spec G4:
  `RestrictedSandboxAdapter` and the subprocess child both collect the whole
  stdout into a `StringIO` and truncate afterwards, so a larger cap bounds no
  memory, and a sandbox-side truncation leaves the controller nothing to
  spill; docker exposes no stdout-cap plumbing through `create_sandbox()`.
  Instead `application/services/inspect_dispatcher.py` takes
  `(content, InspectAction)` and calls `peek/grep/chunk/select/peek_file/
  grep_file/outline_file` directly, returning the canonical result text; the
  registry receives that complete value first and the 10,000-char preview is
  derived from it. `_auto_inspect_missing_file` uses the same dispatcher.
  `subcall` (single and batched, item 3) is dispatched by the controller
  too: the child `RunRLMUseCase` runs in-process, never as `subcall(...)`
  Python through the sandbox. That also fixes an existing defect: the
  subprocess child worker re-injects only the content tools and the JSON
  namespace transfer drops the `subcall` closure, so v2 subcalls cannot
  work in that sandbox today. The sandbox remains only for the v1 free-form
  Python path, with today's stdout limits. Resulting pipeline:
  parse JSON action → controller executes deterministic action → registry
  owns the full result → model receives a bounded preview; only arbitrary
  Python crosses the sandbox boundary.
- **D2a — canonical result representation.** `ResultRegistry.register`
  accepts `str | list | dict`; a `str` is stored unchanged, anything else
  as stable JSON (`ensure_ascii=False, indent=2`). `chunk()` returns
  `list[str]` and a batched subcall returns `list[str]`; both become one
  JSON text. `read_result` slices one canonical text by character offset.
  Config names say what they mean: `spill_result_above_bytes` (in-memory
  per-result threshold, 16 MiB), `max_registry_bytes` (64 MiB),
  `max_spill_bytes` (256 MiB). Spilled files are UTF-8 with a sparse
  character→byte checkpoint table (every 64 K characters) recorded while
  writing, so character reads are exact for non-ASCII content.
- **D3 — one implementation, two loops.** `execute` and `execute_async` are
  near-identical ~800-line copies. All new behaviour lives in pure helpers
  under `application/services/` (registry, compaction, ledger, preview) with
  their own unit tests; each loop gets the same few call sites. No new logic
  is written inline in either loop.
- **D4 — `read_result` is a step** (spec OQ-2). It goes through the normal
  parse → dispatch path, records an `execution` trace row with `code` set to
  the display string `read_result('r4', 10000, 20000)`, and never touches the
  sandbox.
- **D4a — compaction API boundary.** `LLMPort` promises only
  `count_tokens(text)`; `LiteLLMAdapter` additionally supports
  `count_tokens(messages=...)`. Do not widen every adapter. Reuse the
  capability-protocol pattern from `history_context.py::TokenCounter`: a
  `MessageTokenCounter` protocol inside `context_compaction.py`, used when
  the adapter satisfies it, else the `last_clamp_info` + chars/4 fallback.
  Order of operations in both loops: build the candidate `call_messages`
  (nudges and the last-step `force_final_nudge` included) → estimate that
  exact payload → compact `messages` if required → rebuild `call_messages`.
- **D5 — compaction is a message transform, not a summarizer.**
  `compact_messages(messages, registry, keep_last)` is a pure function over
  the message list plus the registry's index of message-position → result id.
  The loop maintains that index as it appends execution messages.
- **D6 — children are independent `RunRLMUseCase` runs** with a
  `copy.copy(llm)` and their own sandbox copy, launched by the controller
  (D2); the ledger replaces the `parent_budget_snapshot` / `subcall_usage`
  pair, which is deleted. Cost, tokens and steps are pre-reserved (equal
  split, unused headroom released). **Wall-clock time is not split:** every
  child gets the parent's absolute deadline and the pool wait is bounded by
  it; two parallel children with 90 s remaining may each run up to that
  deadline. Equal-splitting time would make batching weaker than today's
  serial semantics.
- **D7 — `compaction` is a canonical action type.** Today
  `domain/entities.py::TraceStep.action_type` is
  `Literal["inspect","subcall","final","error"]`,
  `server/routes/_helpers.py::_canonical_action_type` maps unknown roles to
  `inspect` and promotes the last row of a successful run to `final`, and
  `trace_to_replay.py` understands only those four. A raw `compaction` row
  would become an `inspect` step or even the `final` one. PR 2 widens the
  Literal, the role map (`compaction` → `compaction`, exempt from `final`
  promotion), telemetry `action_type`, and the replay builder (its own
  step kind). The frontend needs no type change: `TraceStep.action_type`
  in `frontend/src/lib/api.ts` is already `string`; `timeline.tsx` gets a
  display/colour treatment only.

## 2. Reuse map
| Need | Existing |
|---|---|
| Action schema + validation | `core/actions.py` (`InspectAction.validate` tool list, `SubcallAction`) |
| JSON → action | `run_rlm.py::_parse_rlm_response` (kept); `_inspect_to_code` retired for v2 actions (D2) |
| Content tools | `tools/content.py` (`peek`, `grep`, `chunk`, `select`, `peek_file`, `grep_file`, `outline_file`) called directly by the dispatcher |
| Token-counter capability pattern | `application/services/history_context.py::TokenCounter` |
| Canonical trace types | `domain/entities.py::TraceStep`, `server/routes/_helpers.py::_canonical_action_type`, `application/services/trace_to_replay.py` |
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
`src/rlmstudio/application/services/{result_registry,inspect_dispatcher,context_compaction,subcall_ledger,execution_preview}.py` (new),
`src/rlmstudio/domain/entities.py` (`TraceStep.action_type` Literal), `src/rlmstudio/server/routes/_helpers.py` (role map), `src/rlmstudio/application/services/trace_to_replay.py` (compaction step kind) and their tests (PR 2),
`src/rlmstudio/application/use_cases/run_rlm.py`,
`src/rlmstudio/application/dto.py` (RunConfigDTO fields),
`src/rlmstudio/application/sandbox_vars.py` (trace keys/roles),
`src/rlmstudio/core/actions.py`,
`src/rlmstudio/prompts/system_prompt_v2_0.yaml`, `src/rlmstudio/prompts/rlm_messages.yaml`,
`src/rlmstudio/rlm_studio_config.default.yaml`, `src/rlmstudio/server/dependencies.py` (config → RunConfigDTO plumbing),
`src/rlmstudio/telemetry/store.py` (action_type value only; no schema migration expected),
`frontend/src/components/trace/timeline.tsx` (display treatment for `compaction`; `api.ts` needs no change),
`tests/test_result_registry.py`, `tests/test_context_compaction.py`, `tests/test_subcall_ledger.py`, `tests/test_rlm_working_state_loop.py` (new), existing loop tests as needed,
`docs/RLM_Studio_Design_Document.md` (§4, §5, §9.6, §9.9), `docs/rlm-concepts.md` (§4, §9), `docs/rlm-prompt-tuning.md` (one note), `CHANGELOG.md`.

## 4. Steps
### Item 1 — registry + `read_result`
1. `result_registry.py`: `ResultRegistry(spill_result_above_bytes, max_registry_bytes, max_spill_bytes, scratch_dir)` with `register(value: str | list | dict, *, kind, step) -> ResultRef(id, length_chars, spilled)` (canonical text per D2a), `read(name, start, end, max_chars) -> str` (error text for unknown/evicted), `last`, `index` (message position → id), `close()`; spill via `tempfile.mkdtemp(prefix="rlm-run-")`, atomic UTF-8 writes with the character→byte checkpoint table, accounting hook for tests. Pure Python, no ports.
2. `inspect_dispatcher.py`: `dispatch_inspect(content, action: InspectAction) -> str | list | dict` calling the `tools/content.py` functions directly with validated args; raises a typed error for bad args that the loop renders as an execution error row.
3. `execution_preview.py`: `format_preview(text, cap, ref) -> str` producing the marker from spec G3; `_format_execution` delegates to it for registered results.
4. `core/actions.py`: add `read_result` to the tool list with arg validation (`name` str, `start`/`end` int|None, `max_chars` int).
5. `run_rlm.py` (both loops): construct the registry per run (`try/finally close()`); route every v2 `inspect` (incl. `_auto_inspect_missing_file`) through the dispatcher instead of `_inspect_to_code` + sandbox; register the full value, then preview; answer `read_result` from the registry (D4); leave the v1 code path on the sandbox unchanged.
6. `RunConfigDTO`: `spill_result_above_bytes`, `max_registry_bytes`, `max_spill_bytes`; defaults in `rlm_studio_config.default.yaml`; plumb through `server/dependencies.py` and `api.py`.
7. Prompt 2.2: tool table row, one example workflow ("grep returned 200 matches"), marker explanation; fingerprint records the version.

### Item 2 — compaction
8. `context_compaction.py`: `MessageTokenCounter` protocol (D4a); `estimate_tokens(llm, call_messages, fallback_hint) -> int`; `should_compact(estimate, context_window, reserve) -> bool`; `compact_messages(messages, registry, keep_last) -> (messages, CompactionRecord)`; idempotent.
9. Both loops: build candidate `call_messages` (nudges + last-step `force_final_nudge`) → estimate → compact `messages` if needed → rebuild `call_messages`; on overflow exception, compact and retry the same step once (`reason="overflow_retry"`), else existing fallback; append the raw `compaction` trace row (new constants in `sandbox_vars.py`); keep the registry's message index in sync when messages are replaced.
10. `RunConfigDTO`: `compaction_reserve_tokens` (4096), `compaction_keep_last` (4); config + plumbing as in step 6.
11. Canonical types (D7): widen `TraceStep.action_type`; `_ACTION_TYPE_MAP["compaction"] = "compaction"` with no `final` promotion; telemetry `record_step(action_type="compaction")`; `trace_to_replay.py` emits a `compaction` replay step; `timeline.tsx` display treatment; outcome classifier counts compactions (no category change).

### Item 3 — batched subcalls
12. `core/actions.py`: `SubcallAction` accepts either `prompt`+`query` or `calls: list[{prompt, query}]` (≥1, each validated) and optional `max_concurrency`; single form unchanged.
13. `subcall_ledger.py`: `SubcallLedger(config, parent_state, deadline, lock)` with `reserve(n_children) -> list[ChildAllowance]` (equal split of remaining cost/tokens/steps; every allowance carries the parent's absolute deadline), `release(allowance, actual_usage)`; folds actual spend into parent totals under the lock.
14. `run_rlm.py`: controller-dispatched subcalls (D2): delete the `subcall` sandbox binding, `_make_subcall`, `parent_budget_snapshot` and `subcall_usage`; add `_run_subcalls(calls, max_concurrency)` using `ThreadPoolExecutor` with `copy.copy(llm)` and a sandbox copy per child, pool wait bounded by the parent deadline; results in order → one registered list result; per-child failure becomes an error string in that slot, never fails the batch. The single-call form is the one-element batch.
15. Concurrency cap: `RunConfigDTO.subcall_max_concurrency`; `server/dependencies.py` / `api.py` set 2 for local providers, 4 for cloud from the provider catalog; the action's `max_concurrency` is `min`'d against it.
16. Prompt 2.2: batched example after the single-call example; `chunk()` → batched subcall map-reduce workflow.

### Docs / bench
17. Design doc §3 (deterministic actions are controller operations), §4 (new tool), §5 (compaction as a safety valve), §6 (sandbox scope is the v1 path only), retire §9.6/§9.9 text; `rlm-concepts.md` §4 ("batched subcall is the map-reduce step") and §9; prompt-tuning note on `read_result`; CHANGELOG `[Unreleased]` entry per item.
18. Owner: AC-9 regression run (`rlm-studio bench` with an 8K-context local model, before/after), results under `benchmarks/results/<date>/`.

## 5. Test strategy
- Unit (fast, no network): registry (ids, `last`, canonical JSON for lists, character slicing/clamping, spill with checkpoint table on CJK/emoji, eviction order, accounting, cleanup on exception); dispatcher (each tool, bad args); preview marker; compaction (threshold on the candidate `call_messages`, capability vs fallback counter, idempotency, stub text, index sync); canonical type mapping and replay step for `compaction`; ledger (reservation math, release, shared deadline, lock under a thread pool); action parsing (both subcall forms, `read_result` args).
- Loop-level (ScriptedLLM extended with `context_window`, an optional `count_tokens(messages=)`, and a scripted overflow-on-step-N): AC-1..7 on `execute` and `execute_async`, compared on a normalised semantic trace (role, code, preview text, registry ids, budget totals; timing and stream fields dropped). The v1 free-form path keeps its sandbox-parametrised tests (restricted/subprocess; docker as `integration`).
- e2e: one WebSocket streaming case asserting the compaction row arrives as an `on_step` event.
- Bench: AC-9 by the owner; command recorded in tasks.md.

## 6. Risks
| Risk | Mitigation |
|---|---|
| Two loop copies drift | D3: helpers own the logic; a test asserts both paths produce identical traces for the same scripted run |
| Dispatcher and sandbox tool signatures drift | The dispatcher calls the same `tools/content.py` functions the sandbox binds; one parametrised test runs each tool through both and compares |
| Large `grep`/`chunk` results held in the parent process | Complete value goes to the registry first; above `spill_result_above_bytes` it is on disk; AC-3 asserts the in-memory bound |
| Token estimate undercounts → overflow anyway | Overflow retry path (G6) is the backstop; reserve default 4,096 leaves margin |
| Ledger equal split starves a heavy child | Unused headroom is released and re-reservable; a child that runs out stops with a classified error, batch continues; time is shared, never split |
| Local backends flooded by parallel children | Provider-kind cap (2) is authoritative over the model's request |
| Prompt growth on small models | 2.2 adds one table row and two short examples; measured by the same bench run |
