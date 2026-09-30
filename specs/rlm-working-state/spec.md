---
feature_id: rlm-working-state
spec_mode: full
status: draft
date: 2026-09-30
origin:
  urls:
    - https://github.com/gosha70/rlm-studio/issues/77   # epic (revision 3) — this spec implements items 1–3
    - https://github.com/gosha70/rlm-studio/issues/78   # child: evidence-gated refinement, blocked on this
    - https://github.com/PrimeIntellect-ai/prime-agent    # the design being borrowed from
    - https://arxiv.org/abs/2608.23552                    # Prime Agent paper
  transcripts:
    - "Owner + external review, 2026-09-30: design review of #77 closed at revision 3; 'move #77 into implementation planning', order 1 → 2 → 3, item 1 establishes the registry API that compaction and batching both consume."
    - "External review of plan r1, 2026-09-30: keep D1 (controller-side registry) with a corrected rationale; replace D2 (stdout-cap raise) with controller-dispatched deterministic actions; fix registry representation, spill offsets, the compaction API boundary, canonical trace types, and wall-clock semantics in the ledger."
  origin_claim: |
    Studio's RLM loop discards information that should stay in the run's
    working state: inspect output past the stdout cap is lost, a context
    overflow ends reasoning instead of compacting, and subcalls run one
    per turn, serially. Prime Agent's lesson — information can leave the
    model context without leaving the working state — transfers to a
    single-node document workbench without adopting its agent runtime.
spec_mode_justification: >
  Touches the v2 action protocol (new inspect tool, extended subcall
  schema), both copies of the RLM loop, budget semantics under
  concurrency, and the trace/telemetry schema. Needs explicit invariants
  and acceptance tests so the deterministic protocol that keeps local
  models reliable is tightened, not loosened.
---

# Spec — RLM working state: result registry, lossless compaction, batched subcalls

## 1. Problem
`RunRLMUseCase` (`src/rlmstudio/application/use_cases/run_rlm.py`) breaks the
"working state outlives the context" property in three places, verified
against the code (not the design document, which is behind it):

1. `_inspect_to_code` compiles every inspect action to `print(tool(...))`.
   The sandbox caps stdout at 10,000 chars and the tail is gone. `grep()` and
   `chunk()` routinely return more than that; `peek()`/`select()`/
   `outline_file()` truncate inside the tool, so their tail never exists.
2. A context-window exception breaks out of the loop to synthesis (≤12
   inspect snapshots × 500 chars, joined and capped at 6,000) or a limit
   warning. No compaction, no retry. The adapter already exposes
   `context_window`, `count_tokens(messages=...)` and
   `last_clamp_info.estimated_prompt_tokens`.
3. One `subcall` per turn; `_make_subcall` runs children serially on a
   deep-copied sandbox and derives each child's allowance from *completed*
   siblings, which is only correct serially.

The v2 prompt is also internally inconsistent: "exactly one JSON object" and,
later, a bare `print(history[-1])` block. The parser accepts both via the v1
fallback. The new tool tightens this rather than widening it.

## 2. Goals / non-goals
**Goals**
- G0 **Deterministic actions are controller operations.** Every v2 JSON
  action (`inspect` tools, `read_result`, single and batched `subcall`) is
  executed by the controller against `content` directly; the sandbox runs
  only free-form Python (the v1 code-block path). Pipeline:
  parse JSON action → controller executes it → registry owns the full result
  → the model receives a bounded preview.
- G1 **Result registry.** Every *primary* result (inspect tools other than
  `read_result`, single/batched subcalls, automatic coverage inspections) gets
  a monotonically increasing id `r1, r2, …` from a registry counter
  independent of `budget_state.steps`; `last` names the newest primary
  result; `read_result` never registers and never changes `last`.
- G2 **`read_result` inspect tool** in the v2 protocol:
  `{"type":"inspect","tool":"read_result","args":{"name":"last"|"r<K>","start":int,"end":int|null,"max_chars":int}}`.
  Python slice semantics, clamped; unknown/evicted id → one-line error
  string, still a step.
- G3 **Preview marker.** A truncated preview ends with
  `... (preview truncated: N chars total; read_result('r<K>', start=<cap>) for more)`
  naming the id explicitly.
- G4 **Enforceable memory bound.** In-memory per-result threshold
  `spill_result_above_bytes` (16 MiB), in-memory total `max_registry_bytes`
  (64 MiB), spill budget `max_spill_bytes` (256 MiB). A result above the
  threshold, or displaced by the total, spills to a per-run scratch
  directory and is never dropped while spill budget remains; `last` is never
  dropped; scratch dir removed at run end, including on exceptions. The
  registry always receives the *complete* returned value (G0), so nothing is
  lost before it can be spilled.
- G4a **Canonical representation.** A registered result is one text payload:
  a `str` result is stored unchanged; a structured result (`chunk()` →
  `list[str]`, a batched subcall → `list[str]`) is stored as stable JSON
  (`json.dumps(..., ensure_ascii=False, indent=2)`). `read_result` always
  slices that one canonical text by **character** offset.
- G4b **Character offsets on spilled files.** Byte quotas and character
  reads are reconciled by storing, per spilled result, a sparse
  character→byte checkpoint table (every 64 K characters) built while
  writing UTF-8; a read decodes from the nearest checkpoint. Reads on
  non-ASCII content are exact (tested with CJK and emoji).
- G5 **Deterministic, lossless compaction.** Before each LLM call, build
  the candidate `call_messages` first (nudges and the last-step
  `force_final_nudge` included), estimate *that exact payload* through a
  message-token-counter capability (see plan D4) with a `last_clamp_info`
  + chars/4 fallback, compact `messages` if needed, then rebuild
  `call_messages`. Compact when estimate > `context_window − reserve_tokens`
  (4,096). Keep system +
  query + last `keep_last` (4) assistant/execution pairs; older execution
  messages become `[result r4: 31,842 chars; read_result('r4', ...)]`
  stubs, older assistant actions become their one-line code. Idempotent.
- G6 **Overflow → compact → retry once** on the same controller step; a
  second overflow falls through to today's fallback with the step recorded.
- G7 **Compaction is a first-class canonical action.** Raw trace row
  `role: compaction` (step, tokens before/after, ids stubbed, reason
  `threshold` | `overflow_retry`) → canonical `action_type: compaction` in
  `domain/entities.py::TraceStep`, the server canonicaliser and telemetry;
  never coerced to `inspect` and never promoted to `final`; rendered as its
  own Replay step and timeline row.
- G8 **Batched subcall action**, backward compatible:
  `{"type":"subcall","calls":[{prompt,query},…],"max_concurrency":int}`;
  results in order, registered as one primary list result.
- G9 **Budget ledger.** Lock-protected, owned by the parent run. Cost,
  tokens and steps are additive: pre-reserved per child before launch (equal
  split), unused headroom returned on completion. **Wall-clock time is not
  divided among siblings:** every child receives the parent's common
  absolute deadline and must not start another controller/LLM step after
  it; parallel children overlap rather than each consuming a slice.
  In-flight provider calls remain subject to the provider adapter's
  existing request-timeout semantics (`max_time_seconds` is a
  between-actions check and cannot interrupt a call in progress). Children
  get an isolated adapter (`copy.copy`) — `_active_model` is mutable.
- G9a **Subcalls are controller-dispatched.** The `subcall` action no longer
  compiles to `subcall(...)` Python sent through the sandbox; the controller
  runs the child `RunRLMUseCase` itself. This also fixes an existing defect:
  the subprocess sandbox's JSON namespace transfer drops the non-serialisable
  `subcall` closure and the child worker re-injects only the content tools,
  so v2 subcalls cannot work in that sandbox today.
- G10 **Concurrency cap** from profile config: default 2 for local providers
  (Ollama, LM Studio, vLLM), 4 for cloud; the model's request never raises it.
- G11 Both loop copies (`execute`, `execute_async`) behave identically
  (compared on a normalised semantic trace, timing and stream fields
  excluded); deterministic actions no longer depend on the sandbox, so
  sandbox parametrisation applies only to the v1 free-form path.

**Non-goals** (see #77 table): persistent child sessions, agent messaging,
harness CRUD for memories/skills/subagents, daemon sessions, autonomous
mode, unsandboxed execution, summary-based compaction, refinement (#78),
binding `r<K>` into any sandbox namespace (plan D1); raising the sandbox
stdout cap (the rejected plan-r1 D2 — the in-process sandboxes buffer the
whole stdout before truncating, so a larger cap bounds nothing, and a
sandbox-side truncation leaves the controller nothing to spill).

## 3. Invariants
- I1 `last` changes only when a primary result is registered.
- I2 In-memory registry bytes ≤ `max_registry_bytes` at every step; a result
  above `spill_result_above_bytes` is spilled, not held; the registry stores
  the complete returned value before any preview is derived.
- I2a `read_result(name, start, end)` on a spilled result equals the same
  slice of the in-memory canonical text, for any Unicode content.
- I3 After compaction, `read_result(r<K>)` for every stubbed id returns the
  original payload byte-for-byte.
- I4 Sum of children's reserved cost/tokens/steps never exceeds the parent's
  remaining budget at reservation time; the parent's totals fold in every
  child's actual spend; no child starts a controller/LLM step after the
  parent's deadline (an in-flight provider call may still run to the
  adapter's request timeout).
- I4a A v2 `subcall` (single or batched) never executes Python in the sandbox.
- I5 The base v2 prompt gains one tool and one example; the "exactly one JSON
  object" rule is not weakened.
- I6 Existing single-call `subcall` payloads and all current tests keep
  passing unchanged.

## 4. Acceptance
Hard (deterministic fakes, run in CI):
- AC-1 A grep returning > 10,000 chars is fully readable across consecutive
  `read_result` calls on `last`; the id does not change between reads.
- AC-2 Ids are contiguous across a run containing a soft nudge, a repeated-
  output retry and a folded-in subcall; `last` equals the highest id.
- AC-3 A result above `spill_result_above_bytes` is spilled and readable; a
  run whose results exceed `max_registry_bytes` never holds more in memory
  (asserted via the registry's accounting hook); a `chunk()` result and a
  batched-subcall list are stored as canonical JSON and sliced by character.
- AC-3a Character-offset reads on a spilled result containing CJK and emoji
  match the in-memory slice exactly.
- AC-4 Fake adapter with a fixed `context_window` and a chars/4 message
  counter: compaction fires exactly when the candidate `call_messages`
  (including the last-step `force_final_nudge`) crosses the threshold and
  never when headroom suffices; a fake without the message-counter
  capability takes the fallback path and still compacts.
- AC-4a A `compaction` raw row canonicalises to `action_type: compaction`
  in the server helper, telemetry and the replay builder, is never mapped
  to `inspect`, and is not promoted to `final` when it is the terminal row
  of a successful run.
- AC-5 Fake adapter raising overflow once on step N → compaction + successful
  retry of N; raising twice on N → existing fallback with `overflow_at_step`
  recorded.
- AC-6 Four concurrent children never exceed the parent's `max_cost` /
  `max_tokens` / `max_steps`; two children may each consume most of the
  same remaining wall-clock interval, and the ledger does not divide that
  interval by child count; neither child starts a step after the parent's
  deadline; a local-provider profile is capped at 2 regardless of the
  requested `max_concurrency`.
- AC-6a A single and a batched v2 `subcall` complete correctly with the
  subprocess sandbox configured (the sandbox is never invoked for them).
- AC-7 Telemetry attributes each child's spend to the parent run; the batched
  result is readable by `read_result` and stubbed by compaction.
- AC-8 AC-1..7 pass on `execute` and `execute_async` (normalised semantic
  trace equality); the v1 free-form path keeps its existing sandbox tests
  on restricted and subprocess (docker as `integration`).

Regression (benchmark, owner-run, not a merge gate):
- AC-9 `longdoc-v1` with an 8K-window local model: overflow-induced limit
  warnings on the ≥50K cases decrease versus baseline; no deterministic-
  accuracy regression; judge deltas within rep-to-rep range. Reaching
  `final` on every case is not required.

## 5. Open questions (resolved)
All three resolved by the owner on 2026-09-30 (tasks T0.4).
- OQ-1 Bind `r<K>` into the sandbox namespace for the in-process sandboxes
  only? **No.** `r<K>` stays controller-only; G0 makes that the clean
  architecture, and binding selected sandboxes would reintroduce divergent
  behaviour for no v2 benefit.
- OQ-2 Should `read_result` count toward `max_steps`? **Yes.** It is
  another model/controller turn; a free readback would let repeated reads
  evade the convergence guard.
- OQ-3 Bump the v2 prompt version to 2.2? **Yes.** `read_result` now and the
  extended `subcall` later change the deterministic protocol surface.
