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
- G4 **Enforceable memory bound.** Per-result cap `max_result_bytes`
  (16 MiB), in-memory total `max_registry_bytes` (64 MiB), spill budget
  `max_spill_bytes` (256 MiB). Oversized/displaced results spill to a
  per-run scratch directory, never dropped while spill budget remains; `last`
  is never dropped; scratch dir removed at run end, including on exceptions.
- G5 **Deterministic, lossless compaction.** Before each LLM call estimate
  the *pending* `call_messages` (adapter `count_tokens(messages=...)`;
  fallback `last_clamp_info` + chars/4 for what was appended since). Compact
  when estimate > `context_window − reserve_tokens` (4,096). Keep system +
  query + last `keep_last` (4) assistant/execution pairs; older execution
  messages become `[result r4: 31,842 chars; read_result('r4', ...)]`
  stubs, older assistant actions become their one-line code. Idempotent.
- G6 **Overflow → compact → retry once** on the same controller step; a
  second overflow falls through to today's fallback with the step recorded.
- G7 **Compaction trace entry** (`role: compaction`; step, tokens
  before/after, ids stubbed, reason `threshold` | `overflow_retry`) shown in
  Traces and Replay, counted by the outcome classifier.
- G8 **Batched subcall action**, backward compatible:
  `{"type":"subcall","calls":[{prompt,query},…],"max_concurrency":int}`;
  results in order, registered as one primary list result.
- G9 **Budget ledger.** Lock-protected, owned by the parent run; pre-reserves
  a partition of remaining cost/tokens/steps/time per child before launch
  (equal split), returns unused headroom on completion. Children get an
  isolated adapter (`copy.copy`) — `_active_model` is mutable.
- G10 **Concurrency cap** from profile config: default 2 for local providers
  (Ollama, LM Studio, vLLM), 4 for cloud; the model's request never raises it.
- G11 Both loop copies (`execute`, `execute_async`) behave identically; all
  three sandboxes (restricted, subprocess, docker) pass the same tests.

**Non-goals** (see #77 table): persistent child sessions, agent messaging,
harness CRUD for memories/skills/subagents, daemon sessions, autonomous
mode, unsandboxed execution, summary-based compaction, refinement (#78),
binding `r<K>` into the *subprocess* sandbox namespace (see plan §1
deviation D1).

## 3. Invariants
- I1 `last` changes only when a primary result is registered.
- I2 In-memory registry bytes ≤ `max_registry_bytes` at every step; a result
  above `max_result_bytes` is spilled, not held.
- I3 After compaction, `read_result(r<K>)` for every stubbed id returns the
  original payload byte-for-byte.
- I4 Sum of children's reserved budgets never exceeds the parent's remaining
  budget at reservation time; the parent's totals fold in every child's
  actual spend.
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
- AC-3 A result above `max_result_bytes` is spilled and readable; a run whose
  results exceed `max_registry_bytes` never holds more in memory (asserted
  via the registry's accounting hook).
- AC-4 Fake adapter with a fixed `context_window` and a chars/4
  `count_tokens`: compaction fires exactly when the pending prompt crosses
  the threshold and never when headroom suffices.
- AC-5 Fake adapter raising overflow once on step N → compaction + successful
  retry of N; raising twice on N → existing fallback with `overflow_at_step`
  recorded.
- AC-6 Four concurrent children never exceed the parent's `max_cost` /
  `max_tokens` / `max_steps`; a local-provider profile is capped at 2
  regardless of the requested `max_concurrency`.
- AC-7 Telemetry attributes each child's spend to the parent run; the batched
  result is readable by `read_result` and stubbed by compaction.
- AC-8 AC-1..7 pass on `execute` and `execute_async`, and AC-1/3 on all three
  sandboxes.

Regression (benchmark, owner-run, not a merge gate):
- AC-9 `longdoc-v1` with an 8K-window local model: overflow-induced limit
  warnings on the ≥50K cases decrease versus baseline; no deterministic-
  accuracy regression; judge deltas within rep-to-rep range. Reaching
  `final` on every case is not required.

## 5. Open questions (defaults apply if unanswered)
- OQ-1 Bind `r<K>` into the sandbox namespace for the in-process sandboxes
  (restricted/local) only? Default: **no** in this feature; `read_result` is
  the only access path (plan D1).
- OQ-2 Should `read_result` count toward `max_steps`? Default: **yes**
  (it is an LLM turn; budgets stay honest).
- OQ-3 Bump the v2 prompt version to 2.2? Default: **yes**; the loop's
  runtime fingerprint records it.
