# RLM Prompt Tuning Guide

This guide is for teams customizing RLM prompts in RLM Studio.

The main rule is simple: prompt changes must respect the actual tool contract. If a prompt teaches an impossible workflow, better wording will not save it.

## Recommended Defaults

- Keep the built-in v2.0 base prompt as the protocol layer.
- Treat profile prompt text as a supplement, not a replacement.
- For multi-document runs, read the document index first.
- Prefer direct jumps via `content_start` offsets from the index.
- Use `grep(..., context_lines=...)` when you need local context around a marker or keyword.
- If coverage is incomplete, instruct the model to say so explicitly.
- When a preview is truncated, continue it with `read_result` at the offset the marker gives; never re-run the same inspection to "see more".

## Safe Multi-Document Pattern

Use this mental model:

1. `peek(0, 1200)` to read the document index.
2. Use `content_start` from the index to jump into a relevant file.
3. Use `grep()` when you need to find a section or term inside that file space.
4. If `grep()` is used, rely on the returned `char_offset` for follow-up `peek()` calls.
5. Finalize once the answer is supported, and state any coverage gaps.

## Long Results and `read_result`

Every v2 inspect result is kept in full on the controller side as `r1, r2, …` (the newest is also `last`). The model sees at most `result_preview_chars` characters of it, and a truncated preview ends with:

```text
... (preview truncated: 31,842 chars total; read_result('r4', start=10000) for more)
```

The marker names the result id and the exact offset to continue from. A supplement that deals with long outputs should say so in those terms:

- "If a result was truncated, call `read_result` with the id and `start` from the marker."
- "Do not repeat a grep or peek to see more of the same output."
- "Reads never change `last`; use the id from the marker when you have inspected something else since."

Do not tell the model that output is "cut off" or "lost": it is not, and that wording sends it back into re-running tools.

## Anti-Patterns

Do not put these into custom prompts:

- "Use grep, then peek from the returned position" unless grep actually returns a character offset.
- "You must inspect every file before finalizing" without a budget escape hatch.
- "Use your full step budget" or other instructions that fight convergence controls.
- "Re-run the search with a smaller range to see the rest" — the rest is one `read_result` away.
- Prompt text that restates the full JSON protocol differently from the base prompt.
- Model-specific heuristics presented as universal rules.

## Good Supplement Characteristics

A good profile supplement should:

- narrow behavior for a workload, not redefine the protocol
- tell the model how to handle incomplete evidence
- bias toward supported tools and observable outputs
- be short enough that it does not crowd out execution history

## Model-Specific Guidance

For smaller or local models:

- keep supplements shorter
- avoid rigid multi-step recipes
- prefer direct offset-based navigation over inferred navigation
- use lower step budgets unless the model has already shown stable loop behavior

For stronger frontier models:

- encourage per-document summaries before synthesis
- allow deeper inspection when the query truly spans multiple files
- still require explicit uncertainty when evidence is incomplete

## Validation Checklist

Before shipping a prompt change, run these scenarios:

- Single-document summary
- Targeted question about file 2 in a multi-file upload
- Cross-document comparison where both files matter
- Small-model malformed JSON case
- Budget-limited run where full coverage is impossible

The change is not ready if any of those scenarios regress materially in answer quality, step count, or completion behavior.

## Evidence-Gated Refinement

Settings → Profiles can propose a new RLM supplement from a profile's own failed runs and test it before anything changes. Open it with the flask button ("Refine RLM prompt") on a custom profile's card. Built-in profiles do not have it; clone one first.

1. **Propose.** Tick runs of this profile to learn from: the list shows its recent RLM runs with their outcome and a "degraded" mark. Pick the Chat Provider that writes the proposal; it defaults to the one used by the newest ticked run. One model call reads a redacted digest of those runs, the profile's current RLM supplement and the fixed base prompt, and returns a candidate supplement with a short rationale. The panel shows the rationale, a diff, and the current supplement next to the candidate.
2. **Evaluate.** Choose held-out cases from `benchmarks/longdoc-v1.yaml` (the cases tagged `holdout` are preselected), the repetitions (2–5), an evaluating Chat Provider that uses this profile, and optionally a judge. Before you start, the panel shows the run count: cases × repetitions × 2, one run with the current supplement and one with the candidate, at temperature 0. Cost is not estimated up front; it appears in telemetry afterwards. A case whose document is identical to one of the source runs is excluded automatically, so the candidate is never scored on the runs it came from.
3. **Decide.** Results show each case's pass rate for the current and the candidate supplement, plus accuracy, judge, cost, step and outcome deltas. If the candidate passes a case with an expected answer less often than the current supplement, the case is flagged, Reject becomes the default, and Accept needs a tick to confirm. The flag is advisory: you can accept a trade-off on purpose, and the acknowledgement is recorded.
4. **Roll back.** An accepted refinement can be rolled back while its text is still the profile's RLM prompt. Rollback restores the exact prior state; a profile that followed a prompt group follows that group again.

What it guarantees:

- Nothing changes the profile until you click **Accept**. Proposing, evaluating and rejecting never write it.
- Accept refuses (409) when the profile's RLM prompt changed after the proposal was made.
- Accept needs a complete evaluation: every chosen case and repetition ran for both supplements, and at least one case has an expected answer. Judge-only cases are reported but never gate.
- The base prompt is never edited; only the profile's RLM supplement changes.
- Every refinement keeps its evidence: source runs, the exact digest sent, proposal model and prompt version, case and run ids, deltas, the decision and any rollback. Download them as JSONL with **Export history** in the panel (`GET /api/profiles/{id}/refinements/export.jsonl`).

Before you use it:

- **Only RLM runs recorded by this version are listed.** Runs need the profile and a document hash, which older runs do not have.
- **The digest goes to the proposal provider.** API-key and token shapes and lines like `password=…` are redacted, but a secret written in free text is not. Choose source runs and the provider accordingly; the stored digest shows exactly what was sent.
- **Evaluation needs the benchmark dataset.** From a source checkout it is found automatically. An installed Studio needs `RLM_STUDIO_REFINEMENT_DATASET` pointing at a benchmark YAML; until then the case list and Evaluate return 503.
- **One server process.** Refinement locks and the running-evaluation check are in-process. A restart marks a running evaluation `interrupted`; interrupted and failed refinements can only be rejected, so start a new one.
- **Proposals are synchronous.** Under `npm run dev` a slow local model can exceed the frontend proxy timeout; the packaged app has no proxy.
- If Accept or Roll back reports that the profile was saved but recording the decision failed, the profile holds the new text and the refinement's status is out of date. Check the "RLM prompt now" line in the panel; there is no automatic repair.

## Example Supplement

```text
Additional guidance for this profile:
- When the document index lists multiple files, read it first and use each file's content_start offset to jump directly into relevant files.
- Use grep(..., context_lines=...) when you need local context around a marker or keyword; grep returns char_offset values for precise follow-up peeks.
- For cross-document questions, build a short per-document summary before synthesizing.
- If not all relevant files were inspected within budget, say which files were covered and lower confidence accordingly.
```
