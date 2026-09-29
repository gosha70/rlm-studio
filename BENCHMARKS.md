# Benchmarks

Reproducible numbers for the question RLM Studio exists to answer: **does the
Recursive Language Model pattern beat direct prompting or retrieval on long
documents — with which model, at what cost?** Every number on this page is
produced by the tool itself (`rlm-studio bench`) and regenerated between the
markers below; nothing here is hand-edited.

> **This is not the paper's benchmark.** The RLM paper
> ([arXiv:2512.24601](https://arxiv.org/abs/2512.24601), Zhang, Kraska,
> Khattab) reports OOLONG / BrowseComp-Plus results for its own implementation
> ([alexzhang13/rlm](https://github.com/alexzhang13/rlm)). The set here is a
> small, long-document task set designed to be reproducible from a fresh clone
> at a documented cost. It measures *this workbench's engines on these tasks*,
> nothing more.

## What is measured

- **Dataset:** [`benchmarks/longdoc-v1.yaml`](benchmarks/longdoc-v1.yaml) — 14
  cases across three size buckets (~5K / ~50K / ~150K tokens) and four task
  types: *needle* (one fact), *synthesis* (open answer, judge-graded),
  *aggregation* (count occurrences — exact target derived from the text),
  *refusal* (the answer is not in the document). Documents are deterministic
  synthetic texts with planted facts (contamination-free), this repository's
  own docs, and two public texts fetched on demand and pinned by sha256
  (RFC 9112 / 9110, Project Gutenberg #1342). Licences are listed in the
  dataset's `sources:` block.
- **Engines:** `direct`, `rag`, `rlm` (Studio's loop), and `rlm_official`
  (the paper authors' `rlms` package run under Studio's budgets — needs
  `pip install "rlm-studio[interop]"`).
- **Providers:** every cell is one provider × engine; the same model id,
  endpoint and key are used for every engine of a provider.
- **Scoring:** two independent signals. *Accuracy* is a normalised
  exact/contains match against the case's expected answer (needle,
  aggregation, refusal). *Judge* is the pointwise rubric in
  `src/rlmstudio/prompts/judge_pointwise.yaml` sent to the named judge model,
  with the case's rubric hint as grading guidance; reported as mean ± range
  across repetitions. Failed and timed-out runs are counted in their own
  column and never dropped.
- **Determinism aids:** temperature 0 on every slot; the judge model and
  prompt version are recorded; per-case budgets (steps, wall-clock, cost) come
  from the dataset.

## How to reproduce

```bash
pip install "rlm-studio[all]"                    # engines incl. rlm_official
export OPENAI_API_KEY=…  ANTHROPIC_API_KEY=…      # cloud providers you include
rlm-studio bench --config benchmarks/longdoc-v1.yaml \
  --providers openai/gpt-4o-mini,anthropic/claude-haiku-4-5,ollama/qwen3:8b \
  --engines direct,rag,rlm,rlm_official \
  --judge openai/gpt-4o-mini --reps 3 --fetch \
  --out benchmarks/results/$(date +%F)/ --page BENCHMARKS.md
```

`--fetch` downloads the two public texts into `benchmarks/corpus/` (verified
against the pinned digests). Without it those cases are skipped and listed as
such. `--dry-run` runs the whole pipeline on fakes with no network — CI does
this on every push so the runner cannot rot.

Expected cost of a full run at cheap cloud tiers: see the *total cost of this
run* line under the results; the target is under ~$25.

## Caveats

- The judge is an LLM; treat its scores as a ranking aid, not ground truth.
  The accuracy column is deterministic and is the number to trust first.
- `rlm_official` does not stream, so its TTFT is always "—" and it ranks last
  on TTFT by construction. When the engine reports no price and Studio has
  none, the cell shows the run count with unknown cost instead of a $0 that
  would look free.
- The RFC synthesis case is graded purely by the judge on a well-known
  standard; it is flagged `contamination_risk` in the dataset. Every other
  target is either invented (synthetic documents) or computed from the text.
- Local rows name the hardware in the run notes; cloud rows carry the exact
  model id. Numbers drift with provider updates — each run is dated.

## Results

<!-- bench:start -->
_No results yet — run `rlm-studio bench … --page BENCHMARKS.md` to populate this section._
<!-- bench:end -->
