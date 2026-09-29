---
feature_id: benchmarks-v1
spec: ./spec.md
plan: ./plan.md
status: draft
date: 2026-08-15
---

# Tasks — Reproducible benchmarks v1

Branch `feat/benchmarks-v1` after `feat/rebrand-rlm-studio` merges (may run in parallel with `feat/interop-official-rlm`; the `rlm_official` engine row is added when that lands). One commit per phase; CI green each time. Allowlist in `plan.md` §2 is binding. Apply defaults for OQ-1..3 in `spec.md` §6.

## Phase 0 — confirmations (no edits)
- [x] T0.1 Read `src/rlmstudio/benchmark/{dataset,runner,report}.py` and `run_matrix_comparison.py`; confirm the slot-builder wiring in `server/dependencies.py` can be reused from the application layer without importing `server/`. *(2026-09-29: it cannot — per-slot adapter construction lives on `AppState` (`server/dependencies.py:944/:1352`, used from `routes/compare_matrix.py:385-434`). A small slot builder is factored into `application`/`infrastructure` during interop Phase 4 and reused here. See `doc_internal/plans/2026-09-29-release-1.0.0-execution-plan.md` §M2.)*
- [x] T0.2 Pick and record dataset sources + licenses (Gutenberg, RFCs, gov reports, repo docs). *(2026-09-29: recorded in the YAML `sources:` block — deterministic synthetic docs (MIT, generated per run, contamination-free), this repo's docs (MIT), RFC 9112 / RFC 9110 (IETF Trust BCP 78), Project Gutenberg #1342 Pride and Prejudice (public domain / PG License). Gov reports dropped: no stable plain-text URL worth pinning. Public texts are fetched on demand and pinned by sha256 so the tracked set stays ~15 KB.)*

## Phase 1 — dataset
- [x] T1.1 Extend `BenchmarkCase` + loader validation; tests. *(2026-09-29: `task_type`, `min_tokens`, `rubric_hint`, `match` (exact | contains, "a || b" any-of), `budget`, `source`, `content_spec`, `expected_spec`; `BenchmarkDataset.sources`; both loaders share `_case_from_dict`. New `benchmark/corpus.py`: `generator: synthetic` (seeded, chars/4 sizing, facts at start/middle/end, spread repeats), `content_file`, `source_url`+`sha256` (fetch on demand, digest verified), and `expected: {count_word | count_sentence}` derived from the materialised text. `tests/test_benchmark_dataset.py` (22).)*
- [x] T1.2 Author `benchmarks/longdoc-v1.yaml` (≥12 cases across ~5K/50K/150K tokens; four task types; `sources:` block). *(14 cases: small ×4, medium ×5, large ×5; needle ×5, synthesis ×2, aggregation ×5, refusal ×2; per-case budgets. Public-text aggregation targets are counted from the text, so even the novel is contamination-free; the RFC synthesis case is flagged `contamination_risk`.)*

## Phase 2 — matrix driver + scoring
- [x] T2.1 `benchmark/matrix_runner.py` (providers × engines → `MatrixSlotDTO`s; per-case budgets; trace metrics extraction in the application layer). *(2026-09-29: `MatrixBenchmarkRunner` drives `RunMatrixComparisonUseCase` over slots from a pluggable `SlotBuilder`; the default wraps `rlmstudio.api.build_matrix_slots` — made public, seeded by a per-case `base_config` (budgets) and taught `rlm_official` (engine priced by the slot's LLM; catalog endpoint fallback for local backends) — so the public Python client gains the mode too, closing the interop Phase-4 caveat. Outcomes are classified with `classify_execution_outcome` (failures/timeouts counted, never dropped), `cost_known` and median TTFT extracted from the raw trace; `reps`, `limit`, `case_ids`, progress callback; skipped public texts carried into the results.)*
- [x] T2.2 `benchmark/scoring.py` (exact/contains + judge via `LLMPort`, `judge_pointwise.yaml`); tests with fakes. *(`matches()` normalises case/whitespace/edge punctuation and accepts `"a || b"` alternatives; `JudgeScorer` reuses `judge_pointwise.yaml` through any `LLMPort`, appends the case `rubric_hint` as grading guidance, truncates the source like the Studio judge, records model + prompt version, flags unparseable verdicts (fallback 3.0). Tests: `test_benchmark_scoring.py`, `test_benchmark_runner.py`, `test_api_build_matrix_slots.py`.)*

## Phase 3 — report + page
- [ ] T3.1 Markdown table + aggregates in `report.py`; marker-based `BENCHMARKS.md` regeneration; idempotency test.
- [ ] T3.2 `BENCHMARKS.md` skeleton (methodology, caveats, links) with markers.

## Phase 4 — CLI + CI
- [ ] T4.1 `rlm-studio bench` subcommand + `benchmarks/run_benchmark.py` wrapper; `--dry-run` with fakes.
- [ ] T4.2 CI `bench-smoke` step (2 cases × 2 engines, no network).

## Phase 5 — real run (owner)
- [ ] T5.1 Configure providers (env vars per `docs/hosts/`); run cloud ×2, local ×1, engines ×3–4, three reps for cloud; commit `benchmarks/results/<date>/` + regenerated `BENCHMARKS.md`; log total cost.

## Phase 6 — docs
- [ ] T6.1 README "Where RLM Studio shines" cites the table; `docs/rlm-studio-guide.md` "Reproducing the benchmarks"; CHANGELOG.

## Phase 7 — acceptance
- [ ] T7.1 AC-1..AC-4 from `spec.md`; record in `doc_internal/v1.0.0-rlm-studio/MANUAL_TEST_PLAN.md`.
