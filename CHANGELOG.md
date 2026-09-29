# Changelog

All notable changes to RLM Studio (formerly RLMKit) are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Official-engine interop (`rlm_official`)

- **Added:** a fourth execution mode, `rlm_official`, that runs the paper
  authors' implementation ([alexzhang13/rlm](https://github.com/alexzhang13/rlm),
  PyPI `rlms`) as an engine inside the Compare matrix and Chat — under the
  same budgets, traces, telemetry and ranking as Studio's own loop — so one
  run can benchmark Official-RLM vs Studio-RLM vs Direct vs RAG on the same
  document, provider and model. Optional: `pip install "rlm-studio[interop]"`
  (included in `[all]`). `GET /api/engines` reports availability; the UI
  disables the option with the reason while the package is absent, and a
  request naming the mode is rejected with the same reason before any
  session or telemetry side effect. Provider mapping: `openai` / `anthropic`
  use the engine's native clients; `ollama` / `lmstudio` / `vllm` go through
  its OpenAI-compatible client at the provider's endpoint. Studio's `docker`
  sandbox maps to the engine's Docker environment; any other sandbox runs the
  engine's in-process `local` REPL, which is not isolated and is flagged on
  every such run. Cells and traces carry an "official rlms x.y.z" badge.
- **Depth and concurrency semantics:** Studio counts levels of sub-RLM while
  the engine counts the depth at which a node degrades to a plain model call
  and its root already occupies one level, so the recursion budget is
  translated (N → N+1); a budget of 0 cannot be reproduced faithfully (the
  engine downgrades sub-calls instead of refusing them) and says so in the
  run's notes. Because the in-process `local` REPL redirects the whole
  process's output and working directory around every code cell, official
  runs and their batched sub-calls are serialised; a cell may spend up to a
  quarter of its timeout queueing, that time comes off its own budget, and a
  cell still waiting after it reports the engine as busy. The rule covers
  official runs against each other, not Studio's own in-process sandbox. The
  engine's deadline is also set just inside Studio's wall-clock guard so a run
  that runs out of time stops itself and reports its spend, leaving the guard
  as the backstop for a REPL stuck inside one iteration. The Docker sandbox
  has none of these restrictions.
- **Known limitations:** the engine does not stream (no TTFT / decode
  timings, no token events over WebSocket); it reports token usage per run,
  so step rows show 0 tokens while totals are exact; a run whose engine
  reports no price and for which Studio has none is flagged
  (`cost_known=false`) and ranks last on the cost metrics rather than
  counting as free, and a $0 answer from Studio's price table for a *cloud*
  model that did consume tokens counts as no price rather than as free, while a
  locally served model keeps $0 as its real price; a run the engine
  stopped for breaking a cap still reports the spend, steps and partial
  answer it had reached, and a token breach with no price attached is priced
  from the slot's own cost table; conversation memory is not delivered to the engine; the
  mode is UI / REST only — the Python client's `compare_matrix()` does not
  accept it in 1.0. A REPL execution that never returns is stopped by
  Studio's wall-clock guard (the engine only checks its timeout between
  iterations); with the `local` environment the abandoned thread lives on
  until the server restarts, and because the engine runs that code inside the
  server process, code that never finishes keeps the process's output
  redirected and its working directory moved and makes every later official
  run report the engine as busy. The Docker sandbox has none of that: the
  container owns the runaway code.
- **Architecture:** new `RLMEnginePort` (application layer) and
  `infrastructure/engines/rlms_adapter.py`; `RunRLMOfficialUseCase` enforces
  the wall-clock and post-run token / cost caps around the engine. Mode
  literals are now centralised as the `ExecutionMode` / `ChatMode` aliases
  next to the `MODE_*` constants.

### Reproducible benchmarks (`BENCHMARKS.md`)

- **Added:** `rlm-studio bench` — runs a YAML task set across providers ×
  engines through the same code path as the Compare page, scores every
  cell with a deterministic exact/contains match and the pointwise LLM
  judge (`judge_pointwise.yaml`, rubric hints per case, model + prompt
  version recorded), classifies failures and timeouts with the Dashboard's
  outcome classifier so they are counted rather than dropped, and writes
  `results.json` / `results.md` plus the results section of
  `BENCHMARKS.md` between its markers. `benchmarks/run_benchmark.py` is a
  thin wrapper.
- **Added:** `benchmarks/longdoc-v1.yaml` — 14 cases across ~5K / ~50K /
  ~150K tokens and four task types (needle, synthesis, aggregation,
  refusal). Documents are deterministic synthetic texts with planted facts
  (contamination-free), this repository's docs, and two public texts
  (RFC 9112 / 9110, Project Gutenberg #1342) fetched on demand and pinned
  by sha256; aggregation targets on the public texts are counted from the
  text, so they are contamination-free too. Licences are listed in the
  dataset's `sources:` block.
- **Added:** `BENCHMARKS.md` — methodology, how to reproduce, caveats, and
  an explicit "not the paper's benchmark" note; the results block is only
  ever regenerated by the runner.
- **Changed:** the public Python client's `compare_matrix()` now accepts
  `rlm_official`, and `rlmstudio.api.build_matrix_slots` /
  `build_llm_adapter` are public.
- **CI:** a `bench-smoke` step runs the whole pipeline on offline fakes
  (every engine through the real use cases, a failing provider, the judge
  path, page regeneration) on every push.

### Security

- `anyio>=4.14.2` — CVE-2026-63374 (TLSStream IDNA-2003 host-name encoding
  enables certificate spoofing; critical) and CVE-2026-64847 (transitive via
  httpx / anthropic / google-genai).

### Rebrand: RLMKit → RLM Studio

- **BREAKING:** the project is now **RLM Studio**. PyPI distribution
  `rlm-studio` (the `rlmkit` name on PyPI belongs to an unrelated
  project — there is no shim), import package `rlmstudio` (was
  `rlmkit`), console script `rlm-studio` (was `rlmkit`; subcommands
  `studio` / `version` unchanged), GitHub repository
  `gosha70/rlm-studio`.
  **Migration:** replace `import rlmkit` → `import rlmstudio`,
  `rlmkit studio` → `rlm-studio studio`, and install with
  `pip install "rlm-studio[all]"` (or `[studio]` for the API + UI only).
- **BREAKING (one-release compatibility):** environment variables use
  the `RLM_STUDIO_*` prefix; every `RLMKIT_*` name is still honoured with
  a one-time deprecation warning. State lives in `~/.rlm-studio/`
  (override `RLM_STUDIO_DIR`); on the first *server boot* after upgrading,
  an existing `~/.rlmkit/` is copied — never moved or deleted — to
  `~/.rlm-studio/`. The copy runs only from the FastAPI startup path
  (`rlm-studio studio`, `python -m rlmstudio.server`, uvicorn, Docker),
  never on import, in tests, or on `rlm-studio version`.
  Also: `./rlm_studio_config.{yaml,json}` searched before the legacy
  `./rlmkit_config.*`; OS-keyring service `rlm-studio` with read-fallback
  to `rlmkit` (keys are re-saved under the new name); Docker Compose
  volume `rlm-studio-data` (one-off copy recipe from `rlmkit-data` in the
  compose header); frontend `localStorage` keys `rlm_studio_*` migrated
  from `rlmkit_*` on first load; `NEXT_PUBLIC_RLM_STUDIO_PERF_UI` with
  the legacy `NEXT_PUBLIC_RLMKIT_PERF_UI` honoured; telemetry JSONL export
  extras keys renamed `rlmkit_*` → `rlm_studio_*`; default sandbox image
  `rlm-studio-sandbox`.
- **BREAKING:** the legacy Streamlit prototype (`ui` extra) is removed
  in favour of the Next.js Studio UI. `pip install "rlm-studio[all]"` no
  longer installs the dev toolchain. Extras are now `studio` (API + UI;
  `server` kept as an alias for one release), `providers` (direct
  provider SDKs — currently `anthropic`; `openai` is satisfied through
  the base `litellm` dependency, and the Ollama client uses `requests`,
  which is now an explicit base dependency), `eval`, `interop`
  (placeholder), `dev`; `all = studio + providers + eval + interop`.
- Product identity is centralised in `rlmstudio.branding` (Python) and
  `frontend/src/lib/branding.ts` (TypeScript).
- Docs: `docs/RLMKit_Design_Document.md` → `docs/RLM_Studio_Design_Document.md`
  (Learn slug `rlm-studio-design-document`), logo
  `docs/RLM_Studio_Logo.png`; README repositioned around the workbench
  ("does RLM beat Direct/RAG on your data, with your model?") with paper
  and official-repo attribution; `docs/OPERATIONS.md` §2.1a documents the
  upgrade path.
- **BREAKING:** minimum Python is now **3.11** (was 3.10). `litellm >= 1.98.0`
  — the adapter every LLM call goes through — imports `typing.NotRequired`
  at import time, which is 3.11+, while still advertising `requires_python
  >=3.10`; 1.99.0rc1 carries the same import, so it will not self-heal.
  Pinning 3.10 users below litellm 1.98 would freeze them under future
  litellm security fixes, and Python 3.10 reaches end-of-life in October
  2026. CI, classifiers and docs now target 3.11 / 3.12 / 3.13.
- Release plumbing: `release.yml` requires the tag and the `pyproject.toml`
  version to match exactly (normalising `v1.0.0-rc.1` → `1.0.0rc1`) and
  re-checks the built wheel's version, so a release-candidate run cannot
  upload — and permanently burn — the final release version on an index.
  `pyproject.toml` carries `1.0.0rc1` until the release PR sets `1.0.0`.
- Housekeeping: `debug_parsing.py` removed; `cryptography>=50.0.0`,
  `aiohttp>=3.14.3`, `pypdf>=6.15.0`, `gitpython>=3.1.58` floors
  (pip-audit clean); coverage omit list emptied.

### Prefill / decode telemetry

#### Approved deviations from spec v1.7

The `doc_internal/specs/prefill-decode-telemetry.md` spec is
gitignored, so approved deviations live here to travel with the PR:

- **AC-8**: spec wording asks for a wall-clock assertion ("first
  websocket `token` arrives within `prior_baseline_ttft_ms + 50` ms").
  Implemented as an order-based regression guard instead —
  `tests/test_websocket_streaming.py::test_first_post_connect_event_is_token_not_aggregate`
  asserts the first post-`connected` event is a `token`, not an
  aggregate. Catches the same class of bug (buffering regression
  that holds chunks until end-of-stream) without CI timing flake.
- **Three hot-path call sites → two**: spec §Prerequisite refactor
  names three `complete_stream_async` consumers. `git grep` finds
  only two (`run_rlm.py:1019`, `run_direct.py:198`). Both migrated.
- **Per-adapter test layout**: spec mentions
  `tests/infrastructure/llm/test_*_adapter.py`; this repo has a flat
  `tests/` layout. New tests added there rather than inventing a
  new directory.
- **Legacy adapters' `ttft_ms`**: `openai`/`anthropic`/`ollama`
  wrappers have no streaming backend, so `ttft_ms` is `None` on
  their terminal `StreamChunk.response`. `StreamChunk.response`
  itself is always non-None for successful calls.
- **`_rank` perf-aggregate computation**: the use-case-layer ranker
  computes aggregates from the raw-DTO trace keys directly rather
  than materializing via `server/routes/_helpers.py`, to respect
  the Clean-Architecture dependency rule
  (`application/` must not import from `server/`).
- **`RLMKIT_STREAMED_COMPLETE` scope**: the flag gates
  `complete()` / `complete_async()` only — `complete_stream_async`'s
  Protocol signature change is unconditional (Protocols cannot
  branch on env).
- **`judge_score` backend passthrough**: pre-existing drift —
  frontend `MatrixRankingMetric` included `judge_score` but backend
  rejected it. Widened the backend Literal and added a
  no-op passthrough branch in `_rank` (frontend re-ranks
  client-side) so the request doesn't 422.

- **Phase 6 — DGX Spark cookbook §4b (spec v1.7).** One paragraph added
  to `docs/hosts/dgx-spark.md` calling out that vLLM's
  `--enable-prefix-caching` flag is the single biggest knob for
  RLM-on-Spark wall-time. AC: 10.
- **Phase 5 — Compare perf fields, ranking widen, frontend tab, feature flag (spec v1.7).**
  `CompareMatrixSlotResponse` gains six per-slot perf aggregates
  (`total_prompt_tokens`, `total_completion_tokens`,
  `total_cached_tokens`, `total_decode_ms`, `median_ttft_ms`,
  `cache_hit_rate`), populated from each slot's trace at write time
  so frontend ranking doesn't need a second fetch. All three
  request payload classes (`CompareMatrixRequest`,
  `CompareMatrixRequestV2`, `CompareMatrixUnifiedRequest`) widen
  the `ranking_metric` Literal to include `"ttft"`,
  `"decode_tokens_per_sec"`, `"cache_hit_rate"`; the backend ranker
  in `RunMatrixComparisonUseCase._rank` handles the three new
  metrics (computed directly from the raw-DTO trace keys to respect
  the dependency rule). Frontend `MatrixRankingMetric` union and
  `CompareMatrixSlotResponse` interface mirror the additions; the
  Compare page's client-side ranker switch grows three new branches
  that read the new per-slot fields and sort null-telemetry slots
  last. A Traces "Performance" tab appears when
  `NEXT_PUBLIC_RLMKIT_PERF_UI=1`, rendering a per-step table of
  prompt/completion/cached tokens, TTFT, decode, and duration.
  ACs: 6, 16, 17, 18.
- **Phase 4 — classifier PREFILL_TIMEOUT + write-time persistence + route helpers (spec v1.7).**
  `OutcomeCategory` gains `PREFILL_TIMEOUT`;
  `classify_execution_outcome` accepts an optional `trace:
  ExecutionTrace | None` kwarg and promotes a timeout failure to
  PREFILL_TIMEOUT when ≥2 steps satisfy `ttft_ms / (duration_ms) >
  0.7` — two or more prefill-heavy steps is the signature of
  prefix-cache failure vs. a single cold-start. `TelemetryStore.
  record_run` gains an `outcome_category: str | None` kwarg; the
  store stays a dumb sink that writes verbatim. New route helper
  module `server/routes/_helpers.py` exposes
  `_translate_raw_trace_entry` (raw-DTO → `TraceStep`) and
  `_materialize_trace` (tolerant wrapper); both chat and
  compare_matrix materialize the trace, classify, and pass the
  persisted category through `record_run`. `_RunPoint` in
  `metrics.py` gains `outcome_category`; the dashboard's failure
  aggregator now prefers the persisted value and only re-derives
  for legacy `NULL` rows — legacy TIMEOUT rows are never
  retroactively promoted. New troubleshoot entry
  `prefill-timeout-enable-caching` surfaces on the Learn tab.
  Judge (`server/judge.py`) is unchanged per AC-28. ACs: 7, 13b,
  14, 15, 21, 24, 25, 26, 27, 28.
- **Phase 3 — use-case trace writers + REST/replay shape + CI guard (spec v1.7).**
  Every `TRACE_KEY_ROLE: "assistant"` dict in `application/use_cases/`
  now populates the four new telemetry keys (`ttft_ms`, `decode_ms`,
  `cached_tokens`, `cache_write_tokens`) from the returning
  `LLMResponseDTO`. Eleven assistant-role sites covered across
  `run_rlm.py` (8), `run_direct.py` (2 — sync + async branches), and
  the Phase-2 `run_rag.py` site. The server-side `TraceStep` Pydantic
  model, the `/api/traces/{id}` response builder, and the Learn-tab
  replay converter (`trace_to_replay.py`) mirror the six new fields;
  frontend `TraceStep` and `LearnReplayStepMetrics` interfaces get
  matching optional fields. A new CI guard test
  (`tests/test_trace_writers.py`) greps the use-case tree for every
  assistant dict and fails if any is missing the four keys — blocks
  future PRs that add a new assistant-role writer without attending
  to telemetry. ACs: 9, 29.
- **Phase 2 — data model + cache extraction + store migration + RAG two-step trace (spec v1.7).**
  `TraceStep` gains six fields (`prompt_tokens`, `completion_tokens`,
  `ttft_ms`, `decode_ms`, `cached_tokens`, `cache_write_tokens`) plus a
  `TraceStep.from_dict` classmethod (AC-23). `ExecutionTrace` gains four
  derived properties (`total_prompt_tokens`, `total_completion_tokens`,
  `cache_hit_rate` capped at 1.0, `median_ttft_ms`) — AC-4.
  `LiteLLMAdapter._extract_cache_tokens` handles Anthropic
  (`cache_read_input_tokens` + `cache_creation_input_tokens`) and OpenAI
  (`prompt_tokens_details.cached_tokens`) schemas; returns `(0, 0)` for
  unknown providers — AC-3. Cache counts are now populated on every
  streamed LiteLLM response. Telemetry store gains a migration harness
  keyed on `PRAGMA user_version`: `_SCHEMA_SQL` stays frozen at the v1
  baseline; `_MIGRATIONS` is the single source of truth for post-v1
  schema changes. The v2 block adds the six step columns +
  `runs.outcome_category` (the column is added now; writers wire up in
  Phase 4) — AC-5, AC-20, AC-22. `run_rag.py` now emits a two-entry
  trace: step 0 is `rag_retrieval` (retrieval-side metrics unchanged),
  step 1 is `assistant` with the four new telemetry keys populated from
  the LLM response DTO — AC-13a. Route-layer materialization (AC-13b)
  arrives in Phase 4.
- **Phase 1 — streaming-under-the-hood (spec v1.7).** `LLMPort.
  complete_stream_async` now yields `StreamChunk` events; the terminal
  chunk carries a populated `LLMResponseDTO` with four new fields
  (`ttft_ms`, `decode_ms`, `cached_tokens`, `cache_write_tokens`).
  `LiteLLMAdapter.complete()` and `.complete_async()` stream under
  the hood by default so TTFT and decode_ms are measured on every
  call. The legacy `openai` / `anthropic` / `ollama` / `mock`
  adapters migrate to the new `StreamChunk` signature; `ttft_ms` is
  `None` on the three wrappers because their underlying clients have
  no streaming backend. The two in-repo consumers (`run_rlm.py`,
  `run_direct.py`) read `chunk.response` on the final chunk and drop
  the previous approximate-token synthesis. Panic lever:
  `RLMKIT_STREAMED_COMPLETE=0` restores the pre-Phase-1
  non-streaming behavior for the two sync paths (`complete`,
  `complete_async`); the `complete_stream_async` signature change
  is unconditional because Protocol signatures cannot branch on
  env. Cache-token extraction, TraceStep changes, and store
  migration land in Phase 2.

Learn tab V2 — a self-contained surface that teaches the RLM loop
through a scrubbable, step-by-step replay. Shipped in three slices
(V1 → V2a → V2b) across PRs #20, #23, and #24.

### Added
- **Learn tab (V1, PR #20)**: new `/learn` surface with Concepts,
  Cookbook, and Troubleshooting sub-pages. Mode chooser, diagnostics
  strip, and markdown-backed guides for `rlm-concepts`, `rlm-prompt-tuning`,
  `rlm-studio-guide`, and `lessons-from-ai-copilots`.
- **Replay walkthrough widget (V2a, PR #23)**: 6-node SVG diagram,
  play / pause / step / reset controls at 1× / 1.5× / 2× speeds,
  three-pane layout (controls / diagram / step list + detail). Mounts
  on the Concepts page with a bundled demo replay. Advanced details
  tray is folded into the right pane and resets to collapsed on
  step change.
- **Trace-backed replay (V2b, PR #24)**: deep-link from any execution
  to its own walkthrough.
  - `GET /api/replays/{execution_id}` — pure-service converter
    (`src/rlmkit/application/services/trace_to_replay.py`) that turns
    a canonical `TraceResponse` into a `LearnReplay`. Returns a bare
    `LearnReplay` (no wrapper); 404 with the standard error envelope
    when the id is unknown. Omits null optional fields on the wire to
    match bundled replay JSON.
  - `/learn/replay/[executionId]` — SWR-based page that mirrors
    Concepts' loading / error / ready pattern; reuses the shared
    `ReplayWalkthrough` widget exactly (no second renderer).
    Surfaces a truncation banner when `metadata.truncated` fires.
  - Traces row CTA — small outline "Replay in Learn" button per row
    navigates to the new page; keyboard-safe (Enter/Space on the
    button no longer bubbles into the row's open-trace handler).
  - Shared canonical trace loader — fixes the live-execution path
    so both `/api/traces/{id}` and `/api/replays/{id}` see the same
    `inspect | subcall | final | error` action-type enum.
  - Contract refinement: code-less `inspect` steps (the RLM
    controller's "Runtime fingerprint" preamble) are dropped from
    trace-sourced replays rather than rendered as empty code cards.

### Tests
- +5 converter unit tests for the code-less-inspect refinement
  (real-trace fixture + whitespace-only / null / post-skip-subcall
  edges).
- +5 e2e tests for the replay route (unknown id → 404, telemetry
  round-trip, in-memory canonicalization regression guard, failed
  run with `metadata.failed` + folded error, null-omission wire
  shape, bare `LearnReplay` response).
- +34 unit tests for the `trace_to_replay` service across bookends,
  kind inference, metadata, failure handling, truncation, step
  bounds, ids/titles, and final-step folding.
- +8 frontend tests (5 for the page, 3 for the Traces CTA) covering
  loading / error / ready / truncated states and the keyboard /
  mouse stopPropagation contract.

### Fixed
- Failed-run replays now preserve the trace-side failing output in
  `answer.details.output` instead of duplicating the run-level
  error string from `summary`.
- Truncation banner copy now matches the metadata it cites
  ("The full replay would have had N steps" — `originalStepCount`
  is the pre-truncation replay length, not the raw trace step count).

---

Evaluation & Ops — a cluster of features landed in the same window
that together close the "is my RLMKit deployment healthy, and are
my results any good?" feedback loop. Grouped separately from the
Learn tab work because it ships independently.

### Added (Evaluation & Ops)
- **LLM Tuner / Compare matrix** — new `/compare` page backed by
  `POST /api/chat/compare-matrix`. Runs the same query across a
  Provider × Mode grid in parallel, ranks cells by cost, tokens,
  latency, `answer_per_cost`, or `judge_score`. Ephemeral Chat
  Providers are built on the fly from the picked combinations and
  don't persist to Settings. Synchronous endpoint — the UI renders
  once every cell has completed or errored.
- **LLM-as-judge rubric v2.0** — pointwise scoring across five
  dimensions (relevance, correctness, completeness, coherence,
  conciseness, each 1–5) plus a pairwise variant with an explicit
  `a`/`b`/`tie` winner. `overall_score` is the mean of dimensions,
  rounded to 2 decimals, clamped to [1.0, 5.0]. Rubric prompts live
  in `src/rlmkit/prompts/judge_pointwise.yaml` and
  `judge_pairwise.yaml`. Judge provider is configured app-wide via
  `judge_chat_provider_id`. Non-usable outcomes auto-score without
  calling the judge: 1.0 for `budget_exhausted` with ≥50 chars of
  answer, 0.0 for all other failures — keeps `judge_score` sortable
  across a mix of successes and failures without wasting judge
  tokens.
- **Outcome classifier + failure metrics** — every execution is
  classified into one of `success`, `timeout`, `budget_exhausted`,
  `context_overflow`, `general_error`. Non-success outcomes are
  excluded from cost / latency / token aggregations across the
  Dashboard. New endpoint `GET /api/metrics/failures/{session_id}`
  returns failure rate plus breakdowns by category, provider, and
  mode. Dashboard gains a failure chart.
- **Scheduled connection testing** — new global setting
  `connection_test_interval_minutes` (0–1440, 0 disables). A
  background daemon re-tests up to 5 providers in parallel with a
  10-second per-test timeout. A provider flips to `offline` only
  after 2 consecutive failures (flap avoidance); a single manual
  `Test Connection` success flips it back to connected immediately.
  Per-provider audit fields: `last_tested_at`, `last_tested_by`
  (`manual` | `background`), `consecutive_failures`. Settings UI
  surfaces all three.
- **Conversation memory toggle** — `conversation_memory_enabled`
  flag on `ChatProviderConfig`. RLM / RAG / Auto modes bind prior
  turns as a Python variable `history` inside the sandbox REPL
  (byte-capped, model reads on demand — zero token cost when
  unused). Direct / Compare modes deliver history as an in-prompt
  "Previous conversation:" prefix, token-budgeted (default 30% of
  context window). Flip the toggle off for stateless benchmarking.
- **Trace bulk delete** — Gmail-style selection on the Traces page,
  plus single-row delete with confirm dialog. Backend endpoints:
  `DELETE /api/executions/{id}` for single and
  `DELETE /api/executions` to clear a session.
- **LLM-provider UX polish** — Test Connection in the edit form
  (not just the list row), provider logo in the sidebar, fix for
  the model-dropdown not refreshing after a Test Connection that
  invalidated the previous model list.

### Fixed (Evaluation & Ops)
- Stall breaker now accepts plain-text answers once all uploaded
  files have been inspected, even if the model hasn't emitted an
  explicit final-answer JSON action.
- JSON action parser tolerates trailing braces, prose after the
  JSON block, and `<think>…</think>` wrappers — matches how
  reasoning-tuned models actually emit tool calls.
- PDFs ≥ 50 MB no longer fail with a truncation error during
  upload; the size cap was lifted and chunking handles large files
  incrementally.
- Anthropic requests guard against `temperature` + `top_p` being
  sent together (the Anthropic API rejects this combination with
  an empty response). When a profile sets a custom temperature,
  `top_p` is cleared automatically.
- Timeout warnings surface in the UI before the wall-clock cap
  fires, so users see a warning banner rather than silent
  truncation.

## [1.0.0] - 2026-03-28

Cycle 3 (Quality, DevOps & Release) + Cool-down 3.

### Changed
- **BREAKING:** `ExecutionConfig.default_safe_mode` now defaults to `True` — sandbox restrictions (blocked modules, restricted builtins via RestrictedPython) are active by default. Pass `safe_mode=False` explicitly to opt into unrestricted execution.
- **BREAKING:** `RecursiveController.max_depth` now defaults to `1` (aligned with the RLM paper's evaluated regime). Deeper recursion remains fully supported — pass `max_depth=N` explicitly. Previous default was `5`.

### Added
- CI pipeline: Codecov coverage upload, pip-audit vulnerability scan, Docker sandbox build + smoke-test job
- Coverage gate raised to 80%; omit list scoped to Streamlit-bound files only — pure-Python `ui/data/` and `ui/services/` remain measured
- Pre-commit mypy hook (`uv run mypy src/rlmkit/ --ignore-missing-imports`); ruff scope expanded to all of `src/rlmkit/`
- `execute_async()` on all three sandbox adapters (`LocalSandboxAdapter`, `RestrictedSandboxAdapter`, `DockerSandboxAdapter`) — satisfies `SandboxPort` protocol
- `complete_async()` and `complete_stream_async()` on all legacy LLM adapters (`MockLLMAdapter`, `OpenAIAdapter`, `AnthropicAdapter`, `OllamaAdapter`) — satisfies `LLMPort` protocol; streaming wrappers delegate to `complete_async` via `asyncio.to_thread` to remain non-blocking
- Async API variants: `interact_async()`, `complete_async()` with `api_base` and `timeout` parameters
- Storybook 10 with 27 component stories across 7 components
- Frontend test suite: 175 passing + 21 skipped tests (vitest, Node ≥ 22)
- D1: Unified `interact()` / `interact_async()` API consolidation
- D2: Dashboard charts polished (chart types, color palette, tooltips)
- D3: Frontend cleanup — dead chat components removed, dead API functions removed, error toasts added (`sonner`), Dashboard→Traces deep-link, Traces "Load more" pagination

### Fixed
- Security (CodeQL `py/clear-text-storage-sensitive-data`): replaced bespoke `_update_env_file()` with `SecretStore` dispatch — OS keyring when available, JSON file (`~/.rlmkit/api_keys.json`, chmod 600) as fallback; startup `_reload_stored_api_keys()` restores persisted keys with correct precedence (real env > SecretStore > legacy `.env`)
- `_create_llm_adapter()` return type narrowed from `object` to `LLMPort`; branch-local variable names fixed to prevent mypy incompatible-assignment errors
- `ExecutionSlot.mode` cast to `Literal["rlm","direct","rag"]` in configuration panel
- Anthropic default model updated to `claude-sonnet-4-6` across `api.py` and `RLMKitClient`
- Local providers (`ollama`, `lmstudio`) now require an explicit `model=` argument; no silent default (avoids "model not found" errors on deployments with different models pulled)
- ESLint `globalIgnores` includes `storybook-static/`

### Known Limitations
- **No timeout enforcement in non-main threads (e.g., Streamlit):** When the REPL environment runs outside the main thread, signal-based timeouts are unavailable and process-based timeouts would break variable persistence. Code execution proceeds without a timeout guard in these contexts. A warning is now logged when this path is taken. Out-of-process sandbox execution is planned for v1.1.0.

## [0.2.0-alpha.3] - 2026-02-11

### Added
- WCAG 2.1 AA accessibility: 32 issues fixed across 20 files (ARIA labels, keyboard navigation, focus management, color contrast, reduced motion support)
- End-to-end WebSocket streaming with `ExecutionEventEmitter` protocol
- Token-by-token response streaming in chat UI
- Live step progress during RLM execution
- `execute_async()` on all use cases with event emitter support

## [0.2.0-alpha.2] - 2026-02-10

### Added
- Next.js 16 frontend with 4 pages: Chat, Dashboard, Traces, Settings
- 38 custom React components built on shadcn/ui and Radix UI
- Dark/light theme with system preference detection via next-themes
- SWR-based data fetching with real-time dashboard refresh
- WebSocket reconnection with exponential backoff

### Fixed
- 8 of 9 server code review findings resolved (error format, validation, race conditions)

## [0.2.0-alpha.1] - 2026-02-10

### Added
- FastAPI backend with 12 REST + 1 WebSocket endpoint
- 30+ Pydantic v2 request/response models
- AppState singleton with dependency injection
- Docker Sandbox Adapter implementing `SandboxPort`
- Async port methods: `complete_async()`, `complete_stream_async()`, `execute_async()`
- Two-model switching in `RunRLMUseCase` (root + recursive model)

## [0.1.0] - 2026-02-09

Initial release after Cycle 1: Foundation & Core Alignment.

### Added
- Clean Architecture: domain, application, infrastructure, and public API layers
- `interact()` and `complete()` as main programmatic entry points
- Three execution modes: Direct, RAG, RLM with Auto selection
- LiteLLM integration for 100+ LLM providers through a single adapter
- Two-model cost optimization (root model + recursive model)
- Deep recursion support with configurable `max_depth`
- RestrictedPython sandbox for safe code execution
- Budget controls: token limits, cost caps, step limits, timeouts
- Execution tracing with per-step metrics
- `RLMKitClient` public client with backward compatibility
- 814+ test functions across unit, integration, and E2E suites
- CI/CD pipeline with GitHub Actions (lint, typecheck, test, security)
- Pre-commit hooks (ruff, trailing whitespace, YAML/JSON validation)
