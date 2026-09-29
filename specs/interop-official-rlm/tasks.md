---
feature_id: interop-official-rlm
spec: ./spec.md
plan: ./plan.md
status: draft
date: 2026-08-15
---

# Tasks — Official-engine interop

Branch `feat/interop-official-rlm` off `master` after `feat/rebrand-rlm-studio` merges. One commit per phase; CI green each time. Allowlist in `plan.md` §2 is binding. Apply defaults for OQ-1..3 in `spec.md` §6.

## Phase 0 — confirmations (no edits)
- [x] T0.1 `pip index versions rlms` — record latest 0.1.x and its Python floor; read `rlm.RLM` signature and logger/trajectory API for the pinned version. *(2026-09-29: `rlms==0.1.3`, `>=3.11`; signatures, `RLMLogger.get_trajectory()`, `UsageSummary`, exported exceptions and the `[studio]+rlms` resolution check recorded in `doc_internal/plans/2026-09-29-release-1.0.0-execution-plan.md` §M1.)*
- [x] T0.2 Locate where mode literals/constants live today (`grep -rn 'Literal\["auto"' src`; `SlotMode`); decide the single constants module. *(2026-09-29: `application/sandbox_vars.py:57-61` is the home — `MODE_RLM_OFFICIAL` goes there; no new `application/constants.py`. Other literal sites: `run_matrix_comparison.py:32/48`, `server/models.py:219/638/682/711`, `frontend/src/lib/api.ts:51/377/392/405`.)*

## Phase 1 — constants + literals
- [x] T1.1 `MODE_RLM_OFFICIAL` constant; widen `SlotMode`, `server/models.py` literals; frontend type mirror. *(2026-09-29: constant + `ExecutionMode`/`ChatMode` `Literal` aliases in `sandbox_vars.py`; `SlotMode` in the use case **and** the duplicate in `routes/compare_matrix.py:58` both re-export the alias; matrix `modes` `max_length` 3→4; frontend `MODE_RLM_OFFICIAL`, `api.ts` unions, `MODE_DESCRIPTIONS` entry. `_SUPPORTED_MODES` widens in Phase 4 together with dispatch so a slot is never silently accepted-then-failed.)*
- [x] T1.2 Tests: request validation accepts the mode; `auto` never routes to it. *(`tests/test_mode_rlm_official.py`, 14 tests; `test_all_execution_modes_accepted` extended.)*

## Phase 2 — port + fake + use case
- [x] T2.1 `application/ports/rlm_engine_port.py` (`RLMEnginePort`). *(2026-09-29: `version` property + `is_available()` + `run`/`run_async`; exported from `application.ports`; compliance test in `tests/test_port_compliance.py`.)*
- [x] T2.2 `tests/fakes/fake_rlm_engine.py` (scripted trajectories). *(`tests/` is a package, so `tests/fakes/` is importable; the fake sleeps with `asyncio.sleep` on the async path so `wait_for` cancels cleanly.)*
- [x] T2.3 `application/use_cases/run_rlm_official.py`; tests: happy, error, timeout, budget breach → correct outcome categories. *(Wall-clock backstop = daemon thread joined with a timeout, not a ThreadPoolExecutor, so an engine stuck in a REPL can neither block the caller nor interpreter exit. Post-run token/cost breach → degraded `⚠️ … budget …` answer (success=True, classifier → BUDGET_EXHAUSTED), mirroring `RunRLMUseCase`'s limit warning. `tests/test_run_rlm_official.py`, 16 tests, all asserting the classifier category.)*

## Phase 3 — adapter
- [x] T3.1 `infrastructure/engines/rlms_adapter.py`: lazy import, `is_available()`, client factory (native openai/anthropic; openai-compatible base_url; unsupported → clear error), run, trajectory→`TraceStep` mapping, usage→tokens/cost. *(2026-09-29: `RlmsEngineAdapter`; `prompt=content, root_prompt=query`; budgets → `max_iterations/max_depth/max_timeout/max_tokens/max_budget`; ollama/lmstudio/vllm → rlms `openai` backend with `/v1`-normalised `base_url`; `docker` sandbox → rlms docker env, else `local` + `NOTE_LOCAL_ENV`; cost = engine-reported → slot `cost_fn` → `cost_known=False` (OQ-2); rlms `Budget/TokenLimit/Timeout` exceptions → classifier-keyword errors. Deviation: the adapter **cannot** tear down a stuck REPL — rlms' local env `exec()`s in-process inside `_spawn_completion_context`, unreachable from outside — so AC-3 rests on the use case's wall-clock guard; recorded in `engine_notes`.)*
- [x] T3.2 Recorded trajectory fixture + pure mapping tests. *(`tests/test_rlms_adapter.py`: fixture mirrors `RLMLogger.get_trajectory()`/`to_dict()` for 0.1.3 incl. a nested `rlm_calls` entry; `run()` exercised against a fake `rlm` module in `sys.modules` — 35 tests, no network, no extra needed.)*
- [x] T3.3 `--runslow` test on 3.11 against an OpenAI-compatible stub; AC-3 timeout test. *(`tests/test_rlms_engine_real.py`: stdlib `ThreadingHTTPServer` stub speaking `chat/completions`; real `rlms==0.1.3` run → 2 root calls, `inspect`×≥1 + `final`×1 via `_canonical_action_type`; AC-3 uses `time.sleep(30)` in the REPL instead of `while True: pass` so the abandoned in-process thread doesn't hold the GIL for the rest of the session — same mechanism. Verified with `uv run --with rlms==0.1.3 pytest … --runslow` (2 passed, 6.6 s); skips when `rlm` is absent. CI wiring of the extra is Phase 5.)*

## Phase 4 — matrix + routes + wiring
- [x] T4.1 `_execute_slot` dispatch + slot validation + `_copy_config_for_slot`. *(2026-09-29: `MatrixSlotDTO.engine`, `_SUPPORTED_MODES` now built from the `MODE_*` constants, "requires an engine adapter" validation, dispatch to `RunRLMOfficialUseCase`; `_copy_config_for_slot` needed no change. OQ-2: `_rank` sorts `cost_known=False` slots last for `cost` and `answer_per_cost`. The public client `api.compare_matrix()` keeps rejecting `rlm_official` — it has no provider config to build an engine from; server/UI only in 1.0, documented in Phase 7.)*
- [x] T4.2 `server/routes/engines.py` (`GET /api/engines`), register in `app.py`; chat/compare 400 path; `get_rlm_engine()` in `dependencies.py`. *(`EngineStatus`/`EnginesResponse`; `AppState.rlm_engine_availability()`, `create_rlm_engine_for_chat_provider(cp_id, llm)`, `create_rlm_engine(llm)`; Chat Provider backend resolution extracted into `_resolve_backend_for_chat_provider()` and the active-provider one into `_resolve_active_backend()` so the LiteLLM adapter and the engine can't drift. Both compare slot sites build the engine next to the LLM adapter; `_build_slot_run_config` applies the step/time knobs via `MODES_RLM_INTERNAL` (which now includes `rlm_official`) but the profile prompt only to Studio's loop. Chat REST + WS dispatch (`ENGINE_UNAVAILABLE` error frame on WS); `_prepare_history_context` short-circuits — the engine has no history channel. Availability checks fail fast before any session/telemetry side effect.)*
- [x] T4.3 e2e tests in `tests/e2e/test_api_endpoints.py`. *(Placed in `tests/test_engines_route.py` next to `test_compare_matrix_route.py` (same fake-injection convention) rather than the e2e file: `/api/engines` ×3, compare-matrix ×3 (official slot next to a direct slot, 400 + no side effects when unavailable, availability not consulted for built-in modes), chat ×3 (400, CP mode honoured, 202 when available). Matrix use-case dispatch/ranking in `tests/test_run_matrix_comparison_official.py`.)*

## Phase 5 — packaging
- [ ] T5.1 `interop` extra with version marker; `all` includes it; `uv lock`; CI installs the extra on 3.11+ only.

## Phase 6 — frontend
- [ ] T6.1 Compare slot picker + Chat mode option; availability gating from `/api/engines`; engine badge in Traces/Compare.
- [ ] T6.2 vitest coverage for gating + badge.

## Phase 7 — telemetry + docs
- [ ] T7.1 Telemetry mode enum/labels; Dashboard grouping check.
- [ ] T7.2 Docs: studio guide, concepts, hosts provider-mapping matrix, README table row; CHANGELOG.

## Phase 8 — acceptance
- [ ] T8.1 Run AC-1..AC-5 from `spec.md`; record results in `doc_internal/v1.0.0-rlm-studio/MANUAL_TEST_PLAN.md`.
