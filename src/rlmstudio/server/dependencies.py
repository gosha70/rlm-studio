"""FastAPI dependency injection: adapters, use cases, and shared state."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

if TYPE_CHECKING:
    from rlmstudio.application.services.provider_tester import ProviderTestResult

from rlmstudio.application.dto import RunConfigDTO
from rlmstudio.application.ports.llm_port import LLMPort
from rlmstudio.application.ports.rlm_engine_port import RLMEnginePort
from rlmstudio.application.sandbox_vars import MODE_AUTO, MODE_DIRECT, MODE_RAG, MODE_RLM
from rlmstudio.branding import env, env_name, state_dir
from rlmstudio.infrastructure.engines.rlms_adapter import RlmsEngineAdapter, rlms_availability
from rlmstudio.infrastructure.llm.litellm_adapter import LiteLLMAdapter
from rlmstudio.infrastructure.sandbox.sandbox_factory import create_sandbox
from rlmstudio.server.models import (
    BudgetConfig,
    ChatProviderConfig,
    ConfigResponse,
    LLMProviderConfig,
    RunProfile,
    RuntimeSettings,
    SystemPrompts,
)
from rlmstudio.telemetry.store import TelemetryStore
from rlmstudio.ui.data.providers_catalog import PROVIDERS_BY_KEY
from rlmstudio.ui.services.secret_store import FileSecretStore, KeyringSecretStore

logger = logging.getLogger(__name__)


def _get_instance_api_key(llm_provider_id: str, backend: str) -> str | None:
    """Return the stored API key for a named LLM Provider, falling back to env var."""
    key_name = f"llm_provider:{llm_provider_id}"
    store: KeyringSecretStore | FileSecretStore = (
        KeyringSecretStore() if KeyringSecretStore.is_available() else FileSecretStore()
    )
    raw_key: str | None = store.get(key_name)
    if raw_key:
        return raw_key
    entry = PROVIDERS_BY_KEY.get(backend)
    if entry and entry.env_var:
        return os.environ.get(entry.env_var)
    return None


# Providers that default to zero retries (local servers; retries multiply hangs)
_LOCAL_PROVIDERS: frozenset[str] = frozenset({"ollama", "lmstudio", "vllm"})


class _ResolvedBackend(NamedTuple):
    """Where a Chat Provider's requests actually go: backend key, model, endpoint, key."""

    provider_key: str
    model: str
    api_base: str | None
    api_key: str | None
    llm_provider: LLMProviderConfig | None


# Config lives in the product state dir (see rlmstudio.branding.state_dir) so
# the path is stable regardless of CWD.  Migrate from legacy
# .rlmkit_config.json in CWD on first run if present.
_STATE_DIR = state_dir()
_CONFIG_FILE = _STATE_DIR / "config.json"
_SESSIONS_FILE = _STATE_DIR / "sessions.json"
_EVALUATIONS_FILE = _STATE_DIR / "evaluations.json"
_FILES_FILE = _STATE_DIR / "files.json"
_LEGACY_CONFIG_FILE = Path(".rlmkit_config.json")
_LEGACY_SESSIONS_FILE = Path(".rlmkit_sessions.json")
_LEGACY_EVALUATIONS_FILE = Path(".rlmkit_evaluations.json")
_MAX_PERSISTED_SESSIONS = 50
_MAX_PERSISTED_FILES = 100

# ---------------------------------------------------------------------------
# In-memory stores (replaced by real persistence in production)
# ---------------------------------------------------------------------------


@dataclass
class FileRecord:
    id: str
    name: str
    size_bytes: int
    content_type: str
    text_content: str
    token_count: int
    created_at: datetime


@dataclass
class SessionRecord:
    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    messages: list[dict[str, Any]] = field(default_factory=list)
    conversations: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict
    )  # keyed by chat_provider_id


@dataclass
class ExecutionRecord:
    execution_id: str
    session_id: str
    query: str
    mode: str
    status: str = "running"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    chat_provider_id: str | None = None
    chat_provider_name: str | None = None


class AppState:
    """Shared application state holding in-memory stores and configuration."""

    def __init__(self, *, load_from_disk: bool = True) -> None:
        self.start_time = time.time()
        self.config = ConfigResponse()
        self.files: dict[str, FileRecord] = {}
        self.sessions: dict[str, SessionRecord] = {}
        self.executions: dict[str, ExecutionRecord] = {}
        self.user_profiles: list[RunProfile] = []
        self.system_prompts = SystemPrompts()
        self.evaluations: dict[str, list[dict[str, Any]]] = {
            "thumb_ratings": [],
            "best_picks": [],
            "judge_scores": [],
            "judge_pairwise": [],
        }
        # Reentrant lock protecting config.llm_providers reads/writes and all
        # save_config() calls.  Acquired by CRUD route handlers, the manual
        # test route, the PUT /api/config handler, and (in a later commit)
        # the background connection-test thread.  See doc_internal/specs/
        # scheduled-connection-testing.md §Concurrency model.  Reentrant
        # because some paths acquire it and then call helpers that also
        # acquire it.  Lock-ordering discipline: this is the only
        # application-level lock; never hold it while trying to acquire
        # another application-level lock.
        self._config_lock: threading.RLock = threading.RLock()
        # Scheduled-connection-testing thread state.  Always initialized to
        # None/unset — _start_connection_testing() spawns the thread lazily
        # based on config.connection_test_interval_minutes.
        self._connection_test_thread: threading.Thread | None = None
        self._connection_test_stop: threading.Event = threading.Event()
        # Persistent telemetry store (SQLite-backed).
        # In-memory ":memory:" for tests, on-disk for production.
        self.telemetry: TelemetryStore = TelemetryStore(":memory:" if not load_from_disk else None)
        if load_from_disk:
            self._load_config()
            self._load_sessions()
            self._load_files()
            self._load_evaluations()
            self._migrate_chat_providers()
            self._assign_default_profiles()
        # MUST be the last line of __init__ (see spec §Background thread).
        # Any later initialization would race with the first cycle's reads
        # of self.config.  If a future refactor needs post-startup work,
        # factor it into _post_init_warmup() and call that BEFORE this.
        self._start_connection_testing()

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        """Load persisted config from disk if available.

        Checks <state dir>/config.json first.  If absent, migrates from the
        legacy CWD-relative .rlmkit_config.json (created by older versions).
        """
        config_path = _CONFIG_FILE
        if not config_path.exists() and _LEGACY_CONFIG_FILE.exists():
            # Migrate: copy legacy file to the new home-dir location and remove it
            try:
                _STATE_DIR.mkdir(parents=True, exist_ok=True)
                config_path.write_text(_LEGACY_CONFIG_FILE.read_text())
                _LEGACY_CONFIG_FILE.unlink()
                logger.info("Migrated config from %s to %s", _LEGACY_CONFIG_FILE, config_path)
            except Exception as exc:
                logger.warning("Migration failed, falling back to legacy path: %s", exc)
                config_path = _LEGACY_CONFIG_FILE

        if not config_path.exists():
            return
        try:
            raw = json.loads(config_path.read_text())
            self.config = ConfigResponse.model_validate(raw.get("config", {}))
            self.system_prompts = SystemPrompts.model_validate(raw.get("system_prompts", {}))
            self.user_profiles = [
                RunProfile.model_validate(p) for p in raw.get("user_profiles", [])
            ]
            self._reconcile_active_provider()
            logger.info(
                "Loaded config: provider=%s model=%s (%d provider configs)",
                self.config.active_provider,
                self.config.active_model,
                len(self.config.provider_configs),
            )
        except Exception as exc:
            logger.warning("Failed to load config: %s", exc)

    def _reconcile_active_provider(self) -> None:
        """Ensure active_provider has an enabled entry in provider_configs.

        Fixes corrupted state from the old dual-save-path bug where
        active_provider/active_model could drift from provider_configs.
        """
        from rlmstudio.server.models import ProviderConfig

        active = self.config.active_provider
        configs = self.config.provider_configs

        # Find the active provider's entry
        active_entry: ProviderConfig | None = None
        for pc in configs:
            if pc.provider == active:
                active_entry = pc
                break

        changed = False

        # If the active provider has no entry, create one
        if active_entry is None:
            catalog_entry = PROVIDERS_BY_KEY.get(active)
            default_model = self.config.active_model
            if not default_model and catalog_entry and catalog_entry.models:
                default_model = catalog_entry.models[0].name
            active_entry = ProviderConfig(
                provider=active,
                model=default_model or "",
                enabled=True,
            )
            configs.append(active_entry)
            changed = True

        # Ensure the active provider's entry is enabled
        if not active_entry.enabled:
            active_entry.enabled = True
            changed = True

        # Ensure active_model matches the entry's model
        if active_entry.model and active_entry.model != self.config.active_model:
            self.config.active_model = active_entry.model
            changed = True

        # Disable all other providers (only one active at a time)
        for pc in configs:
            if pc.provider != active and pc.enabled:
                pc.enabled = False
                changed = True

        if changed:
            logger.info(
                "Reconciled config: set %s as active provider (model=%s)",
                active,
                self.config.active_model,
            )
            self.save_config()

    def _migrate_chat_providers(self) -> None:
        """Auto-create Chat Providers from existing provider_configs on first load.

        Also performs a second pass to replace stale model IDs that are no longer
        present in the hardcoded catalog with the provider's current default model.
        """
        if not self.config.chat_providers:
            now = datetime.now(timezone.utc)
            for pc in self.config.provider_configs:
                if not pc.enabled:
                    continue
                mode_label = "DIRECT"
                cp = ChatProviderConfig(
                    id=str(uuid.uuid4()),
                    name=f"{mode_label}-{pc.provider.upper()}",
                    llm_provider=pc.provider,
                    llm_model=pc.model,
                    execution_mode=MODE_DIRECT,
                    profile_id="builtin-accurate",
                    runtime_settings=pc.runtime_settings.model_copy(),
                    created_at=now,
                    updated_at=now,
                )
                self.config.chat_providers.append(cp)
            if self.config.chat_providers:
                logger.info(
                    "Migrated %d Chat Provider(s) from existing provider configs",
                    len(self.config.chat_providers),
                )
                self.save_config()

        self._migrate_to_llm_providers()

    def _migrate_to_llm_providers(self) -> None:
        """Create named LLM Provider instances from legacy ChatProvider configs.

        Old ChatProviderConfig has llm_provider (backend key) + llm_model.
        New design references an LLMProviderConfig by UUID.
        This migration creates the LLM Provider instances and updates the references.
        """
        # Check if any chat providers still use the old format (no llm_provider_id)
        legacy_cps = [cp for cp in self.config.chat_providers if not cp.llm_provider_id]
        if not legacy_cps:
            return

        from rlmstudio.ui.data.providers_catalog import PROVIDERS_BY_KEY

        now = datetime.now(timezone.utc)
        changed = False

        # Deduplicate: one LLMProviderConfig per unique (backend, model, endpoint)
        key_to_id: dict[tuple[str, str, str | None], str] = {}
        for lp in self.config.llm_providers:
            key_to_id[(lp.backend, lp.model, lp.endpoint)] = lp.id

        for cp in legacy_cps:
            backend = cp.llm_provider or "openai"
            model = cp.llm_model or ""
            # Get endpoint from matching provider_config
            endpoint: str | None = None
            for pc in self.config.provider_configs:
                if pc.provider == backend and pc.endpoint:
                    endpoint = pc.endpoint
                    break

            combo = (backend, model, endpoint)
            if combo not in key_to_id:
                # Create a new LLMProviderConfig
                catalog = PROVIDERS_BY_KEY.get(backend)
                display = catalog.display_name if catalog else backend.capitalize()
                lp_name = f"{display} ({model})" if model else display
                # Ensure name is unique
                existing_names = {lp.name.lower() for lp in self.config.llm_providers}
                base_name = lp_name
                suffix = 2
                while lp_name.lower() in existing_names:
                    lp_name = f"{base_name} {suffix}"
                    suffix += 1

                # Use the provider_config's runtime_settings if available
                rt = RuntimeSettings()
                for pc in self.config.provider_configs:
                    if pc.provider == backend:
                        rt = pc.runtime_settings.model_copy()
                        break

                lp = LLMProviderConfig(
                    id=str(uuid.uuid4()),
                    name=lp_name,
                    backend=backend,
                    model=model,
                    endpoint=endpoint,
                    runtime_settings=rt,
                    created_at=now,
                    updated_at=now,
                )
                self.config.llm_providers.append(lp)
                key_to_id[combo] = lp.id
                logger.info(
                    "Created LLM Provider '%s' (backend=%s model=%s)", lp.name, backend, model
                )

            cp.llm_provider_id = key_to_id[combo]
            changed = True

        if changed:
            logger.info(
                "Migrated %d Chat Provider(s) to reference named LLM Providers",
                len(legacy_cps),
            )
            self.save_config()

    def _assign_default_profiles(self) -> None:
        """Assign a default profile to Chat Providers that don't have one."""
        changed = False
        for cp in self.config.chat_providers:
            if cp.profile_id:
                continue
            if cp.execution_mode == MODE_RLM:
                cp.profile_id = "builtin-rlm-deep"
            elif cp.execution_mode == MODE_RAG:
                cp.profile_id = "builtin-rag"
            else:
                cp.profile_id = "builtin-accurate"
            changed = True
            logger.info(
                "Assigned profile %s to Chat Provider %s",
                cp.profile_id,
                cp.name,
            )
        if changed:
            self.save_config()

    # ------------------------------------------------------------------
    # Scheduled connection testing
    # See doc_internal/specs/scheduled-connection-testing.md for design.
    # ------------------------------------------------------------------

    # Per-provider test timeout used by the background cycle.  Manual route
    # uses 15s; background uses 10s because the cycle budget must bound
    # cleanly.  10 providers × 10s / 5 workers = ~20s worst-case cycle.
    _PROVIDER_TEST_TIMEOUT_S: float = 10.0
    # Connected → offline requires this many consecutive failures.
    _OFFLINE_FAILURE_THRESHOLD: int = 2
    # Parallelism inside a single cycle.
    _CYCLE_MAX_WORKERS: int = 5
    # Test-only override for the interval, in seconds (never documented to
    # users).  Set via monkeypatch.setenv in tests — see spec Open Q #3.
    _INTERVAL_OVERRIDE_ENV: str = env_name("CONNECTION_TEST_INTERVAL_SECONDS_OVERRIDE")

    _connection_test_logger = logging.getLogger("rlmstudio.server.connection_test_thread")

    def _resolve_interval_seconds(self) -> float | None:
        """Return the interval-to-sleep, or None if testing is disabled.

        Respects the test-only env var override (in seconds) so integration
        tests can run cycles at sub-minute frequency without changing the
        user-facing minutes-based knob.
        """
        override = env("CONNECTION_TEST_INTERVAL_SECONDS_OVERRIDE")
        if override:
            try:
                val = float(override)
                if val > 0:
                    return val
            except ValueError:
                self._connection_test_logger.warning(
                    "invalid %s=%r — ignoring override", self._INTERVAL_OVERRIDE_ENV, override
                )
        minutes = self.config.connection_test_interval_minutes
        if minutes <= 0:
            return None
        return float(minutes * 60)

    def _start_connection_testing(self) -> None:
        """Spawn the background cycle thread if configured.

        Idempotent: if a thread is already running, this is a no-op.
        If connection_test_interval_minutes == 0 (and no override), no
        thread is spawned.
        """
        if self._connection_test_thread is not None and self._connection_test_thread.is_alive():
            return
        interval_s = self._resolve_interval_seconds()
        if interval_s is None:
            return
        # Fresh stop event per thread lifecycle.
        self._connection_test_stop = threading.Event()
        self._connection_test_thread = threading.Thread(
            target=self._connection_test_loop,
            name="rlmkit-connection-test",
            daemon=True,
        )
        self._connection_test_thread.start()
        self._connection_test_logger.info(
            "connection test thread started, interval_s=%.1f", interval_s
        )

    def _stop_connection_testing(self) -> None:
        """Signal the thread to stop and join with a bounded timeout.

        In-flight cycle results are discarded (the loop exits before the
        "apply results" step).  This is intentional: stopping means the
        user changed the interval or is shutting down, so this cycle's
        observations are no longer authoritative.
        """
        thread = self._connection_test_thread
        if thread is None:
            return
        self._connection_test_stop.set()
        thread.join(timeout=2.0)
        if thread.is_alive():
            self._connection_test_logger.warning(
                "connection test thread did not exit within 2s; leaving it daemon"
            )
        self._connection_test_thread = None

    def restart_connection_testing(self) -> None:
        """Stop any running thread and start one with the current config.

        Called by the PUT /api/config handler when
        connection_test_interval_minutes changes.  Only that specific field
        triggers a restart — other config changes do not churn the thread.
        """
        self._stop_connection_testing()
        self._start_connection_testing()

    def _connection_test_loop(self) -> None:
        """Background cycle loop.  Sleeps first, then tests.

        Sleeps before the first cycle so a --reload restart in dev does not
        hammer every provider on every code edit.  Uses the stop event's
        wait() as the sleep primitive so interval changes take effect
        immediately instead of waiting for the current sleep.
        """
        log = self._connection_test_logger
        while not self._connection_test_stop.is_set():
            interval_s = self._resolve_interval_seconds()
            if interval_s is None:
                log.info("interval resolved to None mid-loop; exiting")
                return
            if self._connection_test_stop.wait(timeout=interval_s):
                break  # stop requested during sleep
            cycle_started = time.monotonic()
            try:
                self._run_test_cycle()
            except Exception:  # noqa: BLE001 — cycle must not kill the thread
                log.exception("unhandled exception in connection test cycle")
            cycle_duration = time.monotonic() - cycle_started
            if cycle_duration > interval_s:
                log.warning(
                    "connection test cycle took %.1fs > interval %.1fs; "
                    "skipping next cycle to catch up",
                    cycle_duration,
                    interval_s,
                )
        log.info("connection test thread exiting")

    def _provider_fingerprint(self, lp: LLMProviderConfig) -> str:
        """Fingerprint of the test-relevant fields of a provider.

        Used by the stale-result guard: if a provider's fingerprint at
        snapshot time does not match the fingerprint at apply time, the
        user edited the provider mid-cycle and our test result no longer
        reflects what they configured.  We discard it rather than vouch
        for an endpoint we never probed.

        Fields included MUST match every field that test_provider reads.
        The api_key is identified by reference (id), not value — rotating
        the key is indistinguishable from setting one for the first time
        via the usual (SecretStore) path, so fingerprinting includes the
        provider id (which is already present as the map key) and the
        backend/model/endpoint tuple.
        """
        import hashlib

        fields = (lp.id, lp.backend, lp.model, lp.endpoint or "")
        tuple_repr = repr(fields)
        return hashlib.sha1(tuple_repr.encode("utf-8"), usedforsecurity=False).hexdigest()

    def _safe_test(self, lp: LLMProviderConfig) -> ProviderTestResult:
        """Call test_provider and catch any unhandled exception.

        A single provider raising inside test_provider must not kill the
        cycle for the other providers.  test_provider itself promises not
        to raise, but we add this belt-and-braces so a future bug in the
        probe does not take down the thread.
        """
        from rlmstudio.application.services.provider_tester import (
            ProviderTestResult,
            test_provider,
        )

        try:
            return test_provider(lp, timeout_s=self._PROVIDER_TEST_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001
            self._connection_test_logger.exception(
                "test_provider raised unexpectedly for provider_id=%s", lp.id
            )
            return ProviderTestResult(
                status="error",
                tested_at=datetime.now(timezone.utc),
                latency_ms=None,
                error_message=f"internal error: {exc!s}"[:200],
            )

    def _run_test_cycle(self) -> None:
        """One cycle: snapshot → probe in parallel → apply under lock.

        Lock discipline: held only during the snapshot and the batch-apply.
        The actual network probes run in the thread pool OUTSIDE the lock
        so CRUD handlers can still mutate providers while a cycle is in
        flight.  Typical critical-section duration: milliseconds, vs
        cycle duration of ~20s with 10 providers.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        log = self._connection_test_logger

        # Step 1: snapshot under lock
        with self._config_lock:
            provider_snapshot: dict[str, LLMProviderConfig] = {
                lp.id: lp.model_copy(deep=True) for lp in self.config.llm_providers
            }
            provider_count = len(provider_snapshot)

        log.info("cycle started, provider_count=%d", provider_count)
        if not provider_snapshot:
            log.info("cycle complete, nothing to test")
            return

        # Step 2: probe in parallel OUTSIDE the lock
        results: dict[str, ProviderTestResult] = {}
        with ThreadPoolExecutor(
            max_workers=self._CYCLE_MAX_WORKERS, thread_name_prefix="rlmkit-probe"
        ) as pool:
            futures = {
                pool.submit(self._safe_test, provider_snapshot[pid]): pid
                for pid in provider_snapshot
            }
            for future in as_completed(futures, timeout=None):
                pid = futures[future]
                try:
                    results[pid] = future.result()
                except Exception:  # noqa: BLE001
                    log.exception("future.result() failed for provider_id=%s", pid)

        # Before applying, check if stop was signaled during the probe.
        # The spec requires that in-flight cycle results be discarded on
        # stop — persisting them would vouch for connectivity based on a
        # cycle the operator explicitly abandoned (interval change or
        # shutdown).
        if self._connection_test_stop.is_set():
            log.info(
                "stop requested during cycle; discarding %d probe result(s) without persisting",
                len(results),
            )
            return

        # Step 3: apply results under lock
        with self._config_lock:
            transitions, discarded = self._apply_test_results(
                results, snapshot=provider_snapshot, source="background"
            )
            self.save_config()

        # Log transitions AFTER releasing the lock
        tested = len(results)
        failed = sum(1 for r in results.values() if r.status == "offline")
        errored = sum(1 for r in results.values() if r.status == "error")
        log.info(
            "cycle complete, tested=%d failed=%d errored=%d transitions=%d discarded_stale=%d",
            tested,
            failed,
            errored,
            len(transitions),
            len(discarded),
        )
        for t in transitions:
            log.info(
                "provider %s transitioned %s → %s after %d consecutive failures (tested_by=%s)",
                t["provider_id"],
                t["old_status"],
                t["new_status"],
                t["consecutive_failures"],
                t["tested_by"],
            )
        for d in discarded:
            log.info(
                "discarded stale test result for %s: config changed during cycle "
                "(fingerprint %s → %s)",
                d["provider_id"],
                d["snapshot_fp"],
                d["current_fp"],
            )

    def _apply_test_results(
        self,
        results: dict[str, ProviderTestResult],
        *,
        snapshot: dict[str, LLMProviderConfig],
        source: Literal["background", "manual"],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Merge cycle results back into self.config.llm_providers.

        Returns (transitions, discarded) for logging AFTER the lock is
        released.  MUST be called with ``_config_lock`` held.

        Stale-result guard: if a provider's fingerprint changed between
        snapshot and apply (user edited base_url / api_key / model), the
        result is discarded without mutating consecutive_failures or any
        other field.  The next cycle will probe the new config.
        """
        transitions: list[dict[str, Any]] = []
        discarded: list[dict[str, Any]] = []
        current_by_id = {lp.id: lp for lp in self.config.llm_providers}

        for pid, result in results.items():
            current = current_by_id.get(pid)
            if current is None:
                # Deleted mid-cycle.  Treat as a discard with no log noise
                # beyond the summary counter (logs are per-provider noisy).
                discarded.append(
                    {
                        "provider_id": pid,
                        "snapshot_fp": "deleted",
                        "current_fp": "deleted",
                    }
                )
                continue
            snap = snapshot.get(pid)
            if snap is None:
                # Added mid-cycle but somehow showed up in results?  Skip.
                continue
            snap_fp = self._provider_fingerprint(snap)
            curr_fp = self._provider_fingerprint(current)
            if snap_fp != curr_fp:
                discarded.append(
                    {
                        "provider_id": pid,
                        "snapshot_fp": snap_fp[:8],
                        "current_fp": curr_fp[:8],
                    }
                )
                continue

            old_status = current.status
            current.last_tested_at = result.tested_at
            current.last_tested_by = source

            if result.status == "connected":
                current.status = "connected"
                current.consecutive_failures = 0
            elif result.status == "error":
                # Error is treated as a failure for the counter, but the
                # status flips immediately to make the UI actionable.
                current.consecutive_failures += 1
                current.status = "error"
            else:  # offline (graceful failure)
                current.consecutive_failures += 1
                if current.consecutive_failures >= self._OFFLINE_FAILURE_THRESHOLD:
                    current.status = "offline"
                # else: stay at old status (likely "connected"); surface
                # the brewing failure via consecutive_failures counter in UI.

            if current.status != old_status:
                transitions.append(
                    {
                        "provider_id": pid,
                        "old_status": old_status,
                        "new_status": current.status,
                        "consecutive_failures": current.consecutive_failures,
                        "tested_by": source,
                    }
                )
        return transitions, discarded

    # ------------------------------------------------------------------
    # Chat Provider helpers
    # ------------------------------------------------------------------

    def get_chat_provider(self, chat_provider_id: str) -> ChatProviderConfig | None:
        """Look up a Chat Provider by ID."""
        for cp in self.config.chat_providers:
            if cp.id == chat_provider_id:
                return cp
        return None

    def get_llm_provider(self, llm_provider_id: str) -> LLMProviderConfig | None:
        """Look up a named LLM Provider instance by ID."""
        for lp in self.config.llm_providers:
            if lp.id == llm_provider_id:
                return lp
        return None

    def get_llm_provider_by_name(self, name: str) -> LLMProviderConfig | None:
        """Look up a named LLM Provider instance by display name (case-insensitive)."""
        for lp in self.config.llm_providers:
            if lp.name.lower() == name.lower():
                return lp
        return None

    def get_conversation(self, session_id: str, chat_provider_id: str) -> list[dict[str, Any]]:
        """Get conversation history for a specific Chat Provider in a session."""
        session = self.sessions.get(session_id)
        if not session:
            return []
        return session.conversations.get(chat_provider_id, [])

    def find_profile(self, profile_id: str) -> RunProfile | None:
        """Look up a profile by ID (builtins + user-created)."""
        from rlmstudio.server.routes.profiles import BUILTIN_PROFILES

        for p in BUILTIN_PROFILES:
            if p.id == profile_id:
                return p
        for p in self.user_profiles:
            if p.id == profile_id:
                return p
        return None

    def resolve_chat_provider(self, cp: ChatProviderConfig) -> ChatProviderConfig:
        """Return a copy of cp with settings resolved from its LLM Provider and profile.

        When the linked LLM Provider has a ``context_window``, the resolved
        ``runtime_settings.max_output_tokens`` is clamped so that at least
        25 % of the context window remains for input.  This prevents vLLM
        from rejecting requests when a generous Profile max_tokens is paired
        with a small-context model.
        """
        resolved = cp.model_copy()

        # Resolve LLM Provider name for display
        lp = None
        if cp.llm_provider_id:
            lp = self.get_llm_provider(cp.llm_provider_id)
            if lp:
                resolved.llm_provider_name = lp.name

        if not cp.profile_id:
            return resolved
        profile = self.find_profile(cp.profile_id)
        if not profile:
            logger.warning("Profile %s not found for Chat Provider %s", cp.profile_id, cp.id)
            return resolved
        resolved.profile_name = profile.name
        resolved.execution_mode = profile.strategy  # type: ignore[assignment]
        resolved.runtime_settings = (
            profile.runtime_settings.model_copy()
            if hasattr(profile.runtime_settings, "model_copy")
            else profile.runtime_settings
        )
        budget = (
            profile.budget
            if isinstance(profile.budget, BudgetConfig)
            else BudgetConfig.model_validate(profile.budget)
        )
        resolved.rlm_max_steps = budget.max_steps
        resolved.rlm_timeout_seconds = budget.max_time_seconds
        if hasattr(budget, "repeat_limit"):
            resolved.rlm_repeat_limit = budget.repeat_limit
        if hasattr(budget, "nudge_at_fraction"):
            resolved.rlm_nudge_at_fraction = budget.nudge_at_fraction

        # Clamp max_output_tokens against the LLM Provider's context window.
        # Reserve at least 25% of context for the input prompt so the RLM loop
        # has room for system prompt + conversation history.
        if lp and lp.context_window and lp.context_window > 0:
            max_safe_output = int(lp.context_window * 0.75)
            profile_max = resolved.runtime_settings.max_output_tokens
            if profile_max > max_safe_output:
                logger.info(
                    "Clamping max_output_tokens %d → %d for provider %s "
                    "(context_window=%d, 25%% reserved for input)",
                    profile_max,
                    max_safe_output,
                    lp.name,
                    lp.context_window,
                )
                resolved.runtime_settings.max_output_tokens = max_safe_output

        return resolved

    def resolve_execution_context(self, execution_id: str) -> tuple[str, str, str, str] | None:
        """Resolve (session_id, query, response_content, chat_provider_id) for an execution.

        First checks in-memory executions (live data), then falls back to
        persisted session messages (survives restarts).  Returns None if
        the execution cannot be found anywhere.
        """
        rec = self.executions.get(execution_id)
        if rec:
            # Live execution record exists — get response from session messages
            session = self.sessions.get(rec.session_id)
            response_content = ""
            if session:
                for msg in session.messages:
                    if msg.get("execution_id") == execution_id and msg.get("role") == "assistant":
                        response_content = msg.get("content", "")
                        break
            if not response_content and rec.result:
                response_content = rec.result.get("answer", "")
            return rec.session_id, rec.query, response_content, rec.chat_provider_id or ""

        # Fallback: scan persisted session messages
        for session in self.sessions.values():
            user_query = ""
            for msg in session.messages:
                if msg.get("role") == "user":
                    user_query = msg.get("content", "")
                elif msg.get("execution_id") == execution_id and msg.get("role") == "assistant":
                    return (
                        session.id,
                        user_query,
                        msg.get("content", ""),
                        msg.get("chat_provider_id", ""),
                    )
        return None

    def get_execution_ids_for_turn(self, session_id: str, execution_id: str) -> set[str]:
        """Find all execution_ids that belong to the same turn as the given execution_id.

        A 'turn' is defined by the user message that immediately precedes
        the assistant messages.  Works from persisted session messages so it
        survives restarts.
        """
        session = self.sessions.get(session_id)
        if not session:
            return set()

        # Build a mapping: for each assistant message, record its execution_id
        # grouped by the preceding user message id.
        current_user_msg_id: str | None = None
        turn_groups: dict[str, set[str]] = {}  # user_msg_id -> set of exec_ids
        exec_to_user: dict[str, str] = {}  # exec_id -> user_msg_id

        for msg in session.messages:
            if msg.get("role") == "user":
                current_user_msg_id = msg.get("id", "")
                if current_user_msg_id not in turn_groups:
                    turn_groups[current_user_msg_id] = set()
            elif msg.get("role") == "assistant" and current_user_msg_id:
                eid = msg.get("execution_id")
                if eid:
                    turn_groups[current_user_msg_id].add(eid)
                    exec_to_user[eid] = current_user_msg_id

        # Find which turn group contains our execution_id
        user_msg_id = exec_to_user.get(execution_id)
        if user_msg_id:
            return turn_groups.get(user_msg_id, set())

        # Fallback for in-memory-only executions: use query matching
        rec = self.executions.get(execution_id)
        if rec:
            return {
                eid
                for eid, r in self.executions.items()
                if r.session_id == session_id and r.query == rec.query
            }

        return set()

    def add_message(
        self,
        session_id: str,
        message: dict[str, Any],
        chat_provider_id: str | None = None,
    ) -> None:
        """Add a message to a session, tracking it per Chat Provider."""
        session = self.sessions.get(session_id)
        if not session:
            return
        # Always add to legacy flat messages for backward compat
        session.messages.append(message)
        # Also add to per-Chat-Provider conversations
        if chat_provider_id:
            if chat_provider_id not in session.conversations:
                session.conversations[chat_provider_id] = []
            session.conversations[chat_provider_id].append(message)

    def _resolve_backend_for_chat_provider(self, cp: ChatProviderConfig) -> _ResolvedBackend:
        """Resolve a (resolved) Chat Provider to its backend key, model, endpoint and API key.

        Shared by the LiteLLM adapter factory and the official-engine factory
        so the two can never disagree about which provider a slot talks to.
        """
        # Resolve backend config through LLM Provider instance
        lp = self.get_llm_provider(cp.llm_provider_id) if cp.llm_provider_id else None
        api_key: str | None = None
        if lp:
            provider_key = lp.backend
            model = lp.model
            api_base = lp.endpoint
            api_key = _get_instance_api_key(lp.id, lp.backend)
            # Fall back to catalog default endpoint for local providers
            if not api_base:
                entry = PROVIDERS_BY_KEY.get(provider_key)
                if entry and entry.default_endpoint:
                    api_base = entry.default_endpoint
        else:
            # Fallback for unmigrated legacy chat providers
            provider_key = cp.llm_provider or "openai"
            model = cp.llm_model or ""
            entry = PROVIDERS_BY_KEY.get(provider_key)
            api_base = None
            for pc in self.config.provider_configs:
                if pc.provider == provider_key and pc.endpoint:
                    api_base = pc.endpoint
                    break
            if not api_base and entry and entry.default_endpoint:
                api_base = entry.default_endpoint
        return _ResolvedBackend(provider_key, model, api_base, api_key, lp)

    def create_rlm_engine_for_chat_provider(
        self,
        chat_provider_id: str,
        llm: LLMPort | None = None,
    ) -> RLMEnginePort:
        """Official-engine adapter for a Chat Provider's provider/model.

        ``llm`` is the slot's LiteLLM adapter; its cost table prices the run
        when the engine reports no price.
        """
        cp = self.get_chat_provider(chat_provider_id)
        if not cp:
            raise ValueError(f"Chat Provider {chat_provider_id} not found")
        backend = self._resolve_backend_for_chat_provider(self.resolve_chat_provider(cp))
        return self._build_rlm_engine(backend, llm)

    def create_rlm_engine(self, llm: LLMPort | None = None) -> RLMEnginePort:
        """Official-engine adapter for the globally active provider (no Chat Provider)."""
        provider_key, model, api_base, _runtime = self._resolve_active_backend()
        entry = PROVIDERS_BY_KEY.get(provider_key)
        api_key = os.environ.get(entry.env_var) if entry and entry.env_var else None
        return self._build_rlm_engine(
            _ResolvedBackend(provider_key, model, api_base, api_key, None), llm
        )

    def _build_rlm_engine(self, backend: _ResolvedBackend, llm: LLMPort | None) -> RLMEnginePort:
        cost_fn = getattr(llm, "get_completion_cost", None) if llm is not None else None
        return RlmsEngineAdapter(
            backend=backend.provider_key,
            model=backend.model,
            api_key=backend.api_key,
            base_url=backend.api_base,
            sandbox_type=self.config.sandbox.type,
            docker_image=self.config.sandbox.docker_image,
            cost_fn=cost_fn,
        )

    @staticmethod
    def rlm_engine_availability() -> tuple[bool, str, str | None]:
        """Package-level availability of the official engine: ``(available, reason, version)``."""
        availability: tuple[bool, str, str | None] = rlms_availability()
        return availability

    def create_llm_adapter_for_chat_provider(
        self,
        chat_provider_id: str,
        num_retries: int | None = None,
    ) -> LiteLLMAdapter:
        """Create an LLM adapter configured for a specific Chat Provider."""
        cp = self.get_chat_provider(chat_provider_id)
        if not cp:
            raise ValueError(f"Chat Provider {chat_provider_id} not found")

        cp = self.resolve_chat_provider(cp)

        provider_key, model, api_base, api_key, lp = self._resolve_backend_for_chat_provider(cp)

        prefixed_model = self._litellm_model_name(provider_key, model)
        # runtime_settings are resolved from the Profile by resolve_chat_provider()
        runtime = cp.runtime_settings

        effective_retries = (
            num_retries
            if num_retries is not None
            else cp.num_retries
            if cp.num_retries is not None
            else 0
            if provider_key in _LOCAL_PROVIDERS
            else 2
        )

        context_window = lp.context_window if lp else None

        return LiteLLMAdapter(
            model=prefixed_model,
            api_base=api_base,
            api_key=api_key,
            temperature=runtime.temperature,
            top_p=runtime.top_p,
            max_tokens=runtime.max_output_tokens,
            timeout=float(runtime.timeout_seconds),
            num_retries=effective_retries,
            context_window=context_window,
        )

    def save_config(self) -> None:
        """Persist current config, profiles, and prompts to disk atomically.

        Ephemeral Chat Providers (created by compare-matrix) are excluded
        from the persisted data — they are session-local and should not
        survive a server restart.

        Atomicity
        ---------
        Writes go to a temp file in the same directory as ``_CONFIG_FILE``,
        then ``os.replace`` promotes it to the target path.  ``os.replace``
        is atomic on POSIX and on Windows (Python 3.3+).  If the process
        dies between the temp-file write and the rename, the previous valid
        config stays on disk; there is no partial-file window.  The feature
        that motivates this change (scheduled connection testing) increases
        save frequency ~1500×/day with 10 providers at 1-min cycles, which
        makes even low-probability non-atomic writes a reliability risk.

        Thread safety
        -------------
        Acquires ``_config_lock`` around the snapshot + write so concurrent
        CRUD mutations and background-cycle applies cannot interleave into
        a corrupted serialization.  The lock is reentrant, so route
        handlers that already hold it (e.g. after a CRUD mutation) can
        still call save_config() without deadlocking.
        """
        with self._config_lock:
            self._save_config_locked()

    def _save_config_locked(self) -> None:
        """Do the actual write. MUST be called with ``_config_lock`` held."""
        try:
            _STATE_DIR.mkdir(parents=True, exist_ok=True)
            config_dump = self.config.model_dump()
            # Filter out ephemeral CPs from the persisted chat_providers list.
            config_dump["chat_providers"] = [
                cp for cp in config_dump.get("chat_providers", []) if not cp.get("ephemeral", False)
            ]
            data = {
                "config": config_dump,
                "system_prompts": self.system_prompts.model_dump(),
                "user_profiles": [p.model_dump() for p in self.user_profiles],
            }
            serialized = json.dumps(data, indent=2, default=str)

            # Write to temp file in the SAME directory as the target.  Cross-
            # filesystem os.replace is not atomic; same-dir is.  delete=False
            # because the "with" block's __exit__ would unlink the file
            # before os.replace could see it — we need it alive for the
            # rename, and on Windows it must be closed before the rename.
            target_dir = _CONFIG_FILE.parent
            tmp = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(target_dir),
                prefix=".config.",
                suffix=".tmp",
                delete=False,
            )
            tmp_path = tmp.name
            try:
                try:
                    tmp.write(serialized)
                    tmp.flush()
                    os.fsync(tmp.fileno())
                finally:
                    tmp.close()
            except BaseException:
                # Write-path failure (ENOSPC, EIO, quota exceeded, read-only
                # remount, SIGTERM between flush and fsync, ...).  The temp
                # file already exists on disk; unlink it before re-raising
                # so the outer except doesn't leave a .config.*.tmp behind.
                # BaseException on purpose: SystemExit / KeyboardInterrupt
                # can also interrupt the write path.
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            try:
                os.replace(tmp_path, _CONFIG_FILE)
            except OSError:
                # Rename failed — clean up the dangling temp file so we do
                # not leak one per failed save.
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            logger.info(
                "Saved config: provider=%s model=%s (%d provider configs)",
                self.config.active_provider,
                self.config.active_model,
                len(self.config.provider_configs),
            )
        except Exception as exc:
            logger.warning("Failed to save config: %s", exc)

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def _load_sessions(self) -> None:
        """Load persisted sessions from disk if available."""
        sessions_path = _SESSIONS_FILE
        if not sessions_path.exists() and _LEGACY_SESSIONS_FILE.exists():
            try:
                _STATE_DIR.mkdir(parents=True, exist_ok=True)
                sessions_path.write_text(_LEGACY_SESSIONS_FILE.read_text())
                _LEGACY_SESSIONS_FILE.unlink()
                logger.info("Migrated sessions from %s to %s", _LEGACY_SESSIONS_FILE, sessions_path)
            except Exception as exc:
                logger.warning("Sessions migration failed: %s", exc)
                sessions_path = _LEGACY_SESSIONS_FILE
        if not sessions_path.exists():
            return
        try:
            raw = json.loads(sessions_path.read_text())
            for s in raw:
                rec = SessionRecord(
                    id=s["id"],
                    name=s["name"],
                    created_at=datetime.fromisoformat(s["created_at"]),
                    updated_at=datetime.fromisoformat(s["updated_at"]),
                    messages=s.get("messages", []),
                    conversations=s.get("conversations", {}),
                )
                self.sessions[rec.id] = rec
            logger.info("Loaded %d sessions from disk", len(self.sessions))
        except Exception as exc:
            logger.warning("Failed to load sessions: %s", exc)

    def save_sessions(self) -> None:
        """Persist sessions to disk (most recent N only)."""
        try:
            _STATE_DIR.mkdir(parents=True, exist_ok=True)
            # Sort by updated_at descending and cap
            sorted_sessions = sorted(
                self.sessions.values(),
                key=lambda s: s.updated_at,
                reverse=True,
            )[:_MAX_PERSISTED_SESSIONS]
            data = [
                {
                    "id": s.id,
                    "name": s.name,
                    "created_at": s.created_at.isoformat(),
                    "updated_at": s.updated_at.isoformat(),
                    "messages": s.messages,
                    "conversations": s.conversations,
                }
                for s in sorted_sessions
            ]
            _SESSIONS_FILE.write_text(json.dumps(data, indent=2, default=str))
        except Exception as exc:
            logger.warning("Failed to save sessions: %s", exc)

    # ------------------------------------------------------------------
    # File persistence
    # ------------------------------------------------------------------

    def _load_files(self) -> None:
        """Load persisted uploaded files from disk if available.

        Files must survive server restarts so that a user who uploads a
        document, gets rescheduled (e.g. dev-server ``--reload``), and then
        continues chatting in a persisted session can still reference the
        file by id.  Without this, ``_resolve_file_content`` returns 404
        for every restored session that has ``file_ids`` in its messages.
        """
        if not _FILES_FILE.exists():
            return
        try:
            raw = json.loads(_FILES_FILE.read_text())
            for f in raw:
                rec = FileRecord(
                    id=f["id"],
                    name=f["name"],
                    size_bytes=f["size_bytes"],
                    content_type=f["content_type"],
                    text_content=f["text_content"],
                    token_count=f["token_count"],
                    created_at=datetime.fromisoformat(f["created_at"]),
                )
                self.files[rec.id] = rec
            logger.info("Loaded %d files from disk", len(self.files))
        except Exception as exc:
            logger.warning("Failed to load files: %s", exc)

    def save_files(self) -> None:
        """Persist uploaded files to disk (most recent N only).

        Capped at :data:`_MAX_PERSISTED_FILES` by ``created_at`` descending
        so the on-disk JSON never grows unbounded on long-running servers.
        """
        try:
            _STATE_DIR.mkdir(parents=True, exist_ok=True)
            sorted_files = sorted(
                self.files.values(),
                key=lambda f: f.created_at,
                reverse=True,
            )[:_MAX_PERSISTED_FILES]
            data = [
                {
                    "id": f.id,
                    "name": f.name,
                    "size_bytes": f.size_bytes,
                    "content_type": f.content_type,
                    "text_content": f.text_content,
                    "token_count": f.token_count,
                    "created_at": f.created_at.isoformat(),
                }
                for f in sorted_files
            ]
            _FILES_FILE.write_text(json.dumps(data, indent=2, default=str))
        except Exception as exc:
            logger.warning("Failed to save files: %s", exc)

    # ------------------------------------------------------------------
    # Execution persistence (SQLite via TelemetryStore)
    # ------------------------------------------------------------------

    def save_executions(self) -> None:
        """No-op: executions are now persisted via TelemetryStore on completion."""

    # ------------------------------------------------------------------
    # Evaluation persistence
    # ------------------------------------------------------------------

    def _load_evaluations(self) -> None:
        """Load persisted evaluations from disk if available."""
        evals_path = _EVALUATIONS_FILE
        if not evals_path.exists() and _LEGACY_EVALUATIONS_FILE.exists():
            try:
                _STATE_DIR.mkdir(parents=True, exist_ok=True)
                evals_path.write_text(_LEGACY_EVALUATIONS_FILE.read_text())
                _LEGACY_EVALUATIONS_FILE.unlink()
                logger.info(
                    "Migrated evaluations from %s to %s", _LEGACY_EVALUATIONS_FILE, evals_path
                )
            except Exception as exc:
                logger.warning("Evaluations migration failed: %s", exc)
                evals_path = _LEGACY_EVALUATIONS_FILE
        if not evals_path.exists():
            return
        try:
            raw = json.loads(evals_path.read_text())
            self.evaluations = {
                "thumb_ratings": raw.get("thumb_ratings", []),
                "best_picks": raw.get("best_picks", []),
                "judge_scores": raw.get("judge_scores", []),
                "judge_pairwise": raw.get("judge_pairwise", []),
            }
            total = sum(len(v) for v in self.evaluations.values())
            logger.info("Loaded %d evaluations from disk", total)
        except Exception as exc:
            logger.warning("Failed to load evaluations: %s", exc)

    def save_evaluations(self) -> None:
        """Persist evaluations to disk."""
        try:
            _STATE_DIR.mkdir(parents=True, exist_ok=True)
            _EVALUATIONS_FILE.write_text(json.dumps(self.evaluations, indent=2, default=str))
        except Exception as exc:
            logger.warning("Failed to save evaluations: %s", exc)

    def get_or_create_session(self, session_id: str | None = None) -> SessionRecord:
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        now = datetime.now(timezone.utc)
        sid = session_id or str(uuid.uuid4())
        session = SessionRecord(
            id=sid,
            name=f"Session {now.strftime('%Y-%m-%d %H:%M')}",
            created_at=now,
            updated_at=now,
        )
        self.sessions[sid] = session
        return session

    def _resolve_active_backend(self) -> tuple[str, str, str | None, RuntimeSettings]:
        """``(provider_key, model, api_base, runtime)`` for the globally active provider.

        Shared by :meth:`create_llm_adapter` and :meth:`create_rlm_engine`.
        """
        provider_key = self.config.active_provider
        model = self.config.active_model

        # Trust the user-selected model — it may have been fetched live from
        # the provider API and won't be in the static catalog.
        entry = PROVIDERS_BY_KEY.get(provider_key)

        # Prefer persisted endpoint over catalog default
        api_base: str | None = None
        runtime = self.config.default_runtime_settings
        for pc in self.config.provider_configs:
            if pc.provider == provider_key and pc.enabled:
                if pc.endpoint:
                    api_base = pc.endpoint
                runtime = pc.runtime_settings
                break
        if not api_base and entry and entry.default_endpoint:
            api_base = entry.default_endpoint
        return provider_key, model, api_base, runtime

    def create_llm_adapter(self, num_retries: int | None = None) -> LiteLLMAdapter:
        provider_key, model, api_base, runtime = self._resolve_active_backend()

        # Apply LiteLLM provider prefix (e.g. "ollama/" for Ollama models)
        prefixed_model = self._litellm_model_name(provider_key, model)

        # Priority: per-request > local-provider default > 2
        effective_retries = (
            num_retries if num_retries is not None else 0 if provider_key in _LOCAL_PROVIDERS else 2
        )

        return LiteLLMAdapter(
            model=prefixed_model,
            api_base=api_base,
            temperature=runtime.temperature,
            top_p=runtime.top_p,
            max_tokens=runtime.max_output_tokens,
            timeout=float(runtime.timeout_seconds),
            num_retries=effective_retries,
        )

    @staticmethod
    def _litellm_model_name(provider_key: str, model: str) -> str:
        """Prepend the LiteLLM provider prefix if needed.

        HuggingFace-style IDs (e.g. 'Qwen/Qwen2.5-7B-Instruct') contain a
        slash but are not yet prefixed — only skip when the model already
        starts with this provider's own LiteLLM prefix.
        """
        _prefixes = {
            "anthropic": "anthropic/",
            "ollama": "ollama/",
            "lmstudio": "openai/",
            # hosted_vllm/, not openai/ — see docs/hosts/dgx-spark-vllm.md §7
            # Blocker #4. Must stay in sync with the tables in
            # server/routes/providers.py and api.py.
            "vllm": "hosted_vllm/",
        }
        prefix = _prefixes.get(provider_key, "")
        if not prefix or model.startswith(prefix):
            return model
        return f"{prefix}{model}"

    def create_sandbox(self):  # type: ignore[no-untyped-def]
        sandbox_type = self.config.sandbox.type
        # Server runs in worker threads where signal-based timeouts don't work.
        # Auto-select subprocess for "local" only; "restricted" stays as-is
        # because it provides AST-level security that subprocess doesn't.
        if sandbox_type == "local":
            sandbox_type = "subprocess"
        return create_sandbox(sandbox_type=sandbox_type)

    def create_run_config(self, mode: str = MODE_AUTO) -> RunConfigDTO:
        b = self.config.budget
        return RunConfigDTO(
            mode=mode,
            max_steps=b.max_steps,
            max_tokens=b.max_tokens,
            max_cost=b.max_cost_usd,
            max_time_seconds=float(b.max_time_seconds),
            max_recursion_depth=b.max_recursion_depth,
            repeat_limit=b.repeat_limit,
            nudge_at_fraction=b.nudge_at_fraction,
            **_registry_config_kwargs(),
        )


def _registry_config_kwargs() -> dict[str, int]:
    """Result-registry limits from the library config (``execution:`` section).

    The Studio config file (``~/.rlm-studio/config.json``) has no UI for these
    yet; they come from the YAML library config and fall back to the DTO
    defaults when it cannot be read.
    """
    from rlmstudio.config import get_default_config

    try:
        kwargs: dict[str, int] = get_default_config().execution.registry_kwargs()
    except Exception:
        return {}
    return kwargs


# Singleton state -- created once at startup, injected into routes
_state: AppState | None = None


def get_state() -> AppState:
    global _state
    if _state is None:
        _state = AppState()
    return _state


def reset_state() -> None:
    """Reset state (used in tests). Creates a fresh AppState without loading from disk.

    Also disables disk persistence so tests never overwrite the user's config
    and stops any background connection-test thread left over from a
    previous test (the env-var override can keep one running).
    """
    global _state
    if _state is not None:
        _state._stop_connection_testing()
    _state = AppState(load_from_disk=False)
    _state.save_config = lambda: None  # type: ignore[assignment]
    _state.save_sessions = lambda: None  # type: ignore[assignment]
    _state.save_files = lambda: None  # type: ignore[assignment]
    _state.save_evaluations = lambda: None  # type: ignore[assignment]
