"""Unit tests for the scheduled-connection-testing background thread.

Covers the invariants from doc_internal/specs/scheduled-connection-testing.md:

- Thread lifecycle: start/stop/restart, interval=0 means no thread.
- First cycle happens AFTER the interval, not at startup.
- Failure threshold (N=2) debounces transient blips.
- Manual vs background status transitions use distinct code paths.
- Stale-result guard discards results for providers edited mid-cycle.
- Stop signal discards in-flight results.
- Single provider raising does not kill the cycle.
- save_config is batched (once per cycle, not once per provider).

All tests use the RLM_STUDIO_CONNECTION_TEST_INTERVAL_SECONDS_OVERRIDE env var
via ``monkeypatch.setenv`` so they can run cycles at sub-second frequency
without touching the minutes-based user-facing knob.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import pytest

from rlmstudio.application.services.provider_failure import ProviderFailureKind
from rlmstudio.application.services.provider_tester import ProviderTestResult
from rlmstudio.server.dependencies import AppState
from rlmstudio.server.models import LLMProviderConfig

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _make_lp(
    *,
    lp_id: str = "prov-1",
    name: str = "Test Provider",
    backend: str = "openai",
    model: str = "gpt-4o-mini",
    endpoint: str | None = None,
    status: str = "connected",
    consecutive_failures: int = 0,
) -> LLMProviderConfig:
    return LLMProviderConfig(
        id=lp_id,
        name=name,
        backend=backend,
        model=model,
        endpoint=endpoint,
        status=status,
        consecutive_failures=consecutive_failures,
    )


def _connected_result() -> ProviderTestResult:
    return ProviderTestResult(
        status="connected",
        tested_at=datetime.now(timezone.utc),
        latency_ms=42,
        error_message=None,
    )


def _offline_result(msg: str = "timeout") -> ProviderTestResult:
    return ProviderTestResult(
        status="offline",
        tested_at=datetime.now(timezone.utc),
        latency_ms=None,
        error_message=msg,
    )


def _error_result() -> ProviderTestResult:
    return ProviderTestResult(
        status="error",
        tested_at=datetime.now(timezone.utc),
        latency_ms=None,
        error_message="boom",
    )


def _auth_failure_result() -> ProviderTestResult:
    return ProviderTestResult(
        status="offline",
        tested_at=datetime.now(timezone.utc),
        latency_ms=None,
        error_message="invalid x-api-key",
        failure_kind=ProviderFailureKind.AUTH,
        failure_cause="The API key was rejected.",
        failure_fix="Check the key has not been revoked or regenerated, then update it in Settings → Providers.",
    )


@pytest.fixture
def state() -> Iterator[AppState]:
    """Fresh state with save_config disabled (in-memory only)."""
    s = AppState(load_from_disk=False)
    s.save_config = lambda: None  # type: ignore[assignment]
    yield s
    s._stop_connection_testing()


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


def test_interval_zero_does_not_start_thread(state: AppState) -> None:
    assert state.config.connection_test_interval_minutes == 0
    state._start_connection_testing()
    assert state._connection_test_thread is None


def test_interval_positive_starts_thread(state: AppState, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RLM_STUDIO_CONNECTION_TEST_INTERVAL_SECONDS_OVERRIDE", "10")
    state._start_connection_testing()
    assert state._connection_test_thread is not None
    assert state._connection_test_thread.is_alive()


def test_stop_joins_with_2s_timeout(state: AppState, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RLM_STUDIO_CONNECTION_TEST_INTERVAL_SECONDS_OVERRIDE", "10")
    state._start_connection_testing()
    thread = state._connection_test_thread
    assert thread is not None

    start = time.monotonic()
    state._stop_connection_testing()
    elapsed = time.monotonic() - start

    assert elapsed < 2.5, "stop should return within 2s join timeout"
    assert not thread.is_alive()
    assert state._connection_test_thread is None


def test_restart_picks_up_new_interval(state: AppState, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RLM_STUDIO_CONNECTION_TEST_INTERVAL_SECONDS_OVERRIDE", "10")
    state._start_connection_testing()
    first_thread = state._connection_test_thread
    assert first_thread is not None

    state.restart_connection_testing()
    second_thread = state._connection_test_thread
    assert second_thread is not None
    assert second_thread is not first_thread
    assert second_thread.is_alive()


def test_restart_with_interval_zero_stops_thread(
    state: AppState, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RLM_STUDIO_CONNECTION_TEST_INTERVAL_SECONDS_OVERRIDE", "10")
    state._start_connection_testing()
    assert state._connection_test_thread is not None

    monkeypatch.delenv("RLM_STUDIO_CONNECTION_TEST_INTERVAL_SECONDS_OVERRIDE")
    # interval_minutes is 0 by default → no thread
    state.restart_connection_testing()
    assert state._connection_test_thread is None


def test_first_cycle_begins_after_interval_not_immediately(
    state: AppState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sleep-first invariant: no probe should happen before the first interval."""
    call_count = {"n": 0}

    def _fake_safe_test(lp: LLMProviderConfig) -> ProviderTestResult:
        call_count["n"] += 1
        return _connected_result()

    lp = _make_lp()
    state.config.llm_providers = [lp]
    monkeypatch.setenv("RLM_STUDIO_CONNECTION_TEST_INTERVAL_SECONDS_OVERRIDE", "5")
    with patch.object(AppState, "_safe_test", _fake_safe_test):
        state._start_connection_testing()
        # Give the thread a moment to start and enter sleep.
        time.sleep(0.2)
        # It MUST NOT have run a cycle yet (interval is 5s).
        assert call_count["n"] == 0


# --------------------------------------------------------------------------
# Cycle behavior
# --------------------------------------------------------------------------


def test_run_cycle_applies_connected_result(state: AppState) -> None:
    lp = _make_lp(status="offline", consecutive_failures=3)
    state.config.llm_providers = [lp]

    with patch.object(AppState, "_safe_test", lambda self, p: _connected_result()):
        state._run_test_cycle()

    got = state.config.llm_providers[0]
    assert got.status == "connected"
    assert got.consecutive_failures == 0
    assert got.last_tested_by == "background"
    assert got.last_tested_at is not None


def test_cycle_persists_the_failure_classification(state: AppState) -> None:
    """Even below the offline threshold, the cause is recorded for the row."""
    lp = _make_lp(status="connected", consecutive_failures=0)
    state.config.llm_providers = [lp]

    with patch.object(AppState, "_safe_test", lambda self, p: _auth_failure_result()):
        state._run_test_cycle()

    got = state.config.llm_providers[0]
    assert got.status == "connected"  # below the N=2 threshold
    assert got.last_failure_kind == "auth"
    assert got.last_failure_cause is not None
    assert "rejected" in got.last_failure_cause
    assert got.last_failure_fix


def test_cycle_success_clears_the_failure_classification(state: AppState) -> None:
    lp = _make_lp(status="offline")
    lp.last_failure_kind = "auth"
    lp.last_failure_cause = "The API key was rejected."
    lp.last_failure_fix = "Check the key."
    state.config.llm_providers = [lp]

    with patch.object(AppState, "_safe_test", lambda self, p: _connected_result()):
        state._run_test_cycle()

    got = state.config.llm_providers[0]
    assert got.last_failure_kind is None
    assert got.last_failure_cause is None
    assert got.last_failure_fix is None


def test_consecutive_failure_threshold(state: AppState) -> None:
    """N=2: one failure stays connected, two flip to offline."""
    lp = _make_lp(status="connected", consecutive_failures=0)
    state.config.llm_providers = [lp]

    with patch.object(AppState, "_safe_test", lambda self, p: _offline_result()):
        state._run_test_cycle()
    # After 1 failure: still connected, counter=1.
    got = state.config.llm_providers[0]
    assert got.status == "connected"
    assert got.consecutive_failures == 1

    with patch.object(AppState, "_safe_test", lambda self, p: _offline_result()):
        state._run_test_cycle()
    # After 2 failures: flip.
    got = state.config.llm_providers[0]
    assert got.status == "offline"
    assert got.consecutive_failures == 2


def test_single_success_flips_offline_back_to_connected(state: AppState) -> None:
    lp = _make_lp(status="offline", consecutive_failures=5)
    state.config.llm_providers = [lp]

    with patch.object(AppState, "_safe_test", lambda self, p: _connected_result()):
        state._run_test_cycle()

    got = state.config.llm_providers[0]
    assert got.status == "connected"
    assert got.consecutive_failures == 0


def test_error_status_distinct_from_offline(state: AppState) -> None:
    lp = _make_lp(status="connected", consecutive_failures=0)
    state.config.llm_providers = [lp]

    with patch.object(AppState, "_safe_test", lambda self, p: _error_result()):
        state._run_test_cycle()

    got = state.config.llm_providers[0]
    assert got.status == "error"  # error flips immediately, not after N
    assert got.consecutive_failures == 1


def test_single_provider_exception_does_not_kill_cycle(state: AppState) -> None:
    """Three providers; middle one's probe raises.  First and third are
    still tested and persisted."""
    providers = [
        _make_lp(lp_id="p1", name="P1", status="offline"),
        _make_lp(lp_id="p2", name="P2", status="connected"),
        _make_lp(lp_id="p3", name="P3", status="offline"),
    ]
    state.config.llm_providers = providers

    def _flaky_test(self: AppState, p: LLMProviderConfig) -> ProviderTestResult:
        if p.id == "p2":
            raise RuntimeError("simulated probe crash")
        return _connected_result()

    with patch.object(AppState, "_safe_test", _flaky_test):
        state._run_test_cycle()

    # NOTE: _safe_test is supposed to swallow exceptions.  We patched it
    # away so we need to also patch _run_test_cycle's handling directly.
    # Let's switch to patching test_provider instead to go through the
    # real _safe_test path.
    for p in providers:
        p.status = "offline"
        p.consecutive_failures = 0
        p.last_tested_by = None

    def _real_test(provider: LLMProviderConfig, timeout_s: float) -> ProviderTestResult:
        if provider.id == "p2":
            raise RuntimeError("simulated probe crash")
        return _connected_result()

    with patch(
        "rlmstudio.application.services.provider_tester.test_provider",
        side_effect=_real_test,
    ):
        state._run_test_cycle()

    got = {p.id: p for p in state.config.llm_providers}
    # p1 and p3 should still be tested and connected.
    assert got["p1"].status == "connected"
    assert got["p3"].status == "connected"
    # p2 should be classified as error by _safe_test's belt-and-braces.
    assert got["p2"].status == "error"
    assert got["p2"].consecutive_failures == 1


def test_stale_result_discarded_when_provider_edited_mid_cycle(
    state: AppState,
) -> None:
    """Edit the provider's base_url between snapshot and apply.  Cycle's
    result must be discarded without mutating consecutive_failures."""
    lp = _make_lp(
        status="connected",
        consecutive_failures=0,
        endpoint="http://original",
    )
    state.config.llm_providers = [lp]

    probe_called = threading.Event()
    edit_done = threading.Event()

    def _slow_probe(provider: LLMProviderConfig, timeout_s: float) -> ProviderTestResult:
        # Signal the main thread that the probe started, then block until
        # the edit is done before returning.
        probe_called.set()
        edit_done.wait(timeout=5.0)
        # Return offline to distinguish: if the result is NOT discarded,
        # counter would become 1.
        return _offline_result()

    def _edit_in_background() -> None:
        probe_called.wait(timeout=5.0)
        # Mutate while cycle is in flight.
        with state._config_lock:
            state.config.llm_providers[0].endpoint = "http://changed"
        edit_done.set()

    edit_thread = threading.Thread(target=_edit_in_background)
    edit_thread.start()
    try:
        with patch(
            "rlmstudio.application.services.provider_tester.test_provider",
            side_effect=_slow_probe,
        ):
            state._run_test_cycle()
    finally:
        edit_thread.join(timeout=5.0)

    got = state.config.llm_providers[0]
    # Fingerprint changed → result discarded → counter unchanged.
    assert got.consecutive_failures == 0
    assert got.status == "connected"
    # The endpoint was in fact changed.
    assert got.endpoint == "http://changed"


def test_deleted_provider_result_dropped(state: AppState) -> None:
    """Provider deleted mid-cycle → result dropped, no exception, no crash."""
    lp = _make_lp(status="connected")
    state.config.llm_providers = [lp]

    def _probe_and_delete(provider: LLMProviderConfig, timeout_s: float) -> ProviderTestResult:
        # Simulate the delete happening while our probe runs.
        with state._config_lock:
            state.config.llm_providers = []
        return _offline_result()

    with patch(
        "rlmstudio.application.services.provider_tester.test_provider",
        side_effect=_probe_and_delete,
    ):
        state._run_test_cycle()

    # No exception.  Provider list is empty.
    assert state.config.llm_providers == []


def test_save_config_batched_once_per_cycle(state: AppState) -> None:
    providers = [_make_lp(lp_id=f"p{i}", name=f"P{i}", status="connected") for i in range(5)]
    state.config.llm_providers = providers

    save_call_count = {"n": 0}

    def _count_save() -> None:
        save_call_count["n"] += 1

    state.save_config = _count_save  # type: ignore[assignment]

    with patch.object(AppState, "_safe_test", lambda self, p: _connected_result()):
        state._run_test_cycle()

    # Exactly ONE save per cycle, not one per provider.
    assert save_call_count["n"] == 1


def test_empty_provider_list_runs_no_probes(state: AppState) -> None:
    state.config.llm_providers = []
    call_count = {"n": 0}

    def _track(self: AppState, p: LLMProviderConfig) -> ProviderTestResult:
        call_count["n"] += 1
        return _connected_result()

    with patch.object(AppState, "_safe_test", _track):
        state._run_test_cycle()

    assert call_count["n"] == 0


def test_shutdown_hook_joins_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """FastAPI lifespan shutdown handler joins the thread cleanly.

    Pins the wiring between create_app()'s lifespan and
    AppState._stop_connection_testing — without it the thread would still
    die (daemon), but only after the logger has shut down.
    """
    from fastapi.testclient import TestClient

    from rlmstudio.server.app import create_app
    from rlmstudio.server.dependencies import get_state, reset_state

    monkeypatch.setenv("RLM_STUDIO_CONNECTION_TEST_INTERVAL_SECONDS_OVERRIDE", "30")
    reset_state()
    app = create_app()

    # TestClient enters/exits the lifespan context manager on __enter__
    # / __exit__, so use it as a context manager here.
    with TestClient(app):
        state = get_state()
        # Manually start the thread since reset_state created a fresh
        # AppState(load_from_disk=False) which skipped __init__'s start.
        state._start_connection_testing()
        assert state._connection_test_thread is not None
        thread = state._connection_test_thread
    # On context exit, shutdown runs.
    assert not thread.is_alive(), "shutdown hook did not join the thread"


def test_first_cycle_sleeps_before_testing(
    state: AppState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Thread must call stop.wait BEFORE the first _run_test_cycle call."""
    wait_calls: list[float | None] = []
    run_cycle_called = threading.Event()

    real_wait = threading.Event.wait

    def _tracking_wait(self: threading.Event, timeout: float | None = None) -> bool:
        wait_calls.append(timeout)
        return real_wait(self, timeout)

    def _track_cycle(self: AppState) -> None:
        run_cycle_called.set()

    monkeypatch.setenv("RLM_STUDIO_CONNECTION_TEST_INTERVAL_SECONDS_OVERRIDE", "0.5")
    with patch.object(threading.Event, "wait", _tracking_wait):
        with patch.object(AppState, "_run_test_cycle", _track_cycle):
            state._start_connection_testing()
            run_cycle_called.wait(timeout=3.0)
            state._stop_connection_testing()

    # At least one wait with timeout ~0.5 happened before _run_test_cycle.
    assert any(t is not None and 0.4 < float(t) < 0.7 for t in wait_calls)


# --------------------------------------------------------------------------
# Model check from the probe (#95 Scope 2)
# --------------------------------------------------------------------------


def _model_missing_result() -> ProviderTestResult:
    return ProviderTestResult(
        status="offline",
        tested_at=datetime.now(timezone.utc),
        latency_ms=None,
        error_message="model not found",
        failure_kind=ProviderFailureKind.MODEL_UNAVAILABLE,
        failure_cause="The provider does not offer this model.",
        failure_fix="Pick a model from the provider's list.",
    )


def test_cycle_records_an_available_model_on_connect(state: AppState) -> None:
    lp = _make_lp(status="offline")
    lp.model_check_status = "unavailable"
    state.config.llm_providers = [lp]

    with patch.object(AppState, "_safe_test", lambda self, p: _connected_result()):
        state._run_test_cycle()

    assert state.config.llm_providers[0].model_check_status == "available"


def test_cycle_records_an_unavailable_model(state: AppState) -> None:
    lp = _make_lp(status="connected")
    state.config.llm_providers = [lp]

    with patch.object(AppState, "_safe_test", lambda self, p: _model_missing_result()):
        state._run_test_cycle()

    assert state.config.llm_providers[0].model_check_status == "unavailable"


# --------------------------------------------------------------------------
# One-shot startup probe (#95 Scope 4)
# --------------------------------------------------------------------------


def _startup_state(monkeypatch: pytest.MonkeyPatch) -> AppState:
    """AppState whose constructor-time startup probe is suppressed."""
    monkeypatch.setenv("RLM_STUDIO_STARTUP_PROBE", "0")
    s = AppState(load_from_disk=False)
    s.save_config = lambda: None  # type: ignore[assignment]
    return s


def _enable_startup_probe(monkeypatch: pytest.MonkeyPatch, delay: str = "0") -> None:
    monkeypatch.setenv("RLM_STUDIO_STARTUP_PROBE", "1")
    monkeypatch.setenv("RLM_STUDIO_STARTUP_PROBE_DELAY_SECONDS_OVERRIDE", delay)


def test_startup_probe_is_enabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RLM_STUDIO_STARTUP_PROBE", raising=False)
    monkeypatch.setenv("RLM_STUDIO_STARTUP_PROBE_DELAY_SECONDS_OVERRIDE", "30")
    state = AppState(load_from_disk=False)
    state.save_config = lambda: None  # type: ignore[assignment]
    try:
        assert state._startup_probe_thread is not None
    finally:
        state._stop_connection_testing()
        thread = state._startup_probe_thread
        if thread is not None:
            thread.join(timeout=2.0)


def test_startup_probe_disabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RLM_STUDIO_STARTUP_PROBE", "0")
    state = AppState(load_from_disk=False)
    state.save_config = lambda: None  # type: ignore[assignment]

    assert state._startup_probe_thread is None


def test_startup_probe_runs_one_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _startup_state(monkeypatch)
    lp = _make_lp(status="configured")
    state.config.llm_providers = [lp]
    _enable_startup_probe(monkeypatch, "0")

    with patch.object(AppState, "_safe_test", lambda self, p: _connected_result()):
        state._start_startup_probe()
        thread = state._startup_probe_thread
        assert thread is not None
        thread.join(timeout=5.0)

    assert lp.status == "connected"
    assert lp.last_tested_by == "background"


def test_startup_probe_is_not_rerun(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _startup_state(monkeypatch)
    _enable_startup_probe(monkeypatch, "0")

    with patch.object(AppState, "_run_test_cycle", lambda self: None):
        state._start_startup_probe()
        first = state._startup_probe_thread
        assert first is not None
        first.join(timeout=5.0)

        state._start_startup_probe()  # guard: no second thread
        state.restart_connection_testing()  # never restarts the one-shot
        assert state._startup_probe_thread is first


def test_startup_probe_aborts_on_shutdown_while_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _startup_state(monkeypatch)
    calls: list[int] = []
    _enable_startup_probe(monkeypatch, "5")

    with patch.object(AppState, "_run_test_cycle", lambda self: calls.append(1)):
        state._start_startup_probe()
        state._stop_connection_testing()  # sets the shared stop event
        thread = state._startup_probe_thread
        assert thread is not None
        thread.join(timeout=2.0)

    assert calls == []


def test_startup_cycle_discards_results_on_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _startup_state(monkeypatch)
    lp = _make_lp(status="configured")
    state.config.llm_providers = [lp]
    entered = threading.Event()
    release = threading.Event()

    def _blocking_test(_self: AppState, _p: LLMProviderConfig) -> ProviderTestResult:
        entered.set()
        release.wait(timeout=5.0)
        return _connected_result()

    _enable_startup_probe(monkeypatch, "0")
    with patch.object(AppState, "_safe_test", _blocking_test):
        state._start_startup_probe()
        assert entered.wait(timeout=2.0)
        state._stop_connection_testing()
        release.set()
        thread = state._startup_probe_thread
        assert thread is not None
        thread.join(timeout=5.0)

    assert lp.status == "configured"
    assert lp.last_tested_at is None


def test_startup_cycle_discards_results_when_provider_edited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _startup_state(monkeypatch)
    lp = _make_lp(status="configured")
    state.config.llm_providers = [lp]
    entered = threading.Event()
    release = threading.Event()

    def _blocking_test(_self: AppState, _p: LLMProviderConfig) -> ProviderTestResult:
        entered.set()
        release.wait(timeout=5.0)
        return _connected_result()

    _enable_startup_probe(monkeypatch, "0")
    with patch.object(AppState, "_safe_test", _blocking_test):
        state._start_startup_probe()
        assert entered.wait(timeout=2.0)
        with state._config_lock:
            lp.model = "edited-mid-cycle"
        release.set()
        thread = state._startup_probe_thread
        assert thread is not None
        thread.join(timeout=5.0)

    assert lp.status == "configured"
    assert lp.last_tested_at is None


def test_repeated_start_connection_testing_spawns_no_startup_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _startup_state(monkeypatch)
    _enable_startup_probe(monkeypatch, "5")
    try:
        state._start_startup_probe()
        first = state._startup_probe_thread
        assert first is not None
        for _ in range(3):
            state._start_connection_testing()
        assert state._startup_probe_thread is first
    finally:
        state._stop_connection_testing()
        first = state._startup_probe_thread
        if first is not None:
            first.join(timeout=2.0)


def test_startup_probe_does_not_shorten_the_periodic_first_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _startup_state(monkeypatch)
    lp = _make_lp(status="configured")
    state.config.llm_providers = [lp]
    _enable_startup_probe(monkeypatch, "0")
    monkeypatch.setenv("RLM_STUDIO_CONNECTION_TEST_INTERVAL_SECONDS_OVERRIDE", "5")

    with patch.object(AppState, "_safe_test", lambda self, p: _connected_result()):
        state._start_connection_testing()  # periodic loop: sleeps 5s first
        state._start_startup_probe()  # one-shot: runs now
        thread = state._startup_probe_thread
        assert thread is not None
        thread.join(timeout=5.0)
        assert lp.status == "connected"

        # Reset and wait: the periodic loop must still be in its first sleep.
        lp.status = "configured"
        time.sleep(0.3)
        assert lp.status == "configured"

    state._stop_connection_testing()


def test_startup_cycle_discards_results_after_a_key_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key-only edit changes nothing in the model/endpoint fingerprint.

    The revision (updated_at) must catch it, or the old-key cycle restores
    the state the update just cleared.
    """
    state = _startup_state(monkeypatch)
    lp = _make_lp(status="configured")
    state.config.llm_providers = [lp]
    entered = threading.Event()
    release = threading.Event()

    def _blocking_test(_self: AppState, _p: LLMProviderConfig) -> ProviderTestResult:
        entered.set()
        release.wait(timeout=5.0)
        return _connected_result()

    _enable_startup_probe(monkeypatch, "0")
    with patch.object(AppState, "_safe_test", _blocking_test):
        state._start_startup_probe()
        assert entered.wait(timeout=2.0)
        with state._config_lock:
            lp.updated_at = datetime.now(timezone.utc) + timedelta(seconds=1)
            lp.status = "configured"
            lp.last_tested_at = None
        release.set()
        thread = state._startup_probe_thread
        assert thread is not None
        thread.join(timeout=5.0)

    assert lp.status == "configured"
    assert lp.last_tested_at is None


def test_restart_discards_an_inflight_startup_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restart replaces the stop event; the in-flight cycle must keep
    checking the event it started under, not the replacement."""
    state = _startup_state(monkeypatch)
    lp = _make_lp(status="configured")
    state.config.llm_providers = [lp]
    entered = threading.Event()
    release = threading.Event()

    def _blocking_test(_self: AppState, _p: LLMProviderConfig) -> ProviderTestResult:
        entered.set()
        release.wait(timeout=5.0)
        return _connected_result()

    _enable_startup_probe(monkeypatch, "0")
    monkeypatch.setenv("RLM_STUDIO_CONNECTION_TEST_INTERVAL_SECONDS_OVERRIDE", "5")
    try:
        with patch.object(AppState, "_safe_test", _blocking_test):
            state._start_startup_probe()
            assert entered.wait(timeout=2.0)
            state.restart_connection_testing()  # sets the old event, installs a new one
            release.set()
            thread = state._startup_probe_thread
            assert thread is not None
            thread.join(timeout=5.0)
    finally:
        state._stop_connection_testing()

    assert lp.status == "configured"
    assert lp.last_tested_at is None


def test_restart_while_the_startup_probe_waits_aborts_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _startup_state(monkeypatch)
    calls: list[int] = []
    _enable_startup_probe(monkeypatch, "5")

    def _cycle(_self: AppState, _event: object = None) -> None:
        calls.append(1)

    with patch.object(AppState, "_run_test_cycle", _cycle):
        state._start_startup_probe()
        state.restart_connection_testing()
        thread = state._startup_probe_thread
        assert thread is not None
        thread.join(timeout=2.0)

    assert calls == []


def test_startup_populates_alternatives_from_the_account_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scope 2's load-time contract: the startup path fetches /v1/models.

    The completion probe alone cannot report what a retired model was
    replaced by; this is the account-list comparison running on load.
    """
    from rlmstudio.server.models import ModelInfo

    state = _startup_state(monkeypatch)
    lp = _make_lp(status="configured", model="retired-model")
    state.config.llm_providers = [lp]
    _enable_startup_probe(monkeypatch, "0")

    async def _models(*_args: object, **_kwargs: object) -> list[ModelInfo]:
        return [ModelInfo(name="gpt-4o"), ModelInfo(name="gpt-4o-mini")]

    with (
        # Auth failure says nothing about the model, so only discovery can set it.
        patch.object(AppState, "_safe_test", lambda self, p: _auth_failure_result()),
        patch("rlmstudio.server.routes.providers._fetch_models_from_api", _models),
    ):
        state._start_startup_probe()
        thread = state._startup_probe_thread
        assert thread is not None
        thread.join(timeout=5.0)

    assert lp.model_check_status == "unavailable"
    assert lp.model_check_alternatives == ["gpt-4o", "gpt-4o-mini"]


class _RecordingThread:
    """Captures a thread's target/args instead of starting it."""

    instances: list[_RecordingThread] = []

    def __init__(
        self,
        target: Any = None,
        args: tuple[Any, ...] = (),
        name: str | None = None,
        daemon: bool | None = None,
    ) -> None:
        self.target = target
        self.args = args
        self.name = name
        self.daemon = daemon
        _RecordingThread.instances.append(self)

    def start(self) -> None:
        return None

    def join(self, timeout: float | None = None) -> None:
        return None

    def is_alive(self) -> bool:
        return False


def test_startup_probe_captures_its_event_before_spawning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restart between start() and execution must not revive the probe.

    The target must use the event captured at scheduling time, not the
    attribute a restart may have replaced.
    """
    state = _startup_state(monkeypatch)
    _enable_startup_probe(monkeypatch, "1")
    calls: list[int] = []
    _RecordingThread.instances = []

    with (
        patch("rlmstudio.server.dependencies.threading.Thread", _RecordingThread),
        patch.object(AppState, "_run_test_cycle", lambda self, event=None: calls.append(1)),
    ):
        state._start_startup_probe()
        rec = _RecordingThread.instances[-1]
        scheduled_event = rec.args[1]
        assert scheduled_event is state._connection_test_stop

        # Simulate a restart between start() and the target running.
        state._connection_test_stop = threading.Event()
        scheduled_event.set()
        rec.target(*rec.args)

    assert calls == []


def test_connection_loop_uses_the_event_it_was_started_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A loop whose join timed out must exit, not follow the replacement event."""
    state = _startup_state(monkeypatch)
    monkeypatch.setenv("RLM_STUDIO_CONNECTION_TEST_INTERVAL_SECONDS_OVERRIDE", "5")
    calls: list[int] = []
    _RecordingThread.instances = []

    with patch.object(AppState, "_run_test_cycle", lambda self, event=None: calls.append(1)):
        # Scope the Thread patch to the spawn only: the worker below must be a
        # real thread, or it would never run the loop.
        with patch("rlmstudio.server.dependencies.threading.Thread", _RecordingThread):
            state._start_connection_testing()
            rec = _RecordingThread.instances[-1]
            started_event = rec.args[0]
            assert started_event is state._connection_test_stop

            # Simulate a restart that replaced the attribute after a
            # timed-out join.
            replacement = threading.Event()
            state._connection_test_stop = replacement  # unset replacement
            started_event.set()  # the lifecycle this loop belongs to stopped

        # Run the loop with a timeout: the fixed version exits immediately;
        # a regression that follows the replacement would sleep instead, so
        # unblock it after the timeout to keep the test failing, not hanging.
        worker = threading.Thread(target=rec.target, args=rec.args, daemon=True)
        worker.start()
        worker.join(timeout=1.0)
        still_running = worker.is_alive()
        replacement.set()

    assert not still_running
    assert calls == []


def test_the_listed_status_follows_the_background_check(state: AppState) -> None:
    """Browser pre-pass H2: after a manual Test said Connected, the background
    check flipped the provider offline but the list kept showing Connected (an
    in-memory cache of the manual result won), and an offline or error
    provider was reported as merely "configured", also after a restart."""
    from rlmstudio.server.routes.llm_providers import _compute_status

    lp = _make_lp(status="connected", consecutive_failures=0)
    state.config.llm_providers = [lp]
    assert _compute_status(state.config.llm_providers[0]) == "connected"

    with patch.object(AppState, "_safe_test", lambda self, p: _offline_result()):
        state._run_test_cycle()
        state._run_test_cycle()

    assert _compute_status(state.config.llm_providers[0]) == "offline"
    # A restart reloads the persisted status; it must not decay to "configured".
    assert _compute_status(_make_lp(status="offline")) == "offline"
    assert _compute_status(_make_lp(status="error")) == "error"
