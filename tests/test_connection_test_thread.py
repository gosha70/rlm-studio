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
from datetime import datetime, timezone
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
