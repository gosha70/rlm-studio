"""Tests for TelemetryStore."""

from __future__ import annotations

import json
import time

import pytest

from rlmstudio.telemetry.store import TelemetryStore


@pytest.fixture
def store() -> TelemetryStore:
    """Create an in-memory telemetry store."""
    return TelemetryStore(":memory:")


class TestRecordRun:
    """Recording and retrieving runs."""

    def test_record_and_list(self, store: TelemetryStore) -> None:
        rid = store.record_run(
            created_at=time.time(),
            mode="direct",
            provider="openai",
            model="gpt-4o",
            query="What is 2+2?",
            answer="4",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            total_cost=0.001,
            elapsed_seconds=0.5,
            success=True,
            steps_count=1,
        )
        runs = store.list_runs()
        assert len(runs) == 1
        assert runs[0].id == rid
        assert runs[0].mode == "direct"
        assert runs[0].total_tokens == 15

    def test_record_with_explicit_id(self, store: TelemetryStore) -> None:
        rid = store.record_run(
            run_id="custom-id-123",
            created_at=time.time(),
            mode="rlm",
        )
        assert rid == "custom-id-123"
        detail = store.get_run("custom-id-123")
        assert detail is not None
        assert detail.mode == "rlm"

    def test_count_runs(self, store: TelemetryStore) -> None:
        assert store.count_runs() == 0
        for i in range(5):
            store.record_run(created_at=time.time() + i, mode="direct")
        assert store.count_runs() == 5

    def test_list_runs_with_filters(self, store: TelemetryStore) -> None:
        store.record_run(created_at=1.0, mode="direct", session_id="s1")
        store.record_run(created_at=2.0, mode="rlm", session_id="s1")
        store.record_run(created_at=3.0, mode="direct", session_id="s2")

        assert len(store.list_runs(mode="direct")) == 2
        assert len(store.list_runs(mode="rlm")) == 1
        assert len(store.list_runs(session_id="s1")) == 2
        assert len(store.list_runs(session_id="s2")) == 1

    def test_list_runs_ordered_by_created_at_desc(self, store: TelemetryStore) -> None:
        store.record_run(run_id="old", created_at=1.0, mode="direct")
        store.record_run(run_id="new", created_at=2.0, mode="direct")
        runs = store.list_runs()
        assert runs[0].id == "new"
        assert runs[1].id == "old"

    def test_list_runs_limit_offset(self, store: TelemetryStore) -> None:
        for i in range(10):
            store.record_run(run_id=f"r{i}", created_at=float(i), mode="direct")
        page = store.list_runs(limit=3, offset=0)
        assert len(page) == 3
        assert page[0].id == "r9"  # newest first
        page2 = store.list_runs(limit=3, offset=3)
        assert len(page2) == 3
        assert page2[0].id == "r6"

    def test_failed_run(self, store: TelemetryStore) -> None:
        store.record_run(
            created_at=time.time(),
            mode="rlm",
            success=False,
            error="Timeout exceeded",
        )
        runs = store.list_runs(success=False)
        assert len(runs) == 1
        assert runs[0].error == "Timeout exceeded"
        assert not runs[0].success


class TestRecordStep:
    """Recording and retrieving steps."""

    def test_steps_attached_to_run(self, store: TelemetryStore) -> None:
        rid = store.record_run(run_id="run1", created_at=time.time(), mode="rlm", steps_count=2)
        store.record_step(
            run_id=rid,
            step_index=0,
            action_type="inspect",
            code="print(len(P))",
            output="1234",
            input_tokens=50,
            output_tokens=20,
            duration=0.3,
        )
        store.record_step(
            run_id=rid,
            step_index=1,
            action_type="final",
            output="The answer is 42.",
            input_tokens=60,
            output_tokens=30,
            duration=0.5,
        )
        detail = store.get_run(rid)
        assert detail is not None
        assert len(detail.steps) == 2
        assert detail.steps[0]["action_type"] == "inspect"
        assert detail.steps[0]["code"] == "print(len(P))"
        assert detail.steps[1]["action_type"] == "final"


class TestRecordCall:
    """Recording provider calls."""

    def test_record_call(self, store: TelemetryStore) -> None:
        rid = store.record_run(run_id="run1", created_at=time.time(), mode="direct")
        cid = store.record_call(
            run_id=rid,
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            cost=0.005,
            latency_ms=800,
        )
        assert cid  # non-empty


class TestAddRating:
    """Adding ratings."""

    def test_add_rating(self, store: TelemetryStore) -> None:
        rid = store.record_run(run_id="run1", created_at=time.time(), mode="direct")
        rating_id = store.add_rating(
            run_id=rid,
            rating=5,
            comment="Great answer",
            created_at=time.time(),
        )
        assert rating_id  # non-empty


class TestGetRunDetail:
    """Full run detail retrieval."""

    def test_get_run_not_found(self, store: TelemetryStore) -> None:
        assert store.get_run("nonexistent") is None

    def test_get_run_full_detail(self, store: TelemetryStore) -> None:
        rid = store.record_run(
            run_id="detail-test",
            created_at=100.0,
            mode="rlm",
            provider="anthropic",
            model="claude-sonnet",
            query="Explain X",
            answer="X is ...",
            content_length=5000,
            input_tokens=200,
            output_tokens=100,
            total_tokens=300,
            total_cost=0.01,
            elapsed_seconds=2.5,
            success=True,
            session_id="sess1",
            chat_provider_id="cp1",
            chat_provider_name="My Provider",
            steps_count=3,
        )
        detail = store.get_run(rid)
        assert detail is not None
        assert detail.answer == "X is ..."
        assert detail.content_length == 5000
        assert detail.input_tokens == 200
        assert detail.output_tokens == 100
        assert detail.session_id == "sess1"
        assert detail.chat_provider_name == "My Provider"


class TestAggregation:
    """Aggregation queries."""

    def test_aggregate_by_mode(self, store: TelemetryStore) -> None:
        store.record_run(created_at=1.0, mode="direct", total_tokens=100, total_cost=0.01)
        store.record_run(created_at=2.0, mode="direct", total_tokens=200, total_cost=0.02)
        store.record_run(created_at=3.0, mode="rlm", total_tokens=500, total_cost=0.05)

        agg = store.aggregate_by_mode()
        modes = {r.key: r for r in agg}
        assert "direct" in modes
        assert "rlm" in modes
        assert modes["direct"].run_count == 2
        assert modes["direct"].total_tokens == 300
        assert modes["rlm"].run_count == 1

    def test_aggregate_by_provider(self, store: TelemetryStore) -> None:
        store.record_run(created_at=1.0, mode="direct", provider="openai")
        store.record_run(created_at=2.0, mode="direct", provider="anthropic")
        store.record_run(created_at=3.0, mode="rlm", provider="openai")

        agg = store.aggregate_by_provider()
        providers = {r.key: r for r in agg}
        assert providers["openai"].run_count == 2
        assert providers["anthropic"].run_count == 1


class TestExportJsonl:
    """JSONL export — matches the upstream RLM visualizer schema.

    See ``alexzhang13/rlm/visualizer/src/lib/types.ts`` for the exact
    interface definitions we're targeting.
    """

    def test_export_not_found(self, store: TelemetryStore) -> None:
        assert store.export_jsonl("nonexistent") is None

    def test_metadata_line_is_upstream_compatible(self, store: TelemetryStore) -> None:
        """Line 1 must be ``{"type": "metadata", ...}`` — upstream's
        parseJSONL branches on ``parsed.type === 'metadata'``."""
        rid = store.record_run(
            run_id="meta-test",
            created_at=1000.0,
            mode="rlm",
            provider="openai",
            model="gpt-4o",
            query="test query",
            total_tokens=100,
            total_cost=0.01,
            elapsed_seconds=1.5,
            success=True,
            steps_count=1,  # one raw step → one grouped iteration
            answer="final answer",
        )
        store.record_step(run_id=rid, step_index=0, action_type="final", output="done")

        jsonl = store.export_jsonl(rid)
        assert jsonl is not None
        lines = jsonl.strip().split("\n")
        assert len(lines) == 2  # 1 metadata + 1 iteration

        metadata = json.loads(lines[0])
        # Upstream discriminator
        assert metadata["type"] == "metadata"
        # Upstream RLMConfigMetadata fields (all must be present, even if null)
        for key in (
            "root_model",
            "max_depth",
            "max_iterations",
            "backend",
            "backend_kwargs",
            "environment_type",
            "environment_kwargs",
            "other_backends",
        ):
            assert key in metadata, f"missing required metadata key: {key}"
        # Concrete values we do populate
        assert metadata["root_model"] == "gpt-4o"
        assert metadata["backend"] == "openai"
        # max_iterations must match the number of iteration lines actually
        # emitted — not detail.steps_count, which is the raw pre-grouping
        # step count and can diverge whenever inspect/subcall steps fold
        # into one upstream iteration.
        assert metadata["max_iterations"] == len(lines) - 1  # == 1 here
        assert metadata["environment_type"] == "subprocess"
        # Non-standard extras are allowed (upstream ignores unknown fields)
        assert metadata["rlm_studio_query"] == "test query"
        assert metadata["rlm_studio_total_tokens"] == 100
        # The raw step count is still available as an extra, for
        # cross-referencing with the telemetry store.
        assert metadata["rlm_studio_raw_steps_count"] == 1

    def test_iteration_line_shape(self, store: TelemetryStore) -> None:
        """Each iteration line must have the upstream RLMIteration fields."""
        rid = store.record_run(
            run_id="iter-test",
            created_at=2000.0,
            mode="rlm",
            provider="openai",
            model="gpt-4o",
            query="q",
            success=True,
            steps_count=2,
            answer="done",
        )
        store.record_step(
            run_id=rid,
            step_index=0,
            action_type="inspect",
            code="print(len(P))",
            output="LLM response text",
            duration=0.3,
        )
        store.record_step(run_id=rid, step_index=1, action_type="final", output="done")

        jsonl = store.export_jsonl(rid)
        assert jsonl is not None
        lines = jsonl.strip().split("\n")
        # 1 metadata + 2 iterations (one inspect, one final)
        assert len(lines) == 3

        iter0 = json.loads(lines[1])
        # Upstream RLMIteration required fields
        for key in (
            "iteration",
            "timestamp",
            "prompt",
            "response",
            "code_blocks",
            "final_answer",
            "iteration_time",
        ):
            assert key in iter0, f"missing required iteration key: {key}"
        assert iter0["iteration"] == 0
        assert isinstance(iter0["prompt"], list)  # upstream expects []
        assert iter0["response"] == "LLM response text"
        assert iter0["final_answer"] is None  # inspect, not final
        assert iter0["iteration_time"] == 0.3
        # The inspect step's code should appear as a code_block
        assert len(iter0["code_blocks"]) == 1
        assert iter0["code_blocks"][0]["code"] == "print(len(P))"

        iter1 = json.loads(lines[2])
        assert iter1["iteration"] == 1
        assert iter1["final_answer"] == "done"

    def test_subcall_steps_collapse_into_code_blocks(self, store: TelemetryStore) -> None:
        """Following ``subcall`` steps must attach to the preceding
        assistant iteration's ``code_blocks``, not spawn new iterations."""
        rid = store.record_run(
            run_id="subcall-test",
            created_at=3000.0,
            mode="rlm",
            provider="openai",
            model="gpt-4o",
            success=True,
            steps_count=3,
            answer="done",
        )
        store.record_step(
            run_id=rid,
            step_index=0,
            action_type="inspect",
            code="print(x)",
            output="assistant says run this",
            duration=0.1,
        )
        store.record_step(
            run_id=rid,
            step_index=1,
            action_type="subcall",
            code="print(x)",
            output="42",
            duration=0.2,
        )
        store.record_step(run_id=rid, step_index=2, action_type="final", output="done")

        jsonl = store.export_jsonl(rid)
        assert jsonl is not None
        lines = jsonl.strip().split("\n")
        # 1 metadata + 2 iterations (the subcall collapses into iter 0)
        assert len(lines) == 3, f"expected 3 lines, got {len(lines)}"

        # ``max_iterations`` in metadata must match the number of
        # iteration lines (2), NOT the raw step count (3).  Regression
        # guard for the "metadata says 3, body has 2" inconsistency.
        metadata = json.loads(lines[0])
        assert metadata["max_iterations"] == 2
        assert metadata["rlm_studio_raw_steps_count"] == 3

        iter0 = json.loads(lines[1])
        # Iter 0 should have 2 code blocks: one from inspect (empty result),
        # one from subcall (real stdout).
        assert len(iter0["code_blocks"]) == 2
        inspect_block, subcall_block = iter0["code_blocks"]
        assert inspect_block["code"] == "print(x)"
        assert inspect_block["result"]["stdout"] == ""
        assert subcall_block["result"]["stdout"] == "42"
        assert subcall_block["result"]["execution_time"] == 0.2
        # Execution time rolls up into iteration_time (float sum approx)
        assert iter0["iteration_time"] == pytest.approx(0.3)  # 0.1 + 0.2

    def test_code_block_result_has_all_upstream_fields(self, store: TelemetryStore) -> None:
        """Each ``code_blocks[i].result`` must have the upstream REPLResult
        shape even when we don't have data for every field."""
        rid = store.record_run(
            run_id="repl-shape-test",
            created_at=4000.0,
            mode="rlm",
            provider="openai",
            model="gpt-4o",
            success=True,
            steps_count=1,
        )
        store.record_step(
            run_id=rid,
            step_index=0,
            action_type="subcall",
            code="y = 1",
            output="ran",
            duration=0.05,
        )

        jsonl = store.export_jsonl(rid)
        assert jsonl is not None
        iter_line = json.loads(jsonl.strip().split("\n")[1])
        result = iter_line["code_blocks"][0]["result"]
        for key in ("stdout", "stderr", "locals", "execution_time", "rlm_calls"):
            assert key in result, f"missing required REPLResult key: {key}"
        assert result["stdout"] == "ran"
        assert result["stderr"] == ""
        assert result["locals"] == {}
        assert result["rlm_calls"] == []
        assert result["execution_time"] == 0.05

    def test_error_step_yields_iteration_without_final_answer(self, store: TelemetryStore) -> None:
        """An ``error`` step becomes an iteration whose ``final_answer`` is null."""
        rid = store.record_run(
            run_id="err-test",
            created_at=5000.0,
            mode="rlm",
            provider="openai",
            model="gpt-4o",
            success=False,
            error="budget exceeded",
            steps_count=1,
        )
        store.record_step(
            run_id=rid,
            step_index=0,
            action_type="error",
            output="Something went wrong",
        )
        jsonl = store.export_jsonl(rid)
        assert jsonl is not None
        iter_line = json.loads(jsonl.strip().split("\n")[1])
        assert iter_line["final_answer"] is None

    def test_iteration_timestamps_advance_monotonically(self, store: TelemetryStore) -> None:
        """Each iteration's ``timestamp`` must advance by the previous
        iteration's ``iteration_time``, not repeat ``run_created_at``.

        Regression guard: the original implementation used
        ``run_created_at`` as-is for every ``_new_iteration()`` call,
        so a multi-step run emitted identical timestamps for every
        line and the visualizer lost per-iteration timing/order.
        """
        from datetime import datetime, timezone

        rid = store.record_run(
            run_id="ts-test",
            created_at=1_000_000.0,
            mode="rlm",
            provider="openai",
            model="gpt-4o",
            success=True,
            steps_count=4,
            answer="done",
        )
        # Iteration 0: inspect (0.5s) + subcall (1.0s) → iteration_time 1.5s
        store.record_step(
            run_id=rid,
            step_index=0,
            action_type="inspect",
            code="print(1)",
            output="first assistant reply",
            duration=0.5,
        )
        store.record_step(
            run_id=rid,
            step_index=1,
            action_type="subcall",
            code="print(1)",
            output="42",
            duration=1.0,
        )
        # Iteration 1: inspect (0.25s) + subcall (0.75s) → iteration_time 1.0s
        store.record_step(
            run_id=rid,
            step_index=2,
            action_type="inspect",
            code="print(2)",
            output="second assistant reply",
            duration=0.25,
        )
        store.record_step(
            run_id=rid,
            step_index=3,
            action_type="subcall",
            code="print(2)",
            output="84",
            duration=0.75,
        )

        jsonl = store.export_jsonl(rid)
        assert jsonl is not None
        lines = jsonl.strip().split("\n")
        # 1 metadata + 2 iterations
        assert len(lines) == 3

        iter0 = json.loads(lines[1])
        iter1 = json.loads(lines[2])

        ts0 = datetime.fromisoformat(iter0["timestamp"])
        ts1 = datetime.fromisoformat(iter1["timestamp"])

        # Iteration 0 must start at run_created_at exactly.
        expected_start = datetime.fromtimestamp(1_000_000.0, tz=timezone.utc)
        assert ts0 == expected_start

        # Iteration 1 must start at run_created_at + iteration_time(iter0),
        # i.e. 1_000_001.5, not duplicate iter0's timestamp.
        delta = (ts1 - ts0).total_seconds()
        assert delta == pytest.approx(1.5), (
            f"iteration 1 should start 1.5s after iteration 0, got delta={delta:.3f}s"
        )

        # And the iteration_time of iter1 must reflect its own 1.0s span.
        assert iter1["iteration_time"] == pytest.approx(1.0)


class TestDeleteRun:
    """Deleting runs."""

    def test_delete_run(self, store: TelemetryStore) -> None:
        rid = store.record_run(run_id="del-test", created_at=time.time(), mode="direct")
        store.record_step(run_id=rid, step_index=0, action_type="final")
        assert store.get_run(rid) is not None

        store.delete_run(rid)
        assert store.get_run(rid) is None
        assert store.count_runs() == 0


class TestConcurrency:
    """Thread safety under concurrent writes and reads."""

    def test_concurrent_record_run(self) -> None:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        store = TelemetryStore(":memory:")
        total = 50

        def _write(i: int) -> None:
            store.record_run(
                run_id=f"run-{i}",
                created_at=float(i),
                mode="direct",
                query=f"q{i}",
                total_tokens=i,
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_write, i) for i in range(total)]
            for f in as_completed(futures):
                f.result()

        assert store.count_runs() == total

    def test_concurrent_record_run_and_steps(self) -> None:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        store = TelemetryStore(":memory:")
        total = 30

        def _write(i: int) -> None:
            rid = store.record_run(run_id=f"r{i}", created_at=float(i), mode="rlm")
            for step_idx in range(3):
                store.record_step(
                    run_id=rid,
                    step_index=step_idx,
                    action_type="inspect",
                    code=f"print({step_idx})",
                )

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_write, i) for i in range(total)]
            for f in as_completed(futures):
                f.result()

        assert store.count_runs() == total
        for i in range(total):
            detail = store.get_run(f"r{i}")
            assert detail is not None
            assert len(detail.steps) == 3

    def test_concurrent_mixed_read_write(self) -> None:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        store = TelemetryStore(":memory:")
        # Pre-populate some data
        for i in range(10):
            store.record_run(run_id=f"seed-{i}", created_at=float(i), mode="direct")

        errors: list[Exception] = []

        def _mixed(i: int) -> None:
            try:
                if i % 2 == 0:
                    store.record_run(run_id=f"new-{i}", created_at=100.0 + i, mode="rlm")
                else:
                    store.list_runs(limit=5)
                    store.count_runs()
            except Exception as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_mixed, i) for i in range(40)]
            for f in as_completed(futures):
                f.result()

        assert errors == []
        assert store.count_runs() == 10 + 20  # 10 seeds + 20 new


class TestDbPathCreation:
    """Store creates its parent directory on first use."""

    def test_creates_parent_dir(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        nested = tmp_path / "does" / "not" / "exist" / "telemetry.db"
        assert not nested.parent.exists()
        store = TelemetryStore(str(nested))
        assert nested.parent.exists()
        store.record_run(created_at=1.0, mode="direct")
        assert store.count_runs() == 1


class TestServerIntegration:
    """Telemetry store wired into AppState."""

    def test_appstate_has_telemetry(self) -> None:
        from rlmstudio.server.dependencies import AppState

        state = AppState(load_from_disk=False)
        assert hasattr(state, "telemetry")
        assert isinstance(state.telemetry, TelemetryStore)

    def test_reset_state_has_telemetry(self) -> None:
        from rlmstudio.server.dependencies import get_state, reset_state

        reset_state()
        state = get_state()
        assert isinstance(state.telemetry, TelemetryStore)

    def test_save_executions_is_noop(self) -> None:
        from rlmstudio.server.dependencies import AppState

        state = AppState(load_from_disk=False)
        # Should not raise
        state.save_executions()


class TestRecordTelemetryHelper:
    """_record_telemetry persists structured backend/model, not display labels."""

    def test_structured_provider_and_model_recorded(self) -> None:
        from datetime import datetime, timezone

        from rlmstudio.application.dto import RunResultDTO
        from rlmstudio.server.dependencies import AppState, ExecutionRecord
        from rlmstudio.server.routes.chat import _record_telemetry

        state = AppState(load_from_disk=False)
        execution = ExecutionRecord(
            execution_id="exec-ws-1",
            session_id="sess-1",
            query="hello",
            mode="direct",
            status="complete",
            started_at=datetime.now(timezone.utc),
            chat_provider_id="cp-1",
            chat_provider_name="DIRECT-CLAUDE",
        )
        result = RunResultDTO(
            answer="hi",
            mode_used="direct",
            success=True,
            steps=1,
            input_tokens=10,
            output_tokens=5,
            total_cost=0.001,
            elapsed_time=0.2,
            trace=[],
        )

        # Simulate WS path: display name "My Claude" must NOT leak into provider column.
        _record_telemetry(
            state,
            execution,
            result,
            provider="anthropic",
            model="claude-sonnet-4-6",
        )

        detail = state.telemetry.get_run("exec-ws-1")
        assert detail is not None
        assert detail.provider == "anthropic"
        assert detail.model == "claude-sonnet-4-6"

        # aggregate_by_provider sees the backend key, not a display label
        agg = state.telemetry.aggregate_by_provider()
        keys = {row.key for row in agg}
        assert "anthropic" in keys
        assert "My Claude" not in keys

    def test_subcall_row_tokens_and_cost_reconcile_with_run_totals(self) -> None:
        from datetime import datetime, timezone

        from rlmstudio.application.dto import RunResultDTO
        from rlmstudio.server.dependencies import AppState, ExecutionRecord
        from rlmstudio.server.routes.chat import _record_telemetry

        state = AppState(load_from_disk=False)
        execution = ExecutionRecord(
            execution_id="exec-subcall-1",
            session_id="sess-1",
            query="batch them",
            mode="rlm",
            status="complete",
            started_at=datetime.now(timezone.utc),
        )
        result = RunResultDTO(
            answer="done",
            mode_used="rlm",
            success=True,
            steps=3,
            input_tokens=60,
            output_tokens=30,
            total_cost=0.002,
            elapsed_time=0.2,
            trace=[
                {"role": "assistant", "input_tokens": 10, "output_tokens": 5, "model": "gpt-4o"},
                {
                    "role": "execution",
                    "content": '["answer child-0", "answer child-1"]',
                    "input_tokens": 40,
                    "output_tokens": 20,
                    "cost": 0.002,
                    "cost_known": True,
                },
                {"role": "assistant", "input_tokens": 10, "output_tokens": 5, "model": "gpt-4o"},
            ],
        )

        _record_telemetry(state, execution, result, provider="openai", model="gpt-4o")

        detail = state.telemetry.get_run("exec-subcall-1")
        assert detail is not None
        assert sum(int(step.get("input_tokens", 0)) for step in detail.steps) == 60
        assert sum(int(step.get("output_tokens", 0)) for step in detail.steps) == 30
        subcall = next(step for step in detail.steps if step["action_type"] == "subcall")
        assert float(subcall["cost"]) == 0.002
        assert subcall["cost_known"] == 1

    def test_step_action_types_normalized(self) -> None:
        """Raw trace roles (assistant/execution) are normalized to canonical
        ExecutionTrace action types; the last successful step becomes 'final'."""
        from datetime import datetime, timezone

        from rlmstudio.application.dto import RunResultDTO
        from rlmstudio.server.dependencies import AppState, ExecutionRecord
        from rlmstudio.server.routes.chat import _record_telemetry

        state = AppState(load_from_disk=False)
        execution = ExecutionRecord(
            execution_id="exec-norm-1",
            session_id="sess-1",
            query="q",
            mode="rlm",
            status="complete",
            started_at=datetime.now(timezone.utc),
        )
        result = RunResultDTO(
            answer="done",
            mode_used="rlm",
            success=True,
            steps=4,
            trace=[
                {"role": "assistant", "code": "print(len(P))"},
                {"role": "execution", "content": "1234"},
                {"role": "assistant", "code": "print(P[:10])"},
                {"role": "assistant", "content": "The answer is ..."},
            ],
        )
        _record_telemetry(state, execution, result, provider="openai", model="gpt-4o")

        detail = state.telemetry.get_run("exec-norm-1")
        assert detail is not None
        action_types = [s["action_type"] for s in detail.steps]
        # assistant → inspect, execution → subcall, last step (successful) → final
        assert action_types == ["inspect", "subcall", "inspect", "final"]

    def test_step_action_types_on_failure_no_final(self) -> None:
        """On a failed run, the last step is NOT promoted to 'final'."""
        from datetime import datetime, timezone

        from rlmstudio.application.dto import RunResultDTO
        from rlmstudio.server.dependencies import AppState, ExecutionRecord
        from rlmstudio.server.routes.chat import _record_telemetry

        state = AppState(load_from_disk=False)
        execution = ExecutionRecord(
            execution_id="exec-norm-2",
            session_id="sess-1",
            query="q",
            mode="rlm",
            status="error",
            started_at=datetime.now(timezone.utc),
        )
        result = RunResultDTO(
            answer="",
            mode_used="rlm",
            success=False,
            error="budget exceeded",
            steps=2,
            trace=[
                {"role": "assistant", "code": "x=1"},
                {"role": "execution", "content": "ok"},
            ],
        )
        _record_telemetry(state, execution, result, provider="openai", model="gpt-4o")

        detail = state.telemetry.get_run("exec-norm-2")
        assert detail is not None
        action_types = [s["action_type"] for s in detail.steps]
        assert action_types == ["inspect", "subcall"]
        assert "final" not in action_types


class TestExecutionsListingSortOrder:
    """GET /api/executions returns newest-first across in-memory and persisted."""

    def test_combined_sorted_by_started_at(self) -> None:
        from datetime import datetime, timezone

        from fastapi.testclient import TestClient

        from rlmstudio.server.app import app
        from rlmstudio.server.dependencies import ExecutionRecord, get_state, reset_state

        reset_state()
        state = get_state()

        # Persist an older completed run via telemetry store (epoch=100)
        state.telemetry.record_run(
            run_id="old-persisted",
            created_at=100.0,
            mode="direct",
            query="old q",
            success=True,
        )

        # Add a newer in-memory running execution (epoch=200)
        mid_dt = datetime.fromtimestamp(200.0, tz=timezone.utc)
        state.executions["mid-in-memory"] = ExecutionRecord(
            execution_id="mid-in-memory",
            session_id="s1",
            query="mid q",
            mode="direct",
            status="running",
            started_at=mid_dt,
        )

        # Persist the newest completed run via telemetry store (epoch=300)
        state.telemetry.record_run(
            run_id="new-persisted",
            created_at=300.0,
            mode="rlm",
            query="new q",
            success=True,
        )

        with TestClient(app) as client:
            resp = client.get("/api/executions?limit=10")
            assert resp.status_code == 200
            data = resp.json()

        ids_in_order = [row["execution_id"] for row in data]
        # Newest-first across both sources
        assert ids_in_order[:3] == ["new-persisted", "mid-in-memory", "old-persisted"]

    def test_limit_respected_after_merge(self) -> None:
        from fastapi.testclient import TestClient

        from rlmstudio.server.app import app
        from rlmstudio.server.dependencies import get_state, reset_state

        reset_state()
        state = get_state()

        for i in range(5):
            state.telemetry.record_run(
                run_id=f"r{i}",
                created_at=float(i),
                mode="direct",
                query=f"q{i}",
            )

        with TestClient(app) as client:
            resp = client.get("/api/executions?limit=3")
            assert resp.status_code == 200
            data = resp.json()

        assert len(data) == 3
        # Newest first: r4, r3, r2
        assert [row["execution_id"] for row in data] == ["r4", "r3", "r2"]


class TestGetTraceFromTelemetry:
    """GET /api/traces/{id} served from the telemetry-backed path."""

    def test_persisted_steps_return_canonical_action_types(self) -> None:
        """End-to-end: when a run is only in the telemetry store (not in
        state.executions), /api/traces/{id} must still return canonical
        action types ('inspect', 'subcall', 'final'), matching in-memory
        traces and JSONL exports."""
        from datetime import datetime, timezone

        from fastapi.testclient import TestClient

        from rlmstudio.application.dto import RunResultDTO
        from rlmstudio.server.app import app
        from rlmstudio.server.dependencies import ExecutionRecord, get_state, reset_state
        from rlmstudio.server.routes.chat import _record_telemetry

        reset_state()
        state = get_state()

        execution = ExecutionRecord(
            execution_id="persisted-only-1",
            session_id="sess-1",
            query="what is in the doc?",
            mode="rlm",
            status="complete",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )
        result = RunResultDTO(
            answer="found it",
            mode_used="rlm",
            success=True,
            steps=4,
            input_tokens=100,
            output_tokens=50,
            total_cost=0.003,
            elapsed_time=1.2,
            trace=[
                {"role": "assistant", "code": "print(len(P))", "model": "gpt-4o"},
                {"role": "execution", "content": "8321", "model": "gpt-4o"},
                {"role": "assistant", "code": "print(P[:80])", "model": "gpt-4o"},
                {"role": "assistant", "content": "The answer is ...", "model": "gpt-4o"},
            ],
        )

        # Write through the production helper, then DO NOT register the
        # execution in state.executions — this forces the route to serve
        # from the telemetry store.
        _record_telemetry(
            state,
            execution,
            result,
            provider="openai",
            model="gpt-4o",
        )
        assert "persisted-only-1" not in state.executions

        with TestClient(app) as client:
            resp = client.get("/api/traces/persisted-only-1")
            assert resp.status_code == 200
            data = resp.json()

        # Canonical action types match the in-memory / JSONL schema
        action_types = [step["action_type"] for step in data["steps"]]
        assert action_types == ["inspect", "subcall", "inspect", "final"]

        # Structured provider/model round-tripped through the store
        assert data["result"]["answer"] == "found it"
        assert data["result"]["success"] is True
        assert data["mode"] == "rlm"
        assert data["status"] == "complete"

    def test_persisted_run_not_found(self) -> None:
        """Unknown execution_id returns 404 from the telemetry-backed path."""
        from fastapi.testclient import TestClient

        from rlmstudio.server.app import app
        from rlmstudio.server.dependencies import reset_state

        reset_state()

        with TestClient(app) as client:
            resp = client.get("/api/traces/does-not-exist")
            assert resp.status_code == 404
