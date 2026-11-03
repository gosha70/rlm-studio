"""Loop-level tests for controller-dispatched single and batched subcalls.

Covers specs/rlm-working-state AC-6 (budget shares and concurrency cap),
AC-6a (the subprocess sandbox is never invoked), AC-7 (the batch is one
registered, readable, compactable result), and sync/async parity.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

from rlmstudio.application.dto import LLMResponseDTO, RunConfigDTO, RunResultDTO
from rlmstudio.application.sandbox_vars import (
    TRACE_KEY_IDS_STUBBED,
    TRACE_KEY_READBACK_OF,
    TRACE_KEY_RESULT_ID,
    TRACE_KEY_ROLE,
    TRACE_ROLE_COMPACTION,
)
from rlmstudio.application.use_cases.run_rlm import RunRLMUseCase
from rlmstudio.infrastructure.sandbox.subprocess_sandbox import SubprocessSandboxAdapter
from tests.test_rlm_working_state_loop import _semantic
from tests.test_use_cases import FakeSandbox


def _batch(*queries: str, max_concurrency: int | None = None) -> str:
    return json.dumps(
        {
            "type": "subcall",
            "calls": [{"prompt": f"content for {query}", "query": query} for query in queries],
            "max_concurrency": max_concurrency or len(queries),
        }
    )


def _single(query: str) -> str:
    return json.dumps({"type": "subcall", "prompt": f"content for {query}", "query": query})


def _final(answer: str) -> str:
    return json.dumps({"type": "final", "answer": answer})


def _inspect(tool: str, **args: Any) -> str:
    return json.dumps({"type": "inspect", "tool": tool, "args": args})


class BatchRoutingLLM:
    """Thread-safe fake that routes each child by its query.

    ``copy.copy`` shares ``_state``, so every child adapter draws from the
    same response stream and concurrency counters.  Child runs (whose query
    starts with ``child-``) always answer with a final derived from the
    query; the parent consumes ``parent_responses`` in order.
    """

    def __init__(
        self,
        parent_responses: list[str],
        *,
        context_window: int | None = None,
        fail_queries: set[str] | None = None,
        delay: float = 0.02,
        pricing: dict[str, float] | None = None,
    ) -> None:
        self._parent_responses = list(parent_responses)
        self._fail_queries = fail_queries or set()
        self._delay = delay
        self._pricing = pricing or {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}
        self.context_window = context_window
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "parent_idx": 0,
            "active": 0,
            "peak": 0,
            "calls": 0,
            "seen": [],
        }

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        query = messages[1]["content"] if len(messages) > 1 else ""
        is_child = query.startswith("child-")
        with self._lock:
            self._state["active"] += 1
            self._state["peak"] = max(self._state["peak"], self._state["active"])
            self._state["seen"].append(list(messages))
        try:
            if self._delay:
                time.sleep(self._delay)
            if query in self._fail_queries:
                raise RuntimeError("provider down")
            if is_child:
                text = _final(f"answer {query}")
            else:
                with self._lock:
                    index = min(self._state["parent_idx"], len(self._parent_responses) - 1)
                    self._state["parent_idx"] += 1
                text = self._parent_responses[index]
            return LLMResponseDTO(
                content=text,
                model="fake",
                input_tokens=10,
                output_tokens=5,
            )
        finally:
            with self._lock:
                self._state["active"] -= 1
                self._state["calls"] += 1

    def count_tokens(
        self,
        text: str | None = None,
        *,
        messages: list[dict[str, str]] | None = None,
    ) -> int:
        if messages is not None:
            return max(1, sum(len(m.get("content", "")) for m in messages) // 4 + 10)
        return max(1, len(text or "") // 4)

    def get_pricing(self) -> dict[str, float]:
        return self._pricing

    @property
    def peak(self) -> int:
        return int(self._state["peak"])

    @property
    def calls(self) -> int:
        return int(self._state["calls"])

    @property
    def seen(self) -> list[list[dict[str, str]]]:
        return list(self._state["seen"])


def _execution_rows(result: RunResultDTO) -> list[dict[str, Any]]:
    return [t for t in result.trace if t.get(TRACE_KEY_ROLE) == "execution"]


def _run_async(coro):
    """Run one async test coroutine and leave an event loop for later tests.

    The existing suite uses ``asyncio.get_event_loop().run_until_complete``;
    ``asyncio.run`` clears that loop, which makes later tests fail when this
    file shares an xdist worker with them.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


class TestBatchedSubcalls:
    def test_runs_concurrently_and_keeps_input_order(self) -> None:
        llm = BatchRoutingLLM(
            [_batch("child-0", "child-1", "child-2", "child-3"), _final("parent done")],
            delay=0.03,
        )
        config = RunConfigDTO(
            mode="rlm",
            max_steps=6,
            max_recursion_depth=1,
            subcall_max_concurrency=4,
        )
        result = RunRLMUseCase(llm, FakeSandbox()).execute("doc", "batch them", config)

        assert result.success
        assert result.answer == "parent done"
        assert result.steps == 6
        assert llm.peak >= 2

        rows = _execution_rows(result)
        assert rows[0][TRACE_KEY_RESULT_ID] == "r1"
        content = rows[0].get("content", "")
        positions = [content.index(f"answer child-{i}") for i in range(4)]
        assert positions == sorted(positions)

    def test_action_concurrency_is_capped_by_config(self) -> None:
        llm = BatchRoutingLLM(
            [_batch("child-0", "child-1", "child-2", "child-3", max_concurrency=8), _final("done")],
            delay=0.03,
        )
        config = RunConfigDTO(
            mode="rlm",
            max_steps=8,
            max_recursion_depth=1,
            subcall_max_concurrency=2,
        )
        result = RunRLMUseCase(llm, FakeSandbox()).execute("doc", "batch them", config)

        assert result.success
        assert llm.peak <= 2

    def test_result_is_readable_through_read_result(self) -> None:
        llm = BatchRoutingLLM(
            [
                _batch("child-0", "child-1"),
                _inspect("read_result", name="last", start=0, end=10000),
                _final("done"),
            ]
        )
        config = RunConfigDTO(
            mode="rlm",
            max_steps=6,
            max_recursion_depth=1,
            subcall_max_concurrency=2,
        )
        result = RunRLMUseCase(llm, FakeSandbox()).execute("doc", "batch them", config)

        assert result.success
        rows = _execution_rows(result)
        assert rows[0][TRACE_KEY_RESULT_ID] == "r1"
        assert rows[1][TRACE_KEY_READBACK_OF] == "r1"
        assert "answer child-0" in rows[1]["content"]
        assert "answer child-1" in rows[1]["content"]

    def test_child_failure_is_isolated(self) -> None:
        llm = BatchRoutingLLM(
            [_batch("child-0", "child-1", "child-2"), _final("parent done")],
            fail_queries={"child-1"},
        )
        config = RunConfigDTO(
            mode="rlm",
            max_steps=8,
            max_recursion_depth=1,
            subcall_max_concurrency=3,
        )
        result = RunRLMUseCase(llm, FakeSandbox()).execute("doc", "batch them", config)

        assert result.success
        assert result.answer == "parent done"
        content = _execution_rows(result)[0]["content"]
        assert "answer child-0" in content
        assert "answer child-2" in content
        assert "Error in subcall" in content

    def test_batched_result_is_stubbed_by_compaction(self) -> None:
        llm = BatchRoutingLLM(
            [
                _batch("child-0", "child-1"),
                _inspect("peek", start=0, end=20),
                _final("done"),
            ],
            context_window=100,
            delay=0.0,
        )
        config = RunConfigDTO(
            mode="rlm",
            max_steps=7,
            max_recursion_depth=1,
            subcall_max_concurrency=2,
            compaction_keep_last=0,
            compaction_reserve_tokens=0,
        )
        result = RunRLMUseCase(llm, FakeSandbox()).execute("doc", "batch them", config)

        assert result.success
        compaction = [t for t in result.trace if t.get(TRACE_KEY_ROLE) == TRACE_ROLE_COMPACTION]
        assert compaction
        assert "r1" in compaction[0][TRACE_KEY_IDS_STUBBED]
        assert any(
            message["content"].startswith("[result r1:") for call in llm.seen for message in call
        )

    def test_async_matches_sync(self) -> None:
        script = [_batch("child-0", "child-1"), _final("done")]
        config = RunConfigDTO(
            mode="rlm",
            max_steps=5,
            max_recursion_depth=1,
            subcall_max_concurrency=2,
        )
        sync_result = RunRLMUseCase(
            BatchRoutingLLM(list(script), delay=0.0), FakeSandbox()
        ).execute("doc", "batch them", config)
        async_result = _run_async(
            RunRLMUseCase(BatchRoutingLLM(list(script), delay=0.0), FakeSandbox()).execute_async(
                "doc", "batch them", config
            )
        )

        assert sync_result.answer == async_result.answer == "done"
        assert sync_result.steps == async_result.steps
        assert sync_result.input_tokens == async_result.input_tokens
        assert _semantic(sync_result.trace) == _semantic(async_result.trace)


class TestSubcallBudgets:
    def test_token_share_keeps_the_parent_within_budget(self) -> None:
        llm = BatchRoutingLLM(
            [_batch("child-0", "child-1", "child-2", "child-3"), _final("done")],
            delay=0.0,
        )
        config = RunConfigDTO(
            mode="rlm",
            max_steps=6,
            max_tokens=100,
            max_recursion_depth=1,
            subcall_max_concurrency=4,
        )
        result = RunRLMUseCase(llm, FakeSandbox()).execute("doc", "batch them", config)

        assert result.success
        assert result.input_tokens + result.output_tokens <= 100

    def test_each_child_gets_an_equal_budget_share(self, monkeypatch) -> None:
        child_configs: list[RunConfigDTO] = []
        original_execute = RunRLMUseCase.execute
        calls = [0]

        def tracking_execute(
            inner: RunRLMUseCase,
            content: str,
            query: str,
            config: RunConfigDTO | None = None,
        ) -> RunResultDTO:
            calls[0] += 1
            if calls[0] > 1 and config is not None:
                child_configs.append(config)
            return original_execute(inner, content, query, config=config)

        monkeypatch.setattr(RunRLMUseCase, "execute", tracking_execute)
        llm = BatchRoutingLLM(
            [_batch("child-0", "child-1", "child-2", "child-3"), _final("done")],
            delay=0.0,
        )
        config = RunConfigDTO(
            mode="rlm",
            max_steps=6,
            max_tokens=100,
            max_cost=0.1,
            max_recursion_depth=1,
            subcall_max_concurrency=4,
        )
        result = RunRLMUseCase(llm, FakeSandbox()).execute("doc", "batch them", config)

        assert result.success
        assert len(child_configs) == 4
        # Parent used step 1 and 15 tokens before reserving; cost pricing is 0.
        assert all(cfg.max_steps == 1 for cfg in child_configs)
        assert all(cfg.max_tokens == 21 for cfg in child_configs)
        assert all(abs((cfg.max_cost or 0.0) - 0.025) < 1e-9 for cfg in child_configs)

    def test_children_share_the_absolute_deadline(self, monkeypatch) -> None:
        child_configs: list[RunConfigDTO] = []
        original_execute = RunRLMUseCase.execute
        calls = [0]

        def tracking_execute(
            inner: RunRLMUseCase,
            content: str,
            query: str,
            config: RunConfigDTO | None = None,
        ) -> RunResultDTO:
            calls[0] += 1
            if calls[0] > 1 and config is not None:
                child_configs.append(config)
            return original_execute(inner, content, query, config=config)

        monkeypatch.setattr(RunRLMUseCase, "execute", tracking_execute)

        llm = BatchRoutingLLM([_batch("child-0", "child-1"), _final("done")], delay=0.0)
        config = RunConfigDTO(
            mode="rlm",
            max_steps=5,
            max_recursion_depth=1,
            max_time_seconds=10.0,
            subcall_max_concurrency=2,
        )
        result = RunRLMUseCase(llm, FakeSandbox()).execute("doc", "batch them", config)

        assert result.success
        assert len(child_configs) == 2
        times = [cfg.max_time_seconds for cfg in child_configs]
        assert all(t is not None and t > 9.0 for t in times)
        numeric = [t for t in times if t is not None]
        # The ledger gives every child the common deadline, not 10/2 each.
        assert max(numeric) - min(numeric) < 0.5

    def test_cost_share_keeps_the_parent_within_budget(self) -> None:
        llm = BatchRoutingLLM(
            [_batch("child-0", "child-1", "child-2", "child-3"), _final("done")],
            delay=0.0,
            pricing={"input_cost_per_1m": 1000.0, "output_cost_per_1m": 1000.0},
        )
        config = RunConfigDTO(
            mode="rlm",
            max_steps=6,
            max_cost=0.1,
            max_recursion_depth=1,
            subcall_max_concurrency=4,
        )
        result = RunRLMUseCase(llm, FakeSandbox()).execute("doc", "batch them", config)

        assert result.success
        assert result.total_cost <= 0.1


class TestSubprocessSandbox:
    def test_batch_never_invokes_the_subprocess_sandbox(self) -> None:
        llm = BatchRoutingLLM([_batch("child-0", "child-1"), _final("done")], delay=0.0)
        sandbox = SubprocessSandboxAdapter(max_exec_time_s=5.0)
        result = RunRLMUseCase(llm, sandbox).execute(
            "doc",
            "batch them",
            RunConfigDTO(mode="rlm", max_steps=5, max_recursion_depth=1),
        )
        assert result.success
        assert result.answer == "done"

    def test_single_call_never_invokes_the_subprocess_sandbox(self) -> None:
        llm = BatchRoutingLLM([_single("child-0"), _final("done")], delay=0.0)
        sandbox = SubprocessSandboxAdapter(max_exec_time_s=5.0)
        result = RunRLMUseCase(llm, sandbox).execute(
            "doc",
            "one call",
            RunConfigDTO(mode="rlm", max_steps=4, max_recursion_depth=1),
        )
        assert result.success
        assert result.answer == "done"
