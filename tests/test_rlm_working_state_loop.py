"""Loop-level tests for the result registry in ``RunRLMUseCase``.

Covers specs/rlm-working-state AC-1 (readback across steps, stable id),
AC-2 (contiguous ids across nudges, repeats and auto-inspect), AC-3 (spill
inside a run, scratch cleanup) and the sync/async parity check, using the
same fakes as ``tests/test_use_cases.py``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from rlmstudio.application.dto import RunConfigDTO, RunResultDTO
from rlmstudio.application.sandbox_vars import (
    TRACE_KEY_CODE,
    TRACE_KEY_CONTENT,
    TRACE_KEY_RESULT_CHARS,
    TRACE_KEY_RESULT_ID,
    TRACE_KEY_ROLE,
    TRACE_KEY_SEQ,
    TRACE_KEY_STEP,
)
from rlmstudio.application.services.result_registry import ResultRef, ResultRegistry
from rlmstudio.application.use_cases.run_rlm import RunRLMUseCase
from rlmstudio.tools import grep
from tests.test_use_cases import CapturingLLM, FakeLLM, FakeSandbox

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BIG_CONTENT = "\n".join(f"needle line {i:05d} with some filler text" for i in range(1500))
"""grep('needle') over this yields well over 10,000 chars of output."""

TWO_FILES = (
    "[DOCUMENT INDEX — 2 files attached]\n"
    "Read this index first with peek(0, 1200).\n"
    "Each file entry includes exact character offsets:\n"
    '  1. "alpha.txt" (file_start=300, content_start=320, file_end_exclusive=400)\n'
    '  2. "beta.txt" (file_start=405, content_start=424, file_end_exclusive=500)\n'
    "[END DOCUMENT INDEX]\n\n"
    "[File 1: alpha.txt]\n"
    "Alpha Title\nChapter 1\nAlpha body text.\n"
    "---\n"
    "[File 2: beta.txt]\n"
    "Beta Title\nChapter 1\nBeta body text.\n"
)


def _inspect(tool: str, **args: Any) -> str:
    return json.dumps({"type": "inspect", "tool": tool, "args": args})


def _final(answer: str) -> str:
    return json.dumps({"type": "final", "answer": answer})


def _execution_rows(result: RunResultDTO) -> list[dict[str, Any]]:
    return [t for t in result.trace if t.get(TRACE_KEY_ROLE) == "execution"]


def _exec_messages(llm: CapturingLLM) -> list[str]:
    """Every 'Execution result:' user message the model was shown, in order."""
    seen: list[str] = []
    for call in llm.call_messages:
        for m in call:
            if m["role"] == "user" and m["content"].startswith("Execution result:"):
                if m["content"] not in seen:
                    seen.append(m["content"])
    return seen


# ---------------------------------------------------------------------------
# AC-1: a large result is fully readable across read_result calls on `last`
# ---------------------------------------------------------------------------


class TestReadbackAcrossSteps:
    def test_grep_over_preview_cap_is_readable_and_id_is_stable(self) -> None:
        llm = CapturingLLM(
            [
                _inspect("grep", pattern="needle", context_lines=0, max_matches=2000),
                _inspect("read_result", name="last", start=10000, end=20000),
                _inspect("read_result", name="last", start=20000, end=30000),
                _inspect("peek", start=0, end=50),
                _final("done"),
            ]
        )
        uc = RunRLMUseCase(llm, FakeSandbox())
        result = uc.execute(BIG_CONTENT, "find the needles", RunConfigDTO(mode="rlm", max_steps=8))
        assert result.success
        assert result.answer == "done"

        full = grep(BIG_CONTENT, pattern="needle", context_lines=0, max_matches=2000)
        assert len(full) > 10000

        rows = _execution_rows(result)
        assert rows[0][TRACE_KEY_RESULT_ID] == "r1"
        assert rows[0][TRACE_KEY_RESULT_CHARS] == len(full)
        # Readbacks are not registered and do not move `last`.
        assert TRACE_KEY_RESULT_ID not in rows[1]
        assert TRACE_KEY_RESULT_ID not in rows[2]
        assert rows[3][TRACE_KEY_RESULT_ID] == "r2"

        shown = _exec_messages(llm)
        # The first preview ends in a marker naming r1 and the exact offset.
        assert "read_result('r1', start=" in shown[0]
        # Each readback returned the requested slice of the same result.
        assert full[10000:10100] in shown[1]
        assert full[20000:20100] in shown[2]
        # The readback's own continuation footer survives the message bound
        # and points at the exact next offset.
        assert f"of {len(full):,} in r1; read_result('r1', start=" in shown[1]
        assert rows[1]["readback_of"] == "r1"

    def test_readback_of_unknown_id_is_an_execution_error_not_a_crash(self) -> None:
        llm = FakeLLM([_inspect("read_result", name="r7"), _final("ok")])
        result = RunRLMUseCase(llm, FakeSandbox()).execute(
            BIG_CONTENT, "q", RunConfigDTO(mode="rlm", max_steps=4)
        )
        assert result.success
        rows = _execution_rows(result)
        assert "unknown result 'r7'" in rows[0][TRACE_KEY_CONTENT]


# ---------------------------------------------------------------------------
# AC-2: ids stay contiguous across nudges, repeats and coverage auto-inspects
# ---------------------------------------------------------------------------


class TestContiguousIds:
    def test_ids_contiguous_across_nudge_repeat_and_auto_inspect(self) -> None:
        llm = FakeLLM(
            [
                _inspect("outline_file", file_no=1, max_lines=5),
                _inspect("outline_file", file_no=1, max_lines=5),  # repeat → nudge
                _final("summary of both"),  # coverage gate → auto-inspect file 2
                _final("summary of both"),
            ]
        )
        config = RunConfigDTO(mode="rlm", max_steps=8, nudge_at_fraction=0.3)
        result = RunRLMUseCase(llm, FakeSandbox()).execute(
            TWO_FILES, "summarize both documents", config
        )
        assert result.success
        assert result.answer == "summary of both"
        rows = _execution_rows(result)
        ids = [r[TRACE_KEY_RESULT_ID] for r in rows]
        assert ids == ["r1", "r2", "r3"]
        assert rows[2].get("note") == "auto coverage inspect for file 2"
        # Steps advanced past the ids (nudge and repeat turns), so ids are
        # not step numbers.
        assert result.steps > 3


# ---------------------------------------------------------------------------
# AC-3: spill inside a run, exact reads after spill, scratch cleanup
# ---------------------------------------------------------------------------


class TestSpillInsideRun:
    def test_spilled_result_reads_exactly_and_scratch_is_removed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        peak = 0

        def account(in_mem: int, _spilled: int) -> None:
            nonlocal peak
            peak = max(peak, in_mem)

        def new_registry(config: RunConfigDTO) -> ResultRegistry:
            return ResultRegistry(
                spill_result_above_bytes=config.spill_result_above_bytes,
                max_registry_bytes=config.max_registry_bytes,
                max_spill_bytes=config.max_spill_bytes,
                scratch_dir=tmp_path,
                on_account=account,
            )

        monkeypatch.setattr(RunRLMUseCase, "_new_registry", staticmethod(new_registry))
        llm = CapturingLLM(
            [
                _inspect("grep", pattern="needle", context_lines=0, max_matches=2000),
                _inspect("read_result", name="r1", start=30000, end=30200),
                _final("ok"),
            ]
        )
        config = RunConfigDTO(
            mode="rlm",
            max_steps=5,
            spill_result_above_bytes=4096,
            max_registry_bytes=8192,
        )
        result = RunRLMUseCase(llm, FakeSandbox()).execute(BIG_CONTENT, "q", config)
        assert result.success
        full = grep(BIG_CONTENT, pattern="needle", context_lines=0, max_matches=2000)
        shown = _exec_messages(llm)
        assert full[30000:30200] in shown[1]
        assert peak <= 8192
        assert not list(tmp_path.iterdir())  # scratch removed at run end

    def test_scratch_removed_when_the_run_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def new_registry(config: RunConfigDTO) -> ResultRegistry:
            return ResultRegistry(spill_result_above_bytes=1, scratch_dir=tmp_path)

        monkeypatch.setattr(RunRLMUseCase, "_new_registry", staticmethod(new_registry))

        class ExplodingLLM(FakeLLM):
            def complete(self, messages: list[dict[str, str]]) -> Any:
                if self._idx == 1:
                    raise RuntimeError("provider down")
                return super().complete(messages)

        llm = ExplodingLLM([_inspect("peek", start=0, end=100), _final("never")])
        result = RunRLMUseCase(llm, FakeSandbox()).execute(
            BIG_CONTENT, "q", RunConfigDTO(mode="rlm", max_steps=4)
        )
        assert not result.success
        assert not list(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# Sync / async parity on a normalised semantic trace
# ---------------------------------------------------------------------------

_SEMANTIC_KEYS = (
    TRACE_KEY_STEP,
    TRACE_KEY_SEQ,
    TRACE_KEY_ROLE,
    TRACE_KEY_CODE,
    TRACE_KEY_CONTENT,
    TRACE_KEY_RESULT_ID,
    TRACE_KEY_RESULT_CHARS,
    "note",
)


def _semantic(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in trace:
        if row.get(TRACE_KEY_CONTENT) == "Runtime fingerprint":
            continue  # carries execution_path, which differs by design
        out.append({k: row[k] for k in _SEMANTIC_KEYS if k in row})
    return out


class TestSyncAsyncParity:
    def test_same_script_yields_same_semantic_trace(self) -> None:
        script = [
            _inspect("grep", pattern="needle", context_lines=0, max_matches=2000),
            _inspect("read_result", name="last", start=10000, end=12000),
            _inspect("peek", start=0, end=40),
            _final("parity"),
        ]
        config = RunConfigDTO(mode="rlm", max_steps=6)
        sync_result = RunRLMUseCase(FakeLLM(list(script)), FakeSandbox()).execute(
            BIG_CONTENT, "q", config
        )
        async_result = asyncio.run(
            RunRLMUseCase(FakeLLM(list(script)), FakeSandbox()).execute_async(
                BIG_CONTENT, "q", config
            )
        )
        assert sync_result.answer == async_result.answer == "parity"
        assert sync_result.steps == async_result.steps
        assert sync_result.input_tokens == async_result.input_tokens
        assert _semantic(sync_result.trace) == _semantic(async_result.trace)
        assert any(TRACE_KEY_RESULT_ID in row for row in sync_result.trace)


# ---------------------------------------------------------------------------
# Message-side bound keeps the registry marker
# ---------------------------------------------------------------------------


class TestBoundExecContent:
    def _ref(self, length: int) -> ResultRef:
        return ResultRef(id="r9", kind="inspect", step=1, length=length, byte_length=length)

    def test_unregistered_output_uses_generic_note(self) -> None:
        out = RunRLMUseCase._bound_exec_content("Output:\n" + "x" * 100, 50, None, 10000)
        assert out.endswith("[truncated, 58 chars omitted]")

    def test_registered_output_cut_ends_in_marker_with_exact_offset(self) -> None:
        formatted = "Output:\n" + "x" * 100
        out = RunRLMUseCase._bound_exec_content(formatted, 58, self._ref(100), 10000)
        assert out.startswith("Output:\n" + "x" * 50)
        assert out.endswith("read_result('r9', start=50) for more)")

    def test_cut_that_would_only_clip_the_marker_keeps_it_intact(self) -> None:
        preview = (
            "x" * 100
            + "\n... (preview truncated: 300 chars total; read_result('r9', start=100) for more)"
        )
        formatted = "Output:\n" + preview
        out = RunRLMUseCase._bound_exec_content(formatted, 120, self._ref(300), 100)
        assert out == formatted
