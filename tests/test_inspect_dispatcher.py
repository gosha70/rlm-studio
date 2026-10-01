"""Tests for controller-side inspect dispatch (specs/rlm-working-state, plan D2)."""

from __future__ import annotations

import time

import pytest

from rlmstudio.application.services.inspect_dispatcher import (
    InspectOutcome,
    dispatch_inspect,
    display_code,
)
from rlmstudio.application.services.result_registry import ResultRegistry
from rlmstudio.core.actions import InspectAction, ParseError, parse_action
from rlmstudio.infrastructure.sandbox.restricted_sandbox import RestrictedSandboxAdapter

CONTENT = (
    "[DOCUMENT INDEX — 2 files attached]\n"
    "Read this index first with peek(0, 1200).\n"
    "Each file entry includes exact character offsets:\n"
    '  1. "alpha.txt" (file_start=0, content_start=0, file_end_exclusive=0)\n'
    '  2. "beta.txt" (file_start=0, content_start=0, file_end_exclusive=0)\n'
    "[END DOCUMENT INDEX]\n\n"
    "[File 1: alpha.txt]\n"
    "Chapter One\nAlpha needle here.\nMore alpha text.\n"
    "---\n"
    "[File 2: beta.txt]\n"
    "Chapter Two\nBeta needle here.\nMore beta text.\n"
)

PLAIN = "\n".join(f"line {i}: needle" if i % 7 == 0 else f"line {i}: filler" for i in range(400))


def _action(tool: str, **args: object) -> InspectAction:
    return InspectAction(tool=tool, args=dict(args))


class TestDispatchMatchesSandbox:
    """Each tool through the dispatcher equals the same call inside the sandbox."""

    @pytest.mark.parametrize(
        "action",
        [
            _action("peek", start=0, end=120),
            _action("peek", start=-30),
            _action("grep", pattern="needle", context_lines=1),
            _action("grep", pattern="NEEDLE", ignore_case=True, max_matches=3),
            _action("grep", pattern=r"line \d+0:", use_regex=True),
            _action("select", ranges=[[0, 20], [100, 130]]),
            _action("chunk", size=500, overlap=50, max_chunks=4),
        ],
        ids=lambda a: display_code(a),
    )
    def test_plain_content_tools(self, action: InspectAction) -> None:
        outcome = dispatch_inspect(PLAIN, action)
        assert outcome.error is None, outcome.error
        sandbox = RestrictedSandboxAdapter(max_stdout_chars=10_000_000)
        sandbox.set_variable("P", PLAIN)
        result = sandbox.execute(display_code(action))
        assert result.exception is None, result.exception
        if action.tool == "chunk":
            # The sandbox printed the Python repr; the dispatcher keeps the list.
            assert isinstance(outcome.value, list)
            assert result.stdout.strip() == repr(outcome.value)
        else:
            assert result.stdout.strip() == str(outcome.value).strip()

    @pytest.mark.parametrize(
        "action",
        [
            _action("peek_file", file_no=2, start=0, end=40),
            _action("grep_file", file_no=1, pattern="needle"),
            _action("outline_file", file_no=2, max_lines=5),
        ],
        ids=lambda a: display_code(a),
    )
    def test_multi_file_tools(self, action: InspectAction) -> None:
        outcome = dispatch_inspect(CONTENT, action)
        assert outcome.error is None, outcome.error
        sandbox = RestrictedSandboxAdapter(max_stdout_chars=10_000_000)
        sandbox.set_variable("P", CONTENT)
        result = sandbox.execute(display_code(action))
        assert result.exception is None, result.exception
        assert result.stdout.strip() == str(outcome.value).strip()


class TestErrors:
    def test_missing_required_arg_is_error_text(self) -> None:
        out = dispatch_inspect(PLAIN, _action("grep"))
        assert out.value is None
        assert out.error is not None
        assert "'pattern' is required" in out.error

    def test_wrong_type_is_error_text(self) -> None:
        out = dispatch_inspect(PLAIN, _action("peek", start="ten"))
        assert out.error is not None
        assert "'start' must be an integer" in out.error

    def test_tool_failure_is_error_text(self) -> None:
        out = dispatch_inspect(PLAIN, _action("chunk", size=0))
        assert out.error is not None
        assert out.error.startswith("Error: chunk() failed")

    def test_bad_ranges(self) -> None:
        out = dispatch_inspect(PLAIN, _action("select", ranges=[[1]]))
        assert out.error is not None
        assert "[start, end] pair" in out.error

    def test_bool_coercion(self) -> None:
        out = dispatch_inspect(PLAIN, _action("grep", pattern="needle", ignore_case="true"))
        assert out.error is None
        out = dispatch_inspect(PLAIN, _action("grep", pattern="needle", ignore_case="yes"))
        assert out.error is not None
        assert "must be a boolean" in out.error


class TestReadResult:
    def test_readback_is_not_registered_and_defaults_to_last(self) -> None:
        with ResultRegistry() as reg:
            reg.register("0123456789" * 3, kind="inspect", step=1)
            out = dispatch_inspect(PLAIN, _action("read_result", start=5, end=12), reg)
            assert isinstance(out, InspectOutcome)
            assert out.is_readback is True
            assert out.error is None
            assert str(out.value).startswith("5678901")
            assert reg.ids() == ["r1"]

    def test_readback_without_registry_is_error(self) -> None:
        out = dispatch_inspect(PLAIN, _action("read_result"))
        assert out.error is not None
        assert "not available" in out.error

    def test_display_code_for_readback(self) -> None:
        code = display_code(_action("read_result", name="r4", start=10, end=20))
        assert code == "read_result(name='r4', start=10, end=20, max_chars=10000)"


class TestActionSchema:
    def test_read_result_parses(self) -> None:
        kind, action = parse_action(
            '{"type": "inspect", "tool": "read_result", "args": {"name": "last", "start": 10}}'
        )
        assert kind == "inspect"
        assert action.tool == "read_result"

    @pytest.mark.parametrize(
        "args",
        [
            '{"name": ""}',
            '{"name": 3}',
            '{"start": "ten"}',
            '{"start": true}',
            '{"max_chars": 0}',
        ],
    )
    def test_read_result_rejects_bad_args(self, args: str) -> None:
        with pytest.raises(ParseError, match="Invalid inspect action"):
            parse_action(f'{{"type": "inspect", "tool": "read_result", "args": {args}}}')

    def test_unknown_tool_lists_read_result(self) -> None:
        with pytest.raises(ParseError, match="read_result"):
            parse_action('{"type": "inspect", "tool": "nope", "args": {}}')


class TestPatternBudget:
    """A model-written pattern cannot run unbounded in the controller process.

    These tools used to execute inside the sandbox, which cut any call off
    after its own timeout.  Running them here removed that limit, and a
    nested quantifier backtracks exponentially in the length of a single
    line — 40 characters is enough to run for hours — while `re` holds the
    GIL, so neither a signal nor a watchdog thread can interrupt it.
    """

    # Backtracks in the `regex` engine too, so the deadline is what stops it.
    CATASTROPHIC = r"(a|aa)+$"
    BUDGET = 0.5
    LONG_RUN = "intro line\n" + "a" * 60 + "b\n" + "tail line\n"

    def test_a_backtracking_pattern_is_cut_off_at_the_budget(self) -> None:
        started = time.monotonic()
        outcome = dispatch_inspect(
            self.LONG_RUN,
            _action("grep", pattern=self.CATASTROPHIC, use_regex=True),
            None,
            pattern_timeout=self.BUDGET,
        )
        elapsed = time.monotonic() - started

        assert outcome.error is not None
        assert "timed out" in outcome.error
        assert elapsed < self.BUDGET * 10  # not "eventually": promptly

    def test_the_error_tells_the_model_what_to_do(self) -> None:
        outcome = dispatch_inspect(
            self.LONG_RUN,
            _action("grep", pattern=self.CATASTROPHIC, use_regex=True),
            None,
            pattern_timeout=self.BUDGET,
        )

        assert outcome.error is not None
        assert "simplify" in outcome.error
        assert outcome.value is None

    def test_the_budget_covers_the_whole_call_not_each_line(self) -> None:
        """Many bad lines must not multiply the budget by the line count."""
        many = "".join("a" * 60 + "b\n" for _ in range(50))
        started = time.monotonic()
        dispatch_inspect(
            many,
            _action("grep", pattern=self.CATASTROPHIC, use_regex=True),
            None,
            pattern_timeout=self.BUDGET,
        )
        elapsed = time.monotonic() - started

        assert elapsed < self.BUDGET * 10

    def test_an_ordinary_regex_still_matches(self) -> None:
        outcome = dispatch_inspect(PLAIN, _action("grep", pattern=r"needle", use_regex=True), None)

        assert outcome.error is None
        assert "needle" in str(outcome.value)

    def test_a_literal_search_still_matches(self) -> None:
        outcome = dispatch_inspect(PLAIN, _action("grep", pattern="needle"), None)

        assert outcome.error is None
        assert "needle" in str(outcome.value)

    def test_an_invalid_pattern_is_reported_the_same_way_as_before(self) -> None:
        outcome = dispatch_inspect(
            PLAIN, _action("grep", pattern="(unclosed", use_regex=True), None
        )

        assert outcome.error is None  # an ordinary result, not a step failure
        assert "Invalid regex pattern" in str(outcome.value)


class TestReadResultIsBounded:
    """``read_result`` output lands verbatim in the stored, streamed trace row.

    A stored result can be hundreds of megabytes, so the per-step cap is what
    keeps one readback from writing all of it into the execution record.
    """

    def test_an_oversized_request_is_clamped_to_the_preview_cap(self) -> None:
        registry = ResultRegistry()
        ref = registry.register("x" * 200_000, kind="inspect", step=1)

        outcome = dispatch_inspect(
            "doc",
            _action("read_result", name=ref.id, max_chars=50_000_000),
            registry,
            preview_chars=10_000,
        )

        assert outcome.error is None
        body = str(outcome.value)
        assert 10_000 <= len(body) < 11_000  # the cap plus the continuation footer

    def test_a_smaller_request_is_still_honoured(self) -> None:
        registry = ResultRegistry()
        ref = registry.register("x" * 200_000, kind="inspect", step=1)

        outcome = dispatch_inspect(
            "doc",
            _action("read_result", name=ref.id, max_chars=500),
            registry,
            preview_chars=10_000,
        )

        assert 500 <= len(str(outcome.value)) < 1_000

    def test_the_whole_result_is_still_reachable_in_successive_reads(self) -> None:
        """Clamping bounds one step, it does not hide the rest of the result."""
        registry = ResultRegistry()
        ref = registry.register("abcdefghij" * 3_000, kind="inspect", step=1)

        first = dispatch_inspect(
            "doc",
            _action("read_result", name=ref.id, start=0, max_chars=10_000_000),
            registry,
            preview_chars=5_000,
        )
        second = dispatch_inspect(
            "doc",
            _action("read_result", name=ref.id, start=5_000, max_chars=10_000_000),
            registry,
            preview_chars=5_000,
        )

        assert str(first.value)[:10] == "abcdefghij"
        assert str(second.value)[:10] == "abcdefghij"
        assert first.readback is not None
        assert second.readback is not None
        assert second.readback.lo == 5_000
