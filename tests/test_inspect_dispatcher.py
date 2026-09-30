"""Tests for controller-side inspect dispatch (specs/rlm-working-state, plan D2)."""

from __future__ import annotations

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
