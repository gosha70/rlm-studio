"""Controller-side execution of v2 ``inspect`` actions.

The JSON protocol's inspect tools are deterministic functions over the
content string, so the controller runs them directly against ``content``
and hands the complete return value to the result registry.  Only the v1
free-form Python path still crosses the sandbox boundary.  The display
form recorded in the trace (``print(peek(start=0, end=2000, …))``) is kept
byte-identical to the code the sandbox used to receive, so traces, replays
and the file-coverage heuristics are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rlmstudio.application.sandbox_vars import RLM_DEFAULT_PATTERN_TIMEOUT_SECONDS
from rlmstudio.application.services.result_registry import ReadSlice, ResultRegistry
from rlmstudio.core.actions import InspectAction
from rlmstudio.tools import (
    PatternTimeoutError,
    chunk,
    grep,
    grep_file,
    outline_file,
    peek,
    peek_file,
    select,
)

READ_RESULT_TOOL = "read_result"

_TOOL_ERRORS = (ValueError, TypeError, IndexError, KeyError, OverflowError)


class InspectArgError(ValueError):
    """An inspect action's ``args`` do not fit the tool's signature."""


@dataclass(frozen=True)
class InspectOutcome:
    """What one dispatched inspect action produced.

    Attributes:
        display_code: The trace/replay form of the call.
        value: The tool's complete return value (``None`` on error).
        error: One-line error text when the call could not run.
        is_readback: ``True`` for ``read_result`` (never registered).
    """

    display_code: str
    value: str | list[str] | None = None
    error: str | None = None
    is_readback: bool = False
    readback: ReadSlice | None = None
    """For a successful ``read_result``: which result and range was read."""


def display_code(action: InspectAction) -> str:
    """The code string recorded in the trace for ``action``."""
    tool, args = action.tool, action.args
    if tool == "grep":
        return (
            f"print(grep("
            f"pattern={args.get('pattern')!r}, "
            f"context_lines={args.get('context_lines', 2)}, "
            f"max_matches={args.get('max_matches', 100)}, "
            f"ignore_case={args.get('ignore_case', False)}, "
            f"use_regex={args.get('use_regex', False)}))"
        )
    if tool == "peek":
        return (
            f"print(peek("
            f"start={args.get('start', 0)}, "
            f"end={args.get('end')}, "
            f"max_chars={args.get('max_chars', 10000)}))"
        )
    if tool == "peek_file":
        return (
            f"print(peek_file("
            f"file_no={args.get('file_no')}, "
            f"start={args.get('start', 0)}, "
            f"end={args.get('end')}, "
            f"max_chars={args.get('max_chars', 10000)}))"
        )
    if tool == "grep_file":
        return (
            f"print(grep_file("
            f"file_no={args.get('file_no')}, "
            f"pattern={args.get('pattern')!r}, "
            f"context_lines={args.get('context_lines', 2)}, "
            f"max_matches={args.get('max_matches', 100)}, "
            f"ignore_case={args.get('ignore_case', False)}, "
            f"use_regex={args.get('use_regex', False)}))"
        )
    if tool == "outline_file":
        return (
            f"print(outline_file("
            f"file_no={args.get('file_no')}, "
            f"max_lines={args.get('max_lines', 40)}, "
            f"max_chars={args.get('max_chars', 8000)}))"
        )
    if tool == "select":
        return f"print(select(ranges={args.get('ranges')}))"
    if tool == "chunk":
        return (
            f"print(chunk("
            f"size={args.get('size', 1000)}, "
            f"overlap={args.get('overlap', 0)}, "
            f"by={args.get('by', 'chars')!r}, "
            f"max_chunks={args.get('max_chunks', 100)}))"
        )
    if tool == READ_RESULT_TOOL:
        return (
            f"read_result("
            f"name={args.get('name', 'last')!r}, "
            f"start={args.get('start', 0)}, "
            f"end={args.get('end')}, "
            f"max_chars={args.get('max_chars', 10000)})"
        )
    return f"{tool}({args!r})"


def dispatch_inspect(
    content: str,
    action: InspectAction,
    registry: ResultRegistry | None = None,
    *,
    preview_chars: int = 10000,
    pattern_timeout: float = RLM_DEFAULT_PATTERN_TIMEOUT_SECONDS,
) -> InspectOutcome:
    """Run ``action`` against ``content`` and return its complete result.

    Errors (bad args, a tool raising) come back as ``error`` text rather
    than exceptions so the loop can show them as an ordinary execution
    failure and let the model recover.

    ``pattern_timeout`` bounds the regex engine for the pattern tools.  These
    tools used to run inside the sandbox, which cut any call off after its own
    timeout; running them here removed that limit, and a nested quantifier in
    a model-written pattern can otherwise backtrack for hours while holding
    the GIL.
    """
    code = display_code(action)
    try:
        if action.tool == READ_RESULT_TOOL:
            return _read_back(code, action, registry, preview_chars)
        value = _run(content, action, registry, preview_chars, pattern_timeout)
    except InspectArgError as exc:
        return InspectOutcome(display_code=code, error=f"Error: {exc}")
    except PatternTimeoutError as exc:
        return InspectOutcome(
            display_code=code,
            error=(
                f"Error: {action.tool}() timed out after {exc.budget:.1f}s. "
                "The pattern backtracks too much on this content — simplify it "
                "(avoid a quantifier inside a quantified group) or search a "
                "narrower range."
            ),
        )
    except _TOOL_ERRORS as exc:
        return InspectOutcome(display_code=code, error=f"Error: {action.tool}() failed: {exc}")
    return InspectOutcome(display_code=code, value=value)


def _read_back(
    code: str,
    action: InspectAction,
    registry: ResultRegistry | None,
    preview_chars: int,
) -> InspectOutcome:
    """Answer ``read_result`` from the registry; never registers anything."""
    if registry is None:
        raise InspectArgError("read_result is not available in this run")
    args = action.args
    # Clamped, not just validated: a result can be hundreds of megabytes, and
    # whatever comes back is written verbatim into the trace row that is stored
    # with the execution and streamed to the UI.  The per-step limit is the
    # same preview cap a fresh inspect result gets.
    requested = _int(args, "max_chars", preview_chars)
    piece = registry.read_slice(
        _str(args, "name", "last"),
        start=_int(args, "start", 0),
        end=_opt_int(args, "end"),
        max_chars=min(requested, preview_chars),
    )
    if isinstance(piece, str):
        # Unknown or evicted id: shown as an ordinary (non-fatal) result.
        return InspectOutcome(display_code=code, value=piece, is_readback=True)
    return InspectOutcome(
        display_code=code, value=piece.body + piece.footer(), is_readback=True, readback=piece
    )


def _run(
    content: str,
    action: InspectAction,
    registry: ResultRegistry | None,
    preview_chars: int,
    pattern_timeout: float,
) -> str | list[str]:
    tool, args = action.tool, action.args
    text: str
    if tool == "peek":
        text = peek(
            content,
            start=_int(args, "start", 0),
            end=_opt_int(args, "end"),
            max_chars=_int(args, "max_chars", 10000),
        )
        return text
    if tool == "grep":
        text = grep(
            content,
            pattern=_str(args, "pattern"),
            context_lines=_int(args, "context_lines", 2),
            max_matches=_int(args, "max_matches", 100),
            ignore_case=_bool(args, "ignore_case", False),
            use_regex=_bool(args, "use_regex", False),
            timeout=pattern_timeout,
        )
        return text
    if tool == "peek_file":
        text = peek_file(
            content,
            file_no=_int(args, "file_no"),
            start=_int(args, "start", 0),
            end=_opt_int(args, "end"),
            max_chars=_int(args, "max_chars", 10000),
        )
        return text
    if tool == "grep_file":
        text = grep_file(
            content,
            file_no=_int(args, "file_no"),
            pattern=_str(args, "pattern"),
            context_lines=_int(args, "context_lines", 2),
            max_matches=_int(args, "max_matches", 100),
            ignore_case=_bool(args, "ignore_case", False),
            use_regex=_bool(args, "use_regex", False),
            timeout=pattern_timeout,
        )
        return text
    if tool == "outline_file":
        text = outline_file(
            content,
            file_no=_int(args, "file_no"),
            max_lines=_int(args, "max_lines", 40),
            max_chars=_int(args, "max_chars", 8000),
        )
        return text
    if tool == "select":
        text = select(content, ranges=_ranges(args))
        return text
    if tool == "chunk":
        chunks: list[str] = chunk(
            content,
            size=_int(args, "size", 1000),
            overlap=_int(args, "overlap", 0),
            by=str(args.get("by", "chars")),
            max_chunks=_int(args, "max_chunks", 100),
        )
        return chunks
    raise InspectArgError(f"unknown inspect tool {tool!r}")


# -- argument coercion ---------------------------------------------------------

_MISSING = object()


def _int(args: dict[str, Any], key: str, default: Any = _MISSING) -> int:
    value = args.get(key, default)
    if value is _MISSING:
        raise InspectArgError(f"{key!r} is required")
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise InspectArgError(f"{key!r} must be an integer, got {value!r}")
    try:
        return int(value)
    except ValueError as exc:
        raise InspectArgError(f"{key!r} must be an integer, got {value!r}") from exc


def _opt_int(args: dict[str, Any], key: str) -> int | None:
    if args.get(key) is None:
        return None
    return _int(args, key)


def _bool(args: dict[str, Any], key: str, default: bool) -> bool:
    value = args.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise InspectArgError(f"{key!r} must be a boolean, got {value!r}")


def _str(args: dict[str, Any], key: str, default: Any = _MISSING) -> str:
    value = args.get(key, default)
    if value is _MISSING or value is None:
        raise InspectArgError(f"{key!r} is required")
    return str(value)


def _ranges(args: dict[str, Any]) -> list[tuple[int, int]]:
    raw = args.get("ranges")
    if not isinstance(raw, list) or not raw:
        raise InspectArgError("'ranges' must be a non-empty list of [start, end] pairs")
    out: list[tuple[int, int]] = []
    for item in raw:
        if not isinstance(item, list | tuple) or len(item) != 2:
            raise InspectArgError(f"each range must be a [start, end] pair, got {item!r}")
        out.append((_int({"v": item[0]}, "v"), _int({"v": item[1]}, "v")))
    return out
