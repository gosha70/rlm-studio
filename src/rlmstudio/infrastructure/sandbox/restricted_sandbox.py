"""Restricted sandbox adapter using RestrictedPython.

Compiles LLM-generated code through RestrictedPython's AST transformer which
blocks dangerous patterns at compile time:
- Attribute access to dunder methods (``__class__``, ``__bases__``, etc.)
- Direct ``import`` of modules not in the allow-list
- ``exec`` / ``eval`` / ``compile`` (not present in ``safe_builtins``)
- File I/O (``open`` not present in ``safe_builtins``)
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
from typing import Any

from RestrictedPython import compile_restricted, safe_builtins, safe_globals
from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getiter
from RestrictedPython.Guards import (
    guarded_unpack_sequence,
    safer_getattr,
)
from RestrictedPython.PrintCollector import PrintCollector

from rlmstudio.application.dto import ExecutionResultDTO
from rlmstudio.domain.exceptions import SecurityViolationError
from rlmstudio.tools import chunk, grep, grep_file, outline_file, peek, peek_file, select

# Modules considered safe for LLM-generated code (subset of rlmstudio defaults)
_DEFAULT_ALLOWED_MODULES = frozenset(
    {
        "json",
        "re",
        "math",
        "datetime",
        "time",
        "calendar",
        "decimal",
        "fractions",
        "statistics",
        "random",
        "string",
        "textwrap",
        "unicodedata",
        "itertools",
        "functools",
        "operator",
        "collections",
        "heapq",
        "bisect",
        "array",
        "copy",
        "pprint",
        "enum",
        "dataclasses",
        "typing",
    }
)


def _make_safe_import(allowed_modules: frozenset[str]) -> Any:
    """Return a restricted ``__import__`` that only allows *allowed_modules*."""
    import builtins as _builtins

    def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
        base = name.split(".")[0]
        if base not in allowed_modules:
            raise SecurityViolationError(
                f"Import of module '{name}' is not allowed. "
                f"Allowed modules: {', '.join(sorted(allowed_modules)[:10])}..."
            )
        return _builtins.__import__(name, *args, **kwargs)

    return _safe_import


class RestrictedSandboxAdapter:
    """Sandbox that uses RestrictedPython for compile-time and runtime safety.

    This adapter provides stronger isolation than the basic
    :class:`LocalSandboxAdapter` by using RestrictedPython's AST transformer
    to block dangerous patterns *before* the code runs.

    Args:
        allowed_modules: Set of module names that ``import`` may load.
        max_stdout_chars: Maximum characters captured from stdout/stderr.
    """

    def __init__(
        self,
        allowed_modules: frozenset[str] | None = None,
        max_stdout_chars: int = 10_000,
    ) -> None:
        self._allowed_modules = allowed_modules or _DEFAULT_ALLOWED_MODULES
        self._max_stdout_chars = max_stdout_chars
        self._namespace: dict[str, Any] = {}
        self._build_globals()

    # ------------------------------------------------------------------
    # SandboxPort interface
    # ------------------------------------------------------------------

    def execute(self, code: str) -> ExecutionResultDTO:
        """Compile and execute *code* under RestrictedPython.

        Args:
            code: Python source code to execute.

        Returns:
            :class:`ExecutionResultDTO` with captured output.
        """
        # 1. Compile through RestrictedPython
        try:
            byte_code = compile_restricted(code, "<sandbox>", "exec")
        except SyntaxError as exc:
            return ExecutionResultDTO(exception=f"SyntaxError: {exc}")

        if byte_code is None:
            return ExecutionResultDTO(exception="RestrictedPython compilation failed")

        # 2. Execute in the restricted namespace
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        exception_msg: str | None = None
        truncated = False

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(byte_code, self._globals, self._namespace)  # noqa: S102
        except SecurityViolationError as exc:
            exception_msg = f"SecurityViolationError: {exc}"
        except Exception as exc:
            exception_msg = f"{type(exc).__name__}: {exc}"

        # Collect printed output from PrintCollector
        printer = self._namespace.pop("_print", None)
        printed_text = ""
        if printer is not None:
            printed_text = printer()

        stdout = printed_text + stdout_buf.getvalue()
        stderr = stderr_buf.getvalue()

        if len(stdout) > self._max_stdout_chars:
            stdout = stdout[: self._max_stdout_chars] + "\n... (truncated)"
            truncated = True

        return ExecutionResultDTO(
            stdout=stdout,
            stderr=stderr,
            exception=exception_msg,
            truncated=truncated,
        )

    def reset(self) -> None:
        """Reset the execution namespace to a clean state."""
        self._namespace.clear()
        self._build_globals()

    def is_healthy(self) -> bool:
        """The restricted sandbox is always healthy if RestrictedPython is importable."""
        return True

    def get_variable(self, name: str) -> Any | None:
        """Retrieve a variable from the sandbox namespace.

        Args:
            name: Variable name.

        Returns:
            The variable's value, or ``None`` if not found.
        """
        return self._namespace.get(name)

    async def execute_async(self, code: str) -> ExecutionResultDTO:
        """Execute code asynchronously by delegating to the sync method."""
        import asyncio

        return await asyncio.to_thread(self.execute, code)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def set_variable(self, name: str, value: Any) -> None:
        """Inject a variable into the sandbox namespace.

        When ``name`` is ``"P"`` (the content variable), also rebinds the
        content-navigation tools (``peek``, ``peek_file``, ``grep``,
        ``grep_file``, ``outline_file``, ``chunk``, ``select``)
        as ``partial`` functions so that LLM-generated code can call them
        without passing content explicitly.

        Args:
            name: Variable name.
            value: Variable value.
        """
        self._namespace[name] = value
        if name == "P" and isinstance(value, str):
            content = value
            self._globals["peek"] = partial(peek, content)
            self._globals["peek_file"] = partial(peek_file, content)
            self._globals["grep"] = partial(grep, content)
            self._globals["grep_file"] = partial(grep_file, content)
            self._globals["outline_file"] = partial(outline_file, content)
            self._globals["chunk"] = partial(chunk, content)
            self._globals["select"] = partial(select, content)

    def _build_globals(self) -> None:
        """(Re)build the restricted globals dict."""
        self._globals: dict[str, Any] = {**safe_globals}

        # RestrictedPython guard hooks
        self._globals["_print_"] = PrintCollector
        self._globals["_getattr_"] = safer_getattr
        self._globals["_getiter_"] = default_guarded_getiter
        self._globals["_getitem_"] = default_guarded_getitem
        self._globals["_iter_unpack_sequence_"] = guarded_unpack_sequence

        # Provide a restricted __import__
        builtins_dict = dict(safe_builtins)
        builtins_dict["__import__"] = _make_safe_import(self._allowed_modules)
        self._globals["__builtins__"] = builtins_dict

        # Content-navigation tools (unbound; rebound to P in set_variable)
        self._globals["peek"] = peek
        self._globals["peek_file"] = peek_file
        self._globals["grep"] = grep
        self._globals["grep_file"] = grep_file
        self._globals["outline_file"] = outline_file
        self._globals["chunk"] = chunk
        self._globals["select"] = select
