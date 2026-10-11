# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""Python REPL execution environment for RLM."""

import io
import logging
import traceback
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
from typing import Any

from rlmstudio.envs.sandbox import create_safe_globals
from rlmstudio.envs.timeout import TimeoutError as ExecTimeoutError
from rlmstudio.envs.timeout import create_timeout
from rlmstudio.tools import chunk, grep, grep_file, outline_file, peek, peek_file, select

logger = logging.getLogger("rlmstudio.envs")


class PyReplEnv:
    """
    Python REPL(Read-Eval-Print Loop)-like execution environment for RLM.

    Executes Python code in a controlled environment with:
    - Persistent globals (state across executions)
    - Stdout/stderr capture
    - Exception handling
    - Optional sandboxing (safe mode)
    - Timeout and output limits

    Example:
        >>> env = PyReplEnv()
        >>> result = env.execute("x = 1 + 1\\nprint(x)")
        >>> result['stdout']
        '2\\n'
    """

    def __init__(
        self,
        safe_mode: bool = False,
        allowed_imports: list[str] | None = None,
        max_exec_time_s: float = 5.0,
        max_stdout_chars: int = 10000,
    ) -> None:
        """
        Initialize PyReplEnv.

        Args:
            safe_mode: Enable sandbox restrictions
            allowed_imports: List of allowed module names (only in safe mode)
            max_exec_time_s: Maximum execution time per code cell
            max_stdout_chars: Maximum characters in stdout/stderr
        """
        self.safe_mode = safe_mode
        self.allowed_imports = allowed_imports or ["json", "re", "math", "datetime"]
        self.max_exec_time_s = max_exec_time_s
        self.max_stdout_chars = max_stdout_chars

        # Content storage (set via set_content)
        self._content: str | None = None
        self._timeout_warning_emitted: bool = False

        # Persistent environment globals
        if safe_mode:
            # Use restricted globals in safe mode
            self.env_globals = create_safe_globals(allowed_imports=self.allowed_imports)
        else:
            # Use full builtins in unsafe mode
            self.env_globals = {
                "__builtins__": __builtins__,
                "__name__": "__rlm__",
            }

        # Add content navigation tools
        self.env_globals["peek"] = peek
        self.env_globals["peek_file"] = peek_file
        self.env_globals["grep"] = grep
        self.env_globals["grep_file"] = grep_file
        self.env_globals["outline_file"] = outline_file
        self.env_globals["chunk"] = chunk
        self.env_globals["select"] = select

    def set_content(self, content: str) -> None:
        """
        Set the large content P that tools will operate on.

        Args:
            content: The large text content to analyze
        """
        self._content = content
        self.env_globals["P"] = content

        # Create wrapper functions that automatically pass content as first arg
        # This allows LLM to call peek(0, 10) instead of peek(P, 0, 10)
        # Use functools.partial instead of lambdas to avoid pickling issues
        self.env_globals["peek"] = partial(peek, content)
        self.env_globals["peek_file"] = partial(peek_file, content)
        self.env_globals["grep"] = partial(grep, content)
        self.env_globals["grep_file"] = partial(grep_file, content)
        self.env_globals["outline_file"] = partial(outline_file, content)
        self.env_globals["chunk"] = partial(chunk, content)
        self.env_globals["select"] = partial(select, content)

    def get_var(self, name: str) -> Any:
        """
        Get a variable value from the environment.

        Args:
            name: Variable name

        Returns:
            Variable value or None if not found
        """
        return self.env_globals.get(name)

    def _exec_code(self, code: str) -> None:
        """Helper method to execute code (needed for pickling with process-based timeout)."""
        exec(code, self.env_globals)

    def execute(self, code: str) -> dict[str, Any]:
        """
        Execute Python code and return result.

        Args:
            code: Python code to execute

        Returns:
            Dictionary with keys:
                - stdout: str - captured stdout
                - stderr: str - captured stderr
                - exception: str | None - exception message if any
                - timeout: bool - whether execution timed out
                - truncated: bool - whether output was truncated
        """
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        exception_msg: str | None = None
        timeout_occurred = False
        truncated = False

        try:
            # Determine if we can use signal-based timeout
            # signal.alarm() only works in the main thread
            import signal
            import threading

            # Check if SIGALRM is available (Unix-like systems)
            has_sigalrm = hasattr(signal, "SIGALRM")

            # Check if we're in the main thread
            is_main_thread = threading.current_thread() == threading.main_thread()

            # Redirect stdout and stderr
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # CRITICAL: In non-main threads (like Streamlit), we CANNOT use:
                # - Signal-based timeout (raises ValueError)
                # - Process-based timeout (loses variable state)
                # Better to have no timeout than lose variable persistence!

                if has_sigalrm and is_main_thread:
                    # Main thread: Use signal-based timeout
                    timeout_ctx = create_timeout(self.max_exec_time_s, use_signal=True)
                    try:
                        with timeout_ctx:
                            exec(code, self.env_globals)
                    except ValueError as e:
                        # Shouldn't happen (we checked is_main_thread)
                        # But if it does, fall back to no timeout
                        if "signal only works in main thread" in str(e):
                            exec(code, self.env_globals)
                        else:
                            raise
                else:
                    # Non-main thread OR no SIGALRM: Execute without timeout
                    # This preserves variable state across executions
                    # Trade-off: No timeout protection, but RLM remains functional
                    if not self._timeout_warning_emitted:
                        logger.warning(
                            "Executing code without timeout protection "
                            "(non-main thread or no SIGALRM). "
                            "Long-running code may hang the process."
                        )
                        self._timeout_warning_emitted = True
                    exec(code, self.env_globals)

        except ExecTimeoutError as e:
            # Timeout occurred
            timeout_occurred = True
            exception_msg = str(e)
        except Exception as e:
            # Other exception
            exception_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        # Get output
        stdout = stdout_buffer.getvalue()
        stderr = stderr_buffer.getvalue()

        # Truncate if needed
        if len(stdout) > self.max_stdout_chars:
            stdout = stdout[: self.max_stdout_chars] + "\n... (truncated)"
            truncated = True

        if len(stderr) > self.max_stdout_chars:
            stderr = stderr[: self.max_stdout_chars] + "\n... (truncated)"
            truncated = True

        return {
            "stdout": stdout,
            "stderr": stderr,
            "exception": exception_msg,
            "timeout": timeout_occurred,
            "truncated": truncated,
        }

    def reset(self) -> None:
        """Reset the environment to initial state."""
        self.env_globals.clear()

        # Restore appropriate globals based on safe_mode
        if self.safe_mode:
            self.env_globals.update(create_safe_globals(allowed_imports=self.allowed_imports))
        else:
            self.env_globals["__builtins__"] = __builtins__
            self.env_globals["__name__"] = "__rlm__"

        # Restore content if set
        if self._content is not None:
            content = self._content
            self.env_globals["P"] = content
            # Bind tools to content using functools.partial (avoids pickling issues)
            self.env_globals["peek"] = partial(peek, content)
            self.env_globals["peek_file"] = partial(peek_file, content)
            self.env_globals["grep"] = partial(grep, content)
            self.env_globals["grep_file"] = partial(grep_file, content)
            self.env_globals["outline_file"] = partial(outline_file, content)
            self.env_globals["chunk"] = partial(chunk, content)
            self.env_globals["select"] = partial(select, content)
        else:
            # No content set, use unbound tools
            self.env_globals["peek"] = peek
            self.env_globals["peek_file"] = peek_file
            self.env_globals["grep"] = grep
            self.env_globals["grep_file"] = grep_file
            self.env_globals["outline_file"] = outline_file
            self.env_globals["chunk"] = chunk
            self.env_globals["select"] = select
