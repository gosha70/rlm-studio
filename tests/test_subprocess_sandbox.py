"""Tests for SubprocessSandboxAdapter."""

from __future__ import annotations

import time
from typing import Any

import pytest

from rlmstudio.infrastructure.sandbox.subprocess_sandbox import SubprocessSandboxAdapter


@pytest.fixture
def sandbox() -> SubprocessSandboxAdapter:
    """Create a fresh subprocess sandbox with a generous timeout.

    10 s is chosen to absorb ``spawn`` cold-start jitter on macOS
    Python 3.13 under full-suite load.  Spawn re-launches a fresh
    interpreter for every ``process.start()``, which normally takes
    ~200-500 ms but can spike above 3 s when the suite runs many
    subprocess sandbox tests back-to-back (observed intermittent
    "Code execution timed out after 3.0s" failures on unrelated
    tests like ``test_variable_persists_across_calls``).

    Tests that explicitly verify timeout behavior (e.g.
    ``TestTimeoutEnforcement.test_infinite_loop_times_out``) create
    their own ``SubprocessSandboxAdapter(max_exec_time_s=1.0)``
    outside this fixture, so they are unaffected by the bump.
    """
    return SubprocessSandboxAdapter(max_exec_time_s=10.0, max_stdout_chars=5000)


class TestBasicExecution:
    """Basic code execution tests."""

    def test_print_hello(self, sandbox: SubprocessSandboxAdapter) -> None:
        result = sandbox.execute('print("hello")')
        assert result.success
        assert result.stdout.strip() == "hello"
        assert result.exception is None
        assert not result.timeout

    def test_arithmetic(self, sandbox: SubprocessSandboxAdapter) -> None:
        result = sandbox.execute("x = 2 + 3")
        assert result.success
        assert sandbox.get_variable("x") == 5

    def test_multiline_code(self, sandbox: SubprocessSandboxAdapter) -> None:
        code = "a = 1\nb = 2\nc = a + b\nprint(c)"
        result = sandbox.execute(code)
        assert result.success
        assert result.stdout.strip() == "3"
        assert sandbox.get_variable("c") == 3

    def test_syntax_error(self, sandbox: SubprocessSandboxAdapter) -> None:
        result = sandbox.execute("def f(:\n  pass")
        assert not result.success
        assert result.exception is not None
        assert "SyntaxError" in result.exception

    def test_runtime_error(self, sandbox: SubprocessSandboxAdapter) -> None:
        result = sandbox.execute("1 / 0")
        assert not result.success
        assert result.exception is not None
        assert "ZeroDivisionError" in result.exception

    def test_name_error(self, sandbox: SubprocessSandboxAdapter) -> None:
        result = sandbox.execute("print(undefined_var)")
        assert not result.success
        assert "NameError" in (result.exception or "")


class TestVariablePersistence:
    """Variable persistence across executions."""

    def test_set_then_get(self, sandbox: SubprocessSandboxAdapter) -> None:
        sandbox.set_variable("x", 1)
        result = sandbox.execute("x += 1")
        assert result.success
        assert sandbox.get_variable("x") == 2

    def test_variable_persists_across_calls(self, sandbox: SubprocessSandboxAdapter) -> None:
        sandbox.execute("counter = 10")
        sandbox.execute("counter += 5")
        assert sandbox.get_variable("counter") == 15

    def test_get_variable_not_found(self, sandbox: SubprocessSandboxAdapter) -> None:
        assert sandbox.get_variable("nonexistent") is None

    def test_set_variable_dict(self, sandbox: SubprocessSandboxAdapter) -> None:
        sandbox.set_variable("data", {"key": "value", "count": 42})
        result = sandbox.execute('print(data["key"])')
        assert result.success
        assert result.stdout.strip() == "value"

    def test_set_variable_list(self, sandbox: SubprocessSandboxAdapter) -> None:
        sandbox.set_variable("items", [1, 2, 3])
        result = sandbox.execute("items.append(4)\nprint(len(items))")
        assert result.success
        assert result.stdout.strip() == "4"
        assert sandbox.get_variable("items") == [1, 2, 3, 4]


class TestHistoryVariableSurvivesMultipleExecutions:
    """Regression: ``history`` must persist across consecutive execute() calls.

    The subprocess sandbox excludes ``history`` from the child → parent
    return payload (``skip_return``) to avoid bandwidth cost.  But the
    parent-side namespace merge must also exclude it from deletion
    detection (``skip_keys``), otherwise the first ``execute()`` deletes
    ``history`` from the parent namespace and subsequent steps hit
    ``NameError``.  This is the exact multi-step RLM failure mode that
    the ``SANDBOX_SKIP_RETURN_VARS`` fix in ``subprocess_sandbox.py``
    prevents.
    """

    def test_history_readable_across_three_executions(self, sandbox: SubprocessSandboxAdapter):
        history = [
            {"turn": 0, "user": "What is X?", "assistant": "X is foo"},
            {"turn": 1, "user": "And Y?", "assistant": "Y is bar"},
        ]
        sandbox.set_variable("history", history)

        # Step 1: read history
        r1 = sandbox.execute("step1_result = history[-1]['user']")
        assert r1.success, r1.exception
        assert sandbox.get_variable("step1_result") == "And Y?"

        # Step 2: history must still be accessible (this was the bug)
        r2 = sandbox.execute("step2_result = len(history)")
        assert r2.success, r2.exception
        assert sandbox.get_variable("step2_result") == 2

        # Step 3: read first entry
        r3 = sandbox.execute("step3_result = history[0]['assistant']")
        assert r3.success, r3.exception
        assert sandbox.get_variable("step3_result") == "X is foo"

    def test_history_survives_alongside_regular_variable_changes(
        self, sandbox: SubprocessSandboxAdapter
    ):
        """history persists even when other variables are created/deleted."""
        sandbox.set_variable("history", [{"turn": 0, "user": "Q", "assistant": "A"}])

        sandbox.execute("x = 42")
        sandbox.execute("del x")
        # history must survive despite x being created then deleted
        r = sandbox.execute("result = history[0]['user']")
        assert r.success, r.exception
        assert sandbox.get_variable("result") == "Q"


class TestTimeoutEnforcement:
    """Timeout enforcement with hard kill."""

    def test_infinite_loop_times_out(self) -> None:
        sandbox = SubprocessSandboxAdapter(max_exec_time_s=1.0)
        start = time.monotonic()
        result = sandbox.execute("while True: pass")
        elapsed = time.monotonic() - start

        assert result.timeout
        assert not result.success
        assert "timed out" in (result.exception or "").lower()
        # Should complete within the timeout plus a generous spawn
        # cold-start budget — see the ``sandbox`` fixture docstring.
        # What this test actually verifies is that the hard-kill fires,
        # not that spawn is fast.
        assert elapsed < 1.0 + 9.0, f"Took too long: {elapsed:.1f}s"

    def test_sleep_within_timeout(self, sandbox: SubprocessSandboxAdapter) -> None:
        result = sandbox.execute("import time; time.sleep(0.1); print('done')")
        assert result.success
        assert result.stdout.strip() == "done"


class TestToolFunctions:
    """Content-navigation tool availability in the child process."""

    def test_peek_with_content(self, sandbox: SubprocessSandboxAdapter) -> None:
        sandbox.set_variable("P", "hello world, this is a test")
        result = sandbox.execute("print(peek(start=0, end=5))")
        assert result.success
        assert "hello" in result.stdout

    def test_grep_with_content(self, sandbox: SubprocessSandboxAdapter) -> None:
        sandbox.set_variable("P", "line one\nline two\nline three\n")
        result = sandbox.execute('print(grep("two"))')
        assert result.success
        assert "two" in result.stdout

    def test_chunk_with_content(self, sandbox: SubprocessSandboxAdapter) -> None:
        sandbox.set_variable("P", "abcdefghij" * 10)
        result = sandbox.execute("chunks = chunk(size=20)\nprint(len(chunks))")
        assert result.success
        assert int(result.stdout.strip()) == 5

    def test_tools_without_content(self, sandbox: SubprocessSandboxAdapter) -> None:
        # Tools should still be importable; they just need content arg explicitly
        result = sandbox.execute(
            "from rlmstudio.tools.content import peek\nprint(peek('abc', 0, 2))"
        )
        assert result.success
        assert "ab" in result.stdout


class TestNonSerializableVariables:
    """Non-serializable variables are skipped gracefully."""

    def test_lambda_skipped(self, sandbox: SubprocessSandboxAdapter) -> None:
        sandbox.set_variable("fn", lambda x: x + 1)
        sandbox.set_variable("x", 10)
        result = sandbox.execute("y = x + 1\nprint(y)")
        assert result.success
        assert result.stdout.strip() == "11"
        # fn is not available in child (non-serializable), but no crash
        assert sandbox.get_variable("y") == 11

    def test_class_instance_skipped(self, sandbox: SubprocessSandboxAdapter) -> None:
        class Custom:
            pass

        sandbox.set_variable("obj", Custom())
        sandbox.set_variable("num", 5)
        result = sandbox.execute("print(num)")
        assert result.success
        assert result.stdout.strip() == "5"


class TestReset:
    """Reset clears the namespace."""

    def test_reset_clears_variables(self, sandbox: SubprocessSandboxAdapter) -> None:
        sandbox.set_variable("x", 42)
        assert sandbox.get_variable("x") == 42
        sandbox.reset()
        assert sandbox.get_variable("x") is None

    def test_reset_then_execute(self, sandbox: SubprocessSandboxAdapter) -> None:
        sandbox.execute("x = 100")
        sandbox.reset()
        result = sandbox.execute("print(x)")
        assert not result.success  # x is undefined after reset
        assert "NameError" in (result.exception or "")


class TestHealthCheck:
    """is_healthy always returns True."""

    def test_is_healthy(self, sandbox: SubprocessSandboxAdapter) -> None:
        assert sandbox.is_healthy() is True


class TestStdoutTruncation:
    """Stdout truncation when output exceeds max_stdout_chars."""

    def test_large_output_truncated(self) -> None:
        # 10 s budget — same reason as the shared ``sandbox`` fixture:
        # spawn cold-start on macOS Python 3.13 can spike above 3 s
        # under load and cause spurious timeout failures.
        sandbox = SubprocessSandboxAdapter(max_exec_time_s=10.0, max_stdout_chars=100)
        result = sandbox.execute("print('x' * 500)")
        assert result.truncated
        assert len(result.stdout) < 200  # truncated + marker


class TestStderr:
    """Stderr capture."""

    def test_stderr_captured(self, sandbox: SubprocessSandboxAdapter) -> None:
        result = sandbox.execute("import sys; sys.stderr.write('warning\\n')")
        assert "warning" in result.stderr


class TestNamespaceDeletionSemantics:
    """P1 fix: deletions and non-serializable reassignments are reflected."""

    def test_del_removes_variable(self, sandbox: SubprocessSandboxAdapter) -> None:
        sandbox.set_variable("x", 42)
        result = sandbox.execute("del x")
        assert result.success
        assert sandbox.get_variable("x") is None

    def test_reassign_to_non_serializable_clears(self, sandbox: SubprocessSandboxAdapter) -> None:
        sandbox.set_variable("x", 10)
        # Child reassigns x to a lambda — not serializable, so parent
        # should see x removed (not stale value 10).
        result = sandbox.execute("x = lambda: 1")
        assert result.success
        assert sandbox.get_variable("x") is None

    def test_non_serializable_parent_var_preserved(self, sandbox: SubprocessSandboxAdapter) -> None:
        # Non-serializable vars that were never sent to child stay intact.
        sandbox.set_variable("fn", lambda: 99)
        sandbox.set_variable("y", 5)
        result = sandbox.execute("y += 1")
        assert result.success
        assert sandbox.get_variable("y") == 6
        # fn was never serialized/sent, so it should survive.
        assert callable(sandbox.get_variable("fn"))


class TestContentNotRoundTripped:
    """P2 fix: P is not returned from child, avoiding O(doc-size) overhead."""

    def test_large_p_not_in_child_return(self, sandbox: SubprocessSandboxAdapter) -> None:
        big_content = "x" * 100_000
        sandbox.set_variable("P", big_content)
        sandbox.set_variable("counter", 0)
        result = sandbox.execute("counter = 1")
        assert result.success
        assert sandbox.get_variable("counter") == 1
        # P should still be present in the parent namespace (never removed).
        assert sandbox.get_variable("P") == big_content


class TestServerSandboxSelection:
    """P0 fix: server does NOT override 'restricted' to 'subprocess'."""

    def test_restricted_type_passes_through(self) -> None:
        from unittest.mock import patch

        from rlmstudio.server.dependencies import AppState

        state = AppState(load_from_disk=False)
        state.config.sandbox.type = "restricted"
        with patch("rlmstudio.server.dependencies.create_sandbox") as mock_create:
            state.create_sandbox()
            mock_create.assert_called_once_with(sandbox_type="restricted")

    def test_local_type_becomes_subprocess(self) -> None:
        from unittest.mock import patch

        from rlmstudio.server.dependencies import AppState

        state = AppState(load_from_disk=False)
        state.config.sandbox.type = "local"
        with patch("rlmstudio.server.dependencies.create_sandbox") as mock_create:
            state.create_sandbox()
            mock_create.assert_called_once_with(sandbox_type="subprocess")


class TestSafeMode:
    """safe_mode restricts builtins and imports in the child process."""

    @pytest.fixture
    def safe_sandbox(self) -> SubprocessSandboxAdapter:
        # 10 s budget — see sandbox fixture above for rationale.
        return SubprocessSandboxAdapter(safe_mode=True, max_exec_time_s=10.0)

    def test_safe_mode_blocks_os_import(self, safe_sandbox: SubprocessSandboxAdapter) -> None:
        result = safe_sandbox.execute("import os")
        assert not result.success
        assert result.exception is not None
        assert "not allowed" in result.exception.lower() or "Security" in result.exception

    def test_safe_mode_blocks_subprocess(self, safe_sandbox: SubprocessSandboxAdapter) -> None:
        result = safe_sandbox.execute("import subprocess")
        assert not result.success
        assert result.exception is not None

    def test_safe_mode_blocks_open(self, safe_sandbox: SubprocessSandboxAdapter) -> None:
        result = safe_sandbox.execute("f = open('/etc/passwd')")
        assert not result.success
        assert result.exception is not None

    def test_safe_mode_allows_safe_imports(self, safe_sandbox: SubprocessSandboxAdapter) -> None:
        result = safe_sandbox.execute("import json\nprint(json.dumps([1,2]))")
        assert result.success
        assert "[1, 2]" in result.stdout

    def test_safe_mode_allows_math(self, safe_sandbox: SubprocessSandboxAdapter) -> None:
        result = safe_sandbox.execute("import math\nprint(math.pi)")
        assert result.success
        assert "3.14" in result.stdout

    def test_unsafe_mode_allows_os(self, sandbox: SubprocessSandboxAdapter) -> None:
        # Default fixture has safe_mode=False
        result = sandbox.execute("import os; print(os.getpid())")
        assert result.success
        assert result.stdout.strip().isdigit()

    def test_factory_forwards_safe_mode(self) -> None:
        """Factory passes safe_mode through to SubprocessSandboxAdapter."""
        from rlmstudio.infrastructure.sandbox.sandbox_factory import create_sandbox

        sb = create_sandbox(sandbox_type="subprocess", safe_mode=True)
        result = sb.execute("import os")
        assert not result.success
        assert result.exception is not None
        assert "not allowed" in result.exception.lower() or "Security" in result.exception


class TestAsyncExecution:
    """Async execution wrapper."""

    @pytest.mark.asyncio
    async def test_execute_async(self, sandbox: SubprocessSandboxAdapter) -> None:
        result = await sandbox.execute_async('print("async hello")')
        assert result.success
        assert "async hello" in result.stdout


@pytest.mark.slow
class TestTopLevelScriptEntrypoint:
    """Regression coverage for the ``__main__`` script path.

    Pytest itself runs the adapter from imported test modules, so it never
    exercises what happens when a plain ``python script.py`` invocation
    drives the adapter.  That is exactly the path that broke twice during
    this work:

    1. The original spawn-based implementation required an
       ``if __name__ == '__main__':`` guard; scripts without it raised
       ``RuntimeError: An attempt has been made to start a new process
       before the current process has finished its bootstrapping phase``.
    2. A subsequent "fix" switched to fork to remove that requirement,
       but fork from the multi-threaded server parent is unsafe.

    These tests pin both the working idiom and the failing idiom so any
    future change to ``_MP_CTX`` is caught in CI instead of at runtime.

    Marked ``slow`` because each test launches 1-3 fresh Python
    interpreters via ``subprocess.run``, compounding spawn cost to
    ~5-18 s per test.  Default dev runs skip these; CI passes
    ``--runslow`` to include them.
    """

    _SANDBOX_IMPORT = (
        "from rlmstudio.infrastructure.sandbox.subprocess_sandbox import SubprocessSandboxAdapter"
    )

    def _run_script(
        self, tmp_path: Any, source: str, timeout: float = 60.0
    ) -> tuple[int, str, str]:
        """Write *source* to a .py file and run it under the current Python.

        Uses :mod:`subprocess` (stdlib) rather than invoking pytest from
        within pytest, so the child is a genuinely fresh interpreter.
        Returns ``(returncode, stdout, stderr)``.

        The child is given an explicit working directory instead of inheriting
        the test process's.  A sibling test elsewhere in the suite can leave the
        worker chdir'd into a directory it is about to delete, and the child
        would then die on the first ``os.getcwd()`` — in ``litellm`` at import
        time, or in ``multiprocessing``'s spawn preparation — for reasons that
        have nothing to do with the sandbox.
        """
        import subprocess  # noqa: S404 — stdlib, only used to spawn ourselves
        import sys as _sys

        script = tmp_path / "uc_script.py"
        script.write_text(source)
        proc = subprocess.run(  # noqa: S603 — executable is sys.executable, trusted
            [_sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(tmp_path),
        )
        return proc.returncode, proc.stdout, proc.stderr

    def test_top_level_script_with_main_guard_succeeds(self, tmp_path: Any) -> None:
        """The documented ``if __name__ == '__main__':`` idiom works end-to-end.

        This is the contract for anyone using the adapter from a plain
        Python script: wrap entrypoint calls in ``main()`` under the
        standard guard, and execute() returns normally.
        """
        source = f"""
{self._SANDBOX_IMPORT}


def main() -> None:
    sb = SubprocessSandboxAdapter(max_exec_time_s=5.0)
    result = sb.execute("print('hello from child')")
    assert result.success, f"execute failed: {{result.exception}}"
    assert "hello from child" in result.stdout, result.stdout
    print("SCRIPT_OK")


if __name__ == "__main__":
    main()
"""
        rc, out, err = self._run_script(tmp_path, source)
        assert rc == 0, f"script exited {rc}\nstdout:\n{out}\nstderr:\n{err}"
        assert "SCRIPT_OK" in out, f"missing sentinel; stdout:\n{out}"

    def test_top_level_script_without_main_guard_fails_cleanly(self, tmp_path: Any) -> None:
        """Without the ``__main__`` guard, spawn's re-import recursion must
        raise the documented ``RuntimeError`` — not hang or segfault.

        This pins the failure mode so a future "let's use fork to make
        unguarded scripts work" change trips this test on the way in.
        The assertion checks for the canonical multiprocessing error
        string; Python 3.13 emits it via
        ``_check_not_importing_main`` during child bootstrap.
        """
        source = f"""
{self._SANDBOX_IMPORT}

# Intentionally NO `if __name__ == '__main__':` guard.  This replicates
# the exact scenario a user hits when they copy the adapter into a
# quick debug script and run `python uc_script.py`.
sb = SubprocessSandboxAdapter(max_exec_time_s=5.0)
sb.execute("print('this never runs')")
"""
        rc, out, err = self._run_script(tmp_path, source)
        # The parent script fails — either the exception bubbles up
        # (non-zero rc) or the child's RuntimeError shows in stderr.
        assert rc != 0 or "bootstrapping phase" in err, (
            f"expected script failure or bootstrapping error\n"
            f"rc={rc}\nstdout:\n{out}\nstderr:\n{err}"
        )
        assert "bootstrapping phase" in err or "_check_not_importing_main" in err, (
            f"expected spawn's bootstrapping-phase error in stderr\nstderr:\n{err}"
        )

    def test_documented_uc_script_passes(self, tmp_path: Any) -> None:
        """Exact replica of the manual-test script from the docs — this is
        UC-1.1 + UC-1.2 + UC-1.3 rolled into one fresh-interpreter run.

        If anyone updates the adapter and silently breaks the scripted
        entry path, this test fails loudly.
        """
        source = f"""
import time

{self._SANDBOX_IMPORT}


def test_timeout() -> None:
    sb = SubprocessSandboxAdapter(max_exec_time_s=2.0)
    start = time.monotonic()
    result = sb.execute("while True: pass")
    elapsed = time.monotonic() - start
    assert result.timeout, f"expected timeout=True, got {{result}}"
    # Generous upper bound: 2 s budget + 10 s spawn slack.  The test
    # verifies hard-kill fires, not that spawn is fast.
    assert elapsed < 12.0, f"took too long: {{elapsed:.2f}}s"


def test_persistence() -> None:
    sb = SubprocessSandboxAdapter()
    sb.set_variable("x", 0)
    sb.execute("x = x + 42")
    assert sb.get_variable("x") == 42
    sb.execute("x = x * 2")
    assert sb.get_variable("x") == 84


def test_non_serializable() -> None:
    sb = SubprocessSandboxAdapter()
    sb.set_variable("f", lambda y: y + 1)
    sb.set_variable("n", 10)
    result = sb.execute("print(n)")
    assert result.success, result.exception
    assert "10" in result.stdout


def main() -> None:
    test_timeout()
    test_persistence()
    test_non_serializable()
    print("ALL_UC_PASSED")


if __name__ == "__main__":
    main()
"""
        rc, out, err = self._run_script(tmp_path, source, timeout=90.0)
        assert rc == 0, f"script exited {rc}\nstdout:\n{out}\nstderr:\n{err}"
        assert "ALL_UC_PASSED" in out, f"missing sentinel; stdout:\n{out}"
