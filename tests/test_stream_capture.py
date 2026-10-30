"""Tests for thread-local stdout/stderr capture (issue #90).

The defect: both in-process cell runners used ``contextlib.redirect_stdout``,
which rebinds the *process-global* ``sys.stdout``. Two concurrent Compare cells
therefore interleaved into each other's buffers, and whichever context manager
exited last restored what it saw on entry — leaving the global pointing at a
``StringIO`` belonging to a finished cell.

Two properties are pinned here: output isolation between overlapping cells,
and stream restoration afterwards, on the success, raise and timeout paths.
"""

from __future__ import annotations

import io
import sys
import threading
import time

import pytest

from rlmstudio.envs.stream_capture import capture_streams, is_installed

LINES_PER_CELL = 200


def _print_slowly(count: int, tag: str, stagger: float = 0.0) -> None:
    """Emit *count* tagged lines with a yield between each.

    The sleep is what makes the two threads genuinely overlap: without it the
    GIL may let one cell finish before the other starts, and the test would
    pass without ever exercising concurrency.
    """
    for i in range(count):
        print(f"{tag}-{i}")
        if stagger:
            time.sleep(stagger)


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def test_overlapping_cells_capture_only_their_own_output() -> None:
    """The reproduction from the issue: two cells, two threads, interleaved."""
    buf_a, buf_b = io.StringIO(), io.StringIO()
    barrier = threading.Barrier(2)

    def cell(buf: io.StringIO, tag: str) -> None:
        with capture_streams(buf, io.StringIO()):
            barrier.wait()  # guarantee both are inside the capture together
            _print_slowly(LINES_PER_CELL, tag, stagger=0.0005)

    threads = [
        threading.Thread(target=cell, args=(buf_a, "A")),
        threading.Thread(target=cell, args=(buf_b, "B")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive(), "cell thread did not finish"

    a_lines = buf_a.getvalue().splitlines()
    b_lines = buf_b.getvalue().splitlines()

    assert len(a_lines) == LINES_PER_CELL, f"A captured {len(a_lines)} lines"
    assert len(b_lines) == LINES_PER_CELL, f"B captured {len(b_lines)} lines"
    assert all(line.startswith("A-") for line in a_lines), "A captured B's output"
    assert all(line.startswith("B-") for line in b_lines), "B captured A's output"


def test_stderr_is_isolated_independently_of_stdout() -> None:
    """A cell's stderr must not land in another cell's stderr buffer, and the
    two slots must not be confused with each other."""
    out_a, err_a = io.StringIO(), io.StringIO()
    out_b, err_b = io.StringIO(), io.StringIO()
    barrier = threading.Barrier(2)

    def cell(out: io.StringIO, err: io.StringIO, tag: str) -> None:
        with capture_streams(out, err):
            barrier.wait()
            for i in range(50):
                print(f"out-{tag}-{i}")
                print(f"err-{tag}-{i}", file=sys.stderr)
                time.sleep(0.0005)

    threads = [
        threading.Thread(target=cell, args=(out_a, err_a, "A")),
        threading.Thread(target=cell, args=(out_b, err_b, "B")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert all(ln.startswith("out-A-") for ln in out_a.getvalue().splitlines())
    assert all(ln.startswith("err-A-") for ln in err_a.getvalue().splitlines())
    assert all(ln.startswith("out-B-") for ln in out_b.getvalue().splitlines())
    assert all(ln.startswith("err-B-") for ln in err_b.getvalue().splitlines())


def test_uncaptured_thread_still_writes_through() -> None:
    """A thread with no capture target set must reach the real stream.

    This is what keeps the proxy transparent to everything in the server
    process that is not a cell — logging, the reloader, library warnings.
    """
    captured = io.StringIO()
    seen: list[str] = []

    class _Recorder(io.StringIO):
        def write(self, s: str) -> int:
            seen.append(s)
            return len(s)

    with capture_streams(captured, io.StringIO()):
        # Installed now. Swap the proxy's original so we can observe
        # pass-through without writing to the test runner's stdout.
        original = sys.stdout._original  # type: ignore[union-attr]
        sys.stdout._original = _Recorder()  # type: ignore[union-attr]
        try:
            done = threading.Event()

            def other_thread() -> None:
                print("from an uncaptured thread")
                done.set()

            t = threading.Thread(target=other_thread)
            t.start()
            t.join(timeout=10)
            assert done.is_set()
        finally:
            sys.stdout._original = original  # type: ignore[union-attr]

    assert any("from an uncaptured thread" in s for s in seen)
    assert "from an uncaptured thread" not in captured.getvalue()


# ---------------------------------------------------------------------------
# Restoration — the half that left the process broken
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["both_succeed", "one_raises", "one_times_out"])
def test_streams_are_restored_after_overlapping_cells(mode: str) -> None:
    """After two overlapping cells, no capture target may be left set.

    `sys.stdout` itself stays the installed proxy by design — it is
    process-wide and permanent — so what must be clean is the thread-local
    target. A leaked target is what sent later output into a dead buffer.
    """
    # Install first, then snapshot. The proxy replaces sys.stdout once and
    # permanently — that is the design, not a leak — so snapshotting before
    # installation would compare a pre-install stream against the proxy and
    # fail for the wrong reason.
    from rlmstudio.envs.stream_capture import install

    install()
    before_out, before_err = sys.stdout, sys.stderr
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def cell(tag: str, misbehave: bool) -> None:
        try:
            with capture_streams(io.StringIO(), io.StringIO()):
                barrier.wait()
                print(f"{tag}-working")
                if misbehave and mode == "one_raises":
                    raise RuntimeError("boom inside exec")
                if misbehave and mode == "one_times_out":
                    # Stand-in for the timeout path: the body is abandoned by
                    # an exception raised from a timer, as create_timeout does.
                    raise TimeoutError("cell exceeded its budget")
        except (RuntimeError, TimeoutError):
            pass  # the runner catches these; the context manager must still unwind
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [
        threading.Thread(target=cell, args=("A", False)),
        threading.Thread(target=cell, args=("B", mode != "both_succeed")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive()

    assert not errors, f"unexpected errors: {errors}"
    # The proxy must still be the installed one — not replaced by a cell's
    # buffer, which is exactly what the old redirect_stdout left behind.
    assert sys.stdout is before_out, "sys.stdout was swapped"
    assert sys.stderr is before_err, "sys.stderr was swapped"
    assert is_installed(), "the proxy was removed"
    # No thread-local target may survive the cell on this thread either.
    from rlmstudio.envs import stream_capture

    assert getattr(stream_capture._state, "stdout_target", None) is None
    assert getattr(stream_capture._state, "stderr_target", None) is None


def test_nested_capture_restores_the_outer_target() -> None:
    """A cell that itself captures must hand the outer target back."""
    outer, inner = io.StringIO(), io.StringIO()

    with capture_streams(outer, io.StringIO()):
        print("outer-before")
        with capture_streams(inner, io.StringIO()):
            print("inner-only")
        print("outer-after")

    assert outer.getvalue().splitlines() == ["outer-before", "outer-after"]
    assert inner.getvalue().splitlines() == ["inner-only"]


def test_install_is_idempotent_and_does_not_chain_proxies() -> None:
    """Repeated installs must not wrap a proxy in another proxy — that would
    make every write walk a longer chain and eventually recurse."""
    from rlmstudio.envs.stream_capture import install

    with capture_streams(io.StringIO(), io.StringIO()):
        assert is_installed()
        first_out, first_err = sys.stdout, sys.stderr
        install()
        install()
        assert sys.stdout is first_out
        assert sys.stderr is first_err
        # And the proxies must not be each other's original.
        assert sys.stdout._original is not sys.stderr  # type: ignore[union-attr]
        assert sys.stderr._original is not sys.stdout  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# The real call sites (#90 scope: both in-process adapters)
# ---------------------------------------------------------------------------


def test_pyrepl_env_cells_are_isolated_across_threads() -> None:
    """Two PyReplEnv cells on two threads must not see each other's output.

    This is the adapter the issue was reported against. `print` is the right
    vehicle here — PyReplEnv has no PrintCollector, so print goes through
    sys.stdout, which is what used to be swapped process-wide.
    """
    from rlmstudio.envs.pyrepl_env import PyReplEnv

    results: dict[str, str] = {}
    barrier = threading.Barrier(2)

    def cell(tag: str) -> None:
        env = PyReplEnv()
        barrier.wait()
        out = env.execute(
            f"import time\nfor i in range({LINES_PER_CELL}):\n"
            f"    print('{tag}-' + str(i))\n    time.sleep(0.0005)\n"
        )
        results[tag] = out.get("stdout", "") if isinstance(out, dict) else str(out)

    threads = [threading.Thread(target=cell, args=(t,)) for t in ("A", "B")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
        assert not t.is_alive()

    for tag, other in (("A", "B"), ("B", "A")):
        lines = [ln for ln in results[tag].splitlines() if ln.strip()]
        assert lines, f"{tag} captured nothing"
        assert all(ln.startswith(f"{tag}-") for ln in lines), (
            f"{tag} captured {other}'s output: {lines[:5]}"
        )


def test_restricted_sandbox_direct_stream_writes_are_isolated() -> None:
    """The restricted adapter, exercised through the path the fix changes.

    RestrictedPython routes `print` through PrintCollector in the
    per-instance namespace, which was never shared between cells — so a
    print-based assertion here would pass whether or not the proxy works.
    Writing to the stream directly is what used to go through the swapped
    global, so that is the vehicle this test uses.
    """
    from rlmstudio.infrastructure.sandbox.restricted_sandbox import (
        RestrictedSandboxAdapter,
    )

    results: dict[str, str] = {}
    barrier = threading.Barrier(2)

    code = (
        "import sys\n"
        "import time\n"
        "for i in range(100):\n"
        "    sys.stdout.write('{tag}-' + str(i) + '\\n')\n"
        "    time.sleep(0.0005)\n"
    )

    def cell(tag: str) -> None:
        adapter = RestrictedSandboxAdapter(allowed_modules=frozenset({"sys", "time"}))
        barrier.wait()
        result = adapter.execute(code.format(tag=tag))
        results[tag] = result.stdout or ""

    threads = [threading.Thread(target=cell, args=(t,)) for t in ("A", "B")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
        assert not t.is_alive()

    for tag, other in (("A", "B"), ("B", "A")):
        lines = [ln for ln in results[tag].splitlines() if ln.strip()]
        # If the sandbox blocked sys.stdout.write entirely there is nothing to
        # assert about isolation — fail loudly rather than pass vacuously.
        assert lines, (
            f"{tag} captured no direct stream writes; the test is not "
            f"exercising the path the fix changes"
        )
        assert all(ln.startswith(f"{tag}-") for ln in lines), (
            f"{tag} captured {other}'s output: {lines[:5]}"
        )
