"""Thread-local capture of ``sys.stdout`` / ``sys.stderr`` for in-process cells.

``contextlib.redirect_stdout`` rebinds the *process-global* ``sys.stdout``.
That is fine for one cell at a time and wrong the moment two run concurrently,
which is exactly what the Compare grid does with the in-process sandboxes:

* Two cells overlap. B's ``redirect_stdout`` replaces the global while A is
  still executing, so A's remaining output lands in B's buffer and B's buffer
  contains both.
* Whichever context manager exits last restores what it saw on entry, so the
  global can be left pointing at a ``StringIO`` that belongs to a finished
  cell. Everything printed afterwards — including by unrelated library code in
  the server process — is written into a buffer nobody reads.

The fix is to stop moving the global. ``install()`` replaces ``sys.stdout``
and ``sys.stderr`` once, with a proxy that dispatches each write to a
thread-local target when the calling thread has set one, and to the original
stream otherwise. Concurrent cells then capture only their own output, and
no cell ever restores a stream on another's behalf.

Scope: the in-process paths only — ``PyReplEnv`` and
``RestrictedSandboxAdapter``. The subprocess and Docker sandboxes redirect
inside their own process or container and are unaffected. What a library
caller does with ``sys.stdout`` outside a cell is also unaffected: with no
thread-local target set, the proxy is a pass-through to the original stream.

See issue #90.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO, Any

__all__ = ["capture_streams", "install", "is_installed"]

_install_lock = threading.Lock()
_state = threading.local()


class _ThreadLocalStreamProxy:
    """Dispatch writes to this thread's capture target, else the real stream.

    Deliberately not a subclass of ``io.TextIOBase``: the object has to stand
    in for whatever ``sys.stdout`` already was, including a pytest capture
    object or a Jupyter shim, so it forwards unknown attributes to the
    original rather than imposing a file-like contract of its own.
    """

    def __init__(self, original: IO[str], slot: str) -> None:
        self._original = original
        self._slot = slot

    @property
    def _target(self) -> IO[str]:
        return getattr(_state, self._slot, None) or self._original

    # -- the parts that actually matter for capture --

    def write(self, s: str) -> int:
        return self._target.write(s)

    def writelines(self, lines: Any) -> None:
        self._target.writelines(lines)

    def flush(self) -> None:
        # A captured StringIO has flush(); a closed one does not tolerate it.
        try:
            self._target.flush()
        except ValueError:
            # "I/O operation on closed file" — a cell whose buffer was closed
            # after it finished must not break an unrelated thread's write.
            pass

    def isatty(self) -> bool:
        # Capture targets are never a terminal. Reporting the original's
        # answer while capturing would make libraries emit ANSI colour into
        # a StringIO that the UI then renders as escape codes.
        return False if getattr(_state, self._slot, None) else self._original.isatty()

    @property
    def encoding(self) -> str:
        return getattr(self._original, "encoding", "utf-8")

    def __getattr__(self, name: str) -> Any:
        # Everything else (fileno, buffer, errors, …) comes from the real
        # stream. Reached only for names not defined above.
        return getattr(self._original, name)


def is_installed() -> bool:
    """True when the proxies are in place on ``sys.stdout``/``sys.stderr``."""
    return isinstance(sys.stdout, _ThreadLocalStreamProxy) and isinstance(
        sys.stderr, _ThreadLocalStreamProxy
    )


def install() -> None:
    """Install the proxies once. Idempotent and safe to call concurrently.

    Called lazily by :func:`capture_streams` rather than at import time, so
    importing the package never mutates the process's streams.
    """
    with _install_lock:
        if is_installed():
            return
        # Read both before assigning either: if stdout were replaced first and
        # stderr's original were then read back through the new proxy, the
        # proxies would chain instead of sitting side by side.
        original_out, original_err = sys.stdout, sys.stderr
        sys.stdout = _ThreadLocalStreamProxy(original_out, "stdout_target")  # type: ignore[assignment]
        sys.stderr = _ThreadLocalStreamProxy(original_err, "stderr_target")  # type: ignore[assignment]


@contextmanager
def capture_streams(stdout: IO[str], stderr: IO[str]) -> Iterator[None]:
    """Route this thread's ``print`` / stream writes into *stdout* / *stderr*.

    Replaces ``with redirect_stdout(buf), redirect_stderr(buf2):`` at the
    in-process cell boundaries. Unlike that pair, it touches no global state,
    so an overlapping cell on another thread is unaffected and no cell can
    leave a finished buffer installed process-wide.

    Restores the previous thread-local targets on exit, including when the
    body raises, so nesting (a cell that itself captures) behaves.
    """
    install()
    previous_out = getattr(_state, "stdout_target", None)
    previous_err = getattr(_state, "stderr_target", None)
    _state.stdout_target = stdout
    _state.stderr_target = stderr
    try:
        yield
    finally:
        _state.stdout_target = previous_out
        _state.stderr_target = previous_err
