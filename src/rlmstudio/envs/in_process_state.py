"""Shared/exclusive gate over process-global state used by in-process cells.

Issue #90.  Studio's own cell runners capture streams per thread (see
:mod:`rlmstudio.envs.stream_capture`), so any number of them may execute
concurrently.  The official engine's ``rlms`` ``local`` environment is
different: inside that dependency, every code cell assigns
``sys.stdout``/``sys.stderr`` and changes the working directory for the whole
process.  A Studio cell running in that window has its output written into the
official cell's buffer, and its own captured result is incomplete.

The gate is the boundary between the two kinds of in-process execution:

* Studio's cell runners hold the **shared** side for the duration of a cell;
  any number may run at once.
* An official ``local`` run holds the **exclusive** side for the whole run; no
  Studio cell and no other official run may execute while it does.

The Docker environment and the subprocess sandbox mutate nothing in this
process and never acquire the gate.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

__all__ = ["acquire_exclusive", "release_exclusive", "shared_scope"]

_thread_state = threading.local()


class _SharedExclusiveGate:
    """A writer-preferring read/write lock.

    Writers are preferred so an official run cannot starve behind a steady
    stream of Studio cells.  The shared side is reentrant per thread: a cell
    that nests a capture must not queue behind a writer that is itself waiting
    for the outer capture to end.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._shared = 0
        self._exclusive = False
        self._waiting_exclusive = 0

    def _acquire_shared(self) -> None:
        depth = getattr(_thread_state, "shared_depth", 0)
        if depth == 0:
            with self._condition:
                while self._exclusive or self._waiting_exclusive:
                    self._condition.wait()
                self._shared += 1
        _thread_state.shared_depth = depth + 1

    def _release_shared(self) -> None:
        depth = getattr(_thread_state, "shared_depth", 1) - 1
        _thread_state.shared_depth = depth
        if depth == 0:
            with self._condition:
                self._shared -= 1
                if self._shared == 0:
                    self._condition.notify_all()

    def acquire_exclusive(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(timeout, 0.0)
        with self._condition:
            self._waiting_exclusive += 1
            try:
                while self._exclusive or self._shared:
                    if deadline is None:
                        self._condition.wait()
                        continue
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._condition.wait(remaining)
                self._exclusive = True
                return True
            finally:
                self._waiting_exclusive -= 1
                # A writer that gave up may have been the only reason readers
                # were queued rather than entering.
                self._condition.notify_all()

    def release_exclusive(self) -> None:
        with self._condition:
            if not self._exclusive:
                raise RuntimeError("the exclusive in-process scope is not held")
            self._exclusive = False
            self._condition.notify_all()


_gate = _SharedExclusiveGate()


@contextmanager
def shared_scope() -> Iterator[None]:
    """Hold the shared side of the gate while a Studio cell executes."""
    _gate._acquire_shared()
    try:
        yield
    finally:
        _gate._release_shared()


def acquire_exclusive(timeout: float | None = None) -> bool:
    """Try to hold the exclusive side, waiting at most *timeout* seconds.

    Returns ``False`` when the timeout elapses first.  The caller owns the
    exclusive side until it calls :func:`release_exclusive`.
    """
    return _gate.acquire_exclusive(timeout)


def release_exclusive() -> None:
    """Release the exclusive side acquired by :func:`acquire_exclusive`."""
    _gate.release_exclusive()
