"""Controller-side registry of RLM inspect and subcall results.

Every deterministic v2 action (an inspect tool, a subcall) produces a
*primary result*.  The loop registers the complete returned value here
before showing the model a bounded preview, so information can leave the
model's context without leaving the run's working state; the model reads
any part of it back through the ``read_result`` inspect tool.

Design (``specs/rlm-working-state``):

* ids ``r1, r2, …`` come from this registry's own counter, never from the
  step counter (which also counts nudges, retries and folded-in child
  steps); ``last`` is the most recent primary result and readbacks never
  change it;
* one canonical text per result: a ``str`` is stored unchanged, anything
  structured (``chunk()`` lists, batched subcall answers) as stable JSON;
  reads slice that text by **character** offset;
* an enforceable memory bound: a result above ``spill_result_above_bytes``,
  or displaced when the in-memory total would exceed ``max_registry_bytes``,
  is spilled to a per-run scratch directory (kept, not dropped) with a
  sparse character→byte checkpoint table so character reads stay exact on
  non-ASCII content; spilled results are evicted, oldest first and never
  ``last``, only when ``max_spill_bytes`` is exhausted.

The registry lives in the parent process and never touches a sandbox.
"""

from __future__ import annotations

import bisect
import json
import os
import shutil
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rlmstudio.application.sandbox_vars import (
    RLM_DEFAULT_MAX_REGISTRY_BYTES,
    RLM_DEFAULT_MAX_SPILL_BYTES,
    RLM_DEFAULT_SPILL_RESULT_ABOVE_BYTES,
)

LAST_RESULT_NAME = "last"
"""Alias the model may pass to ``read_result`` for the newest primary result."""

CHECKPOINT_EVERY_CHARS = 65_536
"""Spacing of the character→byte checkpoints recorded for a spilled result."""

_SCRATCH_PREFIX = "rlm-run-"

AccountingHook = Callable[[int, int], None]
"""``(in_memory_bytes, spilled_bytes)`` after every change; used by tests."""


@dataclass(frozen=True)
class ReadSlice:
    """One character-range read of a registered result.

    Attributes:
        ref: The result read.
        lo: First character index returned (clamped).
        hi: One past the last character returned (clamped, bounded by ``max_chars``).
        body: The characters ``[lo, hi)``.
    """

    ref: ResultRef
    lo: int
    hi: int
    body: str

    def footer(self, *, hi: int | None = None) -> str:
        """The continuation line for a read that stops before the end."""
        stop = self.hi if hi is None else hi
        if stop >= self.ref.length:
            return ""
        return (
            f"\n... (showing chars {self.lo}-{stop} of {self.ref.length:,} in {self.ref.id}; "
            f"read_result('{self.ref.id}', start={stop}) for more)"
        )


@dataclass(frozen=True)
class ResultRef:
    """Handle to one registered result.

    Attributes:
        id: ``r<K>``; stable for the life of the run.
        kind: ``inspect`` or ``subcall`` (what produced it).
        step: Controller step that produced it.
        length: Length of the canonical text in characters.
        byte_length: Length of the canonical text in UTF-8 bytes.
        spilled: Whether the text currently lives on disk.
        truncated: Whether the text was cut to fit ``max_spill_bytes``.
    """

    id: str
    kind: str
    step: int
    length: int
    byte_length: int
    spilled: bool = False
    truncated: bool = False


def canonical_text(value: Any) -> str:
    """Return the one text form a result is stored and sliced as."""
    if isinstance(value, str):
        return value
    if isinstance(value, list | dict | tuple):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


class _Entry:
    __slots__ = ("checkpoints", "evicted_at_step", "path", "ref", "text")

    def __init__(self, ref: ResultRef, text: str) -> None:
        self.ref = ref
        self.text: str | None = text
        self.path: Path | None = None
        # (char_index, byte_offset) pairs, ascending; only for spilled entries.
        self.checkpoints: list[tuple[int, int]] = []
        self.evicted_at_step: int | None = None

    @property
    def evicted(self) -> bool:
        return self.evicted_at_step is not None


class ResultRegistry:
    """Bounded, spill-backed store of a run's primary results."""

    def __init__(
        self,
        *,
        spill_result_above_bytes: int = RLM_DEFAULT_SPILL_RESULT_ABOVE_BYTES,
        max_registry_bytes: int = RLM_DEFAULT_MAX_REGISTRY_BYTES,
        max_spill_bytes: int = RLM_DEFAULT_MAX_SPILL_BYTES,
        scratch_dir: str | Path | None = None,
        on_account: AccountingHook | None = None,
    ) -> None:
        if spill_result_above_bytes <= 0 or max_registry_bytes <= 0 or max_spill_bytes <= 0:
            raise ValueError("registry byte limits must be positive")
        self._spill_above = spill_result_above_bytes
        self._max_memory = max_registry_bytes
        self._max_spill = max_spill_bytes
        self._scratch_root = Path(scratch_dir) if scratch_dir is not None else None
        self._scratch: Path | None = None
        self._owns_scratch = False
        self._on_account = on_account
        self._lock = threading.RLock()
        self._entries: dict[str, _Entry] = {}
        self._counter = 0
        self._last_id: str | None = None
        self._in_memory_bytes = 0
        self._spilled_bytes = 0
        self._closed = False

    # -- introspection -------------------------------------------------------

    @property
    def last(self) -> ResultRef | None:
        """The most recent primary result, or ``None`` before the first one."""
        with self._lock:
            return self._entries[self._last_id].ref if self._last_id else None

    @property
    def in_memory_bytes(self) -> int:
        return self._in_memory_bytes

    @property
    def spilled_bytes(self) -> int:
        return self._spilled_bytes

    def ids(self) -> list[str]:
        """Registered ids in creation order (evicted ids included)."""
        with self._lock:
            return list(self._entries)

    def describe(self, name: str) -> ResultRef | None:
        """The ref for ``name`` (``last`` or ``r<K>``), or ``None`` if unknown."""
        with self._lock:
            entry = self._resolve(name)
            return entry.ref if entry is not None else None

    # -- registration --------------------------------------------------------

    def register(self, value: Any, *, kind: str, step: int) -> ResultRef:
        """Store ``value`` as the next primary result and make it ``last``."""
        text = canonical_text(value)
        truncated = False
        byte_length = len(text.encode("utf-8"))
        if byte_length > self._max_spill:
            text, byte_length = _truncate_to_bytes(text, self._max_spill)
            truncated = True
        with self._lock:
            self._ensure_open()
            self._counter += 1
            ref = ResultRef(
                id=f"r{self._counter}",
                kind=kind,
                step=step,
                length=len(text),
                byte_length=byte_length,
                truncated=truncated,
            )
            entry = _Entry(ref, text)
            self._entries[ref.id] = entry
            self._last_id = ref.id
            # Count it in memory first so ``_spill`` always subtracts what
            # was added, whether it spills now or is displaced later.
            self._in_memory_bytes += byte_length
            if byte_length > self._spill_above:
                self._spill(entry)
            else:
                self._enforce_memory_bound()
            self._enforce_spill_bound()
            self._account()
            return entry.ref

    # -- reads ----------------------------------------------------------------

    def read_slice(
        self,
        name: str,
        start: int = 0,
        end: int | None = None,
        max_chars: int = 10000,
    ) -> ReadSlice | str:
        """Slice a result by character offset (Python slice semantics, clamped).

        Unknown or evicted names yield a one-line ``Error: …`` string instead
        of raising, so the model sees the problem as an ordinary execution
        result.
        """
        with self._lock:
            entry = self._resolve(name)
            if entry is None:
                known = f"r1..r{self._counter}" if self._counter else "none yet"
                return f"Error: unknown result {name!r} (registered results: {known})"
            if entry.evicted:
                return (
                    f"Error: {entry.ref.id} was evicted "
                    f"(spill budget exhausted at step {entry.evicted_at_step})"
                )
            lo, hi = _clamp_slice(start, end, entry.ref.length)
            if max_chars > 0 and hi - lo > max_chars:
                hi = lo + max_chars
            return ReadSlice(ref=entry.ref, lo=lo, hi=hi, body=self._slice(entry, lo, hi))

    def read(
        self,
        name: str,
        start: int = 0,
        end: int | None = None,
        max_chars: int = 10000,
    ) -> str:
        """:meth:`read_slice` rendered as text: the body plus a continuation footer."""
        piece = self.read_slice(name, start=start, end=end, max_chars=max_chars)
        if isinstance(piece, str):
            return piece
        return piece.body + piece.footer()

    def head(self, name: str, n: int) -> str:
        """The first ``n`` characters of a result, no footer (for previews)."""
        with self._lock:
            entry = self._resolve(name)
            if entry is None or entry.evicted:
                return ""
            return self._slice(entry, 0, min(n, entry.ref.length))

    def get(self, name: str) -> str | None:
        """The whole canonical text, or ``None`` if unknown or evicted."""
        with self._lock:
            entry = self._resolve(name)
            if entry is None or entry.evicted:
                return None
            return self._slice(entry, 0, entry.ref.length)

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Drop every result and remove the scratch directory (idempotent)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._entries.clear()
            self._in_memory_bytes = 0
            self._spilled_bytes = 0
            if self._scratch is not None and self._owns_scratch:
                shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None
            self._account()

    def __enter__(self) -> ResultRegistry:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals ----------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ResultRegistry is closed")

    def _resolve(self, name: str) -> _Entry | None:
        key = (name or "").strip()
        if key == LAST_RESULT_NAME:
            key = self._last_id or ""
        return self._entries.get(key)

    def _account(self) -> None:
        if self._on_account is not None:
            self._on_account(self._in_memory_bytes, self._spilled_bytes)

    def _scratch_dir(self) -> Path:
        if self._scratch is None:
            if self._scratch_root is not None:
                self._scratch_root.mkdir(parents=True, exist_ok=True)
                self._scratch = Path(
                    tempfile.mkdtemp(prefix=_SCRATCH_PREFIX, dir=self._scratch_root)
                )
            else:
                self._scratch = Path(tempfile.mkdtemp(prefix=_SCRATCH_PREFIX))
            self._owns_scratch = True
        return self._scratch

    def _spill(self, entry: _Entry) -> None:
        """Move an in-memory entry to disk, recording checkpoints as it goes."""
        text = entry.text
        if text is None:
            return
        path = self._scratch_dir() / f"{entry.ref.id}.txt"
        tmp = path.with_suffix(".tmp")
        checkpoints: list[tuple[int, int]] = []
        byte_offset = 0
        with open(tmp, "wb") as fh:
            for char_index in range(0, len(text), CHECKPOINT_EVERY_CHARS):
                checkpoints.append((char_index, byte_offset))
                data = text[char_index : char_index + CHECKPOINT_EVERY_CHARS].encode("utf-8")
                fh.write(data)
                byte_offset += len(data)
            if not checkpoints:
                checkpoints.append((0, 0))
        os.replace(tmp, path)
        if not entry.ref.spilled:
            self._in_memory_bytes -= entry.ref.byte_length
        self._spilled_bytes += entry.ref.byte_length
        entry.text = None
        entry.path = path
        entry.checkpoints = checkpoints
        entry.ref = replace(entry.ref, spilled=True)

    def _enforce_memory_bound(self) -> None:
        """Spill oldest in-memory entries until the in-memory total fits."""
        if self._in_memory_bytes <= self._max_memory:
            return
        for entry in self._entries.values():
            if self._in_memory_bytes <= self._max_memory:
                break
            if entry.text is not None and not entry.evicted:
                self._spill(entry)

    def _enforce_spill_bound(self) -> None:
        """Evict oldest spilled entries (never ``last``) until spill fits."""
        if self._spilled_bytes <= self._max_spill:
            return
        for entry in self._entries.values():
            if self._spilled_bytes <= self._max_spill:
                break
            if entry.ref.id == self._last_id or entry.evicted or entry.path is None:
                continue
            self._evict(entry)

    def _evict(self, entry: _Entry) -> None:
        if entry.path is not None:
            try:
                entry.path.unlink()
            except OSError:
                pass
        self._spilled_bytes -= entry.ref.byte_length
        entry.path = None
        entry.checkpoints = []
        entry.evicted_at_step = self._current_step()

    def _current_step(self) -> int:
        return self._entries[self._last_id].ref.step if self._last_id else 0

    def _slice(self, entry: _Entry, lo: int, hi: int) -> str:
        if hi <= lo:
            return ""
        if entry.text is not None:
            return entry.text[lo:hi]
        if entry.path is None:
            return ""
        chars = [c for c, _ in entry.checkpoints]
        idx = max(0, bisect.bisect_right(chars, lo) - 1)
        cp_char, cp_byte = entry.checkpoints[idx]
        needed = hi - cp_char
        with open(entry.path, "rb") as fh:
            fh.seek(cp_byte)
            # Every character is at most 4 UTF-8 bytes, so this covers the
            # first ``needed`` characters; a partial trailing character is
            # beyond the requested range and is dropped by ``ignore``.
            data = fh.read(needed * 4)
        decoded = data.decode("utf-8", errors="ignore")
        return decoded[lo - cp_char : hi - cp_char]


def _clamp_slice(start: int, end: int | None, length: int) -> tuple[int, int]:
    lo = start if start >= 0 else max(0, length + start)
    hi = length if end is None else (end if end >= 0 else max(0, length + end))
    lo = max(0, min(lo, length))
    hi = max(lo, min(hi, length))
    return lo, hi


def _truncate_to_bytes(text: str, limit: int) -> tuple[str, int]:
    """Cut ``text`` so its UTF-8 encoding is at most ``limit`` bytes."""
    data = text.encode("utf-8")[:limit]
    cut = data.decode("utf-8", errors="ignore")
    return cut, len(cut.encode("utf-8"))
