"""Bounded previews of registered results for the model's context.

A registered result stays complete in the :class:`ResultRegistry`; the
model sees at most ``cap`` characters of it per step, ending in a marker
that names the result id and the offset to continue from, so a follow-up
``read_result`` is unambiguous.
"""

from __future__ import annotations

from rlmstudio.application.services.result_registry import ResultRef, ResultRegistry


def truncation_marker(ref: ResultRef, shown_chars: int) -> str:
    """The one-line marker appended when a preview stops at ``shown_chars``."""
    note = " (stored copy cut at the spill budget)" if ref.truncated else ""
    return (
        f"\n... (preview truncated: {ref.length:,} chars total{note}; "
        f"read_result('{ref.id}', start={shown_chars}) for more)"
    )


def format_preview(registry: ResultRegistry, ref: ResultRef, *, cap: int) -> str:
    """The first ``cap`` characters of ``ref``'s text, marked when truncated."""
    if ref.length <= cap:
        whole: str = registry.head(ref.id, ref.length)
        return whole
    head: str = registry.head(ref.id, cap)
    return head + truncation_marker(ref, cap)
