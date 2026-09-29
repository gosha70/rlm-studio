"""RLM engine port: interface for running a whole RLM loop as an opaque engine.

Studio's own loop (``RunRLMUseCase``) drives an :class:`LLMPort` and a
:class:`SandboxPort` step by step.  A third-party engine — the paper
authors' ``rlms`` package is the first — owns its loop end to end, so it is
modelled as a single port that takes Studio's run configuration and returns
Studio's run result.  See ``specs/interop-official-rlm``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rlmstudio.application.dto import RunConfigDTO, RunResultDTO


@runtime_checkable
class RLMEnginePort(Protocol):
    """Protocol for third-party RLM engines.

    Implementations map the :class:`RunConfigDTO` budgets onto the engine's
    own limits (``max_steps`` → iteration limit, ``max_recursion_depth`` →
    depth, ``max_time_seconds`` → the engine's timeout) and return a
    :class:`RunResultDTO` whose ``trace`` uses the raw trace shape defined in
    ``application/sandbox_vars.py`` (``TRACE_KEY_*``), so Traces, replay,
    telemetry and ranking work unchanged.
    """

    @property
    def version(self) -> str | None:
        """Engine package version, or ``None`` when it cannot be determined."""
        ...

    def is_available(self) -> tuple[bool, str]:
        """Report whether the engine can run in this process.

        Returns:
            ``(available, reason)``.  ``reason`` explains an unavailable
            engine (for example, the optional extra is not installed) and
            is shown to users verbatim.
        """
        ...

    def run(self, content: str, query: str, config: RunConfigDTO) -> RunResultDTO:
        """Run the engine's full loop over *content* for *query*.

        Args:
            content: Document text the engine explores.
            query: User question about the content.
            config: Run configuration; budgets are mapped onto the engine.

        Returns:
            RunResultDTO with the answer, token/cost totals and raw trace.
        """
        ...

    async def run_async(self, content: str, query: str, config: RunConfigDTO) -> RunResultDTO:
        """Async version of :meth:`run`.

        Implementations may wrap the sync method with ``asyncio.to_thread``.
        """
        ...
