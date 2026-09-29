"""Use case: run the paper authors' RLM implementation as an opaque engine.

The engine (see :class:`RLMEnginePort`) owns the whole loop, so Studio's
budgets are enforced around it rather than inside it:

- ``max_steps`` / ``max_recursion_depth`` / ``max_time_seconds`` are mapped
  onto the engine's own limits by the adapter.
- The wall-clock budget is *also* enforced here as a backstop.  The
  engine's timeout is checked between iterations, so a REPL execution that
  never returns (``while True: pass``) would otherwise hang the slot.
- Token and cost caps are checked after the run, because the engine only
  reports usage once it has finished.

Failures are shaped so :func:`classify_execution_outcome` categorises them
exactly like Studio's own loop: ``timeout`` / ``budget`` keywords in the
error or the ``⚠️`` answer prefix.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
import time
from typing import Any

from rlmstudio.application.dto import RunConfigDTO, RunResultDTO
from rlmstudio.application.ports.event_port import ExecutionEventEmitter
from rlmstudio.application.ports.rlm_engine_port import RLMEnginePort
from rlmstudio.application.sandbox_vars import (
    MODE_RLM_OFFICIAL,
    RESULT_KEY_COST_KNOWN,
    RESULT_KEY_ENGINE_VERSION,
)

logger = logging.getLogger(__name__)


class _WallClockExceededError(Exception):
    """Raised internally when the engine did not return within the wall-clock budget."""


class RunRLMOfficialUseCase:
    """Orchestrates one run of a third-party RLM engine under Studio's budgets.

    Args:
        engine: Engine port adapter (for example the ``rlms`` adapter).
    """

    def __init__(self, engine: RLMEnginePort) -> None:
        self._engine = engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        content: str,
        query: str,
        config: RunConfigDTO | None = None,
    ) -> RunResultDTO:
        """Run the engine synchronously.

        Args:
            content: Document text to analyze.
            query: User question about the content.
            config: Optional run configuration; ``mode`` is forced to
                ``rlm_official`` on the result.

        Returns:
            RunResultDTO with the engine's answer, metrics and raw trace, or
            a classified failure (unavailable engine, error, timeout).
        """
        config = config or RunConfigDTO(mode=MODE_RLM_OFFICIAL)
        start = time.time()

        unavailable = self._unavailable_result(start)
        if unavailable is not None:
            return unavailable

        try:
            result = self._run_within_wall_clock(content, query, config)
        except _WallClockExceededError:
            return self._timeout_result(config, start)
        except Exception as exc:
            return self._error_result(str(exc), start)

        return self._finalize(result, config, start)

    async def execute_async(
        self,
        content: str,
        query: str,
        config: RunConfigDTO | None = None,
        event_emitter: ExecutionEventEmitter | None = None,
    ) -> RunResultDTO:
        """Async run.  Trace entries and totals are emitted once the engine returns.

        The engine does not stream, so there are no ``on_token`` events; each
        raw trace entry is emitted through ``on_step`` after completion so
        WebSocket clients render the same steps as a synchronous run.
        """
        config = config or RunConfigDTO(mode=MODE_RLM_OFFICIAL)
        start = time.time()

        unavailable = self._unavailable_result(start)
        if unavailable is not None:
            return unavailable

        try:
            result = await asyncio.wait_for(
                self._engine.run_async(content, query, config),
                timeout=config.max_time_seconds,
            )
        except TimeoutError:
            return self._timeout_result(config, start)
        except Exception as exc:
            return self._error_result(str(exc), start)

        result = self._finalize(result, config, start)

        if event_emitter is not None:
            for entry in result.trace:
                await event_emitter.on_step(entry)
            await event_emitter.on_metrics(
                {
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "total_tokens": result.input_tokens + result.output_tokens,
                    "cost_usd": result.total_cost,
                    "steps": result.steps,
                    "elapsed_seconds": result.elapsed_time,
                }
            )
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_within_wall_clock(
        self, content: str, query: str, config: RunConfigDTO
    ) -> RunResultDTO:
        """Call ``engine.run`` and give up waiting after ``max_time_seconds``.

        The engine runs on a daemon thread that is *not* joined on timeout:
        the caller gets its classified result on time, the thread cannot
        block interpreter exit, and the engine's own timeout (mapped by the
        adapter) is what eventually stops it.
        """
        timeout = config.max_time_seconds
        if timeout is None:
            return self._engine.run(content, query, config)

        box: dict[str, Any] = {}

        def _target() -> None:
            try:
                box["result"] = self._engine.run(content, query, config)
            except BaseException as exc:  # re-raised on the caller's thread
                box["error"] = exc

        worker = threading.Thread(target=_target, name="rlm-engine", daemon=True)
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            logger.warning(
                "Official RLM engine exceeded the %.1fs wall-clock budget; "
                "abandoning the worker thread",
                timeout,
            )
            raise _WallClockExceededError
        if "error" in box:
            raise box["error"]
        return box["result"]

    def _finalize(self, result: RunResultDTO, config: RunConfigDTO, start: float) -> RunResultDTO:
        """Stamp mode/engine metadata and apply the post-run token/cost caps."""
        metadata = dict(result.metadata)
        metadata.setdefault(RESULT_KEY_ENGINE_VERSION, self._engine.version)
        answer = result.answer
        if result.success:
            breach = self._budget_breach(result, config)
            if breach is not None:
                # Degraded, not failed: the engine finished and the answer is
                # real, but ranking must not reward a run that blew the cap.
                # The ``⚠️`` prefix + "budget" is what the classifier keys on.
                answer = f"⚠️ **Budget exceeded** — {breach}.\n\n{answer}"
        return dataclasses.replace(
            result,
            answer=answer,
            mode_used=MODE_RLM_OFFICIAL,
            elapsed_time=result.elapsed_time or (time.time() - start),
            metadata=metadata,
        )

    @staticmethod
    def _budget_breach(result: RunResultDTO, config: RunConfigDTO) -> str | None:
        total_tokens = result.input_tokens + result.output_tokens
        if config.max_tokens is not None and total_tokens > config.max_tokens:
            return f"the engine used {total_tokens:,} tokens against a cap of {config.max_tokens:,}"
        if config.max_cost is not None and result.total_cost > config.max_cost:
            return (
                f"the engine spent ${result.total_cost:.4f} against a cap of ${config.max_cost:.4f}"
            )
        return None

    def _unavailable_result(self, start: float) -> RunResultDTO | None:
        available, reason = self._engine.is_available()
        if available:
            return None
        error = f"Official RLM engine unavailable: {reason}"
        return RunResultDTO(
            answer=f"⚠️ **Execution error**.\n\n{error}",
            mode_used=MODE_RLM_OFFICIAL,
            success=False,
            error=error,
            elapsed_time=time.time() - start,
            metadata={RESULT_KEY_ENGINE_VERSION: self._engine.version},
        )

    def _timeout_result(self, config: RunConfigDTO, start: float) -> RunResultDTO:
        seconds = config.max_time_seconds or 0.0
        error = f"Official RLM engine timed out after {seconds:.0f}s (wall-clock budget)"
        return RunResultDTO(
            answer=(
                f"⚠️ **RLM run timed out** after {seconds:.0f}s.\n\n"
                "The official engine did not finish within the wall-clock budget.\n\n"
                "**How to fix** — in Settings → Profile → Runtime Settings:\n"
                "- Increase **Timeout (s)**, or reduce **Max steps**."
            ),
            mode_used=MODE_RLM_OFFICIAL,
            success=False,
            error=error,
            elapsed_time=time.time() - start,
            # The engine was abandoned mid-run, so it had spent something this
            # guard cannot see.  Reporting $0 as a known cost would understate
            # exactly the runs that ran longest.
            metadata={
                RESULT_KEY_ENGINE_VERSION: self._engine.version,
                RESULT_KEY_COST_KNOWN: False,
            },
        )

    def _error_result(self, error: str, start: float) -> RunResultDTO:
        return RunResultDTO(
            answer=f"⚠️ **Execution error**.\n\n{error}",
            mode_used=MODE_RLM_OFFICIAL,
            success=False,
            error=error,
            elapsed_time=time.time() - start,
            # The engine may have raised part-way through a run it had already
            # paid for, so the cost is unknown rather than zero.
            metadata={
                RESULT_KEY_ENGINE_VERSION: self._engine.version,
                RESULT_KEY_COST_KNOWN: False,
            },
        )
