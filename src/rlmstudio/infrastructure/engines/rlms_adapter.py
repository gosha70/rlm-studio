"""``rlms`` engine adapter: the paper authors' RLM implementation behind RLMEnginePort.

``rlms`` (https://github.com/alexzhang13/rlm) is an optional dependency — the
``interop`` extra.  It is imported lazily so the rest of Studio never pays for
it, and :meth:`RlmsEngineAdapter.is_available` reports a user-facing reason
when it is missing or the provider cannot be mapped.

Provider mapping (spec G5): Studio's ``openai`` / ``anthropic`` backends use the
engine's native clients; the OpenAI-compatible local servers (``lmstudio``,
``vllm``, ``ollama``) go through the engine's ``openai`` client with
``base_url`` pointing at the server's ``/v1`` endpoint.

Trace mapping (spec FR-3): every code block in the engine's trajectory becomes
one ``assistant`` entry (the model's response, with the code) followed by one
``execution`` entry (stdout/stderr); nested ``llm_query`` sub-calls become
extra ``execution`` entries carrying ``recursion_depth``, and a sub-call that
ran its own REPL loop contributes that loop's entries at its own depth; the
final answer closes the trace as a last ``assistant`` entry so the normaliser
in ``server/routes/_helpers.py`` promotes it to ``final``.

Known limitations, recorded on every result under ``engine_notes``: the
engine does not stream (no TTFT / decode timings), it reports token usage per
run rather than per step, and its ``local`` environment executes model-written
code in-process — a runaway loop there cannot be stopped, only abandoned by
the use case's wall-clock guard.  That environment also mutates process-global
state (``sys.stdout``/``sys.stderr`` and the working directory) around every
code cell under a per-instance lock, so this adapter serialises ``local`` runs
process-wide; see :data:`_LOCAL_ENV_LOCK`.  Select the Docker sandbox to run
the engine in a container instead, which removes both limits.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import importlib.metadata
import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any

from rlmstudio.application.dto import RunConfigDTO, RunResultDTO
from rlmstudio.application.sandbox_vars import (
    MODE_RLM_OFFICIAL,
    RESULT_KEY_COST_KNOWN,
    RESULT_KEY_ENGINE_NOTES,
    RESULT_KEY_ENGINE_VERSION,
    TRACE_KEY_CODE,
    TRACE_KEY_CONTENT,
    TRACE_KEY_ELAPSED_SECONDS,
    TRACE_KEY_INPUT_TOKENS,
    TRACE_KEY_MODE,
    TRACE_KEY_MODEL,
    TRACE_KEY_OUTPUT_TOKENS,
    TRACE_KEY_RECURSION_DEPTH,
    TRACE_KEY_ROLE,
    TRACE_KEY_STEP,
)

logger = logging.getLogger(__name__)

RLMS_DISTRIBUTION = "rlms"
RLMS_MODULE = "rlm"
INTEROP_EXTRA = "interop"
UNAVAILABLE_REASON = (
    f"The official RLM engine needs the `{INTEROP_EXTRA}` extra: "
    f'pip install "rlm-studio[{INTEROP_EXTRA}]"'
)

# Studio provider backends the engine speaks natively (Studio key → rlms backend).
_NATIVE_BACKENDS: dict[str, str] = {"openai": "openai", "anthropic": "anthropic"}
# Environment variable each native backend's key is read from when the provider
# instance carries none.  The engine's OpenAI client only reads the environment
# at *import* time, so the key is resolved here and passed in explicitly.
_ENV_VAR_BY_NATIVE_BACKEND: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
# Studio backends reached through the engine's OpenAI client plus a base_url.
_OPENAI_COMPATIBLE_BACKENDS: frozenset[str] = frozenset({"lmstudio", "vllm", "ollama"})
_RLMS_OPENAI_BACKEND = "openai"
_OPENAI_V1_SUFFIX = "/v1"
# Local servers ignore the key, but the OpenAI SDK refuses to start without one.
_PLACEHOLDER_API_KEY = "not-needed"

SANDBOX_TYPE_DOCKER = "docker"
_ENV_LOCAL = "local"
_ENV_DOCKER = "docker"

# Studio counts *levels of sub-RLM*; the engine counts the depth at which a node
# stops looping and degrades to a plain model call.  See _engine_max_depth.
_RLMS_DEPTH_OFFSET = 1
_RLMS_MIN_DEPTH = 1

# The engine's in-process ``local`` environment swaps ``sys.stdout`` /
# ``sys.stderr`` and chdirs into a temp directory around every code cell, under
# a lock that only covers one instance.  Two concurrent slots would therefore
# restore each other's streams and leave the server in a deleted directory, so
# every ``local`` run in this process takes this lock instead.
_LOCAL_ENV_LOCK = threading.Lock()
# Waiting cap for a slot that has no wall-clock budget of its own.
_LOCAL_ENV_LOCK_TIMEOUT_SECONDS = 30.0
# Share of a slot's budget it may spend queueing, so a run that does get the
# REPL still has most of its budget to use it with.
_LOCAL_ENV_LOCK_WAIT_FRACTION = 0.25
# The engine's own timeout is set this far inside the caller's wall-clock guard
# so the engine stops itself first and can report what it spent; the guard stays
# as the backstop for a REPL stuck inside a single iteration.
_ENGINE_TIMEOUT_MARGIN_FRACTION = 0.1
_ENGINE_TIMEOUT_MARGIN_CAP_SECONDS = 5.0
# The same global state rules out the engine's parallel batched sub-calls, which
# would otherwise give several in-process REPLs to one run: ``llm_query_batched``
# fans out over a thread pool, and every child that is allowed its own REPL
# swaps this process's streams and directory.  One at a time inside a run, too.
_LOCAL_MAX_CONCURRENT_SUBCALLS = 1

# Deepest sub-call nesting converted into trace entries.  The trajectory comes
# from another library, so the walk is bounded rather than trusting its shape.
MAX_TRACE_RECURSION_DEPTH = 8

NOTE_NO_STREAMING = (
    "The official engine does not stream: TTFT and decode timings are unavailable for this slot."
)
NOTE_NO_STEP_TOKENS = (
    "The official engine reports token usage per run, not per step; step-level "
    "token counts are shown as 0 and the totals are exact."
)
NOTE_LOCAL_ENV = (
    "The official engine ran its REPL in-process (rlms 'local' environment), which is "
    "not isolated and redirects this process's output and working directory while each "
    "code cell runs; official runs and their sub-calls are therefore serialised one at "
    "a time. Select the Docker sandbox in Settings to isolate them and run them in "
    "parallel."
)
NOTE_COST_UNKNOWN = (
    "No price is known for this model; cost is shown as 0 and the slot is excluded "
    "from cost ranking."
)
NOTE_NO_SUBCALL_FLOOR = (
    "Recursion is set to 0 levels, which the official engine cannot enforce: its "
    "sub-calls degrade to plain model calls instead of being refused."
)
NOTE_INTERRUPTED_TOTALS = (
    "The run was stopped by a cap, so the engine reported one combined token total "
    "instead of an input/output split; the total is recorded as input tokens."
)
NOTE_INTERRUPTED_TOKENS_UNKNOWN = (
    "The run was stopped by a cap that reported no token count, so this run's tokens "
    "are shown as 0; the cost, where known, is the figure the cap was checked against."
)
BUSY_REASON = (
    "Another official RLM run is still holding the in-process REPL, which only one run "
    "can use at a time. Select the Docker sandbox in Settings to run official slots "
    "side by side."
)


def _installed_version() -> str | None:
    try:
        return importlib.metadata.version(RLMS_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return None


def rlms_availability() -> tuple[bool, str, str | None]:
    """Package-level availability of the engine: ``(available, reason, version)``.

    Provider mapping is not involved — that is per slot and reported by
    :meth:`RlmsEngineAdapter.is_available`.  ``reason`` is user-facing either
    way: the version when installed, the install hint when not.
    """
    try:
        importlib.import_module(RLMS_MODULE)
    except ImportError:
        return False, UNAVAILABLE_REASON, None
    version = _installed_version()
    return True, f"{RLMS_DISTRIBUTION} {version or 'unknown'}", version


def _openai_v1(url: str) -> str:
    """Return *url* with the ``/v1`` suffix the OpenAI SDK expects (idempotent)."""
    trimmed = url.rstrip("/")
    return trimmed if trimmed.endswith(_OPENAI_V1_SUFFIX) else trimmed + _OPENAI_V1_SUFFIX


def engine_max_depth(max_recursion_depth: int) -> int:
    """Translate Studio's recursion budget into the engine's ``max_depth``.

    Studio counts *levels of sub-RLM*: ``max_recursion_depth=1`` lets the root
    loop spawn one child that runs its own REPL loop.  ``rlms`` instead counts
    the depth at which a node stops looping and becomes a plain model call —
    ``RLM.completion`` falls back when ``depth >= max_depth`` and the root node
    is at ``depth == 0`` — so the root loop already costs one level and
    Studio's *N* maps to *N + 1*.

    ``max_depth=0`` must never reach the engine: it would skip the loop
    entirely, answer from the document without ever being shown the question,
    and return a bare string where the adapter expects a completion object.
    Studio's 0 (no sub-calls at all) is therefore clamped to the engine's
    minimum, where sub-calls degrade to plain model calls; the result carries
    :data:`NOTE_NO_SUBCALL_FLOOR` to say so.
    """
    return max(_RLMS_MIN_DEPTH, max_recursion_depth + _RLMS_DEPTH_OFFSET)


def engine_timeout(max_time_seconds: float | None) -> float | None:
    """Give the engine a deadline just inside the caller's wall-clock budget.

    Both clocks measure the same budget, but the use case's guard starts first,
    so an equal deadline means the guard always wins — and the guard can only
    abandon the run, reporting no tokens, no cost and no trace.  Stopping the
    engine a little earlier instead lets it raise its own timeout, which carries
    the usage and the trajectory to report.  The guard remains the backstop for
    the case the engine cannot handle: a REPL execution stuck inside one
    iteration, which it never gets to check between.
    """
    if max_time_seconds is None:
        return None
    margin = min(
        max_time_seconds * _ENGINE_TIMEOUT_MARGIN_FRACTION,
        _ENGINE_TIMEOUT_MARGIN_CAP_SECONDS,
    )
    return max(max_time_seconds - margin, max_time_seconds / 2)


def sum_usage(usage: dict[str, Any]) -> tuple[int, int, float | None]:
    """Total ``(input_tokens, output_tokens, cost)`` from an ``rlms`` ``UsageSummary`` dict.

    ``cost`` is ``None`` when the engine reported no price for any model.
    """
    input_tokens = 0
    output_tokens = 0
    per_model_costs: list[float] = []
    for model_usage in (usage.get("model_usage_summaries") or {}).values():
        input_tokens += int(model_usage.get("total_input_tokens") or 0)
        output_tokens += int(model_usage.get("total_output_tokens") or 0)
        if model_usage.get("total_cost") is not None:
            per_model_costs.append(float(model_usage["total_cost"]))
    if usage.get("total_cost") is not None:
        return input_tokens, output_tokens, float(usage["total_cost"])
    return input_tokens, output_tokens, (sum(per_model_costs) if per_model_costs else None)


def trajectory_to_trace(
    trajectory: dict[str, Any] | None,
    *,
    final_answer: str,
    model: str,
) -> list[dict[str, Any]]:
    """Translate an ``RLMLogger.get_trajectory()`` dict into Studio's raw trace shape.

    Per-step token counts are unknown (the engine reports totals only) and are
    recorded as 0; nested sub-calls carry their own usage.

    Sub-calls nest: a child that was allowed to run its own REPL loop records
    that loop under its ``metadata`` (the engine attaches the child's full
    trajectory there).  Its iterations are converted at the child's depth,
    ahead of the entry carrying the child's answer, so a trace reads
    parent-step → child-steps → child-answer.  The walk stops at
    :data:`MAX_TRACE_RECURSION_DEPTH`; deeper children still contribute their
    answer entry, just not their internal steps.
    """
    entries: list[dict[str, Any]] = []

    def _entry(role: str, content: str, *, depth: int = 0, **extra: Any) -> None:
        entries.append(
            {
                TRACE_KEY_STEP: len(entries),
                TRACE_KEY_ROLE: role,
                TRACE_KEY_CONTENT: content,
                TRACE_KEY_MODE: MODE_RLM_OFFICIAL,
                TRACE_KEY_INPUT_TOKENS: 0,
                TRACE_KEY_OUTPUT_TOKENS: 0,
                TRACE_KEY_MODEL: model,
                TRACE_KEY_ELAPSED_SECONDS: 0.0,
                **({TRACE_KEY_RECURSION_DEPTH: depth} if depth else {}),
                **extra,
            }
        )

    def _walk(iterations: Any, *, node_model: str, depth: int) -> None:
        for iteration in iterations or []:
            response = iteration.get("response") or ""
            blocks = iteration.get("code_blocks") or []
            if not blocks:
                _entry(
                    "assistant",
                    response,
                    depth=depth,
                    **{
                        TRACE_KEY_MODEL: node_model,
                        TRACE_KEY_ELAPSED_SECONDS: iteration.get("iteration_time") or 0.0,
                    },
                )
                continue
            for index, block in enumerate(blocks):
                code = block.get("code") or ""
                _entry(
                    "assistant",
                    response if index == 0 else code,
                    depth=depth,
                    **{TRACE_KEY_MODEL: node_model, TRACE_KEY_CODE: code},
                )
                result = block.get("result") or {}
                stdout = result.get("stdout") or ""
                stderr = (result.get("stderr") or "").strip()
                output = f"{stdout}\n{stderr}".strip() if stderr else stdout
                _entry(
                    "execution",
                    output,
                    depth=depth,
                    **{
                        TRACE_KEY_MODEL: node_model,
                        TRACE_KEY_ELAPSED_SECONDS: result.get("execution_time") or 0.0,
                    },
                    **({"error": stderr} if stderr else {}),
                )
                for call in result.get("rlm_calls") or []:
                    _walk_call(call, parent_model=node_model, depth=depth + 1)

    def _walk_call(call: dict[str, Any], *, parent_model: str, depth: int) -> None:
        call_model = call.get("root_model") or parent_model
        call_in, call_out, _ = sum_usage(call.get("usage_summary") or {})
        nested = call.get("metadata")
        if isinstance(nested, dict) and depth < MAX_TRACE_RECURSION_DEPTH:
            _walk(nested.get("iterations"), node_model=call_model, depth=depth)
        _entry(
            "execution",
            call.get("response") or "",
            depth=depth,
            **{
                TRACE_KEY_MODEL: call_model,
                TRACE_KEY_INPUT_TOKENS: call_in,
                TRACE_KEY_OUTPUT_TOKENS: call_out,
                TRACE_KEY_ELAPSED_SECONDS: call.get("execution_time") or 0.0,
            },
        )

    _walk((trajectory or {}).get("iterations"), node_model=model, depth=0)

    # Close with the final answer so the last step promotes to ``final`` —
    # unless the last iteration already was that answer.
    last = entries[-1] if entries else None
    if not (
        last is not None
        and last[TRACE_KEY_ROLE] == "assistant"
        and last[TRACE_KEY_CONTENT] == final_answer
    ):
        _entry("assistant", final_answer)
    return entries


class RlmsEngineAdapter:
    """Runs the ``rlms`` engine for one Studio provider/model under Studio's budgets.

    Args:
        backend: Studio provider backend key (``openai``, ``anthropic``,
            ``lmstudio``, ``vllm``, ``ollama``).
        model: Bare model identifier as the provider knows it.
        api_key: API key for cloud providers; ignored by local servers.
        base_url: Endpoint of a local OpenAI-compatible server.
        sandbox_type: Studio sandbox setting; ``docker`` maps to the engine's
            Docker environment, everything else to its in-process ``local`` one.
        docker_image: Image for the Docker environment (engine default if ``None``).
        cost_fn: ``(input_tokens, output_tokens) -> USD`` from the slot's LLM
            adapter, used when the engine reports no price.
    """

    def __init__(
        self,
        *,
        backend: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        sandbox_type: str = "restricted",
        docker_image: str | None = None,
        cost_fn: Callable[[int, int], float] | None = None,
    ) -> None:
        self._backend = backend
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._sandbox_type = sandbox_type
        self._docker_image = docker_image
        self._cost_fn = cost_fn

    # ------------------------------------------------------------------
    # RLMEnginePort
    # ------------------------------------------------------------------

    @property
    def version(self) -> str | None:
        return _installed_version()

    def is_available(self) -> tuple[bool, str]:
        available, reason, _version = rlms_availability()
        if not available:
            return False, reason
        try:
            self._engine_backend()
        except ValueError as exc:
            return False, str(exc)
        return True, reason

    def run(self, content: str, query: str, config: RunConfigDTO) -> RunResultDTO:
        environment, environment_kwargs = self._engine_environment()
        if environment != _ENV_LOCAL:
            return self._run_engine(content, query, config, environment, environment_kwargs)

        # The in-process REPL mutates process-global state, so only one run at a
        # time.  The wait is a *slice* of the slot's budget, never the whole of
        # it: the use case's wall-clock guard is counting the same budget from
        # earlier, so waiting the full amount would guarantee the caller gave up
        # first — the queued run would never get to report itself busy, and its
        # abandoned thread would then run a fresh full-length engine call that
        # nobody is waiting for while still holding this lock.
        queued_at = time.time()
        if not _LOCAL_ENV_LOCK.acquire(timeout=self._lock_wait(config)):
            logger.warning("Official RLM engine busy: in-process REPL still held")
            # ``queued_at`` so the failed slot reports the time it spent waiting.
            return self._failed(BUSY_REASON, queued_at, environment, config)
        try:
            queued_config = self._config_after_queueing(config, queued_at)
            return self._run_engine(content, query, queued_config, environment, environment_kwargs)
        finally:
            _LOCAL_ENV_LOCK.release()

    @staticmethod
    def _lock_wait(config: RunConfigDTO) -> float:
        """How long a queued run may wait for the in-process REPL."""
        budget = config.max_time_seconds
        if budget is None:
            return _LOCAL_ENV_LOCK_TIMEOUT_SECONDS
        return float(min(budget * _LOCAL_ENV_LOCK_WAIT_FRACTION, _LOCAL_ENV_LOCK_TIMEOUT_SECONDS))

    @staticmethod
    def _config_after_queueing(config: RunConfigDTO, queued_at: float) -> RunConfigDTO:
        """Charge the time spent queueing to the run's budget.

        Without this the engine would be handed the full budget after part of it
        had already gone on waiting, and would still be working for a caller
        that had given up.  Waiting can only ever consume the slice
        :meth:`_lock_wait` allows, so what is left is always enough to attempt
        the run.
        """
        budget = config.max_time_seconds
        if budget is None:
            return config
        remaining = max(budget - (time.time() - queued_at), 0.0)
        return dataclasses.replace(config, max_time_seconds=remaining)

    def _run_engine(
        self,
        content: str,
        query: str,
        config: RunConfigDTO,
        environment: str,
        environment_kwargs: dict[str, Any],
    ) -> RunResultDTO:
        rlm = importlib.import_module(RLMS_MODULE)
        rlm_logger = importlib.import_module(f"{RLMS_MODULE}.logger")
        backend, backend_kwargs = self._engine_backend()

        trajectory = rlm_logger.RLMLogger()
        engine_kwargs: dict[str, Any] = {}
        if environment == _ENV_LOCAL:
            engine_kwargs["max_concurrent_subcalls"] = _LOCAL_MAX_CONCURRENT_SUBCALLS
        engine = rlm.RLM(
            backend=backend,
            backend_kwargs=backend_kwargs,
            environment=environment,
            environment_kwargs=environment_kwargs,
            max_depth=engine_max_depth(config.max_recursion_depth),
            max_iterations=config.max_steps,
            max_timeout=engine_timeout(config.max_time_seconds),
            max_tokens=config.max_tokens,
            max_budget=config.max_cost,
            logger=trajectory,
            verbose=False,
            **engine_kwargs,
        )
        start = time.time()
        try:
            # ``prompt`` is the context the REPL exposes; ``root_prompt`` is the question.
            completion = engine.completion(prompt=content, root_prompt=query)
        except (rlm.BudgetExceededError, rlm.TokenLimitExceededError) as exc:
            return self._interrupted(
                f"Budget exceeded: {exc}", exc, trajectory, start, environment, config
            )
        except rlm.TimeoutExceededError as exc:
            return self._interrupted(
                f"Official RLM engine timed out: {exc}", exc, trajectory, start, environment, config
            )
        finally:
            engine.close()
        return self._to_result(completion, start, environment, config)

    async def run_async(self, content: str, query: str, config: RunConfigDTO) -> RunResultDTO:
        """Run on a daemon thread and await the result.

        Deliberately not ``asyncio.to_thread``: that borrows the loop's default
        executor, whose threads are **not** daemons, so a run stuck in the
        engine's in-process REPL would be joined at interpreter exit and hang
        the server's shutdown.  The caller's timeout (the use case wraps this
        in ``asyncio.wait_for``) abandons the thread instead, matching what the
        synchronous path already does.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[RunResultDTO] = loop.create_future()

        def _settle(setter: Any, value: Any) -> None:
            # The awaiting side may already have timed out and cancelled.
            if not future.done():
                setter(value)

        def _target() -> None:
            try:
                result = self.run(content, query, config)
            except BaseException as exc:  # delivered to the awaiting caller
                self._hand_back(loop, _settle, future.set_exception, exc)
            else:
                self._hand_back(loop, _settle, future.set_result, result)

        threading.Thread(target=_target, name="rlm-engine-async", daemon=True).start()
        return await future

    @staticmethod
    def _hand_back(
        loop: asyncio.AbstractEventLoop,
        settle: Any,
        setter: Any,
        value: Any,
    ) -> None:
        """Deliver a worker-thread outcome to the loop, tolerating a closed loop."""
        try:
            loop.call_soon_threadsafe(settle, setter, value)
        except RuntimeError:
            # The loop is gone (server shut down while the engine was running);
            # nobody is waiting for this result any more.
            logger.debug("Official RLM engine finished after its event loop closed")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _engine_backend(self) -> tuple[str, dict[str, Any]]:
        """Map Studio's provider onto an ``rlms`` backend + client kwargs, or raise ``ValueError``."""
        if self._backend in _NATIVE_BACKENDS:
            env_var = _ENV_VAR_BY_NATIVE_BACKEND[self._backend]
            # Both native backends are treated alike: a missing key is reported
            # rather than left for the engine to discover.  The environment is
            # read here because the engine's OpenAI client captured it at import
            # time and would not see a key exported afterwards.
            api_key = self._api_key or os.environ.get(env_var)
            if not api_key:
                raise ValueError(
                    f"The official RLM engine needs an API key for the {self._backend!r} "
                    f"provider; set one in Settings → LLM Providers or export {env_var}."
                )
            return _NATIVE_BACKENDS[self._backend], {
                "model_name": self._model,
                "api_key": api_key,
            }
        if self._backend in _OPENAI_COMPATIBLE_BACKENDS:
            if not self._base_url:
                raise ValueError(
                    f"The official RLM engine needs an endpoint for the {self._backend!r} provider."
                )
            return _RLMS_OPENAI_BACKEND, {
                "model_name": self._model,
                "base_url": _openai_v1(self._base_url),
                "api_key": self._api_key or _PLACEHOLDER_API_KEY,
            }
        supported = ", ".join(sorted({*_NATIVE_BACKENDS, *_OPENAI_COMPATIBLE_BACKENDS}))
        raise ValueError(
            f"Provider backend {self._backend!r} is not supported by the official RLM engine "
            f"(supported: {supported})."
        )

    def _price(
        self, input_tokens: int, output_tokens: int, reported_cost: float | None
    ) -> tuple[float, bool]:
        """Return ``(cost, cost_known)`` for a run's usage.

        The engine's own figure wins when it has one.  Otherwise the slot's cost
        table is asked, and a 0.0 from it is read carefully: the table answers
        0.0 both for a model it has no price for and for a lookup that raised,
        so for a cloud model that really did consume tokens the honest answer is
        "price unknown" rather than "free" — ranking must not reward it as the
        cheapest slot.  A locally served model is the exception: $0 is its real
        price, the same way Studio's own cells report it, so it stays known and
        keeps its place in the cost ranking.
        """
        if reported_cost is not None:
            return reported_cost, True
        if self._cost_fn is None:
            return 0.0, False
        total_cost = float(self._cost_fn(input_tokens, output_tokens))
        if total_cost > 0.0 or (input_tokens + output_tokens) == 0:
            return total_cost, True
        return total_cost, self._is_locally_served()

    def _is_locally_served(self) -> bool:
        """True when the provider is a local server, where $0 is a real price."""
        return self._backend in _OPENAI_COMPATIBLE_BACKENDS

    def _engine_environment(self) -> tuple[str, dict[str, Any]]:
        if self._sandbox_type == SANDBOX_TYPE_DOCKER:
            return _ENV_DOCKER, ({"image": self._docker_image} if self._docker_image else {})
        return _ENV_LOCAL, {}

    def _notes(
        self,
        environment: str,
        config: RunConfigDTO,
        *,
        cost_known: bool,
        interrupted_tokens: int | None = None,
    ) -> list[str]:
        """Notes for one result.

        ``interrupted_tokens`` is the token figure a cap breach reported: a
        count when it had one, 0 when the breach reported none, and ``None``
        for a run that was not interrupted.  The two cases get different notes
        because claiming a combined total was recorded would be false when no
        count was ever reported.
        """
        notes = [NOTE_NO_STREAMING, NOTE_NO_STEP_TOKENS]
        if environment == _ENV_LOCAL:
            notes.append(NOTE_LOCAL_ENV)
        if config.max_recursion_depth <= 0:
            notes.append(NOTE_NO_SUBCALL_FLOOR)
        if not cost_known:
            notes.append(NOTE_COST_UNKNOWN)
        if interrupted_tokens is not None:
            notes.append(
                NOTE_INTERRUPTED_TOTALS if interrupted_tokens else NOTE_INTERRUPTED_TOKENS_UNKNOWN
            )
        return notes

    def _to_result(
        self, completion: Any, start: float, environment: str, config: RunConfigDTO
    ) -> RunResultDTO:
        usage_summary = getattr(completion, "usage_summary", None)
        usage = usage_summary.to_dict() if usage_summary is not None else {}
        input_tokens, output_tokens, reported_cost = sum_usage(usage)
        total_cost, cost_known = self._price(input_tokens, output_tokens, reported_cost)

        trajectory = completion.metadata if isinstance(completion.metadata, dict) else None
        answer = completion.response or ""
        trace = trajectory_to_trace(trajectory, final_answer=answer, model=completion.root_model)
        steps = len((trajectory or {}).get("iterations") or [])
        elapsed = float(completion.execution_time or (time.time() - start))
        metadata = {
            RESULT_KEY_ENGINE_VERSION: self.version,
            RESULT_KEY_COST_KNOWN: cost_known,
            RESULT_KEY_ENGINE_NOTES: self._notes(environment, config, cost_known=cost_known),
        }
        error = getattr(completion, "error", None)
        if error:
            return RunResultDTO(
                answer=f"⚠️ **Execution error**.\n\n{error}",
                mode_used=MODE_RLM_OFFICIAL,
                success=False,
                error=str(error),
                steps=steps,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_cost=total_cost,
                elapsed_time=elapsed,
                trace=trace,
                metadata=metadata,
            )
        return RunResultDTO(
            answer=answer,
            mode_used=MODE_RLM_OFFICIAL,
            success=True,
            steps=steps,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_cost=total_cost,
            elapsed_time=elapsed,
            trace=trace,
            metadata=metadata,
        )

    def _failed(
        self, error: str, start: float, environment: str, config: RunConfigDTO
    ) -> RunResultDTO:
        """Failed result for a run that never reached the engine (nothing was spent)."""
        return RunResultDTO(
            answer=f"⚠️ **Execution error**.\n\n{error}",
            mode_used=MODE_RLM_OFFICIAL,
            success=False,
            error=error,
            elapsed_time=time.time() - start,
            metadata={
                RESULT_KEY_ENGINE_VERSION: self.version,
                RESULT_KEY_COST_KNOWN: True,
                RESULT_KEY_ENGINE_NOTES: self._notes(environment, config, cost_known=True),
            },
        )

    def _interrupted(
        self,
        error: str,
        exc: BaseException,
        trajectory: Any,
        start: float,
        environment: str,
        config: RunConfigDTO,
    ) -> RunResultDTO:
        """Failed result that keeps what the engine had already spent and done.

        The engine raises its cap breaches from inside the loop, so no
        completion object comes back and the usage the cap was checked against
        would be lost.  It survives in two places: the exception carries the
        figure that tripped the cap (``spent`` for a budget breach,
        ``tokens_used`` for a token breach, neither for a timeout), and the
        logger passed into the engine is ours, so the iterations it captured are
        still readable.  Reporting both is what keeps the most expensive runs
        from being recorded as the cheapest.

        What each breach knows differs, and neither figure may be invented from
        the other: a token breach reports one combined token count, recorded as
        input tokens under :data:`NOTE_INTERRUPTED_TOTALS`; a budget breach
        reports money but no tokens, so the token count stays 0 under
        :data:`NOTE_INTERRUPTED_TOKENS_UNKNOWN`; a timeout reports neither, so
        the cost is unknown rather than $0 — a run does not become free by
        being cut short.
        """
        captured = trajectory.get_trajectory() if hasattr(trajectory, "get_trajectory") else None
        partial = str(getattr(exc, "partial_answer", None) or "")
        trace = (
            trajectory_to_trace(captured, final_answer=partial, model=self._model)
            if captured
            else []
        )
        tokens_used = int(getattr(exc, "tokens_used", 0) or 0)
        spent: float | None = getattr(exc, "spent", None)
        if spent is not None:
            total_cost, cost_known = float(spent), True
        elif tokens_used:
            # Tokens known, money not: the slot's cost table can price them.
            total_cost, cost_known = self._price(tokens_used, 0, None)
        else:
            # Neither figure survived.  Zero tokens here means "not reported",
            # not "none used", so the cost-table shortcut in _price for a run
            # that genuinely used nothing must not apply.
            total_cost, cost_known = 0.0, False
        answer = f"⚠️ **Execution error**.\n\n{error}"
        if partial:
            answer = f"{answer}\n\n**Partial answer**\n\n{partial}"
        return RunResultDTO(
            answer=answer,
            mode_used=MODE_RLM_OFFICIAL,
            success=False,
            error=error,
            steps=len((captured or {}).get("iterations") or []),
            input_tokens=tokens_used,
            total_cost=total_cost,
            elapsed_time=time.time() - start,
            trace=trace,
            metadata={
                RESULT_KEY_ENGINE_VERSION: self.version,
                RESULT_KEY_COST_KNOWN: cost_known,
                RESULT_KEY_ENGINE_NOTES: self._notes(
                    environment, config, cost_known=cost_known, interrupted_tokens=tokens_used
                ),
            },
        )
