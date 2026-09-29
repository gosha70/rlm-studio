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
extra ``execution`` entries carrying ``recursion_depth``; the final answer
closes the trace as a last ``assistant`` entry so the normaliser in
``server/routes/_helpers.py`` promotes it to ``final``.

Known limitations, recorded on every result under ``engine_notes``: the
engine does not stream (no TTFT / decode timings), it reports token usage per
run rather than per step, and its ``local`` environment executes model-written
code in-process — a runaway loop there cannot be stopped, only abandoned by
the use case's wall-clock guard.  Select the Docker sandbox to run the engine
in a container instead.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import logging
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
# Studio backends reached through the engine's OpenAI client plus a base_url.
_OPENAI_COMPATIBLE_BACKENDS: frozenset[str] = frozenset({"lmstudio", "vllm", "ollama"})
_RLMS_OPENAI_BACKEND = "openai"
_OPENAI_V1_SUFFIX = "/v1"
# Local servers ignore the key, but the OpenAI SDK refuses to start without one.
_PLACEHOLDER_API_KEY = "not-needed"

SANDBOX_TYPE_DOCKER = "docker"
_ENV_LOCAL = "local"
_ENV_DOCKER = "docker"

NOTE_NO_STREAMING = (
    "The official engine does not stream: TTFT and decode timings are unavailable for this slot."
)
NOTE_NO_STEP_TOKENS = (
    "The official engine reports token usage per run, not per step; step-level "
    "token counts are shown as 0 and the totals are exact."
)
NOTE_LOCAL_ENV = (
    "The official engine ran its REPL in-process (rlms 'local' environment), which is "
    "not isolated; select the Docker sandbox in Settings to run it in a container."
)
NOTE_COST_UNKNOWN = (
    "No price is known for this model; cost is shown as 0 and the slot is excluded "
    "from cost ranking."
)


def _openai_v1(url: str) -> str:
    """Return *url* with the ``/v1`` suffix the OpenAI SDK expects (idempotent)."""
    trimmed = url.rstrip("/")
    return trimmed if trimmed.endswith(_OPENAI_V1_SUFFIX) else trimmed + _OPENAI_V1_SUFFIX


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
    """
    entries: list[dict[str, Any]] = []

    def _entry(role: str, content: str, **extra: Any) -> None:
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
                **extra,
            }
        )

    for iteration in (trajectory or {}).get("iterations") or []:
        response = iteration.get("response") or ""
        blocks = iteration.get("code_blocks") or []
        if not blocks:
            _entry(
                "assistant",
                response,
                **{TRACE_KEY_ELAPSED_SECONDS: iteration.get("iteration_time") or 0.0},
            )
            continue
        for index, block in enumerate(blocks):
            code = block.get("code") or ""
            _entry("assistant", response if index == 0 else code, **{TRACE_KEY_CODE: code})
            result = block.get("result") or {}
            stdout = result.get("stdout") or ""
            stderr = (result.get("stderr") or "").strip()
            output = f"{stdout}\n{stderr}".strip() if stderr else stdout
            _entry(
                "execution",
                output,
                **{TRACE_KEY_ELAPSED_SECONDS: result.get("execution_time") or 0.0},
                **({"error": stderr} if stderr else {}),
            )
            for call in result.get("rlm_calls") or []:
                call_in, call_out, _ = sum_usage(call.get("usage_summary") or {})
                _entry(
                    "execution",
                    call.get("response") or "",
                    **{
                        TRACE_KEY_MODEL: call.get("root_model") or model,
                        TRACE_KEY_INPUT_TOKENS: call_in,
                        TRACE_KEY_OUTPUT_TOKENS: call_out,
                        TRACE_KEY_ELAPSED_SECONDS: call.get("execution_time") or 0.0,
                        TRACE_KEY_RECURSION_DEPTH: 1,
                    },
                )

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
        try:
            return importlib.metadata.version(RLMS_DISTRIBUTION)
        except importlib.metadata.PackageNotFoundError:
            return None

    def is_available(self) -> tuple[bool, str]:
        try:
            importlib.import_module(RLMS_MODULE)
        except ImportError:
            return False, UNAVAILABLE_REASON
        try:
            self._engine_backend()
        except ValueError as exc:
            return False, str(exc)
        return True, f"{RLMS_DISTRIBUTION} {self.version or 'unknown'}"

    def run(self, content: str, query: str, config: RunConfigDTO) -> RunResultDTO:
        rlm = importlib.import_module(RLMS_MODULE)
        rlm_logger = importlib.import_module(f"{RLMS_MODULE}.logger")
        backend, backend_kwargs = self._engine_backend()
        environment, environment_kwargs = self._engine_environment()

        trajectory = rlm_logger.RLMLogger()
        engine = rlm.RLM(
            backend=backend,
            backend_kwargs=backend_kwargs,
            environment=environment,
            environment_kwargs=environment_kwargs,
            max_depth=config.max_recursion_depth,
            max_iterations=config.max_steps,
            max_timeout=config.max_time_seconds,
            max_tokens=config.max_tokens,
            max_budget=config.max_cost,
            logger=trajectory,
            verbose=False,
        )
        start = time.time()
        try:
            # ``prompt`` is the context the REPL exposes; ``root_prompt`` is the question.
            completion = engine.completion(prompt=content, root_prompt=query)
        except (rlm.BudgetExceededError, rlm.TokenLimitExceededError) as exc:
            return self._failed(f"Budget exceeded: {exc}", start, environment)
        except rlm.TimeoutExceededError as exc:
            return self._failed(f"Official RLM engine timed out: {exc}", start, environment)
        finally:
            engine.close()
        return self._to_result(completion, start, environment)

    async def run_async(self, content: str, query: str, config: RunConfigDTO) -> RunResultDTO:
        return await asyncio.to_thread(self.run, content, query, config)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _engine_backend(self) -> tuple[str, dict[str, Any]]:
        """Map Studio's provider onto an ``rlms`` backend + client kwargs, or raise ``ValueError``."""
        if self._backend in _NATIVE_BACKENDS:
            kwargs: dict[str, Any] = {"model_name": self._model}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            elif self._backend == "anthropic":
                raise ValueError(
                    "The official RLM engine needs an API key for the Anthropic provider; "
                    "set one in Settings → LLM Providers."
                )
            return _NATIVE_BACKENDS[self._backend], kwargs
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

    def _engine_environment(self) -> tuple[str, dict[str, Any]]:
        if self._sandbox_type == SANDBOX_TYPE_DOCKER:
            return _ENV_DOCKER, ({"image": self._docker_image} if self._docker_image else {})
        return _ENV_LOCAL, {}

    def _notes(self, environment: str, *, cost_known: bool) -> list[str]:
        notes = [NOTE_NO_STREAMING, NOTE_NO_STEP_TOKENS]
        if environment == _ENV_LOCAL:
            notes.append(NOTE_LOCAL_ENV)
        if not cost_known:
            notes.append(NOTE_COST_UNKNOWN)
        return notes

    def _to_result(self, completion: Any, start: float, environment: str) -> RunResultDTO:
        usage_summary = getattr(completion, "usage_summary", None)
        usage = usage_summary.to_dict() if usage_summary is not None else {}
        input_tokens, output_tokens, reported_cost = sum_usage(usage)
        cost_known = True
        if reported_cost is not None:
            total_cost = reported_cost
        elif self._cost_fn is not None:
            total_cost = float(self._cost_fn(input_tokens, output_tokens))
        else:
            total_cost = 0.0
            cost_known = False

        trajectory = completion.metadata if isinstance(completion.metadata, dict) else None
        answer = completion.response or ""
        trace = trajectory_to_trace(trajectory, final_answer=answer, model=completion.root_model)
        steps = len((trajectory or {}).get("iterations") or [])
        elapsed = float(completion.execution_time or (time.time() - start))
        metadata = {
            RESULT_KEY_ENGINE_VERSION: self.version,
            RESULT_KEY_COST_KNOWN: cost_known,
            RESULT_KEY_ENGINE_NOTES: self._notes(environment, cost_known=cost_known),
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

    def _failed(self, error: str, start: float, environment: str) -> RunResultDTO:
        return RunResultDTO(
            answer=f"⚠️ **Execution error**.\n\n{error}",
            mode_used=MODE_RLM_OFFICIAL,
            success=False,
            error=error,
            elapsed_time=time.time() - start,
            metadata={
                RESULT_KEY_ENGINE_VERSION: self.version,
                RESULT_KEY_COST_KNOWN: True,
                RESULT_KEY_ENGINE_NOTES: self._notes(environment, cost_known=True),
            },
        )
