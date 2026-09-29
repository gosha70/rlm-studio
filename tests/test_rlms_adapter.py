"""Tests for the ``rlms`` engine adapter (specs/interop-official-rlm FR-3).

The ``rlm`` package is not a test dependency: ``run()`` is exercised against a
fake ``rlm`` module injected into ``sys.modules`` that records the engine
constructor kwargs and the ``completion()`` call, and returns a completion
object shaped exactly like ``rlm.core.types.RLMChatCompletion``.  The
trajectory fixture mirrors ``RLMLogger.get_trajectory()`` / ``to_dict()``
output for ``rlms==0.1.3``.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import sys
import threading
import types
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from rlmstudio.application.dto import RunConfigDTO
from rlmstudio.application.sandbox_vars import (
    MODE_RLM_OFFICIAL,
    RESULT_KEY_COST_KNOWN,
    RESULT_KEY_ENGINE_NOTES,
    RESULT_KEY_ENGINE_VERSION,
    TRACE_KEY_CODE,
    TRACE_KEY_CONTENT,
    TRACE_KEY_INPUT_TOKENS,
    TRACE_KEY_MODE,
    TRACE_KEY_MODEL,
    TRACE_KEY_OUTPUT_TOKENS,
    TRACE_KEY_RECURSION_DEPTH,
    TRACE_KEY_ROLE,
    TRACE_KEY_STEP,
)
from rlmstudio.application.services.outcome_classifier import (
    OutcomeCategory,
    classify_execution_outcome,
)
from rlmstudio.infrastructure.engines import rlms_adapter
from rlmstudio.infrastructure.engines.rlms_adapter import (
    BUSY_REASON,
    MAX_TRACE_RECURSION_DEPTH,
    NOTE_COST_UNKNOWN,
    NOTE_INTERRUPTED_TOTALS,
    NOTE_LOCAL_ENV,
    NOTE_NO_STREAMING,
    NOTE_NO_SUBCALL_FLOOR,
    UNAVAILABLE_REASON,
    RlmsEngineAdapter,
    engine_max_depth,
    sum_usage,
    trajectory_to_trace,
)

# ---------------------------------------------------------------------------
# Fixture: an rlms 0.1.3 trajectory as RLMLogger.get_trajectory() returns it
# ---------------------------------------------------------------------------

ROOT_MODEL = "gpt-4o-mini"
FINAL = "The document has 3 sections."

SUBCALL_USAGE = {
    "model_usage_summaries": {
        ROOT_MODEL: {"total_calls": 1, "total_input_tokens": 40, "total_output_tokens": 8}
    }
}

TRAJECTORY: dict[str, Any] = {
    "run_metadata": {
        "root_model": ROOT_MODEL,
        "max_depth": 1,
        "max_iterations": 5,
        "backend": "openai",
        "backend_kwargs": {"model_name": ROOT_MODEL},
        "environment_type": "local",
        "environment_kwargs": {},
        "other_backends": None,
    },
    "iterations": [
        {
            "type": "iteration",
            "iteration": 1,
            "timestamp": "2026-09-29T12:00:00",
            "prompt": [{"role": "user", "content": "..."}],
            "response": "Let me look.\n```repl\nprint(len(context))\n```",
            "code_blocks": [
                {
                    "code": "print(len(context))",
                    "result": {
                        "stdout": "1234\n",
                        "stderr": "",
                        "locals": {},
                        "execution_time": 0.02,
                        "rlm_calls": [
                            {
                                "root_model": ROOT_MODEL,
                                "prompt": "summarise section 1",
                                "response": "Section 1 is the intro.",
                                "usage_summary": SUBCALL_USAGE,
                                "execution_time": 0.4,
                            }
                        ],
                        "final_answer": None,
                    },
                }
            ],
            "final_answer": None,
            "iteration_time": 1.1,
        },
        {
            "type": "iteration",
            "iteration": 2,
            "timestamp": "2026-09-29T12:00:02",
            "prompt": [{"role": "user", "content": "..."}],
            "response": f'```repl\nanswer["content"] = "{FINAL}"\nanswer["ready"] = True\n```',
            "code_blocks": [
                {
                    "code": f'answer["content"] = "{FINAL}"\nanswer["ready"] = True',
                    "result": {
                        "stdout": "",
                        "stderr": "",
                        "locals": {},
                        "execution_time": 0.001,
                        "rlm_calls": [],
                        "final_answer": FINAL,
                    },
                }
            ],
            "final_answer": FINAL,
            "iteration_time": 0.6,
        },
    ],
}

USAGE_WITH_COST = {
    "model_usage_summaries": {
        ROOT_MODEL: {
            "total_calls": 3,
            "total_input_tokens": 1500,
            "total_output_tokens": 120,
            "total_cost": 0.0031,
        }
    },
    "total_cost": 0.0031,
}
USAGE_NO_COST = {
    "model_usage_summaries": {
        "qwen3:8b": {"total_calls": 3, "total_input_tokens": 1500, "total_output_tokens": 120}
    }
}


# ---------------------------------------------------------------------------
# Pure mapping
# ---------------------------------------------------------------------------


class TestSumUsage:
    def test_totals_and_reported_cost(self) -> None:
        assert sum_usage(USAGE_WITH_COST) == (1500, 120, 0.0031)

    def test_no_price_reported_is_none_not_zero(self) -> None:
        assert sum_usage(USAGE_NO_COST) == (1500, 120, None)

    def test_sums_across_models_and_per_model_costs(self) -> None:
        usage = {
            "model_usage_summaries": {
                "a": {"total_input_tokens": 10, "total_output_tokens": 1, "total_cost": 0.5},
                "b": {"total_input_tokens": 20, "total_output_tokens": 2, "total_cost": 0.25},
            }
        }
        assert sum_usage(usage) == (30, 3, 0.75)

    def test_empty(self) -> None:
        assert sum_usage({}) == (0, 0, None)


class TestEngineMaxDepth:
    """The two projects count recursion differently; the mapping is the contract.

    Studio counts levels of sub-RLM, the engine counts the depth at which a node
    degrades to a plain model call, and the engine's root node is at depth 0.
    """

    @pytest.mark.parametrize(
        ("studio_levels", "engine_depth"),
        [(0, 1), (1, 2), (2, 3), (5, 6)],
    )
    def test_studio_levels_map_to_engine_depth_plus_one(
        self, studio_levels: int, engine_depth: int
    ) -> None:
        assert engine_max_depth(studio_levels) == engine_depth

    def test_zero_is_clamped_so_the_engine_never_skips_its_loop(self) -> None:
        """``max_depth=0`` would answer from the document without the question.

        The engine returns a bare string in that case instead of a completion
        object, so the floor is 1 rather than a faithful 0.
        """
        assert engine_max_depth(0) >= 1
        assert engine_max_depth(-3) >= 1


class TestTrajectoryToTrace:
    def test_roles_follow_the_block_structure(self) -> None:
        trace = trajectory_to_trace(TRAJECTORY, final_answer=FINAL, model=ROOT_MODEL)

        roles = [e[TRACE_KEY_ROLE] for e in trace]
        # iteration 1: assistant(code) → execution(stdout) → execution(sub-call)
        # iteration 2: assistant(FINAL code) → execution → closing assistant(final answer)
        assert roles == [
            "assistant",
            "execution",
            "execution",
            "assistant",
            "execution",
            "assistant",
        ]
        assert [e[TRACE_KEY_STEP] for e in trace] == list(range(6))
        assert all(e[TRACE_KEY_MODE] == MODE_RLM_OFFICIAL for e in trace)

    def test_assistant_entry_carries_response_and_code(self) -> None:
        first = trajectory_to_trace(TRAJECTORY, final_answer=FINAL, model=ROOT_MODEL)[0]

        assert first[TRACE_KEY_CONTENT].startswith("Let me look.")
        assert first[TRACE_KEY_CODE] == "print(len(context))"
        assert first[TRACE_KEY_MODEL] == ROOT_MODEL

    def test_execution_entry_carries_stdout(self) -> None:
        execution = trajectory_to_trace(TRAJECTORY, final_answer=FINAL, model=ROOT_MODEL)[1]

        assert execution[TRACE_KEY_CONTENT] == "1234\n"
        assert TRACE_KEY_RECURSION_DEPTH not in execution

    def test_subcall_becomes_depth_one_execution_with_its_usage(self) -> None:
        subcall = trajectory_to_trace(TRAJECTORY, final_answer=FINAL, model=ROOT_MODEL)[2]

        assert subcall[TRACE_KEY_ROLE] == "execution"
        assert subcall[TRACE_KEY_CONTENT] == "Section 1 is the intro."
        assert subcall[TRACE_KEY_RECURSION_DEPTH] == 1
        assert (subcall[TRACE_KEY_INPUT_TOKENS], subcall[TRACE_KEY_OUTPUT_TOKENS]) == (40, 8)

    def test_closes_with_the_final_answer(self) -> None:
        last = trajectory_to_trace(TRAJECTORY, final_answer=FINAL, model=ROOT_MODEL)[-1]

        assert last[TRACE_KEY_ROLE] == "assistant"
        assert last[TRACE_KEY_CONTENT] == FINAL

    def test_does_not_duplicate_a_text_only_final_iteration(self) -> None:
        trajectory = {"iterations": [{"response": FINAL, "code_blocks": [], "iteration_time": 0.3}]}
        trace = trajectory_to_trace(trajectory, final_answer=FINAL, model=ROOT_MODEL)

        assert len(trace) == 1
        assert trace[0][TRACE_KEY_CONTENT] == FINAL

    def test_stderr_is_appended_and_flagged(self) -> None:
        trajectory = {
            "iterations": [
                {
                    "response": "```repl\n1/0\n```",
                    "code_blocks": [
                        {
                            "code": "1/0",
                            "result": {
                                "stdout": "",
                                "stderr": "\nZeroDivisionError: division by zero",
                            },
                        }
                    ],
                }
            ]
        }
        execution = trajectory_to_trace(trajectory, final_answer="", model=ROOT_MODEL)[1]

        assert execution[TRACE_KEY_CONTENT] == "ZeroDivisionError: division by zero"
        assert execution["error"] == "ZeroDivisionError: division by zero"

    def test_missing_trajectory_yields_only_the_final_answer(self) -> None:
        trace = trajectory_to_trace(None, final_answer="42", model=ROOT_MODEL)

        assert [e[TRACE_KEY_CONTENT] for e in trace] == ["42"]

    def test_second_code_block_in_one_iteration_carries_its_own_code(self) -> None:
        """One model turn can emit several ```repl blocks; each becomes a pair.

        The iteration's prose belongs to the first block only, so later blocks
        show their code as the assistant content instead of repeating it.
        """
        trajectory = {
            "iterations": [
                {
                    "response": "Two steps.\n```repl\na=1\n```\n```repl\nb=2\n```",
                    "code_blocks": [
                        {"code": "a=1", "result": {"stdout": "first\n", "stderr": ""}},
                        {"code": "b=2", "result": {"stdout": "second\n", "stderr": ""}},
                    ],
                }
            ]
        }
        trace = trajectory_to_trace(trajectory, final_answer=FINAL, model=ROOT_MODEL)

        assert [e[TRACE_KEY_ROLE] for e in trace] == [
            "assistant",
            "execution",
            "assistant",
            "execution",
            "assistant",
        ]
        assert trace[0][TRACE_KEY_CONTENT].startswith("Two steps.")
        assert trace[2][TRACE_KEY_CONTENT] == "b=2"
        assert trace[2][TRACE_KEY_CODE] == "b=2"
        assert [e[TRACE_KEY_CONTENT] for e in trace if e[TRACE_KEY_ROLE] == "execution"] == [
            "first\n",
            "second\n",
        ]


def _nested_call(*, response: str, model: str, nested: dict[str, Any] | None) -> dict[str, Any]:
    """An ``rlm_calls`` entry; ``nested`` is the child's own trajectory (engine ``metadata``)."""
    call: dict[str, Any] = {
        "root_model": model,
        "prompt": "sub question",
        "response": response,
        "usage_summary": SUBCALL_USAGE,
        "execution_time": 0.3,
    }
    if nested is not None:
        call["metadata"] = nested
    return call


def _one_block_trajectory(*, response: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
    """A one-iteration, one-code-block trajectory whose block made *calls*."""
    return {
        "iterations": [
            {
                "response": response,
                "code_blocks": [
                    {
                        "code": "out = llm_query(chunk)",
                        "result": {"stdout": "", "stderr": "", "rlm_calls": calls},
                    }
                ],
            }
        ]
    }


class TestNestedSubCalls:
    """A sub-call that ran its own REPL loop contributes that loop to the trace.

    The engine attaches a child's full trajectory to the sub-call's
    ``metadata``, so anything deeper than one level used to be dropped and
    every sub-call was labelled depth 1 regardless of where it ran.
    """

    def _two_level_trace(self) -> list[dict[str, Any]]:
        grandchild = _nested_call(response="leaf answer", model="child-model", nested=None)
        child = _nested_call(
            response="child answer",
            model="child-model",
            nested=_one_block_trajectory(response="child reasoning", calls=[grandchild]),
        )
        trace: list[dict[str, Any]] = trajectory_to_trace(
            _one_block_trajectory(response="root reasoning", calls=[child]),
            final_answer=FINAL,
            model=ROOT_MODEL,
        )
        return trace

    def test_child_steps_precede_the_child_answer(self) -> None:
        contents = [e[TRACE_KEY_CONTENT] for e in self._two_level_trace()]

        assert contents.index("child reasoning") < contents.index("child answer")
        assert contents.index("leaf answer") < contents.index("child answer")

    def test_each_level_carries_its_own_depth(self) -> None:
        depths = {
            e[TRACE_KEY_CONTENT]: e.get(TRACE_KEY_RECURSION_DEPTH) for e in self._two_level_trace()
        }

        assert depths["root reasoning"] is None  # the root loop is not recursion
        assert depths["child reasoning"] == 1
        assert depths["child answer"] == 1
        assert depths["leaf answer"] == 2

    def test_child_entries_are_attributed_to_the_child_model(self) -> None:
        by_content = {e[TRACE_KEY_CONTENT]: e for e in self._two_level_trace()}

        assert by_content["root reasoning"][TRACE_KEY_MODEL] == ROOT_MODEL
        assert by_content["child reasoning"][TRACE_KEY_MODEL] == "child-model"

    def test_walk_stops_at_the_depth_cap_but_still_reports_the_answer(self) -> None:
        """Nesting comes from another library, so the walk is bounded."""
        call = _nested_call(response="depth-limit answer", model=ROOT_MODEL, nested=None)
        for level in range(MAX_TRACE_RECURSION_DEPTH + 3):
            call = _nested_call(
                response=f"answer {level}",
                model=ROOT_MODEL,
                nested=_one_block_trajectory(response=f"reasoning {level}", calls=[call]),
            )
        trace = trajectory_to_trace(
            _one_block_trajectory(response="root reasoning", calls=[call]),
            final_answer=FINAL,
            model=ROOT_MODEL,
        )

        depths = [e.get(TRACE_KEY_RECURSION_DEPTH) or 0 for e in trace]
        assert max(depths) == MAX_TRACE_RECURSION_DEPTH
        assert "depth-limit answer" not in [e[TRACE_KEY_CONTENT] for e in trace]
        assert trace[-1][TRACE_KEY_CONTENT] == FINAL


# ---------------------------------------------------------------------------
# Availability and provider mapping (no rlm needed)
# ---------------------------------------------------------------------------


def _adapter(**overrides: Any) -> RlmsEngineAdapter:
    kwargs: dict[str, Any] = {"backend": "openai", "model": ROOT_MODEL, "api_key": "sk-test"}
    kwargs.update(overrides)
    return RlmsEngineAdapter(**kwargs)


class TestAvailability:
    def test_missing_package_reports_the_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "rlm", None)  # import_module raises ImportError

        available, reason = _adapter().is_available()

        assert not available
        assert reason == UNAVAILABLE_REASON
        assert "rlm-studio[interop]" in reason

    def test_version_is_none_when_distribution_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _missing(name: str) -> str:
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, "version", _missing)

        assert _adapter().version is None

    def test_unsupported_backend_is_reported_even_when_installed(self, fake_rlm: _FakeRlm) -> None:
        available, reason = _adapter(backend="gemini").is_available()

        assert not available
        assert "'gemini'" in reason
        assert "not supported" in reason

    @pytest.mark.parametrize("backend", ["anthropic", "openai"])
    def test_native_backend_without_any_key_is_unavailable(
        self, fake_rlm: _FakeRlm, monkeypatch: pytest.MonkeyPatch, backend: str
    ) -> None:
        """Both cloud backends are reported alike; neither is silently allowed."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        available, reason = _adapter(backend=backend, api_key=None).is_available()

        assert not available
        assert "API key" in reason

    @pytest.mark.parametrize(
        ("backend", "env_var"),
        [("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")],
    )
    def test_key_falls_back_to_the_provider_environment_variable(
        self, fake_rlm: _FakeRlm, monkeypatch: pytest.MonkeyPatch, backend: str, env_var: str
    ) -> None:
        """The engine's own client reads the environment at import time only.

        Studio therefore resolves the variable itself and passes the key in, so
        a key exported after the process started is still used.
        """
        monkeypatch.setenv(env_var, "sk-from-env")

        _adapter(backend=backend, api_key=None).run("d", "q", _CONFIG)

        assert fake_rlm.init_kwargs["backend_kwargs"]["api_key"] == "sk-from-env"

    def test_an_explicit_key_wins_over_the_environment(
        self, fake_rlm: _FakeRlm, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

        _adapter(api_key="sk-explicit").run("d", "q", _CONFIG)

        assert fake_rlm.init_kwargs["backend_kwargs"]["api_key"] == "sk-explicit"

    def test_local_backend_without_endpoint_is_unavailable(self, fake_rlm: _FakeRlm) -> None:
        available, reason = _adapter(backend="ollama", base_url=None).is_available()

        assert not available
        assert "endpoint" in reason

    def test_available_reports_engine_version(
        self, fake_rlm: _FakeRlm, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.1.3")

        assert _adapter().is_available() == (True, "rlms 0.1.3")


# ---------------------------------------------------------------------------
# run() against a fake `rlm` module
# ---------------------------------------------------------------------------


class _FakeUsage:
    def __init__(self, usage: dict[str, Any]) -> None:
        self._usage = usage

    def to_dict(self) -> dict[str, Any]:
        return self._usage


class _FakeCompletion:
    def __init__(
        self,
        *,
        response: str = FINAL,
        usage: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
        execution_time: float = 2.5,
    ) -> None:
        self.root_model = ROOT_MODEL
        self.prompt = "..."
        self.response = response
        self.usage_summary = _FakeUsage(USAGE_WITH_COST if usage is None else usage)
        self.execution_time = execution_time
        self.metadata = TRAJECTORY if metadata is None else metadata
        self.error = error


class _FakeRlm:
    """State shared between the fake module and the test: constructor kwargs, calls, script."""

    def __init__(self) -> None:
        self.init_kwargs: dict[str, Any] = {}
        self.completion_calls: list[dict[str, Any]] = []
        self.closed = False
        self.completion: _FakeCompletion = _FakeCompletion()
        self.raise_on_completion: BaseException | None = None
        # What the logger the adapter owns would hand back mid-run; the adapter
        # reads it when the engine raises instead of returning a completion.
        self.logged_trajectory: dict[str, Any] | None = TRAJECTORY


@pytest.fixture
def fake_rlm(monkeypatch: pytest.MonkeyPatch) -> _FakeRlm:
    """Install a fake ``rlm`` + ``rlm.logger`` in ``sys.modules`` and return its recorder."""
    state = _FakeRlm()

    class BudgetExceededError(Exception): ...

    class TokenLimitExceededError(Exception): ...

    class TimeoutExceededError(Exception): ...

    class RLM:
        def __init__(self, **kwargs: Any) -> None:
            state.init_kwargs = kwargs

        def completion(self, prompt: Any, root_prompt: str | None = None) -> _FakeCompletion:
            state.completion_calls.append({"prompt": prompt, "root_prompt": root_prompt})
            if state.raise_on_completion is not None:
                raise state.raise_on_completion
            return state.completion

        def close(self) -> None:
            state.closed = True

    class RLMLogger:
        def __init__(self, log_dir: str | None = None, file_name: str = "rlm") -> None:
            self.log_dir = log_dir

        def get_trajectory(self) -> dict[str, Any] | None:
            return state.logged_trajectory

    module = types.ModuleType("rlm")
    module.RLM = RLM  # type: ignore[attr-defined]
    module.BudgetExceededError = BudgetExceededError  # type: ignore[attr-defined]
    module.TokenLimitExceededError = TokenLimitExceededError  # type: ignore[attr-defined]
    module.TimeoutExceededError = TimeoutExceededError  # type: ignore[attr-defined]
    logger_module = types.ModuleType("rlm.logger")
    logger_module.RLMLogger = RLMLogger  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rlm", module)
    monkeypatch.setitem(sys.modules, "rlm.logger", logger_module)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.1.3")
    return state


_CONFIG = RunConfigDTO(
    mode=MODE_RLM_OFFICIAL,
    max_steps=7,
    max_recursion_depth=2,
    max_time_seconds=90.0,
    max_tokens=50_000,
    max_cost=0.25,
)


class TestRun:
    def test_budgets_map_onto_engine_limits(self, fake_rlm: _FakeRlm) -> None:
        _adapter().run("doc", "q", _CONFIG)

        kw = fake_rlm.init_kwargs
        assert kw["max_iterations"] == 7
        # Studio's 2 levels of sub-RLM are 3 engine levels: see engine_max_depth.
        assert kw["max_depth"] == 3
        assert kw["max_timeout"] == 90.0
        assert kw["max_tokens"] == 50_000
        assert kw["max_budget"] == 0.25
        assert kw["verbose"] is False
        assert kw["logger"] is not None

    def test_content_is_the_context_and_query_is_the_root_prompt(self, fake_rlm: _FakeRlm) -> None:
        _adapter().run("the document", "the question", _CONFIG)

        assert fake_rlm.completion_calls == [
            {"prompt": "the document", "root_prompt": "the question"}
        ]
        assert fake_rlm.closed

    def test_openai_backend_is_native(self, fake_rlm: _FakeRlm) -> None:
        _adapter(backend="openai", api_key="sk-test").run("d", "q", _CONFIG)

        assert fake_rlm.init_kwargs["backend"] == "openai"
        assert fake_rlm.init_kwargs["backend_kwargs"] == {
            "model_name": ROOT_MODEL,
            "api_key": "sk-test",
        }

    def test_anthropic_backend_is_native(self, fake_rlm: _FakeRlm) -> None:
        _adapter(backend="anthropic", model="claude-x", api_key="sk-ant").run("d", "q", _CONFIG)

        assert fake_rlm.init_kwargs["backend"] == "anthropic"
        assert fake_rlm.init_kwargs["backend_kwargs"] == {
            "model_name": "claude-x",
            "api_key": "sk-ant",
        }

    @pytest.mark.parametrize(
        ("backend", "endpoint", "expected_base_url"),
        [
            ("ollama", "http://localhost:11434", "http://localhost:11434/v1"),
            ("lmstudio", "http://localhost:1234/v1", "http://localhost:1234/v1"),
            ("vllm", "http://spark:8000/v1/", "http://spark:8000/v1"),
        ],
    )
    def test_local_servers_go_through_the_openai_client(
        self, fake_rlm: _FakeRlm, backend: str, endpoint: str, expected_base_url: str
    ) -> None:
        _adapter(backend=backend, model="qwen3:8b", api_key=None, base_url=endpoint).run(
            "d", "q", _CONFIG
        )

        assert fake_rlm.init_kwargs["backend"] == "openai"
        assert fake_rlm.init_kwargs["backend_kwargs"] == {
            "model_name": "qwen3:8b",
            "base_url": expected_base_url,
            "api_key": "not-needed",
        }

    def test_docker_sandbox_maps_to_docker_environment(self, fake_rlm: _FakeRlm) -> None:
        _adapter(sandbox_type="docker", docker_image="rlm-studio-sandbox").run("d", "q", _CONFIG)

        assert fake_rlm.init_kwargs["environment"] == "docker"
        assert fake_rlm.init_kwargs["environment_kwargs"] == {"image": "rlm-studio-sandbox"}

    def test_other_sandboxes_map_to_local_environment_with_a_warning(
        self, fake_rlm: _FakeRlm
    ) -> None:
        result = _adapter(sandbox_type="restricted").run("d", "q", _CONFIG)

        assert fake_rlm.init_kwargs["environment"] == "local"
        assert NOTE_LOCAL_ENV in result.metadata[RESULT_KEY_ENGINE_NOTES]

    def test_result_fields(self, fake_rlm: _FakeRlm) -> None:
        result = _adapter().run("d", "q", _CONFIG)

        assert result.success
        assert result.answer == FINAL
        assert result.mode_used == MODE_RLM_OFFICIAL
        assert result.steps == 2
        assert (result.input_tokens, result.output_tokens) == (1500, 120)
        assert result.total_cost == 0.0031
        assert result.elapsed_time == 2.5
        assert len(result.trace) == 6
        assert result.metadata[RESULT_KEY_ENGINE_VERSION] == "0.1.3"
        assert result.metadata[RESULT_KEY_COST_KNOWN] is True
        assert NOTE_NO_STREAMING in result.metadata[RESULT_KEY_ENGINE_NOTES]
        assert classify_execution_outcome(result.success, result.error, result.answer).category is (
            OutcomeCategory.SUCCESS
        )

    def test_cost_falls_back_to_the_slot_cost_table(self, fake_rlm: _FakeRlm) -> None:
        fake_rlm.completion = _FakeCompletion(usage=USAGE_NO_COST)
        seen: list[tuple[int, int]] = []

        def cost_fn(input_tokens: int, output_tokens: int) -> float:
            seen.append((input_tokens, output_tokens))
            return 0.042

        result = _adapter(cost_fn=cost_fn).run("d", "q", _CONFIG)

        assert seen == [(1500, 120)]
        assert result.total_cost == 0.042
        assert result.metadata[RESULT_KEY_COST_KNOWN] is True

    def test_unknown_cost_is_flagged_not_zeroed_silently(self, fake_rlm: _FakeRlm) -> None:
        fake_rlm.completion = _FakeCompletion(usage=USAGE_NO_COST)

        result = _adapter(cost_fn=None).run("d", "q", _CONFIG)

        assert result.total_cost == 0.0
        assert result.metadata[RESULT_KEY_COST_KNOWN] is False
        assert NOTE_COST_UNKNOWN in result.metadata[RESULT_KEY_ENGINE_NOTES]

    def test_a_zero_from_the_cost_table_is_unknown_not_free(self, fake_rlm: _FakeRlm) -> None:
        """A priceless model must not rank as the cheapest slot.

        The slot's cost table answers 0.0 both for a model it has no price for
        and for a lookup that raised, so a run that consumed tokens and came
        back free is "price unknown".
        """
        fake_rlm.completion = _FakeCompletion(usage=USAGE_NO_COST)

        result = _adapter(cost_fn=lambda _in, _out: 0.0).run("d", "q", _CONFIG)

        assert result.total_cost == 0.0
        assert result.metadata[RESULT_KEY_COST_KNOWN] is False
        assert NOTE_COST_UNKNOWN in result.metadata[RESULT_KEY_ENGINE_NOTES]

    def test_zero_cost_on_a_run_that_used_no_tokens_stays_known(self, fake_rlm: _FakeRlm) -> None:
        fake_rlm.completion = _FakeCompletion(usage={})

        result = _adapter(cost_fn=lambda _in, _out: 0.0).run("d", "q", _CONFIG)

        assert result.metadata[RESULT_KEY_COST_KNOWN] is True

    def test_zero_recursion_is_reported_as_unenforceable(self, fake_rlm: _FakeRlm) -> None:
        config = dataclasses.replace(_CONFIG, max_recursion_depth=0)

        result = _adapter().run("d", "q", config)

        assert fake_rlm.init_kwargs["max_depth"] == 1
        assert NOTE_NO_SUBCALL_FLOOR in result.metadata[RESULT_KEY_ENGINE_NOTES]

    def test_positive_recursion_carries_no_floor_note(self, fake_rlm: _FakeRlm) -> None:
        result = _adapter().run("d", "q", _CONFIG)

        assert NOTE_NO_SUBCALL_FLOOR not in result.metadata[RESULT_KEY_ENGINE_NOTES]

    def test_engine_error_completion_is_a_failed_result(self, fake_rlm: _FakeRlm) -> None:
        fake_rlm.completion = _FakeCompletion(
            response="Error: Timeout exhausted (90.0s of 90.0s)",
            error="Timeout exhausted (90.0s of 90.0s)",
        )

        result = _adapter().run("d", "q", _CONFIG)

        assert not result.success
        assert result.error == "Timeout exhausted (90.0s of 90.0s)"
        assert result.answer.startswith("⚠️")
        assert classify_execution_outcome(result.success, result.error, result.answer).category is (
            OutcomeCategory.TIMEOUT
        )

    def test_budget_exceptions_classify_as_budget(self, fake_rlm: _FakeRlm) -> None:
        rlm = sys.modules["rlm"]
        fake_rlm.raise_on_completion = rlm.TokenLimitExceededError("50000 tokens")  # type: ignore[attr-defined]

        result = _adapter().run("d", "q", _CONFIG)

        assert not result.success
        assert fake_rlm.closed
        assert classify_execution_outcome(result.success, result.error, result.answer).category is (
            OutcomeCategory.BUDGET_EXHAUSTED
        )

    def test_timeout_exception_classifies_as_timeout(self, fake_rlm: _FakeRlm) -> None:
        rlm = sys.modules["rlm"]
        fake_rlm.raise_on_completion = rlm.TimeoutExceededError("90s")  # type: ignore[attr-defined]

        result = _adapter().run("d", "q", _CONFIG)

        assert classify_execution_outcome(result.success, result.error, result.answer).category is (
            OutcomeCategory.TIMEOUT
        )

    def test_unexpected_exception_propagates_to_the_use_case(self, fake_rlm: _FakeRlm) -> None:
        fake_rlm.raise_on_completion = RuntimeError("connection refused")

        with pytest.raises(RuntimeError, match="connection refused"):
            _adapter().run("d", "q", _CONFIG)
        assert fake_rlm.closed

    def test_run_async_wraps_run(self, fake_rlm: _FakeRlm) -> None:
        import asyncio

        result = asyncio.run(_adapter().run_async("d", "q", _CONFIG))

        assert result.success
        assert fake_rlm.completion_calls[0]["root_prompt"] == "q"


def _raise(exc_name: str, message: str, **attrs: Any) -> BaseException:
    """Build one of the fake ``rlm`` limit exceptions with the engine's attributes."""
    exc: BaseException = getattr(sys.modules["rlm"], exc_name)(message)
    for name, value in attrs.items():
        setattr(exc, name, value)
    return exc


class TestInterruptedRunKeepsWhatWasSpent:
    """A run stopped by a cap must not be recorded as having spent nothing.

    The engine raises from inside its loop, so no completion comes back: the
    figure that tripped the cap lives on the exception and the steps live in the
    logger the adapter owns.  Reporting neither made the most expensive runs
    look like the cheapest.
    """

    def test_token_breach_reports_the_tokens_and_the_trajectory(self, fake_rlm: _FakeRlm) -> None:
        fake_rlm.raise_on_completion = _raise(
            "TokenLimitExceededError",
            "50,000 of 50,000 tokens",
            tokens_used=50_000,
            partial_answer="3 sections so far",
        )

        result = _adapter().run("d", "q", _CONFIG)

        assert not result.success
        assert result.total_tokens == 50_000
        assert result.steps == 2  # the two iterations the logger captured
        assert len(result.trace) > 0
        assert "3 sections so far" in result.answer
        assert NOTE_INTERRUPTED_TOTALS in result.metadata[RESULT_KEY_ENGINE_NOTES]
        assert classify_execution_outcome(result.success, result.error, result.answer).category is (
            OutcomeCategory.BUDGET_EXHAUSTED
        )

    def test_budget_breach_reports_the_money_it_spent(self, fake_rlm: _FakeRlm) -> None:
        fake_rlm.raise_on_completion = _raise(
            "BudgetExceededError", "spent $0.310000 of $0.250000", spent=0.31
        )

        result = _adapter().run("d", "q", _CONFIG)

        assert result.total_cost == 0.31
        assert result.metadata[RESULT_KEY_COST_KNOWN] is True

    def test_timeout_reports_no_cost_as_unknown_rather_than_zero(self, fake_rlm: _FakeRlm) -> None:
        """A timeout carries no usage at all, so cost is unknown, not $0."""
        fake_rlm.raise_on_completion = _raise(
            "TimeoutExceededError", "90.0s of 90.0s", elapsed=90.0
        )

        result = _adapter().run("d", "q", _CONFIG)

        assert result.total_cost == 0.0
        assert result.metadata[RESULT_KEY_COST_KNOWN] is False
        assert result.steps == 2
        assert classify_execution_outcome(result.success, result.error, result.answer).category is (
            OutcomeCategory.TIMEOUT
        )

    def test_an_empty_logger_still_yields_a_classified_failure(self, fake_rlm: _FakeRlm) -> None:
        fake_rlm.logged_trajectory = None
        fake_rlm.raise_on_completion = _raise("TimeoutExceededError", "90.0s of 90.0s")

        result = _adapter().run("d", "q", _CONFIG)

        assert not result.success
        assert result.trace == []
        assert result.steps == 0


class TestLocalEnvironmentIsSerialised:
    """The engine's in-process REPL mutates process-global state.

    ``local`` swaps ``sys.stdout`` / ``sys.stderr`` and chdirs into a temp
    directory around every code cell under a lock that covers one instance only,
    so two concurrent slots could restore each other's streams and leave the
    server in a deleted directory.  Official runs therefore queue.
    """

    def test_a_second_local_run_waits_and_then_reports_the_engine_as_busy(
        self, fake_rlm: _FakeRlm
    ) -> None:
        config = dataclasses.replace(_CONFIG, max_time_seconds=0.05)
        with self._lock_held():
            result = _adapter(sandbox_type="restricted").run("d", "q", config)

        assert not result.success
        assert result.error == BUSY_REASON
        assert fake_rlm.completion_calls == []  # the engine was never started
        assert result.elapsed_time >= config.max_time_seconds  # the wait is reported
        assert classify_execution_outcome(result.success, result.error, result.answer).category is (
            OutcomeCategory.GENERAL_ERROR
        )

    def test_a_docker_run_is_not_serialised(self, fake_rlm: _FakeRlm) -> None:
        """Container runs touch no shared process state, so they never queue."""
        config = dataclasses.replace(_CONFIG, max_time_seconds=0.05)
        with self._lock_held():
            result = _adapter(sandbox_type="docker").run("d", "q", config)

        assert result.success

    def test_batched_sub_calls_are_serialised_in_process(self, fake_rlm: _FakeRlm) -> None:
        """One run must not give itself several in-process REPLs either.

        The engine fans ``llm_query_batched`` out over a thread pool, and every
        child allowed its own REPL swaps the same process-global streams and
        working directory.
        """
        _adapter(sandbox_type="restricted").run("d", "q", _CONFIG)

        assert fake_rlm.init_kwargs["max_concurrent_subcalls"] == 1

    def test_docker_runs_keep_the_engine_default_concurrency(self, fake_rlm: _FakeRlm) -> None:
        _adapter(sandbox_type="docker").run("d", "q", _CONFIG)

        assert "max_concurrent_subcalls" not in fake_rlm.init_kwargs

    def test_the_lock_is_released_after_a_run(self, fake_rlm: _FakeRlm) -> None:
        _adapter(sandbox_type="restricted").run("d", "q", _CONFIG)

        assert _adapter(sandbox_type="restricted").run("d", "q", _CONFIG).success

    def test_the_lock_is_released_when_the_engine_raises(self, fake_rlm: _FakeRlm) -> None:
        fake_rlm.raise_on_completion = RuntimeError("connection refused")

        with pytest.raises(RuntimeError):
            _adapter(sandbox_type="restricted").run("d", "q", _CONFIG)

        fake_rlm.raise_on_completion = None
        assert _adapter(sandbox_type="restricted").run("d", "q", _CONFIG).success

    @staticmethod
    @contextmanager
    def _lock_held() -> Iterator[None]:
        """Hold the adapter's process-wide local-environment lock on another thread.

        A thread rather than the test's own thread because the lock is not
        reentrant and the adapter must observe it as held by someone else.
        """
        acquired = threading.Event()
        release = threading.Event()

        def _hold() -> None:
            with rlms_adapter._LOCAL_ENV_LOCK:
                acquired.set()
                release.wait(timeout=10.0)

        holder = threading.Thread(target=_hold, name="lock-holder", daemon=True)
        holder.start()
        assert acquired.wait(timeout=10.0), "the holder thread never took the lock"
        try:
            yield
        finally:
            release.set()
            holder.join(timeout=10.0)
