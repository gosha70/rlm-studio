"""Tests for the ``rlms`` engine adapter (specs/interop-official-rlm FR-3).

The ``rlm`` package is not a test dependency: ``run()`` is exercised against a
fake ``rlm`` module injected into ``sys.modules`` that records the engine
constructor kwargs and the ``completion()`` call, and returns a completion
object shaped exactly like ``rlm.core.types.RLMChatCompletion``.  The
trajectory fixture mirrors ``RLMLogger.get_trajectory()`` / ``to_dict()``
output for ``rlms==0.1.3``.
"""

from __future__ import annotations

import importlib.metadata
import sys
import types
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
from rlmstudio.infrastructure.engines.rlms_adapter import (
    NOTE_COST_UNKNOWN,
    NOTE_LOCAL_ENV,
    NOTE_NO_STREAMING,
    UNAVAILABLE_REASON,
    RlmsEngineAdapter,
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

    def test_anthropic_without_key_is_unavailable(self, fake_rlm: _FakeRlm) -> None:
        available, reason = _adapter(backend="anthropic", api_key=None).is_available()

        assert not available
        assert "API key" in reason

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
        assert kw["max_depth"] == 2
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
