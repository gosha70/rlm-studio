"""Tests for port protocol compliance.

Verify that all infrastructure adapters correctly implement their
corresponding port Protocol classes (LLMPort, SandboxPort, etc.).
"""

import pytest

from rlmstudio.application.dto import ExecutionResultDTO, LLMResponseDTO
from rlmstudio.application.ports.llm_port import LLMPort

# ---------------------------------------------------------------------------
# MockLLMAdapter compliance
# ---------------------------------------------------------------------------


def _has_port_methods(obj: object, method_names: list[str]) -> bool:
    """Check that *obj* has all required callable methods."""
    return all(callable(getattr(obj, name, None)) for name in method_names)


_LLM_PORT_METHODS = ["complete", "complete_stream", "count_tokens", "get_pricing"]
_SANDBOX_PORT_METHODS = ["execute", "reset", "is_healthy", "set_variable", "get_variable"]
_RLM_ENGINE_PORT_METHODS = ["run", "run_async", "is_available"]


class TestFakeRLMEngineCompliance:
    """Verify the shared FakeRLMEngine satisfies RLMEnginePort structurally.

    The real ``rlms`` adapter is checked the same way once it exists; keeping
    the fake honest is what makes the use-case tests meaningful.
    """

    def _make_engine(self):
        from tests.fakes.fake_rlm_engine import FakeRLMEngine

        return FakeRLMEngine()

    def test_has_all_rlm_engine_port_methods(self):
        assert _has_port_methods(self._make_engine(), _RLM_ENGINE_PORT_METHODS)

    def test_exposes_version(self):
        assert isinstance(self._make_engine().version, str)

    def test_is_available_returns_flag_and_reason(self):
        available, reason = self._make_engine().is_available()
        assert isinstance(available, bool)
        assert isinstance(reason, str)

    def test_isinstance_rlm_engine_port(self):
        from rlmstudio.application.ports.rlm_engine_port import RLMEnginePort

        assert isinstance(self._make_engine(), RLMEnginePort)


class TestRlmsEngineAdapterCompliance:
    """Verify the ``rlms`` adapter satisfies RLMEnginePort structurally.

    Constructing the adapter and calling ``is_available`` never imports
    ``rlm`` eagerly, so this passes with or without the ``interop`` extra.
    """

    def _make_engine(self):
        from rlmstudio.infrastructure.engines import RlmsEngineAdapter

        return RlmsEngineAdapter(backend="openai", model="gpt-4o-mini", api_key="sk-test")

    def test_has_all_rlm_engine_port_methods(self):
        assert _has_port_methods(self._make_engine(), _RLM_ENGINE_PORT_METHODS)

    def test_exposes_version_attribute(self):
        engine = self._make_engine()
        assert engine.version is None or isinstance(engine.version, str)

    def test_is_available_returns_flag_and_reason(self):
        available, reason = self._make_engine().is_available()
        assert isinstance(available, bool)
        assert isinstance(reason, str)
        assert reason  # never silent — either the version or the install hint

    def test_isinstance_rlm_engine_port(self):
        from rlmstudio.application.ports.rlm_engine_port import RLMEnginePort

        assert isinstance(self._make_engine(), RLMEnginePort)


class TestMockLLMAdapterCompliance:
    """Verify MockLLMAdapter satisfies LLMPort structurally."""

    def _make_adapter(self):
        from rlmstudio.infrastructure.llm.mock_adapter import MockLLMAdapter

        return MockLLMAdapter(["test response"])

    def test_has_all_llm_port_methods(self):
        adapter = self._make_adapter()
        assert _has_port_methods(adapter, _LLM_PORT_METHODS)

    def test_complete_returns_dto(self):
        adapter = self._make_adapter()
        result = adapter.complete([{"role": "user", "content": "hi"}])
        assert isinstance(result, LLMResponseDTO)
        assert result.content == "test response"
        assert result.model == "mock"

    def test_complete_stream_yields_strings(self):
        adapter = self._make_adapter()
        chunks = list(adapter.complete_stream([{"role": "user", "content": "hi"}]))
        assert all(isinstance(c, str) for c in chunks)
        assert len(chunks) >= 1

    def test_count_tokens_returns_int(self):
        adapter = self._make_adapter()
        count = adapter.count_tokens("hello world test text")
        assert isinstance(count, int)
        assert count > 0

    def test_get_pricing_returns_dict(self):
        adapter = self._make_adapter()
        pricing = adapter.get_pricing()
        assert isinstance(pricing, dict)
        assert "input_cost_per_1m" in pricing
        assert "output_cost_per_1m" in pricing

    def test_call_history_recorded(self):
        adapter = self._make_adapter()
        msgs = [{"role": "user", "content": "hello"}]
        adapter.complete(msgs)
        assert len(adapter.call_history) == 1
        assert adapter.call_history[0] == msgs

    def test_responses_cycle(self):
        from rlmstudio.infrastructure.llm.mock_adapter import MockLLMAdapter

        adapter = MockLLMAdapter(["first", "second"])
        r1 = adapter.complete([{"role": "user", "content": "1"}])
        r2 = adapter.complete([{"role": "user", "content": "2"}])
        r3 = adapter.complete([{"role": "user", "content": "3"}])
        assert r1.content == "first"
        assert r2.content == "second"
        # After exhaustion, repeats last
        assert r3.content == "second"

    def test_reset(self):
        from rlmstudio.infrastructure.llm.mock_adapter import MockLLMAdapter

        adapter = MockLLMAdapter(["a", "b"])
        adapter.complete([{"role": "user", "content": "x"}])
        adapter.reset()
        r = adapter.complete([{"role": "user", "content": "y"}])
        assert r.content == "a"
        assert len(adapter.call_history) == 1

    def test_requires_at_least_one_response(self):
        from rlmstudio.infrastructure.llm.mock_adapter import MockLLMAdapter

        with pytest.raises(ValueError):
            MockLLMAdapter([])


# ---------------------------------------------------------------------------
# LiteLLMAdapter compliance (structural)
# ---------------------------------------------------------------------------


class TestLiteLLMAdapterCompliance:
    """Verify LiteLLMAdapter structurally satisfies LLMPort."""

    def test_isinstance_llm_port(self):
        from rlmstudio.infrastructure.llm.litellm_adapter import LiteLLMAdapter

        adapter = LiteLLMAdapter(model="gpt-4o")
        assert isinstance(adapter, LLMPort)

    def test_has_required_methods(self):
        from rlmstudio.infrastructure.llm.litellm_adapter import LiteLLMAdapter

        adapter = LiteLLMAdapter()
        assert callable(getattr(adapter, "complete", None))
        assert callable(getattr(adapter, "complete_stream", None))
        assert callable(getattr(adapter, "count_tokens", None))
        assert callable(getattr(adapter, "get_pricing", None))


# ---------------------------------------------------------------------------
# LocalSandboxAdapter compliance
# ---------------------------------------------------------------------------


class TestLocalSandboxAdapterCompliance:
    """Verify LocalSandboxAdapter satisfies SandboxPort structurally."""

    def _make_adapter(self):
        from rlmstudio.infrastructure.sandbox.local_sandbox import LocalSandboxAdapter

        return LocalSandboxAdapter()

    def test_has_all_sandbox_port_methods(self):
        adapter = self._make_adapter()
        assert _has_port_methods(adapter, _SANDBOX_PORT_METHODS)

    def test_execute_returns_dto(self):
        adapter = self._make_adapter()
        result = adapter.execute("x = 1 + 1")
        assert isinstance(result, ExecutionResultDTO)
        assert result.success is True

    def test_execute_captures_stdout(self):
        adapter = self._make_adapter()
        result = adapter.execute("print(42)")
        assert "42" in result.stdout

    def test_set_and_get_variable(self):
        adapter = self._make_adapter()
        adapter.set_variable("test_var", 123)
        assert adapter.get_variable("test_var") == 123

    def test_get_variable_not_found(self):
        adapter = self._make_adapter()
        assert adapter.get_variable("nonexistent") is None

    def test_reset(self):
        adapter = self._make_adapter()
        adapter.set_variable("x", 42)
        adapter.reset()
        assert adapter.get_variable("x") is None

    def test_is_healthy(self):
        adapter = self._make_adapter()
        assert adapter.is_healthy() is True


# ---------------------------------------------------------------------------
# RestrictedSandboxAdapter compliance
# ---------------------------------------------------------------------------


class TestRestrictedSandboxAdapterCompliance:
    """Verify RestrictedSandboxAdapter satisfies SandboxPort structurally."""

    def _make_adapter(self):
        from rlmstudio.infrastructure.sandbox.restricted_sandbox import RestrictedSandboxAdapter

        return RestrictedSandboxAdapter()

    def test_has_all_sandbox_port_methods(self):
        adapter = self._make_adapter()
        assert _has_port_methods(adapter, _SANDBOX_PORT_METHODS)

    def test_execute_returns_dto(self):
        adapter = self._make_adapter()
        result = adapter.execute("x = 2 + 2")
        assert isinstance(result, ExecutionResultDTO)
        assert result.success is True

    def test_set_and_get_variable(self):
        adapter = self._make_adapter()
        adapter.set_variable("val", [1, 2, 3])
        assert adapter.get_variable("val") == [1, 2, 3]

    def test_reset(self):
        adapter = self._make_adapter()
        adapter.execute("y = 99")
        adapter.reset()
        assert adapter.get_variable("y") is None

    def test_is_healthy(self):
        adapter = self._make_adapter()
        assert adapter.is_healthy() is True


# ---------------------------------------------------------------------------
# SandboxFactory compliance
# ---------------------------------------------------------------------------


class TestSandboxFactory:
    """Verify create_sandbox returns SandboxPort-compliant objects."""

    def test_local_sandbox_has_sandbox_port_methods(self):
        from rlmstudio.infrastructure.sandbox.sandbox_factory import create_sandbox

        sb = create_sandbox(sandbox_type="local")
        assert _has_port_methods(sb, _SANDBOX_PORT_METHODS)

    def test_restricted_sandbox_has_sandbox_port_methods(self):
        from rlmstudio.infrastructure.sandbox.sandbox_factory import create_sandbox

        sb = create_sandbox(sandbox_type="restricted")
        assert _has_port_methods(sb, _SANDBOX_PORT_METHODS)

    def test_unknown_sandbox_raises(self):
        from rlmstudio.infrastructure.sandbox.sandbox_factory import create_sandbox

        with pytest.raises(ValueError, match="Unknown sandbox type"):
            create_sandbox(sandbox_type="nonexistent")


# ---------------------------------------------------------------------------
# DTO tests
# ---------------------------------------------------------------------------


class TestDTOs:
    """Tests for application-layer Data Transfer Objects."""

    def test_llm_response_dto(self):
        dto = LLMResponseDTO(content="hello", model="gpt-4o", input_tokens=10, output_tokens=5)
        assert dto.content == "hello"
        assert dto.model == "gpt-4o"

    def test_execution_result_dto_success(self):
        dto = ExecutionResultDTO(stdout="42\n")
        assert dto.success is True
        assert dto.exception is None
        assert dto.timeout is False

    def test_execution_result_dto_failure(self):
        dto = ExecutionResultDTO(exception="NameError: x not defined")
        assert dto.success is False

    def test_execution_result_dto_timeout(self):
        dto = ExecutionResultDTO(timeout=True)
        assert dto.success is False

    def test_run_result_dto_total_tokens(self):
        from rlmstudio.application.dto import RunResultDTO

        dto = RunResultDTO(input_tokens=100, output_tokens=50)
        assert dto.total_tokens == 150

    def test_run_config_dto_defaults(self):
        from rlmstudio.application.dto import RunConfigDTO

        cfg = RunConfigDTO()
        assert cfg.mode == "auto"
        assert cfg.max_steps == 16
        assert cfg.max_recursion_depth == 1
