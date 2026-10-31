"""
Test suite for Phase 1 Infrastructure.

Run with: pytest tests/ui/test_infrastructure.py -v
"""

from datetime import datetime

import pytest

from rlmstudio.application.dto import RunConfigDTO, RunResultDTO
from rlmstudio.ui.services import (
    ChatManager,
    ChatMessage,
    ComparisonMetrics,
    ExecutionMetrics,
    LLMConfigManager,
    LLMProviderConfig,
    MemoryMonitor,
    MetricsCollector,
    Response,
    SessionMetrics,
)
from rlmstudio.ui.services.profile_store import resolve_system_prompt


class TestDataModels:
    """Test data class definitions."""

    def test_response_creation(self) -> None:
        """Test Response dataclass."""
        response = Response(
            content="Hello, world!",
            stop_reason="stop",
        )
        assert response.content == "Hello, world!"
        assert response.stop_reason == "stop"

    def test_execution_metrics_creation(self) -> None:
        """Test ExecutionMetrics dataclass."""
        metrics = ExecutionMetrics(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.05,
            cost_breakdown={"input": 0.03, "output": 0.02},
            execution_time_seconds=2.5,
            steps_taken=3,
            memory_used_mb=45.2,
            memory_peak_mb=62.1,
            success=True,
        )
        assert metrics.total_tokens == 150
        assert metrics.success is True

    def test_chat_message_creation(self) -> None:
        """Test ChatMessage dataclass."""
        message = ChatMessage(
            user_query="What is RLM?",
            mode="rlm_only",
        )
        assert message.user_query == "What is RLM?"
        assert message.mode == "rlm_only"
        assert message.id is not None
        assert message.timestamp is not None

    def test_chat_message_has_responses(self) -> None:
        """Test response checking methods."""
        message = ChatMessage(user_query="Test")
        assert not message.has_rlm_response()
        assert not message.has_direct_response()

        # Add RLM response
        message.rlm_response = Response(content="RLM answer", stop_reason="stop")
        assert message.has_rlm_response()
        assert not message.has_direct_response()

    def test_llm_provider_config_creation(self) -> None:
        """Test LLMProviderConfig dataclass."""
        config = LLMProviderConfig(
            provider="openai",
            model="gpt-4",
            api_key_env_var="OPENAI_API_KEY",
            input_cost_per_1k_tokens=0.03,
            output_cost_per_1k_tokens=0.06,
        )
        assert config.provider == "openai"
        assert config.is_configured is True  # Has model and env var
        assert config.is_ready is False  # Not ready until test_successful=True

    def test_session_metrics_creation(self) -> None:
        """Test SessionMetrics dataclass."""
        metrics = SessionMetrics(
            total_messages=5,
            total_tokens=10000,
            total_cost=0.10,
        )
        assert metrics.total_messages == 5
        assert metrics.total_tokens == 10000


class TestChatManager:
    """Test ChatManager service."""

    def test_chat_manager_init(self) -> None:
        """Test ChatManager initialization."""
        session: dict[str, object] = {}
        manager = ChatManager(session)
        assert "messages" in session
        assert "conversation_id" in session
        assert len(manager.get_messages()) == 0

    def test_get_messages(self) -> None:
        """Test getting messages."""
        session: dict[str, object] = {}
        manager = ChatManager(session)
        messages = manager.get_messages()
        assert isinstance(messages, list)
        assert len(messages) == 0


class TestMemoryMonitor:
    """Test MemoryMonitor service."""

    def test_memory_monitor_init(self) -> None:
        """Test MemoryMonitor initialization."""
        monitor = MemoryMonitor()
        assert monitor.baseline_memory_mb >= 0
        assert monitor.peak_memory_mb >= 0

    def test_memory_capture(self) -> None:
        """Test memory capture."""
        monitor = MemoryMonitor()
        monitor.current_mb()

        monitor.capture()
        stats = monitor.get_stats()

        assert "current" in stats
        assert "peak" in stats
        assert "average" in stats


class TestMetricsCollector:
    """Test MetricsCollector service."""

    def test_metrics_collector_init(self) -> None:
        """Test MetricsCollector initialization."""
        collector = MetricsCollector()
        assert "openai" in collector.provider_pricing
        assert "anthropic" in collector.provider_pricing

    def test_add_provider_pricing(self) -> None:
        """Test adding provider pricing."""
        collector = MetricsCollector()
        collector.add_provider_pricing(
            provider="custom",
            model="custom-model",
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
        )
        assert "custom" in collector.provider_pricing
        assert "custom-model" in collector.provider_pricing["custom"]

    def test_calculate_cost_openai_gpt4(self) -> None:
        """Test cost calculation for OpenAI GPT-4."""
        collector = MetricsCollector()
        # GPT-4: input $0.03/1k, output $0.06/1k
        # 1000 input tokens = $0.03
        # 500 output tokens = $0.03
        # Total = $0.06
        cost = collector._calculate_cost(1000, 500, "openai", "gpt-4")
        assert abs(cost - 0.06) < 0.0001

    def test_calculate_cost_anthropic_haiku(self) -> None:
        """Test cost calculation for Anthropic Claude Haiku."""
        collector = MetricsCollector()
        # Derived from the catalog rather than restated here. This assertion
        # previously pinned $0.008/$0.024 per 1k — the figure Anthropic's
        # published table contradicts ($1/$5 per MTok, i.e. 0.001/0.005) — so
        # it enforced the wrong price instead of catching it. The published
        # rate itself is pinned in tests/test_model_defaults_are_current.py.
        from rlmstudio.domain.model_catalog import price_per_1k

        rates = price_per_1k("claude-haiku-4-5-20251001")
        assert rates is not None
        expected = 1000 / 1000 * rates[0] + 1000 / 1000 * rates[1]

        cost = collector._calculate_cost(1000, 1000, "anthropic", "claude-haiku-4-5-20251001")
        assert abs(cost - expected) < 0.001

    def test_calculate_cost_unknown_provider(self) -> None:
        """Test cost calculation with unknown provider."""
        collector = MetricsCollector()
        cost = collector._calculate_cost(1000, 500, "unknown-provider", "model")
        assert cost == 0.0

    def test_calculate_cost_unknown_model(self) -> None:
        """Test cost calculation with unknown model."""
        collector = MetricsCollector()
        cost = collector._calculate_cost(1000, 500, "openai", "unknown-model")
        assert cost == 0.0

    def test_cost_breakdown_openai_gpt4(self) -> None:
        """Test cost breakdown for OpenAI GPT-4."""
        collector = MetricsCollector()
        # GPT-4: input $0.03/1k, output $0.06/1k
        # 1000 input tokens = $0.03
        # 500 output tokens = $0.03
        breakdown = collector._cost_breakdown(1000, 500, "openai", "gpt-4")
        assert abs(breakdown["input"] - 0.03) < 0.0001
        assert abs(breakdown["output"] - 0.03) < 0.0001

    def test_cost_breakdown_structure(self) -> None:
        """Test cost breakdown returns correct structure."""
        collector = MetricsCollector()
        breakdown = collector._cost_breakdown(1000, 500, "openai", "gpt-4")
        assert isinstance(breakdown, dict)
        assert "input" in breakdown
        assert "output" in breakdown
        assert isinstance(breakdown["input"], float)
        assert isinstance(breakdown["output"], float)

    def test_cost_breakdown_unknown_provider(self) -> None:
        """Test cost breakdown with unknown provider."""
        collector = MetricsCollector()
        breakdown = collector._cost_breakdown(1000, 500, "unknown", "model")
        assert breakdown["input"] == 0.0
        assert breakdown["output"] == 0.0

    def test_cost_calculation_consistency(self) -> None:
        """Test that _calculate_cost equals sum of breakdown."""
        collector = MetricsCollector()
        input_tokens = 1234
        output_tokens = 567
        provider = "openai"
        model = "gpt-4-turbo"

        total_cost = collector._calculate_cost(input_tokens, output_tokens, provider, model)
        breakdown = collector._cost_breakdown(input_tokens, output_tokens, provider, model)
        breakdown_sum = breakdown["input"] + breakdown["output"]

        assert abs(total_cost - breakdown_sum) < 0.0001


class TestLLMConfigManager:
    """Test LLMConfigManager service."""

    def test_llm_config_manager_init(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Test LLMConfigManager initialization."""
        manager = LLMConfigManager(config_dir=tmp_path)
        assert manager.config_dir.exists()
        assert len(manager.list_providers()) == 0

    def test_list_providers(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Test listing providers."""
        manager = LLMConfigManager(config_dir=tmp_path)
        providers = manager.list_providers()
        assert isinstance(providers, list)

    def test_test_connection_openai_valid(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Test connection to OpenAI with valid key."""
        manager = LLMConfigManager(config_dir=tmp_path)
        # Valid OpenAI key format (sk-...) and valid model
        success = manager.test_connection(
            provider="openai",
            model="gpt-4",
            api_key="sk-test-key-valid",
        )
        assert success is True

    def test_test_connection_openai_invalid_key_format(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Test OpenAI connection with invalid key format."""
        manager = LLMConfigManager(config_dir=tmp_path)
        # Invalid key format (doesn't start with sk-)
        success = manager.test_connection(
            provider="openai",
            model="gpt-4",
            api_key="invalid-key",
        )
        assert success is False

    def test_test_connection_openai_invalid_model(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Test OpenAI connection with invalid model."""
        manager = LLMConfigManager(config_dir=tmp_path)
        # Invalid model name
        success = manager.test_connection(
            provider="openai",
            model="invalid-model",
            api_key="sk-test-key",
        )
        assert success is False

    def test_test_connection_openai_gpt35(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Test OpenAI connection with GPT-3.5."""
        manager = LLMConfigManager(config_dir=tmp_path)
        success = manager.test_connection(
            provider="openai",
            model="gpt-3.5-turbo",
            api_key="sk-test-key",
        )
        assert success is True

    def test_test_connection_anthropic_valid(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Test connection to Anthropic with valid key."""
        manager = LLMConfigManager(config_dir=tmp_path)
        # Valid Anthropic key format (sk-ant-...)
        success = manager.test_connection(
            provider="anthropic",
            model="claude-3-opus",
            api_key="sk-ant-test-key",
        )
        assert success is True

    def test_test_connection_anthropic_invalid_key_format(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Test Anthropic connection with invalid key format."""
        manager = LLMConfigManager(config_dir=tmp_path)
        # Invalid Anthropic key format
        success = manager.test_connection(
            provider="anthropic",
            model="claude-3-opus",
            api_key="sk-test-key",  # OpenAI format, not Anthropic
        )
        assert success is False

    def test_test_connection_anthropic_invalid_model(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Test Anthropic connection with invalid model."""
        manager = LLMConfigManager(config_dir=tmp_path)
        success = manager.test_connection(
            provider="anthropic",
            model="invalid-model",
            api_key="sk-ant-test-key",
        )
        assert success is False

    def test_test_connection_ollama_valid(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Test connection to Ollama with valid model."""
        manager = LLMConfigManager(config_dir=tmp_path)
        # Ollama doesn't need API key
        success = manager.test_connection(
            provider="ollama",
            model="llama2",
            api_key="",  # Ollama doesn't use API key
        )
        assert success is True

    def test_test_connection_ollama_no_model(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Test Ollama connection without model."""
        manager = LLMConfigManager(config_dir=tmp_path)
        success = manager.test_connection(
            provider="ollama",
            model="",  # No model
            api_key="",
        )
        assert success is False

    def test_test_connection_lmstudio_valid(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Test connection to LMStudio with valid model."""
        manager = LLMConfigManager(config_dir=tmp_path)
        # LMStudio doesn't need API key
        success = manager.test_connection(
            provider="lmstudio",
            model="mistral",
            api_key="",  # LMStudio doesn't use API key
        )
        assert success is True

    def test_test_connection_unknown_provider(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Test connection to unknown provider."""
        manager = LLMConfigManager(config_dir=tmp_path)
        success = manager.test_connection(
            provider="unknown-provider",
            model="some-model",
            api_key="key",
        )
        assert success is False

    def test_test_connection_no_api_key_openai(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Test OpenAI connection without API key."""
        manager = LLMConfigManager(config_dir=tmp_path)
        success = manager.test_connection(
            provider="openai",
            model="gpt-4",
            api_key=None,
            api_key_env_var=None,
        )
        assert success is False

    def test_test_connection_env_var_fallback(self, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """Test using environment variable for API key."""
        manager = LLMConfigManager(config_dir=tmp_path)
        # Set environment variable
        monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test-key")

        success = manager.test_connection(
            provider="openai",
            model="gpt-4",
            api_key=None,
            api_key_env_var="TEST_OPENAI_KEY",
        )
        assert success is True

    def test_test_connection_returns_bool(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Test that test_connection always returns bool."""
        manager = LLMConfigManager(config_dir=tmp_path)
        result = manager.test_connection("openai", "gpt-4", "sk-key")
        assert isinstance(result, bool)


# ============================================================================
# PHASE 2.1: Direct Execution & Comparison
# ============================================================================


class TestChatManagerDirectExecution:
    """Test ChatManager._execute_direct() method."""

    @pytest.mark.asyncio
    async def test_execute_rlm_uses_run_use_case_with_session_settings(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_provider_config: LLMProviderConfig,
    ) -> None:
        """Local UI RLM should delegate to RunRLMUseCase with active profile settings."""
        session_state = {
            "max_steps": 32,
            "rlm_timeout": 120,
            "system_prompt_mode": "custom",
            "system_prompt_custom": {"rlm": "Use the detailed explainer supplement."},
            "default_temperature": 0.4,
            "default_top_p": 1.0,
            "default_max_tokens": 4096,
        }
        manager = ChatManager(session_state)
        captured: dict[str, object] = {}

        class FakeLegacyClient:
            model = "gpt-4o-mini"

            def complete(self, messages):  # pragma: no cover - should not be called
                raise AssertionError("Legacy client should not be called directly in this test")

            def complete_with_metadata(self, messages):  # pragma: no cover - should not be called
                raise AssertionError("Legacy client should not be called directly in this test")

        class FakeRunRLMUseCase:
            def __init__(self, llm, sandbox):
                captured["llm"] = llm
                captured["sandbox"] = sandbox

            def execute(self, content, query, config):
                captured["content"] = content
                captured["query"] = query
                captured["config"] = config
                return RunResultDTO(
                    answer="Unified RLM answer",
                    mode_used="rlm",
                    success=True,
                    steps=7,
                    input_tokens=120,
                    output_tokens=45,
                    total_cost=0.0,
                    elapsed_time=1.2,
                    trace=[{"step": 1, "role": "assistant", "content": "step"}],
                )

        monkeypatch.setattr(
            ChatManager,
            "_resolve_provider",
            lambda self, provider_name=None, api_key=None: (fake_provider_config, "sk-test"),
        )
        monkeypatch.setattr(
            "rlmstudio.ui.services.chat_manager.get_llm_client",
            lambda **kwargs: FakeLegacyClient(),
        )
        monkeypatch.setattr(
            "rlmstudio.ui.services.chat_manager.create_sandbox",
            lambda: object(),
        )
        monkeypatch.setattr(
            "rlmstudio.ui.services.chat_manager.RunRLMUseCase",
            FakeRunRLMUseCase,
        )

        result = await manager._execute_rlm("Summarize both files", file_context="doc content")

        config = captured["config"]
        assert isinstance(config, RunConfigDTO)
        assert result["response"].content == "Unified RLM answer"
        assert result["metrics"].steps_taken == 7
        assert result["trace"] == [{"step": 1, "role": "assistant", "content": "step"}]
        assert config.max_steps == 32
        assert config.max_time_seconds == 120.0
        assert config.system_prompt_extra == resolve_system_prompt("rlm", manager.session_state)

    @pytest.mark.asyncio
    async def test_execute_direct_returns_dict(self) -> None:
        """Test that _execute_direct returns proper dict structure."""
        manager = ChatManager()
        result = await manager._execute_direct("What is RLM?")

        assert isinstance(result, dict)
        assert "response" in result
        assert "metrics" in result
        assert "trace" not in result  # Direct has no trace

    @pytest.mark.asyncio
    async def test_execute_direct_response_structure(self, mock_execute_methods: None) -> None:
        """Test Response object from direct execution."""
        manager = ChatManager()
        result = await manager._execute_direct("Summarize this")

        response = result["response"]
        assert isinstance(response, Response)
        assert response.content
        assert response.stop_reason == "stop"

    @pytest.mark.asyncio
    async def test_execute_direct_metrics_structure(self, mock_execute_methods: None) -> None:
        """Test ExecutionMetrics from direct execution."""
        manager = ChatManager()
        result = await manager._execute_direct("Test query")

        metrics = result["metrics"]
        assert isinstance(metrics, ExecutionMetrics)
        assert metrics.input_tokens > 0
        assert metrics.output_tokens > 0
        assert metrics.total_tokens > 0
        assert metrics.cost_usd >= 0
        assert metrics.execution_time_seconds > 0
        assert metrics.steps_taken == 0  # Direct has no steps
        assert metrics.execution_type == "direct"

    @pytest.mark.asyncio
    async def test_execute_direct_metrics_breakdown(self) -> None:
        """Test cost breakdown in direct metrics."""
        manager = ChatManager()
        result = await manager._execute_direct("Test")

        metrics = result["metrics"]
        assert "input" in metrics.cost_breakdown
        assert "output" in metrics.cost_breakdown
        assert metrics.cost_breakdown["input"] >= 0
        assert metrics.cost_breakdown["output"] >= 0

    @pytest.mark.asyncio
    async def test_execute_direct_faster_than_rlm(self, mock_execute_methods: None) -> None:
        """Test that direct execution is faster than RLM."""

        manager = ChatManager()

        # Time RLM execution
        start_rlm = datetime.now()
        rlm_result = await manager._execute_rlm("Test query")
        (datetime.now() - start_rlm).total_seconds()

        # Time Direct execution
        start_direct = datetime.now()
        direct_result = await manager._execute_direct("Test query")
        (datetime.now() - start_direct).total_seconds()

        # Direct should be faster in real-world scenarios
        # (though simulation is fast for both)
        rlm_metrics = rlm_result["metrics"]
        direct_metrics = direct_result["metrics"]

        assert direct_metrics.execution_time_seconds < rlm_metrics.execution_time_seconds

    @pytest.mark.asyncio
    async def test_execute_direct_less_tokens_than_rlm(self, mock_execute_methods: None) -> None:
        """Test that direct execution uses fewer tokens than RLM."""
        manager = ChatManager()

        rlm_result = await manager._execute_rlm("Test query")
        direct_result = await manager._execute_direct("Test query")

        rlm_metrics = rlm_result["metrics"]
        direct_metrics = direct_result["metrics"]

        # Direct should use fewer tokens (no exploration steps)
        assert direct_metrics.total_tokens < rlm_metrics.total_tokens

    @pytest.mark.asyncio
    async def test_execute_direct_success(self, mock_execute_methods: None) -> None:
        """Test that direct execution marks success."""
        manager = ChatManager()
        result = await manager._execute_direct("Test")

        metrics = result["metrics"]
        assert metrics.success is True
        assert metrics.error is None


class TestChatManagerComparison:
    """Test ChatManager._compare_metrics() method."""

    def test_compare_metrics_returns_comparison_metrics(self) -> None:
        """Test that _compare_metrics returns ComparisonMetrics."""
        manager = ChatManager()

        rlm_metrics = ExecutionMetrics(
            input_tokens=600,
            output_tokens=250,
            total_tokens=850,
            cost_usd=0.05,
            cost_breakdown={"input": 0.03, "output": 0.02},
            execution_time_seconds=1.6,
            steps_taken=3,
            memory_used_mb=45.2,
            memory_peak_mb=62.1,
            success=True,
            execution_type="rlm",
        )

        direct_metrics = ExecutionMetrics(
            input_tokens=150,
            output_tokens=120,
            total_tokens=270,
            cost_usd=0.015,
            cost_breakdown={"input": 0.01, "output": 0.005},
            execution_time_seconds=0.8,
            steps_taken=0,
            memory_used_mb=12.5,
            memory_peak_mb=18.3,
            success=True,
            execution_type="direct",
        )

        comparison = manager._compare_metrics(rlm_metrics, direct_metrics)
        assert isinstance(comparison, ComparisonMetrics)

    def test_compare_metrics_cost_delta(self) -> None:
        """Test cost delta calculation."""
        manager = ChatManager()

        rlm_metrics = ExecutionMetrics(
            input_tokens=600,
            output_tokens=250,
            total_tokens=850,
            cost_usd=0.050,
            cost_breakdown={"input": 0.03, "output": 0.02},
            execution_time_seconds=1.6,
            steps_taken=3,
            memory_used_mb=45.2,
            memory_peak_mb=62.1,
            success=True,
            execution_type="rlm",
        )

        direct_metrics = ExecutionMetrics(
            input_tokens=150,
            output_tokens=120,
            total_tokens=270,
            cost_usd=0.015,
            cost_breakdown={"input": 0.01, "output": 0.005},
            execution_time_seconds=0.8,
            steps_taken=0,
            memory_used_mb=12.5,
            memory_peak_mb=18.3,
            success=True,
            execution_type="direct",
        )

        comparison = manager._compare_metrics(rlm_metrics, direct_metrics)

        assert comparison.cost_delta_usd == pytest.approx(0.035, abs=0.001)
        assert comparison.rlm_cost_usd == 0.050
        assert comparison.direct_cost_usd == 0.015

    def test_compare_metrics_time_delta(self) -> None:
        """Test time delta calculation."""
        manager = ChatManager()

        rlm_metrics = ExecutionMetrics(
            input_tokens=600,
            output_tokens=250,
            total_tokens=850,
            cost_usd=0.050,
            cost_breakdown={"input": 0.03, "output": 0.02},
            execution_time_seconds=1.6,
            steps_taken=3,
            memory_used_mb=45.2,
            memory_peak_mb=62.1,
            success=True,
            execution_type="rlm",
        )

        direct_metrics = ExecutionMetrics(
            input_tokens=150,
            output_tokens=120,
            total_tokens=270,
            cost_usd=0.015,
            cost_breakdown={"input": 0.01, "output": 0.005},
            execution_time_seconds=0.8,
            steps_taken=0,
            memory_used_mb=12.5,
            memory_peak_mb=18.3,
            success=True,
            execution_type="direct",
        )

        comparison = manager._compare_metrics(rlm_metrics, direct_metrics)

        assert comparison.time_delta_seconds == pytest.approx(0.8, abs=0.001)
        assert comparison.rlm_time_seconds == 1.6
        assert comparison.direct_time_seconds == 0.8

    def test_compare_metrics_token_delta(self) -> None:
        """Test token delta calculation."""
        manager = ChatManager()

        rlm_metrics = ExecutionMetrics(
            input_tokens=600,
            output_tokens=250,
            total_tokens=850,
            cost_usd=0.050,
            cost_breakdown={"input": 0.03, "output": 0.02},
            execution_time_seconds=1.6,
            steps_taken=3,
            memory_used_mb=45.2,
            memory_peak_mb=62.1,
            success=True,
            execution_type="rlm",
        )

        direct_metrics = ExecutionMetrics(
            input_tokens=150,
            output_tokens=120,
            total_tokens=270,
            cost_usd=0.015,
            cost_breakdown={"input": 0.01, "output": 0.005},
            execution_time_seconds=0.8,
            steps_taken=0,
            memory_used_mb=12.5,
            memory_peak_mb=18.3,
            success=True,
            execution_type="direct",
        )

        comparison = manager._compare_metrics(rlm_metrics, direct_metrics)

        assert comparison.token_delta == 580
        assert comparison.rlm_tokens == 850
        assert comparison.direct_tokens == 270

    def test_compare_metrics_percentages(self) -> None:
        """Test percentage calculations."""
        manager = ChatManager()

        rlm_metrics = ExecutionMetrics(
            input_tokens=600,
            output_tokens=250,
            total_tokens=850,
            cost_usd=0.050,
            cost_breakdown={"input": 0.03, "output": 0.02},
            execution_time_seconds=1.6,
            steps_taken=3,
            memory_used_mb=45.2,
            memory_peak_mb=62.1,
            success=True,
            execution_type="rlm",
        )

        direct_metrics = ExecutionMetrics(
            input_tokens=150,
            output_tokens=120,
            total_tokens=270,
            cost_usd=0.015,
            cost_breakdown={"input": 0.01, "output": 0.005},
            execution_time_seconds=0.8,
            steps_taken=0,
            memory_used_mb=12.5,
            memory_peak_mb=18.3,
            success=True,
            execution_type="direct",
        )

        comparison = manager._compare_metrics(rlm_metrics, direct_metrics)

        # RLM is 233% more expensive
        assert comparison.cost_delta_percent == pytest.approx(233, abs=1)
        # RLM takes 100% more time (2x)
        assert comparison.time_delta_percent == pytest.approx(100, abs=1)

    def test_compare_metrics_generates_recommendation(self) -> None:
        """Test that recommendation is generated."""
        manager = ChatManager()

        rlm_metrics = ExecutionMetrics(
            input_tokens=600,
            output_tokens=250,
            total_tokens=850,
            cost_usd=0.050,
            cost_breakdown={"input": 0.03, "output": 0.02},
            execution_time_seconds=1.6,
            steps_taken=3,
            memory_used_mb=45.2,
            memory_peak_mb=62.1,
            success=True,
            execution_type="rlm",
        )

        direct_metrics = ExecutionMetrics(
            input_tokens=150,
            output_tokens=120,
            total_tokens=270,
            cost_usd=0.015,
            cost_breakdown={"input": 0.01, "output": 0.005},
            execution_time_seconds=0.8,
            steps_taken=0,
            memory_used_mb=12.5,
            memory_peak_mb=18.3,
            success=True,
            execution_type="direct",
        )

        comparison = manager._compare_metrics(rlm_metrics, direct_metrics)

        assert comparison.recommendation
        assert isinstance(comparison.recommendation, str)
        assert len(comparison.recommendation) > 0
        # Should mention that Direct is cheaper
        assert "cheaper" in comparison.recommendation.lower()


# ============================================================================
# PHASE 2.2: Metrics Collection
# ============================================================================


class TestMetricsCollectorCollection:
    """Test MetricsCollector collection methods."""

    @pytest.mark.asyncio
    async def test_collect_rlm_metrics_returns_execution_metrics(self) -> None:
        """Test that collect_rlm_metrics returns ExecutionMetrics."""
        collector = MetricsCollector()
        monitor = MemoryMonitor()

        # Simulate RLM execution with 3 steps
        trace = [
            {"step": 1, "input_tokens": 200, "output_tokens": 100, "duration_seconds": 0.5},
            {"step": 2, "input_tokens": 250, "output_tokens": 130, "duration_seconds": 0.6},
            {"step": 3, "input_tokens": 300, "output_tokens": 150, "duration_seconds": 0.7},
        ]
        result: dict[str, object] = {"response": Response("RLM result", "stop")}

        metrics = await collector.collect_rlm_metrics(result, trace, monitor, "openai", "gpt-4")

        assert isinstance(metrics, ExecutionMetrics)
        assert metrics.execution_type == "rlm"

    @pytest.mark.asyncio
    async def test_collect_rlm_metrics_sums_tokens(self) -> None:
        """Test that RLM metrics correctly sum tokens from trace."""
        collector = MetricsCollector()
        monitor = MemoryMonitor()

        trace = [
            {"step": 1, "input_tokens": 200, "output_tokens": 100, "duration_seconds": 0.5},
            {"step": 2, "input_tokens": 250, "output_tokens": 130, "duration_seconds": 0.6},
            {"step": 3, "input_tokens": 300, "output_tokens": 150, "duration_seconds": 0.7},
        ]
        result: dict[str, object] = {}

        metrics = await collector.collect_rlm_metrics(result, trace, monitor, "openai", "gpt-4")

        # Input: 200 + 250 + 300 = 750
        assert metrics.input_tokens == 750
        # Output: 100 + 130 + 150 = 380
        assert metrics.output_tokens == 380
        # Total: 750 + 380 = 1130
        assert metrics.total_tokens == 1130

    @pytest.mark.asyncio
    async def test_collect_rlm_metrics_sums_execution_time(self) -> None:
        """Test that RLM metrics sum execution time from trace."""
        collector = MetricsCollector()
        monitor = MemoryMonitor()

        trace = [
            {"step": 1, "input_tokens": 200, "output_tokens": 100, "duration_seconds": 0.5},
            {"step": 2, "input_tokens": 250, "output_tokens": 130, "duration_seconds": 0.6},
            {"step": 3, "input_tokens": 300, "output_tokens": 150, "duration_seconds": 0.7},
        ]
        result: dict[str, object] = {}

        metrics = await collector.collect_rlm_metrics(result, trace, monitor, "openai", "gpt-4")

        # Total time: 0.5 + 0.6 + 0.7 = 1.8
        assert metrics.execution_time_seconds == pytest.approx(1.8, abs=0.01)

    @pytest.mark.asyncio
    async def test_collect_rlm_metrics_steps_taken(self) -> None:
        """Test that RLM metrics records number of steps."""
        collector = MetricsCollector()
        monitor = MemoryMonitor()

        trace = [
            {"step": 1, "input_tokens": 200, "output_tokens": 100, "duration_seconds": 0.5},
            {"step": 2, "input_tokens": 250, "output_tokens": 130, "duration_seconds": 0.6},
            {"step": 3, "input_tokens": 300, "output_tokens": 150, "duration_seconds": 0.7},
        ]
        result: dict[str, object] = {}

        metrics = await collector.collect_rlm_metrics(result, trace, monitor, "openai", "gpt-4")

        assert metrics.steps_taken == 3

    @pytest.mark.asyncio
    async def test_collect_rlm_metrics_calculates_cost(self) -> None:
        """Test that RLM metrics calculates cost correctly."""
        collector = MetricsCollector()
        monitor = MemoryMonitor()

        # Simple trace for cost calculation
        trace = [
            {"input_tokens": 1000, "output_tokens": 1000, "duration_seconds": 1.0},
        ]
        result: dict[str, object] = {}

        metrics = await collector.collect_rlm_metrics(result, trace, monitor, "openai", "gpt-4")

        # OpenAI GPT-4: input $0.03/1K, output $0.06/1K
        # 1000 input * 0.03/1K + 1000 output * 0.06/1K = 0.03 + 0.06 = 0.09
        assert metrics.cost_usd == pytest.approx(0.09, abs=0.001)

    @pytest.mark.asyncio
    async def test_collect_direct_metrics_returns_execution_metrics(self) -> None:
        """Test that collect_direct_metrics returns ExecutionMetrics."""
        collector = MetricsCollector()
        monitor = MemoryMonitor()

        result: dict[str, object] = {
            "input_tokens": 150,
            "output_tokens": 120,
            "execution_time": 0.8,
        }

        metrics = await collector.collect_direct_metrics(result, monitor, "openai", "gpt-4")

        assert isinstance(metrics, ExecutionMetrics)
        assert metrics.execution_type == "direct"

    @pytest.mark.asyncio
    async def test_collect_direct_metrics_no_steps(self) -> None:
        """Test that direct metrics has zero steps."""
        collector = MetricsCollector()
        monitor = MemoryMonitor()

        result: dict[str, object] = {
            "input_tokens": 150,
            "output_tokens": 120,
            "execution_time": 0.8,
        }

        metrics = await collector.collect_direct_metrics(result, monitor, "openai", "gpt-4")

        assert metrics.steps_taken == 0

    @pytest.mark.asyncio
    async def test_collect_direct_metrics_fewer_tokens(self) -> None:
        """Test that direct metrics has fewer tokens than RLM."""
        collector = MetricsCollector()
        monitor = MemoryMonitor()

        direct_result: dict[str, object] = {
            "input_tokens": 150,
            "output_tokens": 120,
            "execution_time": 0.8,
        }

        direct_metrics = await collector.collect_direct_metrics(
            direct_result, monitor, "openai", "gpt-4"
        )

        assert direct_metrics.total_tokens == 270
        assert direct_metrics.input_tokens == 150
        assert direct_metrics.output_tokens == 120


class TestMetricsCollectorComparison:
    """Test MetricsCollector.compare_metrics() method."""

    def test_metrics_collector_compare_returns_comparison_metrics(self) -> None:
        """Test that compare_metrics returns ComparisonMetrics."""
        collector = MetricsCollector()

        rlm_metrics = ExecutionMetrics(
            input_tokens=750,
            output_tokens=380,
            total_tokens=1130,
            cost_usd=0.090,
            cost_breakdown={"input": 0.0225, "output": 0.0675},
            execution_time_seconds=1.8,
            steps_taken=3,
            memory_used_mb=45.2,
            memory_peak_mb=62.1,
            success=True,
            execution_type="rlm",
        )

        direct_metrics = ExecutionMetrics(
            input_tokens=150,
            output_tokens=120,
            total_tokens=270,
            cost_usd=0.015,
            cost_breakdown={"input": 0.0045, "output": 0.0105},
            execution_time_seconds=0.8,
            steps_taken=0,
            memory_used_mb=12.5,
            memory_peak_mb=18.3,
            success=True,
            execution_type="direct",
        )

        comparison = collector.compare_metrics(rlm_metrics, direct_metrics)

        assert isinstance(comparison, ComparisonMetrics)

    def test_metrics_collector_compare_cost_delta(self) -> None:
        """Test cost delta calculation in collector."""
        collector = MetricsCollector()

        rlm_metrics = ExecutionMetrics(
            input_tokens=750,
            output_tokens=380,
            total_tokens=1130,
            cost_usd=0.090,
            cost_breakdown={"input": 0.0225, "output": 0.0675},
            execution_time_seconds=1.8,
            steps_taken=3,
            memory_used_mb=45.2,
            memory_peak_mb=62.1,
            success=True,
            execution_type="rlm",
        )

        direct_metrics = ExecutionMetrics(
            input_tokens=150,
            output_tokens=120,
            total_tokens=270,
            cost_usd=0.015,
            cost_breakdown={"input": 0.0045, "output": 0.0105},
            execution_time_seconds=0.8,
            steps_taken=0,
            memory_used_mb=12.5,
            memory_peak_mb=18.3,
            success=True,
            execution_type="direct",
        )

        comparison = collector.compare_metrics(rlm_metrics, direct_metrics)

        # RLM is $0.075 more expensive (0.090 - 0.015 = 0.075)
        assert comparison.cost_delta_usd == pytest.approx(0.075, abs=0.001)
        # That's 500% more expensive
        assert comparison.cost_delta_percent == pytest.approx(500, abs=1)

    def test_metrics_collector_compare_token_delta(self) -> None:
        """Test token delta calculation in collector."""
        collector = MetricsCollector()

        rlm_metrics = ExecutionMetrics(
            input_tokens=750,
            output_tokens=380,
            total_tokens=1130,
            cost_usd=0.090,
            cost_breakdown={"input": 0.0225, "output": 0.0675},
            execution_time_seconds=1.8,
            steps_taken=3,
            memory_used_mb=45.2,
            memory_peak_mb=62.1,
            success=True,
            execution_type="rlm",
        )

        direct_metrics = ExecutionMetrics(
            input_tokens=150,
            output_tokens=120,
            total_tokens=270,
            cost_usd=0.015,
            cost_breakdown={"input": 0.0045, "output": 0.0105},
            execution_time_seconds=0.8,
            steps_taken=0,
            memory_used_mb=12.5,
            memory_peak_mb=18.3,
            success=True,
            execution_type="direct",
        )

        comparison = collector.compare_metrics(rlm_metrics, direct_metrics)

        # RLM uses 860 more tokens (1130 - 270 = 860)
        assert comparison.token_delta == 860

    def test_metrics_collector_compare_time_delta(self) -> None:
        """Test time delta calculation in collector."""
        collector = MetricsCollector()

        rlm_metrics = ExecutionMetrics(
            input_tokens=750,
            output_tokens=380,
            total_tokens=1130,
            cost_usd=0.090,
            cost_breakdown={"input": 0.0225, "output": 0.0675},
            execution_time_seconds=1.8,
            steps_taken=3,
            memory_used_mb=45.2,
            memory_peak_mb=62.1,
            success=True,
            execution_type="rlm",
        )

        direct_metrics = ExecutionMetrics(
            input_tokens=150,
            output_tokens=120,
            total_tokens=270,
            cost_usd=0.015,
            cost_breakdown={"input": 0.0045, "output": 0.0105},
            execution_time_seconds=0.8,
            steps_taken=0,
            memory_used_mb=12.5,
            memory_peak_mb=18.3,
            success=True,
            execution_type="direct",
        )

        comparison = collector.compare_metrics(rlm_metrics, direct_metrics)

        # RLM is 1.0 second slower (1.8 - 0.8 = 1.0)
        assert comparison.time_delta_seconds == pytest.approx(1.0, abs=0.01)
        # That's 125% slower
        assert comparison.time_delta_percent == pytest.approx(125, abs=1)

    def test_metrics_collector_compare_recommendation(self) -> None:
        """Test recommendation generation in collector."""
        collector = MetricsCollector()

        rlm_metrics = ExecutionMetrics(
            input_tokens=750,
            output_tokens=380,
            total_tokens=1130,
            cost_usd=0.090,
            cost_breakdown={"input": 0.0225, "output": 0.0675},
            execution_time_seconds=1.8,
            steps_taken=3,
            memory_used_mb=45.2,
            memory_peak_mb=62.1,
            success=True,
            execution_type="rlm",
        )

        direct_metrics = ExecutionMetrics(
            input_tokens=150,
            output_tokens=120,
            total_tokens=270,
            cost_usd=0.015,
            cost_breakdown={"input": 0.0045, "output": 0.0105},
            execution_time_seconds=0.8,
            steps_taken=0,
            memory_used_mb=12.5,
            memory_peak_mb=18.3,
            success=True,
            execution_type="direct",
        )

        comparison = collector.compare_metrics(rlm_metrics, direct_metrics)

        assert comparison.recommendation
        # Should mention Direct is cheaper and faster
        assert "cheaper" in comparison.recommendation.lower()
        assert "faster" in comparison.recommendation.lower()


class TestChatManagerProcessMessage:
    """Test ChatManager.process_message() routing and execution."""

    @pytest.mark.asyncio
    async def test_process_message_rlm_only_mode(self, mock_execute_methods: None) -> None:
        """Test process_message in RLM-only mode."""
        manager = ChatManager()

        message = await manager.process_message(
            user_query="Summarize the document", mode="rlm_only"
        )

        # Should have RLM response and metrics
        assert message.rlm_response is not None
        assert message.rlm_metrics is not None
        assert message.rlm_trace is not None
        # Should NOT have direct response
        assert message.direct_response is None
        assert message.direct_metrics is None
        # Should NOT have comparison
        assert message.comparison_metrics is None
        # RLM should take at least 1 step (actual count varies by content)
        assert message.rlm_metrics.steps_taken >= 1

    @pytest.mark.asyncio
    async def test_process_message_direct_only_mode(self) -> None:
        """Test process_message in Direct-only mode."""
        manager = ChatManager()

        message = await manager.process_message(user_query="Quick answer", mode="direct_only")

        # Should NOT have RLM response
        assert message.rlm_response is None
        assert message.rlm_metrics is None
        # Should have direct response and metrics
        assert message.direct_response is not None
        assert message.direct_metrics is not None
        # Should NOT have comparison
        assert message.comparison_metrics is None
        # Should have 0 steps (direct call)
        assert message.direct_metrics.steps_taken == 0

    @pytest.mark.asyncio
    async def test_process_message_compare_mode_both_paths(self) -> None:
        """Test process_message in Compare mode executes both paths."""
        manager = ChatManager()

        message = await manager.process_message(user_query="Compare approaches", mode="compare")

        # Should have BOTH RLM and Direct responses
        assert message.rlm_response is not None
        assert message.rlm_metrics is not None
        assert message.rlm_trace is not None
        assert message.direct_response is not None
        assert message.direct_metrics is not None
        # Should have comparison
        assert message.comparison_metrics is not None

    @pytest.mark.asyncio
    async def test_process_message_comparison_metrics_calculated(
        self, mock_execute_methods: None
    ) -> None:
        """Test that comparison metrics are calculated correctly."""
        manager = ChatManager()

        message = await manager.process_message(user_query="Compare", mode="compare")

        comparison = message.comparison_metrics
        # Should have cost delta
        assert comparison.cost_delta_usd > 0  # RLM is more expensive
        assert comparison.cost_delta_percent > 0
        # Should have token delta
        assert comparison.token_delta > 0  # RLM has more tokens
        # Should have time delta
        assert comparison.time_delta_seconds > 0  # RLM is slower
        assert comparison.time_delta_percent > 0
        # Should have recommendation
        assert comparison.recommendation

    @pytest.mark.asyncio
    async def test_process_message_adds_to_session_state(self) -> None:
        """Test that process_message adds message to session state."""
        session_state: dict[str, object] = {}
        manager = ChatManager(session_state)

        # Initial state should have empty messages
        assert len(session_state["messages"]) == 0  # type: ignore[arg-type]

        message1 = await manager.process_message("First query")
        assert len(session_state["messages"]) == 1  # type: ignore[arg-type]
        assert session_state["messages"][0] == message1  # type: ignore[index]

        message2 = await manager.process_message("Second query")
        assert len(session_state["messages"]) == 2  # type: ignore[arg-type]
        assert session_state["messages"][1] == message2  # type: ignore[index]

    @pytest.mark.asyncio
    async def test_process_message_with_file_context(self) -> None:
        """Test process_message with file context."""
        manager = ChatManager()

        message = await manager.process_message(
            user_query="Summarize",
            mode="rlm_only",
            file_context="This is the document content...",
            file_info={"name": "document.txt", "size_bytes": 5000},
        )

        assert message.file_context == "This is the document content..."
        assert message.file_info["name"] == "document.txt"
        assert message.rlm_response is not None

    @pytest.mark.asyncio
    async def test_process_message_rlm_faster_than_direct_false(
        self, mock_execute_methods: None
    ) -> None:
        """Test that RLM is slower than Direct (as expected)."""
        manager = ChatManager()

        message = await manager.process_message(user_query="Test", mode="compare")

        # RLM should take longer (3 steps) than Direct (single call)
        assert (
            message.rlm_metrics.execution_time_seconds
            > message.direct_metrics.execution_time_seconds
        )

    @pytest.mark.asyncio
    async def test_process_message_rlm_more_tokens_than_direct(
        self, mock_execute_methods: None
    ) -> None:
        """Test that RLM uses more tokens than Direct."""
        manager = ChatManager()

        message = await manager.process_message(user_query="Test", mode="compare")

        # RLM with 3 steps uses more tokens than single Direct call
        assert message.rlm_metrics.total_tokens > message.direct_metrics.total_tokens

    @pytest.mark.asyncio
    async def test_process_message_error_handling_rlm(self) -> None:
        """Test error handling if RLM fails (graceful degradation)."""
        manager = ChatManager()

        # Mock a failure scenario by using an invalid mode that should still work
        # In real scenario, this would be a network error, etc.
        # For now, just test that the method completes without raising
        try:
            message = await manager.process_message("Test", mode="compare")
            # Should complete even if internal error occurs
            assert message.user_query == "Test"
        except Exception:
            pytest.fail("process_message should handle errors gracefully")


class TestChatManagerExportConversation:
    """Test ChatManager.export_conversation() for different formats."""

    @pytest.mark.asyncio
    async def test_export_conversation_json_format(self) -> None:
        """Test exporting conversation as JSON."""
        manager = ChatManager()

        await manager.process_message("Test query 1", mode="compare")

        export = manager.export_conversation("json")

        # Should be valid JSON
        import json

        data = json.loads(export)

        # Should have required fields
        assert "conversation_id" in data
        assert "message_count" in data
        assert "messages" in data
        assert data["message_count"] == 1

        # First message should have query and metrics
        msg_data = data["messages"][0]
        assert msg_data["user_query"] == "Test query 1"
        assert msg_data["mode"] == "compare"
        assert msg_data["rlm_metrics"] is not None
        assert msg_data["direct_metrics"] is not None
        assert msg_data["comparison"] is not None

    @pytest.mark.asyncio
    async def test_export_conversation_markdown_format(self) -> None:
        """Test exporting conversation as Markdown."""
        manager = ChatManager()

        await manager.process_message("Summarize", mode="rlm_only")

        export = manager.export_conversation("markdown")

        # Should have markdown structure
        assert "# Conversation Report" in export
        assert "## Query 1" in export
        assert "**Mode:**" in export
        assert "**Time:**" in export
        assert "### User Query" in export
        assert "### RLM Response" in export
        assert "**RLM Metrics:**" in export

    @pytest.mark.asyncio
    async def test_export_conversation_csv_format(self) -> None:
        """Test exporting conversation as CSV."""
        manager = ChatManager()

        await manager.process_message("Test query", mode="compare")

        export = manager.export_conversation("csv")

        # Should have CSV structure
        lines = export.strip().split("\n")

        # Should have header
        assert "Query" in lines[0]
        assert "Mode" in lines[0]
        assert "RLM_Tokens" in lines[0]
        assert "Direct_Tokens" in lines[0]

        # Should have data row
        assert len(lines) == 2  # Header + 1 message

    @pytest.mark.asyncio
    async def test_export_conversation_multiple_messages(self) -> None:
        """Test exporting conversation with multiple messages."""
        manager = ChatManager()

        await manager.process_message("Query 1", mode="rlm_only")
        await manager.process_message("Query 2", mode="direct_only")
        await manager.process_message("Query 3", mode="compare")

        json_export = manager.export_conversation("json")

        import json

        data = json.loads(json_export)
        assert data["message_count"] == 3
        assert len(data["messages"]) == 3

    @pytest.mark.asyncio
    async def test_export_conversation_json_has_cost_metrics(
        self, mock_execute_methods: None
    ) -> None:
        """Test that JSON export includes cost metrics."""
        manager = ChatManager()

        await manager.process_message("Test", mode="compare")

        export = manager.export_conversation("json")

        import json

        data = json.loads(export)
        msg = data["messages"][0]

        # Should have cost in metrics
        assert msg["rlm_metrics"]["cost_usd"] > 0
        assert msg["direct_metrics"]["cost_usd"] > 0
        assert msg["comparison"]["cost_delta_usd"] > 0

    @pytest.mark.asyncio
    async def test_export_conversation_markdown_has_recommendation(self) -> None:
        """Test that Markdown export includes recommendation."""
        manager = ChatManager()

        await manager.process_message("Test", mode="compare")

        export = manager.export_conversation("markdown")

        # Should include recommendation
        assert "### Comparison" in export
        assert "Deltas:" in export

    @pytest.mark.asyncio
    async def test_export_conversation_csv_escapes_quotes(self) -> None:
        """Test that CSV export properly escapes quotes in queries."""
        manager = ChatManager()

        query_with_quotes = 'What is "Python"?'
        await manager.process_message(query_with_quotes, mode="rlm_only")

        export = manager.export_conversation("csv")

        # CSV should have escaped quotes
        assert '""' in export  # Double quotes for escaping

    @pytest.mark.asyncio
    async def test_export_conversation_empty_conversation(self) -> None:
        """Test exporting empty conversation."""
        manager = ChatManager()

        json_export = manager.export_conversation("json")

        import json

        data = json.loads(json_export)
        assert data["message_count"] == 0
        assert len(data["messages"]) == 0

    @pytest.mark.asyncio
    async def test_export_conversation_invalid_format(self) -> None:
        """Test that invalid format raises ValueError."""
        manager = ChatManager()

        with pytest.raises(ValueError, match="Unsupported format"):
            manager.export_conversation("xml")

    @pytest.mark.asyncio
    async def test_export_conversation_preserves_timestamps(self) -> None:
        """Test that export preserves message timestamps."""
        manager = ChatManager()

        await manager.process_message("Test", mode="rlm_only")

        export = manager.export_conversation("json")

        import json

        data = json.loads(export)

        # Timestamps should be ISO format
        timestamp = data["messages"][0]["timestamp"]
        assert "T" in timestamp  # ISO format includes T
        assert isinstance(timestamp, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
