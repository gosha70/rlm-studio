"""Phase A verification tests: provider config persistence, runtime settings, cost calculation.

These tests verify the fixes for Bugs #1, #3, and #7 from the outstanding bugs handoff.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from rlmstudio.server.app import app
from rlmstudio.server.dependencies import AppState, reset_state
from rlmstudio.server.models import ProviderConfig, RuntimeSettings


@pytest.fixture(autouse=True)
def _clean_state() -> None:
    reset_state()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# Bug #1: Config persistence — single save path, reconciliation
# ---------------------------------------------------------------------------


class TestConfigPersistence:
    def test_save_provider_sets_active(self, client: TestClient) -> None:
        """PUT /api/providers/{name} with enabled=True sets active_provider/model."""
        resp = client.put(
            "/api/providers/anthropic",
            json={"model": "claude-sonnet-4-5-20250929", "enabled": True},
        )
        assert resp.status_code == 200
        assert resp.json()["saved"] is True

        cfg = client.get("/api/config").json()
        assert cfg["active_provider"] == "anthropic"
        assert cfg["active_model"] == "claude-sonnet-4-5-20250929"

    def test_save_provider_disables_others(self, client: TestClient) -> None:
        """Enabling one provider disables all others."""
        # Enable openai first
        client.put(
            "/api/providers/openai",
            json={"model": "gpt-4o", "enabled": True},
        )
        # Now enable anthropic
        client.put(
            "/api/providers/anthropic",
            json={"model": "claude-sonnet-4-5-20250929", "enabled": True},
        )
        cfg = client.get("/api/config").json()
        for pc in cfg["provider_configs"]:
            if pc["provider"] == "anthropic":
                assert pc["enabled"] is True
            else:
                assert pc["enabled"] is False

    def test_update_config_ignores_active_fields(self, client: TestClient) -> None:
        """PUT /api/config cannot change active_provider/active_model."""
        initial = client.get("/api/config").json()
        resp = client.put(
            "/api/config",
            json={"active_provider": "anthropic", "active_model": "claude-3-opus"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_provider"] == initial["active_provider"]
        assert data["active_model"] == initial["active_model"]

    def test_provider_configs_replacement_reconciles_active(self, client: TestClient) -> None:
        """PUT /api/config with provider_configs reconciles active_provider/model."""
        # First, set up openai as the active provider (default state)
        initial = client.get("/api/config").json()
        assert initial["active_provider"] == "openai"
        assert initial["active_model"] == "gpt-4o"

        # Now replace provider_configs with one where openai has a different model
        resp = client.put(
            "/api/config",
            json={
                "provider_configs": [
                    {
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "enabled": True,
                        "runtime_settings": {
                            "temperature": 0.7,
                            "top_p": 1.0,
                            "max_output_tokens": 4096,
                            "timeout_seconds": 30,
                        },
                    }
                ]
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        # After reconciliation, active_model must match the provider_configs entry
        assert data["active_provider"] == "openai"
        assert data["active_model"] == "gpt-4o-mini"

    def test_reconcile_active_provider_on_load(self) -> None:
        """_reconcile_active_provider fixes a missing provider_configs entry."""
        state = AppState(load_from_disk=False)
        state.config.active_provider = "anthropic"
        state.config.active_model = "claude-sonnet-4-5-20250929"
        state.config.provider_configs = []  # Missing entry!
        state.save_config = lambda: None  # type: ignore[assignment]

        state._reconcile_active_provider()

        # Should auto-create an enabled entry for the active provider
        assert len(state.config.provider_configs) == 1
        pc = state.config.provider_configs[0]
        assert pc.provider == "anthropic"
        assert pc.enabled is True
        assert pc.model == "claude-sonnet-4-5-20250929"

    def test_reconcile_fixes_provider_model_mismatch(self) -> None:
        """_reconcile_active_provider aligns active_model with entry's model."""
        state = AppState(load_from_disk=False)
        state.config.active_provider = "openai"
        state.config.active_model = "claude-sonnet-4-5-20250929"  # Wrong!
        state.config.provider_configs = [
            ProviderConfig(provider="openai", model="gpt-4o", enabled=True),
        ]
        state.save_config = lambda: None  # type: ignore[assignment]

        state._reconcile_active_provider()

        # active_model should be corrected to match the provider entry
        assert state.config.active_model == "gpt-4o"


# ---------------------------------------------------------------------------
# Bug #7: Runtime settings applied to LLM adapter
# ---------------------------------------------------------------------------


class TestRuntimeSettingsApplied:
    def test_create_llm_adapter_uses_provider_runtime_settings(self) -> None:
        """create_llm_adapter() passes runtime_settings from the active provider."""
        state = AppState(load_from_disk=False)
        state.config.active_provider = "openai"
        state.config.active_model = "gpt-4o"
        custom_settings = RuntimeSettings(
            temperature=0.2, top_p=0.9, max_output_tokens=2048, timeout_seconds=60
        )
        state.config.provider_configs = [
            ProviderConfig(
                provider="openai",
                model="gpt-4o",
                enabled=True,
                runtime_settings=custom_settings,
            ),
        ]

        adapter = state.create_llm_adapter()

        assert adapter._temperature == 0.2
        assert adapter._max_tokens == 2048
        assert adapter._timeout == 60.0

    def test_create_llm_adapter_falls_back_to_defaults(self) -> None:
        """create_llm_adapter() uses default_runtime_settings when no provider config."""
        state = AppState(load_from_disk=False)
        state.config.active_provider = "openai"
        state.config.active_model = "gpt-4o"
        state.config.provider_configs = []  # No configs
        state.config.default_runtime_settings = RuntimeSettings(
            temperature=0.4, max_output_tokens=8192, timeout_seconds=120
        )

        adapter = state.create_llm_adapter()

        assert adapter._temperature == 0.4
        assert adapter._max_tokens == 8192
        assert adapter._timeout == 120.0

    def test_chat_provider_adapter_uses_resolved_settings(self) -> None:
        """create_llm_adapter_for_chat_provider() applies profile runtime settings."""
        from rlmstudio.server.models import ChatProviderConfig

        state = AppState(load_from_disk=False)
        cp = ChatProviderConfig(
            id="test-cp-1",
            name="Test Provider",
            llm_provider="openai",
            llm_model="gpt-4o",
            execution_mode="direct",
            runtime_settings=RuntimeSettings(
                temperature=0.1, max_output_tokens=1024, timeout_seconds=15
            ),
        )
        state.config.chat_providers = [cp]

        adapter = state.create_llm_adapter_for_chat_provider("test-cp-1")

        assert adapter._temperature == 0.1
        assert adapter._max_tokens == 1024
        assert adapter._timeout == 15.0

    def test_lm_studio_gets_a_placeholder_key_never_the_openai_env_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LM Studio is keyless but routed as openai/<model>; without a key
        LiteLLM fell back to OPENAI_API_KEY (failing without it, leaking it
        to the local server with it). Chat, the official engine and the
        global adapter must all send the placeholder instead."""
        from rlmstudio.infrastructure.engines.rlms_adapter import RlmsEngineAdapter
        from rlmstudio.server import dependencies
        from rlmstudio.server.models import ChatProviderConfig, LLMProviderConfig
        from rlmstudio.ui.data.providers_catalog import KEYLESS_API_KEY_PLACEHOLDER

        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai")
        monkeypatch.setattr(dependencies, "_get_instance_api_key", lambda _id, _backend: None)
        state = AppState(load_from_disk=False)
        state.config.llm_providers = [
            LLMProviderConfig(id="lp-lm", name="LM", backend="lmstudio", model="qwen")
        ]
        state.config.chat_providers = [
            ChatProviderConfig(id="cp-lm", name="LM chat", llm_provider_id="lp-lm")
        ]

        adapter = state.create_llm_adapter_for_chat_provider("cp-lm")
        engine = state.create_rlm_engine_for_chat_provider("cp-lm")
        state.config.active_provider = "lmstudio"
        state.config.active_model = "qwen"
        global_adapter = state.create_llm_adapter()

        assert isinstance(engine, RlmsEngineAdapter)
        assert adapter._api_key == KEYLESS_API_KEY_PLACEHOLDER
        assert engine._api_key == KEYLESS_API_KEY_PLACEHOLDER
        assert global_adapter._api_key == KEYLESS_API_KEY_PLACEHOLDER

    def test_cloud_provider_without_a_key_keeps_litellm_env_lookup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The placeholder is for keyless local servers only: an OpenAI
        provider with no stored key still leaves the key to LiteLLM."""
        state = AppState(load_from_disk=False)
        state.config.active_provider = "openai"
        state.config.active_model = "gpt-4o"

        assert state.create_llm_adapter()._api_key is None


# ---------------------------------------------------------------------------
# Bug #3: Cost calculation
# ---------------------------------------------------------------------------


class TestCostCalculation:
    def test_compute_cost_uses_get_completion_cost(self) -> None:
        """_compute_cost prefers get_completion_cost() over get_pricing()."""
        from rlmstudio.application.use_cases.run_direct import RunDirectUseCase

        class FakeLLM:
            def get_completion_cost(self, input_tokens: int, output_tokens: int) -> float:
                return 0.0042

            def get_pricing(self) -> dict[str, float]:
                # Should NOT be called
                raise AssertionError("get_pricing should not be called")

        uc = RunDirectUseCase(FakeLLM())  # type: ignore[arg-type]
        cost = uc._compute_cost(1000, 500)
        assert cost == 0.0042

    def test_compute_cost_falls_back_to_get_pricing(self) -> None:
        """_compute_cost falls back to get_pricing() if get_completion_cost is missing."""
        from rlmstudio.application.use_cases.run_direct import RunDirectUseCase

        class FakeLLM:
            def get_pricing(self) -> dict[str, float]:
                return {"input_cost_per_1m": 3.0, "output_cost_per_1m": 15.0}

        uc = RunDirectUseCase(FakeLLM())  # type: ignore[arg-type]
        cost = uc._compute_cost(1000, 500)
        # 1000 * 3.0 / 1M + 500 * 15.0 / 1M = 0.003 + 0.0075 = 0.0105
        assert abs(cost - 0.0105) < 1e-6

    def test_get_completion_cost_logs_on_failure(self) -> None:
        """get_completion_cost logs at debug level when model not found."""
        from rlmstudio.infrastructure.llm.litellm_adapter import LiteLLMAdapter

        adapter = LiteLLMAdapter(model="nonexistent/fake-model-xyz")
        # Should not raise, should return 0.0
        cost = adapter.get_completion_cost(100, 50)
        assert cost == 0.0

    def test_get_pricing_strips_prefix(self) -> None:
        """get_pricing tries both prefixed and stripped model names."""
        from rlmstudio.infrastructure.llm.litellm_adapter import LiteLLMAdapter

        adapter = LiteLLMAdapter(model="anthropic/claude-sonnet-4-5-20250929")
        pricing = adapter.get_pricing()
        # Should return a dict (may have 0.0 values if model unknown to litellm)
        assert "input_cost_per_1m" in pricing
        assert "output_cost_per_1m" in pricing


# ---------------------------------------------------------------------------
# LiteLLM model name prefixing
# ---------------------------------------------------------------------------


class TestModelNamePrefixing:
    def test_openai_no_prefix(self) -> None:
        assert AppState._litellm_model_name("openai", "gpt-4o") == "gpt-4o"

    def test_anthropic_prefix(self) -> None:
        result = AppState._litellm_model_name("anthropic", "claude-sonnet-4-5-20250929")
        assert result == "anthropic/claude-sonnet-4-5-20250929"

    def test_ollama_prefix(self) -> None:
        assert AppState._litellm_model_name("ollama", "qwen3") == "ollama/qwen3"

    def test_already_prefixed_unchanged(self) -> None:
        assert (
            AppState._litellm_model_name("anthropic", "anthropic/claude-3-opus")
            == "anthropic/claude-3-opus"
        )

    def test_lmstudio_uses_openai_prefix(self) -> None:
        assert AppState._litellm_model_name("lmstudio", "local-model") == "openai/local-model"

    def test_vllm_uses_hosted_vllm_prefix(self) -> None:
        """vLLM must use hosted_vllm/, not openai/.

        With LiteLLM >= 1.50 an ``openai/*`` model on a vLLM upstream
        auto-routes to ``/v1/responses``, which returns HTTP 400 on the second
        turn of a multi-turn conversation (Blocker #4 in
        ``docs/hosts/dgx-spark-vllm.md`` §7).
        """
        result = AppState._litellm_model_name("vllm", "RedHatAI/Qwen3-Coder-Next-NVFP4")
        assert result == "hosted_vllm/RedHatAI/Qwen3-Coder-Next-NVFP4"
        assert not result.startswith("openai/")

    def test_vllm_hf_style_id_is_prefixed_once(self) -> None:
        """A HuggingFace id contains a slash but is not already prefixed."""
        assert (
            AppState._litellm_model_name("vllm", "hosted_vllm/RedHatAI/Qwen3-Coder-Next-NVFP4")
            == "hosted_vllm/RedHatAI/Qwen3-Coder-Next-NVFP4"
        )

    def test_vllm_prefix_agrees_across_all_three_tables(self) -> None:
        """The prefix table is duplicated in three modules; they must not drift.

        ``api.py`` serves ``interact()``, ``server/routes/providers.py`` serves
        the Studio provider form and ``provider_tester``, and
        ``server/dependencies.py`` serves the request dispatch path. A vLLM fix
        applied to only one of them leaves the other paths broken.
        """
        from rlmstudio.api import _PROVIDER_PREFIXES
        from rlmstudio.server.routes.providers import _LITELLM_PREFIXES

        assert _PROVIDER_PREFIXES["vllm"] == "hosted_vllm/"
        assert _LITELLM_PREFIXES["vllm"] == "hosted_vllm/"
        assert AppState._litellm_model_name("vllm", "m") == "hosted_vllm/m"

    def test_vllm_model_name_reaches_chat_completions_route(self) -> None:
        """The Studio provider path must produce a hosted_vllm/ model string.

        This is the regression guard for the Studio-side half of Blocker #4:
        ``provider_tester`` and the provider form both route through
        ``server/routes/providers._litellm_model_name``.
        """
        from rlmstudio.server.routes.providers import _litellm_model_name

        assert (
            _litellm_model_name("vllm", "RedHatAI/Qwen3-Coder-Next-NVFP4")
            == "hosted_vllm/RedHatAI/Qwen3-Coder-Next-NVFP4"
        )

    def test_vllm_model_string_resolves_to_hosted_vllm_route(self) -> None:
        """LiteLLM must resolve our vLLM model string to the hosted_vllm provider.

        This is the end of the chain Blocker #4 breaks: LiteLLM picks the
        upstream route from the provider it infers. ``hosted_vllm`` pins to
        ``/v1/chat/completions``; ``openai`` is auto-detected as the Responses
        API and routed to ``/v1/responses``, which vLLM rejects on multi-turn.
        Resolution is offline — no request is made.
        """
        from litellm import get_llm_provider

        model_str = AppState._litellm_model_name("vllm", "RedHatAI/Qwen3-Coder-Next-NVFP4")
        _, provider, _, _ = get_llm_provider(model=model_str, api_base="http://spark:8000/v1")

        assert provider == "hosted_vllm", (
            f"vLLM model string {model_str!r} resolved to provider {provider!r}; "
            "'openai' would route multi-turn requests to /v1/responses (Blocker #4)"
        )
