"""
Tests for LLM provider auto-configuration.

This tests the Bet 2 implementation - auto-detection of providers from environment variables.
"""

import pytest

from rlmstudio.llm.auto import (
    ConfigurationError,
    auto_detect_provider,
    get_default_client_config,
    get_provider_info,
    list_available_providers,
)


class TestAutoDetectProvider:
    """Test auto-detection of providers from environment."""

    def test_detects_openai_first(self, monkeypatch):
        """Test that OpenAI is detected when OPENAI_API_KEY is set."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")

        provider = auto_detect_provider()
        assert provider == "openai"

    def test_detects_anthropic_when_no_openai(self, monkeypatch):
        """Test that Anthropic is detected when only ANTHROPIC_API_KEY is set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        provider = auto_detect_provider()
        assert provider == "anthropic"

    def test_detects_ollama_when_no_api_keys(self, monkeypatch):
        """Test that Ollama is detected when only OLLAMA_BASE_URL is set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

        provider = auto_detect_provider()
        assert provider == "ollama"

    def test_returns_none_when_no_providers(self, monkeypatch):
        """Test that None is returned when no providers are configured."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("LMSTUDIO_BASE_URL", raising=False)
        monkeypatch.delenv("VLLM_BASE_URL", raising=False)

        provider = auto_detect_provider()
        assert provider is None

    def test_priority_order(self, monkeypatch):
        """Test that providers are detected in correct priority order."""
        # Set all providers
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # OpenAI should be first
        provider = auto_detect_provider()
        assert provider == "openai"

        # Remove OpenAI, Anthropic should be next
        monkeypatch.delenv("OPENAI_API_KEY")
        provider = auto_detect_provider()
        assert provider == "anthropic"

        # Remove Anthropic, Ollama should be next
        monkeypatch.delenv("ANTHROPIC_API_KEY")
        provider = auto_detect_provider()
        assert provider == "ollama"


class TestListAvailableProviders:
    """Test listing of available providers."""

    def test_lists_all_configured_providers(self, monkeypatch):
        """Test that all configured providers are listed."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

        providers = list_available_providers()
        assert "openai" in providers
        assert "anthropic" in providers
        assert "ollama" not in providers

    def test_empty_list_when_no_providers(self, monkeypatch):
        """Test that empty list is returned when no providers configured."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("LMSTUDIO_BASE_URL", raising=False)
        monkeypatch.delenv("VLLM_BASE_URL", raising=False)

        providers = list_available_providers()
        assert providers == []

    def test_lists_local_providers(self, monkeypatch):
        """Test that local providers (Ollama, LM Studio) are listed."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://localhost:1234")

        providers = list_available_providers()
        assert "ollama" in providers
        assert "lmstudio" in providers
        assert len(providers) == 2


class TestGetProviderInfo:
    """Test getting detailed provider information."""

    def test_shows_configured_status(self, monkeypatch):
        """Test that configured status is correctly reported."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        info = get_provider_info()

        assert info["openai"]["configured"] is True
        assert info["openai"]["status"] == "ready"
        assert info["anthropic"]["configured"] is False
        assert info["anthropic"]["status"] == "missing"

    def test_includes_key_preview(self, monkeypatch):
        """Test that API key preview is included (first 8 chars)."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123456789")

        info = get_provider_info()

        assert info["openai"]["key_preview"] == "sk-test1..."

    def test_includes_base_url_for_local(self, monkeypatch):
        """Test that base URL is included for local providers."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

        info = get_provider_info()

        assert info["ollama"]["base_url"] == "http://localhost:11434"
        assert info["ollama"]["configured"] is True


class TestGetDefaultClientConfig:
    """Test getting default client configuration."""

    def test_returns_provider_and_model(self, monkeypatch):
        """Test that provider and default model are returned."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        provider, model = get_default_client_config()

        assert provider == "openai"
        assert model == "gpt-4o-mini"

    def test_raises_error_when_no_provider(self, monkeypatch):
        """Test that ConfigurationError is raised when no provider configured."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("LMSTUDIO_BASE_URL", raising=False)
        monkeypatch.delenv("VLLM_BASE_URL", raising=False)

        with pytest.raises(ConfigurationError) as exc_info:
            get_default_client_config()

        assert "No LLM provider configured" in str(exc_info.value)
        assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_returns_correct_defaults_for_each_provider(self, monkeypatch):
        """Test that correct default models are returned for each provider."""
        # Test OpenAI
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        provider, model = get_default_client_config()
        assert provider == "openai"
        assert model == "gpt-4o-mini"

        # Test Anthropic
        monkeypatch.delenv("OPENAI_API_KEY")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        provider, model = get_default_client_config()
        assert provider == "anthropic"
        # Catalog-derived, not a literal: the pinned value was retired.
        from rlmstudio.domain.model_catalog import DEFAULT_ANTHROPIC_MODEL

        assert model == DEFAULT_ANTHROPIC_MODEL

        # Test Ollama
        monkeypatch.delenv("ANTHROPIC_API_KEY")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        provider, model = get_default_client_config()
        assert provider == "ollama"
        assert model == "llama3.2"


class TestInteractAutoConfig:
    """Test that interact() uses auto-configuration."""

    def test_interact_uses_auto_detection(self, monkeypatch):
        """Test that interact() auto-detects provider when not specified."""
        from unittest.mock import patch as _patch

        from rlmstudio.api import interact
        from rlmstudio.application.dto import RunResultDTO

        fake = RunResultDTO(answer="Test answer", mode_used="direct", success=True)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with (
            _patch("rlmstudio.api.RunDirectUseCase") as mock_uc,
            _patch("rlmstudio.api.LiteLLMAdapter"),
        ):
            mock_uc.return_value.execute.return_value = fake
            result = interact("Test content", "Test query", mode="direct")

        assert result.answer == "Test answer"

    def test_interact_raises_error_with_no_provider(self, monkeypatch):
        """Test that interact() raises ValueError when no provider available."""
        from rlmstudio.api import interact

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

        with pytest.raises(ValueError, match="No LLM provider configured"):
            interact("Test content", "Test query")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
