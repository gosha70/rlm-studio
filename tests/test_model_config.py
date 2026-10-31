"""
Tests for multi-model configuration.

This tests the Bet 4 implementation - using different models for root vs sub-agents.
"""

import pytest

from rlmstudio.core.model_config import ModelConfig, ModelMetrics


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_create_basic_config(self):
        """Test creating basic model configuration."""
        config = ModelConfig(
            root_model="gpt-4o",
            sub_model="gpt-4o-mini",
            root_provider="openai",
            sub_provider="openai",
        )

        assert config.root_model == "gpt-4o"
        assert config.sub_model == "gpt-4o-mini"
        assert config.root_provider == "openai"
        assert config.sub_provider == "openai"

    def test_cross_provider_config(self):
        """Test configuration with different providers."""
        config = ModelConfig(
            root_model="claude-3-opus",
            root_provider="anthropic",
            sub_model="llama3",
            sub_provider="ollama",
        )

        assert config.root_provider == "anthropic"
        assert config.sub_provider == "ollama"

    def test_from_single_model(self):
        """Test creating config from single model."""
        config = ModelConfig.from_single_model(model="gpt-4o", provider="openai")

        assert config.root_model == "gpt-4o"
        assert config.sub_model == "gpt-4o"
        assert config.root_provider == "openai"
        assert config.sub_provider == "openai"

    def test_cost_optimized_openai(self):
        """Test cost-optimized configuration for OpenAI."""
        config = ModelConfig.cost_optimized("openai")

        assert config.root_model == "gpt-4o"
        assert config.sub_model == "gpt-4o-mini"
        assert config.root_provider == "openai"
        assert config.sub_provider == "openai"

    def test_cost_optimized_anthropic(self):
        """Test cost-optimized configuration for Anthropic."""
        config = ModelConfig.cost_optimized("anthropic")

        # Against the catalog rather than a literal: this test previously
        # pinned claude-3-5-sonnet-20241022, which Anthropic has since
        # retired, so it enforced the bug instead of catching it.
        from rlmstudio.ui.data.providers_catalog import (
            DEFAULT_ANTHROPIC_CHEAP_MODEL,
            DEFAULT_ANTHROPIC_MODEL,
        )

        assert config.root_model == DEFAULT_ANTHROPIC_MODEL
        assert config.sub_model == DEFAULT_ANTHROPIC_CHEAP_MODEL
        assert config.root_provider == "anthropic"
        assert config.sub_provider == "anthropic"

    def test_cost_optimized_mixed(self):
        """Test cost-optimized mixed configuration."""
        config = ModelConfig.cost_optimized("mixed")

        assert config.root_provider == "anthropic"
        assert config.sub_provider == "ollama"
        assert "claude" in config.root_model.lower()
        assert "llama" in config.sub_model.lower()

    def test_cost_optimized_invalid_provider(self):
        """Test that invalid provider raises error."""
        with pytest.raises(ValueError) as exc_info:
            ModelConfig.cost_optimized("invalid")

        assert "Unknown provider" in str(exc_info.value)

    def test_str_representation(self):
        """Test string representation."""
        config = ModelConfig(
            root_model="gpt-4o",
            sub_model="gpt-4o-mini",
            root_provider="openai",
            sub_provider="openai",
        )

        str_repr = str(config)
        assert "openai:gpt-4o" in str_repr
        assert "openai:gpt-4o-mini" in str_repr


class TestModelMetrics:
    """Test ModelMetrics dataclass."""

    def test_create_empty_metrics(self):
        """Test creating empty metrics."""
        metrics = ModelMetrics()

        assert metrics.root_calls == 0
        assert metrics.root_tokens == 0
        assert metrics.root_cost == 0.0
        assert metrics.sub_calls == 0
        assert metrics.sub_tokens == 0
        assert metrics.sub_cost == 0.0

    def test_total_calls(self):
        """Test total calls calculation."""
        metrics = ModelMetrics(root_calls=2, sub_calls=8)

        assert metrics.total_calls == 10

    def test_total_tokens(self):
        """Test total tokens calculation."""
        metrics = ModelMetrics(root_tokens=1000, sub_tokens=5000)

        assert metrics.total_tokens == 6000

    def test_total_cost(self):
        """Test total cost calculation."""
        metrics = ModelMetrics(root_cost=0.15, sub_cost=0.02)

        assert abs(metrics.total_cost - 0.17) < 0.001

    def test_savings_calculation(self):
        """Test savings vs single model calculation."""
        metrics = ModelMetrics(root_tokens=1000, root_cost=0.10, sub_tokens=5000, sub_cost=0.05)

        # If we used expensive model for everything
        # 6000 tokens * 0.0001 = 0.60
        # Actual cost: 0.15
        # Savings: 0.45
        single_model_cost_per_token = 0.0001
        savings = metrics.savings_vs_single_model(single_model_cost_per_token)

        assert savings > 0  # We saved money
        assert abs(savings - 0.45) < 0.001

    def test_to_dict(self):
        """Test converting to dictionary."""
        metrics = ModelMetrics(
            root_calls=2,
            root_tokens=1000,
            root_cost=0.10,
            sub_calls=8,
            sub_tokens=5000,
            sub_cost=0.05,
        )

        metrics_dict = metrics.to_dict()

        assert metrics_dict["root_calls"] == 2
        assert metrics_dict["root_tokens"] == 1000
        assert metrics_dict["root_cost"] == 0.10
        assert metrics_dict["sub_calls"] == 8
        assert metrics_dict["sub_tokens"] == 5000
        assert metrics_dict["sub_cost"] == 0.05
        assert metrics_dict["total_calls"] == 10
        assert metrics_dict["total_tokens"] == 6000
        assert abs(metrics_dict["total_cost"] - 0.15) < 0.001


class TestModelConfigIntegration:
    """Integration tests for model configuration."""

    def test_realistic_cost_optimization_scenario(self):
        """Test a realistic cost optimization scenario."""
        # Scenario: Large document analysis with multiple subcalls
        ModelConfig.cost_optimized("openai")

        # Simulate metrics from actual usage
        metrics = ModelMetrics()

        # Root model: 2 calls for main reasoning (expensive)
        metrics.root_calls = 2
        metrics.root_tokens = 5000
        metrics.root_cost = 0.15  # GPT-4o pricing

        # Sub model: 8 subcalls for exploration (cheap)
        metrics.sub_calls = 8
        metrics.sub_tokens = 20000
        metrics.sub_cost = 0.02  # GPT-4o-mini pricing

        # Calculate savings
        # If we used GPT-4o for everything: 25000 * 0.00003 = 0.75
        # Actual cost: 0.17
        # Savings: 0.58 (77% cost reduction!)

        gpt4_cost_per_token = 0.00003
        savings = metrics.savings_vs_single_model(gpt4_cost_per_token)

        assert savings > 0.5  # Significant savings
        assert metrics.total_cost < 0.20

        # Verify the cost breakdown makes sense
        assert metrics.root_cost > metrics.sub_cost  # Root is more expensive per token
        assert metrics.sub_tokens > metrics.root_tokens  # But subs do more work

    def test_mixed_provider_scenario(self):
        """Test mixed provider configuration."""
        config = ModelConfig.cost_optimized("mixed")

        # This uses Claude Opus for root (quality) and local Llama for subs (free!)
        assert config.root_provider == "anthropic"
        assert config.sub_provider == "ollama"

        # Simulate metrics
        metrics = ModelMetrics()
        metrics.root_calls = 2
        metrics.root_tokens = 5000
        metrics.root_cost = 0.15  # Claude pricing

        # Local model is free!
        metrics.sub_calls = 10
        metrics.sub_tokens = 30000
        metrics.sub_cost = 0.00  # Ollama is local/free

        assert metrics.total_cost == 0.15

        # Maximum savings: Only pay for root calls
        claude_cost_per_token = 0.00003
        savings = metrics.savings_vs_single_model(claude_cost_per_token)

        # Savings = (35000 * 0.00003) - 0.15 = 1.05 - 0.15 = 0.90
        assert savings > 0.8  # Huge savings with local model!


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
