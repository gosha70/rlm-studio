# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""LLM provider configuration."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelPricing:
    """Pricing information for a specific model."""

    input_cost_per_1m: float
    """Cost per 1 million input tokens in USD"""

    output_cost_per_1m: float
    """Cost per 1 million output tokens in USD"""

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "input_cost_per_1m": self.input_cost_per_1m,
            "output_cost_per_1m": self.output_cost_per_1m,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "ModelPricing":
        """Create from dictionary."""
        return cls(
            input_cost_per_1m=data["input_cost_per_1m"],
            output_cost_per_1m=data["output_cost_per_1m"],
        )


@dataclass
class LLMProviderConfig:
    """Configuration for LLM provider."""

    provider: str
    """Provider name (e.g., 'openai', 'anthropic', 'ollama', 'vllm')"""

    model: str
    """Model identifier"""

    api_key: str | None = None
    """API key (if None, will use environment variable)"""

    base_url: str | None = None
    """Custom base URL for API"""

    temperature: float = 0.7
    """Sampling temperature (0.0-1.0)"""

    max_tokens: int | None = None
    """Maximum tokens to generate"""

    organization: str | None = None
    """Organization ID (OpenAI only)"""

    extra_params: dict[str, Any] = field(default_factory=dict)
    """Additional provider-specific parameters"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "organization": self.organization,
            "extra_params": self.extra_params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMProviderConfig":
        """Create from dictionary."""
        return cls(
            provider=data["provider"],
            model=data["model"],
            api_key=data.get("api_key"),
            base_url=data.get("base_url"),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens"),
            organization=data.get("organization"),
            extra_params=data.get("extra_params", {}),
        )


@dataclass
class LLMConfig:
    """LLM configuration including provider settings and pricing."""

    # Default provider to use
    default_provider: LLMProviderConfig | None = None

    # Model pricing database
    pricing: dict[str, ModelPricing] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_provider": self.default_provider.to_dict() if self.default_provider else None,
            "pricing": {model: pricing.to_dict() for model, pricing in self.pricing.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMConfig":
        """Create from dictionary."""
        default_provider = None
        if data.get("default_provider"):
            default_provider = LLMProviderConfig.from_dict(data["default_provider"])

        pricing = {}
        if "pricing" in data:
            pricing = {
                model: ModelPricing.from_dict(price_data)
                for model, price_data in data["pricing"].items()
            }

        return cls(
            default_provider=default_provider,
            pricing=pricing,
        )

    def get_pricing(self, model: str) -> ModelPricing | None:
        """
        Get pricing for a model.

        Args:
            model: Model identifier

        Returns:
            ModelPricing if found, None otherwise
        """
        return self.pricing.get(model)
