# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""
Multi-model configuration for RLM.

This module enables using different models for root reasoning vs. sub-agent exploration,
allowing significant cost optimization without sacrificing quality.
"""

from dataclasses import dataclass

from rlmstudio.domain.model_catalog import (
    DEFAULT_ANTHROPIC_CHEAP_MODEL,
    DEFAULT_ANTHROPIC_HIGH_QUALITY_MODEL,
    DEFAULT_ANTHROPIC_MODEL,
)

from ..llm import get_llm_client
from ..llm.base import BaseLLMProvider


@dataclass
class ModelConfig:
    """
    Configuration for multi-model RLM execution.

    This allows specifying different models for:
    - Root agent: Main reasoning and final answer generation
    - Sub-agents: Exploration and information gathering subcalls

    This can reduce costs by 50-80% while maintaining quality, as exploration
    tasks can use cheaper models while critical reasoning uses powerful models.

    Attributes:
        root_model: Model name for root agent (e.g., "gpt-4o", "claude-3-opus")
        root_provider: Provider for root model (e.g., "openai", "anthropic")
        sub_model: Model name for sub-agents (e.g., "gpt-4o-mini", "llama3")
        sub_provider: Provider for sub-agents
        root_api_key: API key for root provider (if not in environment)
        sub_api_key: API key for sub provider (if not in environment)

    Examples:
        >>> # Use GPT-4 for root, GPT-4-mini for subs (same provider)
        >>> config = ModelConfig(
        ...     root_model="gpt-4o",
        ...     sub_model="gpt-4o-mini",
        ...     root_provider="openai",
        ...     sub_provider="openai"
        ... )

        >>> # Cross-provider: Claude for root, local Llama for subs
        >>> config = ModelConfig(
        ...     root_model="claude-3-opus",
        ...     root_provider="anthropic",
        ...     sub_model="llama3-70b",
        ...     sub_provider="ollama"
        ... )
    """

    root_model: str
    sub_model: str
    root_provider: str | None = None
    sub_provider: str | None = None
    root_api_key: str | None = None
    sub_api_key: str | None = None

    def get_root_client(self) -> BaseLLMProvider:
        """
        Get configured LLM client for root agent.

        Returns:
            Configured root client
        """
        if self.root_provider is None:
            from ..llm.auto import auto_detect_provider

            self.root_provider = auto_detect_provider()
            if self.root_provider is None:
                raise ValueError(
                    "No provider specified for root model and auto-detection failed. "
                    "Please specify root_provider or set environment variables."
                )

        return get_llm_client(
            provider=self.root_provider, model=self.root_model, api_key=self.root_api_key
        )

    def get_sub_client(self) -> BaseLLMProvider:
        """
        Get configured LLM client for sub-agents.

        Returns:
            Configured sub-agent client
        """
        if self.sub_provider is None:
            # If no sub provider specified, use same as root
            if self.root_provider is None:
                from ..llm.auto import auto_detect_provider

                self.sub_provider = auto_detect_provider()
                if self.sub_provider is None:
                    raise ValueError(
                        "No provider specified for sub model and auto-detection failed. "
                        "Please specify sub_provider or set environment variables."
                    )
            else:
                self.sub_provider = self.root_provider

        return get_llm_client(
            provider=self.sub_provider, model=self.sub_model, api_key=self.sub_api_key
        )

    @classmethod
    def from_single_model(
        cls, model: str, provider: str | None = None, api_key: str | None = None
    ) -> "ModelConfig":
        """
        Create ModelConfig using same model for both root and sub-agents.

        This is equivalent to single-model RLM execution.

        Args:
            model: Model name to use for both root and subs
            provider: Provider name (auto-detected if None)
            api_key: API key (from environment if None)

        Returns:
            ModelConfig with same model for root and subs
        """
        return cls(
            root_model=model,
            sub_model=model,
            root_provider=provider,
            sub_provider=provider,
            root_api_key=api_key,
            sub_api_key=api_key,
        )

    @classmethod
    def cost_optimized(cls, provider: str = "openai") -> "ModelConfig":
        """
        Create a cost-optimized configuration for common providers.

        This uses best-in-class models for root and cheap models for subs.

        Args:
            provider: Provider to use ("openai", "anthropic", or "mixed")

        Returns:
            Cost-optimized ModelConfig

        Examples:
            >>> # OpenAI: GPT-4o for root, GPT-4o-mini for subs
            >>> config = ModelConfig.cost_optimized("openai")

            >>> # Anthropic: Sonnet for root, Haiku for subs
            >>> config = ModelConfig.cost_optimized("anthropic")

            >>> # Mixed: high-quality Claude for root, local Llama for subs
            >>> config = ModelConfig.cost_optimized("mixed")
        """
        if provider == "openai":
            return cls(
                root_model="gpt-4o",
                sub_model="gpt-4o-mini",
                root_provider="openai",
                sub_provider="openai",
            )
        elif provider == "anthropic":
            return cls(
                root_model=DEFAULT_ANTHROPIC_MODEL,
                sub_model=DEFAULT_ANTHROPIC_CHEAP_MODEL,
                root_provider="anthropic",
                sub_provider="anthropic",
            )
        elif provider == "mixed":
            # High-quality Claude for the root, local Llama for the subs.
            # This preset used Opus before the retirement fix and still
            # documents Opus; it keeps the high-quality tier rather than
            # dropping to the balanced default the plain `anthropic` preset
            # uses.
            return cls(
                root_model=DEFAULT_ANTHROPIC_HIGH_QUALITY_MODEL,
                sub_model="llama3",
                root_provider="anthropic",
                sub_provider="ollama",
            )
        else:
            raise ValueError(
                f"Unknown provider '{provider}'. Supported: 'openai', 'anthropic', 'mixed'"
            )

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"ModelConfig(root={self.root_provider or '?'}:{self.root_model}, "
            f"sub={self.sub_provider or '?'}:{self.sub_model})"
        )


@dataclass
class ModelMetrics:
    """
    Metrics broken down by model usage.

    Tracks separate statistics for root and sub-agent model usage,
    enabling cost analysis and optimization insights.

    Attributes:
        root_calls: Number of calls to root model
        root_tokens: Total tokens used by root model
        root_cost: Total cost for root model
        sub_calls: Number of calls to sub-agent model
        sub_tokens: Total tokens used by sub-agent model
        sub_cost: Total cost for sub-agent model
    """

    root_calls: int = 0
    root_tokens: int = 0
    root_cost: float = 0.0
    sub_calls: int = 0
    sub_tokens: int = 0
    sub_cost: float = 0.0

    @property
    def total_calls(self) -> int:
        """Total number of LLM calls."""
        return self.root_calls + self.sub_calls

    @property
    def total_tokens(self) -> int:
        """Total tokens across all models."""
        return self.root_tokens + self.sub_tokens

    @property
    def total_cost(self) -> float:
        """Total cost across all models."""
        return self.root_cost + self.sub_cost

    def savings_vs_single_model(self, single_model_cost_per_token: float) -> float:
        """
        Calculate cost savings vs. using a single model for everything.

        Args:
            single_model_cost_per_token: Cost per token if using only root model

        Returns:
            Savings amount (positive = saved money)
        """
        hypothetical_cost = self.total_tokens * single_model_cost_per_token
        return hypothetical_cost - self.total_cost

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "root_calls": self.root_calls,
            "root_tokens": self.root_tokens,
            "root_cost": self.root_cost,
            "sub_calls": self.sub_calls,
            "sub_tokens": self.sub_tokens,
            "sub_cost": self.sub_cost,
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
        }
