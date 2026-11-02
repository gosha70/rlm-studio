"""Domain value objects: immutable types with no external dependencies.

Value objects are distinguished from entities by having no identity --
two value objects with the same fields are considered equal.
All are frozen dataclasses for immutability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenCount:
    """Immutable token usage count.

    Attributes:
        input_tokens: Number of tokens sent to the LLM.
        output_tokens: Number of tokens received from the LLM.
    """

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        """Total tokens (input + output)."""
        return self.input_tokens + self.output_tokens

    def add(self, other: TokenCount) -> TokenCount:
        """Return a new TokenCount with summed values."""
        return TokenCount(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        """Serialize to a plain dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total": self.total,
        }


@dataclass(frozen=True)
class Cost:
    """Immutable monetary cost breakdown.

    Attributes:
        input_cost: Cost for input tokens in USD.
        output_cost: Cost for output tokens in USD.
    """

    input_cost: float = 0.0
    output_cost: float = 0.0

    @property
    def total(self) -> float:
        """Total cost (input + output)."""
        return self.input_cost + self.output_cost

    def add(self, other: Cost) -> Cost:
        """Return a new Cost with summed values."""
        return Cost(
            input_cost=self.input_cost + other.input_cost,
            output_cost=self.output_cost + other.output_cost,
        )

    def to_dict(self) -> dict[str, float]:
        """Serialize to a plain dictionary."""
        return {
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total": self.total,
        }


@dataclass(frozen=True)
class ModelId:
    """Identifies a specific model on a specific provider.

    Attributes:
        provider: Provider identifier (e.g. "openai", "anthropic").
        model_name: Model identifier (e.g. "gpt-4o", "claude-sonnet-4-6").
    """

    provider: str
    model_name: str

    @property
    def display_name(self) -> str:
        """Human-readable display string."""
        return f"{self.provider}/{self.model_name}"

    def __str__(self) -> str:
        return self.display_name


@dataclass(frozen=True)
class ProviderId:
    """Identifies an LLM provider.

    Attributes:
        name: Canonical provider name (e.g. "openai", "anthropic", "ollama").
    """

    name: str

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ProviderId):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass(frozen=True)
class RecursionDepth:
    """Tracks current and maximum recursion depth.

    Attributes:
        current: Current recursion depth (0 = root level).
        maximum: Maximum allowed depth.
    """

    current: int = 0
    maximum: int = 5

    @property
    def is_exceeded(self) -> bool:
        """Whether current depth exceeds the maximum."""
        return self.current >= self.maximum

    def descend(self) -> RecursionDepth:
        """Return a new RecursionDepth one level deeper."""
        return RecursionDepth(current=self.current + 1, maximum=self.maximum)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "current": self.current,
            "maximum": self.maximum,
            "is_exceeded": self.is_exceeded,
        }
