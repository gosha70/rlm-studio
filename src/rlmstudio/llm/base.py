# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""Base classes for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    """Standard response format from LLM providers."""

    content: str
    """The generated text content"""

    model: str
    """Model used for generation"""

    input_tokens: int | None = None
    """Number of input tokens (if provided by API)"""

    output_tokens: int | None = None
    """Number of output tokens (if provided by API)"""

    finish_reason: str | None = None
    """Reason why generation stopped (e.g., 'stop', 'length', 'content_filter')"""

    metadata: dict[str, Any] | None = None
    """Additional provider-specific metadata"""


class BaseLLMProvider(ABC):
    """
    Base class for all LLM providers.

    Provides common interface and functionality for:
    - Cloud providers (OpenAI, Anthropic, Google)
    - Local providers (Ollama, vLLM)
    - Mock providers (for testing)
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize base provider.

        Args:
            model: Model identifier
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs

    @abstractmethod
    def complete(self, messages: list[dict[str, str]]) -> str:
        """
        Generate completion from messages.

        This is the main interface that RLM uses. Providers should implement
        this method to return just the text content.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            Generated text response

        Raises:
            Exception: Provider-specific errors
        """
        pass

    @abstractmethod
    def complete_with_metadata(self, messages: list[dict[str, str]]) -> LLMResponse:
        """
        Generate completion with full metadata.

        Providers should implement this to return detailed response info
        including token counts, finish reason, etc.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            LLMResponse with content and metadata

        Raises:
            Exception: Provider-specific errors
        """
        pass

    def validate_messages(self, messages: list[dict[str, str]]) -> None:
        """
        Validate message format.

        Args:
            messages: Messages to validate

        Raises:
            ValueError: If messages are invalid
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"Message {i} must be a dictionary")

            if "role" not in msg:
                raise ValueError(f"Message {i} missing 'role' key")

            if "content" not in msg:
                raise ValueError(f"Message {i} missing 'content' key")

            if msg["role"] not in ["system", "user", "assistant"]:
                raise ValueError(
                    f"Message {i} has invalid role: {msg['role']}. "
                    f"Must be 'system', 'user', or 'assistant'"
                )

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(model={self.model})"
