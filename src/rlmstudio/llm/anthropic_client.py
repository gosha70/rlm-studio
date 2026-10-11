# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""Anthropic Claude LLM provider implementation."""

import os
from typing import Any

from .base import BaseLLMProvider, LLMResponse


class ClaudeClient(BaseLLMProvider):
    """
    Anthropic Claude API client for RLM.

    Supports Claude 3 (Opus, Sonnet, Haiku) and Claude 2 models.

    Example:
        >>> import os
        >>> os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-...'
        >>>
        >>> client = ClaudeClient(model="claude-3-opus-20240229")
        >>> response = client.complete([
        ...     {"role": "user", "content": "Hello!"}
        ... ])
        >>> print(response)
    """

    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        """
        Initialize Claude client.

        Args:
            model: Model name (e.g., 'claude-3-opus-20240229', 'claude-3-sonnet-20240229')
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            base_url: Custom API base URL (optional)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate (required for Claude)
            **kwargs: Additional parameters passed to Anthropic API
        """
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)

        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Provide via api_key parameter "
                "or ANTHROPIC_API_KEY environment variable."
            )

        self.base_url = base_url

        # Claude requires max_tokens
        if not self.max_tokens:
            self.max_tokens = 4096

        # Lazy import anthropic to avoid requiring it if not used
        try:
            import anthropic

            self._anthropic = anthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic package not installed. Install with: pip install anthropic"
            ) from exc

        # Initialize Anthropic client
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self._client = self._anthropic.Anthropic(**client_kwargs)  # type: ignore[arg-type]

    def _convert_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[str | None, list[dict[str, str]]]:
        """
        Convert messages to Claude format.

        Claude expects:
        - System message separate from conversation
        - Alternating user/assistant messages

        Args:
            messages: Standard message format

        Returns:
            Tuple of (system_message, converted_messages)
        """
        system_message = None
        converted = []

        for msg in messages:
            if msg["role"] == "system":
                # Extract system message
                system_message = msg["content"]
            else:
                # Keep user and assistant messages
                converted.append({"role": msg["role"], "content": msg["content"]})

        return system_message, converted

    def complete(self, messages: list[dict[str, str]]) -> str:
        """
        Generate completion from messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            Generated text response

        Raises:
            ValueError: If messages are invalid
            anthropic.APIError: If API call fails
        """
        self.validate_messages(messages)

        # Convert messages to Claude format
        system_message, converted_messages = self._convert_messages(messages)

        # Build API request parameters
        params = {
            "model": self.model,
            "messages": converted_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if system_message:
            params["system"] = system_message

        # Add extra parameters
        params.update(self.extra_params)

        # Make API call
        try:
            response = self._client.messages.create(**params)  # type: ignore[call-overload]
            # Claude returns content as a list of blocks
            return str(response.content[0].text)
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}") from e

    def complete_with_metadata(self, messages: list[dict[str, str]]) -> LLMResponse:
        """
        Generate completion with full metadata.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            LLMResponse with content and metadata

        Raises:
            ValueError: If messages are invalid
            anthropic.APIError: If API call fails
        """
        self.validate_messages(messages)

        # Convert messages to Claude format
        system_message, converted_messages = self._convert_messages(messages)

        # Build API request parameters
        params = {
            "model": self.model,
            "messages": converted_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if system_message:
            params["system"] = system_message

        # Add extra parameters
        params.update(self.extra_params)

        # Make API call
        try:
            response = self._client.messages.create(**params)  # type: ignore[call-overload]
            usage = response.usage

            return LLMResponse(
                content=response.content[0].text,
                model=response.model,
                input_tokens=usage.input_tokens if usage else None,
                output_tokens=usage.output_tokens if usage else None,
                finish_reason=response.stop_reason,
                metadata={
                    "id": response.id,
                    "type": response.type,
                    "role": response.role,
                },
            )
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}") from e

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for token usage.

        Uses approximate pricing as of 2024. May need updates.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        # Pricing per 1M tokens (approximate, update as needed)
        pricing = {
            "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
            "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
            "claude-2.1": {"input": 8.0, "output": 24.0},
            "claude-2.0": {"input": 8.0, "output": 24.0},
        }

        # Get pricing for model (or use opus as default)
        model_pricing = pricing.get(self.model, pricing["claude-3-opus-20240229"])

        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]

        return input_cost + output_cost
