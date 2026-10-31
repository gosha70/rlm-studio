# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""Anthropic Claude LLM provider implementation."""

import logging
import os
from typing import Any

from rlmstudio.domain.model_catalog import DEFAULT_ANTHROPIC_MODEL, price_per_1k

from .base import BaseLLMProvider, LLMResponse

logger = logging.getLogger("rlmstudio.llm.anthropic")


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
        model: str = DEFAULT_ANTHROPIC_MODEL,
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
        # Priced from the shared catalog rather than a local table. The
        # local one knew only Claude 2 and 3, and fell back to Opus 3 for
        # anything else — so the current default, Sonnet 4.6, was billed at
        # Opus 3 rates: $90 for 1M in + 1M out against a real $18. Any model
        # newer than the table was overcharged roughly fivefold, silently.
        prices = price_per_1k(self.model)
        if prices is None:
            # No fallback. Substituting another model's rate is what produced
            # the error above, and a confidently wrong number is worse than
            # an absent one — callers treat 0.0 as "not priced".
            logger.warning(
                "No price known for model %s; reporting 0.0 rather than "
                "guessing from another model's rate.",
                self.model,
            )
            return 0.0

        # Annotated because the package ships no py.typed (#98), so a
        # cross-module import resolves to Any and warn_return_any fires.
        input_per_1k, output_per_1k = prices
        total: float = (input_tokens / 1000) * input_per_1k + (output_tokens / 1000) * output_per_1k
        return total
