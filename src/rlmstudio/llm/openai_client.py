# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""OpenAI LLM provider implementation."""

import os
import re
from typing import Any

from .base import BaseLLMProvider, LLMResponse


class OpenAIClient(BaseLLMProvider):
    """
    OpenAI API client for RLM.

    Supports GPT-4, GPT-3.5-turbo, and other OpenAI models.

    Example:
        >>> import os
        >>> os.environ['OPENAI_API_KEY'] = 'sk-...'
        >>>
        >>> client = OpenAIClient(model="gpt-4")
        >>> response = client.complete([
        ...     {"role": "user", "content": "Hello!"}
        ... ])
        >>> print(response)
    """

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ):
        """
        Initialize OpenAI client.

        Args:
            model: Model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            organization: OpenAI organization ID (optional)
            base_url: Custom API base URL (for proxies/Azure)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters passed to OpenAI API
        """
        self._context_token_reserve = int(kwargs.pop("context_token_reserve", 128))
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
        self._discovered_context_limit_tokens: int | None = None

        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Provide via api_key parameter "
                "or OPENAI_API_KEY environment variable."
            )

        self.organization = organization or os.getenv("OPENAI_ORGANIZATION")
        self.base_url = base_url

        # Lazy import openai to avoid requiring it if not used
        try:
            import openai

            self._openai = openai
        except ImportError as e:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            ) from e

        # Initialize OpenAI client
        self._client = self._openai.OpenAI(
            api_key=self.api_key,
            organization=self.organization,
            base_url=self.base_url,
        )

    def _build_params(
        self,
        messages: list[dict[str, str]],
        max_tokens_override: int | None = None,
    ) -> dict[str, Any]:
        """Build API request parameters with context-aware max-token clamping."""
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        requested_max_tokens = (
            self.max_tokens if max_tokens_override is None else max_tokens_override
        )
        clamped_max_tokens = self._context_aware_max_tokens(messages, requested_max_tokens)
        if clamped_max_tokens is not None:
            params["max_tokens"] = clamped_max_tokens

        params.update(self.extra_params)
        return params

    def _estimate_message_tokens(self, messages: list[dict[str, str]]) -> int:
        """Estimate prompt token usage for a list of chat messages."""
        total = 0
        for msg in messages:
            total += 4  # rough chat-message framing overhead
            total += self.estimate_tokens(str(msg.get("content", "")))
        return total + 2  # assistant priming

    def _context_aware_max_tokens(
        self,
        messages: list[dict[str, str]],
        requested_max_tokens: int | None,
    ) -> int | None:
        """Clamp output tokens when the model context limit is known."""
        if requested_max_tokens is None or self._discovered_context_limit_tokens is None:
            return requested_max_tokens

        prompt_tokens = self._estimate_message_tokens(messages)
        available = (
            self._discovered_context_limit_tokens - prompt_tokens - self._context_token_reserve
        )
        if available <= 0:
            return 1
        return max(1, min(requested_max_tokens, available))

    def _derive_retry_max_tokens(
        self,
        messages: list[dict[str, str]],
        error_text: str,
        requested_max_tokens: int | None,
    ) -> int | None:
        """Return a smaller max_tokens when an error reports context overflow."""
        max_context: int | None = None
        prompt_tokens: int | None = None

        ctx_match = re.search(r"maximum context length is\s+(\d+)\s+tokens", error_text, re.I)
        if ctx_match:
            max_context = int(ctx_match.group(1))
            self._discovered_context_limit_tokens = max_context

        prompt_match = re.search(
            r"prompt contains(?: at least)?\s+(\d+)\s+input tokens",
            error_text,
            re.I,
        )
        if prompt_match:
            prompt_tokens = int(prompt_match.group(1))

        if prompt_tokens is None and max_context is not None:
            prompt_tokens = self._estimate_message_tokens(messages)

        if max_context is None or prompt_tokens is None:
            return None

        available = max_context - prompt_tokens - self._context_token_reserve
        if available <= 0:
            return None

        if requested_max_tokens is None:
            return available
        return max(1, min(requested_max_tokens, available))

    def complete(self, messages: list[dict[str, str]]) -> str:
        """
        Generate completion from messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            Generated text response

        Raises:
            ValueError: If messages are invalid
            openai.OpenAIError: If API call fails
        """
        self.validate_messages(messages)

        params = self._build_params(messages)

        # Make API call
        try:
            response = self._client.chat.completions.create(**params)  # type: ignore[call-overload]
            return str(response.choices[0].message.content)
        except Exception as e:
            retry_max_tokens = self._derive_retry_max_tokens(
                messages,
                str(e),
                params.get("max_tokens"),
            )
            if retry_max_tokens is not None and retry_max_tokens != params.get("max_tokens"):
                retry_params = self._build_params(messages, max_tokens_override=retry_max_tokens)
                try:
                    response = self._client.chat.completions.create(**retry_params)  # type: ignore[call-overload]
                    return str(response.choices[0].message.content)
                except Exception as retry_exc:
                    raise RuntimeError(f"OpenAI API error: {str(retry_exc)}") from retry_exc
            raise RuntimeError(f"OpenAI API error: {str(e)}") from e

    def complete_with_metadata(self, messages: list[dict[str, str]]) -> LLMResponse:
        """
        Generate completion with full metadata.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            LLMResponse with content and metadata

        Raises:
            ValueError: If messages are invalid
            openai.OpenAIError: If API call fails
        """
        self.validate_messages(messages)

        params = self._build_params(messages)

        # Make API call
        try:
            response = self._client.chat.completions.create(**params)  # type: ignore[call-overload]
            choice = response.choices[0]
            usage = response.usage

            return LLMResponse(
                content=choice.message.content,
                model=response.model,
                input_tokens=usage.prompt_tokens if usage else None,
                output_tokens=usage.completion_tokens if usage else None,
                finish_reason=choice.finish_reason,
                metadata={
                    "id": response.id,
                    "created": response.created,
                    "system_fingerprint": getattr(response, "system_fingerprint", None),
                },
            )
        except Exception as e:
            retry_max_tokens = self._derive_retry_max_tokens(
                messages,
                str(e),
                params.get("max_tokens"),
            )
            if retry_max_tokens is not None and retry_max_tokens != params.get("max_tokens"):
                retry_params = self._build_params(messages, max_tokens_override=retry_max_tokens)
                try:
                    response = self._client.chat.completions.create(**retry_params)  # type: ignore[call-overload]
                    choice = response.choices[0]
                    usage = response.usage
                    return LLMResponse(
                        content=choice.message.content,
                        model=response.model,
                        input_tokens=usage.prompt_tokens if usage else None,
                        output_tokens=usage.completion_tokens if usage else None,
                        finish_reason=choice.finish_reason,
                        metadata={
                            "id": response.id,
                            "created": response.created,
                            "system_fingerprint": getattr(response, "system_fingerprint", None),
                        },
                    )
                except Exception as retry_exc:
                    raise RuntimeError(f"OpenAI API error: {str(retry_exc)}") from retry_exc
            raise RuntimeError(f"OpenAI API error: {str(e)}") from e

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using tiktoken.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        try:
            import tiktoken

            # Get encoding for model
            try:
                encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # Fallback to cl100k_base (used by gpt-4, gpt-3.5-turbo)
                encoding = tiktoken.get_encoding("cl100k_base")

            return len(encoding.encode(text))
        except ImportError:
            # Fallback to simple estimation if tiktoken not available
            return len(text) // 4

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
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            "gpt-4-turbo-preview": {"input": 10.0, "output": 30.0},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
            "gpt-3.5-turbo-16k": {"input": 3.0, "output": 4.0},
        }

        # Get pricing for model (or use gpt-4 as default)
        model_pricing = pricing.get(self.model, pricing["gpt-4"])

        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]

        return input_cost + output_cost
