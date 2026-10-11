# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""LM Studio LLM provider implementation."""

from typing import Any

from .openai_client import OpenAIClient


class LMStudioClient(OpenAIClient):
    """
    LM Studio API client for local LLM deployment.

    LM Studio provides an OpenAI-compatible API for running local models
    with a user-friendly GUI. This client is a thin wrapper around OpenAIClient
    with LM Studio-specific defaults.

    Example:
        >>> # Start LM Studio and load a model
        >>> # Server starts at http://localhost:1234
        >>>
        >>> client = LMStudioClient(model="local-model")
        >>> response = client.complete([
        ...     {"role": "user", "content": "Hello!"}
        ... ])
        >>> print(response)
    """

    def __init__(
        self,
        model: str = "local-model",
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",  # LM Studio doesn't require real API key
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize LM Studio client.

        Args:
            model: Model identifier (use whatever you loaded in LM Studio)
            base_url: LM Studio server URL (default: http://localhost:1234/v1)
            api_key: Not used by LM Studio, but required by OpenAI client (default: "lm-studio")
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate (optional)
            **kwargs: Additional parameters passed to OpenAI API
        """
        # Initialize with LM Studio defaults
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for token usage.

        LM Studio runs locally, so cost is always zero.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Always returns 0.0 (local inference is free)
        """
        return 0.0
