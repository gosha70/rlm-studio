# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""vLLM LLM provider implementation for production GPU deployments."""

from typing import Any

from .openai_client import OpenAIClient


class vLLMClient(OpenAIClient):  # noqa: N801 — third-party naming convention
    """
    vLLM API client for high-performance GPU inference.

    vLLM provides an OpenAI-compatible API optimized for throughput on NVIDIA GPUs.
    Key features:
    - PagedAttention for efficient memory management
    - Continuous batching for parallel request processing
    - Up to 24x higher throughput than standard inference
    - Excellent for production deployments

    Example:
        >>> # Start vLLM server:
        >>> # python -m vllm.entrypoints.openai.api_server \
        >>> #   --model meta-llama/Llama-2-7b-hf \
        >>> #   --port 8000
        >>>
        >>> client = vLLMClient(model="meta-llama/Llama-2-7b-hf")
        >>> response = client.complete([
        ...     {"role": "user", "content": "Hello!"}
        ... ])
        >>> print(response)
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",  # vLLM doesn't require real API key
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ):
        """
        Initialize vLLM client.

        Args:
            model: Model identifier (must match model loaded in vLLM server)
            base_url: vLLM server URL (default: http://localhost:8000/v1)
            api_key: Not used by vLLM, but required by OpenAI client (default: "EMPTY")
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate (optional)
            **kwargs: Additional parameters passed to OpenAI API
                     (e.g., top_p, frequency_penalty, presence_penalty)
        """
        # Initialize with vLLM defaults
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

        vLLM runs locally or on your own infrastructure.
        Cost is zero by default, but you can override if running on paid cloud GPUs.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Always returns 0.0 by default (self-hosted is free)
        """
        # If running on cloud GPUs, override this method with actual costs
        return 0.0
