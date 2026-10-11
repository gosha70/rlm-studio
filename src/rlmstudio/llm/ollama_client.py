# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""Ollama LLM provider implementation for local models."""

import json
from typing import Any

import requests  # type: ignore[import-untyped]

from .base import BaseLLMProvider, LLMResponse


class OllamaClient(BaseLLMProvider):
    """
    Ollama API client for local LLM deployment.

    Ollama provides an easy way to run LLMs locally (Llama 2, Mistral, etc.)
    with a simple HTTP API. Perfect for development and users without cloud APIs.

    Example:
        >>> # Start Ollama: ollama serve
        >>> # Pull a model: ollama pull llama2
        >>>
        >>> client = OllamaClient(model="llama2")
        >>> response = client.complete([
        ...     {"role": "user", "content": "Hello!"}
        ... ])
        >>> print(response)
    """

    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ):
        """
        Initialize Ollama client.

        Args:
            model: Model name (e.g., 'llama2', 'mistral', 'codellama')
            base_url: Ollama server URL (default: http://localhost:11434)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate (optional)
            **kwargs: Additional parameters passed to Ollama API
        """
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)

        self.base_url = base_url.rstrip("/")
        self.chat_endpoint = f"{self.base_url}/api/chat"
        self.generate_endpoint = f"{self.base_url}/api/generate"

        # Check if Ollama is running
        self._check_connection()

    def _check_connection(self) -> None:
        """Check if Ollama server is accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running: 'ollama serve'. "
                f"Error: {str(e)}"
            ) from e

    def complete(self, messages: list[dict[str, str]]) -> str:
        """
        Generate completion from messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            Generated text response

        Raises:
            ValueError: If messages are invalid
            ConnectionError: If Ollama server is not accessible
            RuntimeError: If API call fails
        """
        self.validate_messages(messages)

        # Build API request
        options: dict[str, Any] = {"temperature": self.temperature}
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,  # Get complete response
            "options": options,
        }

        if self.max_tokens:
            options["num_predict"] = self.max_tokens

        # Add extra parameters
        if self.extra_params:
            options.update(self.extra_params)

        # Make API call
        try:
            response = requests.post(
                self.chat_endpoint,
                json=payload,
                timeout=120,  # Longer timeout for local inference
            )
            response.raise_for_status()

            data = response.json()
            return str(data["message"]["content"])

        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"Ollama request timed out. Model '{self.model}' may be slow or overloaded."
            ) from exc
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API error: {str(e)}") from e
        except (KeyError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Invalid Ollama response format: {str(e)}") from e

    def complete_with_metadata(self, messages: list[dict[str, str]]) -> LLMResponse:
        """
        Generate completion with full metadata.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            LLMResponse with content and metadata

        Raises:
            ValueError: If messages are invalid
            ConnectionError: If Ollama server is not accessible
            RuntimeError: If API call fails
        """
        self.validate_messages(messages)

        # Build API request
        options2: dict[str, Any] = {"temperature": self.temperature}
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options2,
        }

        if self.max_tokens:
            options2["num_predict"] = self.max_tokens

        # Add extra parameters
        if self.extra_params:
            options2.update(self.extra_params)

        # Make API call
        try:
            response = requests.post(self.chat_endpoint, json=payload, timeout=120)
            response.raise_for_status()

            data = response.json()

            # Extract token counts if available
            input_tokens = data.get("prompt_eval_count")
            output_tokens = data.get("eval_count")

            return LLMResponse(
                content=data["message"]["content"],
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=data.get("done_reason", "stop"),
                metadata={
                    "total_duration": data.get("total_duration"),
                    "load_duration": data.get("load_duration"),
                    "prompt_eval_duration": data.get("prompt_eval_duration"),
                    "eval_duration": data.get("eval_duration"),
                },
            )

        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"Ollama request timed out. Model '{self.model}' may be slow or overloaded."
            ) from exc
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API error: {str(e)}") from e
        except (KeyError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Invalid Ollama response format: {str(e)}") from e

    def list_models(self) -> list[str]:
        """
        List available models on the Ollama server.

        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()

            data = response.json()
            return [model["name"] for model in data.get("models", [])]

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to list Ollama models: {str(e)}") from e

    def pull_model(self, model: str) -> None:
        """
        Pull a model from Ollama library.

        Note: This is a blocking operation that may take a while.

        Args:
            model: Model name to pull (e.g., 'llama2', 'mistral')
        """
        payload = {"name": model, "stream": False}

        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=3600,  # 1 hour for large downloads
            )
            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to pull model '{model}': {str(e)}") from e
