# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""Embedding provider abstraction for RAG strategy."""

import os
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Embed text into vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    """OpenAI embeddings via the openai library (already in optional deps)."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ):
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client: object | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError as e:
                raise ImportError(
                    "openai package is required for OpenAIEmbedder. "
                    "Install it with: pip install openai"
                ) from e

            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        response = client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]
