# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""RAG strategy — retrieve relevant chunks then generate an answer."""

import math
import time

from rlmstudio.application.sandbox_vars import (
    MODE_RAG,
    TRACE_KEY_CONTENT,
    TRACE_KEY_MODE,
    TRACE_KEY_ROLE,
    TRACE_KEY_STEP,
)
from rlmstudio.core.budget import TokenUsage, estimate_tokens
from rlmstudio.core.rlm import LLMClient
from rlmstudio.prompts import get_mode_system_prompt
from rlmstudio.tools.content import chunk as chunk_content

from .base import StrategyResult
from .embeddings import EmbeddingProvider


class RAGStrategy:
    """Retrieve-then-read: embed chunks, rank by similarity, generate."""

    def __init__(
        self,
        client: LLMClient,
        embedder: EmbeddingProvider,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 5,
        system_prompt: str | None = None,
    ):
        self.client = client
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.system_prompt = system_prompt or get_mode_system_prompt(MODE_RAG)

    @property
    def name(self) -> str:
        return "rag"

    def run(self, content: str, query: str) -> StrategyResult:
        start = time.time()

        try:
            # Step 1: Chunk
            chunks = self._chunk_content(content)
            if not chunks:
                return StrategyResult(
                    strategy=MODE_RAG,
                    answer="",
                    success=False,
                    error="No chunks produced from content",
                    elapsed_time=time.time() - start,
                )

            # Step 2: Embed chunks + query
            chunk_embeddings = self.embedder.embed(chunks)
            query_embedding = self.embedder.embed_query(query)

            # Step 3: Rank and select top-k
            ranked = self._rank_chunks(query_embedding, chunk_embeddings, chunks)
            top_chunks = ranked[: self.top_k]

            # Step 4: Build prompt with retrieved context
            context = self._assemble_context(top_chunks)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ]

            # Step 5: Generate answer
            answer = self.client.complete(messages)

        except Exception as e:
            return StrategyResult(
                strategy=MODE_RAG,
                answer="",
                success=False,
                error=str(e),
                elapsed_time=time.time() - start,
            )

        elapsed = time.time() - start

        # Token accounting (generation only — embedding tokens tracked in metadata)
        context_text = context + query + self.system_prompt
        tokens = TokenUsage()
        tokens.add_input(estimate_tokens(context_text))
        tokens.add_output(estimate_tokens(answer))

        embedding_input = " ".join(chunks) + " " + query
        embedding_tokens = estimate_tokens(embedding_input)

        return StrategyResult(
            strategy=MODE_RAG,
            answer=answer,
            steps=1,
            tokens=tokens,
            elapsed_time=elapsed,
            trace=[
                {
                    TRACE_KEY_STEP: 1,
                    TRACE_KEY_ROLE: "retrieval",
                    "chunks_total": len(chunks),
                    "chunks_retrieved": len(top_chunks),
                    "top_scores": [round(s, 4) for s, _ in top_chunks],
                },
                {
                    TRACE_KEY_STEP: 2,
                    TRACE_KEY_ROLE: "assistant",
                    TRACE_KEY_CONTENT: answer,
                    TRACE_KEY_MODE: MODE_RAG,
                },
            ],
            metadata={
                "chunks_total": len(chunks),
                "chunks_retrieved": len(top_chunks),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "top_k": self.top_k,
                "embedding_tokens": embedding_tokens,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk_content(self, content: str) -> list[str]:
        return list(
            chunk_content(
                content,
                size=self.chunk_size,
                overlap=self.chunk_overlap,
            )
        )

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _rank_chunks(
        self,
        query_emb: list[float],
        chunk_embs: list[list[float]],
        chunks: list[str],
    ) -> list[tuple[float, str]]:
        scored = [
            (self._cosine_similarity(query_emb, emb), text)
            for emb, text in zip(chunk_embs, chunks, strict=False)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def _assemble_context(self, ranked_chunks: list[tuple[float, str]]) -> str:
        parts = []
        for i, (score, text) in enumerate(ranked_chunks, 1):
            parts.append(f"[Chunk {i} (relevance: {score:.3f})]\n{text}")
        return "\n\n".join(parts)
