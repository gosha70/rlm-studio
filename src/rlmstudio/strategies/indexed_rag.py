# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""RAG strategy with persistent vector store — embeds content once, queries many times."""

import hashlib
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
from rlmstudio.storage.vector_store import VectorStore
from rlmstudio.tools.content import chunk as chunk_content

from .base import StrategyResult
from .embeddings import EmbeddingProvider


class IndexedRAGStrategy:
    """RAG with persistent vector store.

    Content is embedded and stored once. Subsequent queries only embed the
    query and search the index — no redundant document embedding.

    Satisfies the LLMStrategy protocol (name + run).
    """

    def __init__(
        self,
        client: LLMClient,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        collection: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 5,
        system_prompt: str | None = None,
    ):
        self.client = client
        self.embedder = embedder
        self.vector_store = vector_store
        self.collection = collection
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.system_prompt = system_prompt or get_mode_system_prompt(MODE_RAG)

    @property
    def name(self) -> str:
        return "rag"

    def index_content(self, content: str, source_id: str | None = None) -> int:
        """Index new content into the vector store.

        If source_id is given and already indexed, skips entirely.
        Returns the number of new chunks added.
        """
        if source_id is None:
            source_id = hashlib.sha256(content.encode()).hexdigest()

        if self.vector_store.has_source(self.collection, source_id):
            return 0

        chunks = chunk_content(content, size=self.chunk_size, overlap=self.chunk_overlap)
        if not chunks:
            return 0

        embeddings = self.embedder.embed(chunks)
        return int(
            self.vector_store.add_chunks(
                self.collection,
                chunks,
                embeddings,
                source_id=source_id,
            )
        )

    def run(self, content: str, query: str) -> StrategyResult:
        """Auto-index content if new, then query index and generate answer."""
        start = time.time()

        try:
            # Step 1: Index content (skips if already indexed)
            chunks_added = self.index_content(content)

            # Step 2: Embed query and search index
            query_embedding = self.embedder.embed_query(query)
            results = self.vector_store.search(
                self.collection,
                query_embedding,
                top_k=self.top_k,
            )

            if not results:
                return StrategyResult(
                    strategy=MODE_RAG,
                    answer="",
                    success=False,
                    error="No indexed chunks found for query",
                    elapsed_time=time.time() - start,
                )

            # Step 3: Build prompt with retrieved context
            context = self._assemble_context(results)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ]

            # Step 4: Generate answer
            answer = self.client.complete(messages)

        except Exception as e:
            return StrategyResult(
                strategy="rag",
                answer="",
                success=False,
                error=str(e),
                elapsed_time=time.time() - start,
            )

        elapsed = time.time() - start

        # Token accounting
        context_text = context + query + self.system_prompt
        tokens = TokenUsage()
        tokens.add_input(estimate_tokens(context_text))
        tokens.add_output(estimate_tokens(answer))

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
                    "chunks_retrieved": len(results),
                    "chunks_added": chunks_added,
                    "top_scores": [round(s, 4) for s, _, _ in results],
                },
                {
                    TRACE_KEY_STEP: 2,
                    TRACE_KEY_ROLE: "assistant",
                    TRACE_KEY_CONTENT: answer,
                    TRACE_KEY_MODE: "indexed_rag",
                },
            ],
            metadata={
                "collection": self.collection,
                "collection_size": self.vector_store.get_collection_size(self.collection),
                "chunks_retrieved": len(results),
                "chunks_added": chunks_added,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "top_k": self.top_k,
                "indexed": True,
            },
        )

    def _assemble_context(self, results: list) -> str:
        """Format retrieved chunks for the LLM prompt."""
        parts = []
        for i, (score, _chunk_id, text) in enumerate(results, 1):
            parts.append(f"[Chunk {i} (relevance: {score:.3f})]\n{text}")
        return "\n\n".join(parts)
