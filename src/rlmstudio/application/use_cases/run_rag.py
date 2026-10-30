"""Use case: run a RAG (Retrieval-Augmented Generation) query.

Chunks the content, embeds it, retrieves the most relevant chunks
for the query, and sends them to the LLM as context.
"""

from __future__ import annotations

import time

from rlmstudio.application.dto import LLMResponseDTO, RunConfigDTO, RunResultDTO
from rlmstudio.application.ports.embedding_port import EmbeddingPort
from rlmstudio.application.ports.llm_port import LLMPort
from rlmstudio.application.ports.storage_port import StoragePort
from rlmstudio.application.sandbox_vars import (
    MODE_RAG,
    TRACE_KEY_CONTENT,
    TRACE_KEY_ELAPSED_SECONDS,
    TRACE_KEY_INPUT_TOKENS,
    TRACE_KEY_MODE,
    TRACE_KEY_MODEL,
    TRACE_KEY_OUTPUT_TOKENS,
    TRACE_KEY_ROLE,
    TRACE_KEY_STEP,
)
from rlmstudio.application.services.provider_failure import failure_kind_value
from rlmstudio.prompts import get_mode_system_prompt


def _humanize_rag_error(exc: Exception) -> str:
    """Return an actionable error message for common RAG failures."""
    msg = str(exc).lower()
    if "context" in msg and ("window" in msg or "length" in msg or "tokens" in msg):
        return (
            "RAG context window exceeded: the retrieved chunks are too large for this model. "
            "Reduce 'chunk_size' or 'top_k' in Settings → Modes → RAG, "
            "or switch to a model with a larger context window."
        )
    if "timeout" in msg or "timed out" in msg:
        return (
            "RAG timed out while embedding or querying. "
            "For large documents this can be slow on the first message. "
            "Try increasing the provider timeout in Settings, or reduce 'chunk_size'."
        )
    if "api key" in msg or "authentication" in msg or "unauthorized" in msg or "401" in msg:
        return (
            "RAG embedding failed: missing or invalid API key. "
            "RAG uses OpenAI embeddings by default — set OPENAI_API_KEY in your environment "
            "or configure an embedding provider in Settings."
        )
    if "max_tokens_per_request" in msg or "rate limit" in msg or "429" in msg:
        return (
            "RAG embedding rate-limited or exceeded tokens-per-request. "
            "Reduce 'chunk_size' in Settings → Modes → RAG to send smaller batches."
        )
    return str(exc)


def _format_rag_warning(exc: Exception, human_msg: str) -> str:
    """Format a ⚠️ warning card for RAG errors."""
    from rlmstudio.infrastructure.llm.litellm_adapter import TIMEOUT_ERROR_PREFIX

    exc_str = str(exc)
    exc_lower = exc_str.lower()
    if (
        exc_str.startswith(TIMEOUT_ERROR_PREFIX)
        or "timeout" in exc_lower
        or "timed out" in exc_lower
    ):
        return (
            "⚠️ **RAG request timed out**.\n\n"
            f"{human_msg}\n\n"
            "**How to fix** — in Settings → Profile → Runtime Settings:\n"
            "- Increase **Timeout (s)** (try 300 or higher for large documents).\n"
            "- Or reduce **chunk_size** in Settings → Modes → RAG."
        )
    if "context" in exc_str.lower() and "window" in exc_str.lower():
        return f"⚠️ **RAG context window exceeded**.\n\n{human_msg}"
    if "api key" in exc_str.lower() or "401" in exc_str:
        return f"⚠️ **RAG embedding authentication failed**.\n\n{human_msg}"
    return ""


class RunRAGUseCase:
    """Orchestrates a RAG pipeline through ports.

    Args:
        llm: LLM port adapter.
        embedder: Embedding port adapter.
        storage: Storage port adapter with vector search capability.
    """

    def __init__(
        self,
        llm: LLMPort,
        embedder: EmbeddingPort,
        storage: StoragePort,
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._storage = storage

    def execute(
        self,
        content: str,
        query: str,
        config: RunConfigDTO | None = None,
        skip_indexing: bool = False,
    ) -> RunResultDTO:
        """Run a RAG query: chunk, embed, retrieve, generate.

        Args:
            content: Document text to analyze.
            query: User question.
            config: Optional run configuration.

        Returns:
            RunResultDTO with the answer and metrics.
        """
        config = config or RunConfigDTO(mode="rag")
        start = time.time()
        top_k = config.extra.get("top_k", 5)
        chunk_size = config.extra.get("chunk_size", 1000)
        collection = config.extra.get("collection", "rag_temp")

        try:
            if not skip_indexing:
                # 1. Chunk the content
                chunks = self._chunk_text(content, chunk_size)

                # 2. Embed all chunks and store (may be slow for large docs)
                embeddings = self._embedder.embed_batch(chunks)

                # 3. Store chunks with embeddings
                self._storage.add_chunks(
                    collection=collection,
                    chunks=chunks,
                    embeddings=embeddings,
                )

            # 4. Embed the query and retrieve top-k
            query_embedding = self._embedder.embed(query)
            results = self._storage.search_chunks(
                collection=collection,
                query_embedding=query_embedding,
                top_k=top_k,
            )

            # 5. Build context from retrieved chunks
            context_chunks = [chunk_text for _, _, chunk_text in results]
            context = "\n\n---\n\n".join(context_chunks)

            # 6. Generate answer
            from rlmstudio.application.sandbox_vars import EXTRA_KEY_HISTORY_MESSAGES

            messages: list[dict[str, str]] = [
                {
                    "role": "system",
                    "content": get_mode_system_prompt("rag"),
                },
            ]
            history_msgs = config.extra.get(EXTRA_KEY_HISTORY_MESSAGES)
            if history_msgs:
                messages.extend(history_msgs)
            messages.append(
                {
                    "role": "user",
                    "content": (f"Context:\n{context}\n\nQuestion: {query}"),
                },
            )

            llm_start = time.time()
            response: LLMResponseDTO = self._llm.complete(messages)
            llm_elapsed = time.time() - llm_start
            elapsed = time.time() - start

            # Collect embedding token usage if the adapter exposes it
            embed_tokens = getattr(self._embedder, "total_tokens", 0)
            embed_cost = getattr(self._embedder, "total_cost", 0.0)

            # Compute LLM completion cost via the adapter's pricing table
            try:
                pricing = self._llm.get_pricing()
                llm_cost = (
                    response.input_tokens * float(pricing.get("input_cost_per_1m", 0.0))
                    + response.output_tokens * float(pricing.get("output_cost_per_1m", 0.0))
                ) / 1_000_000
            except Exception:
                llm_cost = 0.0

            # Two-step trace (§3a of the prefill/decode spec): retrieval
            # is step 0 (rag_retrieval → canonicalizes to `inspect`) and
            # the LLM completion is step 1 (assistant → canonicalizes to
            # `final` automatically because it's the last step on a
            # successful run). Splitting the phases keeps embedding
            # latency separate from LLM decode latency.
            return RunResultDTO(
                answer=response.content,
                mode_used=MODE_RAG,
                success=True,
                steps=1,
                input_tokens=response.input_tokens + embed_tokens,
                output_tokens=response.output_tokens,
                total_cost=embed_cost + llm_cost,
                elapsed_time=elapsed,
                trace=[
                    {
                        TRACE_KEY_STEP: 0,
                        TRACE_KEY_ROLE: "rag_retrieval",
                        "chunks_retrieved": len(results),
                        "scores": [score for score, _, _ in results],
                        "embedding_tokens": embed_tokens,
                        "embedding_cost": embed_cost,
                    },
                    {
                        TRACE_KEY_STEP: 1,
                        TRACE_KEY_ROLE: "assistant",
                        TRACE_KEY_CONTENT: response.content,
                        TRACE_KEY_MODE: MODE_RAG,
                        TRACE_KEY_INPUT_TOKENS: response.input_tokens,
                        TRACE_KEY_OUTPUT_TOKENS: response.output_tokens,
                        TRACE_KEY_ELAPSED_SECONDS: llm_elapsed,
                        TRACE_KEY_MODEL: (
                            getattr(self._llm, "active_model", None) or response.model
                        ),
                        "ttft_ms": response.ttft_ms,
                        "decode_ms": response.decode_ms,
                        "cached_tokens": response.cached_tokens,
                        "cache_write_tokens": response.cache_write_tokens,
                    },
                ],
                metadata={
                    "chunks_total": len(chunks) if not skip_indexing else None,
                    "chunks_retrieved": len(results),
                    "collection": collection,
                },
            )
        except Exception as exc:
            elapsed = time.time() - start
            error_msg = _humanize_rag_error(exc)
            answer = _format_rag_warning(exc, error_msg)
            return RunResultDTO(
                answer=answer,
                mode_used=MODE_RAG,
                success=False,
                error=error_msg,
                failure_kind=failure_kind_value(exc),
                elapsed_time=elapsed,
            )

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 1000) -> list[str]:
        """Split text into chunks of approximately chunk_size characters."""
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks
