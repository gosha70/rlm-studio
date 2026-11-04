"""RAG settings and the embedder for one Chat Provider.

Chat and Compare both build a RAG embedder. They share this module so a
Chat Provider's own RAG settings, and a local embedding model, are honoured
the same way on every path.
"""

from __future__ import annotations

import os

from rlmstudio.infrastructure.embedding.litellm_embedding_adapter import LiteLLMEmbeddingAdapter
from rlmstudio.server.dependencies import AppState
from rlmstudio.server.models import ChatProviderConfig, RAGConfig
from rlmstudio.ui.data.providers_catalog import PROVIDERS_BY_KEY, request_api_key

# LiteLLM embedding prefix -> the provider catalog key of the local server it targets.
_LOCAL_EMBEDDING_BACKENDS: dict[str, str] = {
    "ollama/": "ollama",
    "lm_studio/": "lmstudio",
}


def rag_config_for(state: AppState, cp: ChatProviderConfig | None) -> RAGConfig:
    """The Chat Provider's own RAG settings when it has them, else the global ones."""
    if cp is not None and cp.rag_config is not None:
        return cp.rag_config
    return state.config.mode_config.rag_config


def resolve_embedding_api_key(state: AppState, cp: ChatProviderConfig | None) -> str | None:
    """Key for any embedding model that is not a local one (unchanged lookup).

    Preference order: ``OPENAI_API_KEY`` from the environment, then the stored
    key of the Chat Provider's LLM Provider when its backend is ``openai``.
    Non-OpenAI cloud embedding models get ``None`` here, so LiteLLM reads
    their own environment variable.
    """
    api_key: str | None = os.environ.get("OPENAI_API_KEY")
    if api_key or cp is None:
        return api_key
    lp = state.get_llm_provider(cp.llm_provider_id) if cp.llm_provider_id else None
    if lp and lp.backend == "openai":
        from rlmstudio.server.routes.llm_providers import _get_api_key

        api_key = _get_api_key(lp.id, lp.backend)
    return api_key


def build_rag_embedder(
    state: AppState, cp: ChatProviderConfig | None, rag_cfg: RAGConfig
) -> LiteLLMEmbeddingAdapter:
    """Embedder for ``rag_cfg.embedding_model``.

    A local model (``ollama/…``, ``lm_studio/…``) goes to the Chat Provider's
    own endpoint when that provider runs the same server, else to the server's
    default endpoint, and is never given the OpenAI key. Any other model keeps
    the OpenAI key lookup.
    """
    model = rag_cfg.embedding_model
    for prefix, backend in _LOCAL_EMBEDDING_BACKENDS.items():
        if not model.startswith(prefix):
            continue
        lp = state.get_llm_provider(cp.llm_provider_id) if cp and cp.llm_provider_id else None
        entry = PROVIDERS_BY_KEY.get(backend)
        api_base = lp.endpoint if lp is not None and lp.backend == backend else None
        if not api_base and entry is not None:
            api_base = entry.default_endpoint
        return LiteLLMEmbeddingAdapter(
            model=model,
            api_key=request_api_key(backend, None),
            api_base=api_base,
        )
    return LiteLLMEmbeddingAdapter(model=model, api_key=resolve_embedding_api_key(state, cp))
