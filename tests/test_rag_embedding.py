"""RAG settings and embedder resolution for a Chat Provider.

Chat and Compare used the global RAG settings and always looked up the OpenAI
key first, so a Chat Provider's own embedding model was ignored: a local-only
setup either failed or sent document text to OpenAI embeddings.
"""

from __future__ import annotations

import pytest

from rlmstudio.server import rag_embedding
from rlmstudio.server.dependencies import AppState
from rlmstudio.server.models import ChatProviderConfig, LLMProviderConfig, RAGConfig
from rlmstudio.ui.data.providers_catalog import KEYLESS_API_KEY_PLACEHOLDER


def _state_with(lp: LLMProviderConfig | None = None) -> AppState:
    state = AppState(load_from_disk=False)
    state.config.mode_config.rag_config = RAGConfig(embedding_model="text-embedding-3-small")
    if lp is not None:
        state.config.llm_providers = [lp]
    return state


def _cp(rag: RAGConfig | None = None, lp_id: str = "") -> ChatProviderConfig:
    return ChatProviderConfig(id="cp", name="cp", llm_provider_id=lp_id, rag_config=rag)


def test_chat_provider_rag_settings_win_over_the_global_ones() -> None:
    own = RAGConfig(chunk_size=300, top_k=2, embedding_model="ollama/nomic-embed-text")
    state = _state_with()

    assert rag_embedding.rag_config_for(state, _cp(own)) is own
    assert rag_embedding.rag_config_for(state, _cp(None)) is state.config.mode_config.rag_config
    assert rag_embedding.rag_config_for(state, None) is state.config.mode_config.rag_config


def test_ollama_embeddings_use_the_providers_endpoint_and_never_the_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai")
    lp = LLMProviderConfig(
        id="lp", name="Ollama", backend="ollama", model="llama3.2", endpoint="http://gpu-box:11434"
    )
    state = _state_with(lp)

    embedder = rag_embedding.build_rag_embedder(
        state, _cp(lp_id="lp"), RAGConfig(embedding_model="ollama/nomic-embed-text")
    )

    assert embedder._api_key is None
    assert embedder._api_base == "http://gpu-box:11434"


def test_lm_studio_embeddings_get_the_placeholder_key_and_default_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai")
    lp = LLMProviderConfig(id="lp", name="Ollama", backend="ollama", model="llama3.2")
    state = _state_with(lp)

    embedder = rag_embedding.build_rag_embedder(
        state,
        _cp(lp_id="lp"),
        RAGConfig(embedding_model="lm_studio/text-embedding-nomic-embed-text-v1.5"),
    )

    assert embedder._api_key == KEYLESS_API_KEY_PLACEHOLDER
    assert embedder._api_base == "http://localhost:1234/v1"


@pytest.mark.parametrize("model", ["text-embedding-3-small", "openai/text-embedding-3-small"])
def test_openai_embeddings_keep_the_openai_key_lookup(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai")
    state = _state_with()

    embedder = rag_embedding.build_rag_embedder(state, _cp(), RAGConfig(embedding_model=model))

    assert embedder._api_key == "sk-real-openai"
    assert embedder._api_base is None


@pytest.mark.parametrize(
    "model", ["cohere/embed-english-v3.0", "gemini/text-embedding-004", "mistral/mistral-embed"]
)
def test_other_cloud_embeddings_never_get_the_openai_key(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    """The embedding model is free text: a Cohere, Gemini or Mistral model
    must not be handed the user's OpenAI key."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai")
    state = _state_with()

    embedder = rag_embedding.build_rag_embedder(state, _cp(), RAGConfig(embedding_model=model))

    assert embedder._api_key is None
