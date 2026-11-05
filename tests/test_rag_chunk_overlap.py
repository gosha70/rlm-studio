"""CHAT2-05 (part): the RAG Chunk Overlap setting reaches the run.

Only ``chunk_size`` and ``top_k`` were put into ``RunConfigDTO.extra`` by
Chat (REST and WebSocket) and Compare, and ``RunRAGUseCase`` cut the
document into back-to-back chunks, so the saved overlap had no effect.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Generator
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rlmstudio.application.dto import LLMResponseDTO, RunConfigDTO, RunResultDTO
from rlmstudio.application.sandbox_vars import EXTRA_KEY_RAG_CHUNK_OVERLAP, MODE_RAG
from rlmstudio.application.use_cases.run_rag import RunRAGUseCase
from rlmstudio.server.app import app
from rlmstudio.server.dependencies import SessionRecord, get_state, reset_state
from rlmstudio.server.models import (
    ChatProviderConfig,
    LLMProviderConfig,
    RAGConfig,
    RuntimeSettings,
)

_CHUNK_SIZE = 100
_CHUNK_OVERLAP = 30
_RAG = RAGConfig(chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP, top_k=2)
_WAIT_SECONDS = 5.0
_POLL_SECONDS = 0.01


# --- use case ----------------------------------------------------------------


class _FakeLLM:
    active_model = "fake"

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        return LLMResponseDTO(content="answer", model="fake", input_tokens=1, output_tokens=1)

    def get_pricing(self) -> dict[str, float]:
        return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}


class _FakeEmbedder:
    total_tokens = 0
    total_cost = 0.0
    dimension = 2

    def embed(self, text: str) -> list[float]:
        return [0.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0] for _ in texts]


class _RecordingStorage:
    def __init__(self) -> None:
        self.chunks: list[str] = []

    def add_chunks(self, *, collection: str, chunks: list[str], embeddings: Any) -> None:
        self.chunks = chunks

    def search_chunks(self, *, collection: str, query_embedding: Any, top_k: int) -> list[Any]:
        return [(0.9, "id-0", self.chunks[0])]


def test_chunks_share_the_overlap() -> None:
    # Review P2: no trailing "ij" — it lay wholly inside "ghij".
    assert RunRAGUseCase._chunk_text("abcdefghij", 4, 2) == ["abcd", "cdef", "efgh", "ghij"]


@pytest.mark.parametrize(
    ("text", "size", "overlap", "expected"),
    [
        ("abcd", 4, 2, ["abcd"]),  # exactly one chunk long: no duplicate tail
        ("abcd", 4, 0, ["abcd"]),
        ("abcdef", 4, 2, ["abcd", "cdef"]),  # last chunk ends exactly at EOF
        ("abcdefg", 4, 2, ["abcd", "cdef", "efg"]),  # a short final chunk is kept
        ("", 4, 2, []),
    ],
)
def test_chunking_stops_at_the_end_of_the_text(
    text: str, size: int, overlap: int, expected: list[str]
) -> None:
    assert RunRAGUseCase._chunk_text(text, size, overlap) == expected


def test_no_overlap_keeps_back_to_back_chunks() -> None:
    assert RunRAGUseCase._chunk_text("abcdefghij", 4) == ["abcd", "efgh", "ij"]


def test_an_overlap_not_below_the_chunk_size_is_ignored() -> None:
    assert RunRAGUseCase._chunk_text("abcdefghij", 4, 4) == ["abcd", "efgh", "ij"]


def test_the_run_chunks_with_the_configured_overlap() -> None:
    storage = _RecordingStorage()
    document = "x" * 250
    config = RunConfigDTO(
        mode=MODE_RAG,
        extra={"chunk_size": _CHUNK_SIZE, EXTRA_KEY_RAG_CHUNK_OVERLAP: _CHUNK_OVERLAP},
    )

    result = RunRAGUseCase(_FakeLLM(), _FakeEmbedder(), storage).execute(document, "q", config)

    assert result.success
    # Steps of 70 over 250 characters: starts at 0, 70, 140, 210.
    assert [len(c) for c in storage.chunks] == [100, 100, 100, 40]


# --- routes ------------------------------------------------------------------


class _RecordingRAG:
    """Stands in for RunRAGUseCase in a route; records the config it was given."""

    configs: list[RunConfigDTO] = []

    def __init__(self, *_: Any) -> None:
        pass

    def execute(self, content: str, query: str, config: RunConfigDTO, *_: Any) -> RunResultDTO:
        type(self).configs.append(config)
        return RunResultDTO(answer="rag answer", mode_used=MODE_RAG, success=True)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    reset_state()
    _RecordingRAG.configs = []
    monkeypatch.setattr("rlmstudio.server.routes.chat.RunRAGUseCase", _RecordingRAG)
    monkeypatch.setattr("rlmstudio.server.routes.chat.build_rag_embedder", lambda *a: None)
    monkeypatch.setattr(
        "rlmstudio.server.routes.compare_matrix.build_rag_embedder", lambda *a: _FakeEmbedder()
    )
    yield
    reset_state()


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as test_client:
        yield test_client


def _rag_chat_provider() -> ChatProviderConfig:
    state = get_state()
    cp = ChatProviderConfig(
        id=str(uuid.uuid4()),
        name="RAG CP",
        llm_provider="ollama",
        llm_model="llama3.2",
        execution_mode=MODE_RAG,
        runtime_settings=RuntimeSettings(),
        rag_config=_RAG,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    state.config.chat_providers.append(cp)
    state.create_llm_adapter_for_chat_provider = (  # type: ignore[method-assign]
        lambda cp_id, num_retries=None: _FakeLLM()
    )
    return cp


def _await_rag_config() -> RunConfigDTO:
    deadline = time.monotonic() + _WAIT_SECONDS
    while time.monotonic() < deadline:
        if _RecordingRAG.configs:
            return _RecordingRAG.configs[0]
        time.sleep(_POLL_SECONDS)
    raise AssertionError("the RAG run was not started")


def test_rest_chat_passes_the_overlap(client: TestClient) -> None:
    cp = _rag_chat_provider()

    resp = client.post(
        "/api/chat", json={"query": "q", "content": "doc", "chat_provider_id": cp.id}
    )

    assert resp.status_code == 202, resp.text
    assert _await_rag_config().extra[EXTRA_KEY_RAG_CHUNK_OVERLAP] == _CHUNK_OVERLAP


def test_websocket_chat_passes_the_overlap(client: TestClient) -> None:
    cp = _rag_chat_provider()
    now = datetime.now(timezone.utc)
    get_state().sessions["s"] = SessionRecord(id="s", name="S", created_at=now, updated_at=now)

    with client.websocket_connect("/ws/chat/s") as ws:
        assert ws.receive_json()["type"] == "connected"
        ws.send_json(
            {"type": "query", "id": "q1", "query": "q", "content": "doc", "chat_provider_id": cp.id}
        )
        config = _await_rag_config()

    assert config.extra[EXTRA_KEY_RAG_CHUNK_OVERLAP] == _CHUNK_OVERLAP


def _spy_on_slot_configs(monkeypatch: pytest.MonkeyPatch) -> list[RunConfigDTO]:
    from rlmstudio.application.use_cases import run_matrix_comparison as uc_mod

    captured: list[RunConfigDTO] = []
    real_execute_slot = uc_mod.RunMatrixComparisonUseCase._execute_slot

    def _spy(self: Any, slot: Any, content: str, query: str, base_config: Any) -> Any:
        captured.append(slot.config)
        slot.mode = "direct"  # no real RAG run needed
        return real_execute_slot(self, slot, content, query, base_config)

    monkeypatch.setattr(uc_mod.RunMatrixComparisonUseCase, "_execute_slot", _spy)
    return captured


def test_compare_with_chat_providers_passes_the_overlap(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    cp = _rag_chat_provider()
    captured = _spy_on_slot_configs(monkeypatch)

    resp = client.post(
        "/api/chat/compare-matrix",
        json={"query": "q", "content": "doc", "chat_provider_ids": [cp.id], "modes": ["rag"]},
    )

    assert resp.status_code == 200, resp.text
    assert captured[0].extra[EXTRA_KEY_RAG_CHUNK_OVERLAP] == _CHUNK_OVERLAP


def test_llm_tuner_passes_the_overlap(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    state = get_state()
    lp = LLMProviderConfig(
        id=str(uuid.uuid4()),
        name="Local",
        backend="ollama",
        model="llama3.2",
        runtime_settings=RuntimeSettings(),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    state.config.llm_providers.append(lp)
    state.create_llm_adapter_for_chat_provider = (  # type: ignore[method-assign]
        lambda cp_id, num_retries=None: _FakeLLM()
    )
    captured = _spy_on_slot_configs(monkeypatch)

    resp = client.post(
        "/api/chat/compare-matrix",
        json={
            "query": "q",
            "content": "doc",
            "llm_provider_ids": [lp.id],
            "modes": ["rag"],
            "rag_config": _RAG.model_dump(),
        },
    )

    assert resp.status_code == 200, resp.text
    assert captured[0].extra[EXTRA_KEY_RAG_CHUNK_OVERLAP] == _CHUNK_OVERLAP
