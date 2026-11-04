"""Tests for POST /api/chat/compare-matrix."""

from __future__ import annotations

import uuid
from collections.abc import Generator, Iterator
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rlmstudio.application.dto import LLMResponseDTO
from rlmstudio.infrastructure.embedding.litellm_embedding_adapter import (
    LiteLLMEmbeddingAdapter,
)
from rlmstudio.server.app import app
from rlmstudio.server.dependencies import AppState, get_state, reset_state
from rlmstudio.server.models import ChatProviderConfig, RuntimeSettings

# ---------------------------------------------------------------------------
# Fake LLM adapter with LiteLLMAdapter-compatible surface
# ---------------------------------------------------------------------------


class _FakeLLMAdapter:
    """Drop-in replacement for LiteLLMAdapter during tests.

    The matrix use case calls ``complete()`` (sync path, via ``to_thread``),
    so this fake only needs to implement the synchronous LLMPort methods.
    """

    def __init__(
        self,
        *,
        response: str,
        input_tokens: int = 10,
        output_tokens: int = 5,
        name: str = "fake",
    ) -> None:
        self._response = response
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._name = name
        self.active_model = name

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        return LLMResponseDTO(
            content=self._response,
            model=self._name,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )

    def complete_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        yield self._response

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def get_pricing(self) -> dict[str, float]:
        return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}

    def get_completion_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state() -> Generator[None, None, None]:
    reset_state()
    yield
    reset_state()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _make_chat_provider(
    name: str, backend: str = "openai", model: str = "gpt-4o"
) -> ChatProviderConfig:
    return ChatProviderConfig(
        id=str(uuid.uuid4()),
        name=name,
        llm_provider=backend,
        llm_model=model,
        execution_mode="direct",
        runtime_settings=RuntimeSettings(),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _register_fake_adapters(state: AppState, response_by_cp: dict[str, str]) -> None:
    """Monkey-patch state.create_llm_adapter_for_chat_provider to return fakes.

    Each chat_provider_id in *response_by_cp* gets a _FakeLLMAdapter that
    always returns the mapped response string.
    """

    def _fake_factory(cp_id: str, num_retries: int | None = None) -> Any:
        return _FakeLLMAdapter(response=response_by_cp.get(cp_id, "default"))

    state.create_llm_adapter_for_chat_provider = _fake_factory  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_missing_content_and_files_returns_400(self, client: TestClient) -> None:
        state = get_state()
        cp = _make_chat_provider("A")
        state.config.chat_providers.append(cp)
        _register_fake_adapters(state, {cp.id: "hi"})

        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "what?",
                "chat_provider_ids": [cp.id],
                "modes": ["direct"],
            },
        )
        assert resp.status_code == 400
        assert "content or file_ids" in resp.json()["error"]["message"]

    def test_unknown_chat_provider_returns_404(self, client: TestClient) -> None:
        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "what?",
                "content": "doc",
                "chat_provider_ids": ["does-not-exist"],
                "modes": ["direct"],
            },
        )
        assert resp.status_code == 404
        assert "Chat Provider" in resp.json()["error"]["message"]

    def test_too_many_slots_returns_400(self, client: TestClient) -> None:
        state = get_state()
        cps = [_make_chat_provider(f"P{i}") for i in range(6)]
        for cp in cps:
            state.config.chat_providers.append(cp)
        _register_fake_adapters(state, {cp.id: "hi" for cp in cps})

        # 6 providers × 2 modes = 12 slots, exceeds MAX_SLOTS=10
        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "q",
                "content": "doc",
                "chat_provider_ids": [cp.id for cp in cps],
                "modes": ["direct", "rlm"],
            },
        )
        assert resp.status_code == 400
        assert "Too many slots" in resp.json()["error"]["message"]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestExecution:
    def test_runs_matrix_and_returns_all_slots(self, client: TestClient) -> None:
        state = get_state()
        cp_a = _make_chat_provider("Provider A", backend="openai", model="gpt-4o")
        cp_b = _make_chat_provider("Provider B", backend="anthropic", model="claude")
        state.config.chat_providers.append(cp_a)
        state.config.chat_providers.append(cp_b)
        _register_fake_adapters(
            state,
            {cp_a.id: "answer from A", cp_b.id: "answer from B"},
        )

        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "what is it?",
                "content": "document text",
                "chat_provider_ids": [cp_a.id, cp_b.id],
                "modes": ["direct"],
                "ranking_metric": "cost",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["comparison_group_id"]
        assert len(data["slots"]) == 2
        assert data["ranking_metric"] == "cost"
        assert len(data["ranking"]) == 2

        # Slots carry chat_provider_id + answer from each
        answers = {s["chat_provider_id"]: s["answer"] for s in data["slots"]}
        assert answers[cp_a.id] == "answer from A"
        assert answers[cp_b.id] == "answer from B"

        # All slots succeed
        assert all(s["success"] for s in data["slots"])
        # Provider/model are backend keys, not display names
        providers = {s["provider"] for s in data["slots"]}
        assert providers == {"openai", "anthropic"}

    def test_cartesian_product(self, client: TestClient) -> None:
        state = get_state()
        cp = _make_chat_provider("P1")
        state.config.chat_providers.append(cp)
        _register_fake_adapters(state, {cp.id: "ok"})

        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "q",
                "content": "doc",
                "chat_provider_ids": [cp.id],
                "modes": ["direct", "rlm"],
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        # 1 provider × 2 modes = 2 slots
        assert len(data["slots"]) == 2
        modes = {s["mode"] for s in data["slots"]}
        assert modes == {"direct", "rlm"}


# ---------------------------------------------------------------------------
# Telemetry persistence
# ---------------------------------------------------------------------------


class TestTelemetryPersistence:
    def test_each_slot_persisted_with_shared_comparison_group_id(self, client: TestClient) -> None:
        state = get_state()
        cp_a = _make_chat_provider("A", backend="openai", model="gpt-4o")
        cp_b = _make_chat_provider("B", backend="anthropic", model="claude")
        state.config.chat_providers.append(cp_a)
        state.config.chat_providers.append(cp_b)
        _register_fake_adapters(state, {cp_a.id: "A-ans", cp_b.id: "B-ans"})

        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "what?",
                "content": "doc",
                "chat_provider_ids": [cp_a.id, cp_b.id],
                "modes": ["direct"],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        group_id = body["comparison_group_id"]

        # Every slot in the response should show up in the store under the
        # shared comparison_group_id (the whole point of this endpoint).
        grouped = state.telemetry.list_runs(comparison_group_id=group_id, limit=100)
        assert len(grouped) == len(body["slots"]) == 2
        assert {r.comparison_group_id for r in grouped} == {group_id}
        assert {r.id for r in grouped} == {s["execution_id"] for s in body["slots"]}

        # Look up each slot's execution_id in the store for per-slot detail
        for slot in body["slots"]:
            detail = state.telemetry.get_run(slot["execution_id"])
            assert detail is not None, f"Slot {slot['slot_id']} not persisted"
            assert detail.provider in ("openai", "anthropic")
            assert detail.session_id == body["session_id"]
            assert detail.success is True
            assert detail.comparison_group_id == group_id

    def test_chat_provider_fields_persisted(self, client: TestClient) -> None:
        state = get_state()
        cp = _make_chat_provider("MyProvider", backend="openai", model="gpt-4o")
        state.config.chat_providers.append(cp)
        _register_fake_adapters(state, {cp.id: "hi"})

        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "q",
                "content": "doc",
                "chat_provider_ids": [cp.id],
                "modes": ["direct"],
            },
        )
        assert resp.status_code == 200
        body = resp.json()

        detail = state.telemetry.get_run(body["slots"][0]["execution_id"])
        assert detail is not None
        # Structured provider/model (NOT the display name "MyProvider")
        assert detail.provider == "openai"
        assert detail.model == "gpt-4o"
        assert detail.chat_provider_id == cp.id
        assert detail.chat_provider_name == "MyProvider"


# ---------------------------------------------------------------------------
# Lifecycle: matrix runs must enter the session/execution graph
# ---------------------------------------------------------------------------


class TestLifecycleIntegration:
    """Matrix slots must be visible to the session/execution graph the rest
    of the server reads from (judge/best-pick, quality engine, resume, etc.)."""

    def test_execution_records_created_for_every_slot(self, client: TestClient) -> None:
        state = get_state()
        cp_a = _make_chat_provider("A", backend="openai", model="gpt-4o")
        cp_b = _make_chat_provider("B", backend="anthropic", model="claude")
        state.config.chat_providers.append(cp_a)
        state.config.chat_providers.append(cp_b)
        _register_fake_adapters(state, {cp_a.id: "A-ans", cp_b.id: "B-ans"})

        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "what?",
                "content": "doc",
                "chat_provider_ids": [cp_a.id, cp_b.id],
                "modes": ["direct"],
            },
        )
        assert resp.status_code == 200
        body = resp.json()

        for slot in body["slots"]:
            exec_id = slot["execution_id"]
            assert exec_id in state.executions, (
                f"Slot {slot['slot_id']} missing from state.executions"
            )
            rec = state.executions[exec_id]
            assert rec.status == "complete"
            assert rec.completed_at is not None
            assert rec.result is not None
            assert rec.result["success"] is True
            assert rec.session_id == body["session_id"]
            assert rec.chat_provider_id in (cp_a.id, cp_b.id)

    def test_session_has_user_message_and_assistant_per_slot(self, client: TestClient) -> None:
        state = get_state()
        cp_a = _make_chat_provider("A")
        cp_b = _make_chat_provider("B", backend="anthropic", model="claude")
        state.config.chat_providers.append(cp_a)
        state.config.chat_providers.append(cp_b)
        _register_fake_adapters(state, {cp_a.id: "A-ans", cp_b.id: "B-ans"})

        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "please compare",
                "content": "doc",
                "chat_provider_ids": [cp_a.id, cp_b.id],
                "modes": ["direct"],
            },
        )
        assert resp.status_code == 200
        body = resp.json()

        session = state.sessions[body["session_id"]]
        roles = [m["role"] for m in session.messages]
        # 1 user message + 2 assistant messages (one per slot)
        assert roles.count("user") == 1
        assert roles.count("assistant") == 2

        # The user message references "compare" mode
        user_msg = next(m for m in session.messages if m["role"] == "user")
        assert user_msg["content"] == "please compare"
        assert user_msg["mode"] == "compare"

        # Each assistant message links back to its slot execution_id
        assistant_exec_ids = {
            m["execution_id"] for m in session.messages if m["role"] == "assistant"
        }
        slot_exec_ids = {s["execution_id"] for s in body["slots"]}
        assert assistant_exec_ids == slot_exec_ids

        # Each assistant message carries the shared comparison_group_id
        group_id = body["comparison_group_id"]
        for m in session.messages:
            if m["role"] == "assistant":
                assert m.get("comparison_group_id") == group_id

    def test_per_provider_conversations_seeded_with_user_turn(self, client: TestClient) -> None:
        """Each affected chat provider's per-provider conversation must
        contain the matrix user turn BEFORE the slot assistant turn, so
        follow-up /api/chat calls on that provider see a balanced history.
        Without this, ``get_conversation(cp_id)`` returns assistant-only
        history and conversational context gets distorted."""
        state = get_state()
        cp_a = _make_chat_provider("A")
        cp_b = _make_chat_provider("B", backend="anthropic", model="claude")
        state.config.chat_providers.append(cp_a)
        state.config.chat_providers.append(cp_b)
        _register_fake_adapters(state, {cp_a.id: "A-ans", cp_b.id: "B-ans"})

        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "please compare",
                "content": "doc",
                "chat_provider_ids": [cp_a.id, cp_b.id],
                "modes": ["direct"],
            },
        )
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]

        # Every chat provider in the matrix must see a full user→assistant
        # turn, not just the assistant reply.
        for cp_id, expected_answer in ((cp_a.id, "A-ans"), (cp_b.id, "B-ans")):
            conv = state.get_conversation(session_id, cp_id)
            roles = [m["role"] for m in conv]
            assert roles == ["user", "assistant"], (
                f"Provider {cp_id}: expected ['user','assistant'], got {roles}"
            )
            assert conv[0]["content"] == "please compare"
            assert conv[0]["mode"] == "compare"
            assert conv[1]["content"] == expected_answer

    def test_execution_ids_visible_to_turn_grouping(self, client: TestClient) -> None:
        """``get_execution_ids_for_turn()`` (used by best-pick flows) must
        return every slot in the matrix, not an empty set."""
        state = get_state()
        cp_a = _make_chat_provider("A")
        cp_b = _make_chat_provider("B", backend="anthropic", model="claude")
        state.config.chat_providers.append(cp_a)
        state.config.chat_providers.append(cp_b)
        _register_fake_adapters(state, {cp_a.id: "A", cp_b.id: "B"})

        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "q",
                "content": "doc",
                "chat_provider_ids": [cp_a.id, cp_b.id],
                "modes": ["direct"],
            },
        )
        body = resp.json()
        slot_ids = {s["execution_id"] for s in body["slots"]}

        # Pick any slot's execution_id and verify the whole turn is reachable
        any_exec_id = next(iter(slot_ids))
        turn_ids = state.get_execution_ids_for_turn(body["session_id"], any_exec_id)
        assert turn_ids == slot_ids


# ---------------------------------------------------------------------------
# RAG key resolution parity with /api/chat
# ---------------------------------------------------------------------------


class TestRAGKeyResolution:
    """Matrix RAG mode must honor provider-stored OpenAI secrets when
    ``OPENAI_API_KEY`` is not set in the environment (same fallback as
    ``/api/chat`` RAG)."""

    def test_falls_back_to_llm_provider_instance_key(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from rlmstudio.server.dependencies import LLMProviderConfig

        state = get_state()

        # Ensure env var is NOT set so the fallback path must fire.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Register a named OpenAI LLM Provider instance that the chat
        # provider will point at.
        lp = LLMProviderConfig(
            id=str(uuid.uuid4()),
            name="My OpenAI",
            backend="openai",
            model="gpt-4o",
            endpoint=None,
            runtime_settings=RuntimeSettings(),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        state.config.llm_providers.append(lp)

        cp = _make_chat_provider("CP", backend="openai", model="gpt-4o")
        cp.llm_provider_id = lp.id
        state.config.chat_providers.append(cp)
        _register_fake_adapters(state, {cp.id: "rag-ans"})

        # Spy on _get_api_key to make sure the fallback path was hit.
        captured: dict[str, tuple[str, str]] = {}

        def _fake_get_api_key(instance_id: str, backend: str) -> str:
            captured["args"] = (instance_id, backend)
            return "sk-from-provider-secret"

        # Spy on the embedder constructor to capture the api_key it got.
        embedder_keys: list[str | None] = []
        real_init = LiteLLMEmbeddingAdapter.__init__

        def _spy_init(self: Any, *args: Any, **kwargs: Any) -> None:
            embedder_keys.append(kwargs.get("api_key"))
            real_init(self, *args, **kwargs)

        monkeypatch.setattr(
            "rlmstudio.server.routes.llm_providers._get_api_key",
            _fake_get_api_key,
        )
        monkeypatch.setattr(LiteLLMEmbeddingAdapter, "__init__", _spy_init)

        # We don't actually need RAG to succeed — the FakeLLMAdapter returns
        # a stub response, and we only care that the embedder was built with
        # the right key.  Mock out the RAG use case's embed/search calls by
        # using a no-op storage/embedder won't work here; instead point the
        # route at a very small doc and let RAG fail gracefully — the slot
        # result will be a failure, but the spy will still have fired.
        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "q",
                "content": "doc",
                "chat_provider_ids": [cp.id],
                "modes": ["rag"],
            },
        )

        # Whatever the slot outcome (embedding may still fail without a real
        # API), the fallback must have been called with the LP instance.
        assert resp.status_code == 200
        assert captured.get("args") == (lp.id, "openai"), (
            "matrix RAG did not fall back to provider-stored secret"
        )
        assert embedder_keys == ["sk-from-provider-secret"]


# ---------------------------------------------------------------------------
# Per-slot RLM settings from chat-provider profiles
# ---------------------------------------------------------------------------


class TestPerSlotRLMSettings:
    """Matrix RLM slots must apply the same profile-specific RLM knobs that
    ``/api/chat`` applies (max_steps, repeat_limit, nudge_at_fraction,
    system_prompt_extra).  Without this the comparison is misleading."""

    def test_rlm_slot_gets_chat_provider_rlm_settings(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rlmstudio.application.use_cases import run_matrix_comparison as uc_mod

        state = get_state()

        cp = _make_chat_provider("RLM-CP", backend="openai", model="gpt-4o")
        cp.execution_mode = "rlm"
        cp.rlm_max_steps = 42
        cp.rlm_timeout_seconds = 99
        cp.rlm_repeat_limit = 7
        cp.rlm_nudge_at_fraction = 0.6
        state.config.chat_providers.append(cp)
        _register_fake_adapters(state, {cp.id: "done"})

        # Capture the actual RunConfigDTO that the use case hands to each
        # slot by patching _execute_slot to intercept the cfg.
        captured_configs: list[Any] = []
        real_execute_slot = uc_mod.RunMatrixComparisonUseCase._execute_slot

        def _spy_execute_slot(
            self: uc_mod.RunMatrixComparisonUseCase,
            slot: uc_mod.MatrixSlotDTO,
            content: str,
            query: str,
            base_config: Any,
        ) -> Any:
            captured_configs.append(slot.config)
            # Swap to direct mode so we don't need a real RLM loop
            slot.mode = "direct"
            slot.sandbox = None
            return real_execute_slot(self, slot, content, query, base_config)

        monkeypatch.setattr(
            uc_mod.RunMatrixComparisonUseCase,
            "_execute_slot",
            _spy_execute_slot,
        )

        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "q",
                "content": "doc",
                "chat_provider_ids": [cp.id],
                "modes": ["rlm"],
            },
        )
        assert resp.status_code == 200

        assert len(captured_configs) == 1
        slot_cfg = captured_configs[0]
        assert slot_cfg is not None, "RLM slot did not receive a per-slot config"
        # These are the values /api/chat copies from the chat provider.
        assert slot_cfg.max_steps == 42
        assert slot_cfg.max_time_seconds == 99.0
        assert slot_cfg.repeat_limit == 7
        assert slot_cfg.nudge_at_fraction == 0.6

    def test_direct_slot_does_not_get_rlm_knobs(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A direct slot's config should be the base run_config, not the
        RLM knobs — we only apply the RLM-specific settings when mode='rlm'."""
        from rlmstudio.application.use_cases import run_matrix_comparison as uc_mod

        state = get_state()
        cp = _make_chat_provider("Direct-CP")
        cp.rlm_max_steps = 999  # would be noticeable if applied by mistake
        state.config.chat_providers.append(cp)
        _register_fake_adapters(state, {cp.id: "ok"})

        captured_configs: list[Any] = []
        real_execute_slot = uc_mod.RunMatrixComparisonUseCase._execute_slot

        def _spy_execute_slot(
            self: uc_mod.RunMatrixComparisonUseCase,
            slot: uc_mod.MatrixSlotDTO,
            content: str,
            query: str,
            base_config: Any,
        ) -> Any:
            captured_configs.append(slot.config)
            return real_execute_slot(self, slot, content, query, base_config)

        monkeypatch.setattr(
            uc_mod.RunMatrixComparisonUseCase,
            "_execute_slot",
            _spy_execute_slot,
        )

        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "q",
                "content": "doc",
                "chat_provider_ids": [cp.id],
                "modes": ["direct"],
            },
        )
        assert resp.status_code == 200
        assert len(captured_configs) == 1
        cfg = captured_configs[0]
        assert cfg is not None
        assert cfg.mode == "direct"
        # The direct slot should NOT inherit rlm_max_steps=999
        assert cfg.max_steps != 999

    def test_direct_slot_gets_the_slot_max_time(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """M3: the LLM Tuner's Max time reaches Direct slots, not only RLM ones."""
        from rlmstudio.application.use_cases import run_matrix_comparison as uc_mod

        state = get_state()
        cp = _make_chat_provider("Direct-CP")
        cp.rlm_timeout_seconds = 5
        state.config.chat_providers.append(cp)
        _register_fake_adapters(state, {cp.id: "ok"})

        captured_configs: list[Any] = []
        real_execute_slot = uc_mod.RunMatrixComparisonUseCase._execute_slot

        def _spy_execute_slot(
            self: uc_mod.RunMatrixComparisonUseCase,
            slot: uc_mod.MatrixSlotDTO,
            content: str,
            query: str,
            base_config: Any,
        ) -> Any:
            captured_configs.append(slot.config)
            return real_execute_slot(self, slot, content, query, base_config)

        monkeypatch.setattr(uc_mod.RunMatrixComparisonUseCase, "_execute_slot", _spy_execute_slot)

        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "q",
                "content": "doc",
                "chat_provider_ids": [cp.id],
                "modes": ["direct"],
            },
        )
        assert resp.status_code == 200
        assert captured_configs[0].max_time_seconds == 5.0


# ---------------------------------------------------------------------------
# End-to-end: matrix → follow-up /api/chat on one of the providers
# ---------------------------------------------------------------------------


class _CapturingLLMAdapter(_FakeLLMAdapter):
    """FakeLLMAdapter that records every ``messages`` list it receives.

    Used to prove that a follow-up ``/api/chat`` request, after a matrix
    run, builds its prompt from a balanced per-provider history (not from
    the flat session.messages list or an empty conversation).
    """

    def __init__(self, response: str = "follow-up-ok") -> None:
        super().__init__(response=response)
        self.captured_calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        # Copy so subsequent mutations to the input list don't retroactively
        # change what we captured.
        self.captured_calls.append([dict(m) for m in messages])
        return super().complete(messages)


class TestMatrixThenFollowUpChat:
    """Run compare-matrix, then run /api/chat on one of the matrix providers,
    and verify the follow-up chat consumes the balanced per-provider history.

    This closes the residual integration gap flagged during review: the
    state-level seeding test covers what lands in ``session.conversations``,
    but this test proves that the *next* LLM call actually sees it.
    """

    def test_follow_up_chat_uses_matrix_history(self, client: TestClient) -> None:
        import asyncio

        from rlmstudio.server.dependencies import ExecutionRecord
        from rlmstudio.server.routes.chat import _run_execution

        state = get_state()

        # Two chat providers registered in the matrix.
        cp_a = _make_chat_provider("A", backend="openai", model="gpt-4o")
        cp_b = _make_chat_provider("B", backend="anthropic", model="claude")
        state.config.chat_providers.append(cp_a)
        state.config.chat_providers.append(cp_b)

        # Capturing adapter for provider A; vanilla fake for provider B.
        capturing_llm = _CapturingLLMAdapter(response="A-reply")
        adapters: dict[str, Any] = {
            cp_a.id: capturing_llm,
            cp_b.id: _FakeLLMAdapter(response="B-reply"),
        }

        def _fake_factory(cp_id: str, num_retries: int | None = None) -> Any:
            return adapters[cp_id]

        state.create_llm_adapter_for_chat_provider = _fake_factory  # type: ignore[method-assign]

        # --- Step 1: matrix run seeds both per-provider conversations ---
        resp = client.post(
            "/api/chat/compare-matrix",
            json={
                "query": "what are the key risks?",
                "content": "document body",
                "chat_provider_ids": [cp_a.id, cp_b.id],
                "modes": ["direct"],
            },
        )
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]

        # Sanity: provider A's per-provider conversation has [user, assistant]
        conv_a = state.get_conversation(session_id, cp_a.id)
        assert [m["role"] for m in conv_a] == ["user", "assistant"]
        assert conv_a[1]["content"] == "A-reply"

        # Clear captured calls from the matrix run (we only care about the
        # follow-up chat's LLM input).
        capturing_llm.captured_calls.clear()

        # --- Step 2: a direct follow-up chat on provider A ---
        # Instead of going through POST /api/chat (which spawns a background
        # task we'd have to poll), we replicate what the handler does up to
        # _run_execution and then call it directly.  This is deterministic
        # and the test fails if the history-reading logic ever regresses.
        follow_up_query = "and which one is most urgent?"
        now = datetime.now(timezone.utc)
        follow_up_msg = {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": follow_up_query,
            "file_ids": [],
            "mode": "direct",
            "chat_provider_id": cp_a.id,
            "timestamp": now.isoformat(),
        }
        state.add_message(session_id, follow_up_msg, cp_a.id)

        execution = ExecutionRecord(
            execution_id=str(uuid.uuid4()),
            session_id=session_id,
            query=follow_up_query,
            mode="direct",
            status="running",
            started_at=now,
            chat_provider_id=cp_a.id,
            chat_provider_name=cp_a.name,
        )
        state.executions[execution.execution_id] = execution

        asyncio.run(
            _run_execution(
                state=state,
                execution=execution,
                content="document body",
                query=follow_up_query,
                mode="direct",
                chat_provider_id=cp_a.id,
            )
        )

        # --- Step 3: assert the follow-up LLM saw balanced history ---
        assert len(capturing_llm.captured_calls) == 1, (
            "follow-up chat must have called the LLM exactly once"
        )
        call = capturing_llm.captured_calls[0]
        user_msgs = [m for m in call if m["role"] == "user"]
        assistant_msgs = [m for m in call if m["role"] == "assistant"]

        # Native chat messages: prior turn is injected as separate
        # user/assistant messages before the current user message.
        assert len(user_msgs) == 2, "history user + current user"
        assert user_msgs[0]["content"] == "what are the key risks?"  # matrix user
        assert "and which one is most urgent?" in user_msgs[1]["content"]  # current

        assert len(assistant_msgs) == 1
        assert "A-reply" in assistant_msgs[0]["content"]  # provider A only
        # provider B's reply must NOT leak into provider A's history
        assert "B-reply" not in assistant_msgs[0]["content"]

        # And the execution succeeded end-to-end
        assert execution.status == "complete"
        assert execution.result is not None
        assert execution.result["success"] is True
