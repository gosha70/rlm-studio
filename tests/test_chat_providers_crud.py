"""Acceptance tests for Chat Providers CRUD API (spec 01).

Verifies all acceptance criteria from specs/01-backend-models-crud.md:
  1. GET /api/chat-providers returns [] on fresh install (after reset_state)
  2. POST /api/chat-providers creates/persists; rejects duplicates (409) and
     invalid llm_provider_id (400)
  3. PUT /api/chat-providers/{id} updates fields; validates name uniqueness (409);
     returns 404 for missing ID
  4. DELETE /api/chat-providers/{id} removes; returns 404 for missing; 204 on success
  5. Auto-migration creates one "DIRECT-{PROVIDER}" Chat Provider per enabled
     provider config on first load
  6. ChatRequest model accepts chat_provider_id field
  7. ChatResponse model includes chat_provider_id field
  8. SessionMessage model includes chat_provider_id and chat_provider_name fields
  9. SessionDetail model includes conversations dict field
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from datetime import datetime, timezone
from typing import Any

import pytest
from starlette.testclient import TestClient

from rlmstudio.server.app import app
from rlmstudio.server.dependencies import get_state, reset_state
from rlmstudio.server.models import (
    ChatRequest,
    ChatResponse,
    ProviderConfig,
    RuntimeSettings,
    SessionDetail,
    SessionMessage,
)
from rlmstudio.ui.data.providers_catalog import PROVIDERS_BY_KEY

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state() -> Generator[None, None, None]:
    """Reset in-memory state (no disk I/O) before and after each test."""
    reset_state()
    yield
    reset_state()


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient backed by the real application."""
    return TestClient(app)


@pytest.fixture
def valid_provider_key() -> str:
    """Return the first key from the catalog that is recognised by the route."""
    return str(next(iter(PROVIDERS_BY_KEY)))


@pytest.fixture
def valid_model(valid_provider_key: str) -> str:
    """Return a valid model name for the chosen provider, or a fallback."""
    entry = PROVIDERS_BY_KEY[valid_provider_key]
    if entry.models:
        return str(entry.models[0].name)
    return "test-model"


@pytest.fixture
def created_llm_provider(
    client: TestClient, valid_provider_key: str, valid_model: str
) -> dict[str, Any]:
    """Create one LLM Provider via the API and return the response JSON."""
    resp = client.post(
        "/api/llm-providers",
        json={
            "name": "TEST-LLM-PROVIDER",
            "backend": valid_provider_key,
            "model": valid_model,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()  # type: ignore[no-any-return]


@pytest.fixture
def created_provider(client: TestClient, created_llm_provider: dict[str, Any]) -> dict[str, Any]:
    """Create one Chat Provider via the API and return the response JSON."""
    resp = client.post(
        "/api/chat-providers",
        json={
            "name": "TEST-PROVIDER",
            "llm_provider_id": created_llm_provider["id"],
            "execution_mode": "direct",
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# AC-1: GET /api/chat-providers returns [] on fresh install
# ---------------------------------------------------------------------------


class TestListEmpty:
    def test_returns_empty_list_after_reset(self, client: TestClient) -> None:
        """After reset_state() the list must be empty."""
        resp = client.get("/api/chat-providers")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_response_is_json_array(self, client: TestClient) -> None:
        resp = client.get("/api/chat-providers")
        assert isinstance(resp.json(), list)


# ---------------------------------------------------------------------------
# AC-2: POST /api/chat-providers — create, persist, and reject bad input
# ---------------------------------------------------------------------------


class TestCreateChatProvider:
    def test_creates_with_201(
        self, client: TestClient, created_llm_provider: dict[str, Any]
    ) -> None:
        resp = client.post(
            "/api/chat-providers",
            json={
                "name": "MY-PROVIDER",
                "llm_provider_id": created_llm_provider["id"],
                "execution_mode": "direct",
            },
        )
        assert resp.status_code == 201, resp.text

    def test_response_contains_required_fields(
        self, client: TestClient, created_llm_provider: dict[str, Any]
    ) -> None:
        resp = client.post(
            "/api/chat-providers",
            json={
                "name": "MY-PROVIDER",
                "llm_provider_id": created_llm_provider["id"],
                "execution_mode": "direct",
            },
        )
        data = resp.json()
        assert "id" in data
        assert data["name"] == "MY-PROVIDER"
        assert data["llm_provider_id"] == created_llm_provider["id"]
        assert data["execution_mode"] == "direct"

    def test_assigns_uuid_id(
        self, client: TestClient, created_llm_provider: dict[str, Any]
    ) -> None:
        resp = client.post(
            "/api/chat-providers",
            json={
                "name": "UUID-CHECK",
                "llm_provider_id": created_llm_provider["id"],
                "execution_mode": "direct",
            },
        )
        data = resp.json()
        # Should not raise if it is a valid UUID
        uuid.UUID(data["id"])

    def test_persists_in_state(
        self, client: TestClient, created_llm_provider: dict[str, Any]
    ) -> None:
        """Created provider must be retrievable from state without another POST."""
        client.post(
            "/api/chat-providers",
            json={
                "name": "PERSIST-CHECK",
                "llm_provider_id": created_llm_provider["id"],
                "execution_mode": "direct",
            },
        )
        state = get_state()
        names = [cp.name for cp in state.config.chat_providers]
        assert "PERSIST-CHECK" in names

    def test_appears_in_list_after_create(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        """GET list should include newly created provider."""
        resp = client.get("/api/chat-providers")
        assert resp.status_code == 200
        ids = [cp["id"] for cp in resp.json()]
        assert created_provider["id"] in ids

    def test_duplicate_name_returns_409(
        self, client: TestClient, created_llm_provider: dict[str, Any]
    ) -> None:
        """Second POST with same name must return 409 CONFLICT."""
        payload = {
            "name": "DUPLICATE",
            "llm_provider_id": created_llm_provider["id"],
            "execution_mode": "direct",
        }
        client.post("/api/chat-providers", json=payload)
        resp = client.post("/api/chat-providers", json=payload)
        assert resp.status_code == 409, resp.text

    def test_duplicate_name_case_insensitive(
        self, client: TestClient, created_llm_provider: dict[str, Any]
    ) -> None:
        """Name uniqueness check must be case-insensitive."""
        client.post(
            "/api/chat-providers",
            json={
                "name": "MyProvider",
                "llm_provider_id": created_llm_provider["id"],
                "execution_mode": "direct",
            },
        )
        resp = client.post(
            "/api/chat-providers",
            json={
                "name": "myprovider",
                "llm_provider_id": created_llm_provider["id"],
                "execution_mode": "direct",
            },
        )
        assert resp.status_code == 409, resp.text

    def test_invalid_llm_provider_id_returns_400(self, client: TestClient) -> None:
        """Non-existent llm_provider_id must return 400."""
        resp = client.post(
            "/api/chat-providers",
            json={
                "name": "BAD-PROVIDER",
                "llm_provider_id": str(uuid.uuid4()),
                "execution_mode": "direct",
            },
        )
        assert resp.status_code == 400, resp.text

    def test_all_execution_modes_accepted(
        self, client: TestClient, created_llm_provider: dict[str, Any]
    ) -> None:
        """All four execution modes must be accepted."""
        for mode in ("direct", "rlm", "rag", "rlm_official"):
            resp = client.post(
                "/api/chat-providers",
                json={
                    "name": f"PROVIDER-{mode.upper()}",
                    "llm_provider_id": created_llm_provider["id"],
                    "execution_mode": mode,
                },
            )
            assert resp.status_code == 201, f"mode={mode}: {resp.text}"

    def test_timestamps_set_on_create(
        self, client: TestClient, created_llm_provider: dict[str, Any]
    ) -> None:
        resp = client.post(
            "/api/chat-providers",
            json={
                "name": "TIMESTAMP-CHECK",
                "llm_provider_id": created_llm_provider["id"],
                "execution_mode": "direct",
            },
        )
        data = resp.json()
        assert data["created_at"] is not None
        assert data["updated_at"] is not None


# ---------------------------------------------------------------------------
# AC-3: PUT /api/chat-providers/{id} — update fields
# ---------------------------------------------------------------------------


class TestUpdateChatProvider:
    def test_update_name(self, client: TestClient, created_provider: dict[str, Any]) -> None:
        cp_id = created_provider["id"]
        resp = client.put(
            f"/api/chat-providers/{cp_id}",
            json={"name": "RENAMED"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["name"] == "RENAMED"

    def test_update_llm_provider_id(
        self,
        client: TestClient,
        created_provider: dict[str, Any],
        valid_provider_key: str,
    ) -> None:
        """Switching to a different LLM Provider by ID must succeed."""
        # Create a second LLM Provider
        entry = PROVIDERS_BY_KEY[valid_provider_key]
        model2 = entry.models[1].name if len(entry.models) >= 2 else "gpt-4o-mini"
        lp2_resp = client.post(
            "/api/llm-providers",
            json={"name": "Second LLM Provider", "backend": valid_provider_key, "model": model2},
        )
        assert lp2_resp.status_code == 201
        lp2_id = lp2_resp.json()["id"]

        cp_id = created_provider["id"]
        resp = client.put(
            f"/api/chat-providers/{cp_id}",
            json={"llm_provider_id": lp2_id},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["llm_provider_id"] == lp2_id

    def test_update_execution_mode(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        cp_id = created_provider["id"]
        resp = client.put(
            f"/api/chat-providers/{cp_id}",
            json={"execution_mode": "rlm"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["execution_mode"] == "rlm"

    def test_update_updates_timestamp(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        cp_id = created_provider["id"]
        resp = client.put(
            f"/api/chat-providers/{cp_id}",
            json={"name": "UPDATED"},
        )
        assert resp.status_code == 200
        assert resp.json()["updated_at"] is not None

    def test_update_nonexistent_returns_404(self, client: TestClient) -> None:
        fake_id = str(uuid.uuid4())
        resp = client.put(
            f"/api/chat-providers/{fake_id}",
            json={"name": "WHATEVER"},
        )
        assert resp.status_code == 404, resp.text

    def test_update_name_to_duplicate_returns_409(
        self, client: TestClient, created_llm_provider: dict[str, Any]
    ) -> None:
        """Renaming to an already-used name must return 409."""
        client.post(
            "/api/chat-providers",
            json={
                "name": "FIRST",
                "llm_provider_id": created_llm_provider["id"],
                "execution_mode": "direct",
            },
        )
        resp2 = client.post(
            "/api/chat-providers",
            json={
                "name": "SECOND",
                "llm_provider_id": created_llm_provider["id"],
                "execution_mode": "direct",
            },
        )
        second_id = resp2.json()["id"]
        resp = client.put(
            f"/api/chat-providers/{second_id}",
            json={"name": "FIRST"},
        )
        assert resp.status_code == 409, resp.text

    def test_update_same_name_allowed(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        """Updating to the same name (no actual change) must not raise 409."""
        cp_id = created_provider["id"]
        same_name = created_provider["name"]
        resp = client.put(
            f"/api/chat-providers/{cp_id}",
            json={"name": same_name},
        )
        assert resp.status_code == 200, resp.text

    def test_get_by_id_returns_updated_data(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        cp_id = created_provider["id"]
        client.put(f"/api/chat-providers/{cp_id}", json={"name": "GET-UPDATED"})
        resp = client.get(f"/api/chat-providers/{cp_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "GET-UPDATED"


# ---------------------------------------------------------------------------
# AC-4: DELETE /api/chat-providers/{id}
# ---------------------------------------------------------------------------


class TestDeleteChatProvider:
    def test_delete_returns_204(self, client: TestClient, created_provider: dict[str, Any]) -> None:
        cp_id = created_provider["id"]
        resp = client.delete(f"/api/chat-providers/{cp_id}")
        assert resp.status_code == 204, resp.text

    def test_delete_removes_from_list(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        cp_id = created_provider["id"]
        client.delete(f"/api/chat-providers/{cp_id}")
        resp = client.get("/api/chat-providers")
        ids = [cp["id"] for cp in resp.json()]
        assert cp_id not in ids

    def test_delete_nonexistent_returns_404(self, client: TestClient) -> None:
        fake_id = str(uuid.uuid4())
        resp = client.delete(f"/api/chat-providers/{fake_id}")
        assert resp.status_code == 404, resp.text

    def test_get_by_id_after_delete_returns_404(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        cp_id = created_provider["id"]
        client.delete(f"/api/chat-providers/{cp_id}")
        resp = client.get(f"/api/chat-providers/{cp_id}")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# AC-5: Auto-migration creates "DIRECT-{PROVIDER}" on first load
# ---------------------------------------------------------------------------


class TestAutoMigration:
    def test_migration_creates_chat_provider_for_enabled_config(self) -> None:
        """When provider_configs has an enabled entry, migration should create a CP."""
        state = get_state()
        assert state.config.chat_providers == []

        pc = ProviderConfig(
            provider="openai",
            model="gpt-4o",
            enabled=True,
            runtime_settings=RuntimeSettings(),
        )
        state.config.provider_configs.append(pc)
        state._migrate_chat_providers()

        names = [cp.name for cp in state.config.chat_providers]
        assert "DIRECT-OPENAI" in names

    def test_migration_skips_disabled_provider_configs(self) -> None:
        """Disabled provider configs must not generate Chat Providers."""
        state = get_state()
        pc = ProviderConfig(
            provider="anthropic",
            model="claude-sonnet-4-5",
            enabled=False,
            runtime_settings=RuntimeSettings(),
        )
        state.config.provider_configs.append(pc)
        state._migrate_chat_providers()
        names = [cp.name for cp in state.config.chat_providers]
        assert "DIRECT-ANTHROPIC" not in names

    def test_migration_is_idempotent(self) -> None:
        """Calling migration twice must not create duplicate Chat Providers."""
        state = get_state()
        pc = ProviderConfig(
            provider="openai",
            model="gpt-4o",
            enabled=True,
            runtime_settings=RuntimeSettings(),
        )
        state.config.provider_configs.append(pc)
        state._migrate_chat_providers()
        count_after_first = len(state.config.chat_providers)
        state._migrate_chat_providers()
        count_after_second = len(state.config.chat_providers)
        assert count_after_first == count_after_second

    def test_migration_uses_direct_mode(self) -> None:
        """Migrated Chat Providers must have execution_mode='direct'."""
        state = get_state()
        pc = ProviderConfig(
            provider="openai",
            model="gpt-4o",
            enabled=True,
            runtime_settings=RuntimeSettings(),
        )
        state.config.provider_configs.append(pc)
        state._migrate_chat_providers()
        for cp in state.config.chat_providers:
            assert cp.execution_mode == "direct"

    def test_migration_propagates_runtime_settings(self) -> None:
        """Migrated Chat Providers should copy runtime_settings from ProviderConfig."""
        state = get_state()
        rt = RuntimeSettings(temperature=0.3, max_output_tokens=512)
        pc = ProviderConfig(
            provider="openai",
            model="gpt-4o",
            enabled=True,
            runtime_settings=rt,
        )
        state.config.provider_configs.append(pc)
        state._migrate_chat_providers()
        cp = state.config.chat_providers[0]
        assert cp.runtime_settings.temperature == 0.3
        assert cp.runtime_settings.max_output_tokens == 512

    def test_migration_name_format(self) -> None:
        """Migrated names must follow 'DIRECT-{PROVIDER_UPPERCASE}' pattern."""
        state = get_state()
        for key in ("openai", "anthropic"):
            pc = ProviderConfig(
                provider=key,
                model="some-model",
                enabled=True,
                runtime_settings=RuntimeSettings(),
            )
            state.config.provider_configs.append(pc)
        state._migrate_chat_providers()
        names = [cp.name for cp in state.config.chat_providers]
        for key in ("openai", "anthropic"):
            assert f"DIRECT-{key.upper()}" in names


# ---------------------------------------------------------------------------
# AC-6: ChatRequest model accepts chat_provider_id field
# ---------------------------------------------------------------------------


class TestChatRequestModel:
    def test_chat_request_accepts_chat_provider_id(self) -> None:
        req = ChatRequest(
            query="hello",
            content="some content",
            chat_provider_id="cp-123",
        )
        assert req.chat_provider_id == "cp-123"

    def test_chat_provider_id_is_optional(self) -> None:
        req = ChatRequest(query="hello", content="some content")
        assert req.chat_provider_id is None

    def test_chat_request_via_api_accepts_chat_provider_id(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        """POST /api/chat with chat_provider_id must not be rejected as invalid input."""
        resp = client.post(
            "/api/chat",
            json={
                "query": "test",
                "content": "context",
                "mode": "direct",
                "chat_provider_id": created_provider["id"],
            },
        )
        # 202 = accepted; 404 = cp not found is acceptable here but not a 422
        assert resp.status_code in (202, 404), f"Unexpected status: {resp.status_code} {resp.text}"

    def test_chat_request_without_chat_provider_id_still_works(self, client: TestClient) -> None:
        resp = client.post(
            "/api/chat",
            json={
                "query": "test",
                "content": "context",
                "mode": "direct",
            },
        )
        assert resp.status_code == 202, resp.text


# ---------------------------------------------------------------------------
# AC-7: ChatResponse model includes chat_provider_id field
# ---------------------------------------------------------------------------


class TestChatResponseModel:
    def test_chat_response_has_chat_provider_id_field(self) -> None:
        resp = ChatResponse(
            execution_id="exec-1",
            session_id="sess-1",
            status="running",
            chat_provider_id="cp-abc",
        )
        assert resp.chat_provider_id == "cp-abc"

    def test_chat_provider_id_is_optional_on_response(self) -> None:
        resp = ChatResponse(
            execution_id="exec-1",
            session_id="sess-1",
        )
        assert resp.chat_provider_id is None

    def test_api_response_echoes_chat_provider_id(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        """The API response must echo back the chat_provider_id that was sent."""
        resp = client.post(
            "/api/chat",
            json={
                "query": "echo test",
                "content": "ctx",
                "mode": "direct",
                "chat_provider_id": created_provider["id"],
            },
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "chat_provider_id" in data
        assert data["chat_provider_id"] == created_provider["id"]

    def test_api_response_chat_provider_id_null_when_not_sent(self, client: TestClient) -> None:
        resp = client.post(
            "/api/chat",
            json={"query": "test", "content": "ctx", "mode": "direct"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "chat_provider_id" in data
        assert data["chat_provider_id"] is None


# ---------------------------------------------------------------------------
# AC-8: SessionMessage model includes chat_provider_id and chat_provider_name
# ---------------------------------------------------------------------------


class TestSessionMessageModel:
    def test_session_message_has_chat_provider_id_field(self) -> None:
        msg = SessionMessage(
            id="msg-1",
            role="user",
            content="hello",
            timestamp=datetime.now(timezone.utc),
            chat_provider_id="cp-123",
        )
        assert msg.chat_provider_id == "cp-123"

    def test_session_message_has_chat_provider_name_field(self) -> None:
        msg = SessionMessage(
            id="msg-1",
            role="assistant",
            content="reply",
            timestamp=datetime.now(timezone.utc),
            chat_provider_name="MY-PROVIDER",
        )
        assert msg.chat_provider_name == "MY-PROVIDER"

    def test_session_message_both_fields_optional(self) -> None:
        msg = SessionMessage(
            id="msg-1",
            role="user",
            content="hi",
            timestamp=datetime.now(timezone.utc),
        )
        assert msg.chat_provider_id is None
        assert msg.chat_provider_name is None

    def test_session_message_fields_reflected_in_api(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        """Messages stored with chat_provider_id must be retrievable via GET /api/sessions/{id}."""
        chat_resp = client.post(
            "/api/chat",
            json={
                "query": "hello",
                "content": "ctx",
                "mode": "direct",
                "chat_provider_id": created_provider["id"],
            },
        )
        assert chat_resp.status_code == 202
        session_id = chat_resp.json()["session_id"]

        detail_resp = client.get(f"/api/sessions/{session_id}")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        messages = detail["messages"]
        assert len(messages) > 0
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) > 0
        assert user_msgs[0]["chat_provider_id"] == created_provider["id"]


# ---------------------------------------------------------------------------
# AC-9: SessionDetail model includes conversations dict field
# ---------------------------------------------------------------------------


class TestSessionDetailModel:
    def test_session_detail_has_conversations_field(self) -> None:
        detail = SessionDetail(
            id="s-1",
            name="Test Session",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert hasattr(detail, "conversations")
        assert isinstance(detail.conversations, dict)

    def test_session_detail_conversations_default_empty(self) -> None:
        detail = SessionDetail(
            id="s-1",
            name="Test Session",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert detail.conversations == {}

    def test_session_detail_conversations_keyed_by_provider_id(self) -> None:
        msg = SessionMessage(
            id="m-1",
            role="user",
            content="hi",
            timestamp=datetime.now(timezone.utc),
            chat_provider_id="cp-abc",
        )
        detail = SessionDetail(
            id="s-1",
            name="Test",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            conversations={"cp-abc": [msg]},
        )
        assert "cp-abc" in detail.conversations
        assert detail.conversations["cp-abc"][0].chat_provider_id == "cp-abc"

    def test_api_session_detail_includes_conversations_key(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        """GET /api/sessions/{id} response must include 'conversations' key."""
        chat_resp = client.post(
            "/api/chat",
            json={
                "query": "test",
                "content": "ctx",
                "mode": "direct",
                "chat_provider_id": created_provider["id"],
            },
        )
        assert chat_resp.status_code == 202
        session_id = chat_resp.json()["session_id"]

        detail_resp = client.get(f"/api/sessions/{session_id}")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert "conversations" in detail
        assert isinstance(detail["conversations"], dict)

    def test_api_conversations_keyed_by_chat_provider_id(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        """conversations dict keys must be the chat_provider_id values."""
        chat_resp = client.post(
            "/api/chat",
            json={
                "query": "hello",
                "content": "ctx",
                "mode": "direct",
                "chat_provider_id": created_provider["id"],
            },
        )
        assert chat_resp.status_code == 202
        session_id = chat_resp.json()["session_id"]

        detail_resp = client.get(f"/api/sessions/{session_id}")
        assert detail_resp.status_code == 200
        conversations = detail_resp.json()["conversations"]
        assert created_provider["id"] in conversations


# ---------------------------------------------------------------------------
# GET /api/chat-providers/{id} — supplementary coverage
# ---------------------------------------------------------------------------


class TestGetChatProviderById:
    def test_get_by_id_returns_200(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        cp_id = created_provider["id"]
        resp = client.get(f"/api/chat-providers/{cp_id}")
        assert resp.status_code == 200

    def test_get_by_id_returns_correct_data(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        cp_id = created_provider["id"]
        resp = client.get(f"/api/chat-providers/{cp_id}")
        data = resp.json()
        assert data["id"] == cp_id
        assert data["name"] == created_provider["name"]

    def test_get_nonexistent_returns_404(self, client: TestClient) -> None:
        fake_id = str(uuid.uuid4())
        resp = client.get(f"/api/chat-providers/{fake_id}")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# conversation_memory_enabled + conversation_memory_fraction
#
# These tests pin the API contract the frontend Settings toggle will
# consume in a later session.  They cover POST/PATCH/GET round-trip,
# persistence model_dump/model_validate round-trip, legacy-config
# load (no field = default True), and validation bounds.  No execution
# code reads the field yet — that is wired in Commit 4.
# ---------------------------------------------------------------------------


class TestConversationMemoryField:
    def test_create_defaults_to_enabled_true(
        self, client: TestClient, created_llm_provider: dict[str, Any]
    ) -> None:
        """POST without the field → feature defaults to enabled."""
        resp = client.post(
            "/api/chat-providers",
            json={
                "name": "CP-DEFAULT",
                "llm_provider_id": created_llm_provider["id"],
                "execution_mode": "direct",
            },
        )
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert data["conversation_memory_enabled"] is True
        assert data["conversation_memory_fraction"] == 0.30

    def test_create_honours_disabled_flag(
        self, client: TestClient, created_llm_provider: dict[str, Any]
    ) -> None:
        """POST with conversation_memory_enabled: false round-trips via GET."""
        resp = client.post(
            "/api/chat-providers",
            json={
                "name": "CP-STATELESS",
                "llm_provider_id": created_llm_provider["id"],
                "execution_mode": "direct",
                "conversation_memory_enabled": False,
            },
        )
        assert resp.status_code == 201, resp.text
        cp_id = resp.json()["id"]

        get_resp = client.get(f"/api/chat-providers/{cp_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["conversation_memory_enabled"] is False

    def test_create_honours_custom_fraction(
        self, client: TestClient, created_llm_provider: dict[str, Any]
    ) -> None:
        resp = client.post(
            "/api/chat-providers",
            json={
                "name": "CP-TIGHT",
                "llm_provider_id": created_llm_provider["id"],
                "execution_mode": "direct",
                "conversation_memory_fraction": 0.15,
            },
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["conversation_memory_fraction"] == 0.15

    def test_update_toggles_enabled_flag(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        """PATCH toggles conversation_memory_enabled, GET reflects the change."""
        cp_id = created_provider["id"]
        # Provider was created with the default (True); flip it off.
        put_resp = client.put(
            f"/api/chat-providers/{cp_id}",
            json={"conversation_memory_enabled": False},
        )
        assert put_resp.status_code == 200, put_resp.text
        assert put_resp.json()["conversation_memory_enabled"] is False

        # Flip it back on.
        put_resp = client.put(
            f"/api/chat-providers/{cp_id}",
            json={"conversation_memory_enabled": True},
        )
        assert put_resp.status_code == 200, put_resp.text
        assert put_resp.json()["conversation_memory_enabled"] is True

    def test_update_fraction_bounds_rejected(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        """PATCH with fraction > 0.9 is a 422 validation error."""
        cp_id = created_provider["id"]
        resp = client.put(
            f"/api/chat-providers/{cp_id}",
            json={"conversation_memory_fraction": 1.5},
        )
        assert resp.status_code == 422

    def test_create_fraction_bounds_rejected(
        self, client: TestClient, created_llm_provider: dict[str, Any]
    ) -> None:
        """POST with fraction > 0.9 is a 422 validation error."""
        resp = client.post(
            "/api/chat-providers",
            json={
                "name": "CP-BAD-FRAC",
                "llm_provider_id": created_llm_provider["id"],
                "execution_mode": "direct",
                "conversation_memory_fraction": -0.1,
            },
        )
        assert resp.status_code == 422

    def test_legacy_config_without_field_defaults_to_enabled(self) -> None:
        """A JSON config written before this commit must load with memory on.

        This protects users who upgrade the server while an older
        ``~/.rlmkit/config.json`` exists on disk.  Pydantic fills the
        default; no migration script required.
        """
        from rlmstudio.server.models import ChatProviderConfig

        legacy_payload = {
            "id": "legacy-cp",
            "name": "LEGACY-CP",
            "llm_provider_id": "some-uuid",
            "execution_mode": "direct",
        }
        cp = ChatProviderConfig.model_validate(legacy_payload)
        assert cp.conversation_memory_enabled is True
        assert cp.conversation_memory_fraction == 0.30

    def test_model_dump_round_trip_preserves_field(self) -> None:
        """save_config writes model_dump(); load reads model_validate().

        Pins that the full save→load cycle preserves a non-default
        value, which is the actual contract ``AppState.save_config()``
        relies on.
        """
        from rlmstudio.server.models import ChatProviderConfig

        original = ChatProviderConfig(
            id="cp-1",
            name="TEST",
            llm_provider_id="lp-1",
            execution_mode="rlm",
            conversation_memory_enabled=False,
            conversation_memory_fraction=0.10,
        )
        dumped = original.model_dump()
        assert dumped["conversation_memory_enabled"] is False
        assert dumped["conversation_memory_fraction"] == 0.10

        reloaded = ChatProviderConfig.model_validate(dumped)
        assert reloaded.conversation_memory_enabled is False
        assert reloaded.conversation_memory_fraction == 0.10

    def test_fraction_boundary_values_accepted(self) -> None:
        """0.0 and 0.9 are both inclusive boundaries."""
        from rlmstudio.server.models import ChatProviderConfig

        zero = ChatProviderConfig(
            id="cp-0",
            name="ZERO",
            llm_provider_id="lp",
            conversation_memory_fraction=0.0,
        )
        assert zero.conversation_memory_fraction == 0.0

        nine = ChatProviderConfig(
            id="cp-9",
            name="NINE",
            llm_provider_id="lp",
            conversation_memory_fraction=0.9,
        )
        assert nine.conversation_memory_fraction == 0.9

    def test_create_rlm_with_small_context_window_returns_warning(
        self, client: TestClient, created_llm_provider: dict[str, Any]
    ) -> None:
        """RLM mode + small context window → response includes warning."""
        # Set the LLM Provider's context_window to something tiny
        from rlmstudio.server.dependencies import get_state

        state = get_state()
        lp = state.get_llm_provider(created_llm_provider["id"])
        assert lp is not None
        lp.context_window = 4096  # too small for RLM

        resp = client.post(
            "/api/chat-providers",
            json={
                "name": "CP-RLM-SMALL-CTX",
                "llm_provider_id": created_llm_provider["id"],
                "execution_mode": "rlm",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "context_window_warning" in data
        assert "too small for RLM" in data["context_window_warning"]

    def test_create_direct_with_small_context_window_no_warning(
        self, client: TestClient, created_llm_provider: dict[str, Any]
    ) -> None:
        """Direct mode + small context window → no warning."""
        from rlmstudio.server.dependencies import get_state

        state = get_state()
        lp = state.get_llm_provider(created_llm_provider["id"])
        assert lp is not None
        lp.context_window = 4096

        resp = client.post(
            "/api/chat-providers",
            json={
                "name": "CP-DIRECT-SMALL-CTX",
                "llm_provider_id": created_llm_provider["id"],
                "execution_mode": "direct",
            },
        )
        assert resp.status_code == 201
        assert "context_window_warning" not in resp.json()

    def test_create_rlm_unknown_context_window_no_warning(
        self, client: TestClient, created_llm_provider: dict[str, Any]
    ) -> None:
        """Unknown context window → no warning (can't validate)."""
        from rlmstudio.server.dependencies import get_state

        state = get_state()
        lp = state.get_llm_provider(created_llm_provider["id"])
        assert lp is not None
        lp.context_window = None  # auto-discover, unknown at creation time

        resp = client.post(
            "/api/chat-providers",
            json={
                "name": "CP-RLM-UNKNOWN-CTX",
                "llm_provider_id": created_llm_provider["id"],
                "execution_mode": "rlm",
            },
        )
        assert resp.status_code == 201
        assert "context_window_warning" not in resp.json()

    def test_create_rlm_large_context_window_no_warning(
        self, client: TestClient, created_llm_provider: dict[str, Any]
    ) -> None:
        """RLM mode + large context window → no warning."""
        from rlmstudio.server.dependencies import get_state

        state = get_state()
        lp = state.get_llm_provider(created_llm_provider["id"])
        assert lp is not None
        lp.context_window = 32768

        resp = client.post(
            "/api/chat-providers",
            json={
                "name": "CP-RLM-LARGE-CTX",
                "llm_provider_id": created_llm_provider["id"],
                "execution_mode": "rlm",
            },
        )
        assert resp.status_code == 201
        assert "context_window_warning" not in resp.json()

    def test_update_to_rlm_with_small_context_returns_warning(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        """Updating execution_mode to rlm + small context → warning on update."""
        from rlmstudio.server.dependencies import get_state

        state = get_state()
        cp = state.get_chat_provider(created_provider["id"])
        assert cp is not None
        lp = state.get_llm_provider(cp.llm_provider_id) if cp.llm_provider_id else None
        if lp:
            lp.context_window = 4096

        resp = client.put(
            f"/api/chat-providers/{created_provider['id']}",
            json={"execution_mode": "rlm"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "context_window_warning" in data

    def test_fraction_rejects_bool(
        self, client: TestClient, created_llm_provider: dict[str, Any]
    ) -> None:
        """JSON false/true must not be silently coerced to 0.0/1.0."""
        resp = client.post(
            "/api/chat-providers",
            json={
                "name": "CP-BOOL-FRAC",
                "llm_provider_id": created_llm_provider["id"],
                "execution_mode": "direct",
                "conversation_memory_fraction": False,
            },
        )
        assert resp.status_code == 422
