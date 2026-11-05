"""The LLM Provider list says whether a key is stored, never the key itself."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from rlmstudio.server.app import create_app
from rlmstudio.server.dependencies import get_state, reset_state
from rlmstudio.server.models import LLMProviderConfig
from rlmstudio.ui.services.secret_store import FileSecretStore, KeyringSecretStore

SECRET = "sk-stored-key-for-the-edit-form-test"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    # The suite's state dir is a temp dir; keep the OS keyring out of it.
    monkeypatch.setattr(KeyringSecretStore, "is_available", staticmethod(lambda: False))
    reset_state()
    state = get_state()
    for lp_id, backend in (("with-key", "openai"), ("no-key", "openai"), ("local", "ollama")):
        state.config.llm_providers.append(
            LLMProviderConfig(id=lp_id, name=lp_id, backend=backend, model="m")
        )
    FileSecretStore().set("llm_provider:with-key", SECRET)
    yield TestClient(create_app())
    FileSecretStore().delete("llm_provider:with-key")


def test_list_and_get_say_whether_a_key_is_stored(client: TestClient) -> None:
    listed = {lp["id"]: lp for lp in client.get("/api/llm-providers").json()}

    assert listed["with-key"]["api_key_stored"] is True
    assert listed["no-key"]["api_key_stored"] is False
    assert listed["local"]["api_key_stored"] is False
    assert client.get("/api/llm-providers/with-key").json()["api_key_stored"] is True
    assert SECRET not in client.get("/api/llm-providers").text


def test_the_flag_is_not_written_to_the_config(client: TestClient) -> None:
    client.get("/api/llm-providers")
    dumped = get_state().config.model_dump()
    assert all("api_key_stored" not in lp for lp in dumped["llm_providers"])
