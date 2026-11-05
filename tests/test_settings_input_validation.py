"""Settings values a user types are validated at the API boundary (M12).

The Settings UI validates first; these tests pin the server-side half, so a
bad value sent any other way is a 422 instead of being stored. Stored config
models stay unconstrained, so a config file written before the bounds still
loads.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from rlmstudio.server.app import create_app
from rlmstudio.server.dependencies import reset_state
from rlmstudio.server.models import (
    BUDGET_MAX_COST_USD_LIMIT,
    BUDGET_MAX_REPEAT_LIMIT,
    BUDGET_MAX_STEPS_LIMIT,
    BUDGET_MAX_TIME_SECONDS,
    BUDGET_MAX_TOKENS_LIMIT,
    CONTEXT_WINDOW_MAX_TOKENS,
    CONTEXT_WINDOW_MIN_TOKENS,
    MAX_OUTPUT_TOKENS_LIMIT,
    RAG_MAX_CHUNK_SIZE,
    RAG_MAX_TOP_K,
    RUNTIME_MAX_TIMEOUT_SECONDS,
    BudgetConfig,
    ConfigResponse,
    RunProfile,
)

VALID_BUDGET: dict[str, Any] = {
    "max_steps": 16,
    "max_tokens": 50000,
    "max_cost_usd": 2.0,
    "max_time_seconds": 600,
    "max_recursion_depth": 5,
    "repeat_limit": 2,
    "nudge_at_fraction": 0.4,
}
VALID_RUNTIME: dict[str, Any] = {
    "temperature": 0.7,
    "top_p": 1.0,
    "max_output_tokens": 4096,
    "timeout_seconds": 120,
}


@pytest.fixture
def client() -> TestClient:
    reset_state()
    return TestClient(create_app())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_tokens", -5),
        ("max_tokens", 0),
        ("max_cost_usd", -1.0),
        ("max_steps", 0),
        ("max_time_seconds", -3),
        ("max_recursion_depth", -1),
        # Review P1: the profile form now edits this field; the API agrees on 0..100.
        ("max_recursion_depth", 101),
        ("repeat_limit", 0),
        ("nudge_at_fraction", 1.5),
    ],
)
def test_budget_rejects_out_of_range_values(client: TestClient, field: str, value: float) -> None:
    resp = client.put("/api/config", json={"budget": {**VALID_BUDGET, field: value}})
    assert resp.status_code == 422
    assert client.get("/api/config").json()["budget"][field] != value


def test_budget_accepts_valid_values(client: TestClient) -> None:
    resp = client.put("/api/config", json={"budget": {**VALID_BUDGET, "max_cost_usd": 0.0}})
    assert resp.status_code == 200
    assert resp.json()["budget"]["max_cost_usd"] == 0.0


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("runtime_settings", "max_output_tokens", 99_999_999_999_999),
        ("runtime_settings", "max_output_tokens", 0),
        ("runtime_settings", "temperature", 3.0),
        ("runtime_settings", "top_p", -0.1),
        ("runtime_settings", "timeout_seconds", 1),
        ("budget", "max_tokens", -5),
    ],
)
def test_profile_rejects_out_of_range_values(
    client: TestClient, section: str, field: str, value: float
) -> None:
    base = {"runtime_settings": VALID_RUNTIME, "budget": VALID_BUDGET}
    body = {"name": "p", **base, section: {**base[section], field: value}}
    assert client.post("/api/profiles", json=body).status_code == 422

    created = client.post("/api/profiles", json={"name": "ok", **base})
    assert created.status_code == 201
    update = {section: {**base[section], field: value}}
    resp = client.put(f"/api/profiles/{created.json()['id']}", json=update)
    assert resp.status_code == 422


def test_profile_accepts_the_largest_output_limit(client: TestClient) -> None:
    runtime = {**VALID_RUNTIME, "max_output_tokens": MAX_OUTPUT_TOKENS_LIMIT}
    resp = client.post("/api/profiles", json={"name": "big", "runtime_settings": runtime})
    assert resp.status_code == 201


@pytest.mark.parametrize("value", [-5, 0, CONTEXT_WINDOW_MAX_TOKENS + 1])
def test_llm_provider_create_rejects_bad_context_window(client: TestClient, value: int) -> None:
    body = {"name": "x", "backend": "ollama", "model": "llama3.2", "context_window": value}
    assert client.post("/api/llm-providers", json=body).status_code == 422
    assert client.get("/api/llm-providers").json() == []


@pytest.mark.parametrize("value", [-5, CONTEXT_WINDOW_MAX_TOKENS + 1])
def test_llm_provider_update_rejects_bad_context_window(client: TestClient, value: int) -> None:
    resp = client.put("/api/llm-providers/missing", json={"context_window": value})
    assert resp.status_code == 422


def test_stored_models_still_load_old_out_of_range_values() -> None:
    """A config saved before the bounds must not be lost on load."""
    config = ConfigResponse.model_validate({"budget": {**VALID_BUDGET, "max_tokens": -5}})
    assert config.budget.max_tokens == -5
    profile = RunProfile.model_validate(
        {
            "id": "p",
            "name": "p",
            "runtime_settings": {**VALID_RUNTIME, "max_output_tokens": 99_999_999_999_999},
            "budget": BudgetConfig(**VALID_BUDGET).model_dump(),
        }
    )
    assert profile.runtime_settings.max_output_tokens == 99_999_999_999_999


# --- SET2-02 / SET2-03 / SET2-04 -------------------------------------------

_HUGE = 99_999_999_999_999
VALID_RAG: dict[str, Any] = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k": 5,
    "embedding_model": "text-embedding-3-small",
}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chunk_size", 0),
        ("chunk_size", RAG_MAX_CHUNK_SIZE + 1),
        ("chunk_overlap", -1),
        ("chunk_overlap", 1000),  # equal to chunk_size
        ("top_k", 0),
        ("top_k", RAG_MAX_TOP_K + 1),
        ("embedding_model", "   "),
    ],
)
def test_chat_provider_rag_config_is_bounded(client: TestClient, field: str, value: Any) -> None:
    rag = {**VALID_RAG, field: value}
    create = {"name": "r", "llm_provider_id": "x", "execution_mode": "rag", "rag_config": rag}
    assert client.post("/api/chat-providers", json=create).status_code == 422
    resp = client.put("/api/chat-providers/missing", json={"rag_config": rag})
    assert resp.status_code == 422


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_tokens", BUDGET_MAX_TOKENS_LIMIT + 1),
        ("max_tokens", _HUGE),
        ("max_cost_usd", BUDGET_MAX_COST_USD_LIMIT + 1),
        ("max_cost_usd", _HUGE),
        ("max_cost_usd", None),
        ("max_cost_usd", ""),
        ("max_steps", BUDGET_MAX_STEPS_LIMIT + 1),
        ("max_time_seconds", BUDGET_MAX_TIME_SECONDS + 1),
        ("repeat_limit", BUDGET_MAX_REPEAT_LIMIT + 1),
    ],
)
def test_budget_has_upper_bounds_and_no_blank_cost(
    client: TestClient, field: str, value: Any
) -> None:
    """A blank Max Cost is refused, not saved as $0."""
    before = client.get("/api/config").json()["budget"][field]
    resp = client.put("/api/config", json={"budget": {**VALID_BUDGET, field: value}})
    assert resp.status_code == 422
    assert client.get("/api/config").json()["budget"][field] == before

    base = {"runtime_settings": VALID_RUNTIME, "budget": VALID_BUDGET}
    body = {"name": "p", **base, "budget": {**VALID_BUDGET, field: value}}
    assert client.post("/api/profiles", json=body).status_code == 422


def test_profile_timeout_has_an_upper_bound(client: TestClient) -> None:
    runtime = {**VALID_RUNTIME, "timeout_seconds": RUNTIME_MAX_TIMEOUT_SECONDS + 1}
    resp = client.post("/api/profiles", json={"name": "t", "runtime_settings": runtime})
    assert resp.status_code == 422


def test_the_upper_bounds_themselves_are_accepted(client: TestClient) -> None:
    budget = {
        **VALID_BUDGET,
        "max_tokens": BUDGET_MAX_TOKENS_LIMIT,
        "max_cost_usd": BUDGET_MAX_COST_USD_LIMIT,
        "max_steps": BUDGET_MAX_STEPS_LIMIT,
        "max_time_seconds": BUDGET_MAX_TIME_SECONDS,
        "repeat_limit": BUDGET_MAX_REPEAT_LIMIT,
    }
    runtime = {**VALID_RUNTIME, "timeout_seconds": RUNTIME_MAX_TIMEOUT_SECONDS}
    assert client.put("/api/config", json={"budget": budget}).status_code == 200
    body = {"name": "max", "runtime_settings": runtime, "budget": budget}
    assert client.post("/api/profiles", json=body).status_code == 201


def test_context_window_needs_a_realistic_minimum(client: TestClient) -> None:
    """A context window of 1 left the Chat Provider with max tokens 0."""
    body = {"name": "x", "backend": "ollama", "model": "llama3.2", "context_window": 1}
    assert client.post("/api/llm-providers", json=body).status_code == 422
    too_small = {"context_window": CONTEXT_WINDOW_MIN_TOKENS - 1}
    assert client.put("/api/llm-providers/missing", json=too_small).status_code == 422
    # 0 still clears a manual override on update (validated, then 404 here).
    assert client.put("/api/llm-providers/missing", json={"context_window": 0}).status_code == 404
