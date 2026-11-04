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
    CONTEXT_WINDOW_MAX_TOKENS,
    MAX_OUTPUT_TOKENS_LIMIT,
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
