"""CMP2-05 (backend): the evaluation summary leaves out LLM Tuner providers.

The Tuner runs each slot on a temporary Chat Provider named with
``EPHEMERAL_CP_PREFIX``; once it is gone only its raw id remained.  The
Dashboard's judge cards then showed those ids and the recommendation could
point at a Tuner run, while the other Dashboard metrics already left them out.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rlmstudio.application.sandbox_vars import EPHEMERAL_CP_PREFIX, MODE_DIRECT
from rlmstudio.server.app import app
from rlmstudio.server.dependencies import SessionRecord, get_state, reset_state
from rlmstudio.server.models import ChatProviderConfig, RuntimeSettings

_SESSION = "tuner-session"
_REAL_SCORE = 3.0
_TUNER_SCORE = 5.0  # higher, so an unfiltered summary recommends a Tuner run


@pytest.fixture(autouse=True)
def _clean_state() -> Generator[None, None, None]:
    reset_state()
    yield
    reset_state()


def _chat_provider(name: str, *, ephemeral: bool = False) -> ChatProviderConfig:
    now = datetime.now(timezone.utc)
    return ChatProviderConfig(
        id=str(uuid.uuid4()),
        name=name,
        llm_provider="ollama",
        llm_model="llama3.2",
        execution_mode=MODE_DIRECT,
        runtime_settings=RuntimeSettings(),
        ephemeral=ephemeral,
        created_at=now,
        updated_at=now,
    )


def _answer(cp_id: str, cp_name: str) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "role": "assistant",
        "content": "answer",
        "execution_id": str(uuid.uuid4()),
        "chat_provider_id": cp_id,
        "chat_provider_name": cp_name,
    }


def _judge(cp_id: str, execution_id: str, score: float) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "execution_id": execution_id,
        "session_id": _SESSION,
        "chat_provider_id": cp_id,
        "judge_provider_id": "judge",
        "dimensions": {},
        "overall_score": score,
        "reasoning": "",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def test_summary_and_recommendation_leave_out_tuner_providers() -> None:
    state = get_state()
    real = _chat_provider("Chat Qwen")
    live_tuner = _chat_provider(f"{EPHEMERAL_CP_PREFIX}Qwen · RLM", ephemeral=True)
    state.config.chat_providers.extend([real, live_tuner])
    gone_tuner_id = str(uuid.uuid4())  # its ephemeral CP no longer exists

    messages = [
        _answer(real.id, real.name),
        _answer(live_tuner.id, live_tuner.name),
        _answer(gone_tuner_id, f"{EPHEMERAL_CP_PREFIX}Llama · Direct"),
    ]
    now = datetime.now(timezone.utc)
    state.sessions[_SESSION] = SessionRecord(
        id=_SESSION, name="S", created_at=now, updated_at=now, messages=messages
    )
    state.evaluations["judge_scores"].extend(
        [
            _judge(real.id, messages[0]["execution_id"], _REAL_SCORE),
            _judge(live_tuner.id, messages[1]["execution_id"], _TUNER_SCORE),
            _judge(gone_tuner_id, messages[2]["execution_id"], _TUNER_SCORE),
        ]
    )

    body = TestClient(app).get(f"/api/evaluations/{_SESSION}/summary").json()

    assert set(body["by_chat_provider"]) == {real.id}
    assert body["recommendation"] == real.id
