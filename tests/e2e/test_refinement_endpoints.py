"""Phase B prompt-refinement routes (#78): source runs, proposal, history.

Spec FR-2, FR-3, FR-4 and acceptance criterion 1 (proposal never writes the
profile).  The proposal model is an in-memory fake; no provider is called.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rlmstudio.application.dto import LLMResponseDTO
from rlmstudio.application.refinement_constants import (
    REFINEMENT_MAX_SOURCE_RUNS,
    REFINEMENT_STATUS_PROPOSAL_READY,
)
from rlmstudio.server.dependencies import AppState, get_state
from rlmstudio.server.models import ChatProviderConfig
from rlmstudio.telemetry.store import content_sha256
from rlmstudio.ui.services.profile_store import SYSTEM_PROMPT_TEMPLATES

PATH = "/api/profiles/{pid}/refinements"
DOC = "The quarterly report names Harbour Platform as the launch."
DEGRADED = "⚠️ Partial answer: ran out of steps"
GOOD_REPLY = '{"candidate": "Read the document index first.", "rationale": "Runs skipped it."}'


@dataclass
class FakeLLM:
    reply: str = GOOD_REPLY
    error: Exception | None = None
    prompts: list[str] = field(default_factory=list)

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        self.prompts.append(messages[-1]["content"])
        if self.error is not None:
            raise self.error
        return LLMResponseDTO(
            content=self.reply, model="fake-model", input_tokens=50, output_tokens=9
        )


@dataclass
class Setup:
    state: AppState
    profile_id: str
    llm: FakeLLM
    adapter_calls: list[str]
    saves: list[str]


def _record(state: AppState, run_id: str, created_at: float, **kw: Any) -> None:
    values: dict[str, Any] = {
        "run_id": run_id,
        "created_at": created_at,
        "mode": "rlm",
        "query": "Which platform launches?",
        "answer": "",
        "success": False,
        "error": "wall-clock budget exhausted",
        "outcome_category": "timeout",
        "chat_provider_id": "cp-1",
        "chat_provider_name": "RLM-TEST",
        "content_sha256": content_sha256(DOC),
    }
    values.update(kw)
    state.telemetry.record_run(**values)
    state.telemetry.record_step(run_id=run_id, step_index=0, action_type="inspect")


@pytest.fixture
def setup(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> Setup:
    resp = client.post(
        "/api/profiles",
        json={"name": "Tuned RLM", "strategy": "rlm", "prompt_template_name": "Default"},
    )
    assert resp.status_code == 201
    pid = resp.json()["id"]
    state = get_state()
    for cp_id in ("cp-1", "cp-2"):
        state.config.chat_providers.append(
            ChatProviderConfig(
                id=cp_id,
                name=f"RLM-{cp_id}",
                llm_provider="openai",
                llm_model="gpt-4o",
                profile_id=pid,
                execution_mode="rlm",
            )
        )
    _record(state, "old", 1.0, profile_id=pid)
    _record(state, "new", 2.0, profile_id=pid, success=True, answer=DEGRADED, error=None)
    _record(state, "legacy", 3.0, profile_id=pid, content_sha256=None)
    _record(state, "foreign", 4.0, profile_id="someone-else")
    _record(state, "direct-run", 5.0, profile_id=pid, mode="direct")

    llm = FakeLLM()
    adapter_calls: list[str] = []

    def fake_adapter(cp_id: str, num_retries: int | None = None) -> FakeLLM:
        adapter_calls.append(cp_id)
        return llm

    saves: list[str] = []
    monkeypatch.setattr(state, "create_llm_adapter_for_chat_provider", fake_adapter)
    monkeypatch.setattr(state, "save_config", lambda: saves.append("save_config"))
    monkeypatch.setattr(state, "save_config_checked", lambda: saves.append("checked"))
    return Setup(state, pid, llm, adapter_calls, saves)


def _propose(client: TestClient, s: Setup, **body: Any) -> Any:
    payload: dict[str, Any] = {"source_run_ids": ["old", "new"], **body}
    return client.post(PATH.format(pid=s.profile_id), json=payload)


# ---------------------------------------------------------------------------
# Source runs (FR-2)
# ---------------------------------------------------------------------------


class TestSourceRuns:
    def test_lists_only_hashed_rlm_runs_of_the_profile_newest_first(
        self, client: TestClient, setup: Setup
    ) -> None:
        resp = client.get(PATH.format(pid=setup.profile_id) + "/source-runs")
        assert resp.status_code == 200
        rows = resp.json()
        assert [r["id"] for r in rows] == ["new", "old"]
        assert rows[0]["degraded"] is True
        assert rows[1]["degraded"] is False
        assert rows[1]["outcome_category"] == "timeout"
        assert rows[0]["content_sha256"] == content_sha256(DOC)

    def test_unknown_profile_is_404(self, client: TestClient) -> None:
        assert client.get(PATH.format(pid="nope") + "/source-runs").status_code == 404

    def test_limit_is_validated(self, client: TestClient, setup: Setup) -> None:
        url = PATH.format(pid=setup.profile_id) + "/source-runs?limit=0"
        assert client.get(url).status_code == 422


# ---------------------------------------------------------------------------
# Proposal (FR-3, FR-4, AC-1)
# ---------------------------------------------------------------------------


class TestCreateProposal:
    def test_proposal_is_stored_and_profile_untouched(
        self, client: TestClient, setup: Setup
    ) -> None:
        before = setup.state.find_profile(setup.profile_id)
        assert before is not None
        prompts_before = dict(before.system_prompts)

        resp = _propose(client, setup)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["status"] == REFINEMENT_STATUS_PROPOSAL_READY
        proposal = body["proposal"]

        template_rlm = SYSTEM_PROMPT_TEMPLATES["Default"]["rlm"]
        assert proposal["source_run_ids"] == ["new", "old"]
        assert proposal["source_content_sha256"] == {
            "old": content_sha256(DOC),
            "new": content_sha256(DOC),
        }
        assert proposal["base"]["resolved"] == template_rlm
        assert proposal["base"]["rlm_key_present"] is False
        assert proposal["base"]["rlm_key_value"] is None
        assert proposal["base"]["template_name"] == "Default"
        assert proposal["base"]["sha256"] == hashlib.sha256(template_rlm.encode()).hexdigest()
        assert proposal["candidate"] == "Read the document index first."
        assert proposal["rationale"] == "Runs skipped it."
        assert "+Read the document index first." in proposal["diff"]
        assert proposal["prompt_version"] == "1.0"
        assert proposal["provider"] == "openai"
        assert proposal["model"] == "fake-model"
        evidence = proposal["evidence"]
        assert evidence["included_run_ids"] == ["new", "old"]
        assert evidence["sha256"] == hashlib.sha256(evidence["text"].encode()).hexdigest()

        # One call, through the newest run's Chat Provider, with the evidence.
        assert setup.adapter_calls == ["cp-1"]
        assert len(setup.llm.prompts) == 1
        assert evidence["text"] in setup.llm.prompts[0]
        assert template_rlm in setup.llm.prompts[0]

        # AC-1: proposing writes nothing to the profile or the config.
        assert setup.saves == []
        after = setup.state.find_profile(setup.profile_id)
        assert after is not None
        assert after.system_prompts == prompts_before

    def test_custom_prompt_state_is_recorded_exactly(
        self, client: TestClient, setup: Setup
    ) -> None:
        profile = setup.state.find_profile(setup.profile_id)
        assert profile is not None
        profile.system_prompts = {"rlm": "Be thorough."}
        resp = _propose(client, setup)
        base = resp.json()["proposal"]["base"]
        assert base["rlm_key_present"] is True
        assert base["rlm_key_value"] == "Be thorough."
        assert base["resolved"] == "Be thorough."

    def test_explicit_chat_provider_wins(self, client: TestClient, setup: Setup) -> None:
        resp = _propose(client, setup, proposal_chat_provider_id="cp-2")
        assert resp.status_code == 201
        assert setup.adapter_calls == ["cp-2"]
        assert resp.json()["proposal"]["proposal_chat_provider_id"] == "cp-2"

    @pytest.mark.parametrize(
        ("run_ids", "message"),
        [
            (["missing"], "missing: not found"),
            (["foreign"], "foreign: not a run of this profile"),
            (["legacy"], "legacy: recorded before document hashing"),
            (["direct-run"], "direct-run: not an RLM run (direct)"),
        ],
    )
    def test_unusable_source_runs_are_400(
        self, client: TestClient, setup: Setup, run_ids: list[str], message: str
    ) -> None:
        resp = _propose(client, setup, source_run_ids=run_ids)
        assert resp.status_code == 400
        assert message in resp.json()["error"]["message"]
        assert setup.llm.prompts == []

    def test_request_bounds_are_422(self, client: TestClient, setup: Setup) -> None:
        assert _propose(client, setup, source_run_ids=[]).status_code == 422
        too_many = [f"r{i}" for i in range(REFINEMENT_MAX_SOURCE_RUNS + 1)]
        assert _propose(client, setup, source_run_ids=too_many).status_code == 422

    def test_builtin_profile_is_400(self, client: TestClient, setup: Setup) -> None:
        resp = client.post(PATH.format(pid="builtin-fast"), json={"source_run_ids": ["old"]})
        assert resp.status_code == 400
        assert "Built-in" in resp.json()["error"]["message"]

    def test_unknown_chat_provider_is_400(self, client: TestClient, setup: Setup) -> None:
        resp = _propose(client, setup, proposal_chat_provider_id="cp-missing")
        assert resp.status_code == 400
        assert setup.llm.prompts == []

    def test_provider_failure_is_502_redacted_and_stores_nothing(
        self, client: TestClient, setup: Setup
    ) -> None:
        secret = "sk-" + "a" * 30
        setup.llm.error = RuntimeError(f"401 invalid key {secret}")
        resp = _propose(client, setup)
        assert resp.status_code == 502
        assert secret not in resp.text
        assert setup.state.telemetry.list_refinements(setup.profile_id) == []

    @pytest.mark.parametrize(
        "reply",
        [
            "I cannot help with that.",
            '{"candidate": "Read the document index first.", "rationale": ""}',
        ],
    )
    def test_unusable_reply_is_502_and_stores_nothing(
        self, client: TestClient, setup: Setup, reply: str
    ) -> None:
        setup.llm.reply = reply
        resp = _propose(client, setup)
        assert resp.status_code == 502
        assert "unusable reply" in resp.json()["error"]["message"]
        assert setup.state.telemetry.list_refinements(setup.profile_id) == []


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


class TestHistory:
    def test_list_and_get(self, client: TestClient, setup: Setup) -> None:
        created = _propose(client, setup).json()
        listed = client.get(PATH.format(pid=setup.profile_id)).json()
        assert [r["id"] for r in listed] == [created["id"]]
        one = client.get(PATH.format(pid=setup.profile_id) + f"/{created['id']}")
        assert one.status_code == 200
        assert one.json() == created

    def test_get_under_another_profile_is_404(self, client: TestClient, setup: Setup) -> None:
        created = _propose(client, setup).json()
        resp = client.get(PATH.format(pid="builtin-fast") + f"/{created['id']}")
        assert resp.status_code == 404

    def test_list_for_unknown_profile_is_404(self, client: TestClient) -> None:
        assert client.get(PATH.format(pid="nope")).status_code == 404
