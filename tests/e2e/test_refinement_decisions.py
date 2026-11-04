"""Phase D decisions for prompt refinement (#78): accept, reject, rollback, export.

Spec FR-9..FR-11a, FR-15 and acceptance criteria 1-10 (AC-3, AC-3a and AC-8a
in particular).  Config-save faults use the real checked save writing to a
temporary state directory.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rlmstudio.application.dto import LLMResponseDTO
from rlmstudio.application.refinement_constants import (
    REFINEMENT_STATUS_ACCEPTED,
    REFINEMENT_STATUS_EVALUATED,
    REFINEMENT_STATUS_EVALUATING,
    REFINEMENT_STATUS_FAILED,
    REFINEMENT_STATUS_PROPOSAL_READY,
    REFINEMENT_STATUS_REJECTED,
    REFINEMENT_STATUS_ROLLED_BACK,
)
from rlmstudio.application.sandbox_vars import MODE_RLM
from rlmstudio.application.services.refinement import (
    BaseSupplementState,
    DecisionRecord,
    ProposalRecord,
    build_evidence_digest,
    supplement_sha256,
)
from rlmstudio.application.services.refinement_evaluation import (
    ArmResult,
    CaseRuns,
    EvaluationSetup,
    RepResult,
    summarise_evaluation,
)
from rlmstudio.server import dependencies, refinement_decisions
from rlmstudio.server.dependencies import AppState, get_state
from rlmstudio.server.models import ChatProviderConfig
from rlmstudio.server.routes.chat import _resolve_profile_prompt
from rlmstudio.telemetry.store import content_sha256
from rlmstudio.ui.services.profile_store import SYSTEM_PROMPT_TEMPLATES
from tests.test_prompt_refinement_foundations import _abort_trigger, _fault_patch

PATH = "/api/profiles/{pid}/refinements"
CANDIDATE = "Read the document index first, then grep before peeking."
TEMPLATE = "Default"
TEMPLATE_RLM = SYSTEM_PROMPT_TEMPLATES[TEMPLATE]["rlm"]


def _arm(passed: bool) -> ArmResult:
    return ArmResult(
        run_id="run",
        success=True,
        usable=True,
        outcome_category="success",
        passed=passed,
        judge_overall=None,
        cost=0.0,
        steps=1,
    )


def _evaluation(
    *, regression: bool = False, complete: bool = True, deterministic: bool = True
) -> dict[str, Any]:
    setup = EvaluationSetup(
        chat_provider_id="cp-1",
        provider="openai",
        model="gpt-4o",
        temperature=0.0,
        reps=2,
        dataset_name="t",
        dataset_version="1",
        case_ids=["a"],
        excluded_cases={},
        skipped_cases={} if complete else {"pub": "not fetched"},
        judge=None,
        started_at=1.0,
    )
    candidate_pass = not regression
    case = CaseRuns(
        case_id="a",
        deterministic=deterministic,
        content_sha256="h",
        reps=[
            RepResult(
                rep=i,
                baseline=_arm(True) if deterministic else _arm_judged(),
                candidate=_arm(candidate_pass) if deterministic else _arm_judged(),
            )
            for i in (1, 2)
        ],
    )
    return summarise_evaluation(setup, [case], finished_at=2.0).to_dict()


def _arm_judged() -> ArmResult:
    return ArmResult(
        run_id="run",
        success=True,
        usable=True,
        outcome_category="success",
        passed=None,
        judge_overall=4.0,
        cost=0.0,
        steps=1,
    )


def _proposal(profile_rlm: str | None) -> dict[str, Any]:
    resolved = profile_rlm if profile_rlm is not None else TEMPLATE_RLM
    return ProposalRecord(
        source_run_ids=["src"],
        source_content_sha256={"src": "x"},
        evidence=build_evidence_digest([]),
        proposal_chat_provider_id="cp-1",
        provider="openai",
        model="gpt-4o",
        prompt_version="1.0",
        prompt_sha256="p",
        input_tokens=1,
        output_tokens=1,
        base=BaseSupplementState(
            resolved=resolved,
            sha256=supplement_sha256(resolved),
            rlm_key_present=profile_rlm is not None,
            rlm_key_value=profile_rlm,
            template_name=TEMPLATE,
            base_prompt_version="2.3",
            base_prompt_sha256="b",
        ),
        candidate=CANDIDATE,
        rationale="why",
        diff="d",
    ).to_dict()


@dataclass
class Env:
    state: AppState
    profile_id: str
    config_path: Path
    saves: list[str]


@pytest.fixture
def env(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(dependencies, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(dependencies, "_CONFIG_FILE", config_path)
    resp = client.post(
        "/api/profiles",
        json={"name": "Tuned RLM", "strategy": "rlm", "prompt_template_name": TEMPLATE},
    )
    state = get_state()
    saves: list[str] = []
    real_checked = AppState.save_config_checked

    def counting_checked() -> None:
        saves.append("checked")
        real_checked(state)

    # reset_state() stubs the checked save; these tests need the real one,
    # writing into tmp_path, and a count of how often it ran.
    monkeypatch.setattr(state, "save_config_checked", counting_checked)
    monkeypatch.setattr(state, "save_config", lambda: saves.append("save_config"))
    return Env(state, resp.json()["id"], config_path, saves)


def _seed(
    e: Env,
    *,
    status: str = REFINEMENT_STATUS_EVALUATED,
    profile_rlm: str | None = None,
    **evaluation: Any,
) -> str:
    profile = e.state.find_profile(e.profile_id)
    assert profile is not None
    if profile_rlm is not None:
        profile.system_prompts = {MODE_RLM: profile_rlm}
    record = e.state.telemetry.create_refinement(
        profile_id=e.profile_id, status=status, proposal=_proposal(profile_rlm)
    )
    if status in (REFINEMENT_STATUS_EVALUATED, REFINEMENT_STATUS_ACCEPTED):
        with e.state.telemetry._lock:
            conn = e.state.telemetry._connect()
            conn.execute(
                "UPDATE refinements SET evaluation_json = ? WHERE id = ?",
                (json.dumps(_evaluation(**evaluation)), record.id),
            )
            conn.commit()
    return record.id


def _post(client: TestClient, e: Env, rid: str, action: str, **body: Any) -> Any:
    return client.post(PATH.format(pid=e.profile_id) + f"/{rid}/{action}", json=body)


def _profile_rlm(e: Env) -> str | None:
    profile = e.state.find_profile(e.profile_id)
    assert profile is not None
    return profile.system_prompts.get(MODE_RLM)


def _disk_rlm(e: Env) -> str | None:
    data = json.loads(e.config_path.read_text())
    (profile,) = [p for p in data["user_profiles"] if p["id"] == e.profile_id]
    value = profile["system_prompts"].get(MODE_RLM)
    return str(value) if value is not None else None


def _status(e: Env, rid: str) -> str:
    record = e.state.telemetry.get_refinement(rid)
    assert record is not None
    return record.status


# ---------------------------------------------------------------------------
# Accept (FR-9, AC-2, AC-3, AC-3a, AC-4, AC-8)
# ---------------------------------------------------------------------------


class TestAccept:
    def test_accept_writes_the_candidate_and_records_the_decision(
        self, client: TestClient, env: Env
    ) -> None:
        rid = _seed(env)
        resp = _post(client, env, rid, "accept", reason="looks right")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == REFINEMENT_STATUS_ACCEPTED
        decision = DecisionRecord.from_dict(body["decision"])
        assert decision.decision == REFINEMENT_STATUS_ACCEPTED
        assert decision.reason == "looks right"
        assert decision.regression is False
        assert _profile_rlm(env) == CANDIDATE
        assert _disk_rlm(env) == CANDIDATE
        assert env.saves == ["checked"]
        # AC-4: the proposal recorded that the template-backed profile had no key.
        assert body["proposal"]["base"]["rlm_key_present"] is False

    @pytest.mark.parametrize(
        ("seed", "status", "message"),
        [
            ({"status": REFINEMENT_STATUS_PROPOSAL_READY}, 409, "Only an evaluated"),
            ({"complete": False}, 409, "incomplete"),
            ({"deterministic": False}, 409, "no case with an expected answer"),
            ({"regression": True}, 400, "acknowledge_regression"),
        ],
    )
    def test_accept_gates(
        self, client: TestClient, env: Env, seed: dict[str, Any], status: int, message: str
    ) -> None:
        rid = _seed(env, **seed)
        resp = _post(client, env, rid, "accept")
        assert resp.status_code == status
        assert message in resp.json()["error"]["message"]
        assert _profile_rlm(env) is None
        assert env.saves == []

    def test_acknowledged_regression_is_accepted_and_recorded(
        self, client: TestClient, env: Env
    ) -> None:
        rid = _seed(env, regression=True)
        resp = _post(client, env, rid, "accept", acknowledge_regression=True)
        assert resp.status_code == 200
        decision = DecisionRecord.from_dict(resp.json()["decision"])
        assert (decision.regression, decision.acknowledge_regression) == (True, True)

    def test_changed_profile_prompt_is_409_and_writes_nothing(
        self, client: TestClient, env: Env
    ) -> None:
        rid = _seed(env)
        profile = env.state.find_profile(env.profile_id)
        assert profile is not None
        profile.system_prompts = {MODE_RLM: "Edited by hand after the proposal."}
        resp = _post(client, env, rid, "accept")
        assert resp.status_code == 409
        assert "changed after this proposal" in resp.json()["error"]["message"]
        assert _profile_rlm(env) == "Edited by hand after the proposal."
        assert _status(env, rid) == REFINEMENT_STATUS_EVALUATED

    @pytest.mark.parametrize("fault", ["write", "fsync", "replace"])
    def test_save_fault_restores_memory_and_keeps_status(
        self, client: TestClient, env: Env, fault: str
    ) -> None:
        rid = _seed(env)
        env.state.save_config_checked()  # a known-good file on disk first
        with _fault_patch(fault):
            resp = _post(client, env, rid, "accept")
        assert resp.status_code == 500
        assert "nothing was changed" in resp.json()["error"]["message"]
        assert _profile_rlm(env) is None
        assert _disk_rlm(env) is None
        assert _status(env, rid) == REFINEMENT_STATUS_EVALUATED

    def test_status_write_failure_after_save_is_a_named_split_brain(
        self, client: TestClient, env: Env
    ) -> None:
        rid = _seed(env)
        _abort_trigger(env.state.telemetry, "UPDATE")
        resp = _post(client, env, rid, "accept")
        assert resp.status_code == 500
        message = resp.json()["error"]["message"]
        assert "saved to the profile, but recording the accept failed" in message
        assert _profile_rlm(env) == CANDIDATE
        assert _disk_rlm(env) == CANDIDATE
        assert _status(env, rid) == REFINEMENT_STATUS_EVALUATED

    def test_builtin_profile_is_400(self, client: TestClient, env: Env) -> None:
        resp = client.post(PATH.format(pid="builtin-fast") + "/x/accept", json={})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Rollback (FR-11, AC-3, AC-3a, AC-4)
# ---------------------------------------------------------------------------


class TestRollback:
    def _accepted(self, client: TestClient, e: Env, profile_rlm: str | None = None) -> str:
        rid = _seed(e, profile_rlm=profile_rlm)
        assert _post(client, e, rid, "accept").status_code == 200
        return rid

    def test_rollback_of_a_template_profile_removes_the_key(
        self, client: TestClient, env: Env
    ) -> None:
        rid = self._accepted(client, env)
        resp = _post(client, env, rid, "rollback")
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == REFINEMENT_STATUS_ROLLED_BACK
        assert _profile_rlm(env) is None
        assert _disk_rlm(env) is None
        profile = env.state.find_profile(env.profile_id)
        assert profile is not None
        assert _resolve_profile_prompt(profile, MODE_RLM) == TEMPLATE_RLM
        decision = DecisionRecord.from_dict(resp.json()["decision"])
        assert decision.decision == REFINEMENT_STATUS_ACCEPTED
        assert decision.rollback is not None
        assert decision.rollback.restored_rlm_key_present is False

    def test_rollback_of_a_custom_profile_restores_the_exact_text(
        self, client: TestClient, env: Env
    ) -> None:
        rid = self._accepted(client, env, profile_rlm="Be thorough.")
        assert _post(client, env, rid, "rollback").status_code == 200
        assert _profile_rlm(env) == "Be thorough."
        assert _disk_rlm(env) == "Be thorough."

    def test_rollback_after_a_manual_edit_is_409(self, client: TestClient, env: Env) -> None:
        rid = self._accepted(client, env)
        profile = env.state.find_profile(env.profile_id)
        assert profile is not None
        profile.system_prompts = {MODE_RLM: "Hand edit"}
        resp = _post(client, env, rid, "rollback")
        assert resp.status_code == 409
        assert _profile_rlm(env) == "Hand edit"

    def test_rollback_needs_an_accepted_refinement(self, client: TestClient, env: Env) -> None:
        rid = _seed(env)
        assert _post(client, env, rid, "rollback").status_code == 409

    @pytest.mark.parametrize("fault", ["write", "fsync", "replace"])
    def test_save_fault_keeps_the_candidate_and_accepted_status(
        self, client: TestClient, env: Env, fault: str
    ) -> None:
        rid = self._accepted(client, env)
        with _fault_patch(fault):
            resp = _post(client, env, rid, "rollback")
        assert resp.status_code == 500
        assert _profile_rlm(env) == CANDIDATE
        assert _disk_rlm(env) == CANDIDATE
        assert _status(env, rid) == REFINEMENT_STATUS_ACCEPTED

    def test_status_write_failure_after_save_is_a_named_split_brain(
        self, client: TestClient, env: Env
    ) -> None:
        rid = self._accepted(client, env)
        _abort_trigger(env.state.telemetry, "UPDATE")
        resp = _post(client, env, rid, "rollback")
        assert resp.status_code == 500
        assert "recording the rollback failed" in resp.json()["error"]["message"]
        assert _profile_rlm(env) is None
        assert _disk_rlm(env) is None
        assert _status(env, rid) == REFINEMENT_STATUS_ACCEPTED


# ---------------------------------------------------------------------------
# Reject (FR-10, AC-1)
# ---------------------------------------------------------------------------


class TestReject:
    @pytest.mark.parametrize(
        "status",
        [REFINEMENT_STATUS_PROPOSAL_READY, REFINEMENT_STATUS_EVALUATED, REFINEMENT_STATUS_FAILED],
    )
    def test_reject_writes_nothing_to_the_profile(
        self, client: TestClient, env: Env, status: str
    ) -> None:
        rid = _seed(env, status=status)
        resp = _post(client, env, rid, "reject", reason="not convinced")
        assert resp.status_code == 200
        assert resp.json()["status"] == REFINEMENT_STATUS_REJECTED
        assert DecisionRecord.from_dict(resp.json()["decision"]).reason == "not convinced"
        assert env.saves == []
        assert _profile_rlm(env) is None

    @pytest.mark.parametrize("status", [REFINEMENT_STATUS_EVALUATING, REFINEMENT_STATUS_ACCEPTED])
    def test_reject_from_other_statuses_is_409(
        self, client: TestClient, env: Env, status: str
    ) -> None:
        rid = _seed(env, status=status)
        assert _post(client, env, rid, "reject").status_code == 409


# ---------------------------------------------------------------------------
# Serialization (FR-10a, AC-8a)
# ---------------------------------------------------------------------------


def test_reject_waits_for_an_in_flight_accept_then_fails(env: Env) -> None:
    rid = _seed(env)
    entered, release = threading.Event(), threading.Event()
    real_checked = env.state.save_config_checked

    def paused_save() -> None:
        real_checked()
        entered.set()  # saved; now hold the lock before the status write
        assert release.wait(10)

    env.state.save_config_checked = paused_save  # type: ignore[method-assign]
    outcomes: dict[str, Any] = {}

    def run(name: str, fn: Any) -> None:
        try:
            outcomes[name] = fn()
        except refinement_decisions.DecisionError as exc:
            outcomes[name] = exc

    accept = threading.Thread(
        target=run,
        args=(
            "accept",
            lambda: refinement_decisions.accept(
                env.state, env.profile_id, rid, acknowledge_regression=False, reason=None
            ),
        ),
    )
    reject = threading.Thread(
        target=run,
        args=(
            "reject",
            lambda: refinement_decisions.reject(env.state, env.profile_id, rid, reason=None),
        ),
    )
    accept.start()
    assert entered.wait(10)
    reject.start()
    time.sleep(0.2)
    assert reject.is_alive(), "reject must wait for the accept's lock"
    release.set()
    accept.join(10)
    reject.join(10)

    assert outcomes["accept"].status == REFINEMENT_STATUS_ACCEPTED
    assert isinstance(outcomes["reject"], refinement_decisions.DecisionConflictError)
    assert _status(env, rid) == REFINEMENT_STATUS_ACCEPTED
    assert _profile_rlm(env) == CANDIDATE


def test_profile_edit_waits_for_the_config_lock(client: TestClient, env: Env) -> None:
    done = threading.Event()
    statuses: list[int] = []

    def edit() -> None:
        resp = client.put(f"/api/profiles/{env.profile_id}", json={"description": "edited"})
        statuses.append(resp.status_code)
        done.set()

    with env.state._config_lock:
        worker = threading.Thread(target=edit)
        worker.start()
        assert not done.wait(0.3), "the profile edit must wait for the config lock"
    assert done.wait(10)
    worker.join(10)
    assert statuses == [200]
    profile = env.state.find_profile(env.profile_id)
    assert profile is not None
    assert profile.description == "edited"


# ---------------------------------------------------------------------------
# Export (FR-15)
# ---------------------------------------------------------------------------


def test_export_is_one_verbatim_line_per_refinement(client: TestClient, env: Env) -> None:
    first = _seed(env, status=REFINEMENT_STATUS_PROPOSAL_READY)
    second = _seed(env)
    resp = client.get(PATH.format(pid=env.profile_id) + "/export.jsonl")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(line) for line in resp.text.splitlines()]
    assert {line["id"] for line in lines} == {first, second}
    for line in lines:
        stored = env.state.telemetry.get_refinement(line["id"])
        assert stored is not None
        assert line["proposal"] == stored.proposal
        assert line["evaluation"] == stored.evaluation


# ---------------------------------------------------------------------------
# End to end (D5): propose -> evaluate -> accept -> run uses it -> rollback
# ---------------------------------------------------------------------------

E2E_DATASET = """
name: decisions-e2e
metadata: {version: "t1"}
cases:
  - id: alpha
    query: "What is the launch platform?"
    content: "Notes. The launch platform is Harbour."
    expected_answer: "Harbour"
    tags: [holdout]
"""


@dataclass
class FlowLLM:
    """Proposer when it sees the evidence block, otherwise a correct RLM arm.

    Records the system prompt of every non-proposal call, so a test can see
    which supplement a run actually sent to the model.
    """

    system_prompts: list[str] = field(default_factory=list)

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        text = "\n".join(m["content"] for m in messages)
        if "<<<EVIDENCE" not in text:
            self.system_prompts.append(messages[0]["content"])
        if "<<<EVIDENCE" in text:
            reply = json.dumps({"candidate": CANDIDATE, "rationale": "Runs skipped the index."})
        else:
            reply = "FINAL: Harbour"
        return LLMResponseDTO(content=reply, model="flow", input_tokens=5, output_tokens=5)

    def complete_stream(self, messages: list[dict[str, str]]) -> Any:
        yield self.complete(messages).content

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def get_pricing(self) -> dict[str, float]:
        return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}

    def get_completion_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0


def _chat_system_prompt(state: AppState, flow: FlowLLM, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run a real RLM chat through cp-1 and return the system prompt it sent.

    /api/chat runs the execution as a background task on the request's event
    loop.  A plain TestClient shuts that loop down after the response, which
    cancels a slow run, so this uses a context-managed client whose loop
    lives until the run finishes.  Its lifespan would copy a legacy state
    directory under the user's home, so that step is stubbed.
    """
    from rlmstudio.server import app as app_module

    monkeypatch.setattr(app_module, "migrate_legacy_state", lambda: None)
    before = len(flow.system_prompts)
    with TestClient(app_module.app) as persistent:
        resp = persistent.post(
            "/api/chat",
            json={
                "query": "What is the launch platform?",
                "content": "Notes. The launch platform is Harbour.",
                "chat_provider_id": "cp-1",
            },
        )
        assert resp.status_code in (200, 202), resp.text
        session_id = resp.json()["session_id"]
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            session = state.sessions.get(session_id)
            if session and any(m["role"] == "assistant" for m in session.messages):
                break
            time.sleep(0.05)
        else:
            raise AssertionError("chat run did not finish")
    assert len(flow.system_prompts) > before
    return flow.system_prompts[before]


def test_full_flow_from_proposal_to_rollback(
    client: TestClient, env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rlmstudio.prompts import templates as prompt_templates

    base_prompt_file = Path(prompt_templates.__file__).parent / "system_prompt_v2_0.yaml"
    base_prompt_bytes = base_prompt_file.read_bytes()

    dataset = tmp_path / "e2e.yaml"
    dataset.write_text(E2E_DATASET, encoding="utf-8")
    monkeypatch.setenv("RLM_STUDIO_REFINEMENT_DATASET", str(dataset))
    state = env.state
    cp = ChatProviderConfig(
        id="cp-1",
        name="RLM-TEST",
        llm_provider="openai",
        llm_model="gpt-4o",
        profile_id=env.profile_id,
        execution_mode="rlm",
    )
    state.config.chat_providers.append(cp)
    flow = FlowLLM()
    monkeypatch.setattr(state, "create_llm_adapter_for_chat_provider", lambda *_a, **_k: flow)
    state.telemetry.record_run(
        run_id="src",
        created_at=1.0,
        mode="rlm",
        query="Which file covers operations?",
        success=False,
        error="wall-clock budget exhausted",
        outcome_category="timeout",
        chat_provider_id="cp-1",
        profile_id=env.profile_id,
        content_sha256=content_sha256("A different source document."),
    )
    base = PATH.format(pid=env.profile_id)

    proposal = client.post(base, json={"source_run_ids": ["src"]})
    assert proposal.status_code == 201, proposal.text
    rid = proposal.json()["id"]
    assert client.post(f"{base}/{rid}/evaluate", json={}).status_code == 202
    deadline = time.monotonic() + 30
    while client.get(f"{base}/{rid}").json()["status"] == REFINEMENT_STATUS_EVALUATING:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    evaluated = client.get(f"{base}/{rid}").json()
    assert evaluated["status"] == REFINEMENT_STATUS_EVALUATED, evaluated
    assert evaluated["evaluation"]["complete"] is True
    assert env.saves == []  # AC-1: nothing written before accept

    assert client.post(f"{base}/{rid}/accept", json={}).status_code == 200
    accepted_prompt = _chat_system_prompt(state, flow, monkeypatch)
    assert CANDIDATE in accepted_prompt
    assert TEMPLATE_RLM not in accepted_prompt

    assert client.post(f"{base}/{rid}/rollback", json={}).status_code == 200
    restored_prompt = _chat_system_prompt(state, flow, monkeypatch)
    assert TEMPLATE_RLM in restored_prompt
    assert CANDIDATE not in restored_prompt

    history = client.get(base).json()
    assert [r["status"] for r in history] == [REFINEMENT_STATUS_ROLLED_BACK]
    assert base_prompt_file.read_bytes() == base_prompt_bytes  # AC-9


def test_rollback_with_an_unreadable_decision_record_is_409(client: TestClient, env: Env) -> None:
    rid = _seed(env)
    assert _post(client, env, rid, "accept").status_code == 200
    with env.state.telemetry._lock:
        conn = env.state.telemetry._connect()
        conn.execute("UPDATE refinements SET decision_json = '{}' WHERE id = ?", (rid,))
        conn.commit()
    resp = _post(client, env, rid, "rollback")
    assert resp.status_code == 409
    assert "unreadable" in resp.json()["error"]["message"]
    assert _profile_rlm(env) == CANDIDATE


def test_deleted_profile_history_is_still_exportable(client: TestClient, env: Env) -> None:
    """Spec §6: deleting a profile leaves its refinement rows exportable."""
    rid = _seed(env, status=REFINEMENT_STATUS_PROPOSAL_READY)
    assert client.delete(f"/api/profiles/{env.profile_id}").status_code == 204
    assert env.state.find_profile(env.profile_id) is None
    resp = client.get(PATH.format(pid=env.profile_id) + "/export.jsonl")
    assert resp.status_code == 200
    assert [json.loads(line)["id"] for line in resp.text.splitlines()] == [rid]
    assert client.get(PATH.format(pid="never-existed") + "/export.jsonl").status_code == 404


@pytest.mark.asyncio
async def test_unrelated_requests_stay_responsive_while_a_profile_edit_waits(env: Env) -> None:
    """A profile edit waiting on the config lock must not block the event loop."""
    import asyncio

    import httpx

    from rlmstudio.server.app import app

    held, release = threading.Event(), threading.Event()

    def hold_lock() -> None:
        with env.state._config_lock:
            held.set()
            release.wait(5)  # bounded, so a regression fails instead of hanging

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert held.wait(5)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            edit = asyncio.create_task(
                ac.put(f"/api/profiles/{env.profile_id}", json={"description": "edited"})
            )
            await asyncio.sleep(0.2)
            started = time.monotonic()
            health = await ac.get("/health")
            elapsed = time.monotonic() - started
            assert health.status_code == 200
            # A blocked event loop shows up as ~5 s (the holder's bound); 3 s leaves
            # room for a loaded CI machine without hiding the regression.
            assert elapsed < 3.0, f"/health waited {elapsed:.2f}s behind the profile edit"
            assert not edit.done(), "the edit should still be waiting for the lock"
            release.set()
            assert (await asyncio.wait_for(edit, 10)).status_code == 200
    finally:
        release.set()
        holder.join(10)


def test_reject_after_a_split_brain_accept_is_409(client: TestClient, env: Env) -> None:
    """An accept saved the candidate but its status write failed; a reject must
    not then record 'rejected' while the candidate is live."""
    rid = _seed(env)
    _abort_trigger(env.state.telemetry, "UPDATE")
    assert _post(client, env, rid, "accept").status_code == 500
    with env.state.telemetry._lock:
        conn = env.state.telemetry._connect()
        conn.execute("DROP TRIGGER inject_update")
        conn.commit()
    assert _status(env, rid) == REFINEMENT_STATUS_EVALUATED
    assert _profile_rlm(env) == CANDIDATE

    resp = _post(client, env, rid, "reject")
    assert resp.status_code == 409
    assert "already uses this candidate" in resp.json()["error"]["message"]
    assert _status(env, rid) == REFINEMENT_STATUS_EVALUATED
    assert _profile_rlm(env) == CANDIDATE
    assert _disk_rlm(env) == CANDIDATE
