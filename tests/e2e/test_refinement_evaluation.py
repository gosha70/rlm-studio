"""Phase C held-out evaluation for prompt refinement (#78).

Spec FR-5..FR-8, FR-12, FR-16 and AC-1, AC-5, AC-6.  The real RLM loop runs
with a scripted model that answers ``FINAL: …`` in one step; the reply is
wrong only when the candidate's marker is in the system prompt, which
proves each arm ran with its own supplement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rlmstudio.application.dto import LLMResponseDTO
from rlmstudio.application.refinement_constants import (
    REFINEMENT_EVALUATION_TEMPERATURE,
    REFINEMENT_STATUS_EVALUATED,
    REFINEMENT_STATUS_EVALUATING,
    REFINEMENT_STATUS_FAILED,
    REFINEMENT_STATUS_INTERRUPTED,
    REFINEMENT_STATUS_PROPOSAL_READY,
)
from rlmstudio.application.services.refinement import (
    BaseSupplementState,
    ProposalRecord,
    build_evidence_digest,
)
from rlmstudio.application.services.refinement_evaluation import EvaluationRecord
from rlmstudio.server import dependencies, refinement_eval
from rlmstudio.server.dependencies import AppState, get_state
from rlmstudio.server.models import ChatProviderConfig
from rlmstudio.telemetry.store import TelemetryStore, content_sha256

PATH = "/api/profiles/{pid}/refinements"
SOURCE_DOC = "The quarterly report names Harbour Platform as the launch."
BASELINE_TEXT = "Read the whole document."
CANDIDATE_MARKER = "CANDIDATE-RULE-7"
ANSWERS = {"launch platform": "Harbour", "memo": "Ada", "summarise": "A short summary."}

DATASET_YAML = f"""
name: refinement-test
metadata:
  version: "t1"
  size_buckets: {{small: 10, large: 1000}}
cases:
  - id: alpha
    query: "What is the launch platform?"
    content: "Notes. The launch platform is Harbour."
    expected_answer: "Harbour"
    tags: [small, holdout]
  - id: beta
    query: "Who signed the memo?"
    content: "Minutes. The memo was signed by Ada."
    expected_answer: "Ada"
    tags: [small, holdout]
    budget: {{max_steps: 3}}
  - id: overlap
    query: "Which launch platform is named?"
    content: "{SOURCE_DOC}"
    expected_answer: "Harbour"
    tags: [small, holdout]
  - id: judged
    query: "Please summarise the notes."
    content: "A long set of notes about nothing much."
    tags: [large, holdout]
  - id: public
    query: "What does the text say?"
    source_url: "https://example.invalid/missing-text.txt"
    tags: [large]
"""


@dataclass
class ArmLLM:
    """Answers correctly unless told to fail when the candidate marker is present."""

    fail_alpha_with_candidate: bool = True
    system_prompts: list[str] = field(default_factory=list)

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        system = messages[0]["content"]
        self.system_prompts.append(system)
        everything = "\n".join(m["content"] for m in messages).lower()
        answer = next((a for k, a in ANSWERS.items() if k in everything), "unknown")
        if self.fail_alpha_with_candidate and CANDIDATE_MARKER in system and answer == "Harbour":
            answer = "Somewhere else"
        return LLMResponseDTO(
            content=f"FINAL: {answer}", model="arm-model", input_tokens=10, output_tokens=3
        )

    def complete_stream(self, messages: list[dict[str, str]]) -> Any:
        yield self.complete(messages).content

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def get_pricing(self) -> dict[str, float]:
        return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}

    def get_completion_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0


@dataclass
class JudgeLLM:
    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        return LLMResponseDTO(
            content='{"dimensions": {"relevance": 4, "correctness": 4}, "reasoning": "ok"}',
            model="judge-model",
        )


@dataclass
class Env:
    state: AppState
    profile_id: str
    refinement_id: str
    arm_llm: ArmLLM
    temperatures: list[float | None]
    saves: list[str]


def _proposal(state: AppState) -> dict[str, Any]:
    record = ProposalRecord(
        source_run_ids=["src-1"],
        source_content_sha256={"src-1": content_sha256(SOURCE_DOC)},
        evidence=build_evidence_digest([]),
        proposal_chat_provider_id="cp-1",
        provider="openai",
        model="gpt-4o",
        prompt_version="1.0",
        prompt_sha256="p",
        input_tokens=1,
        output_tokens=1,
        base=BaseSupplementState(
            resolved=BASELINE_TEXT,
            sha256="s",
            rlm_key_present=True,
            rlm_key_value=BASELINE_TEXT,
            template_name=None,
            base_prompt_version="2.3",
            base_prompt_sha256="b",
        ),
        candidate=f"{BASELINE_TEXT}\n{CANDIDATE_MARKER}: answer fast.",
        rationale="why",
        diff="d",
    )
    return record.to_dict()


@pytest.fixture
def env(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    dataset = tmp_path / "eval.yaml"
    dataset.write_text(DATASET_YAML, encoding="utf-8")
    monkeypatch.setenv("RLM_STUDIO_REFINEMENT_DATASET", str(dataset))

    pid = client.post("/api/profiles", json={"name": "Tuned RLM", "strategy": "rlm"}).json()["id"]
    state = get_state()
    state.config.chat_providers.extend(
        [
            ChatProviderConfig(
                id="cp-1",
                name="RLM-TEST",
                llm_provider="openai",
                llm_model="gpt-4o",
                profile_id=pid,
                execution_mode="rlm",
            ),
            ChatProviderConfig(
                id="cp-other", name="OTHER", llm_provider="openai", llm_model="gpt-4o"
            ),
            ChatProviderConfig(
                id="judge", name="JUDGE", llm_provider="openai", llm_model="judge-model"
            ),
        ]
    )
    arm_llm = ArmLLM()
    temperatures: list[float | None] = []

    def fake_adapter(
        cp_id: str, num_retries: int | None = None, temperature: float | None = None
    ) -> Any:
        if cp_id == "judge":
            return JudgeLLM()
        temperatures.append(temperature)
        return arm_llm

    saves: list[str] = []
    monkeypatch.setattr(state, "create_llm_adapter_for_chat_provider", fake_adapter)
    monkeypatch.setattr(state, "save_config", lambda: saves.append("save_config"))
    monkeypatch.setattr(state, "save_config_checked", lambda: saves.append("checked"))
    rec = state.telemetry.create_refinement(
        profile_id=pid, status=REFINEMENT_STATUS_PROPOSAL_READY, proposal=_proposal(state)
    )
    return Env(state, pid, rec.id, arm_llm, temperatures, saves)


def _evaluate(client: TestClient, e: Env, **body: Any) -> Any:
    return client.post(PATH.format(pid=e.profile_id) + f"/{e.refinement_id}/evaluate", json=body)


def _wait(client: TestClient, e: Env, timeout: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    url = PATH.format(pid=e.profile_id) + f"/{e.refinement_id}"
    while time.monotonic() < deadline:
        body: dict[str, Any] = client.get(url).json()
        if body["status"] != REFINEMENT_STATUS_EVALUATING:
            return body
        time.sleep(0.05)
    raise AssertionError("evaluation did not finish")


# ---------------------------------------------------------------------------
# Holdout cases (FR-16)
# ---------------------------------------------------------------------------


class TestHoldoutCases:
    def test_lists_cases_with_flags(self, client: TestClient, env: Env) -> None:
        resp = client.get(PATH.format(pid=env.profile_id) + "/holdout-cases")
        assert resp.status_code == 200
        rows = {r["id"]: r for r in resp.json()}
        assert set(rows) == {"alpha", "beta", "overlap", "judged", "public"}
        assert rows["alpha"]["holdout"] is True
        assert rows["alpha"]["deterministic"] is True
        assert rows["alpha"]["size"] == "small"
        assert rows["judged"]["deterministic"] is False
        assert rows["public"]["holdout"] is False
        assert rows["public"]["materialisable"] is False
        assert "not fetched" in rows["public"]["skip_reason"]

    def test_missing_dataset_is_503(
        self, client: TestClient, env: Env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("RLM_STUDIO_REFINEMENT_DATASET", str(tmp_path / "absent.yaml"))
        resp = client.get(PATH.format(pid=env.profile_id) + "/holdout-cases")
        assert resp.status_code == 503
        assert "RLM_STUDIO_REFINEMENT_DATASET" in resp.json()["error"]["message"]


# ---------------------------------------------------------------------------
# Evaluation (FR-5..FR-8, AC-1, AC-5, AC-6)
# ---------------------------------------------------------------------------


class TestEvaluation:
    def test_holdout_evaluation_end_to_end(self, client: TestClient, env: Env) -> None:
        resp = _evaluate(client, env, judge_chat_provider_id="judge")
        assert resp.status_code == 202, resp.text
        accepted = resp.json()
        assert accepted["case_ids"] == ["alpha", "beta", "overlap", "judged"]
        assert accepted["run_count"] == 4 * 2 * 2

        body = _wait(client, env)
        assert body["status"] == REFINEMENT_STATUS_EVALUATED, body
        assert body["progress"] is None
        record = EvaluationRecord.from_dict(body["evaluation"])

        # AC-5: the case sharing a source run's document is never evaluated.
        assert record.setup.excluded_cases == {"overlap": refinement_eval.EXCLUDED_SAME_DOCUMENT}
        assert [c.case_id for c in record.cases] == ["alpha", "beta", "judged"]
        assert record.complete, record.incomplete_reasons

        # FR-8: the candidate breaks alpha in both reps, so alpha regresses.
        assert record.regression is True
        assert record.regression_case_ids == ["alpha"]
        assert record.deterministic_case_count == 2
        cases = {c.case_id: c for c in record.cases}
        assert (cases["alpha"].baseline_pass_rate, cases["alpha"].candidate_pass_rate) == (1.0, 0.0)
        assert (cases["beta"].baseline_pass_rate, cases["beta"].candidate_pass_rate) == (1.0, 1.0)
        assert cases["judged"].regressed is False

        # Judge scores are recorded with provenance and never gate.
        assert record.setup.judge is not None
        assert record.setup.judge.model == "judge-model"
        assert record.deltas is not None
        assert record.deltas.judge_baseline is not None
        assert record.deltas.judge_baseline.mean == 4.0

        # FR-5: each arm ran with its own supplement, at temperature 0.
        baseline_prompts = [p for p in env.arm_llm.system_prompts if CANDIDATE_MARKER not in p]
        candidate_prompts = [p for p in env.arm_llm.system_prompts if CANDIDATE_MARKER in p]
        assert len(baseline_prompts) == len(candidate_prompts) == 3 * 2
        assert all(BASELINE_TEXT in p for p in baseline_prompts)
        assert env.temperatures
        assert set(env.temperatures) == {REFINEMENT_EVALUATION_TEMPERATURE}

        # Every arm is a telemetry run with the case's document hash and no profile.
        for case in record.cases:
            for rep in case.reps:
                for arm in (rep.baseline, rep.candidate):
                    assert arm is not None
                    run = env.state.telemetry.get_run(arm.run_id)
                    assert run is not None
                    assert run.content_sha256 == case.content_sha256
                    assert run.profile_id is None

        # AC-1: evaluating never writes the profile or the config.
        assert env.saves == []
        source_runs = client.get(PATH.format(pid=env.profile_id) + "/source-runs").json()
        assert source_runs == []

    def test_skipped_case_makes_the_evaluation_incomplete(
        self, client: TestClient, env: Env
    ) -> None:
        resp = _evaluate(client, env, case_ids=["alpha", "public"])
        assert resp.status_code == 202
        record = EvaluationRecord.from_dict(_wait(client, env)["evaluation"])
        assert record.complete is False
        assert "public" in record.setup.skipped_cases
        assert record.deltas is None
        assert record.regression is False

    def test_all_cases_excluded_is_incomplete(self, client: TestClient, env: Env) -> None:
        _evaluate(client, env, case_ids=["overlap"])
        record = EvaluationRecord.from_dict(_wait(client, env)["evaluation"])
        assert record.complete is False
        assert "no case was evaluated" in record.incomplete_reasons

    def test_matrix_failure_leaves_arms_missing(
        self, client: TestClient, env: Env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("matrix crashed")

        monkeypatch.setattr(refinement_eval.RunMatrixComparisonUseCase, "execute", boom)
        _evaluate(client, env, case_ids=["alpha"])
        record = EvaluationRecord.from_dict(_wait(client, env)["evaluation"])
        assert record.complete is False
        assert any("missing an arm" in r for r in record.incomplete_reasons)

    def test_worker_error_marks_the_refinement_failed(
        self, client: TestClient, env: Env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("dataset vanished")

        monkeypatch.setattr(refinement_eval, "run_evaluation", boom)
        assert _evaluate(client, env).status_code == 202
        body = _wait(client, env)
        assert body["status"] == REFINEMENT_STATUS_FAILED
        assert "dataset vanished" in body["error"]
        # The job slot is released: another refinement can start.
        assert env.state.try_start_refinement_job(env.profile_id)


class TestEvaluateGuards:
    def test_second_evaluation_is_409(self, client: TestClient, env: Env) -> None:
        _evaluate(client, env, case_ids=["alpha"])
        _wait(client, env)
        resp = _evaluate(client, env, case_ids=["alpha"])
        assert resp.status_code == 409
        assert "evaluated" in resp.json()["error"]["message"]

    def test_running_job_for_the_profile_is_409(self, client: TestClient, env: Env) -> None:
        assert env.state.try_start_refinement_job(env.profile_id)
        resp = _evaluate(client, env)
        assert resp.status_code == 409
        record = env.state.telemetry.get_refinement(env.refinement_id)
        assert record is not None
        assert record.status == REFINEMENT_STATUS_PROPOSAL_READY

    def test_chat_provider_of_another_profile_is_400(self, client: TestClient, env: Env) -> None:
        resp = _evaluate(client, env, chat_provider_id="cp-other")
        assert resp.status_code == 400
        assert "does not use this profile" in resp.json()["error"]["message"]

    @pytest.mark.parametrize(
        ("body", "status"),
        [
            ({"case_ids": ["nope"]}, 400),
            ({"reps": 1}, 422),
            ({"case_ids": []}, 422),
            ({"judge_chat_provider_id": "missing-judge"}, 400),
        ],
    )
    def test_bad_requests(
        self, client: TestClient, env: Env, body: dict[str, Any], status: int
    ) -> None:
        assert _evaluate(client, env, **body).status_code == status

    def test_unknown_refinement_is_404(self, client: TestClient, env: Env) -> None:
        url = PATH.format(pid=env.profile_id) + "/missing/evaluate"
        assert client.post(url, json={}).status_code == 404


# ---------------------------------------------------------------------------
# Startup (FR-12)
# ---------------------------------------------------------------------------


def test_startup_marks_running_evaluations_interrupted(monkeypatch: pytest.MonkeyPatch) -> None:
    store = TelemetryStore(":memory:")
    running = store.create_refinement(
        profile_id="p", status=REFINEMENT_STATUS_EVALUATING, proposal={}
    )
    monkeypatch.setattr(dependencies, "TelemetryStore", lambda *_a, **_k: store)
    AppState(load_from_disk=False)
    record = store.get_refinement(running.id)
    assert record is not None
    assert record.status == REFINEMENT_STATUS_INTERRUPTED


def test_case_budget_overrides_only_the_keys_it_sets() -> None:
    from rlmstudio.application.dto import RunConfigDTO

    config = RunConfigDTO(max_steps=10, max_time_seconds=99.0, max_cost=1.0)
    refinement_eval._apply_case_budget(config, {"max_steps": 3, "max_cost": None})
    assert config.max_steps == 3
    assert config.max_time_seconds == 99.0
    assert config.max_cost == 1.0


def test_float_budget_values_are_accepted() -> None:
    from rlmstudio.application.dto import RunConfigDTO

    config = RunConfigDTO(max_steps=10)
    refinement_eval._apply_case_budget(config, {"max_steps": 3.0, "max_recursion_depth": 2.0})
    assert (config.max_steps, config.max_recursion_depth) == (3, 2)


class TestEvaluateFailurePaths:
    def test_worker_that_cannot_start_releases_the_slot(
        self, client: TestClient, env: Env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rlmstudio.server.routes import refinements as routes

        def no_thread(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("can't start new thread")

        monkeypatch.setattr(routes, "start_evaluation", no_thread)
        resp = _evaluate(client, env)
        assert resp.status_code == 500
        record = env.state.telemetry.get_refinement(env.refinement_id)
        assert record is not None
        assert record.status == REFINEMENT_STATUS_FAILED
        assert "did not start" in (record.error or "")
        assert env.state.try_start_refinement_job(env.profile_id)

    def test_malformed_dataset_is_503(
        self, client: TestClient, env: Env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("cases: [unclosed\n", encoding="utf-8")
        monkeypatch.setenv("RLM_STUDIO_REFINEMENT_DATASET", str(bad))
        resp = _evaluate(client, env)
        assert resp.status_code == 503
        assert "unreadable" in resp.json()["error"]["message"]
