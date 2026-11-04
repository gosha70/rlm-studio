/**
 * Refinement panel (#78 Phase E, spec FR-18): proposal, evaluation,
 * decisions and the states in between. API calls are mocked; SWR runs for
 * real with a fresh cache per test.
 */

import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { SWRConfig } from "swr";
import { vi, describe, test, expect, beforeEach } from "vitest";
import { RefinementPanel } from "@/components/settings/refinement-panel";
import {
  apiErrorMessage,
  type ChatProviderConfig,
  type Refinement,
  type RefinementEvaluation,
  type RunProfile,
} from "@/lib/api";
import {
  REFINEMENT_STATUS_ACCEPTED,
  REFINEMENT_STATUS_EVALUATED,
  REFINEMENT_STATUS_EVALUATING,
  REFINEMENT_STATUS_PROPOSAL_READY,
} from "@/lib/constants";

vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    listRefinements: vi.fn(),
    getRefinement: vi.fn(),
    getRefinementSourceRuns: vi.fn(),
    getHoldoutCases: vi.fn(),
    createRefinement: vi.fn(),
    evaluateRefinement: vi.fn(),
    acceptRefinement: vi.fn(),
    rejectRefinement: vi.fn(),
    rollbackRefinement: vi.fn(),
  };
});

import * as api from "@/lib/api";

const mocked = vi.mocked(api);

const PROFILE: RunProfile = {
  id: "prof-1",
  name: "Tuned RLM",
  description: "",
  strategy: "rlm",
  default_provider: null,
  providers_enabled: [],
  runtime_settings: { temperature: 0.7, top_p: 1, max_output_tokens: 4096, timeout_seconds: 120 },
  budget: {
    max_steps: 32,
    max_tokens: 0,
    max_cost_usd: 2,
    max_time_seconds: 300,
    max_recursion_depth: 1,
    repeat_limit: 2,
    nudge_at_fraction: 0.6,
  },
  system_prompts: {},
  prompt_template_name: "Default",
  is_builtin: false,
};

const CPS = [
  { id: "cp-1", name: "RLM-TEST", profile_id: "prof-1" },
  { id: "cp-other", name: "OTHER", profile_id: "someone" },
] as unknown as ChatProviderConfig[];

const CANDIDATE = "Read the document index first.";

function evaluation(overrides: Partial<RefinementEvaluation> = {}): RefinementEvaluation {
  return {
    setup: {
      chat_provider_id: "cp-1",
      provider: "openai",
      model: "gpt-4o",
      temperature: 0,
      reps: 2,
      dataset_name: "longdoc-v1",
      dataset_version: "1.0",
      case_ids: ["alpha"],
      excluded_cases: {},
      skipped_cases: {},
      judge: null,
      started_at: 1,
    },
    cases: [
      {
        case_id: "alpha",
        deterministic: true,
        content_sha256: "h",
        reps: [],
        baseline_pass_rate: 1,
        candidate_pass_rate: 1,
        regressed: false,
      },
    ],
    complete: true,
    incomplete_reasons: [],
    deterministic_case_count: 1,
    regression: false,
    regression_case_ids: [],
    deltas: {
      accuracy_baseline: 1,
      accuracy_candidate: 1,
      judge_baseline: null,
      judge_candidate: null,
      cost_baseline: 0.01,
      cost_candidate: 0.02,
      steps_mean_baseline: 3,
      steps_mean_candidate: 2,
      outcomes_baseline: { success: 2 },
      outcomes_candidate: { success: 2 },
    },
    finished_at: 2,
    ...overrides,
  };
}

function refinement(overrides: Partial<Refinement> = {}): Refinement {
  return {
    id: "ref-1",
    profile_id: "prof-1",
    status: REFINEMENT_STATUS_PROPOSAL_READY,
    created_at: 1_700_000_000,
    updated_at: 1_700_000_000,
    proposal: {
      source_run_ids: ["run-1"],
      source_content_sha256: { "run-1": "h" },
      evidence: {
        text: "",
        sha256: "",
        included_run_ids: ["run-1"],
        excluded_run_ids: [],
        truncated_chars: 0,
        redacted_lines: 0,
      },
      proposal_chat_provider_id: "cp-1",
      provider: "openai",
      model: "gpt-4o",
      prompt_version: "1.0",
      prompt_sha256: "p",
      input_tokens: 1,
      output_tokens: 1,
      base: {
        resolved: "Template text",
        sha256: "s",
        rlm_key_present: false,
        rlm_key_value: null,
        template_name: "Default",
        base_prompt_version: "2.3",
        base_prompt_sha256: "b",
      },
      candidate: CANDIDATE,
      rationale: "Runs skipped the index.",
      diff: "+Read the document index first.",
    },
    evaluation: null,
    decision: null,
    error: null,
    progress: null,
    ...overrides,
  };
}

function renderPanel(onProfileChanged = vi.fn()) {
  render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <RefinementPanel
        profile={PROFILE}
        chatProviders={CPS}
        open
        onOpenChange={vi.fn()}
        onProfileChanged={onProfileChanged}
      />
    </SWRConfig>,
  );
  return onProfileChanged;
}

function showRefinement(r: Refinement) {
  mocked.listRefinements.mockResolvedValue([r]);
  mocked.getRefinement.mockResolvedValue(r);
}

beforeEach(() => {
  vi.clearAllMocks();
  mocked.listRefinements.mockResolvedValue([]);
  mocked.getRefinementSourceRuns.mockResolvedValue([]);
  mocked.getHoldoutCases.mockResolvedValue([]);
});

describe("apiErrorStatus", () => {
  test("reads the HTTP status from an API error", () => {
    expect(api.apiErrorStatus(new Error('API error 500: {"error": {}}'))).toBe(500);
    expect(api.apiErrorStatus(new Error("network down"))).toBe(0);
  });
});

describe("apiErrorMessage", () => {
  test("extracts the server message from an API error", () => {
    const err = new Error('API error 409: {"error": {"code": "ERROR", "message": "already uses this candidate"}}');
    expect(apiErrorMessage(err)).toBe("already uses this candidate");
  });

  test("falls back to the raw text", () => {
    expect(apiErrorMessage(new Error("network down"))).toBe("network down");
  });
});

describe("New proposal", () => {
  test("explains when there are no usable runs", async () => {
    renderPanel();
    expect(await screen.findByText(/No usable runs yet/)).toBeInTheDocument();
  });

  test("proposes from the selected runs through the newest run's Chat Provider", async () => {
    mocked.getRefinementSourceRuns.mockResolvedValue([
      {
        id: "run-new",
        created_at: 2,
        mode: "rlm",
        query: "Which platform?",
        success: false,
        outcome_category: "timeout",
        degraded: false,
        chat_provider_id: "cp-1",
        chat_provider_name: "RLM-TEST",
        total_cost: 0,
        total_tokens: 0,
        steps_count: 3,
        content_sha256: "h",
      },
      {
        id: "run-old",
        created_at: 1,
        mode: "rlm",
        query: "Who signed?",
        success: true,
        outcome_category: "success",
        degraded: true,
        chat_provider_id: "cp-other",
        chat_provider_name: "OTHER",
        total_cost: 0,
        total_tokens: 0,
        steps_count: 3,
        content_sha256: "h2",
      },
    ]);
    const created = refinement();
    mocked.createRefinement.mockResolvedValue(created);
    mocked.getRefinement.mockResolvedValue(created);
    renderPanel();

    expect(await screen.findByText("degraded")).toBeInTheDocument();
    const propose = screen.getByRole("button", { name: /Propose from 0 runs/ });
    expect(propose).toBeDisabled();
    fireEvent.click(screen.getByLabelText("Use run run-old"));
    fireEvent.click(screen.getByLabelText("Use run run-new"));
    fireEvent.click(screen.getByRole("button", { name: /Propose from 2 runs/ }));

    await waitFor(() => expect(mocked.createRefinement).toHaveBeenCalledTimes(1));
    expect(mocked.createRefinement).toHaveBeenCalledWith("prof-1", {
      source_run_ids: ["run-new", "run-old"],
      proposal_chat_provider_id: "cp-1",
    });
    expect(await screen.findByText("Runs skipped the index.")).toBeInTheDocument();
  });

  test("shows the server's reason when a proposal fails", async () => {
    mocked.getRefinementSourceRuns.mockResolvedValue([
      {
        id: "run-1",
        created_at: 1,
        mode: "rlm",
        query: "q",
        success: false,
        outcome_category: "timeout",
        degraded: false,
        chat_provider_id: "cp-1",
        chat_provider_name: "RLM-TEST",
        total_cost: 0,
        total_tokens: 0,
        steps_count: 1,
        content_sha256: "h",
      },
    ]);
    mocked.createRefinement.mockRejectedValue(
      new Error('API error 502: {"error": {"message": "The proposal model returned an unusable reply"}}'),
    );
    renderPanel();
    fireEvent.click(await screen.findByLabelText("Use run run-1"));
    fireEvent.click(screen.getByRole("button", { name: /Propose from 1 run/ }));
    expect(await screen.findByText("The proposal model returned an unusable reply")).toBeInTheDocument();
  });
});

describe("Evaluation", () => {
  test("defaults to runnable holdout cases and shows the run count before starting", async () => {
    showRefinement(refinement());
    mocked.getHoldoutCases.mockResolvedValue([
      { id: "alpha", query: "q", tags: ["holdout"], size: "small", holdout: true, deterministic: true, materialisable: true, skip_reason: null },
      { id: "beta", query: "q", tags: ["holdout"], size: "small", holdout: true, deterministic: true, materialisable: true, skip_reason: null },
      { id: "public", query: "q", tags: [], size: "large", holdout: false, deterministic: true, materialisable: false, skip_reason: "public text not fetched" },
    ]);
    mocked.evaluateRefinement.mockResolvedValue({
      refinement_id: "ref-1",
      status: REFINEMENT_STATUS_EVALUATING,
      case_ids: ["alpha", "beta"],
      reps: 2,
      run_count: 8,
      chat_provider_id: "cp-1",
      judge_chat_provider_id: null,
    });
    renderPanel();

    expect(await screen.findByText(/8 RLM runs \(2 cases × 2 repetitions/)).toBeInTheDocument();
    expect(screen.getByLabelText("Evaluate on public")).toBeDisabled();
    expect(screen.getByText(/Accepting gives it a custom RLM prompt/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Run 8 evaluation runs" }));

    await waitFor(() => expect(mocked.evaluateRefinement).toHaveBeenCalledTimes(1));
    expect(mocked.evaluateRefinement).toHaveBeenCalledWith("prof-1", "ref-1", {
      case_ids: ["alpha", "beta"],
      reps: 2,
      chat_provider_id: "cp-1",
    });
  });

  test("a failed start shows the server's reason", async () => {
    showRefinement(refinement());
    mocked.getHoldoutCases.mockResolvedValue([
      { id: "alpha", query: "q", tags: ["holdout"], size: "small", holdout: true, deterministic: true, materialisable: true, skip_reason: null },
    ]);
    mocked.evaluateRefinement.mockRejectedValue(
      new Error('API error 503: {"error": {"message": "Evaluation dataset not found; set RLM_STUDIO_REFINEMENT_DATASET"}}'),
    );
    renderPanel();
    fireEvent.click(await screen.findByRole("button", { name: "Run 4 evaluation runs" }));
    expect(await screen.findByText(/set RLM_STUDIO_REFINEMENT_DATASET/)).toBeInTheDocument();
  });

  test("polls while evaluating and stops once results arrive", async () => {
    const running = refinement({ status: REFINEMENT_STATUS_EVALUATING, progress: { done: 1, total: 2 } });
    const done = refinement({ status: REFINEMENT_STATUS_EVALUATED, evaluation: evaluation() });
    mocked.listRefinements.mockResolvedValue([running]);
    mocked.getRefinement.mockResolvedValueOnce(running).mockResolvedValue(done);
    renderPanel();
    expect(await screen.findByText(/1 of 2 case × repetition pairs done/)).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: "Accept" }, { timeout: 5000 })).toBeInTheDocument();
    const callsWhenDone = mocked.getRefinement.mock.calls.length;
    await new Promise((resolve) => setTimeout(resolve, 2500));
    expect(mocked.getRefinement.mock.calls.length).toBe(callsWhenDone);
  });

  test("progress with nothing to run never divides by zero", async () => {
    showRefinement(refinement({ status: REFINEMENT_STATUS_EVALUATING, progress: { done: 0, total: 0 } }));
    renderPanel();
    expect(await screen.findByText(/Preparing cases/)).toBeInTheDocument();
    expect(screen.queryByText(/NaN/)).not.toBeInTheDocument();
  });

  test("progress counts finished pairs", async () => {
    showRefinement(refinement({ status: REFINEMENT_STATUS_EVALUATING, progress: { done: 1, total: 4 } }));
    renderPanel();
    expect(await screen.findByText(/1 of 4 case × repetition pairs done/)).toBeInTheDocument();
  });
});

describe("Decisions", () => {
  test("accepts a clean evaluation and refreshes the profile", async () => {
    showRefinement(refinement({ status: REFINEMENT_STATUS_EVALUATED, evaluation: evaluation() }));
    mocked.acceptRefinement.mockResolvedValue(refinement({ status: REFINEMENT_STATUS_ACCEPTED }));
    const onProfileChanged = renderPanel();

    fireEvent.click(await screen.findByRole("button", { name: "Accept" }));
    await waitFor(() => expect(mocked.acceptRefinement).toHaveBeenCalledTimes(1));
    expect(mocked.acceptRefinement).toHaveBeenCalledWith("prof-1", "ref-1", {
      acknowledge_regression: false,
    });
    await waitFor(() => expect(onProfileChanged).toHaveBeenCalledTimes(1));
  });

  test("a regression needs an explicit acknowledgement before Accept", async () => {
    showRefinement(
      refinement({
        status: REFINEMENT_STATUS_EVALUATED,
        evaluation: evaluation({ regression: true, regression_case_ids: ["alpha"] }),
      }),
    );
    mocked.acceptRefinement.mockResolvedValue(refinement({ status: REFINEMENT_STATUS_ACCEPTED }));
    renderPanel();

    const accept = await screen.findByRole("button", { name: "Accept" });
    expect(accept).toBeDisabled();
    expect(screen.getByText(/Reject is the default/)).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText(/Accept despite the regression on alpha/));
    expect(accept).toBeEnabled();
    fireEvent.click(accept);
    await waitFor(() =>
      expect(mocked.acceptRefinement).toHaveBeenCalledWith("prof-1", "ref-1", {
        acknowledge_regression: true,
      }),
    );
  });

  test("an incomplete evaluation cannot be accepted and says why", async () => {
    showRefinement(
      refinement({
        status: REFINEMENT_STATUS_EVALUATED,
        evaluation: evaluation({
          complete: false,
          incomplete_reasons: ["alpha rep 1 candidate: provider failure (timeout), run r9"],
          deltas: null,
        }),
      }),
    );
    renderPanel();
    expect(await screen.findByRole("button", { name: "Accept" })).toBeDisabled();
    expect(screen.getByText(/provider failure \(timeout\)/)).toBeInTheDocument();
  });

  test("a failed accept shows the server's message and still refreshes the profile", async () => {
    showRefinement(refinement({ status: REFINEMENT_STATUS_EVALUATED, evaluation: evaluation() }));
    mocked.acceptRefinement.mockRejectedValue(
      new Error('API error 500: {"error": {"message": "The candidate was saved to the profile, but recording the accept failed"}}'),
    );
    const onProfileChanged = renderPanel();
    fireEvent.click(await screen.findByRole("button", { name: "Accept" }));
    expect(await screen.findByText(/recording the accept failed/)).toBeInTheDocument();
    await waitFor(() => expect(onProfileChanged).toHaveBeenCalledTimes(1));
  });

  test("a refused accept leaves the profile and any edit alone", async () => {
    showRefinement(refinement({ status: REFINEMENT_STATUS_EVALUATED, evaluation: evaluation() }));
    mocked.acceptRefinement.mockRejectedValue(
      new Error('API error 409: {"error": {"message": "The profile\'s RLM prompt changed after this proposal"}}'),
    );
    const onProfileChanged = renderPanel();
    fireEvent.click(await screen.findByRole("button", { name: "Accept" }));
    expect(await screen.findByText(/changed after this proposal/)).toBeInTheDocument();
    expect(onProfileChanged).not.toHaveBeenCalled();
  });

  test("rejects with a reason without touching the profile", async () => {
    showRefinement(refinement());
    mocked.rejectRefinement.mockResolvedValue(refinement());
    const onProfileChanged = renderPanel();
    fireEvent.change(await screen.findByLabelText("Reason (optional)"), {
      target: { value: "not convinced" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Reject" }));
    await waitFor(() =>
      expect(mocked.rejectRefinement).toHaveBeenCalledWith("prof-1", "ref-1", "not convinced"),
    );
    expect(onProfileChanged).not.toHaveBeenCalled();
  });

  test("rolls back an accepted refinement and refreshes the profile", async () => {
    showRefinement(
      refinement({
        status: REFINEMENT_STATUS_ACCEPTED,
        evaluation: evaluation(),
        decision: {
          decision: REFINEMENT_STATUS_ACCEPTED,
          decided_at: 1_700_000_100,
          reason: null,
          regression: false,
          acknowledge_regression: false,
          rollback: null,
        },
      }),
    );
    mocked.rollbackRefinement.mockResolvedValue(refinement());
    const onProfileChanged = renderPanel();
    expect(await screen.findByRole("button", { name: "Roll back" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Accept" })).not.toBeInTheDocument();
    // Advice that only applies before a decision is gone once it is made.
    expect(screen.queryByText(/Accepting gives it a custom RLM prompt/)).not.toBeInTheDocument();
    fireEvent.click(await screen.findByRole("button", { name: "Roll back" }));
    await waitFor(() => expect(mocked.rollbackRefinement).toHaveBeenCalledWith("prof-1", "ref-1"));
    await waitFor(() => expect(onProfileChanged).toHaveBeenCalledTimes(1));
  });
});
