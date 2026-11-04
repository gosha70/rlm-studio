/**
 * LLM Tuner (Compare) result cards and chart.
 *
 * Renders the page, runs a matrix against a stubbed `fetch`, and checks what
 * the user sees for failed and successful slots.
 */

import { render, screen, fireEvent, within } from "@testing-library/react";
import { vi, describe, test, expect, afterEach } from "vitest";
import useSWR from "swr";

vi.mock("swr", () => ({
  default: vi.fn(),
  useSWRConfig: () => ({ mutate: vi.fn() }),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
  useSearchParams: () => ({ get: () => null }),
}));

vi.mock("@/components/shared/app-shell", () => ({
  AppShell: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

import ComparePage from "@/app/compare/page";
import type {
  CompareMatrixResponse,
  CompareMatrixSlotResponse,
  LLMProviderConfig,
} from "@/lib/api";

const PROVIDER: LLMProviderConfig = {
  id: "lp-1",
  name: "LM Studio Qwen",
  backend: "lmstudio",
  model: "qwen2.5-7b-instruct",
  runtime_settings: {
    temperature: 0.2,
    top_p: 1,
    max_output_tokens: 1024,
    timeout_seconds: 60,
  },
  status: "connected",
};

function mockSWR(llmProviders: LLMProviderConfig[] = [PROVIDER]) {
  const stub = (data: unknown) => ({
    data,
    mutate: vi.fn(),
    error: undefined,
    isLoading: false,
    isValidating: false,
  });
  vi.mocked(useSWR).mockImplementation(((key: unknown) => {
    if (key === "llm-providers") return stub(llmProviders);
    if (key === "profiles") return stub([]);
    return stub(undefined);
  }) as typeof useSWR);
}

function makeSlot(overrides: Partial<CompareMatrixSlotResponse> = {}): CompareMatrixSlotResponse {
  return {
    slot_id: "slot-1",
    label: "LM Studio Qwen · direct",
    mode: "direct",
    provider: "lmstudio",
    model: "qwen2.5-7b-instruct",
    chat_provider_id: "cp-1",
    chat_provider_name: null,
    execution_id: "exec-1",
    success: true,
    answer: "The answer.",
    error: null,
    input_tokens: 100,
    output_tokens: 20,
    total_tokens: 120,
    total_cost: 0,
    elapsed_seconds: 1.0,
    steps: 1,
    cost_known: true,
    ...overrides,
  };
}

/** Fill the form, run, and wait for the results grid. */
async function runWith(slots: CompareMatrixSlotResponse[]) {
  mockSWR();
  render(<ComparePage />);
  fireEvent.change(screen.getByPlaceholderText("What would you like to ask?"), {
    target: { value: "What is the title?" },
  });
  fireEvent.change(screen.getByPlaceholderText(/Paste text content here/), {
    target: { value: "a document" },
  });
  const response: CompareMatrixResponse = {
    comparison_group_id: "group-123456789012",
    session_id: "sess-1",
    slots,
    ranking: slots.map((_, i) => i),
    ranking_metric: "cost",
    total_elapsed: 3.4,
  };
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({ ok: true, json: async () => response }),
  );
  fireEvent.click(screen.getByRole("button", { name: /Run Compare/ }));
  await screen.findByText("Results");
}

const card = (slot: CompareMatrixSlotResponse) =>
  screen.getByTestId(`matrix-slot-${slot.mode}-${slot.provider}`);

afterEach(() => vi.unstubAllGlobals());

describe("Failed Compare cell", () => {
  test("shows tokens, steps and elapsed time the failed run spent", async () => {
    const failed = makeSlot({
      mode: "rlm",
      label: "LM Studio Qwen · rlm",
      success: false,
      total_tokens: 4697,
      steps: 2,
      elapsed_seconds: 11.7,
      answer: "⚠️ **Execution error**.\n\nBudget exceeded.",
    });
    await runWith([failed]);

    const c = card(failed);
    expect(within(c).getByText("4,697")).toBeInTheDocument();
    expect(within(c).getByText("2 steps")).toBeInTheDocument();
    expect(within(c).getByText("11.7s")).toBeInTheDocument();
  });

  test("shows the server's message with its fix and the partial answer", async () => {
    const timedOut = makeSlot({
      mode: "rlm",
      label: "LM Studio Qwen · rlm",
      success: false,
      error: "Official RLM engine timed out after 5s (wall-clock budget)",
      answer:
        "⚠️ **RLM run timed out** after 5s.\n\n**How to fix** — increase **Timeout (s)**.\n\n**Partial answer**\n\nThe title is Handbook.",
    });
    await runWith([timedOut]);

    const detail = within(card(timedOut)).getByTestId("slot-failure-detail");
    expect(detail).toHaveTextContent("RLM run timed out after 5s.");
    expect(detail).toHaveTextContent("How to fix — increase Timeout (s).");
    expect(detail).toHaveTextContent("The title is Handbook.");
    // Rendered as markdown, not shown as raw asterisks.
    expect(detail.textContent).not.toContain("**");
  });
});
