/**
 * LLM Tuner (Compare) result cards and chart.
 *
 * Renders the page, runs a matrix against a stubbed `fetch`, and checks what
 * the user sees for failed and successful slots.
 */

import { render, screen, fireEvent, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
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

// Recharts draws nothing in jsdom. The mock keeps what the user would read:
// the tooltip text per bar (through the page's own formatter) and the value
// axis' domain and ticks.
const chart = vi.hoisted(() => ({ data: [] as Array<Record<string, unknown>> }));
vi.mock("recharts", () => {
  const Pass = ({ children }: { children?: React.ReactNode }) => <>{children}</>;
  type Formatter = (v: unknown, n: string, item: { payload: unknown }) => [string, string];
  return {
    ResponsiveContainer: Pass,
    BarChart: ({ data, children }: { data?: Array<Record<string, unknown>>; children?: React.ReactNode }) => {
      chart.data = data ?? [];
      return <div>{children}</div>;
    },
    Bar: Pass,
    Cell: () => null,
    YAxis: () => null,
    CartesianGrid: () => null,
    Legend: () => null,
    ReferenceLine: () => null,
    XAxis: ({ domain, ticks }: { domain?: unknown; ticks?: unknown }) => (
      <div
        data-testid="chart-value-axis"
        data-domain={JSON.stringify(domain ?? null)}
        data-ticks={JSON.stringify(ticks ?? null)}
      />
    ),
    Tooltip: ({ formatter }: { formatter?: Formatter }) => (
      <ul data-testid="chart-tooltip">
        {chart.data.map((d) => (
          <li key={String(d.name)}>
            {`${String(d.name)}: ${formatter ? formatter(d.value, "value", { payload: d })[0] : String(d.value)}`}
          </li>
        ))}
      </ul>
    ),
  };
});

import ComparePage from "@/app/compare/page";
import type {
  CompareMatrixResponse,
  CompareMatrixSlotResponse,
  JudgeScoreData,
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

function mockSWR(llmProviders: LLMProviderConfig[] = [PROVIDER], judgeConfigured = false) {
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
    if (key === "config") return stub(judgeConfigured ? { judge_chat_provider_id: "cp-judge" } : {});
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
async function runWith(slots: CompareMatrixSlotResponse[], judgeScores: JudgeScoreData[] = []) {
  mockSWR([PROVIDER], judgeScores.length > 0);
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
    vi.fn().mockImplementation(async (url: string) => ({
      ok: true,
      json: async () =>
        String(url).includes("/api/evaluations/judge")
          ? { pointwise: judgeScores, pairwise: [] }
          : response,
    })),
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

describe("Failed Compare cell without an answer", () => {
  test("keeps the whole server URL in the connection error", async () => {
    const unreachable = makeSlot({
      success: false,
      answer: "",
      error:
        "Cannot connect to the LLM server at 'http://127.0.0.1:18097/v1'. Make sure the server is running and reachable. (Detail: litellm.InternalServerError: Connection refused)",
    });
    await runWith([unreachable]);

    expect(
      within(card(unreachable)).getByText(
        "Cannot connect to the LLM server at 'http://127.0.0.1:18097/v1'.",
      ),
    ).toBeInTheDocument();
  });

  test("drops markdown asterisks from the shortened message", async () => {
    const exhausted = makeSlot({
      mode: "rlm",
      label: "LM Studio Qwen · rlm",
      success: false,
      answer: "",
      error: "⚠️ **Step budget exhausted** (13 of 16 steps used) without producing a complete answer.",
    });
    await runWith([exhausted]);

    expect(
      within(card(exhausted)).getByText(
        "Step budget exhausted (13 of 16 steps used) without producing a complete answer.",
      ),
    ).toBeInTheDocument();
  });
});

describe("Judge Score chart", () => {
  const judged = makeSlot({ slot_id: "a", execution_id: "exec-a", label: "Ollama Llama · direct", provider: "ollama" });
  const unjudged = makeSlot({ slot_id: "b", execution_id: "exec-b", label: "LM Studio Qwen · direct" });
  const failed = makeSlot({
    slot_id: "c",
    execution_id: "exec-c",
    label: "Mock vLLM · direct",
    provider: "vllm",
    success: false,
    answer: "⚠️ **Cannot reach the provider**",
  });
  const score: JudgeScoreData = {
    id: "j-1",
    execution_id: "exec-a",
    session_id: "sess-1",
    chat_provider_id: "cp-1",
    judge_provider_id: "cp-judge",
    dimensions: { relevance: 4 },
    overall_score: 4.5,
    reasoning: "",
    created_at: "2026-11-04T00:00:00Z",
  };

  async function judgeAndRankByScore() {
    Element.prototype.scrollIntoView = vi.fn();
    Element.prototype.hasPointerCapture = vi.fn().mockReturnValue(false);
    await runWith([judged, unjudged, failed], [score]);
    fireEvent.click(screen.getByRole("button", { name: /Auto-Judge/ }));
    await screen.findByText(/Judge Dimensions/);
    const ranking = screen
      .getAllByRole("combobox")
      .find((el) => el.textContent?.includes("Cost (lowest wins)"))!;
    fireEvent.pointerDown(ranking, { button: 0, ctrlKey: false, pointerType: "mouse" });
    fireEvent.click(await screen.findByRole("option", { name: /Judge Score/ }));
    // The Compare chart is the first chart in the results panel.
    return screen.getAllByTestId("chart-tooltip")[0];
  }

  test("a failed or unjudged slot shows no score, not 0.0/5", async () => {
    const tooltip = await judgeAndRankByScore();
    expect(tooltip).toHaveTextContent("Ollama Llama · Direct: 4.5/5");
    expect(tooltip).toHaveTextContent("LM Studio Qwen · Direct: Not judged");
    expect(tooltip).toHaveTextContent("Mock vLLM · Direct: Failed — no result");
    expect(tooltip).not.toHaveTextContent("0.0/5");
  });

  test("the score axis runs 0–5", async () => {
    await judgeAndRankByScore();
    const axis = screen.getAllByTestId("chart-value-axis")[0];
    expect(JSON.parse(axis.dataset.domain!)).toEqual([0, 5]);
    expect(JSON.parse(axis.dataset.ticks!)).toEqual([0, 1, 2, 3, 4, 5]);
  });
});

describe("LLM Tuner upload", () => {
  test("Upload file is reachable with Tab and opens the file picker from the keyboard", async () => {
    mockSWR();
    render(<ComparePage />);
    const user = userEvent.setup();
    const pickerOpened = vi.spyOn(HTMLInputElement.prototype, "click");

    // Tab from the content box lands on the upload control.
    screen.getByPlaceholderText(/Paste text content here/).focus();
    await user.tab();
    const upload = screen.getByRole("button", { name: "Upload file" });
    expect(upload).toHaveFocus();

    await user.keyboard("{Enter}");
    expect(pickerOpened).toHaveBeenCalled();
    expect(pickerOpened.mock.contexts[0]).toBe(screen.getByTestId("compare-file-input"));
    pickerOpened.mockRestore();
  });
});

describe("LLM provider chips", () => {
  test("a long unbroken name is cut off inside its chip and kept whole in the tooltip", () => {
    const longName = "vLLM" + "dfsdfsdfsdfdfsdfsdfsdgdsgdsgdsgdsg".repeat(6);
    mockSWR([{ ...PROVIDER, id: "lp-long", name: longName }]);
    render(<ComparePage />);

    const name = screen.getByText(longName);
    const chip = name.closest("button")!;
    // jsdom has no layout: check the classes that bound the chip and clip the text.
    expect(chip).toHaveClass("max-w-xs");
    expect(name).toHaveClass("truncate", "max-w-full", "block");
    expect(chip).toHaveAttribute("title", `${longName} — ${PROVIDER.model}`);
  });
});

describe("Slot labels", () => {
  test("spell the official engine for people, never as rlm_official", async () => {
    const official = makeSlot({ mode: "rlm_official", label: "LM Studio Qwen · rlm_official" });
    await runWith([official]);

    const label = "LM Studio Qwen · Official RLM (rlms)";
    expect(within(card(official)).getByText(label)).toBeInTheDocument();
    expect(screen.getByTestId("chart-tooltip")).toHaveTextContent(label);
    expect(document.body.textContent).not.toMatch(/rlm_official/i);
  });

  test("leave a label the server already spelled alone", async () => {
    const custom = makeSlot({ label: "My custom slot" });
    await runWith([custom]);
    expect(within(card(custom)).getByText("My custom slot")).toBeInTheDocument();
  });
});
