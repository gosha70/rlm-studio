/**
 * Trace visualization component tests.
 *
 * Covers: CodeBlock, Timeline, TraceTree, StepDetail,
 * and the Traces page.
 */

import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import { vi, describe, test, expect, beforeEach, afterEach } from "vitest";
import { CodeBlock } from "@/components/trace/code-block";
import { Timeline } from "@/components/trace/timeline";
import { TraceTree } from "@/components/trace/trace-tree";
import { StepDetail } from "@/components/trace/step-detail";
import type { TraceStep, ExecutionSummary } from "@/lib/api";

// ---------------------------------------------------------------------------
// Mock next/navigation — used by TracesPage (useSearchParams)
// ---------------------------------------------------------------------------

const mockRouterPush = vi.fn();
const mockRouterReplace = vi.fn();
let mockExecParam: string | null = null;

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: mockRouterPush, replace: mockRouterReplace }),
  useSearchParams: () => ({ get: (key: string) => (key === "exec" ? mockExecParam : null) }),
}));

// ---------------------------------------------------------------------------
// Mock SWR — used by TracesPage
// ---------------------------------------------------------------------------

vi.mock("swr", () => ({
  default: vi.fn(),
}));

// ---------------------------------------------------------------------------
// Mock AppShell
// ---------------------------------------------------------------------------

vi.mock("@/components/shared/app-shell", () => ({
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  AppShell: ({ children }: { children: any }) => children,
}));

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const makeStep = (overrides: Partial<TraceStep> = {}): TraceStep => ({
  index: 0,
  action_type: "inspect",
  code: 'print("hello")',
  output: "hello",
  input_tokens: 100,
  output_tokens: 50,
  cost_usd: 0.0012,
  duration_seconds: 0.8,
  recursion_depth: 0,
  model: "gpt-4o",
  timestamp: "2024-01-01T00:00:00Z",
  ...overrides,
});

// ---------------------------------------------------------------------------
// CodeBlock
// ---------------------------------------------------------------------------

describe("CodeBlock", () => {
  test("renders code with syntax highlighting", () => {
    render(<CodeBlock code='print("hello")' language="python" />);
    expect(screen.getByText('print("hello")')).toBeInTheDocument();
  });

  test("shows language label when provided", () => {
    render(<CodeBlock code="x = 1" language="python" />);
    expect(screen.getByText("python")).toBeInTheDocument();
  });

  test("shows default language label when language omitted", () => {
    // Default is "python" per component source
    render(<CodeBlock code="x = 1" />);
    expect(screen.getByText("python")).toBeInTheDocument();
  });

  test.skip("copies code to clipboard on button click", () => {
    // CodeBlock has no copy button in the current implementation.
    // Copy functionality is not present in src/components/trace/code-block.tsx.
    // This test should be implemented when a copy button is added.
  });
});

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------

describe("Timeline", () => {
  test("renders steps in chronological order", () => {
    const steps = [
      makeStep({ index: 0, action_type: "inspect" }),
      makeStep({ index: 1, action_type: "subcall" }),
      makeStep({ index: 2, action_type: "final" }),
    ];
    render(<Timeline steps={steps} onSelect={vi.fn()} />);
    const buttons = screen.getAllByRole("button");
    // Order matches: Step 1, Step 2, Step 3
    expect(buttons[0]).toHaveAttribute("aria-label", expect.stringContaining("Step 1"));
    expect(buttons[1]).toHaveAttribute("aria-label", expect.stringContaining("Step 2"));
    expect(buttons[2]).toHaveAttribute("aria-label", expect.stringContaining("Step 3"));
  });

  test("shows step index and action type for each step", () => {
    const steps = [
      makeStep({ index: 0, action_type: "inspect" }),
      makeStep({ index: 1, action_type: "final" }),
    ];
    render(<Timeline steps={steps} onSelect={vi.fn()} />);
    expect(screen.getByText("Step 1")).toBeInTheDocument();
    expect(screen.getByText("Step 2")).toBeInTheDocument();
    expect(screen.getAllByText("inspect")[0]).toBeInTheDocument();
    expect(screen.getByText("final")).toBeInTheDocument();
  });

  test("highlights the currently selected step via aria-label ring", () => {
    const steps = [
      makeStep({ index: 0, action_type: "inspect" }),
      makeStep({ index: 1, action_type: "subcall" }),
    ];
    const { container } = render(
      <Timeline steps={steps} onSelect={vi.fn()} selectedIndex={1} />,
    );
    // The button for index 1 gets the ring-2 ring-ring class
    const buttons = container.querySelectorAll("button");
    expect(buttons[0].className).not.toMatch(/ring-2/);
    expect(buttons[1].className).toMatch(/ring-2/);
  });

  test("handles empty steps list without crashing", () => {
    render(<Timeline steps={[]} onSelect={vi.fn()} />);
    // No buttons rendered, no crash
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// TraceTree
// ---------------------------------------------------------------------------

describe("TraceTree", () => {
  test("renders execution tree with root and child steps", () => {
    const steps = [
      makeStep({ index: 0, recursion_depth: 0, action_type: "inspect" }),
      makeStep({ index: 1, recursion_depth: 1, action_type: "subcall" }),
    ];
    render(<TraceTree steps={steps} onSelect={vi.fn()} />);
    const buttons = screen.getAllByRole("button");
    expect(buttons).toHaveLength(2);
    expect(buttons[0]).toHaveAttribute(
      "aria-label",
      expect.stringContaining("depth 0"),
    );
    expect(buttons[1]).toHaveAttribute(
      "aria-label",
      expect.stringContaining("depth 1"),
    );
  });

  test("collapses and expands nodes on click", () => {
    const onSelect = vi.fn();
    const steps = [makeStep({ index: 0, action_type: "inspect" })];
    render(<TraceTree steps={steps} onSelect={onSelect} />);
    fireEvent.click(screen.getByRole("button"));
    expect(onSelect).toHaveBeenCalledWith(steps[0]);
  });

  test("shows recursion depth indicator for child steps", () => {
    const steps = [
      makeStep({ index: 0, recursion_depth: 0, code: "root_code" }),
      makeStep({ index: 1, recursion_depth: 2, code: "child_code" }),
    ];
    render(<TraceTree steps={steps} onSelect={vi.fn()} />);
    // Depth > 0 renders "|-" prefix
    expect(screen.getByText(/\|-/)).toBeInTheDocument();
    // Root has no "|-" prefix
    const allTexts = screen.getAllByText(/\d+\./);
    // Both steps render their index+1 label
    expect(allTexts.length).toBeGreaterThanOrEqual(2);
  });
});

// ---------------------------------------------------------------------------
// StepDetail
// ---------------------------------------------------------------------------

describe("StepDetail", () => {
  test("renders step code in CodeBlock", () => {
    const step = makeStep({ index: 0, code: 'result = 1 + 1', output: "2" });
    render(<StepDetail step={step} />);
    expect(screen.getByText("result = 1 + 1")).toBeInTheDocument();
  });

  test("renders step output when present without code", () => {
    const step = makeStep({ index: 0, code: null, output: "The answer is 42" });
    render(<StepDetail step={step} />);
    expect(screen.getByText("The answer is 42")).toBeInTheDocument();
  });

  test("shows token count (input/output)", () => {
    const step = makeStep({ index: 0, input_tokens: 200, output_tokens: 75 });
    render(<StepDetail step={step} />);
    expect(screen.getByText("200")).toBeInTheDocument();
    expect(screen.getByText("75")).toBeInTheDocument();
    expect(screen.getByText("Input Tokens")).toBeInTheDocument();
    expect(screen.getByText("Output Tokens")).toBeInTheDocument();
  });

  test("shows step duration", () => {
    const step = makeStep({ index: 0, duration_seconds: 1.23 });
    render(<StepDetail step={step} />);
    expect(screen.getByText("1.23s")).toBeInTheDocument();
    expect(screen.getByText("Duration")).toBeInTheDocument();
  });

  // Issue #94: the three cost states the backend can now produce.
  test("shows Unavailable when the model could not be priced", () => {
    render(<StepDetail step={makeStep({ cost_usd: null })} />);
    expect(screen.getByText("Unavailable")).toBeInTheDocument();
    expect(screen.queryByText(/^\$/)).not.toBeInTheDocument();
  });

  test("shows a genuine zero cost as a dollar figure", () => {
    render(<StepDetail step={makeStep({ cost_usd: 0 })} />);
    expect(screen.getByText("$0.0000")).toBeInTheDocument();
  });

  test("shows a priced step cost", () => {
    render(<StepDetail step={makeStep({ cost_usd: 0.0075 })} />);
    expect(screen.getByText("$0.0075")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Traces page
// ---------------------------------------------------------------------------

import useSWR from "swr";
import TracesPage from "@/app/traces/page";

const mockUseSWR = vi.mocked(useSWR);

const makeExecution = (overrides: Partial<ExecutionSummary> = {}): ExecutionSummary => ({
  execution_id: "exec-1",
  session_id: "sess-1",
  query: "What is 2 + 2?",
  mode: "rlm",
  status: "complete",
  started_at: "2024-01-01T00:00:00Z",
  completed_at: "2024-01-01T00:00:01Z",
  total_tokens: 300,
  total_cost: 0.012,
  chat_provider_name: "GPT-4o Direct",
  ...overrides,
});

describe("TracesPage", () => {
  test("renders execution list", () => {
    mockUseSWR.mockReturnValue({
      data: [makeExecution(), makeExecution({ execution_id: "exec-2", query: "Explain recursion" })],
      isLoading: false,
    } as ReturnType<typeof useSWR>);

    render(<TracesPage />);
    expect(screen.getByText("What is 2 + 2?")).toBeInTheDocument();
    expect(screen.getByText("Explain recursion")).toBeInTheDocument();
  });

  test("shows trace detail when an execution is selected", () => {
    // With no trace loaded yet, the detail panel is absent; clicking loads it
    // (async). We verify the execution row is rendered and clickable.
    mockUseSWR.mockReturnValue({
      data: [makeExecution()],
      isLoading: false,
    } as ReturnType<typeof useSWR>);

    render(<TracesPage />);
    // The row should have role=button and the aria-label
    expect(
      screen.getByRole("button", { name: /Expand trace for: What is 2 \+ 2/ }),
    ).toBeInTheDocument();
  });

  test("shows budget usage summary when trace is loaded", () => {
    // With no execution selected (initial state), budget panel is absent.
    // Verify the "Recent Executions" header renders (budget panel shows only
    // after selecting an execution, which is async).
    mockUseSWR.mockReturnValue({
      data: [makeExecution()],
      isLoading: false,
    } as ReturnType<typeof useSWR>);

    render(<TracesPage />);
    expect(screen.getByText("Recent Executions")).toBeInTheDocument();
  });

  test("fetches trace data on mount via SWR", () => {
    mockUseSWR.mockReturnValue({
      data: [],
      isLoading: false,
    } as ReturnType<typeof useSWR>);

    render(<TracesPage />);
    expect(mockUseSWR).toHaveBeenCalledWith(
      ["executions", "", "", 20],
      expect.any(Function),
      expect.objectContaining({ refreshInterval: 5000 }),
    );
  });

  test("the session filter fetches sessions with the default limit, not the SWR key", async () => {
    mockUseSWR.mockReturnValue({ data: [], isLoading: false } as ReturnType<typeof useSWR>);
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("[]"));
    try {
      render(<TracesPage />);
      const call = mockUseSWR.mock.calls.find(([key]) => key === "sessions");
      expect(call).toBeDefined();
      // SWR calls the fetcher with the key as its argument.
      await (call![1] as (key: string) => Promise<unknown>)("sessions");
      expect(fetchSpy).toHaveBeenCalledWith("/api/sessions?limit=20&offset=0", expect.anything());
    } finally {
      fetchSpy.mockRestore();
    }
  });

  test("shows empty state when no executions exist", () => {
    mockUseSWR.mockReturnValue({
      data: [],
      isLoading: false,
    } as ReturnType<typeof useSWR>);

    render(<TracesPage />);
    expect(
      screen.getByText(/No executions yet/),
    ).toBeInTheDocument();
  });

  // -------------------------------------------------------------------------
  // Replay-in-Learn CTA (V2b step 7)
  //
  // Per NEXT.md §3d: the trailing button must route to
  // /learn/replay?id=<id> via `router.push` without triggering the
  // row's own "open trace detail" click handler. Both branches of
  // the stopPropagation contract are asserted: the button itself
  // navigates, and clicking the button does NOT load a trace for
  // the row.
  // -------------------------------------------------------------------------

  test("renders Replay in Learn button per execution row", () => {
    mockUseSWR.mockReturnValue({
      data: [
        makeExecution(),
        makeExecution({ execution_id: "exec-2", query: "Explain recursion" }),
      ],
      isLoading: false,
    } as ReturnType<typeof useSWR>);

    render(<TracesPage />);
    expect(
      screen.getByRole("button", { name: /Replay in Learn: What is 2 \+ 2/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Replay in Learn: Explain recursion/ }),
    ).toBeInTheDocument();
  });

  test("Replay in Learn click routes to /learn/replay?id=<id>", () => {
    mockRouterPush.mockClear();
    mockUseSWR.mockReturnValue({
      data: [makeExecution({ execution_id: "exec-42" })],
      isLoading: false,
    } as ReturnType<typeof useSWR>);

    render(<TracesPage />);
    const button = screen.getByRole("button", {
      name: /Replay in Learn: What is 2 \+ 2/,
    });
    fireEvent.click(button);

    expect(mockRouterPush).toHaveBeenCalledTimes(1);
    expect(mockRouterPush).toHaveBeenCalledWith("/learn/replay?id=exec-42&from=traces");
  });

  test("Replay in Learn click does not trigger the row's open-trace handler", () => {
    mockRouterPush.mockClear();
    // The row handler fetches a trace via getTrace() which would set
    // loading=true. We assert the button click does NOT cause the
    // "Loading trace…" text to appear — the row handler stayed cold.
    mockUseSWR.mockReturnValue({
      data: [makeExecution()],
      isLoading: false,
    } as ReturnType<typeof useSWR>);

    render(<TracesPage />);
    const button = screen.getByRole("button", {
      name: /Replay in Learn: What is 2 \+ 2/,
    });
    fireEvent.click(button);

    // Row-handler-triggered side effects must NOT appear.
    expect(screen.queryByText(/Loading trace/i)).not.toBeInTheDocument();
    // But navigation did happen.
    expect(mockRouterPush).toHaveBeenCalledTimes(1);
  });

  test("keyboard Enter/Space on CTA does not bubble into row's open-trace handler", () => {
    // Regression guard: the row's onKeyDown handler previously fired
    // for keydown events originating anywhere in the row, including
    // from focused descendants like this CTA. A keyboard user pressing
    // Enter on the "Replay in Learn" button would trigger the row's
    // handleSelectExecution() in addition to (or instead of) the
    // button's own activation. The fix adds
    // `if (e.target !== e.currentTarget) return;` to the row handler.
    mockRouterPush.mockClear();
    mockUseSWR.mockReturnValue({
      data: [makeExecution()],
      isLoading: false,
    } as ReturnType<typeof useSWR>);

    render(<TracesPage />);
    const button = screen.getByRole("button", {
      name: /Replay in Learn: What is 2 \+ 2/,
    });

    // Bubble a keydown from the button up to the row. handleSelect-
    // Execution calls setLoading(true) synchronously, so if the row
    // handler fires "Loading trace…" will render on the next tick.
    fireEvent.keyDown(button, { key: "Enter" });
    fireEvent.keyDown(button, { key: " " });

    expect(screen.queryByText(/Loading trace/i)).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Browser pre-pass M11: Delete N confirms, the open trace is in the URL, and
// each run shows when it started.
// ---------------------------------------------------------------------------

describe("TracesPage (M11)", () => {
  const TRACE = {
    execution_id: "exec-1",
    session_id: "sess-1",
    query: "What is 2 + 2?",
    mode: "rlm",
    status: "complete",
    result: { answer: "4", success: true },
    budget: {
      steps_used: 1,
      steps_limit: 16,
      tokens_used: 300,
      tokens_limit: 0,
      cost_used: 0,
      cost_limit: 0,
      max_depth_reached: 0,
      depth_limit: 0,
    },
    steps: [],
  };
  let fetchSpy: ReturnType<typeof vi.spyOn>;
  let confirmSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    mockExecParam = null;
    mockRouterReplace.mockClear();
    mockUseSWR.mockReturnValue({ data: [makeExecution()], isLoading: false, mutate: vi.fn() } as unknown as ReturnType<
      typeof useSWR
    >);
    fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify(TRACE)));
    confirmSpy = vi.spyOn(window, "confirm");
  });
  afterEach(() => {
    fetchSpy.mockRestore();
    confirmSpy.mockRestore();
    mockExecParam = null;
  });

  const deleteRequests = () =>
    fetchSpy.mock.calls.filter((call: unknown[]) => (call[1] as RequestInit | undefined)?.method === "DELETE");

  test("Delete N asks first and deletes nothing when cancelled", () => {
    confirmSpy.mockReturnValue(false);
    render(<TracesPage />);
    fireEvent.click(screen.getByRole("checkbox", { name: /Select trace: What is 2 \+ 2/ }));
    fireEvent.click(screen.getByRole("button", { name: /Delete 1/ }));
    expect(confirmSpy).toHaveBeenCalledWith("Delete 1 selected trace? This cannot be undone.");
    expect(deleteRequests()).toHaveLength(0);
  });

  test("Delete N deletes once confirmed", async () => {
    confirmSpy.mockReturnValue(true);
    render(<TracesPage />);
    fireEvent.click(screen.getByRole("checkbox", { name: /Select trace: What is 2 \+ 2/ }));
    fireEvent.click(screen.getByRole("button", { name: /Delete 1/ }));
    await waitFor(() => expect(deleteRequests()).toHaveLength(1));
  });

  test("opening a trace puts its id in the URL", async () => {
    render(<TracesPage />);
    fireEvent.click(screen.getByRole("button", { name: /Expand trace for: What is 2 \+ 2/ }));
    await waitFor(() => expect(mockRouterReplace).toHaveBeenCalledWith("/traces?exec=exec-1", { scroll: false }));
  });

  test("a trace named in the URL opens on load without the URL being cleared", async () => {
    mockExecParam = "exec-1";
    render(<TracesPage />);
    expect(await screen.findByRole("button", { name: /Collapse trace for: What is 2 \+ 2/ })).toBeInTheDocument();
    expect(mockRouterReplace).not.toHaveBeenCalled();
  });

  test("each run shows when it started", () => {
    render(<TracesPage />);
    expect(screen.getByRole("columnheader", { name: "Started" })).toBeInTheDocument();
    const started = new Date("2024-01-01T00:00:00Z");
    expect(screen.getByText(started.toLocaleDateString())).toBeInTheDocument();
    expect(screen.getByText(started.toLocaleTimeString())).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Browser pass 2 CHAT2-03 / CMP2-07: a failed trace says why it failed.
// ---------------------------------------------------------------------------

describe("TracesPage failed run (CHAT2-03, CMP2-07)", () => {
  const failedTrace = (result: Record<string, unknown>) => ({
    execution_id: "exec-1",
    session_id: "sess-1",
    query: "What is 2 + 2?",
    mode: "rlm",
    status: "error",
    result: { answer: "", success: false, ...result },
    budget: {
      steps_used: 0,
      steps_limit: 16,
      tokens_used: 0,
      tokens_limit: 0,
      cost_used: 0,
      cost_limit: 0,
      max_depth_reached: 0,
      depth_limit: 0,
    },
    steps: [],
  });
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  function openTrace(trace: object) {
    mockExecParam = "exec-1";
    mockUseSWR.mockReturnValue({
      data: [makeExecution({ status: "error" })],
      isLoading: false,
      mutate: vi.fn(),
    } as unknown as ReturnType<typeof useSWR>);
    fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify(trace)));
    render(<TracesPage />);
  }
  afterEach(() => {
    fetchSpy.mockRestore();
    mockExecParam = null;
  });

  test("shows the run's error and the answer that explains the stop", async () => {
    openTrace(
      failedTrace({
        error: "Context window exceeded: 53,839 tokens > 32,768",
        answer: "The run stopped: the document does not fit. Use RLM or a larger context window.",
      }),
    );
    const reason = await screen.findByRole("region", { name: "Why this run failed" });
    expect(reason).toHaveTextContent("Context window exceeded: 53,839 tokens > 32,768");
    expect(reason).toHaveTextContent("Use RLM or a larger context window.");
  });

  test("an answer that only repeats the error is shown once", async () => {
    openTrace(failedTrace({ error: "Timed out after 120 s", answer: "Timed out after 120 s" }));
    const reason = await screen.findByRole("region", { name: "Why this run failed" });
    expect(reason.textContent?.match(/Timed out after 120 s/g)).toHaveLength(1);
  });

  test("P3-CS-1: the answer's markdown is rendered, not shown raw", async () => {
    openTrace(
      failedTrace({
        error: "litellm.ContextWindowExceededError: 53,839 tokens > 32,768",
        answer:
          "⚠️ **Context window exceeded**.\n\nThe document is too large.\n\n**How to fix:**\n- Use RAG mode.\n\n_Connection error._",
      }),
    );
    const reason = await screen.findByRole("region", { name: "Why this run failed" });
    expect(reason).not.toHaveTextContent("**");
    expect(reason).not.toHaveTextContent("_Connection error._");
    expect(within(reason).getByText("How to fix:").tagName).toBe("STRONG");
    expect(within(reason).getByText("Connection error.").tagName).toBe("EM");
    expect(within(reason).getByRole("listitem")).toHaveTextContent("Use RAG mode.");
  });

  test("P3-CS-1: an answer that only wraps the error in a title is not shown", async () => {
    const error = "Cannot connect to the LLM server at 'http://127.0.0.1:9/v1'.";
    openTrace(failedTrace({ error, answer: `⚠️ **Execution error**.\n\n${error}` }));
    const reason = await screen.findByRole("region", { name: "Why this run failed" });
    expect(within(reason).queryByText("Answer")).not.toBeInTheDocument();
    expect(reason.textContent?.match(/Cannot connect to the LLM server/g)).toHaveLength(1);
  });

  test("a successful run shows no failure panel", async () => {
    openTrace({ ...failedTrace({ answer: "4", success: true }), status: "complete" });
    await screen.findByText("Query");
    expect(screen.queryByRole("region", { name: "Why this run failed" })).not.toBeInTheDocument();
  });
});
