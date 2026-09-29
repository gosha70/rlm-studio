/**
 * Official-engine gating (specs/interop-official-rlm FR-7).
 *
 * - `unavailableReasonFor` is the single rule every picker applies.
 * - `EngineBadge` renders only for `rlm_official` runs and carries the
 *   installed engine version — in the traces table and its detail panel.
 * - The Compare mode picker disables the option and shows the server's
 *   reason as the tooltip when `GET /api/engines` says unavailable, and
 *   drops the mode from the selection if it was picked before the status
 *   arrived, so a run can never carry a mode the backend rejects.
 */

import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { vi, describe, test, expect, beforeEach, afterEach } from "vitest";
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

import { EngineBadge } from "@/components/shared/engine-badge";
import { ENGINES_SWR_KEY, unavailableReasonFor } from "@/components/shared/use-engines";
import {
  OFFICIAL_STRATEGY_UNAVAILABLE_SUFFIX,
  StrategySelectItems,
} from "@/components/settings/strategy-select-items";
import { Select, SelectContent, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  ALL_EXECUTION_MODES,
  displayModeName,
  MODE_DIRECT,
  MODE_DISPLAY_NAMES,
  MODE_RAG,
  MODE_RLM,
  MODE_RLM_OFFICIAL,
} from "@/lib/constants";
import type { EnginesResponse, ExecutionSummary, LLMProviderConfig } from "@/lib/api";
import ComparePage from "@/app/compare/page";
import TracesPage from "@/app/traces/page";

const INSTALL_HINT = 'The official RLM engine needs the `interop` extra: pip install "rlm-studio[interop]"';

/** Every picker labels a mode through the shared display-name map. */
const OFFICIAL_LABEL = displayModeName(MODE_RLM_OFFICIAL);
const RLM_LABEL = displayModeName(MODE_RLM);

const AVAILABLE: EnginesResponse = {
  rlm_official: { available: true, reason: "rlms 0.1.3", version: "0.1.3" },
};
const UNAVAILABLE: EnginesResponse = {
  rlm_official: { available: false, reason: INSTALL_HINT, version: null },
};

const CONNECTED_PROVIDER: LLMProviderConfig = {
  id: "lp-1",
  name: "GPT-4o",
  backend: "openai",
  model: "gpt-4o",
  runtime_settings: {
    temperature: 0.2,
    top_p: 1,
    max_output_tokens: 1024,
    timeout_seconds: 60,
  },
  status: "connected",
};

const makeExecution = (overrides: Partial<ExecutionSummary> = {}): ExecutionSummary => ({
  execution_id: "exec-1",
  session_id: "sess-1",
  query: "What is 2 + 2?",
  mode: MODE_RLM,
  status: "complete",
  started_at: "2024-01-01T00:00:00Z",
  completed_at: "2024-01-01T00:00:01Z",
  total_tokens: 300,
  total_cost: 0.012,
  chat_provider_name: "GPT-4o Direct",
  ...overrides,
});

interface MockData {
  /** Engine status, or undefined to model "not answered yet". */
  engines?: EnginesResponse;
  llmProviders?: LLMProviderConfig[];
  executions?: ExecutionSummary[];
}

function mockSWR(engines: EnginesResponse | undefined, extra: Omit<MockData, "engines"> = {}) {
  const stub = (data: unknown) => ({
    data,
    mutate: vi.fn(),
    error: undefined,
    isLoading: false,
    isValidating: false,
  });
  vi.mocked(useSWR).mockImplementation(((key: unknown) => {
    if (key === ENGINES_SWR_KEY) return stub(engines);
    if (key === "llm-providers") return stub(extra.llmProviders ?? []);
    if (key === "profiles" || key === "sessions") return stub([]);
    // The traces page keys its two execution fetches as arrays.
    if (Array.isArray(key) && String(key[0]).startsWith("executions")) {
      return stub(extra.executions ?? []);
    }
    return stub(undefined);
  }) as typeof useSWR);
}

describe("unavailableReasonFor", () => {
  test("built-in modes are never gated", () => {
    expect(unavailableReasonFor(MODE_DIRECT, UNAVAILABLE)).toBeNull();
    expect(unavailableReasonFor(MODE_RLM, UNAVAILABLE)).toBeNull();
  });

  test("rlm_official returns the server's reason when unavailable", () => {
    expect(unavailableReasonFor(MODE_RLM_OFFICIAL, UNAVAILABLE)).toBe(INSTALL_HINT);
  });

  test("rlm_official is selectable when available or while unknown", () => {
    expect(unavailableReasonFor(MODE_RLM_OFFICIAL, AVAILABLE)).toBeNull();
    expect(unavailableReasonFor(MODE_RLM_OFFICIAL, undefined)).toBeNull();
  });
});

describe("EngineBadge", () => {
  beforeEach(() => mockSWR(AVAILABLE));

  test("renders nothing for Studio's own modes", () => {
    const { container } = render(<EngineBadge mode={MODE_RLM} />);
    expect(container).toBeEmptyDOMElement();
  });

  test("shows the installed engine version for rlm_official", () => {
    render(<EngineBadge mode={MODE_RLM_OFFICIAL} />);
    expect(screen.getByText("official rlms 0.1.3")).toBeInTheDocument();
  });

  test("omits the version when it is unknown", () => {
    mockSWR(undefined);
    render(<EngineBadge mode={MODE_RLM_OFFICIAL} />);
    expect(screen.getByText("official rlms")).toBeInTheDocument();
  });
});

describe("Compare mode picker", () => {
  test("disables rlm_official and explains why when the engine is unavailable", () => {
    mockSWR(UNAVAILABLE);
    render(<ComparePage />);

    const button = screen.getByRole("button", { name: OFFICIAL_LABEL });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute("title", INSTALL_HINT);
    // Built-in modes are untouched by the engine status.
    expect(screen.getByRole("button", { name: RLM_LABEL })).toBeEnabled();
  });

  test("offers rlm_official normally when the engine is available", () => {
    mockSWR(AVAILABLE);
    render(<ComparePage />);

    const button = screen.getByRole("button", { name: OFFICIAL_LABEL });
    expect(button).toBeEnabled();
    expect(button).toHaveAttribute("title", "Paper authors' reference implementation (rlms)");
  });
});

describe("Compare selection when the engine status arrives late", () => {
  afterEach(() => vi.unstubAllGlobals());

  /** Pick rlm_official while availability is still unknown, then answer. */
  const selectThenReport = (answer: EnginesResponse, extra: Omit<MockData, "engines"> = {}) => {
    mockSWR(undefined, extra);
    const view = render(<ComparePage />);
    fireEvent.click(screen.getByRole("button", { name: OFFICIAL_LABEL }));
    expect(screen.getByRole("button", { name: OFFICIAL_LABEL })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    mockSWR(answer, extra);
    view.rerender(<ComparePage />);
    return view;
  };

  test("deselects the mode once the engine is reported unavailable", () => {
    selectThenReport(UNAVAILABLE);

    const button = screen.getByRole("button", { name: OFFICIAL_LABEL });
    // Still un-clickable, but no longer part of the run.
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByText("Modes (1 selected)")).toBeInTheDocument();
  });

  test("keeps the mode selected when the engine is reported available", () => {
    selectThenReport(AVAILABLE);

    const button = screen.getByRole("button", { name: OFFICIAL_LABEL });
    expect(button).toBeEnabled();
    expect(button).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("Modes (2 selected)")).toBeInTheDocument();
  });

  test("submits a matrix without the unavailable mode, so the run cannot 400", async () => {
    selectThenReport(UNAVAILABLE, { llmProviders: [CONNECTED_PROVIDER] });

    fireEvent.change(screen.getByPlaceholderText("What would you like to ask?"), {
      target: { value: "Summarize this" },
    });
    fireEvent.change(screen.getByPlaceholderText(/Paste text content here/), {
      target: { value: "a long document" },
    });

    // submitCompareMatrix posts with bare fetch, so the wire body is visible here.
    const fetchMock = vi.fn().mockRejectedValue(new Error("stub: body already captured"));
    vi.stubGlobal("fetch", fetchMock);

    const run = screen.getByRole("button", { name: /Run Compare/ });
    expect(run).toBeEnabled();
    fireEvent.click(run);

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("/api/chat/compare-matrix");
    expect(JSON.parse(String(init.body)).modes).toEqual([MODE_DIRECT]);
  });
});

describe("useEngineAvailability fetch options", () => {
  test("caps error retries so a backend without /api/engines is not polled forever", () => {
    mockSWR(undefined);
    render(<EngineBadge mode={MODE_RLM_OFFICIAL} />);

    expect(vi.mocked(useSWR)).toHaveBeenCalledWith(
      ENGINES_SWR_KEY,
      expect.any(Function),
      expect.objectContaining({ revalidateOnFocus: false, errorRetryCount: 2 }),
    );
  });
});

describe("Traces engine badge", () => {
  test("labels an official-engine run in the table, not just the detail panel", () => {
    mockSWR(AVAILABLE, {
      executions: [
        makeExecution({ execution_id: "exec-official", mode: MODE_RLM_OFFICIAL }),
        makeExecution({ execution_id: "exec-studio", query: "Explain recursion" }),
      ],
    });
    render(<TracesPage />);

    // One row is rlm_official, the other is Studio's own RLM: exactly one badge.
    expect(screen.getAllByText("official rlms 0.1.3")).toHaveLength(1);
    expect(screen.getByText("RLM_OFFICIAL")).toBeInTheDocument();
  });

  test("shows no engine badge when every run used a built-in mode", () => {
    mockSWR(AVAILABLE, { executions: [makeExecution()] });
    render(<TracesPage />);

    expect(screen.queryByText(/official rlms/)).not.toBeInTheDocument();
  });
});

describe("Profile strategy options", () => {
  // Radix Select renders its items only while open; jsdom lacks the two
  // pointer/scroll APIs the primitive touches on open.
  beforeEach(() => {
    Element.prototype.scrollIntoView = vi.fn();
    Element.prototype.hasPointerCapture = vi.fn().mockReturnValue(false);
  });

  const renderOpen = () =>
    render(
      <Select open value={MODE_DIRECT}>
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <StrategySelectItems />
        </SelectContent>
      </Select>,
    );

  test("lists the official engine as a disabled option with the reason when unavailable", () => {
    mockSWR(UNAVAILABLE);
    renderOpen();

    const option = screen.getByRole("option", {
      name: OFFICIAL_LABEL + OFFICIAL_STRATEGY_UNAVAILABLE_SUFFIX,
    });
    expect(option).toHaveAttribute("aria-disabled", "true");
    expect(option).toHaveAttribute("title", INSTALL_HINT);
    expect(screen.getByRole("option", { name: RLM_LABEL })).not.toHaveAttribute(
      "aria-disabled",
      "true",
    );
  });

  test("offers the official engine normally when available", () => {
    mockSWR(AVAILABLE);
    renderOpen();

    const option = screen.getByRole("option", { name: OFFICIAL_LABEL });
    expect(option).not.toHaveAttribute("aria-disabled", "true");
  });
});

describe("Mode wording", () => {
  test("the display-name map is the only spelling of each mode", () => {
    // Guards the wording itself: the selects, the Compare picker and the
    // settings badges all read these strings, so a change here is a change
    // everywhere rather than a fourth spelling.
    expect(MODE_DISPLAY_NAMES).toEqual({
      [MODE_DIRECT]: "Direct",
      [MODE_RLM]: "RLM",
      [MODE_RAG]: "RAG",
      [MODE_RLM_OFFICIAL]: "Official RLM (rlms)",
    });
  });

  test("an unknown mode falls back to its identifier", () => {
    expect(displayModeName("some_future_mode")).toBe("some_future_mode");
  });

  test("the Compare picker labels modes, never raw identifiers", () => {
    mockSWR(AVAILABLE);
    render(<ComparePage />);

    for (const mode of ALL_EXECUTION_MODES) {
      expect(screen.getByRole("button", { name: displayModeName(mode) })).toBeInTheDocument();
      // `rlm_official` is the only identifier that differs from its label.
      if (displayModeName(mode) !== mode) {
        expect(screen.queryByRole("button", { name: mode })).not.toBeInTheDocument();
      }
    }
  });

  test("the profile strategy select labels modes, never raw identifiers", () => {
    Element.prototype.scrollIntoView = vi.fn();
    Element.prototype.hasPointerCapture = vi.fn().mockReturnValue(false);
    mockSWR(AVAILABLE);
    render(
      <Select open value={MODE_DIRECT}>
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <StrategySelectItems />
        </SelectContent>
      </Select>,
    );

    expect(screen.getByRole("option", { name: OFFICIAL_LABEL })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: MODE_RLM_OFFICIAL })).not.toBeInTheDocument();
  });
});
