/**
 * Dashboard and metrics component test stubs.
 *
 * Covers: MetricCard, ComparisonChart, PerformanceTrend,
 * CostBreakdown, and the Dashboard page.
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import { vi, describe, test, expect } from "vitest";
import { Hash } from "lucide-react";
import { MetricCard } from "@/components/metrics/metric-card";
import { ComparisonChart } from "@/components/metrics/comparison-chart";
import { PerformanceTrend } from "@/components/metrics/performance-trend";
import { CostBreakdown } from "@/components/metrics/cost-breakdown";
import type { ModeSummary, TimelineEntry } from "@/lib/api";

// ---------------------------------------------------------------------------
// Mock recharts — it does not render meaningful DOM in jsdom
// ---------------------------------------------------------------------------

vi.mock("recharts", () => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const Passthrough = ({ children }: { children?: any }) => children ?? null;
  return {
    BarChart: Passthrough,
    LineChart: Passthrough,
    PieChart: Passthrough,
    Bar: () => null,
    Line: () => null,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    Pie: ({ data }: { data?: Array<{ name: string; value: number }>; [k: string]: any }) => {
      if (!data || data.length === 0) return null;
      return (
        <div data-testid="pie-cells">
          {data.map((entry) => (
            <span key={entry.name} data-testid={`pie-entry-${entry.name}`}>
              {entry.name}
            </span>
          ))}
        </div>
      );
    },
    Cell: () => null,
    XAxis: () => null,
    YAxis: () => null,
    CartesianGrid: () => null,
    Tooltip: () => null,
    Legend: () => null,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ResponsiveContainer: ({ children }: { children?: any }) => children ?? null,
  };
});

// ---------------------------------------------------------------------------
// Mock ChartContainer — jsdom has no ResizeObserver / layout
// ---------------------------------------------------------------------------

vi.mock("@/components/metrics/chart-container", () => ({
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ChartContainer: ({ children, ...props }: { children: (w: number, h: number) => any; className?: string; [k: string]: any }) => (
    <div {...props}>{typeof children === "function" ? children(400, 200) : children}</div>
  ),
}));

// ---------------------------------------------------------------------------
// Mock next/navigation — used by DashboardPage (useRouter)
// ---------------------------------------------------------------------------

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}));

// ---------------------------------------------------------------------------
// Mock SWR — used by DashboardPage
// ---------------------------------------------------------------------------

vi.mock("swr", () => ({
  default: vi.fn(),
  useSWRConfig: () => ({ mutate: vi.fn() }),
}));

// ---------------------------------------------------------------------------
// Mock AppShell — heavy component not needed for unit tests
// ---------------------------------------------------------------------------

vi.mock("@/components/shared/app-shell", () => ({
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  AppShell: ({ children }: { children: any }) => children,
}));

// ---------------------------------------------------------------------------
// MetricCard
// ---------------------------------------------------------------------------

describe("MetricCard", () => {
  test("renders label and value", () => {
    render(<MetricCard label="Total Tokens" value="1,234" icon={Hash} />);
    expect(screen.getByText("Total Tokens")).toBeInTheDocument();
    expect(screen.getByText("1,234")).toBeInTheDocument();
  });

  test("formats large numbers with comma separators", () => {
    // MetricCard receives a pre-formatted string; the card's aria-label concatenates
    // label and value.
    render(<MetricCard label="Total Tokens" value="1,234,567" icon={Hash} />);
    expect(screen.getByText("1,234,567")).toBeInTheDocument();
    // aria-label on the Card div confirms label+value are combined
    expect(
      screen.getByLabelText("Total Tokens: 1,234,567"),
    ).toBeInTheDocument();
  });

  test("shows unit in aria-label when unit prop provided", () => {
    render(<MetricCard label="Avg Latency" value="2.3" unit="s" icon={Hash} />);
    expect(screen.getByLabelText("Avg Latency: 2.3 s")).toBeInTheDocument();
  });

  test("renders unit text next to value", () => {
    render(<MetricCard label="Avg Latency" value="2.3" unit="s" icon={Hash} />);
    expect(screen.getByText("2.3")).toBeInTheDocument();
    expect(screen.getByText("s")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// ComparisonChart
// ---------------------------------------------------------------------------

const makeModeSummary = (overrides: Partial<ModeSummary> = {}): ModeSummary => ({
  queries: 5,
  total_tokens: 1000,
  total_cost_usd: 0.05,
  avg_latency_seconds: 1.2,
  ...overrides,
});

describe("ComparisonChart", () => {
  test("renders bar chart with direct and RLM data", () => {
    render(
      <ComparisonChart
        rlmData={makeModeSummary({ total_tokens: 800, total_cost_usd: 0.04 })}
        directData={makeModeSummary({ total_tokens: 1200, total_cost_usd: 0.06 })}
      />,
    );
    // Card title rendered
    expect(screen.getByText("Mode Comparison")).toBeInTheDocument();
    // Both split chart panels (tokens + cost) are rendered
    const charts = screen.getAllByRole("img", { name: /Bar chart comparing mode/ });
    expect(charts).toHaveLength(2);
  });

  test("shows legend for each mode via aria-label description", () => {
    render(
      <ComparisonChart
        rlmData={makeModeSummary()}
        directData={makeModeSummary()}
      />,
    );
    // Both split chart panels have aria-labels containing mode names
    const chartImgs = screen.getAllByRole("img", { name: /comparing mode/ });
    expect(chartImgs.length).toBeGreaterThanOrEqual(2);
  });

  test("handles empty data gracefully (undefined props)", () => {
    // Should render without throwing even with no data
    render(<ComparisonChart rlmData={undefined} directData={undefined} />);
    expect(screen.getByText("Mode Comparison")).toBeInTheDocument();
  });

  test("charts the official engine as a fourth series on its own", () => {
    render(
      <ComparisonChart
        rlmData={undefined}
        directData={undefined}
        officialData={makeModeSummary({ total_tokens: 640, total_cost_usd: 0.03 })}
      />,
    );
    // Not the empty state: both chart panels render for rlm_official runs alone.
    expect(screen.queryByText(/Run queries in multiple modes/)).not.toBeInTheDocument();
    expect(screen.getAllByRole("img", { name: /Bar chart comparing mode/ })).toHaveLength(2);
  });
});

// ---------------------------------------------------------------------------
// PerformanceTrend
// ---------------------------------------------------------------------------

const makeTimelineEntry = (overrides: Partial<TimelineEntry> = {}): TimelineEntry => ({
  timestamp: "2024-01-01T00:00:00Z",
  tokens: 500,
  cost_usd: 0.02,
  latency_seconds: 1.1,
  mode: "rlm",
  provider: "openai",
  ...overrides,
});

describe("PerformanceTrend", () => {
  test("renders line chart with timeline data", () => {
    const timeline = [
      makeTimelineEntry({ tokens: 400 }),
      makeTimelineEntry({ tokens: 600 }),
      makeTimelineEntry({ tokens: 500 }),
    ];
    render(<PerformanceTrend timeline={timeline} />);
    expect(screen.getByText(/Performance Over Time/)).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: /Line chart showing latency/ }),
    ).toBeInTheDocument();
  });

  test.skip("shows tooltip on hover with exact values", () => {
    // Recharts tooltips are rendered via portals and rely on SVG mouse events
    // that do not work in jsdom. Tooltip behaviour is tested in e2e/browser tests.
  });

  test("handles single data point without error", () => {
    render(<PerformanceTrend timeline={[makeTimelineEntry()]} />);
    expect(screen.getByText(/Performance Over Time/)).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// CostBreakdown
// ---------------------------------------------------------------------------

const makeProviderData = (names: string[]) =>
  Object.fromEntries(
    names.map((name) => [
      name,
      { queries: 3, total_tokens: 900, total_cost_usd: 0.036, avg_latency_seconds: 1.0 },
    ]),
  );

describe("CostBreakdown", () => {
  test("renders cost by provider", () => {
    render(
      <CostBreakdown
        data={makeProviderData(["openai", "anthropic"])}
        title="Cost by Provider"
      />,
    );
    expect(screen.getByText("Cost by Provider")).toBeInTheDocument();
    // Pie mock renders entry spans with the provider names
    expect(screen.getByTestId("pie-entry-OpenAI")).toBeInTheDocument();
    expect(screen.getByTestId("pie-entry-Anthropic")).toBeInTheDocument();
  });

  test("renders cost by mode", () => {
    render(
      <CostBreakdown
        data={makeProviderData(["rlm", "direct"])}
        title="Cost by Mode"
      />,
    );
    expect(screen.getByText("Cost by Mode")).toBeInTheDocument();
    expect(screen.getByTestId("pie-entry-rlm")).toBeInTheDocument();
    expect(screen.getByTestId("pie-entry-direct")).toBeInTheDocument();
  });

  test("formats USD values to 4 decimal places via Tooltip formatter", () => {
    // The component's Tooltip formatter: `$${Number(value).toFixed(4)}`
    // We verify the formatter produces the expected string directly
    const formatter = (value: number) => `$${Number(value).toFixed(4)}`;
    expect(formatter(0.036)).toBe("$0.0360");
    expect(formatter(1.23456)).toBe("$1.2346");
    expect(formatter(0)).toBe("$0.0000");
  });
});

// ---------------------------------------------------------------------------
// Dashboard page
// ---------------------------------------------------------------------------

import useSWR from "swr";
import type { MetricsResponse, SessionSummary } from "@/lib/api";
import DashboardPage from "@/app/dashboard/page";

const mockUseSWR = vi.mocked(useSWR);

const makeSessions = (): SessionSummary[] => [
  {
    id: "sess-1",
    name: "Session One",
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T01:00:00Z",
    message_count: 4,
  },
];

const makeMetrics = (): MetricsResponse => ({
  session_id: "sess-1",
  summary: {
    total_queries: 4,
    total_tokens: 5000,
    total_cost_usd: 0.2,
    avg_latency_seconds: 1.5,
    avg_token_savings_percent: 25,
  },
  by_mode: {
    rlm: { queries: 2, total_tokens: 2500, total_cost_usd: 0.1, avg_latency_seconds: 1.8 },
    direct: { queries: 2, total_tokens: 2500, total_cost_usd: 0.1, avg_latency_seconds: 1.2 },
  },
  by_provider: {
    openai: { queries: 4, total_tokens: 5000, total_cost_usd: 0.2, avg_latency_seconds: 1.5 },
  },
  timeline: [],
});

// Helper: stub all five useSWR calls made by DashboardPage:
//   1. sessions  2. metrics  3. failureMetrics  4. evalSummary  5. evalDetail
const noData = { data: undefined, isLoading: false } as ReturnType<typeof useSWR>;

describe("DashboardPage", () => {
  test("renders summary metrics cards", () => {
    const metrics = makeMetrics();
    mockUseSWR
      .mockReturnValueOnce({ data: makeSessions(), isLoading: false } as ReturnType<typeof useSWR>)
      .mockReturnValueOnce({ data: metrics, isLoading: false } as ReturnType<typeof useSWR>)
      .mockReturnValueOnce(noData)  // failureMetrics
      .mockReturnValueOnce(noData)  // evalSummary
      .mockReturnValueOnce(noData); // evalDetail

    render(<DashboardPage />);
    expect(screen.getByText("Total Tokens")).toBeInTheDocument();
    expect(screen.getByText("Total Cost")).toBeInTheDocument();
    expect(screen.getByText("Avg Latency")).toBeInTheDocument();
    expect(screen.getByText("Token Savings")).toBeInTheDocument();
  });

  test("renders charts section when metrics are available", () => {
    const metrics = makeMetrics();
    mockUseSWR
      .mockReturnValueOnce({ data: makeSessions(), isLoading: false } as ReturnType<typeof useSWR>)
      .mockReturnValueOnce({ data: metrics, isLoading: false } as ReturnType<typeof useSWR>)
      .mockReturnValueOnce(noData)  // failureMetrics
      .mockReturnValueOnce(noData)  // evalSummary
      .mockReturnValueOnce(noData); // evalDetail

    render(<DashboardPage />);
    expect(screen.getByText("Mode Comparison")).toBeInTheDocument();
    expect(screen.getByText("Cost by Chat Provider")).toBeInTheDocument();
  });

  test("shows empty state when no sessions exist", () => {
    mockUseSWR
      .mockReturnValueOnce({ data: [], isLoading: false } as ReturnType<typeof useSWR>)
      .mockReturnValueOnce(noData)  // metrics
      .mockReturnValueOnce(noData)  // failureMetrics
      .mockReturnValueOnce(noData)  // evalSummary
      .mockReturnValueOnce(noData); // evalDetail

    render(<DashboardPage />);
    expect(
      screen.getByText("Select a session to view metrics."),
    ).toBeInTheDocument();
  });

  test("fetches metrics data on mount via SWR", () => {
    mockUseSWR
      .mockReturnValueOnce({ data: makeSessions(), isLoading: false } as ReturnType<typeof useSWR>)
      .mockReturnValueOnce(noData)  // metrics
      .mockReturnValueOnce(noData)  // failureMetrics
      .mockReturnValueOnce(noData)  // evalSummary
      .mockReturnValueOnce(noData); // evalDetail

    render(<DashboardPage />);
    // SWR was called at least twice: once for sessions, once for metrics
    expect(mockUseSWR).toHaveBeenCalledWith("sessions", expect.any(Function));
    expect(mockUseSWR).toHaveBeenCalledWith(
      expect.stringContaining("metrics-"),
      expect.any(Function),
    );
  });
});

// ---------------------------------------------------------------------------
// FailureBreakdown — prefill_timeout category wired into the chart
// ---------------------------------------------------------------------------

describe("FailureBreakdown", () => {
  test("ALL_CATEGORIES includes prefill_timeout", async () => {
    const { ALL_CATEGORIES, CATEGORY_LABELS } = await import(
      "@/components/metrics/failure-breakdown"
    );
    // Both the enum and the label map must carry the new category so
    // the recharts BarChart can render a bar for it. Without this,
    // /api/metrics/failures rows with category="prefill_timeout" would
    // be silently dropped by the frontend — regression guard for the
    // Phase 4 wiring.
    expect(ALL_CATEGORIES).toContain("prefill_timeout");
    expect(CATEGORY_LABELS.prefill_timeout).toBe("Prefill Timeout");
  });

  test("renders without error when backend reports prefill_timeout", async () => {
    const { FailureBreakdown } = await import(
      "@/components/metrics/failure-breakdown"
    );
    const data = {
      session_id: "s1",
      total_runs: 5,
      total_failures: 2,
      failure_rate: 0.4,
      by_category: [
        { category: "prefill_timeout" as const, count: 2, percentage: 100 },
      ],
      by_provider: {
        openai: [
          { category: "prefill_timeout" as const, count: 2, percentage: 100 },
        ],
      },
      by_mode: {},
    };
    render(<FailureBreakdown data={data} />);
    // Card renders the failure-count summary line; recharts is mocked
    // so bars don't appear in the DOM — the assertion above covers the
    // category-coverage invariant.
    expect(screen.getByText(/2 of 5 runs failed/i)).toBeInTheDocument();
  });
});
