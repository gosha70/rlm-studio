/**
 * Learn replay page tests — V2b step 6.
 *
 * Covers the three states NEXT.md §3c pins for the
 * `/learn/replay?id=<executionId>` route:
 *
 *   1. loading  — SWR has no data yet, no error
 *   2. error    — SWR errored (404 / network), alert + back link render
 *   3. ready    — payload renders through the shared widget; the
 *      `metadata.truncated` banner surfaces when applicable
 *
 * Kept in its own file (separate from learn-replay.test.tsx) because
 * it mocks `swr` module-wide, which would collide with the widget
 * tests that need real SWR / controls hooks.
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";

import type { LearnReplay } from "@/lib/api";

// ---------------------------------------------------------------------------
// Module mocks. We're testing the page's fetch-and-branch wiring, not
// the widget internals — those have their own suite.
// ---------------------------------------------------------------------------

let mockSearch: Record<string, string> = { id: "exec-123" };
vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(mockSearch),
}));

vi.mock("swr", () => ({ default: vi.fn() }));

vi.mock("@/components/shared/app-shell", () => ({
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  AppShell: ({ children }: { children: any }) => children,
}));

vi.mock("@/components/learn/replay-walkthrough", () => ({
  ReplayWalkthrough: ({ replay }: { replay: LearnReplay }) => (
    <div data-testid="replay-walkthrough">rendered:{replay.id}</div>
  ),
}));

import useSWR from "swr";
import ReplayPage from "@/app/learn/replay/page";

const mockUseSWR = vi.mocked(useSWR);

const makeReplay = (overrides: Partial<LearnReplay> = {}): LearnReplay => ({
  id: "trace-exec-123",
  title: "Replay",
  description: "Reconstructed from a trace.",
  steps: [
    {
      id: "question",
      kind: "question",
      title: "Question received",
      summary: "What is 2 + 2?",
    },
    {
      id: "answer",
      kind: "answer",
      title: "Final answer",
      summary: "4",
    },
  ],
  metadata: {
    source: "trace",
    executionId: "exec-123",
    originalStepCount: 2,
    convertorVersion: 1,
  },
  ...overrides,
});

describe("ReplayPage", () => {
  test("loading state shows loading text, not the widget", () => {
    mockUseSWR.mockReturnValue({
      data: undefined,
      error: undefined,
    } as ReturnType<typeof useSWR>);

    render(<ReplayPage />);
    expect(screen.getByText(/Loading replay/i)).toBeInTheDocument();
    expect(screen.queryByTestId("replay-walkthrough")).not.toBeInTheDocument();
  });

  test("error state renders alert with back link to Traces", () => {
    mockUseSWR.mockReturnValue({
      data: undefined,
      error: new Error("404 Not Found"),
    } as ReturnType<typeof useSWR>);

    render(<ReplayPage />);
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent(/Couldn.t load this replay/);
    const link = screen.getByRole("link", { name: /Back to Traces/i });
    expect(link).toHaveAttribute("href", "/traces");
    expect(screen.queryByTestId("replay-walkthrough")).not.toBeInTheDocument();
  });

  test("ready state mounts ReplayWalkthrough with the fetched payload", () => {
    const replay = makeReplay();
    mockUseSWR.mockReturnValue({
      data: replay,
      error: undefined,
    } as ReturnType<typeof useSWR>);

    render(<ReplayPage />);
    expect(screen.getByTestId("replay-walkthrough")).toHaveTextContent(
      "rendered:trace-exec-123",
    );
    // No truncation banner when metadata.truncated is absent.
    expect(screen.queryByRole("note")).not.toBeInTheDocument();
  });

  test("truncated metadata surfaces a visible notice above the widget", () => {
    const replay = makeReplay({
      metadata: {
        source: "trace",
        executionId: "exec-123",
        originalStepCount: 128,
        truncated: true,
        convertorVersion: 1,
      },
    });
    mockUseSWR.mockReturnValue({
      data: replay,
      error: undefined,
    } as ReturnType<typeof useSWR>);

    render(<ReplayPage />);
    const note = screen.getByRole("note");
    expect(note).toHaveTextContent(/truncated/i);
    expect(note).toHaveTextContent(/128/);
    // Copy must reflect what metadata.originalStepCount actually is
    // (the pre-truncation *replay* length, including synthetic bookends
    // and with final/error folded into answer — NOT the raw trace
    // step count). The prior "The original trace had N steps" copy
    // misattributed the value.
    expect(note).toHaveTextContent(/full replay would have had/i);
    expect(note.textContent ?? "").not.toMatch(/original trace had/i);
    expect(screen.getByTestId("replay-walkthrough")).toBeInTheDocument();
  });

  test("back-to-learn link is always present on the page", () => {
    mockUseSWR.mockReturnValue({
      data: undefined,
      error: undefined,
    } as ReturnType<typeof useSWR>);

    render(<ReplayPage />);
    const back = screen.getByRole("link", { name: /Back to Learn/i });
    expect(back).toHaveAttribute("href", "/learn");
  });

  test("without an id it says so and links to Traces instead of loading forever (M18)", () => {
    mockSearch = {};
    try {
      mockUseSWR.mockReturnValue({ data: undefined, error: undefined } as ReturnType<typeof useSWR>);
      render(<ReplayPage />);
      expect(screen.queryByText(/Loading replay/i)).not.toBeInTheDocument();
      expect(screen.getByRole("alert")).toHaveTextContent(/No execution chosen to replay/);
      expect(screen.getByRole("link", { name: "Traces" })).toHaveAttribute("href", "/traces");
    } finally {
      mockSearch = { id: "exec-123" };
    }
  });
});
