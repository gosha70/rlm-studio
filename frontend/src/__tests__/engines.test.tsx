/**
 * Official-engine gating (specs/interop-official-rlm FR-7).
 *
 * - `unavailableReasonFor` is the single rule every picker applies.
 * - `EngineBadge` renders only for `rlm_official` runs and carries the
 *   installed engine version.
 * - The Compare mode picker disables the option and shows the server's
 *   reason as the tooltip when `GET /api/engines` says unavailable.
 */

import { render, screen } from "@testing-library/react";
import { vi, describe, test, expect, beforeEach } from "vitest";
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
import { StrategySelectItems } from "@/components/settings/strategy-select-items";
import { Select, SelectContent, SelectTrigger, SelectValue } from "@/components/ui/select";
import { MODE_DIRECT, MODE_RLM, MODE_RLM_OFFICIAL } from "@/lib/constants";
import type { EnginesResponse } from "@/lib/api";
import ComparePage from "@/app/compare/page";

const INSTALL_HINT = 'The official RLM engine needs the `interop` extra: pip install "rlm-studio[interop]"';

const AVAILABLE: EnginesResponse = {
  rlm_official: { available: true, reason: "rlms 0.1.3", version: "0.1.3" },
};
const UNAVAILABLE: EnginesResponse = {
  rlm_official: { available: false, reason: INSTALL_HINT, version: null },
};

function mockSWR(engines: EnginesResponse | undefined) {
  const stub = (data: unknown) => ({
    data,
    mutate: vi.fn(),
    error: undefined,
    isLoading: false,
    isValidating: false,
  });
  vi.mocked(useSWR).mockImplementation(((key: unknown) => {
    if (key === ENGINES_SWR_KEY) return stub(engines);
    if (key === "llm-providers" || key === "profiles") return stub([]);
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

    const button = screen.getByRole("button", { name: MODE_RLM_OFFICIAL });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute("title", INSTALL_HINT);
    // Built-in modes are untouched by the engine status.
    expect(screen.getByRole("button", { name: MODE_RLM })).toBeEnabled();
  });

  test("offers rlm_official normally when the engine is available", () => {
    mockSWR(AVAILABLE);
    render(<ComparePage />);

    const button = screen.getByRole("button", { name: MODE_RLM_OFFICIAL });
    expect(button).toBeEnabled();
    expect(button).toHaveAttribute("title", "Paper authors' reference implementation (rlms)");
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

    const option = screen.getByRole("option", { name: /Official RLM \(rlms\) — not installed/ });
    expect(option).toHaveAttribute("aria-disabled", "true");
    expect(option).toHaveAttribute("title", INSTALL_HINT);
    expect(screen.getByRole("option", { name: "RLM" })).not.toHaveAttribute("aria-disabled", "true");
  });

  test("offers the official engine normally when available", () => {
    mockSWR(AVAILABLE);
    renderOpen();

    const option = screen.getByRole("option", { name: "Official RLM (rlms)" });
    expect(option).not.toHaveAttribute("aria-disabled", "true");
  });
});
