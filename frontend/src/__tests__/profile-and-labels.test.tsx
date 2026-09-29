/**
 * Tests for new features introduced in the last 12 commits.
 *
 * Covers:
 *   - FieldLabel: renders label text, help button is focusable with accessible name
 *   - ProfileCard: displays prompt group, editing form shows Prompt Group dropdown,
 *     no per-mode textareas for custom prompts
 */

import React from "react";
import { render, screen, fireEvent, within } from "@testing-library/react";
import { vi, describe, test, expect, beforeEach } from "vitest";
import { FieldLabel } from "@/components/settings/field-label";
import { ProfileCard } from "@/components/settings/profile-card";
import { displayModeName, MODE_RLM, MODE_RLM_OFFICIAL } from "@/lib/constants";
import type { RunProfile, SystemPromptTemplate } from "@/lib/api";

// ---------------------------------------------------------------------------
// Module-level mocks
// ---------------------------------------------------------------------------

vi.mock("swr", () => ({
  default: vi.fn(),
}));

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    getPromptTemplates: vi.fn(),
    createProfile: vi.fn().mockResolvedValue({}),
    deleteProfile: vi.fn().mockResolvedValue(undefined),
    updateProfile: vi.fn().mockResolvedValue({}),
  };
});

import useSWR from "swr";

const mockUseSWR = vi.mocked(useSWR);

// ---------------------------------------------------------------------------
// Test data factories
// ---------------------------------------------------------------------------

const MOCK_TEMPLATES: SystemPromptTemplate[] = [
  {
    name: "Default",
    description: "Balanced, general-purpose",
    prompts: { direct: "d", rlm: "r", rag: "g" },
  },
  {
    name: "Concise analyst",
    description: "Short and sharp",
    prompts: { direct: "cd", rlm: "cr", rag: "cg" },
  },
];

function makeProfile(overrides?: Partial<RunProfile>): RunProfile {
  return {
    id: "test-id-1",
    name: "Test Profile",
    description: "A test profile",
    strategy: "rlm",
    default_provider: null,
    providers_enabled: [],
    runtime_settings: {
      temperature: 0.7,
      top_p: 1.0,
      max_output_tokens: 4096,
      timeout_seconds: 120,
    },
    budget: {
      max_steps: 32,
      max_tokens: 100000,
      max_cost_usd: 2.0,
      max_time_seconds: 300,
      max_recursion_depth: 5,
      repeat_limit: 2,
      nudge_at_fraction: 0.6,
    },
    system_prompts: {},
    prompt_template_name: null,
    is_builtin: false,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// FieldLabel
// ---------------------------------------------------------------------------

describe("FieldLabel", () => {
  test("renders label text", () => {
    render(<FieldLabel tooltip="Help text">Temperature</FieldLabel>);
    expect(screen.getByText("Temperature")).toBeInTheDocument();
  });

  test("renders a focusable help button", () => {
    render(<FieldLabel tooltip="Help text">Temperature</FieldLabel>);
    const helpBtn = screen.getByRole("button", { name: "Help" });
    expect(helpBtn).toBeInTheDocument();
    // Should be a real button element, not an SVG
    expect(helpBtn.tagName).toBe("BUTTON");
  });

  test("help button has type=button (does not submit forms)", () => {
    render(<FieldLabel tooltip="Help text">Temperature</FieldLabel>);
    const helpBtn = screen.getByRole("button", { name: "Help" });
    expect(helpBtn).toHaveAttribute("type", "button");
  });

  test("HelpCircle SVG is aria-hidden", () => {
    render(<FieldLabel tooltip="Help text">Temperature</FieldLabel>);
    const helpBtn = screen.getByRole("button", { name: "Help" });
    const svg = helpBtn.querySelector("svg");
    expect(svg).toHaveAttribute("aria-hidden", "true");
  });

  test("associates label with htmlFor", () => {
    render(<FieldLabel htmlFor="temp-input" tooltip="Help text">Temperature</FieldLabel>);
    const label = screen.getByText("Temperature");
    expect(label).toHaveAttribute("for", "temp-input");
  });

  test("applies custom className", () => {
    const { container } = render(
      <FieldLabel className="text-xs" tooltip="tip">Temp</FieldLabel>
    );
    expect(container.firstChild).toHaveClass("text-xs");
  });
});

// ---------------------------------------------------------------------------
// ProfileCard — read-only view
// ---------------------------------------------------------------------------

describe("ProfileCard — read-only view", () => {
  beforeEach(() => {
    mockUseSWR.mockReturnValue({
      data: MOCK_TEMPLATES,
      error: undefined,
      isLoading: false,
      isValidating: false,
      mutate: vi.fn(),
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);
  });

  test("displays profile name and strategy badge", () => {
    render(<ProfileCard profile={makeProfile()} />);
    expect(screen.getByText("Test Profile")).toBeInTheDocument();
    // The badge shows the mode's display name, never the raw identifier —
    // `rlm_official` would otherwise read as "Rlm_official".
    expect(screen.getByText(displayModeName(MODE_RLM))).toBeInTheDocument();
    expect(screen.queryByText(MODE_RLM)).not.toBeInTheDocument();
  });

  test("displays the official engine's name for an rlm_official profile", () => {
    render(<ProfileCard profile={makeProfile({ strategy: MODE_RLM_OFFICIAL })} />);
    expect(screen.getByText(displayModeName(MODE_RLM_OFFICIAL))).toBeInTheDocument();
    expect(screen.queryByText(MODE_RLM_OFFICIAL)).not.toBeInTheDocument();
  });

  test("shows prompt template name when set", () => {
    render(
      <ProfileCard profile={makeProfile({ prompt_template_name: "Default" })} />
    );
    expect(screen.getByText(/Default/)).toBeInTheDocument();
  });

  test("shows 'Global defaults' when no template is set", () => {
    render(
      <ProfileCard profile={makeProfile({ prompt_template_name: null })} />
    );
    expect(screen.getByText(/Global defaults/)).toBeInTheDocument();
  });

  test("shows Built-in badge for builtin profiles", () => {
    render(
      <ProfileCard profile={makeProfile({ is_builtin: true })} />
    );
    expect(screen.getByText("Built-in")).toBeInTheDocument();
  });

  test("does not show edit button for builtin profiles", () => {
    render(
      <ProfileCard profile={makeProfile({ is_builtin: true })} />
    );
    expect(
      screen.queryByRole("button", { name: /edit/i })
    ).not.toBeInTheDocument();
  });

  test("shows edit button for user profiles", () => {
    render(<ProfileCard profile={makeProfile()} />);
    expect(
      screen.getByRole("button", { name: /edit/i })
    ).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// ProfileCard — editing form
// ---------------------------------------------------------------------------

describe("ProfileCard — editing form", () => {
  beforeEach(() => {
    mockUseSWR.mockReturnValue({
      data: MOCK_TEMPLATES,
      error: undefined,
      isLoading: false,
      isValidating: false,
      mutate: vi.fn(),
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);
  });

  function openEditor() {
    render(<ProfileCard profile={makeProfile()} />);
    const editBtn = screen.getByRole("button", { name: /edit/i });
    fireEvent.click(editBtn);
  }

  test("shows Prompt Group label when editing", () => {
    openEditor();
    expect(screen.getByText("Prompt Group")).toBeInTheDocument();
  });

  test("does NOT show per-mode textareas for custom prompts", () => {
    openEditor();
    // Ensure no textareas with per-mode prompt labels exist
    expect(screen.queryByLabelText(/direct mode/i)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/rlm mode/i)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/rag mode/i)).not.toBeInTheDocument();
  });

  test("shows FieldLabel tooltips for numeric fields", () => {
    openEditor();
    // FieldLabel renders a "Help" button for each numeric field
    const helpButtons = screen.getAllByRole("button", { name: "Help" });
    // Temperature, Top P, Max Tokens, Timeout, Max Steps, Max Cost,
    // Repeat Limit, Nudge at Fraction = 8 FieldLabels
    expect(helpButtons.length).toBeGreaterThanOrEqual(8);
  });

  test("shows Save and Cancel buttons when editing", () => {
    openEditor();
    expect(screen.getByRole("button", { name: /save/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /cancel/i })).toBeInTheDocument();
  });

  test("Cancel closes the edit form", () => {
    openEditor();
    fireEvent.click(screen.getByRole("button", { name: /cancel/i }));
    // After cancel, the Save button (edit-form-only) should not be visible;
    // the read-only settings grid may still show labels like "Prompt Group".
    expect(screen.queryByRole("button", { name: /^Save$/i })).not.toBeInTheDocument();
  });
});
