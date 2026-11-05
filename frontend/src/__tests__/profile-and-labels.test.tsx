/**
 * Tests for new features introduced in the last 12 commits.
 *
 * Covers:
 *   - FieldLabel: renders label text, help button is focusable with accessible name
 *   - ProfileCard: displays prompt group, editing form shows Prompt Group dropdown,
 *     a single custom RLM prompt textarea, and save/clone keep system_prompts
 */

import React from "react";
import { render, screen, fireEvent, within, act } from "@testing-library/react";
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

// The panel has its own test file; here a stub exposes the callback the card
// passes, so the card's reaction to an accept or rollback can be checked.
const panelProps: { onProfileChanged?: () => void } = {};
vi.mock("@/components/settings/refinement-panel", () => ({
  RefinementPanel: (props: { onProfileChanged: () => void }) => {
    panelProps.onProfileChanged = props.onProfileChanged;
    return <div data-testid="refinement-panel" />;
  },
}));

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
    const helpBtn = screen.getByRole("button", { name: "Help: Temperature" });
    expect(helpBtn).toBeInTheDocument();
    // Should be a real button element, not an SVG
    expect(helpBtn.tagName).toBe("BUTTON");
  });

  test("help button has type=button (does not submit forms)", () => {
    render(<FieldLabel tooltip="Help text">Temperature</FieldLabel>);
    const helpBtn = screen.getByRole("button", { name: "Help: Temperature" });
    expect(helpBtn).toHaveAttribute("type", "button");
  });

  test("HelpCircle SVG is aria-hidden", () => {
    render(<FieldLabel tooltip="Help text">Temperature</FieldLabel>);
    const helpBtn = screen.getByRole("button", { name: "Help: Temperature" });
    const svg = helpBtn.querySelector("svg");
    expect(svg).toHaveAttribute("aria-hidden", "true");
  });

  test("names the help button after its field, without a slider value", () => {
    render(
      <>
        <FieldLabel tooltip="a">Max Tokens</FieldLabel>
        <FieldLabel tooltip="b">Max Steps: {16}</FieldLabel>
      </>,
    );
    expect(screen.getByRole("button", { name: "Help: Max Tokens" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Help: Max Steps" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Help" })).not.toBeInTheDocument();
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

  test("a long unbroken name wraps and leaves the action buttons on the card", () => {
    const longName = "p" + "x".repeat(200);
    render(<ProfileCard profile={makeProfile({ name: longName })} />);
    expect(screen.getByText(longName)).toHaveClass("wrap-anywhere");
    expect(
      screen.getByRole("button", { name: `Delete ${longName} profile` }).parentElement,
    ).toHaveClass("shrink-0");
  });

  test("a refused delete is explained on the collapsed card", () => {
    const profile = makeProfile();
    render(
      <ProfileCard
        profile={profile}
        chatProviders={[{ id: "cp-1", name: "Qwen RLM", profile_id: profile.id } as never]}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: `Delete ${profile.name} profile` }));
    expect(screen.getByRole("status")).toHaveTextContent("Cannot delete: used by Qwen RLM");
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

  test("shows only the RLM custom prompt textarea (no Direct/RAG editors)", () => {
    openEditor();
    // Only RLM consumes a profile prompt, and prompt refinement (#78) stores
    // accepted candidates there, so it is the one editable per-mode prompt.
    expect(screen.queryByLabelText(/direct mode/i)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/rag mode/i)).not.toBeInTheDocument();
    expect(screen.getByLabelText("Custom RLM prompt")).toBeInTheDocument();
  });

  test("shows FieldLabel tooltips for numeric fields", () => {
    openEditor();
    // FieldLabel renders a "Help: <field>" button for each numeric field
    const helpButtons = screen.getAllByRole("button", { name: /^Help: / });
    // Temperature, Top P, Max Tokens, Timeout, Max Steps, Max Cost,
    // Repeat Limit, Nudge at Fraction = 8 FieldLabels
    expect(helpButtons.length).toBeGreaterThanOrEqual(8);
  });

  test("shows Save and Cancel buttons when editing", () => {
    openEditor();
    expect(screen.getByRole("button", { name: /save/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /cancel/i })).toBeInTheDocument();
  });

  test("rejects an absurd Max Tokens instead of saving it", async () => {
    const { updateProfile } = await import("@/lib/api");
    vi.mocked(updateProfile).mockClear();
    openEditor();
    fireEvent.change(screen.getByLabelText("Max Tokens"), {
      target: { value: "99999999999999" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^save$/i }));
    expect(screen.getByRole("status")).toHaveTextContent(/Max Tokens must be between 1 and 1,000,000/);
    expect(vi.mocked(updateProfile)).not.toHaveBeenCalled();
  });

  test.each([
    ["Timeout (s)", "99999999999999", "Timeout (s) must be between 5 and 3,600."],
    ["Max Steps", "99999999999999", "Max Steps must be between 1 and 500."],
    ["Max Execution Time (s)", "99999999999999", "Max Execution Time (s) must be between 10 and 86,400."],
    ["Max Cost ($)", "99999999999999", "Max Cost ($) must be between 0 and 100."],
    ["Repeat Limit", "99999999999999", "Repeat Limit must be between 1 and 20."],
    ["Max Steps", "2.5", "Max Steps must be a whole number."],
    ["Max Cost ($)", "", "Enter a value for Max Cost ($)."],
  ])("refuses %s of %j with the actual problem", async (label, value, message) => {
    const { updateProfile } = await import("@/lib/api");
    vi.mocked(updateProfile).mockClear();
    openEditor();
    fireEvent.change(screen.getByLabelText(label), { target: { value } });
    fireEvent.click(screen.getByRole("button", { name: /^save$/i }));
    expect(screen.getByRole("status")).toHaveTextContent(message);
    expect(vi.mocked(updateProfile)).not.toHaveBeenCalled();
  });

  test("a stored out-of-range budget can be seen, fixed and saved (review P1)", async () => {
    // Older profiles may hold values the API now refuses; the form used to
    // resend them from hidden fields, so every save failed with no way out.
    const { updateProfile } = await import("@/lib/api");
    vi.mocked(updateProfile).mockClear();
    vi.mocked(updateProfile).mockResolvedValue(makeProfile());
    const base = makeProfile();
    render(
      <ProfileCard
        profile={makeProfile({ budget: { ...base.budget, max_tokens: 99_999_999_999_999, max_recursion_depth: -1 } })}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /edit/i }));

    fireEvent.click(screen.getByRole("button", { name: /^save$/i }));
    expect(screen.getByRole("status")).toHaveTextContent("Max Tokens (budget) must be between 1 and 500,000.");
    fireEvent.change(screen.getByLabelText("Max Tokens (budget)"), { target: { value: "50000" } });
    fireEvent.click(screen.getByRole("button", { name: /^save$/i }));
    expect(screen.getByRole("status")).toHaveTextContent("Max Recursion Depth must be between 0 and 100.");
    fireEvent.change(screen.getByLabelText("Max Recursion Depth"), { target: { value: "3" } });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^save$/i }));
    });

    expect(vi.mocked(updateProfile)).toHaveBeenCalledTimes(1);
    const sent = vi.mocked(updateProfile).mock.calls[0][1] as { budget: { max_tokens: number; max_recursion_depth: number } };
    expect(sent.budget.max_tokens).toBe(50000);
    expect(sent.budget.max_recursion_depth).toBe(3);
  });

  test("SC3-03: editing a field clears the last Save message", async () => {
    const { updateProfile } = await import("@/lib/api");
    vi.mocked(updateProfile).mockClear();
    openEditor();
    fireEvent.change(screen.getByLabelText("Max Steps"), { target: { value: "9999" } });
    fireEvent.click(screen.getByRole("button", { name: /^save$/i }));
    expect(screen.getByRole("status")).toHaveTextContent("Max Steps must be between 1 and 500.");

    fireEvent.change(screen.getByLabelText("Max Steps"), { target: { value: "50" } });
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(vi.mocked(updateProfile)).not.toHaveBeenCalled();
  });

  test("saves values inside the bounds", async () => {
    const { updateProfile } = await import("@/lib/api");
    vi.mocked(updateProfile).mockClear();
    openEditor();
    fireEvent.change(screen.getByLabelText("Max Steps"), { target: { value: "500" } });
    fireEvent.change(screen.getByLabelText("Max Cost ($)"), { target: { value: "100" } });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^save$/i }));
    });
    expect(vi.mocked(updateProfile)).toHaveBeenCalledWith(
      "test-id-1",
      expect.objectContaining({
        budget: expect.objectContaining({ max_steps: 500, max_cost_usd: 100 }),
      }),
    );
  });

  test("Cancel closes the edit form", () => {
    openEditor();
    fireEvent.click(screen.getByRole("button", { name: /cancel/i }));
    // After cancel, the Save button (edit-form-only) should not be visible;
    // the read-only settings grid may still show labels like "Prompt Group".
    expect(screen.queryByRole("button", { name: /^Save$/i })).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// ProfileCard — system_prompts persistence (#78 FR-14)
// ---------------------------------------------------------------------------

describe("ProfileCard — system_prompts persistence", () => {
  beforeEach(async () => {
    mockUseSWR.mockReturnValue({
      data: MOCK_TEMPLATES,
      error: undefined,
      isLoading: false,
      isValidating: false,
      mutate: vi.fn(),
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);
    const api = await import("@/lib/api");
    vi.mocked(api.updateProfile).mockClear();
    vi.mocked(api.createProfile).mockClear();
  });

  const STORED = { [MODE_RLM]: "Read the index first.", direct: "legacy direct text" };

  async function saveWith(rlmText: string | null) {
    render(<ProfileCard profile={makeProfile({ system_prompts: { ...STORED } })} />);
    fireEvent.click(screen.getByRole("button", { name: /edit/i }));
    const textarea = screen.getByLabelText("Custom RLM prompt") as HTMLTextAreaElement;
    expect(textarea.value).toBe(STORED[MODE_RLM]);
    if (rlmText !== null) {
      fireEvent.change(textarea, { target: { value: rlmText } });
    }
    fireEvent.click(screen.getByRole("button", { name: /^Save$/i }));
    const api = await import("@/lib/api");
    await vi.waitFor(() => expect(api.updateProfile).toHaveBeenCalledTimes(1));
    return vi.mocked(api.updateProfile).mock.calls[0][1];
  }

  test("save without touching the prompt keeps every stored key", async () => {
    const payload = await saveWith(null);
    expect(payload.system_prompts).toEqual(STORED);
  });

  test("save sends the edited RLM prompt and keeps other keys", async () => {
    const payload = await saveWith("New supplement");
    expect(payload.system_prompts).toEqual({ ...STORED, [MODE_RLM]: "New supplement" });
  });

  test("clearing the RLM prompt removes only the RLM key", async () => {
    const payload = await saveWith("   ");
    expect(payload.system_prompts).toEqual({ direct: STORED.direct });
  });

  test("clone copies system_prompts", async () => {
    render(<ProfileCard profile={makeProfile({ system_prompts: { ...STORED } })} />);
    fireEvent.click(screen.getByRole("button", { name: /clone/i }));
    const api = await import("@/lib/api");
    await vi.waitFor(() => expect(api.createProfile).toHaveBeenCalledTimes(1));
    expect(vi.mocked(api.createProfile).mock.calls[0][0].system_prompts).toEqual(STORED);
  });
});


// ---------------------------------------------------------------------------
// ProfileCard — prompt refinement entry point (#78 FR-18)
// ---------------------------------------------------------------------------

describe("ProfileCard — refinement", () => {
  beforeEach(() => {
    panelProps.onProfileChanged = undefined;
    mockUseSWR.mockReturnValue({
      data: MOCK_TEMPLATES,
      error: undefined,
      isLoading: false,
      isValidating: false,
      mutate: vi.fn(),
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);
  });

  test("user profiles get a Refine button that opens the panel", () => {
    render(<ProfileCard profile={makeProfile()} />);
    expect(screen.queryByTestId("refinement-panel")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Refine Test Profile RLM prompt" }));
    expect(screen.getByTestId("refinement-panel")).toBeInTheDocument();
  });

  test("built-in profiles have no Refine button", () => {
    render(<ProfileCard profile={makeProfile({ is_builtin: true })} />);
    expect(screen.queryByRole("button", { name: /Refine/ })).not.toBeInTheDocument();
  });

  test("an accept or rollback closes the open edit form and refreshes the profile", () => {
    const onUpdated = vi.fn();
    render(<ProfileCard profile={makeProfile()} onUpdated={onUpdated} />);
    fireEvent.click(screen.getByRole("button", { name: /edit/i }));
    expect(screen.getByRole("button", { name: /^Save$/i })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Refine Test Profile RLM prompt" }));

    act(() => panelProps.onProfileChanged?.());

    // A stale form would save the RLM prompt captured before the decision.
    expect(screen.queryByRole("button", { name: /^Save$/i })).not.toBeInTheDocument();
    expect(onUpdated).toHaveBeenCalledTimes(1);
  });
});
