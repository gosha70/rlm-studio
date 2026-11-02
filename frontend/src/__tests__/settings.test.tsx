/**
 * Settings component tests.
 *
 * Covers: BudgetConfig, ModelSelector, ConnectionTester, ProviderCard, SettingsPage.
 */

import React from "react";
import { act, render, screen, fireEvent, waitFor } from "@testing-library/react";
import { vi, describe, test, expect, beforeEach } from "vitest";
import { BudgetConfig } from "@/components/settings/budget-config";
import { ModelSelector } from "@/components/settings/model-selector";
import { ConnectionTester } from "@/components/settings/connection-tester";
import { ProviderCard } from "@/components/settings/provider-card";
import useSWR from "swr";
import {
  getLLMProviderModelsForConfig,
  testLLMProviderConfig,
  testProvider,
  type ProviderTestResponse,
} from "@/lib/api";
import SettingsPage from "@/app/settings/page";
import type { BudgetConfig as BudgetConfigType, ModelInfo, ProviderInfo } from "@/lib/api";

// ---------------------------------------------------------------------------
// Module-level mocks (vitest hoists vi.mock to the top automatically)
// ---------------------------------------------------------------------------

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    testProvider: vi.fn(),
    saveProvider: vi.fn().mockResolvedValue({ message: "Saved" }),
    // Keep the Settings form's debounced model discovery off the network.
    getProviderModels: vi.fn().mockResolvedValue([]),
    testLLMProviderConfig: vi.fn(),
    getLLMProviderModelsForConfig: vi.fn().mockResolvedValue({
      models: [],
      source: "unavailable",
      error: "No models returned by the endpoint.",
    }),
  };
});

vi.mock("swr", () => ({
  default: vi.fn(),
}));

vi.mock("next-themes", () => ({
  useTheme: () => ({ theme: "system", setTheme: vi.fn() }),
}));

vi.mock("@/components/shared/app-shell", () => ({
  AppShell: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

vi.mock("@/components/settings/system-prompt-editor", () => ({
  SystemPromptEditor: () => <div>SystemPromptEditor</div>,
}));

vi.mock("@/components/settings/profile-card", () => ({
  ProfileCard: () => <div>ProfileCard</div>,
}));

// SettingsPage now reads the Cookbook → Settings deep-link query
// params via next/navigation. The existing tests do not mount a
// real router, so stub both hooks with deterministic no-op values.
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    replace: vi.fn(),
    push: vi.fn(),
    prefetch: vi.fn(),
    refresh: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
  }),
  useSearchParams: () => new URLSearchParams(),
}));

// ---------------------------------------------------------------------------
// BudgetConfig
// ---------------------------------------------------------------------------

const DEFAULT_CONFIG: BudgetConfigType = {
  max_steps: 16,
  max_tokens: 50000,
  max_cost_usd: 2.0,
  max_time_seconds: 600,
  max_recursion_depth: 5,
  repeat_limit: 2,
  nudge_at_fraction: 0.6,
};

describe("BudgetConfig", () => {
  test("renders Max Steps label with current value", () => {
    render(<BudgetConfig config={DEFAULT_CONFIG} onChange={vi.fn()} />);
    expect(screen.getByText(/Max Steps: 16/)).toBeInTheDocument();
  });

  test("renders Max Tokens input with current value", () => {
    render(<BudgetConfig config={DEFAULT_CONFIG} onChange={vi.fn()} />);
    const input = screen.getByLabelText("Max Tokens") as HTMLInputElement;
    expect(input).toBeInTheDocument();
    expect(input.value).toBe("50000");
  });

  test("renders Max Cost input with current value", () => {
    render(<BudgetConfig config={DEFAULT_CONFIG} onChange={vi.fn()} />);
    const input = screen.getByLabelText("Max Cost (USD)") as HTMLInputElement;
    expect(input).toBeInTheDocument();
    expect(input.value).toBe("2");
  });

  test("renders Max Time label with current value", () => {
    render(<BudgetConfig config={DEFAULT_CONFIG} onChange={vi.fn()} />);
    expect(screen.getByText(/Max Time.*: 600/)).toBeInTheDocument();
  });

  test("renders Max Recursion Depth label with current value", () => {
    render(<BudgetConfig config={DEFAULT_CONFIG} onChange={vi.fn()} />);
    expect(screen.getByText(/Max Recursion Depth: 5/)).toBeInTheDocument();
  });

  test("renders Save and Reset to Defaults buttons", () => {
    render(<BudgetConfig config={DEFAULT_CONFIG} onChange={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Save" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reset to Defaults" })).toBeInTheDocument();
  });

  test("calls onChange with current values when Save clicked", () => {
    const onChange = vi.fn();
    render(<BudgetConfig config={DEFAULT_CONFIG} onChange={onChange} />);
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(onChange).toHaveBeenCalledWith(DEFAULT_CONFIG);
  });

  test("updates max_tokens input and calls onChange on Save", () => {
    const onChange = vi.fn();
    render(<BudgetConfig config={DEFAULT_CONFIG} onChange={onChange} />);

    const input = screen.getByLabelText("Max Tokens");
    fireEvent.change(input, { target: { value: "100000" } });

    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(onChange).toHaveBeenCalledWith({
      ...DEFAULT_CONFIG,
      max_tokens: 100000,
    });
  });

  test("updates max_cost_usd input and calls onChange on Save", () => {
    const onChange = vi.fn();
    render(<BudgetConfig config={DEFAULT_CONFIG} onChange={onChange} />);

    const input = screen.getByLabelText("Max Cost (USD)");
    fireEvent.change(input, { target: { value: "5" } });

    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(onChange).toHaveBeenCalledWith({
      ...DEFAULT_CONFIG,
      max_cost_usd: 5,
    });
  });

  test("Reset to Defaults calls onChange with default values", () => {
    const onChange = vi.fn();
    const customConfig: BudgetConfigType = {
      max_steps: 5,
      max_tokens: 10000,
      max_cost_usd: 0.5,
      max_time_seconds: 10,
      max_recursion_depth: 2,
      repeat_limit: 1,
      nudge_at_fraction: 0.3,
    };
    render(<BudgetConfig config={customConfig} onChange={onChange} />);
    fireEvent.click(screen.getByRole("button", { name: "Reset to Defaults" }));
    expect(onChange).toHaveBeenCalledWith({
      max_steps: 16,
      max_tokens: 50000,
      max_cost_usd: 2.0,
      max_time_seconds: 600,
      max_recursion_depth: 5,
      repeat_limit: 2,
      nudge_at_fraction: 0.6,
    });
  });

  test("Reset to Defaults updates displayed values", () => {
    const customConfig: BudgetConfigType = {
      max_steps: 5,
      max_tokens: 10000,
      max_cost_usd: 0.5,
      max_time_seconds: 10,
      max_recursion_depth: 2,
      repeat_limit: 1,
      nudge_at_fraction: 0.3,
    };
    render(<BudgetConfig config={customConfig} onChange={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "Reset to Defaults" }));

    const tokensInput = screen.getByLabelText("Max Tokens") as HTMLInputElement;
    expect(tokensInput.value).toBe("50000");
  });

  test("max-steps slider exists with aria-valuetext", () => {
    render(<BudgetConfig config={DEFAULT_CONFIG} onChange={vi.fn()} />);
    // Radix Slider renders a <span role="slider"> with aria-valuenow; the
    // aria-label is on the parent container span, not the thumb.
    const sliders = screen.getAllByRole("slider");
    const stepsSlider = sliders.find(
      (s) => s.getAttribute("aria-valuenow") === "16"
    );
    expect(stepsSlider).toBeInTheDocument();
  });

  test("max-time slider exists with aria-valuenow", () => {
    render(<BudgetConfig config={DEFAULT_CONFIG} onChange={vi.fn()} />);
    const sliders = screen.getAllByRole("slider");
    const timeSlider = sliders.find(
      (s) => s.getAttribute("aria-valuenow") === "600"
    );
    expect(timeSlider).toBeInTheDocument();
  });

  test("max-depth slider exists with aria-valuenow", () => {
    render(<BudgetConfig config={DEFAULT_CONFIG} onChange={vi.fn()} />);
    const sliders = screen.getAllByRole("slider");
    const depthSlider = sliders.find(
      (s) => s.getAttribute("aria-valuenow") === "5"
    );
    expect(depthSlider).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// ModelSelector
// ---------------------------------------------------------------------------

const makeModel = (name: string): ModelInfo => ({
  name,
  input_cost_per_1k: 0.003,
  output_cost_per_1k: 0.006,
});

describe("ModelSelector", () => {
  test("renders dropdown with available models", () => {
    render(
      <ModelSelector
        models={[makeModel("gpt-4o"), makeModel("gpt-4o-mini")]}
        value={null}
        onChange={vi.fn()}
      />
    );
    // The trigger renders with aria-label="Select model"
    expect(screen.getByRole("combobox", { name: "Select model" })).toBeInTheDocument();
  });

  test("shows current model as selected", () => {
    render(
      <ModelSelector
        models={[makeModel("gpt-4o"), makeModel("gpt-4o-mini")]}
        value="gpt-4o"
        onChange={vi.fn()}
      />
    );
    // The select trigger displays the current value
    const trigger = screen.getByRole("combobox", { name: "Select model" });
    expect(trigger).toHaveTextContent("gpt-4o");
  });

  test("calls onChange when a model is selected via free-text input", () => {
    const onChange = vi.fn();
    render(
      <ModelSelector
        models={[]}
        value="gpt-4o"
        onChange={onChange}
        freeText={true}
      />
    );
    const input = screen.getByRole("textbox", { name: "Model name" });
    fireEvent.change(input, { target: { value: "claude-3-opus" } });
    expect(onChange).toHaveBeenCalledWith("claude-3-opus");
  });

  test("shows 'No models available' state when models list is empty", () => {
    render(
      <ModelSelector
        models={[]}
        value={null}
        onChange={vi.fn()}
      />
    );
    // The trigger still renders; the "No models available" text is in the dropdown content
    expect(screen.getByRole("combobox", { name: "Select model" })).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// ConnectionTester
// ---------------------------------------------------------------------------

describe("ConnectionTester", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  test("renders test button", () => {
    render(<ConnectionTester provider="openai" />);
    expect(screen.getByRole("button", { name: /Test Connection/i })).toBeInTheDocument();
  });

  test("shows loading state during test", async () => {
    // Make testProvider never resolve so we can observe the loading state
    vi.mocked(testProvider).mockReturnValue(new Promise(() => {}));
    render(<ConnectionTester provider="openai" />);
    fireEvent.click(screen.getByRole("button", { name: /Test Connection/i }));
    // Button text changes to "Testing..." while in flight
    expect(screen.getByRole("button", { name: /Testing.../i })).toBeInTheDocument();
  });

  test("shows success state with latency on successful test", async () => {
    vi.mocked(testProvider).mockResolvedValue({ connected: true, latency_ms: 123 });
    render(<ConnectionTester provider="openai" />);
    fireEvent.click(screen.getByRole("button", { name: /Test Connection/i }));
    // StatusIndicator renders the label with latency
    const statusEl = await screen.findByText(/Connected.*123ms/i);
    expect(statusEl).toBeInTheDocument();
  });

  test("shows error message on failed test", async () => {
    vi.mocked(testProvider).mockResolvedValue({ connected: false, error: "Invalid API key" });
    render(<ConnectionTester provider="openai" />);
    fireEvent.click(screen.getByRole("button", { name: /Test Connection/i }));
    const statusEl = await screen.findByText(/Invalid API key/i);
    expect(statusEl).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// ProviderCard
// ---------------------------------------------------------------------------

const makeProviderInfo = (overrides: Partial<ProviderInfo> = {}): ProviderInfo => ({
  name: "openai",
  display_name: "OpenAI",
  status: "connected",
  models: [{ name: "gpt-4o", input_cost_per_1k: 0.005, output_cost_per_1k: 0.015 }],
  default_model: "gpt-4o",
  configured: true,
  requires_api_key: true,
  default_endpoint: null,
  model_input_hint: "",
  masked_api_key: "sk-...xxxx",
  ...overrides,
});

describe("ProviderCard", () => {
  test("renders provider name and status", () => {
    render(<ProviderCard provider={makeProviderInfo()} />);
    expect(screen.getByText("OpenAI")).toBeInTheDocument();
    // StatusIndicator renders a role="status" element
    const statusEl = document.querySelector('[role="status"]');
    expect(statusEl).toBeInTheDocument();
  });

  test("shows API key input field", () => {
    render(<ProviderCard provider={makeProviderInfo({ requires_api_key: true })} />);
    const input = screen.getByLabelText(`API key for OpenAI`);
    expect(input.tagName).toBe("INPUT");
  });

  test("masks API key value when text is entered", () => {
    render(<ProviderCard provider={makeProviderInfo({ requires_api_key: true })} />);
    const input = screen.getByLabelText(`API key for OpenAI`) as HTMLInputElement;
    fireEvent.change(input, { target: { value: "sk-test-key" } });
    expect(input.type).toBe("password");
  });

  test("shows model selector when models provided", () => {
    const models = [
      { name: "gpt-4o", input_cost_per_1k: 0.005, output_cost_per_1k: 0.015 },
      { name: "gpt-4o-mini", input_cost_per_1k: 0.0002, output_cost_per_1k: 0.0006 },
    ];
    render(<ProviderCard provider={makeProviderInfo({ models })} />);
    // ModelSelector renders a combobox (Select trigger)
    expect(screen.getByRole("combobox", { name: "Select model" })).toBeInTheDocument();
  });

  test("shows not_configured status when provider is not configured", () => {
    render(
      <ProviderCard
        provider={makeProviderInfo({ status: "not_configured", configured: false, masked_api_key: null })}
      />
    );
    // StatusIndicator for not_configured has aria-label including "Not configured"
    const statusEl = document.querySelector('[role="status"]');
    expect(statusEl).toHaveAttribute("aria-label", expect.stringContaining("Not configured"));
  });
});

// ---------------------------------------------------------------------------
// SettingsPage
// ---------------------------------------------------------------------------

const MOCK_CONFIG = {
  budget: {
    max_steps: 16,
    max_tokens: 50000,
    max_cost_usd: 2.0,
    max_time_seconds: 30,
    max_recursion_depth: 5,
    repeat_limit: 2,
    nudge_at_fraction: 0.6,
  },
  provider_configs: [],
  active_profile_id: null,
  judge_chat_provider_id: null,
  appearance: { theme: "system", sidebar_collapsed: false },
};

describe("SettingsPage", () => {
  beforeEach(() => {
    const swrStub = (data: unknown) => ({ data, mutate: vi.fn(), error: undefined, isLoading: false, isValidating: false });
    vi.mocked(useSWR).mockImplementation(((key: unknown) => {
      if (key === "config") return swrStub(MOCK_CONFIG);
      if (key === "providers") return swrStub([]);
      if (key === "profiles") return swrStub([]);
      if (key === "chat-providers") return swrStub([]);
      if (key === "llm-providers") return swrStub([]);
      return swrStub(undefined);
    }) as typeof useSWR);
  });

  test("renders provider cards section", () => {
    render(<SettingsPage />);
    expect(screen.getByRole("tab", { name: "LLM Providers" })).toBeInTheDocument();
  });

  test("renders budget tab trigger", () => {
    render(<SettingsPage />);
    const budgetTab = screen.getByRole("tab", { name: "Budget" });
    expect(budgetTab).toBeInTheDocument();
    // Budget tab panel exists in DOM (hidden until activated)
    const panel = document.getElementById(
      budgetTab.getAttribute("aria-controls")!
    );
    expect(panel).toBeInTheDocument();
  });

  test("renders appearance tab trigger", () => {
    render(<SettingsPage />);
    const appearanceTab = screen.getByRole("tab", { name: "Appearance" });
    expect(appearanceTab).toBeInTheDocument();
    const panel = document.getElementById(
      appearanceTab.getAttribute("aria-controls")!
    );
    expect(panel).toBeInTheDocument();
  });

  test("fetches config on mount", () => {
    render(<SettingsPage />);
    expect(vi.mocked(useSWR)).toHaveBeenCalledWith("config", expect.any(Function));
  });

  test.skip("saves config changes via PUT /api/config", () => {
    // Skipped: this test would require triggering the budget save flow end-to-end,
    // which involves clicking Save inside BudgetConfig and asserting the updateConfig
    // API call. The interaction crosses too many component boundaries and the async
    // state update path requires act() wrapping that is fragile in this setup.
    // Covered by e2e tests instead.
  });

  test("renders General tab trigger", () => {
    render(<SettingsPage />);
    const generalTab = screen.getByRole("tab", { name: "General" });
    expect(generalTab).toBeInTheDocument();
    const panel = document.getElementById(generalTab.getAttribute("aria-controls")!);
    expect(panel).toBeInTheDocument();
  });

  test("General tab shows connection-test-interval input", async () => {
    render(<SettingsPage />);
    const user = (await import("@testing-library/user-event")).default.setup();
    await user.click(screen.getByRole("tab", { name: "General" }));
    // Input is rendered with id 'connection-test-interval'.
    const input = await screen.findByLabelText(/Interval/i);
    expect(input).toBeInTheDocument();
    expect((input as HTMLInputElement).type).toBe("number");
  });

  test("LLM Providers tab shows scheduled-testing cross-link notice", () => {
    render(<SettingsPage />);
    // The notice text depends on the current config.connection_test_interval_minutes.
    // Mock's default is 0, so the "disabled — enable on General" message should be visible.
    expect(
      screen.getByText(/Auto-testing of connections is disabled/i),
    ).toBeInTheDocument();
  });

  test("a connected provider shows the absolute test time with seconds", () => {
    const swrStub = (data: unknown) => ({ data, mutate: vi.fn(), error: undefined, isLoading: false, isValidating: false });
    vi.mocked(useSWR).mockImplementation(((key: unknown) => {
      if (key === "config") return swrStub(MOCK_CONFIG);
      if (key === "llm-providers")
        return swrStub([
          {
            id: "lp-spark",
            name: "DGX Spark",
            backend: "vllm",
            model: "qwen38-27b",
            endpoint: "http://spark-c2e5.local:8001/v1",
            runtime_settings: {
              temperature: 0.7,
              top_p: 1.0,
              max_output_tokens: 4096,
              timeout_seconds: 30,
            },
            status: "connected",
            last_tested_at: "2026-11-01T13:05:58Z",
            last_tested_by: "manual",
            consecutive_failures: 0,
          },
        ]);
      return swrStub(undefined);
    }) as typeof useSWR);

    render(<SettingsPage />);
    // Absolute local stamp, seconds included — never "6m ago".
    expect(
      screen.getByText(/Last tested \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}/),
    ).toBeInTheDocument();
    expect(screen.getByText(/\(you\)/)).toBeInTheDocument();
    expect(screen.queryByText(/Last tested .* ago/)).not.toBeInTheDocument();
  });

  test("the model input opts out of browser autofill", async () => {
    render(<SettingsPage />);
    const user = (await import("@testing-library/user-event")).default.setup();
    await user.click(screen.getByRole("button", { name: /add llm provider/i }));
    const input = await screen.findByLabelText("Model");
    expect(input).toHaveAttribute("autocomplete", "off");
  });

  test("model discovery reports an unavailable source instead of fallback data", async () => {
    vi.mocked(getLLMProviderModelsForConfig).mockResolvedValueOnce({
      models: [],
      source: "unavailable",
      error: "Connection refused",
    });
    render(<SettingsPage />);
    const user = (await import("@testing-library/user-event")).default.setup();
    await user.click(screen.getByRole("button", { name: /add llm provider/i }));

    expect(
      await screen.findByText(
        /Could not load models from .*Connection refused/,
        undefined,
        { timeout: 3000 },
      ),
    ).toBeInTheDocument();
  });

  test("editing a cloud gateway sends its saved endpoint to discovery", async () => {
    const swrStub = (data: unknown) => ({ data, mutate: vi.fn(), error: undefined, isLoading: false, isValidating: false });
    vi.mocked(useSWR).mockImplementation(((key: unknown) => {
      if (key === "config") return swrStub(MOCK_CONFIG);
      if (key === "llm-providers")
        return swrStub([
          {
            id: "lp-gw",
            name: "Gateway",
            backend: "openai",
            model: "gpt-4o",
            endpoint: "https://gw.example/v1",
            runtime_settings: {
              temperature: 0.7,
              top_p: 1.0,
              max_output_tokens: 4096,
              timeout_seconds: 30,
            },
            status: "configured",
            last_tested_at: null,
            last_tested_by: null,
            consecutive_failures: 0,
          },
        ]);
      return swrStub(undefined);
    }) as typeof useSWR);
    vi.mocked(getLLMProviderModelsForConfig).mockClear();

    render(<SettingsPage />);
    const user = (await import("@testing-library/user-event")).default.setup();
    await user.click(screen.getByRole("button", { name: "Edit" }));

    // The saved gateway endpoint must reach discovery; sending null would
    // make the backend treat the destination as changed and refuse the key.
    await waitFor(
      () =>
        expect(vi.mocked(getLLMProviderModelsForConfig)).toHaveBeenCalledWith(
          expect.objectContaining({
            backend: "openai",
            endpoint: "https://gw.example/v1",
            llm_provider_id: "lp-gw",
          }),
        ),
      { timeout: 3000 },
    );
  });

  test("an in-flight form test cannot stamp a result after the form changes", async () => {
    let resolveTest: (value: ProviderTestResponse) => void = () => {};
    vi.mocked(testLLMProviderConfig).mockReturnValueOnce(
      new Promise<ProviderTestResponse>((resolve) => {
        resolveTest = resolve;
      }),
    );
    render(<SettingsPage />);
    const user = (await import("@testing-library/user-event")).default.setup();
    await user.click(screen.getByRole("button", { name: /add llm provider/i }));
    await user.type(screen.getByLabelText("Name"), "probe");
    await user.type(screen.getByLabelText("Model"), "qwen38-27b");
    await user.click(screen.getByRole("button", { name: /test connection/i }));

    // Change a probe-affecting field while the request is outstanding.
    await user.type(screen.getByLabelText("Model"), "-awq");

    await act(async () => {
      resolveTest({
        connected: true,
        latency_ms: 5,
        tested_at: "2026-11-01T13:05:58Z",
      });
      await Promise.resolve();
    });

    expect(screen.queryByText(/Connected/)).not.toBeInTheDocument();
  });
});
