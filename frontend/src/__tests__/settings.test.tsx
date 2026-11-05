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
import { ExecutionConfig } from "@/components/settings/execution-config";
import useSWR from "swr";
import {
  createChatProvider,
  createProfile,
  deleteLLMProvider,
  updateChatProvider,
  updateLLMProvider,
  getLLMProviderModelsForConfig,
  testLLMProviderConfig,
  testProvider,
  updateConfig,
  type ProviderTestResponse,
} from "@/lib/api";
import { toast } from "sonner";
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
    updateConfig: vi.fn(),
    createChatProvider: vi.fn(),
    updateChatProvider: vi.fn(),
    updateLLMProvider: vi.fn(),
    createProfile: vi.fn(),
    deleteLLMProvider: vi.fn(),
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

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn(), warning: vi.fn() },
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
      // The server default (RLM_DEFAULT_NUDGE_AT_FRACTION), not 0.6.
      nudge_at_fraction: 0.4,
    });
  });

  test.each([
    ["Max Tokens", "-5", /Max Tokens must be between 1 and 500,000\. Not saved\./],
    ["Max Tokens", "99999999999999", /Max Tokens must be between 1 and 500,000\. Not saved\./],
    ["Max Tokens", "12.5", /Max Tokens must be a whole number\. Not saved\./],
    ["Max Tokens", "", /Enter a value for Max Tokens\. Not saved\./],
    ["Max Cost (USD)", "-1", /Max Cost must be between 0 and 100\. Not saved\./],
    ["Max Cost (USD)", "99999999999999", /Max Cost must be between 0 and 100\. Not saved\./],
    ["Max Cost (USD)", "", /Enter a value for Max Cost\. Not saved\./],
  ])("refuses to save %s = %j and says why", (label, value, message) => {
    const onChange = vi.fn();
    render(<BudgetConfig config={DEFAULT_CONFIG} onChange={onChange} />);
    fireEvent.change(screen.getByLabelText(label), { target: { value } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(screen.getByRole("alert")).toHaveTextContent(message);
    expect(onChange).not.toHaveBeenCalled();
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
// ExecutionConfig
// ---------------------------------------------------------------------------

describe("ExecutionConfig", () => {
  test("offers no Default Mode control: nothing reads default_mode", () => {
    render(
      <ExecutionConfig
        config={{
          enabled_modes: ["direct", "rlm"],
          default_mode: "auto",
          rag_config: { chunk_size: 1000, chunk_overlap: 200, top_k: 5, embedding_model: "text-embedding-3-small" },
          rlm_max_steps: 16,
          rlm_timeout_seconds: 600,
        }}
        onChange={vi.fn()}
      />,
    );
    expect(screen.queryByText("Default Mode")).not.toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: "Default mode" })).not.toBeInTheDocument();
    expect(screen.queryByText(/system decides/)).not.toBeInTheDocument();
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
    // Any string is valid here; do not use a model id that dates the test.
    fireEvent.change(input, { target: { value: "custom-model-id" } });
    expect(onChange).toHaveBeenCalledWith("custom-model-id");
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
      screen.getByText(/Recurring auto-testing is disabled/i),
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

  test("a provider row shows the persisted failure cause and fix", () => {
    const swrStub = (data: unknown) => ({ data, mutate: vi.fn(), error: undefined, isLoading: false, isValidating: false });
    vi.mocked(useSWR).mockImplementation(((key: unknown) => {
      if (key === "config") return swrStub(MOCK_CONFIG);
      if (key === "llm-providers")
        return swrStub([
          {
            id: "lp-bad",
            name: "Stale key",
            backend: "anthropic",
            model: "claude-sonnet-4-6",
            endpoint: null,
            runtime_settings: {
              temperature: 0.7,
              top_p: 1.0,
              max_output_tokens: 4096,
              timeout_seconds: 30,
            },
            status: "offline",
            last_tested_at: "2026-11-01T13:05:58Z",
            last_tested_by: "manual",
            last_failure_kind: "auth",
            last_failure_cause: "The API key was rejected.",
            last_failure_fix:
              "Check the key has not been revoked or regenerated, then update it in Settings → Providers.",
            consecutive_failures: 1,
          },
        ]);
      return swrStub(undefined);
    }) as typeof useSWR);

    render(<SettingsPage />);

    expect(
      screen.getByText(/Last failure: The API key was rejected\./),
    ).toBeInTheDocument();
    expect(screen.getByText(/update it in Settings/)).toBeInTheDocument();
  });

  test.each([
    [0, false],
    [5, true],
  ])(
    "a connected provider with a failure says monitoring only when auto-testing is on (interval %i)",
    (minutes, monitoring) => {
      // SET2-01: with auto-testing off nothing re-checks the provider, so
      // "monitoring" was untrue.
      const swrStub = (data: unknown) => ({ data, mutate: vi.fn(), error: undefined, isLoading: false, isValidating: false });
      vi.mocked(useSWR).mockImplementation(((key: unknown) => {
        if (key === "config") return swrStub({ ...MOCK_CONFIG, connection_test_interval_minutes: minutes });
        if (key === "llm-providers")
          return swrStub([
            {
              id: "lp-flaky",
              name: "Flaky",
              backend: "vllm",
              model: "m",
              endpoint: "http://spark:8001/v1",
              runtime_settings: { temperature: 0.7, top_p: 1.0, max_output_tokens: 4096, timeout_seconds: 30 },
              status: "connected",
              last_tested_at: "2026-11-01T13:05:58Z",
              last_tested_by: "background",
              consecutive_failures: 1,
            },
          ]);
        return swrStub(undefined);
      }) as typeof useSWR);

      render(<SettingsPage />);

      expect(screen.getByText(/1 recent failure/)).toBeInTheDocument();
      expect(screen.queryByText(/monitoring/) !== null).toBe(monitoring);
    },
  );

  test("a provider row flags a model the provider does not offer", () => {
    const swrStub = (data: unknown) => ({ data, mutate: vi.fn(), error: undefined, isLoading: false, isValidating: false });
    vi.mocked(useSWR).mockImplementation(((key: unknown) => {
      if (key === "config") return swrStub(MOCK_CONFIG);
      if (key === "llm-providers")
        return swrStub([
          {
            id: "lp-model",
            name: "Stale model",
            backend: "vllm",
            model: "retired-model",
            endpoint: "http://spark:8001/v1",
            runtime_settings: {
              temperature: 0.7,
              top_p: 1.0,
              max_output_tokens: 4096,
              timeout_seconds: 30,
            },
            status: "configured",
            model_check_status: "unavailable",
            model_check_alternatives: ["qwen38-27b", "qwen38-27b-awq"],
            last_tested_at: null,
            last_tested_by: null,
            consecutive_failures: 0,
          },
        ]);
      return swrStub(undefined);
    }) as typeof useSWR);

    render(<SettingsPage />);

    expect(screen.getByText(/is not offered by this provider/)).toBeInTheDocument();
    expect(
      screen.getByText(/Available: qwen38-27b, qwen38-27b-awq\./),
    ).toBeInTheDocument();
  });

  test("a provider row shows an unknown model check quietly", () => {
    const swrStub = (data: unknown) => ({ data, mutate: vi.fn(), error: undefined, isLoading: false, isValidating: false });
    vi.mocked(useSWR).mockImplementation(((key: unknown) => {
      if (key === "config") return swrStub(MOCK_CONFIG);
      if (key === "llm-providers")
        return swrStub([
          {
            id: "lp-unknown",
            name: "Unreachable",
            backend: "vllm",
            model: "qwen38-27b",
            endpoint: "http://spark:8001/v1",
            runtime_settings: {
              temperature: 0.7,
              top_p: 1.0,
              max_output_tokens: 4096,
              timeout_seconds: 30,
            },
            status: "configured",
            model_check_status: "unknown",
            last_tested_at: null,
            last_tested_by: null,
            consecutive_failures: 0,
          },
        ]);
      return swrStub(undefined);
    }) as typeof useSWR);

    render(<SettingsPage />);

    expect(
      screen.getByText(/Model not checked: the provider's model list could not be read\./),
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

// ---------------------------------------------------------------------------
// SettingsPage — Wave 3 regressions
// ---------------------------------------------------------------------------

const RUNTIME = { temperature: 0.7, top_p: 1.0, max_output_tokens: 4096, timeout_seconds: 30 };

const makeLLMProvider = (overrides: Record<string, unknown>) => ({
  id: "lp-1",
  name: "Local",
  backend: "ollama",
  model: "llama3.2",
  endpoint: null,
  runtime_settings: RUNTIME,
  status: "configured",
  last_tested_at: null,
  last_tested_by: null,
  consecutive_failures: 0,
  ...overrides,
});

const RAG_PROFILE = {
  id: "prof-rag",
  name: "RAG profile",
  description: "",
  strategy: "rag",
  runtime_settings: RUNTIME,
  budget: MOCK_CONFIG.budget,
  system_prompts: {},
  prompt_template_name: null,
  is_builtin: false,
};

/** Stub SWR keys for one render of the Settings page. */
function stubSettingsData(data: Record<string, unknown>) {
  const swrStub = (value: unknown) => ({ data: value, mutate: vi.fn(), error: undefined, isLoading: false, isValidating: false });
  vi.mocked(useSWR).mockImplementation(((key: unknown) => {
    if (key === "config") return swrStub(MOCK_CONFIG);
    if (typeof key === "string" && key in data) return swrStub(data[key]);
    return swrStub(undefined);
  }) as typeof useSWR);
}

describe("SettingsPage — Wave 3", () => {
  beforeEach(() => {
    // Radix Select needs pointer-capture and scrolling APIs jsdom lacks.
    Element.prototype.hasPointerCapture = () => false;
    Element.prototype.releasePointerCapture = () => {};
    Element.prototype.scrollIntoView = () => {};
  });

  const setupUser = async () => (await import("@testing-library/user-event")).default.setup();

  const pick = async (
    user: Awaited<ReturnType<typeof setupUser>>,
    comboboxId: string,
    optionName: RegExp,
  ) => {
    await user.click(document.getElementById(comboboxId)!);
    await user.click(await screen.findByRole("option", { name: optionName }));
  };

  test.each([
    ["ollama", "ollama/nomic-embed-text"],
    ["lmstudio", "lm_studio/text-embedding-nomic-embed-text-v1.5"],
    ["openai", "text-embedding-3-small"],
  ])("a RAG Chat Provider on %s defaults to %s", async (backend, expected) => {
    stubSettingsData({
      "llm-providers": [makeLLMProvider({ backend, name: `My ${backend}` })],
      profiles: [RAG_PROFILE],
      "chat-providers": [],
    });
    render(<SettingsPage />);
    const user = await setupUser();
    await user.click(screen.getByRole("tab", { name: "Chat Providers" }));
    await user.click(screen.getByRole("button", { name: /create chat provider/i }));
    await pick(user, "cp-profile", /RAG profile/);
    await pick(user, "cp-llm-provider", new RegExp(`My ${backend}`));

    const embedding = screen.getByLabelText("Embedding Model") as HTMLInputElement;
    expect(embedding.value).toBe(expected);
    // Still editable.
    await user.clear(embedding);
    await user.type(embedding, "ollama/mxbai-embed-large");
    expect(embedding.value).toBe("ollama/mxbai-embed-large");
  });

  /** Open Create Chat Provider on a RAG profile and set one RAG field. */
  const fillRagChatProvider = async (field: string, value: string) => {
    stubSettingsData({
      "llm-providers": [makeLLMProvider({ name: "My ollama" })],
      profiles: [RAG_PROFILE],
      "chat-providers": [],
    });
    render(<SettingsPage />);
    const user = await setupUser();
    await user.click(screen.getByRole("tab", { name: "Chat Providers" }));
    await user.click(screen.getByRole("button", { name: /create chat provider/i }));
    await user.type(screen.getByLabelText("Name"), "Docs");
    await pick(user, "cp-profile", /RAG profile/);
    await pick(user, "cp-llm-provider", /My ollama/);
    const input = screen.getByLabelText(field);
    await user.clear(input);
    if (value) await user.type(input, value);
    return user;
  };

  test.each([
    ["Chunk Size", "-5", "Chunk Size must be between 1 and 100,000."],
    ["Chunk Size", "99999999999", "Chunk Size must be between 1 and 100,000."],
    ["Chunk Size", "0", "Chunk Size must be between 1 and 100,000."],
    ["Chunk Size", "12.5", "Chunk Size must be a whole number."],
    ["Chunk Size", "", "Enter a value for Chunk Size."],
    ["Chunk Overlap", "-1", "Chunk Overlap must be between 0 and 511."],
    ["Chunk Overlap", "512", "Chunk Overlap must be between 0 and 511."],
    ["Top K", "-1", "Top K must be between 1 and 100."],
    ["Top K", "0", "Top K must be between 1 and 100."],
    ["Top K", "100000", "Top K must be between 1 and 100."],
    ["Embedding Model", "", "Enter an Embedding Model."],
  ])("a RAG %s of %j is refused, not saved", async (field, value, message) => {
    vi.mocked(createChatProvider).mockClear();
    const user = await fillRagChatProvider(field, value);

    expect(screen.getByRole("status")).toHaveTextContent(message);
    const create = screen.getByRole("button", { name: "Create" });
    expect(create).toBeDisabled();
    await user.click(create);
    expect(vi.mocked(createChatProvider)).not.toHaveBeenCalled();
  });

  test("the Chat Provider edit form refuses an invalid RAG value too", async () => {
    vi.mocked(updateChatProvider).mockClear();
    stubSettingsData({
      "llm-providers": [makeLLMProvider({ name: "My ollama" })],
      profiles: [RAG_PROFILE],
      "chat-providers": [
        {
          id: "cp-rag",
          name: "Docs",
          llm_provider_id: "lp-1",
          llm_provider_name: "My ollama",
          profile_id: RAG_PROFILE.id,
          execution_mode: "rag",
          runtime_settings: RUNTIME,
          rlm_max_steps: 16,
          rlm_timeout_seconds: 600,
          rlm_repeat_limit: 2,
          rag_config: {
            chunk_size: 512,
            chunk_overlap: 64,
            top_k: 5,
            embedding_model: "ollama/nomic-embed-text",
          },
        },
      ],
    });
    render(<SettingsPage />);
    const user = await setupUser();
    await user.click(screen.getByRole("tab", { name: "Chat Providers" }));
    await user.click(screen.getByRole("button", { name: "Edit" }));
    const topK = screen.getByLabelText("Top K");
    await user.clear(topK);
    await user.type(topK, "0");

    expect(screen.getByRole("status")).toHaveTextContent("Top K must be between 1 and 100.");
    const save = screen.getByRole("button", { name: "Save" });
    expect(save).toBeDisabled();
    await user.click(save);
    expect(vi.mocked(updateChatProvider)).not.toHaveBeenCalled();
  });

  test("valid RAG settings are saved as typed", async () => {
    vi.mocked(createChatProvider).mockClear();
    vi.mocked(createChatProvider).mockResolvedValueOnce({} as never);
    const user = await fillRagChatProvider("Top K", "12");
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Create" }));

    await waitFor(() =>
      expect(vi.mocked(createChatProvider)).toHaveBeenCalledWith(
        expect.objectContaining({
          rag_config: {
            chunk_size: 512,
            chunk_overlap: 64,
            top_k: 12,
            embedding_model: "ollama/nomic-embed-text",
          },
        }),
      ),
    );
  });

  test.each(["1e3", "-5", "0", "1", "1023", "12abc"])(
    "an LLM Provider context window of %j is refused with a reason",
    async (value) => {
      stubSettingsData({ "llm-providers": [] });
      render(<SettingsPage />);
      const user = await setupUser();
      await user.click(screen.getByRole("button", { name: /add llm provider/i }));
      await user.type(screen.getByLabelText("Name"), "Local");
      await user.type(screen.getByLabelText("Model"), "llama3.2");
      await user.type(screen.getByLabelText("Context Window (tokens)"), value);

      expect(
        screen.getByText(/Enter a whole number of tokens from 1,024 to 10,000,000/),
      ).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Create" })).toBeDisabled();
    },
  );

  test.each(["1024", "8192"])("a context window of %s is accepted", async (value) => {
    stubSettingsData({ "llm-providers": [] });
    render(<SettingsPage />);
    const user = await setupUser();
    await user.click(screen.getByRole("button", { name: /add llm provider/i }));
    await user.type(screen.getByLabelText("Name"), "Local");
    await user.type(screen.getByLabelText("Model"), "llama3.2");
    await user.type(screen.getByLabelText("Context Window (tokens)"), value);

    expect(screen.queryByText(/Enter a whole number of tokens/)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create" })).toBeEnabled();
  });

  test("a negative connection-test interval is refused, not clamped and saved", async () => {
    vi.mocked(updateConfig).mockClear();
    stubSettingsData({});
    render(<SettingsPage />);
    const user = await setupUser();
    await user.click(screen.getByRole("tab", { name: "General" }));
    const input = await screen.findByLabelText(/Interval/i);
    await user.clear(input);
    await user.type(input, "-3");
    await user.tab();

    expect(screen.getByRole("alert")).toHaveTextContent(/from 0 to 1440\. Not saved\./);
    expect(vi.mocked(updateConfig)).not.toHaveBeenCalled();
  });

  test("a failed Budget save says so with the server's reason", async () => {
    vi.mocked(updateConfig).mockRejectedValueOnce(
      new Error('API error 422: {"detail":"max_tokens must be at least 1"}'),
    );
    vi.mocked(toast.error).mockClear();
    stubSettingsData({});
    render(<SettingsPage />);
    const user = await setupUser();
    await user.click(screen.getByRole("tab", { name: "Budget" }));
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() =>
      expect(vi.mocked(toast.error)).toHaveBeenCalledWith(
        "Budget limits not saved: max_tokens must be at least 1",
      ),
    );
  });

  test("a long unbroken name wraps inside its card and keeps the buttons on it", async () => {
    const longName = "vLLM" + "dfs".repeat(60);
    stubSettingsData({
      "llm-providers": [makeLLMProvider({ name: longName })],
      profiles: [RAG_PROFILE],
      "chat-providers": [
        {
          id: "cp-1",
          name: longName,
          llm_provider_id: "lp-1",
          llm_provider_name: longName,
          profile_id: RAG_PROFILE.id,
          execution_mode: "rag",
          runtime_settings: RUNTIME,
          rlm_max_steps: 16,
          rlm_timeout_seconds: 600,
          rlm_repeat_limit: 2,
        },
      ],
    });
    render(<SettingsPage />);
    const user = await setupUser();

    const lpTitle = screen.getByText(longName);
    expect(lpTitle).toHaveClass("wrap-anywhere");
    expect(lpTitle.closest(".min-w-0.flex-1")).not.toBeNull();
    expect(screen.getByRole("button", { name: "Delete" }).parentElement).toHaveClass("shrink-0");

    await user.click(screen.getByRole("tab", { name: "Chat Providers" }));
    const cpTitles = screen.getAllByText(longName);
    expect(cpTitles.some((el) => el.classList.contains("wrap-anywhere"))).toBe(true);
    for (const button of screen.getAllByRole("button", { name: "Delete" })) {
      expect(button.parentElement).toHaveClass("shrink-0");
    }

    // The LLM Provider dropdown stays within the viewport.
    await user.click(screen.getByRole("button", { name: /create chat provider/i }));
    await user.click(document.getElementById("cp-llm-provider")!);
    const option = await screen.findByRole("option", { name: new RegExp(longName) });
    expect(option).toHaveClass("wrap-anywhere");
    expect(screen.getByRole("listbox").closest("[class*='max-w-[calc(100vw-2rem)]']")).not.toBeNull();
  });

  test("deleting an LLM Provider in use shows the server's reason", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    vi.mocked(deleteLLMProvider).mockRejectedValueOnce(
      new Error(
        'API error 409: {"error":{"code":"ERROR","message":"Cannot delete: referenced by Chat Provider(s): Qwen RLM","details":{}}}',
      ),
    );
    vi.mocked(toast.error).mockClear();
    stubSettingsData({ "llm-providers": [makeLLMProvider({ name: "Local Qwen" })] });
    render(<SettingsPage />);
    const user = await setupUser();
    await user.click(screen.getByRole("button", { name: "Delete" }));

    // The dialog no longer claims the Chat Providers "will stop working":
    // the server refuses the delete instead.
    expect(confirmSpy).toHaveBeenCalledWith(expect.stringContaining('"Local Qwen"'));
    expect(confirmSpy).not.toHaveBeenCalledWith(expect.stringContaining("stop working"));
    await waitFor(() =>
      expect(vi.mocked(toast.error)).toHaveBeenCalledWith(
        "LLM Provider not deleted: Cannot delete: referenced by Chat Provider(s): Qwen RLM",
      ),
    );
    confirmSpy.mockRestore();
  });

  test("the edit form says a key is already stored", async () => {
    stubSettingsData({
      "llm-providers": [
        makeLLMProvider({ id: "lp-k", name: "Keyed", backend: "openai", model: "gpt-4o", api_key_stored: true }),
      ],
    });
    render(<SettingsPage />);
    const user = await setupUser();
    await user.click(screen.getByRole("button", { name: "Edit" }));

    const keyInput = screen.getByLabelText("API Key");
    expect(keyInput).toHaveAttribute("placeholder", "Key stored (hidden)");
    expect(screen.getByText(/A key is stored for this provider/)).toBeInTheDocument();
  });

  test("the edit form makes no stored-key claim without a key", async () => {
    stubSettingsData({
      "llm-providers": [
        makeLLMProvider({ id: "lp-n", name: "Keyless", backend: "openai", model: "gpt-4o", api_key_stored: false }),
      ],
    });
    render(<SettingsPage />);
    const user = await setupUser();
    await user.click(screen.getByRole("button", { name: "Edit" }));

    expect(screen.getByLabelText("API Key")).toHaveAttribute("placeholder", "sk-...");
    expect(screen.queryByText(/A key is stored for this provider/)).not.toBeInTheDocument();
  });

  test("a disabled Create says which fields are missing", async () => {
    stubSettingsData({ "llm-providers": [], profiles: [RAG_PROFILE], "chat-providers": [] });
    render(<SettingsPage />);
    const user = await setupUser();
    await user.click(screen.getByRole("button", { name: /add llm provider/i }));
    expect(screen.getByText("Fill in Name, Model to create.")).toBeInTheDocument();
    await user.type(screen.getByLabelText("Name"), "Local");
    expect(screen.getByText("Fill in Model to create.")).toBeInTheDocument();
    await user.type(screen.getByLabelText("Model"), "llama3.2");
    expect(screen.queryByText(/^Fill in/)).not.toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "Chat Providers" }));
    await user.click(screen.getByRole("button", { name: /create chat provider/i }));
    expect(screen.getByText("Fill in Name, LLM Provider, Profile to create.")).toBeInTheDocument();
  });

  test("picking a discovered model does not flip the Select to controlled", async () => {
    vi.mocked(getLLMProviderModelsForConfig).mockResolvedValueOnce({
      models: [{ name: "qwen2.5-7b-instruct", input_cost_per_1k: 0, output_cost_per_1k: 0 }],
      source: "live",
      error: null,
    });
    // Radix reports the flip through console.warn.
    const errors = vi.spyOn(console, "warn").mockImplementation(() => {});
    stubSettingsData({ "llm-providers": [] });
    render(<SettingsPage />);
    const user = await setupUser();
    await user.click(screen.getByRole("button", { name: /add llm provider/i }));
    await screen.findByText(/1 model loaded/, undefined, { timeout: 3000 });
    await pick(user, "llmp-model", /qwen2\.5-7b-instruct/);

    const flips = errors.mock.calls.filter((args) =>
      String(args[0]).includes("from uncontrolled to controlled"),
    );
    errors.mockRestore();
    expect(flips).toEqual([]);
  });

  test("the tab strip wraps inside its container instead of overflowing at 768 px", () => {
    stubSettingsData({});
    render(<SettingsPage />);
    // jsdom has no layout; the classes are what keep all seven tabs on
    // screen (verified in Chromium at 768 px: two rows, no page scroll).
    const tablist = screen.getByRole("tablist");
    expect(tablist).toHaveClass("flex-wrap", "max-w-full", "h-auto");
    expect(tablist).not.toHaveClass("h-10");
  });

  test("a successful Budget save is confirmed", async () => {
    vi.mocked(updateConfig).mockResolvedValueOnce(MOCK_CONFIG as never);
    vi.mocked(toast.success).mockClear();
    stubSettingsData({});
    render(<SettingsPage />);
    const user = await setupUser();
    await user.click(screen.getByRole("tab", { name: "Budget" }));
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() =>
      expect(vi.mocked(toast.success)).toHaveBeenCalledWith("Budget limits saved"),
    );
  });
});

// ---------------------------------------------------------------------------
// SettingsPage — Wave 3c (browser pass 3)
// ---------------------------------------------------------------------------

describe("SettingsPage — Wave 3c", () => {
  beforeEach(() => {
    Element.prototype.hasPointerCapture = () => false;
    Element.prototype.releasePointerCapture = () => {};
    Element.prototype.scrollIntoView = () => {};
    vi.mocked(updateLLMProvider).mockReset();
    vi.mocked(updateLLMProvider).mockResolvedValue({} as never);
    vi.mocked(toast.error).mockClear();
  });

  const setupUser = async () => (await import("@testing-library/user-event")).default.setup();

  /** Open Edit on a saved provider with a stored context window and blank the field. */
  const blankContextWindowOnEdit = async () => {
    stubSettingsData({
      "llm-providers": [makeLLMProvider({ id: "lp-cw", context_window: 8192 })],
    });
    render(<SettingsPage />);
    const user = await setupUser();
    await user.click(screen.getByRole("button", { name: "Edit" }));
    const field = screen.getByLabelText("Context Window (tokens)") as HTMLInputElement;
    expect(field.value).toBe("8192");
    await user.clear(field);
    return user;
  };

  test("SC3-01: blanking the context window on Edit clears the stored value", async () => {
    const user = await blankContextWindowOnEdit();
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() =>
      expect(vi.mocked(updateLLMProvider)).toHaveBeenCalledWith(
        "lp-cw",
        expect.objectContaining({ context_window: 0 }),
      ),
    );
  });

  test("SC3-01: a blank context window with a new endpoint leaves discovery to the server", async () => {
    const user = await blankContextWindowOnEdit();
    await user.type(screen.getByLabelText("Endpoint"), "http://localhost:11500");
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() =>
      expect(vi.mocked(updateLLMProvider)).toHaveBeenCalledWith(
        "lp-cw",
        expect.objectContaining({ context_window: null, endpoint: "http://localhost:11500" }),
      ),
    );
  });

  test("SC3-05: a refused Chat Provider save shows the server's reason", async () => {
    vi.mocked(createChatProvider).mockRejectedValueOnce(
      new Error(
        'API error 409: {"error":{"code":"ERROR","message":"Chat Provider with name \'Docs\' already exists","details":{}}}',
      ),
    );
    stubSettingsData({
      "llm-providers": [makeLLMProvider({ name: "My ollama" })],
      profiles: [RAG_PROFILE],
      "chat-providers": [],
    });
    render(<SettingsPage />);
    const user = await setupUser();
    await user.click(screen.getByRole("tab", { name: "Chat Providers" }));
    await user.click(screen.getByRole("button", { name: /create chat provider/i }));
    await user.type(screen.getByLabelText("Name"), "Docs");
    await user.click(document.getElementById("cp-profile")!);
    await user.click(await screen.findByRole("option", { name: /RAG profile/ }));
    await user.click(document.getElementById("cp-llm-provider")!);
    await user.click(await screen.findByRole("option", { name: /My ollama/ }));
    await user.click(screen.getByRole("button", { name: "Create" }));

    await waitFor(() =>
      expect(vi.mocked(toast.error)).toHaveBeenCalledWith(
        "Chat Provider not saved: Chat Provider with name 'Docs' already exists",
      ),
    );
  });

  test("SC3-05: a refused profile create shows the server's reason", async () => {
    vi.mocked(createProfile).mockRejectedValueOnce(
      new Error('API error 422: {"detail":"strategy must be one of direct, rlm, rag"}'),
    );
    stubSettingsData({ profiles: [] });
    render(<SettingsPage />);
    const user = await setupUser();
    await user.click(screen.getByRole("tab", { name: "Profiles" }));
    await user.type(screen.getByLabelText("New profile name"), "Fresh");
    await user.click(screen.getByRole("button", { name: "Create" }));

    await waitFor(() =>
      expect(vi.mocked(toast.error)).toHaveBeenCalledWith(
        "Profile not created: strategy must be one of direct, rlm, rag",
      ),
    );
  });

  test("SC3-07: the General tab and the LLM Providers banner agree on the startup check", async () => {
    stubSettingsData({});
    render(<SettingsPage />);
    expect(screen.getByText(/A one-time check runs at startup/)).toBeInTheDocument();
    const user = await setupUser();
    await user.click(screen.getByRole("tab", { name: "General" }));
    expect(screen.queryByText(/not at server startup/)).not.toBeInTheDocument();
    expect(screen.getByText(/checked once shortly after the server starts/)).toBeInTheDocument();
  });
});
