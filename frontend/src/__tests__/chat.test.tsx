/**
 * Chat component tests.
 *
 * Covers: ChatInput, TypingIndicator, FileAttachment, ChatProviderSelector,
 * ResponseRating, PickWinner, JudgeScores.
 */

import { render, screen, fireEvent } from "@testing-library/react";
import { vi, describe, test, expect } from "vitest";
import { ChatInput } from "@/components/chat/chat-input";
import { TypingIndicator } from "@/components/chat/typing-indicator";
import { FileAttachment } from "@/components/chat/file-attachment";
import { ChatProviderSelector } from "@/components/chat/chat-provider-selector";
import { ResponseRating } from "@/components/chat/response-rating";
import { PickWinner } from "@/components/chat/pick-winner";
import { JudgeScores } from "@/components/chat/judge-scores";
import type { ChatProviderConfig, LLMProviderConfig } from "@/lib/api";

// ---------------------------------------------------------------------------
// ChatInput
// ---------------------------------------------------------------------------

describe("ChatInput", () => {
  test("renders textarea and send button", () => {
    render(<ChatInput onSend={vi.fn()} />);
    expect(screen.getByLabelText("Message input")).toBeInTheDocument();
    expect(screen.getByLabelText("Send message")).toBeInTheDocument();
  });

  test("disables send button when input is empty", () => {
    render(<ChatInput onSend={vi.fn()} />);
    expect(screen.getByLabelText("Send message")).toBeDisabled();
  });

  test("enables send button when input has text", () => {
    render(<ChatInput onSend={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("Message input"), {
      target: { value: "hello" },
    });
    expect(screen.getByLabelText("Send message")).not.toBeDisabled();
  });

  test("calls onSend with trimmed text on button click", () => {
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} />);
    fireEvent.change(screen.getByLabelText("Message input"), {
      target: { value: "  hello world  " },
    });
    fireEvent.click(screen.getByLabelText("Send message"));
    expect(onSend).toHaveBeenCalledWith("hello world");
  });

  test("clears input after submit", () => {
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} />);
    const textarea = screen.getByLabelText("Message input");
    fireEvent.change(textarea, { target: { value: "hello" } });
    fireEvent.click(screen.getByLabelText("Send message"));
    expect(textarea).toHaveValue("");
  });

  test("Enter key submits and clears input", () => {
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} />);
    const textarea = screen.getByLabelText("Message input");
    fireEvent.change(textarea, { target: { value: "hello" } });
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: false });
    expect(onSend).toHaveBeenCalledWith("hello");
    expect(textarea).toHaveValue("");
  });

  test("Shift+Enter does not submit", () => {
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} />);
    const textarea = screen.getByLabelText("Message input");
    fireEvent.change(textarea, { target: { value: "hello" } });
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: true });
    expect(onSend).not.toHaveBeenCalled();
  });

  test("does not call onSend for whitespace-only input", () => {
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} />);
    fireEvent.change(screen.getByLabelText("Message input"), {
      target: { value: "   " },
    });
    fireEvent.click(screen.getByLabelText("Send message"));
    expect(onSend).not.toHaveBeenCalled();
  });

  test("disables textarea and send button when disabled prop is true", () => {
    render(<ChatInput onSend={vi.fn()} disabled />);
    expect(screen.getByLabelText("Message input")).toBeDisabled();
    expect(screen.getByLabelText("Send message")).toBeDisabled();
  });

  test("shows upload button when onFileUpload provided", () => {
    render(<ChatInput onSend={vi.fn()} onFileUpload={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Upload file" })).toBeInTheDocument();
  });

  test("does not show upload button when onFileUpload not provided", () => {
    render(<ChatInput onSend={vi.fn()} />);
    expect(screen.queryByRole("button", { name: "Upload file" })).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// TypingIndicator
// ---------------------------------------------------------------------------

describe("TypingIndicator", () => {
  test("shows 'Generating...' with no props", () => {
    render(<TypingIndicator />);
    expect(screen.getByText("Generating...")).toBeInTheDocument();
  });

  test("shows mode in text when mode provided", () => {
    render(<TypingIndicator mode="rlm" />);
    expect(screen.getByText("Generating (rlm)...")).toBeInTheDocument();
  });

  test("shows step text when step provided", () => {
    render(<TypingIndicator step="Running code..." />);
    expect(screen.getByText("Running code...")).toBeInTheDocument();
  });

  test("step takes precedence over mode", () => {
    render(<TypingIndicator mode="rlm" step="Running code..." />);
    expect(screen.getByText("Running code...")).toBeInTheDocument();
    expect(screen.queryByText("Generating (rlm)...")).not.toBeInTheDocument();
  });

  test("has role='status' and aria-live='polite'", () => {
    render(<TypingIndicator />);
    const el = screen.getByRole("status");
    expect(el).toBeInTheDocument();
    expect(el).toHaveAttribute("aria-live", "polite");
  });
});

// ---------------------------------------------------------------------------
// FileAttachment
// ---------------------------------------------------------------------------

describe("FileAttachment", () => {
  test("renders file name", () => {
    render(
      <FileAttachment
        name="report.pdf"
        sizeBytes={2048}
        tokenCount={100}
        onRemove={vi.fn()}
      />
    );
    expect(screen.getByText("report.pdf")).toBeInTheDocument();
  });

  test("renders formatted file size in KB", () => {
    render(
      <FileAttachment
        name="report.pdf"
        sizeBytes={2048}
        tokenCount={0}
        onRemove={vi.fn()}
      />
    );
    expect(screen.getByText(/2\.0 KB/)).toBeInTheDocument();
  });

  test("renders file size in bytes for small files", () => {
    render(
      <FileAttachment
        name="tiny.txt"
        sizeBytes={500}
        tokenCount={0}
        onRemove={vi.fn()}
      />
    );
    expect(screen.getByText(/500 B/)).toBeInTheDocument();
  });

  test("renders file size in MB for large files", () => {
    render(
      <FileAttachment
        name="big.pdf"
        sizeBytes={2 * 1024 * 1024}
        tokenCount={0}
        onRemove={vi.fn()}
      />
    );
    expect(screen.getByText(/2\.0 MB/)).toBeInTheDocument();
  });

  test("renders token count when > 0", () => {
    render(
      <FileAttachment
        name="doc.txt"
        sizeBytes={1024}
        tokenCount={1500}
        onRemove={vi.fn()}
      />
    );
    expect(screen.getByText(/1,500 tokens/)).toBeInTheDocument();
  });

  test("does not render token count when 0", () => {
    render(
      <FileAttachment
        name="doc.txt"
        sizeBytes={1024}
        tokenCount={0}
        onRemove={vi.fn()}
      />
    );
    expect(screen.queryByText(/tokens/)).not.toBeInTheDocument();
  });

  test("has remove button with correct aria-label", () => {
    render(
      <FileAttachment
        name="report.pdf"
        sizeBytes={1024}
        tokenCount={0}
        onRemove={vi.fn()}
      />
    );
    expect(screen.getByLabelText("Remove file report.pdf")).toBeInTheDocument();
  });

  test("calls onRemove when remove button clicked", () => {
    const onRemove = vi.fn();
    render(
      <FileAttachment
        name="report.pdf"
        sizeBytes={1024}
        tokenCount={0}
        onRemove={onRemove}
      />
    );
    fireEvent.click(screen.getByLabelText("Remove file report.pdf"));
    expect(onRemove).toHaveBeenCalledOnce();
  });
});

// ---------------------------------------------------------------------------
// ChatProviderSelector
// ---------------------------------------------------------------------------

const makeCP = (overrides: Partial<ChatProviderConfig> = {}): ChatProviderConfig => {
  const defaults: ChatProviderConfig = {
    id: "cp-1",
    name: "GPT-4o Direct",
    llm_provider_id: "llmp-1",
    execution_mode: "direct",
    runtime_settings: {
      temperature: 0.7,
      top_p: 1,
      max_output_tokens: 2048,
      timeout_seconds: 30,
    },
    rlm_max_steps: 16,
    rlm_timeout_seconds: 30,
    rlm_repeat_limit: 2,
    rlm_nudge_at_fraction: 0.6,
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
  };
  return { ...defaults, ...overrides } as ChatProviderConfig;
};

const makeLLMProvider = (overrides: Partial<LLMProviderConfig> = {}): LLMProviderConfig => ({
  id: "llmp-1",
  name: "My OpenAI",
  backend: "openai",
  model: "gpt-4o",
  status: "connected",
  runtime_settings: {
    temperature: 0.7,
    top_p: 1,
    max_output_tokens: 2048,
    timeout_seconds: 30,
  },
  ...overrides,
});

describe("ChatProviderSelector", () => {
  test("shows empty state when no available chat providers", () => {
    // LLM provider is not connected → chatProvider filtered out
    render(
      <ChatProviderSelector
        chatProviders={[makeCP()]}
        llmProviders={[makeLLMProvider({ status: "not_configured" })]}
        selectedIds={[]}
        onSelectionChange={vi.fn()}
      />
    );
    expect(
      screen.getByText(/No Chat Providers available/)
    ).toBeInTheDocument();
  });

  test("renders chip for each available provider", () => {
    const cp1 = makeCP({ id: "cp-1", name: "GPT-4o Direct", llm_provider_id: "llmp-1" });
    const cp2 = makeCP({ id: "cp-2", name: "Claude Direct", llm_provider_id: "llmp-2" });
    const lp1 = makeLLMProvider({ id: "llmp-1" });
    const lp2 = makeLLMProvider({ id: "llmp-2", backend: "anthropic" });

    render(
      <ChatProviderSelector
        chatProviders={[cp1, cp2]}
        llmProviders={[lp1, lp2]}
        selectedIds={["cp-1"]}
        onSelectionChange={vi.fn()}
      />
    );

    expect(screen.getByRole("checkbox", { name: /GPT-4o Direct/ })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /Claude Direct/ })).toBeInTheDocument();
  });

  test("shows All/1 toggle when more than one provider", () => {
    const cp1 = makeCP({ id: "cp-1", name: "GPT-4o Direct", llm_provider_id: "llmp-1" });
    const cp2 = makeCP({ id: "cp-2", name: "Claude Direct", llm_provider_id: "llmp-2" });
    const lp1 = makeLLMProvider({ id: "llmp-1" });
    const lp2 = makeLLMProvider({ id: "llmp-2", backend: "anthropic" });

    render(
      <ChatProviderSelector
        chatProviders={[cp1, cp2]}
        llmProviders={[lp1, lp2]}
        selectedIds={["cp-1"]}
        onSelectionChange={vi.fn()}
      />
    );

    // When not all selected, shows "All"
    expect(screen.getByRole("button", { name: "Select all providers" })).toBeInTheDocument();
  });

  test("toggle button reads 'First only' (not a bare '1') when all selected", () => {
    const cp1 = makeCP({ id: "cp-1", name: "GPT-4o Direct", llm_provider_id: "llmp-1" });
    const cp2 = makeCP({ id: "cp-2", name: "Claude Direct", llm_provider_id: "llmp-2" });
    const lp1 = makeLLMProvider({ id: "llmp-1" });
    const lp2 = makeLLMProvider({ id: "llmp-2", backend: "anthropic" });

    render(
      <ChatProviderSelector
        chatProviders={[cp1, cp2]}
        llmProviders={[lp1, lp2]}
        selectedIds={["cp-1", "cp-2"]}
        onSelectionChange={vi.fn()}
      />
    );

    expect(screen.getByRole("button", { name: "Select first only" })).toHaveTextContent(/^First only$/);
  });

  test("the mode badge letter is explained: chip name and badge tooltip say the mode", () => {
    const cp1 = makeCP({ id: "cp-1", name: "Qwen RLM", execution_mode: "rlm" });
    render(
      <ChatProviderSelector
        chatProviders={[cp1]}
        llmProviders={[makeLLMProvider()]}
        selectedIds={["cp-1"]}
        onSelectionChange={vi.fn()}
      />
    );
    expect(screen.getByRole("checkbox", { name: "Qwen RLM, RLM mode (connected)" })).toBeInTheDocument();
    expect(screen.getByText("R")).toHaveAttribute("title", "RLM mode");
  });

  test("a long provider name truncates inside its chip (M13)", () => {
    const longName = "ChatProvLongName" + "a".repeat(140);
    render(
      <ChatProviderSelector
        chatProviders={[makeCP({ name: longName })]}
        llmProviders={[makeLLMProvider()]}
        selectedIds={["cp-1"]}
        onSelectionChange={vi.fn()}
      />
    );
    const chip = screen.getByRole("checkbox", { name: new RegExp(longName) });
    expect(chip).toHaveAttribute("title", expect.stringContaining(longName));
    const nameSpan = screen.getByText(longName);
    expect(nameSpan.className.split(" ")).toEqual(expect.arrayContaining(["truncate", "min-w-0"]));
  });

  test("calls onSelectionChange when a provider chip is clicked", () => {
    const cp1 = makeCP({ id: "cp-1", name: "GPT-4o Direct", llm_provider_id: "llmp-1" });
    const cp2 = makeCP({ id: "cp-2", name: "Claude Direct", llm_provider_id: "llmp-2" });
    const lp1 = makeLLMProvider({ id: "llmp-1" });
    const lp2 = makeLLMProvider({ id: "llmp-2", backend: "anthropic" });
    const onSelectionChange = vi.fn();

    render(
      <ChatProviderSelector
        chatProviders={[cp1, cp2]}
        llmProviders={[lp1, lp2]}
        selectedIds={["cp-1"]}
        onSelectionChange={onSelectionChange}
      />
    );

    fireEvent.click(screen.getByRole("checkbox", { name: /Claude Direct/ }));
    expect(onSelectionChange).toHaveBeenCalledWith(["cp-1", "cp-2"]);
  });

  test("does not deselect the last selected provider", () => {
    const cp1 = makeCP({ id: "cp-1", name: "GPT-4o Direct" });
    const lp1 = makeLLMProvider({ id: "llmp-1" });
    const onSelectionChange = vi.fn();

    render(
      <ChatProviderSelector
        chatProviders={[cp1]}
        llmProviders={[lp1]}
        selectedIds={["cp-1"]}
        onSelectionChange={onSelectionChange}
      />
    );

    // The single selected chip should be disabled
    const chip = screen.getByRole("checkbox", { name: /GPT-4o Direct/ });
    expect(chip).toBeDisabled();
  });

  test("All toggle selects all providers", () => {
    const cp1 = makeCP({ id: "cp-1", name: "GPT-4o Direct", llm_provider_id: "llmp-1" });
    const cp2 = makeCP({ id: "cp-2", name: "Claude Direct", llm_provider_id: "llmp-2" });
    const lp1 = makeLLMProvider({ id: "llmp-1" });
    const lp2 = makeLLMProvider({ id: "llmp-2", backend: "anthropic" });
    const onSelectionChange = vi.fn();

    render(
      <ChatProviderSelector
        chatProviders={[cp1, cp2]}
        llmProviders={[lp1, lp2]}
        selectedIds={["cp-1"]}
        onSelectionChange={onSelectionChange}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "Select all providers" }));
    expect(onSelectionChange).toHaveBeenCalledWith(["cp-1", "cp-2"]);
  });

  test("1 toggle deselects to first provider only", () => {
    const cp1 = makeCP({ id: "cp-1", name: "GPT-4o Direct", llm_provider_id: "llmp-1" });
    const cp2 = makeCP({ id: "cp-2", name: "Claude Direct", llm_provider_id: "llmp-2" });
    const lp1 = makeLLMProvider({ id: "llmp-1" });
    const lp2 = makeLLMProvider({ id: "llmp-2", backend: "anthropic" });
    const onSelectionChange = vi.fn();

    render(
      <ChatProviderSelector
        chatProviders={[cp1, cp2]}
        llmProviders={[lp1, lp2]}
        selectedIds={["cp-1", "cp-2"]}
        onSelectionChange={onSelectionChange}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "Select first only" }));
    expect(onSelectionChange).toHaveBeenCalledWith(["cp-1"]);
  });

  test("does not show All/1 toggle when only one provider available", () => {
    const cp1 = makeCP({ id: "cp-1", name: "GPT-4o Direct" });
    const lp1 = makeLLMProvider({ id: "llmp-1" });

    render(
      <ChatProviderSelector
        chatProviders={[cp1]}
        llmProviders={[lp1]}
        selectedIds={["cp-1"]}
        onSelectionChange={vi.fn()}
      />
    );

    expect(screen.queryByRole("button", { name: /Select/ })).not.toBeInTheDocument();
  });

  test("has group role with aria-label", () => {
    const cp1 = makeCP();
    const lp1 = makeLLMProvider();

    render(
      <ChatProviderSelector
        chatProviders={[cp1]}
        llmProviders={[lp1]}
        selectedIds={["cp-1"]}
        onSelectionChange={vi.fn()}
      />
    );

    expect(screen.getByRole("group", { name: "Chat providers" })).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// ResponseRating
// ---------------------------------------------------------------------------

describe("ResponseRating", () => {
  const baseProps = {
    executionId: "exec-1",
    sessionId: "sess-1",
    chatProviderId: "cp-1",
    onRate: vi.fn(),
  };

  test("renders thumbs up and thumbs down buttons", () => {
    render(<ResponseRating {...baseProps} />);
    expect(screen.getByLabelText("Thumbs up")).toBeInTheDocument();
    expect(screen.getByLabelText("Thumbs down")).toBeInTheDocument();
  });

  test("calls onRate with 'up' when thumbs up clicked", () => {
    const onRate = vi.fn();
    render(<ResponseRating {...baseProps} onRate={onRate} />);
    fireEvent.click(screen.getByLabelText("Thumbs up"));
    expect(onRate).toHaveBeenCalledWith("exec-1", "up");
  });

  test("calls onRate with 'down' when thumbs down clicked", () => {
    const onRate = vi.fn();
    render(<ResponseRating {...baseProps} onRate={onRate} />);
    fireEvent.click(screen.getByLabelText("Thumbs down"));
    expect(onRate).toHaveBeenCalledWith("exec-1", "down");
  });

  test("toggles off when clicking already-active rating", () => {
    const onRate = vi.fn();
    render(<ResponseRating {...baseProps} currentRating="up" onRate={onRate} />);
    fireEvent.click(screen.getByLabelText("Thumbs up"));
    expect(onRate).toHaveBeenCalledWith("exec-1", null);
  });

  test("switches rating when clicking opposite thumb", () => {
    const onRate = vi.fn();
    render(<ResponseRating {...baseProps} currentRating="up" onRate={onRate} />);
    fireEvent.click(screen.getByLabelText("Thumbs down"));
    expect(onRate).toHaveBeenCalledWith("exec-1", "down");
  });

  test("applies highlight class to active thumbs up", () => {
    render(<ResponseRating {...baseProps} currentRating="up" />);
    const btn = screen.getByLabelText("Thumbs up");
    expect(btn.className).toMatch(/emerald/);
  });

  test("applies highlight class to active thumbs down", () => {
    render(<ResponseRating {...baseProps} currentRating="down" />);
    const btn = screen.getByLabelText("Thumbs down");
    expect(btn.className).toMatch(/red/);
  });

  test("no highlight when no rating selected", () => {
    render(<ResponseRating {...baseProps} />);
    expect(screen.getByLabelText("Thumbs up").className).not.toMatch(/emerald/);
    expect(screen.getByLabelText("Thumbs down").className).not.toMatch(/red/);
  });
});

// ---------------------------------------------------------------------------
// PickWinner
// ---------------------------------------------------------------------------

describe("PickWinner", () => {
  const responses = [
    { executionId: "exec-1", cpId: "cp-1", name: "GPT-4o" },
    { executionId: "exec-2", cpId: "cp-2", name: "Claude" },
  ];

  test("renders null when fewer than 2 responses", () => {
    const { container } = render(
      <PickWinner
        sessionId="sess-1"
        responses={[responses[0]]}
        onPick={vi.fn()}
      />
    );
    expect(container.innerHTML).toBe("");
  });

  test("renders null when zero responses", () => {
    const { container } = render(
      <PickWinner sessionId="sess-1" responses={[]} onPick={vi.fn()} />
    );
    expect(container.innerHTML).toBe("");
  });

  test("renders a button for each response when >= 2", () => {
    render(
      <PickWinner sessionId="sess-1" responses={responses} onPick={vi.fn()} />
    );
    expect(screen.getByRole("button", { name: "GPT-4o" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Claude" })).toBeInTheDocument();
  });

  test("calls onPick with executionId when button clicked", () => {
    const onPick = vi.fn();
    render(
      <PickWinner sessionId="sess-1" responses={responses} onPick={onPick} />
    );
    fireEvent.click(screen.getByRole("button", { name: "Claude" }));
    expect(onPick).toHaveBeenCalledWith("exec-2");
  });

  test("highlights current winner with amber styling", () => {
    render(
      <PickWinner
        sessionId="sess-1"
        responses={responses}
        currentWinnerId="exec-1"
        onPick={vi.fn()}
      />
    );
    const winnerBtn = screen.getByRole("button", { name: "GPT-4o" });
    expect(winnerBtn.className).toMatch(/amber/);
    const otherBtn = screen.getByRole("button", { name: "Claude" });
    expect(otherBtn.className).not.toMatch(/amber/);
  });

  test("no highlight when no winner selected", () => {
    render(
      <PickWinner sessionId="sess-1" responses={responses} onPick={vi.fn()} />
    );
    expect(screen.getByRole("button", { name: "GPT-4o" }).className).not.toMatch(/amber/);
    expect(screen.getByRole("button", { name: "Claude" }).className).not.toMatch(/amber/);
  });

  test("renders 'Best:' label", () => {
    render(
      <PickWinner sessionId="sess-1" responses={responses} onPick={vi.fn()} />
    );
    expect(screen.getByText("Best:")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// JudgeScores
// ---------------------------------------------------------------------------

describe("JudgeScores", () => {
  const dimensions = {
    relevance: 4.2,
    correctness: 3.8,
    completeness: 4.5,
  };

  test("renders abbreviated labels for known dimensions", () => {
    render(<JudgeScores dimensions={dimensions} overallScore={4.0} />);
    expect(screen.getByText("Rel:")).toBeInTheDocument();
    expect(screen.getByText("Cor:")).toBeInTheDocument();
    expect(screen.getByText("Com:")).toBeInTheDocument();
  });

  test("renders raw key for unknown dimensions", () => {
    render(
      <JudgeScores dimensions={{ novelty: 3.0 }} overallScore={3.0} />
    );
    expect(screen.getByText("novelty:")).toBeInTheDocument();
  });

  test("renders score values", () => {
    render(<JudgeScores dimensions={dimensions} overallScore={4.0} />);
    expect(screen.getByText("4.2")).toBeInTheDocument();
    expect(screen.getByText("3.8")).toBeInTheDocument();
    expect(screen.getByText("4.5")).toBeInTheDocument();
  });

  test("renders overall score with /5 suffix", () => {
    render(<JudgeScores dimensions={dimensions} overallScore={4.167} />);
    expect(screen.getByText("4.2/5")).toBeInTheDocument();
  });

  test("renders dimension bars with correct width", () => {
    render(<JudgeScores dimensions={{ relevance: 5 }} overallScore={5.0} />);
    const bar = document.querySelector('[style*="width"]') as HTMLElement;
    expect(bar).toBeTruthy();
    expect(bar.style.width).toBe("100%");
  });

  test("renders 0-score dimension bar with 0% width", () => {
    render(<JudgeScores dimensions={{ relevance: 0 }} overallScore={0.0} />);
    const bar = document.querySelector('[style*="width"]') as HTMLElement;
    expect(bar).toBeTruthy();
    expect(bar.style.width).toBe("0%");
  });

  test("renders empty state when no dimensions", () => {
    const { container } = render(
      <JudgeScores dimensions={{}} overallScore={0.0} />
    );
    expect(screen.getByText("0.0/5")).toBeInTheDocument();
    // No dimension entries rendered
    expect(container.querySelectorAll('[style*="width"]')).toHaveLength(0);
  });
});
