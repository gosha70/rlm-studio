/**
 * Chat page behaviour found by the browser pre-pass (Wave 3, area D).
 *
 * - M7: upload rejections carry the server's reason; empty and oversized
 *   files are refused before any bytes are sent.
 * - M8: a Chat Provider whose LLM provider went offline is dropped from the
 *   selection restored from localStorage, so it cannot run.
 */

import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { SWRConfig } from "swr";
import { vi, describe, test, expect, beforeEach } from "vitest";

vi.mock("sonner", () => ({
  toast: { error: vi.fn(), warning: vi.fn(), success: vi.fn() },
}));
vi.mock("@/components/shared/app-shell", () => ({
  AppShell: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));
vi.mock("@/lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api")>()),
  getChatProviders: vi.fn(),
  getLLMProviders: vi.fn(),
  getConfig: vi.fn(),
  getSession: vi.fn(),
  getSessionEvaluations: vi.fn(),
  getExecutions: vi.fn(),
  getTrace: vi.fn(),
  submitChat: vi.fn(),
  uploadFile: vi.fn(),
}));

import { toast } from "sonner";
import * as api from "@/lib/api";
import ChatPage from "@/app/page";
import { STORAGE_KEY_SELECTED_CHAT_PROVIDERS } from "@/lib/branding";
import type { ChatProviderConfig, LLMProviderConfig } from "@/lib/api";

const mocked = vi.mocked(api);
const MB = 1024 * 1024;

const llm = (overrides: Partial<LLMProviderConfig> = {}): LLMProviderConfig =>
  ({
    id: "llmp-1",
    name: "Local",
    backend: "ollama",
    model: "llama3.2",
    status: "connected",
    runtime_settings: { temperature: 0.7, top_p: 1, max_output_tokens: 2048, timeout_seconds: 30 },
    ...overrides,
  }) as LLMProviderConfig;

const cp = (overrides: Partial<ChatProviderConfig> = {}): ChatProviderConfig =>
  ({
    id: "cp-1",
    name: "Llama Direct",
    llm_provider_id: "llmp-1",
    execution_mode: "direct",
    runtime_settings: { temperature: 0.7, top_p: 1, max_output_tokens: 2048, timeout_seconds: 30 },
    ...overrides,
  }) as ChatProviderConfig;

function renderPage() {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <ChatPage />
    </SWRConfig>,
  );
}

function fileOfSize(name: string, size: number): File {
  const file = new File(["x"], name, { type: "text/plain" });
  Object.defineProperty(file, "size", { value: size });
  return file;
}

async function pickFiles(container: HTMLElement, files: File[]) {
  const input = container.querySelector('input[type="file"]') as HTMLInputElement;
  fireEvent.change(input, { target: { files } });
}

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  Element.prototype.scrollIntoView = vi.fn();
  mocked.getChatProviders.mockResolvedValue([cp()]);
  mocked.getLLMProviders.mockResolvedValue([llm()]);
  mocked.getConfig.mockResolvedValue({} as Awaited<ReturnType<typeof api.getConfig>>);
  mocked.getSessionEvaluations.mockResolvedValue({ thumb_ratings: [], best_picks: [], judge_scores: [] } as unknown as Awaited<
    ReturnType<typeof api.getSessionEvaluations>
  >);
  mocked.getExecutions.mockResolvedValue([]);
});

describe("Chat uploads (M7)", () => {
  test("a rejected upload shows the server's reason", async () => {
    mocked.uploadFile.mockRejectedValue(new Error("Unsupported file type: .zip. Allowed: .csv, .docx"));
    const { container } = renderPage();
    await pickFiles(container, [fileOfSize("fake.txt", 10)]);
    await waitFor(() =>
      expect(toast.error).toHaveBeenCalledWith(
        'Couldn\'t upload "fake.txt": Unsupported file type: .zip. Allowed: .csv, .docx',
      ),
    );
  });

  test("an empty file is refused without uploading", async () => {
    const { container } = renderPage();
    await pickFiles(container, [fileOfSize("empty.txt", 0)]);
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(expect.stringMatching(/"empty.txt" is empty/)));
    expect(mocked.uploadFile).not.toHaveBeenCalled();
  });

  test("a text file over the per-message limit is refused before uploading", async () => {
    const { container } = renderPage();
    await pickFiles(container, [fileOfSize("big60.txt", 60 * MB)]);
    await waitFor(() =>
      expect(toast.error).toHaveBeenCalledWith(expect.stringMatching(/"big60.txt" skipped .* 50 MB per-message limit/)),
    );
    expect(mocked.uploadFile).not.toHaveBeenCalled();
  });

  test("a file over the server's upload limit is refused before uploading", async () => {
    const { container } = renderPage();
    await pickFiles(container, [fileOfSize("huge.pdf", 205 * MB)]);
    await waitFor(() =>
      expect(toast.error).toHaveBeenCalledWith(expect.stringMatching(/"huge.pdf" is larger than the 200 MB upload limit/)),
    );
    expect(mocked.uploadFile).not.toHaveBeenCalled();
  });
});

describe("uploadFile error reason (M7)", () => {
  test("rejects with the server's message, not the status code", async () => {
    const { uploadFile } = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
    class FakeXHR {
      status = 422;
      responseText = JSON.stringify({ error: { code: "ERROR", message: "Could not extract text from file: EOF marker not found" } });
      upload = {};
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      open() {}
      send() {
        this.onload?.();
      }
    }
    vi.stubGlobal("XMLHttpRequest", FakeXHR);
    try {
      await expect(uploadFile(fileOfSize("corrupt.pdf", 10))).rejects.toThrow(
        "Could not extract text from file: EOF marker not found",
      );
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

describe("Chat Provider selection (M8)", () => {
  async function sendMessage(text: string) {
    const input = await screen.findByLabelText("Message input");
    await waitFor(() => expect(input).not.toBeDisabled());
    fireEvent.change(input, { target: { value: text } });
    fireEvent.click(screen.getByLabelText("Send message"));
  }

  beforeEach(() => {
    mocked.submitChat.mockResolvedValue({ execution_id: "exec-1", session_id: "sess-1", status: "accepted" });
    mocked.getTrace.mockReturnValue(new Promise(() => {}));
    mocked.getChatProviders.mockResolvedValue([
      cp({ id: "cp-on", name: "Online", llm_provider_id: "llmp-on" }),
      cp({ id: "cp-off", name: "Offline", llm_provider_id: "llmp-off" }),
      cp({ id: "cp-on2", name: "Online 2", llm_provider_id: "llmp-on" }),
    ]);
    mocked.getLLMProviders.mockResolvedValue([
      llm({ id: "llmp-on", status: "connected" }),
      llm({ id: "llmp-off", status: "offline" }),
    ]);
  });

  test("a saved provider that went offline is not used to run", async () => {
    localStorage.setItem(STORAGE_KEY_SELECTED_CHAT_PROVIDERS, JSON.stringify(["cp-off"]));
    renderPage();
    await sendMessage("hello");
    await waitFor(() => expect(mocked.submitChat).toHaveBeenCalled());
    const used = mocked.submitChat.mock.calls.map(([req]) => req.chat_provider_id);
    expect(used).not.toContain("cp-off");
    expect(used).toEqual(["cp-on"]);
  });

  test("a saved online selection is kept, not reset to the first provider", async () => {
    localStorage.setItem(STORAGE_KEY_SELECTED_CHAT_PROVIDERS, JSON.stringify(["cp-on2"]));
    renderPage();
    await sendMessage("hello");
    await waitFor(() => expect(mocked.submitChat).toHaveBeenCalled());
    expect(mocked.submitChat.mock.calls.map(([req]) => req.chat_provider_id)).toEqual(["cp-on2"]);
  });
});
