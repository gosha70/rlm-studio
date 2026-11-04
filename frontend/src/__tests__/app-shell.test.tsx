/**
 * Header connection status (browser pre-pass H3): with the backend stopped
 * the header kept "Connected" — SWR keeps the last good response and the
 * shell ignored the error — and after an error SWR's interval polling skipped
 * the key, so it never noticed the backend coming back either.
 */

import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { SWRConfig } from "swr";
import { vi, describe, test, expect, beforeEach } from "vitest";

vi.mock("@/lib/constants", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/constants")>()),
  HEALTH_POLL_MS: 40,
  HEALTH_RETRY_MS: 40,
}));
vi.mock("@/lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api")>()),
  getHealth: vi.fn(),
  getSessions: vi.fn(),
}));
vi.mock("@/components/shared/sidebar", () => ({ Sidebar: () => null }));
vi.mock("@/components/shared/theme-toggle", () => ({ ThemeToggle: () => null }));

import * as api from "@/lib/api";
import { AppShell } from "@/components/shared/app-shell";

const mocked = vi.mocked(api);
const OK = { status: "ok" } as Awaited<ReturnType<typeof api.getHealth>>;

function renderShell() {
  render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <AppShell>
        <p>page</p>
      </AppShell>
    </SWRConfig>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  mocked.getSessions.mockResolvedValue([]);
});

describe("AppShell connection status", () => {
  test("a reachable backend shows Connected and no banner", async () => {
    mocked.getHealth.mockResolvedValue(OK);
    renderShell();
    expect(await screen.findByText("Connected")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  test("before the first answer the header neither claims Connected nor Offline", () => {
    mocked.getHealth.mockReturnValue(new Promise(() => {}));
    renderShell();
    expect(screen.getByText("Loading...")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  test("losing the backend after a good check shows Offline and the banner", async () => {
    mocked.getHealth.mockResolvedValueOnce(OK).mockRejectedValue(new TypeError("Failed to fetch"));
    renderShell();
    expect(await screen.findByText("Connected")).toBeInTheDocument();
    expect(await screen.findByText("Offline")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent(/Can.t reach the RLM Studio server/);
  });

  test("the header recovers on its own when the backend comes back", async () => {
    mocked.getHealth
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValue(OK);
    renderShell();
    expect(await screen.findByText("Offline")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("Connected")).toBeInTheDocument(), { timeout: 3000 });
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
