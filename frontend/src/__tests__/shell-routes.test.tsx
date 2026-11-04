/**
 * Shell routing under the bundled UI's /studio base path (browser pre-pass M18).
 *
 * - The Concepts page's bundled replay is a file in public/, which Next
 *   serves under the base path but fetch() does not prefix.
 * - Unknown routes render a not-found page inside the app shell with a way
 *   back, not Next's bare 404.
 */

import { render, screen } from "@testing-library/react";
import { vi, describe, test, expect, afterEach } from "vitest";

vi.mock("@/components/shared/app-shell", () => ({
  AppShell: ({ children }: { children: React.ReactNode }) => <div data-testid="app-shell">{children}</div>,
}));

import NotFound from "@/app/not-found";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
  vi.resetModules();
});

describe("bundled replay fetch (M18)", () => {
  test("is prefixed with the base path in the bundled build", async () => {
    vi.stubEnv("NEXT_PUBLIC_BASE_PATH", "/studio");
    vi.resetModules();
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify({ id: "bundled" })));
    const { getBundledReplay } = await import("@/lib/api");
    await getBundledReplay();
    expect(fetchSpy).toHaveBeenCalledWith("/studio/learn/replays/bundled-rlm-demo.json");
  });

  test("is unprefixed in the dev build", async () => {
    vi.stubEnv("NEXT_PUBLIC_BASE_PATH", undefined);
    vi.resetModules();
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify({ id: "bundled" })));
    const { getBundledReplay } = await import("@/lib/api");
    await getBundledReplay();
    expect(fetchSpy).toHaveBeenCalledWith("/learn/replays/bundled-rlm-demo.json");
  });
});

describe("not-found page (M18)", () => {
  test("keeps the app shell and links back to Chat", () => {
    render(<NotFound />);
    expect(screen.getByTestId("app-shell")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Page not found" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Go to Chat" })).toHaveAttribute("href", "/");
  });
});
