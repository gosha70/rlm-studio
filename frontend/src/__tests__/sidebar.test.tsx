/**
 * Sidebar sessions list (browser pre-pass M10): delete asks first, and
 * rename has a visible action instead of only a hidden double-click.
 */

import { render, screen, fireEvent } from "@testing-library/react";
import { vi, describe, test, expect, beforeEach, afterEach } from "vitest";

vi.mock("next/navigation", () => ({ usePathname: () => "/" }));
vi.mock("swr", () => ({ default: () => ({ data: undefined }) }));

import { Sidebar } from "@/components/shared/sidebar";
import type { SessionSummary } from "@/lib/api";

const SESSION = { id: "s-1", name: "Orion notes" } as SessionSummary;

function renderSidebar(props: Partial<React.ComponentProps<typeof Sidebar>> = {}) {
  return render(<Sidebar collapsed={false} onToggle={vi.fn()} sessions={[SESSION]} {...props} />);
}

describe("Sidebar sessions (M10)", () => {
  let confirmSpy: ReturnType<typeof vi.spyOn>;
  beforeEach(() => {
    confirmSpy = vi.spyOn(window, "confirm");
  });
  afterEach(() => {
    confirmSpy.mockRestore();
  });

  test("delete asks for confirmation and does nothing when cancelled", () => {
    confirmSpy.mockReturnValue(false);
    const onDeleteSession = vi.fn();
    renderSidebar({ onDeleteSession });
    fireEvent.click(screen.getByRole("button", { name: "Delete session Orion notes" }));
    expect(confirmSpy).toHaveBeenCalledWith(expect.stringContaining('Delete session "Orion notes"?'));
    expect(onDeleteSession).not.toHaveBeenCalled();
  });

  test("delete goes ahead once confirmed", () => {
    confirmSpy.mockReturnValue(true);
    const onDeleteSession = vi.fn();
    renderSidebar({ onDeleteSession });
    fireEvent.click(screen.getByRole("button", { name: "Delete session Orion notes" }));
    expect(onDeleteSession).toHaveBeenCalledWith("s-1");
  });

  test("a Rename button opens the name for editing", () => {
    const onRenameSession = vi.fn();
    renderSidebar({ onRenameSession });
    fireEvent.click(screen.getByRole("button", { name: "Rename session Orion notes" }));
    const input = screen.getByLabelText("Rename session");
    fireEvent.change(input, { target: { value: "Orion maintenance" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onRenameSession).toHaveBeenCalledWith("s-1", "Orion maintenance");
  });

  test("the active session's actions are always visible", () => {
    renderSidebar({ activeSessionId: "s-1" });
    expect(screen.getByRole("button", { name: "Rename session Orion notes" }).className.split(" ")).toContain("opacity-100");
    expect(screen.getByRole("button", { name: "Delete session Orion notes" }).className.split(" ")).toContain("opacity-100");
  });
});

describe("Long session names (SH2-1)", () => {
  const LONG = "a".repeat(200);

  test("no row sits inside a display:table box that grows with the name", () => {
    renderSidebar({ sessions: [{ id: "s-long", name: LONG } as SessionSummary, SESSION] });
    const aside = screen.getByRole("complementary");
    for (const button of screen.getAllByRole("button", { name: /^Delete session / })) {
      for (let el = button.parentElement; el && el !== aside; el = el.parentElement) {
        expect(getComputedStyle(el).display).not.toBe("table");
      }
    }
  });

  test("a long name truncates in a shrinkable button and the full name is on hover", () => {
    renderSidebar({ sessions: [{ id: "s-long", name: LONG } as SessionSummary] });
    const name = screen.getByRole("button", { name: LONG });
    expect(name.className.split(" ")).toEqual(expect.arrayContaining(["min-w-0", "flex-1", "truncate"]));
    expect(name).toHaveAttribute("title", LONG);
    for (const label of [`Rename session ${LONG}`, `Delete session ${LONG}`]) {
      expect(screen.getByRole("button", { name: label }).className.split(" ")).toContain("shrink-0");
    }
  });
});

describe("Session actions take space only when shown (P3-CS-2)", () => {
  const classesOf = (name: string) => screen.getByRole("button", { name }).className.split(" ");

  test("an idle row's Rename and Delete take no width until hover or focus", () => {
    renderSidebar();
    for (const name of ["Rename session Orion notes", "Delete session Orion notes"]) {
      const classes = classesOf(name);
      expect(classes).toEqual(expect.arrayContaining(["w-0", "overflow-hidden", "shrink-0"]));
      expect(classes).not.toContain("w-6");
      expect(classes).toEqual(
        expect.arrayContaining(["group-hover:w-6", "group-focus-within:w-6", "group-focus-within:opacity-100"]),
      );
    }
  });

  test("the active row keeps its actions at full width", () => {
    renderSidebar({ activeSessionId: "s-1" });
    for (const name of ["Rename session Orion notes", "Delete session Orion notes"]) {
      expect(classesOf(name)).toEqual(expect.arrayContaining(["w-6", "opacity-100"]));
      expect(classesOf(name)).not.toContain("w-0");
    }
  });

  test("Tab from the session name reaches Rename, then Delete", async () => {
    const user = (await import("@testing-library/user-event")).default.setup();
    renderSidebar();
    screen.getByRole("button", { name: "Orion notes" }).focus();
    await user.tab();
    expect(screen.getByRole("button", { name: "Rename session Orion notes" })).toHaveFocus();
    await user.tab();
    expect(screen.getByRole("button", { name: "Delete session Orion notes" })).toHaveFocus();
  });
});

describe("Sidebar logo (owner #1)", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.resetModules();
  });

  test("the logo URL carries the bundled UI's base path", async () => {
    vi.stubEnv("NEXT_PUBLIC_BASE_PATH", "/studio");
    vi.resetModules();
    const { Sidebar: BundledSidebar } = await import("@/components/shared/sidebar");
    render(<BundledSidebar collapsed={false} onToggle={vi.fn()} />);
    const src = screen.getByRole("img", { name: "RLM Studio" }).getAttribute("src") ?? "";
    expect(decodeURIComponent(src)).toContain("/studio/logo.png");
  });

  test("the logo is shown at a fixed size, not the file's natural size", () => {
    render(<Sidebar collapsed={false} onToggle={vi.fn()} />);
    expect(screen.getByRole("img", { name: "RLM Studio" }).style.width).toBe("28px");
  });
});
