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
