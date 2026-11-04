/**
 * Theme toggle (browser pre-pass M19): with the OS in dark mode and no saved
 * choice, next-themes reports theme "system", so the first click must act on
 * the resolved theme — otherwise it "switches" to dark and nothing changes.
 */

import { render, screen, fireEvent } from "@testing-library/react";
import { vi, describe, test, expect, beforeEach } from "vitest";

const setTheme = vi.fn();
let themeState: { theme?: string; resolvedTheme?: string } = {};
vi.mock("next-themes", () => ({
  useTheme: () => ({ ...themeState, setTheme }),
}));

import { ThemeToggle } from "@/components/shared/theme-toggle";

beforeEach(() => setTheme.mockClear());

describe("ThemeToggle (M19)", () => {
  test("OS dark with no saved choice: the first click switches to light", () => {
    themeState = { theme: "system", resolvedTheme: "dark" };
    render(<ThemeToggle />);
    const button = screen.getByRole("button", { name: "Switch to light theme" });
    fireEvent.click(button);
    expect(setTheme).toHaveBeenCalledWith("light");
  });

  test("OS light with no saved choice: the first click switches to dark", () => {
    themeState = { theme: "system", resolvedTheme: "light" };
    render(<ThemeToggle />);
    fireEvent.click(screen.getByRole("button", { name: "Switch to dark theme" }));
    expect(setTheme).toHaveBeenCalledWith("dark");
  });
});
