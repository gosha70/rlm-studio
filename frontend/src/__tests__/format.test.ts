/** Cost formatting for charts and tables (lib/format.ts). */

import { describe, test, expect } from "vitest";
import { formatUsd, formatUsdTick } from "@/lib/format";

describe("formatUsd", () => {
  test("keeps four decimals for ordinary costs", () => {
    expect(formatUsd(0.036)).toBe("$0.0360");
    expect(formatUsd(1.23456)).toBe("$1.2346");
    expect(formatUsd(0)).toBe("$0.0000");
  });

  test("shows a tiny non-zero cost instead of rounding it to $0.0000", () => {
    expect(formatUsd(6e-7)).toBe("$0.0000006");
    expect(formatUsd(0.00004321)).toBe("$0.000043");
  });
});

describe("formatUsdTick", () => {
  test("never uses scientific notation", () => {
    for (const v of [6e-7, 3e-7, 1.5e-7]) {
      expect(formatUsdTick(v)).not.toMatch(/e-/);
    }
    expect(formatUsdTick(6e-7)).toBe("$0.0000006");
    expect(formatUsdTick(1.5e-7)).toBe("$0.00000015");
  });

  test("keeps round ticks short", () => {
    expect(formatUsdTick(0)).toBe("$0");
    expect(formatUsdTick(2)).toBe("$2");
    expect(formatUsdTick(0.5)).toBe("$0.5");
  });
});
