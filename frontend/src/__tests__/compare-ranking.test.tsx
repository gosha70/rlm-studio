/**
 * AC-17 — client-side ranker on the Compare page.
 *
 * Exercises ``_rankSlots`` directly (pure function, exported from
 * ``compare/page.tsx`` for test coverage). Asserts the null-last
 * discipline for the three Phase-5 prefill/decode ranking metrics:
 * slots missing the relevant telemetry sort after slots with a
 * value, matching the "failed slots last" contract.
 */

import { describe, test, expect, vi } from "vitest";

// next/navigation is imported transitively by compare/page.tsx.
// Stub so the module loads under vitest's jsdom without a router.
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
  useSearchParams: () => ({ get: () => null }),
}));

// SWR import path is used by the compare page for data fetching.
vi.mock("swr", () => ({
  default: () => ({ data: undefined, error: undefined, isLoading: false }),
  useSWRConfig: () => ({ mutate: vi.fn() }),
}));

vi.mock("@/components/shared/app-shell", () => ({
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  AppShell: ({ children }: { children: any }) => children,
}));

import { _rankSlots } from "@/app/compare/page";
import type { CompareMatrixSlotResponse } from "@/lib/api";

function _slot(
  overrides: Partial<CompareMatrixSlotResponse> = {},
): CompareMatrixSlotResponse {
  return {
    slot_id: "s",
    label: "slot",
    mode: "direct",
    provider: "openai",
    model: "gpt-4o",
    chat_provider_id: "cp",
    chat_provider_name: null,
    execution_id: "e",
    success: true,
    answer: "ok",
    error: null,
    input_tokens: 100,
    output_tokens: 20,
    total_tokens: 120,
    total_cost: 0.01,
    elapsed_seconds: 1.0,
    steps: 1,
    total_prompt_tokens: 100,
    total_completion_tokens: 20,
    total_cached_tokens: 0,
    total_decode_ms: 0,
    median_ttft_ms: null,
    cache_hit_rate: 0,
    ...overrides,
  };
}

describe("_rankSlots — TTFT", () => {
  test("sorts ascending by median_ttft_ms with null-TTFT slots last", () => {
    const slots: CompareMatrixSlotResponse[] = [
      _slot({ slot_id: "no-ttft", median_ttft_ms: null }),
      _slot({ slot_id: "slow", median_ttft_ms: 400 }),
      _slot({ slot_id: "fast", median_ttft_ms: 100 }),
      _slot({ slot_id: "mid", median_ttft_ms: 200 }),
    ];

    const ranking = _rankSlots(slots, "ttft", {});

    // Expected order: fast, mid, slow, no-ttft (null sorts last).
    expect(ranking.map((i) => slots[i].slot_id)).toEqual([
      "fast",
      "mid",
      "slow",
      "no-ttft",
    ]);
  });

  test("slots with valid TTFT come before both null-TTFT and failed slots", () => {
    const slots: CompareMatrixSlotResponse[] = [
      _slot({ slot_id: "failed", success: false }),
      _slot({ slot_id: "no-ttft", median_ttft_ms: null }),
      _slot({ slot_id: "fast", median_ttft_ms: 50 }),
    ];

    const ranking = _rankSlots(slots, "ttft", {});
    const order = ranking.map((i) => slots[i].slot_id);

    // `fast` (real TTFT) ranks before both `no-ttft` (null telemetry
    // treated as +Infinity) and `failed` (also +Infinity). Stable
    // sort breaks the tie between the two +Infinity entries by input
    // order — the spec only pins "valid telemetry wins", not the
    // relative order of the two absence cases.
    expect(order[0]).toBe("fast");
    expect(order.slice(1).sort()).toEqual(["failed", "no-ttft"]);
  });
});

describe("_rankSlots — decode_tokens_per_sec", () => {
  test("sorts descending with zero-decode slots last", () => {
    const slots: CompareMatrixSlotResponse[] = [
      _slot({
        slot_id: "slow",
        total_decode_ms: 1000,
        total_completion_tokens: 100,
      }), // 100 tok/s
      _slot({
        slot_id: "no-decode",
        total_decode_ms: 0,
        total_completion_tokens: 50,
      }), // null → last
      _slot({
        slot_id: "fast",
        total_decode_ms: 100,
        total_completion_tokens: 100,
      }), // 1000 tok/s
    ];

    const ranking = _rankSlots(slots, "decode_tokens_per_sec", {});

    expect(ranking.map((i) => slots[i].slot_id)).toEqual([
      "fast",
      "slow",
      "no-decode",
    ]);
  });
});

describe("_rankSlots — cache_hit_rate", () => {
  test("sorts descending with null cache_hit_rate slots last", () => {
    const slots: CompareMatrixSlotResponse[] = [
      _slot({ slot_id: "low", cache_hit_rate: 0.1 }),
      _slot({ slot_id: "high", cache_hit_rate: 0.9 }),
      // Missing cache_hit_rate entirely (simulate legacy bundle).
      _slot({ slot_id: "missing", cache_hit_rate: undefined }),
      _slot({ slot_id: "mid", cache_hit_rate: 0.5 }),
    ];

    const ranking = _rankSlots(slots, "cache_hit_rate", {});

    expect(ranking.map((i) => slots[i].slot_id)).toEqual([
      "high",
      "mid",
      "low",
      "missing",
    ]);
  });
});

describe("_rankSlots — failures, unjudged and unpriced slots", () => {
  const ok = (id: string, extra: Partial<CompareMatrixSlotResponse> = {}) =>
    _slot({ execution_id: id, slot_id: id, ...extra });
  const failed = (id: string) => _slot({ execution_id: id, slot_id: id, success: false, error: "boom" });

  test("judge score: unjudged successes rank above failures, judged ones first", () => {
    // Browser pre-pass: with no judge scores, failures ranked above the three
    // successes because Infinity - 1 === Infinity made the sort arbitrary.
    const slots = [failed("f1"), ok("a"), failed("f2"), ok("b"), ok("c")];
    expect(_rankSlots(slots, "judge_score", {})).toEqual([1, 3, 4, 0, 2]);

    const scores = { c: { execution_id: "c", overall_score: 4.5 } } as never;
    expect(_rankSlots(slots, "judge_score", scores)).toEqual([4, 1, 3, 0, 2]);
  });

  test("cost: an unpriced $0 run ranks after priced runs, before failures", () => {
    const slots = [
      ok("unpriced", { total_cost: 0, cost_known: false }),
      failed("f"),
      ok("priced", { total_cost: 0.02 }),
      ok("free-local", { total_cost: 0 }),
    ];
    expect(_rankSlots(slots, "cost", {})).toEqual([3, 2, 0, 1]);
  });

  test("answer per cost: known-free runs first by answer length, unpriced after priced", () => {
    const slots = [
      ok("unpriced", { total_cost: 0, cost_known: false, answer: "x".repeat(500) }),
      ok("priced", { total_cost: 0.01, answer: "x".repeat(100) }),
      ok("free-short", { total_cost: 0, answer: "x".repeat(10) }),
      ok("free-long", { total_cost: 0, answer: "x".repeat(50) }),
      failed("f"),
    ];
    expect(_rankSlots(slots, "answer_per_cost", {})).toEqual([3, 2, 1, 0, 4]);
  });
});
