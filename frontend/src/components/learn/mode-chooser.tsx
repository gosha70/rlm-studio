"use client";

/**
 * Concepts §B mode chooser — replaces the V1 static decision strip.
 *
 * Pure client-side lookup. Selecting a scenario in the dropdown
 * highlights the recommended mode and surfaces a one-line rationale
 * directly under the strip. The rows are the modes a Chat Provider
 * can run (Settings → Chat Providers). The dropdown opens with the
 * "Not sure…" scenario selected, which recommends comparing modes side
 * by side, so the chooser is never an empty prompt.
 */

import { useState } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  displayModeName,
  MODE_DIRECT,
  MODE_RAG,
  MODE_RLM,
  MODE_RLM_OFFICIAL,
  type ALL_EXECUTION_MODES,
} from "@/lib/constants";
import { cn } from "@/lib/utils";

export type LearnMode = (typeof ALL_EXECUTION_MODES)[number];

interface ModeRow {
  mode: LearnMode;
  body: string;
  dot: string;
}

const MODE_ROWS: ReadonlyArray<ModeRow> = [
  {
    mode: MODE_DIRECT,
    body: "small, self-contained input that fits in one prompt.",
    dot: "bg-emerald-500",
  },
  {
    mode: MODE_RLM,
    body:
      "large or complex content; the model inspects and reasons step-by-step through a sandboxed Python REPL.",
    dot: "bg-blue-500",
  },
  {
    mode: MODE_RAG,
    body:
      "a specific question about a large document; the model answers from the passages most similar to the question.",
    dot: "bg-amber-500",
  },
  {
    mode: MODE_RLM_OFFICIAL,
    body:
      "the RLM paper authors' own implementation (the rlms package), to check Studio's RLM against the reference. Needs the interop extra.",
    dot: "bg-purple-500",
  },
];

/** What the "not sure" scenario recommends instead of a single mode. */
const COMPARE_RECOMMENDATION = "compare side by side";

interface Scenario {
  id: string;
  prompt: string;
  /** The highlighted row; none when the advice is to compare modes. */
  recommendedMode?: LearnMode;
  rationale: string;
}

const SCENARIOS: ReadonlyArray<Scenario> = [
  {
    id: "not-sure",
    prompt: "Not sure which mode fits the task",
    rationale:
      "Select several Chat Providers in Chat — say a Direct one and an RLM one — and each answers the same question side by side with its tokens, cost and time, so you can pick by evidence. The LLM Tuner runs a whole provider × mode grid.",
  },
  {
    id: "short-note",
    prompt: "Summarize a short note",
    recommendedMode: MODE_DIRECT,
    rationale:
      "Direct is one-shot and cheapest — the right call when the input fits comfortably in a single prompt.",
  },
  {
    id: "long-doc",
    prompt: "Answer questions about a long design doc",
    recommendedMode: MODE_RLM,
    rationale:
      "RLM lets the model inspect the document iteratively instead of trying to fit it all into one prompt.",
  },
  {
    id: "fact-lookup",
    prompt: "Look up one fact in a large document",
    recommendedMode: MODE_RAG,
    rationale:
      "RAG retrieves the passages most similar to the question and answers from those — quick and cheap for a lookup, but it can miss answers that need the whole document.",
  },
  {
    id: "reference-check",
    prompt: "Check a result against the paper's own RLM",
    recommendedMode: MODE_RLM_OFFICIAL,
    rationale:
      "Official RLM runs the authors' implementation instead of Studio's RLM loop, so you can tell the RLM idea apart from Studio's implementation of it.",
  },
];

const SCENARIO_BY_ID: ReadonlyMap<string, Scenario> = new Map(
  SCENARIOS.map((s) => [s.id, s] as const),
);

export function getModeForScenario(scenarioId: string): LearnMode | undefined {
  return SCENARIO_BY_ID.get(scenarioId)?.recommendedMode;
}

export function ModeChooser({ className }: { className?: string }) {
  const [scenarioId, setScenarioId] = useState<string>(SCENARIOS[0].id);
  const scenario = SCENARIO_BY_ID.get(scenarioId);
  const recommendedMode = scenario?.recommendedMode;

  return (
    <div className={cn("rounded-lg border bg-card p-4 shadow-sm", className)}>
      <div className="flex flex-col gap-2">
        <label
          htmlFor="mode-chooser-scenario"
          className="text-sm font-medium"
        >
          What does your task look like?
        </label>
        <Select value={scenarioId} onValueChange={setScenarioId}>
          <SelectTrigger
            id="mode-chooser-scenario"
            aria-label="Pick a scenario"
            className="w-full"
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {SCENARIOS.map((s) => (
              <SelectItem key={s.id} value={s.id}>
                {s.prompt}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <ul
        aria-label="Mode options"
        className="mt-4 flex flex-col gap-2 text-sm"
      >
        {MODE_ROWS.map((row) => {
          const isActive = recommendedMode === row.mode;
          return (
            <li
              key={row.mode}
              data-mode={row.mode}
              data-active={isActive ? "true" : "false"}
              aria-current={isActive ? "true" : undefined}
              className={cn(
                "flex items-start gap-3 rounded-md border px-3 py-2 transition-colors",
                isActive
                  ? "border-primary bg-primary/5"
                  : "border-transparent",
              )}
            >
              <span
                aria-hidden="true"
                className={cn(
                  "mt-1 inline-block h-2 w-2 shrink-0 rounded-full",
                  row.dot,
                )}
              />
              <span>
                <span className="font-semibold">{displayModeName(row.mode)}</span> — {row.body}
              </span>
            </li>
          );
        })}
      </ul>

      {scenario && (
        <p
          aria-live="polite"
          className="mt-3 text-sm text-muted-foreground"
        >
          <span className="font-semibold text-foreground">
            Recommendation:{" "}
            {scenario.recommendedMode
              ? displayModeName(scenario.recommendedMode)
              : COMPARE_RECOMMENDATION}
            .
          </span>{" "}
          {scenario.rationale}
        </p>
      )}
    </div>
  );
}
