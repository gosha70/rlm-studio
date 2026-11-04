"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { FieldLabel } from "./field-label";
import type { BudgetConfig as BudgetConfigType } from "@/lib/api";
import { RLM_DEFAULT_NUDGE_AT_FRACTION } from "@/lib/constants";

interface BudgetConfigProps {
  config: BudgetConfigType;
  onChange: (config: BudgetConfigType) => void;
}

const DEFAULTS: BudgetConfigType = {
  max_steps: 16,
  max_tokens: 50000,
  max_cost_usd: 2.0,
  max_time_seconds: 600,
  max_recursion_depth: 5,
  repeat_limit: 2,
  nudge_at_fraction: RLM_DEFAULT_NUDGE_AT_FRACTION,
};

/** Why `budget` cannot be saved, or null when it can. Mirrors the API bounds. */
function budgetError(budget: BudgetConfigType): string | null {
  if (!Number.isInteger(budget.max_tokens) || budget.max_tokens < 1) {
    return "Max Tokens must be a whole number of at least 1.";
  }
  if (!Number.isFinite(budget.max_cost_usd) || budget.max_cost_usd < 0) {
    return "Max Cost must be 0 or more.";
  }
  return null;
}

export function BudgetConfig({ config, onChange }: BudgetConfigProps) {
  const [local, setLocal] = useState(config);
  const [error, setError] = useState<string | null>(null);

  const update = <K extends keyof BudgetConfigType>(key: K, value: BudgetConfigType[K]) => {
    setError(null);
    setLocal((prev) => ({ ...prev, [key]: value }));
  };

  const handleSave = () => {
    const problem = budgetError(local);
    setError(problem);
    if (problem) return;
    onChange(local);
  };

  const handleReset = () => {
    setError(null);
    setLocal(DEFAULTS);
    onChange(DEFAULTS);
  };

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <FieldLabel htmlFor="max-steps" tooltip="Maximum number of LLM reasoning steps per query. Higher values allow deeper exploration in RLM mode but increase cost and latency.">Max Steps: {local.max_steps}</FieldLabel>
        <Slider
          id="max-steps"
          min={1}
          max={50}
          step={1}
          value={[local.max_steps]}
          onValueChange={([v]) => update("max_steps", v)}
          aria-label="Max steps"
          aria-valuetext={`${local.max_steps} steps`}
        />
      </div>

      <div className="space-y-2">
        <FieldLabel htmlFor="max-tokens" tooltip="Maximum total tokens (input + output) allowed per query execution. Prevents runaway cost on long RLM loops.">Max Tokens</FieldLabel>
        <Input
          id="max-tokens"
          type="number"
          min={1000}
          max={500000}
          value={local.max_tokens}
          onChange={(e) => update("max_tokens", Number(e.target.value))}
        />
      </div>

      <div className="space-y-2">
        <FieldLabel htmlFor="max-cost" tooltip="Dollar limit per query execution. The query is stopped once this cost is reached. Local models cost nothing, so keep it above 0.">Max Cost (USD)</FieldLabel>
        <Input
          id="max-cost"
          type="number"
          min={0.01}
          max={100}
          step={0.01}
          value={local.max_cost_usd}
          onChange={(e) => update("max_cost_usd", Number(e.target.value))}
        />
      </div>

      <div className="space-y-2">
        <FieldLabel htmlFor="max-time" tooltip="Wall-clock timeout per query. Includes LLM inference, code execution, and network latency. Increase for complex multi-step RLM queries.">Max Time (seconds): {local.max_time_seconds}</FieldLabel>
        <Slider
          id="max-time"
          min={10}
          max={3600}
          step={10}
          value={[local.max_time_seconds]}
          onValueChange={([v]) => update("max_time_seconds", v)}
          aria-label="Max time in seconds"
          aria-valuetext={`${local.max_time_seconds} seconds`}
        />
      </div>

      <div className="space-y-2">
        <FieldLabel htmlFor="max-depth" tooltip="Maximum depth of recursive sub-calls. Controls how many levels of nested reasoning the RLM agent can spawn.">Max Recursion Depth: {local.max_recursion_depth}</FieldLabel>
        <Slider
          id="max-depth"
          min={1}
          max={20}
          step={1}
          value={[local.max_recursion_depth]}
          onValueChange={([v]) => update("max_recursion_depth", v)}
          aria-label="Max recursion depth"
          aria-valuetext={`${local.max_recursion_depth} levels`}
        />
      </div>

      {error && (
        <p className="text-sm text-destructive" role="alert">
          {error} Not saved.
        </p>
      )}

      <div className="flex gap-2">
        <Button onClick={handleSave}>Save</Button>
        <Button variant="outline" onClick={handleReset}>
          Reset to Defaults
        </Button>
      </div>
    </div>
  );
}
