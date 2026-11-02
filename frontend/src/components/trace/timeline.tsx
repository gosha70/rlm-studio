"use client";

import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import type { TraceStep } from "@/lib/api";

interface TimelineProps {
  steps: TraceStep[];
  onSelect: (step: TraceStep) => void;
  selectedIndex?: number;
}

const actionColors: Record<string, string> = {
  inspect: "bg-blue-500",
  subcall: "bg-violet-500",
  compaction: "bg-amber-500",
  final: "bg-emerald-500",
  error: "bg-red-500",
};

export function Timeline({ steps, onSelect, selectedIndex }: TimelineProps) {
  const maxDuration = Math.max(...steps.map((s) => s.duration_seconds), 0.1);

  return (
    <div className="space-y-2">
      {steps.map((step) => {
        const widthPercent = (step.duration_seconds / maxDuration) * 100;
        return (
          <button
            key={step.index}
            type="button"
            className={cn(
              "flex w-full items-center gap-3 rounded-md border px-3 py-2 text-left text-sm transition-colors hover:bg-muted/50",
              selectedIndex === step.index && "ring-2 ring-ring",
            )}
            onClick={() => onSelect(step)}
            aria-label={`Step ${step.index + 1}: ${step.action_type}, ${step.duration_seconds.toFixed(1)} seconds`}
          >
            <span className="w-16 shrink-0 text-xs text-foreground/70">
              Step {step.index + 1}
            </span>
            <Badge variant="outline" className="shrink-0 text-xs">
              {step.action_type}
            </Badge>
            <div className="flex-1">
              <div className="h-4 w-full rounded-full bg-muted">
                <div
                  className={cn("h-full rounded-full transition-all", actionColors[step.action_type] ?? "bg-primary")}
                  style={{ width: `${widthPercent}%` }}
                />
              </div>
            </div>
            <span className="w-12 shrink-0 text-right text-xs text-foreground/70">
              {step.duration_seconds.toFixed(1)}s
            </span>
            <span className="w-16 shrink-0 text-right text-xs text-foreground/70">
              {(step.input_tokens + step.output_tokens).toLocaleString()} tok
            </span>
          </button>
        );
      })}
    </div>
  );
}
