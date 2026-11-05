"use client";

import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import type { TraceStep } from "@/lib/api";

interface TraceTreeProps {
  steps: TraceStep[];
  onSelect: (step: TraceStep) => void;
  selectedIndex?: number;
}

export function TraceTree({ steps, onSelect, selectedIndex }: TraceTreeProps) {
  return (
    <div className="space-y-1">
      {steps.map((step) => (
        <button
          key={step.index}
          type="button"
          className={cn(
            "flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm transition-colors hover:bg-muted/50",
            selectedIndex === step.index && "ring-2 ring-ring bg-muted/30",
          )}
          style={{ paddingLeft: `${(step.recursion_depth + 1) * 16}px` }}
          onClick={() => onSelect(step)}
          aria-label={`Step ${step.index + 1}: ${step.action_type} at depth ${step.recursion_depth}`}
        >
          <span className="text-xs text-muted-foreground">
            {step.recursion_depth > 0 ? "|-" : ""} {step.index + 1}.
          </span>
          <Badge variant="outline" className="text-xs">
            {step.action_type}
          </Badge>
          <span className="min-w-0 flex-1 truncate font-mono text-xs">
            {step.code ?? step.output ?? "--"}
          </span>
          <span className="text-xs text-muted-foreground">
            {step.duration_seconds.toFixed(1)}s
          </span>
        </button>
      ))}
    </div>
  );
}
