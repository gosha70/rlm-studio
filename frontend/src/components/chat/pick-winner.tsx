"use client";

import { Trophy } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface PickWinnerProps {
  sessionId: string;
  responses: { executionId: string; cpId: string; name: string }[];
  currentWinnerId?: string;
  onPick: (winnerExecutionId: string) => void;
}

export function PickWinner({
  responses,
  currentWinnerId,
  onPick,
}: PickWinnerProps) {
  if (responses.length < 2) return null;

  return (
    <div className="flex items-center gap-2">
      <Trophy className="h-3.5 w-3.5 text-muted-foreground" aria-hidden="true" />
      <span className="text-xs text-muted-foreground">Best:</span>
      {responses.map((r) => (
        <Button
          key={r.executionId}
          variant="outline"
          size="sm"
          className={cn(
            "h-6 px-2 text-xs",
            currentWinnerId === r.executionId &&
              "border-amber-500 bg-amber-50 text-amber-700 dark:bg-amber-950 dark:text-amber-400"
          )}
          onClick={() => onPick(r.executionId)}
          aria-pressed={currentWinnerId === r.executionId}
        >
          {r.name}
        </Button>
      ))}
    </div>
  );
}
