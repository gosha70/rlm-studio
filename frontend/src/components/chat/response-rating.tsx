"use client";

import { ThumbsUp, ThumbsDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ResponseRatingProps {
  executionId: string;
  sessionId: string;
  chatProviderId: string;
  currentRating?: "up" | "down";
  onRate: (executionId: string, rating: "up" | "down" | null) => void;
}

export function ResponseRating({
  executionId,
  currentRating,
  onRate,
}: ResponseRatingProps) {
  const handleClick = (rating: "up" | "down") => {
    if (currentRating === rating) {
      onRate(executionId, null); // toggle off
    } else {
      onRate(executionId, rating);
    }
  };

  return (
    <div className="flex items-center gap-1">
      <Button
        variant="ghost"
        size="icon"
        className={cn(
          "h-6 w-6",
          currentRating === "up" && "text-emerald-600 dark:text-emerald-400"
        )}
        onClick={() => handleClick("up")}
        aria-label="Thumbs up"
        aria-pressed={currentRating === "up"}
      >
        <ThumbsUp className="h-3.5 w-3.5" />
      </Button>
      <Button
        variant="ghost"
        size="icon"
        className={cn(
          "h-6 w-6",
          currentRating === "down" && "text-red-600 dark:text-red-400"
        )}
        onClick={() => handleClick("down")}
        aria-label="Thumbs down"
        aria-pressed={currentRating === "down"}
      >
        <ThumbsDown className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
}
