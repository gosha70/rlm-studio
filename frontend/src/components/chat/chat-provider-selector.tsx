"use client";

import { useState } from "react";
import { type ChatProviderConfig, type LLMProviderConfig } from "@/lib/api";
import { MODE_DIRECT, MODE_RAG, MODE_RLM, MODE_RLM_OFFICIAL } from "@/lib/constants";
import { cn } from "@/lib/utils";

interface ChatProviderSelectorProps {
  chatProviders: ChatProviderConfig[];
  llmProviders: LLMProviderConfig[];
  selectedIds: string[];
  onSelectionChange: (ids: string[]) => void;
  orderedIds?: string[];
  onOrderChange?: (orderedIds: string[]) => void;
  disabled?: boolean;
}

const MODE_BADGES: Record<string, string> = {
  [MODE_DIRECT]: "D",
  [MODE_RLM]: "R",
  [MODE_RAG]: "G",
  [MODE_RLM_OFFICIAL]: "O",
};

export function ChatProviderSelector({
  chatProviders,
  llmProviders,
  selectedIds,
  onSelectionChange,
  orderedIds,
  onOrderChange,
  disabled = false,
}: ChatProviderSelectorProps) {
  const [dragOverId, setDragOverId] = useState<string | null>(null);

  // Create a map of LLM provider IDs to their info for quick lookup
  const llmProviderMap = new Map(llmProviders.map((p) => [p.id, p]));

  // Filter chat providers to only show those with connected or configured underlying LLM providers
  const availableChatProviders = chatProviders.filter((cp) => {
    const lp = llmProviderMap.get(cp.llm_provider_id);
    return lp && (lp.status === "connected" || lp.status === "configured");
  });

  // Sort by orderedIds if provided
  const sortedProviders = orderedIds
    ? [...availableChatProviders].sort((a, b) => {
        const ai = orderedIds.indexOf(a.id);
        const bi = orderedIds.indexOf(b.id);
        // Items not in orderedIds go to end
        return (ai === -1 ? Infinity : ai) - (bi === -1 ? Infinity : bi);
      })
    : availableChatProviders;

  // Get status and badge for a provider
  const getProviderStatus = (llmProviderId: string) => {
    const lp = llmProviderMap.get(llmProviderId);
    if (!lp) return { status: "unavailable", color: "bg-gray-400" };
    if (lp.status === "connected") return { status: "connected", color: "bg-emerald-500" };
    if (lp.status === "configured") return { status: "configured", color: "bg-amber-500" };
    return { status: "unavailable", color: "bg-gray-400" };
  };

  const toggle = (id: string) => {
    if (disabled) return;

    if (selectedIds.includes(id)) {
      // Don't allow deselecting the last provider
      if (selectedIds.length === 1) return;
      onSelectionChange(selectedIds.filter((sid) => sid !== id));
    } else {
      onSelectionChange([...selectedIds, id]);
    }
  };

  const handleDrop = (targetId: string, e: React.DragEvent) => {
    e.preventDefault();
    setDragOverId(null);
    const fromId = e.dataTransfer.getData("text/plain");
    if (fromId && fromId !== targetId && onOrderChange) {
      const ids = sortedProviders.map((c) => c.id);
      const filtered = ids.filter((id) => id !== fromId);
      const targetIdx = filtered.indexOf(targetId);
      filtered.splice(targetIdx, 0, fromId);
      onOrderChange(filtered);
    }
  };

  if (availableChatProviders.length === 0) {
    return (
      <div
        className={cn(
          "flex items-center rounded-lg border bg-muted p-3 text-xs text-muted-foreground",
          disabled && "opacity-50",
        )}
      >
        No Chat Providers available. Configure one in Settings.
      </div>
    );
  }

  return (
    <div
      className={cn(
        "flex flex-wrap gap-2 rounded-lg border bg-muted p-2",
        disabled && "opacity-50",
      )}
      role="group"
      aria-label="Chat providers"
    >
      {sortedProviders.length > 1 && (
        <button
          type="button"
          disabled={disabled}
          onClick={() => {
            if (selectedIds.length === sortedProviders.length) {
              onSelectionChange([sortedProviders[0].id]);
            } else {
              onSelectionChange(sortedProviders.map((cp) => cp.id));
            }
          }}
          className={cn(
            "rounded-md px-2 py-1.5 text-xs font-medium transition-colors",
            "border border-dashed border-muted-foreground/30",
            "text-muted-foreground hover:text-foreground hover:bg-background/50",
            "disabled:cursor-not-allowed",
          )}
          aria-label={selectedIds.length === sortedProviders.length ? "Select first only" : "Select all providers"}
        >
          {selectedIds.length === sortedProviders.length ? "1" : "All"}
        </button>
      )}
      {sortedProviders.map((cp) => {
        const isSelected = selectedIds.includes(cp.id);
        const { status, color } = getProviderStatus(cp.llm_provider_id);
        const modeBadge = MODE_BADGES[cp.execution_mode] || "?";
        const isDragOver = dragOverId === cp.id;

        return (
          <button
            key={cp.id}
            role="checkbox"
            aria-checked={isSelected}
            aria-label={`${cp.name} (${status})`}
            disabled={disabled || (isSelected && selectedIds.length === 1)}
            onClick={() => toggle(cp.id)}
            draggable={!!onOrderChange && !disabled}
            onDragStart={(e) => e.dataTransfer.setData("text/plain", cp.id)}
            onDragOver={(e) => { e.preventDefault(); setDragOverId(cp.id); }}
            onDragLeave={() => setDragOverId(null)}
            onDrop={(e) => handleDrop(cp.id, e)}
            className={cn(
              "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
              "disabled:cursor-not-allowed",
              isSelected
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground hover:bg-background/50",
              isDragOver && "ring-2 ring-primary/50",
            )}
          >
            {/* Provider name + profile */}
            <span>
              {cp.name}
              {cp.profile_name && (
                <span className="ml-1 font-normal opacity-60">{cp.profile_name}</span>
              )}
            </span>

            {/* Mode badge */}
            <span className="rounded bg-muted px-1 py-0.5 text-xs font-semibold text-muted-foreground">
              {modeBadge}
            </span>

            {/* Status dot */}
            <span
              className={cn("inline-block h-2 w-2 rounded-full", color)}
              title={status}
              aria-label={`Provider status: ${status}`}
            />
          </button>
        );
      })}
    </div>
  );
}
