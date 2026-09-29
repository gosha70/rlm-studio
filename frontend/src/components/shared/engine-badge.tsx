"use client";

/**
 * "official rlms x.y.z" badge for runs executed by the paper authors'
 * implementation. Renders nothing for Studio's own modes, so callers can
 * drop it next to any mode label without a condition.
 *
 * The version shown is the one installed in the running server (from
 * ``GET /api/engines``); per-run metadata is not part of the trace API.
 */

import { Badge } from "@/components/ui/badge";
import { MODE_RLM_OFFICIAL } from "@/lib/constants";
import { useEngineAvailability } from "./use-engines";

export const ENGINE_BADGE_TITLE = "Run by the paper authors' implementation (rlms)";

export function EngineBadge({ mode }: { mode: string }) {
  const { rlmOfficial } = useEngineAvailability();
  if (mode !== MODE_RLM_OFFICIAL) return null;
  const version = rlmOfficial?.version;
  return (
    <Badge variant="outline" className="text-xs" title={ENGINE_BADGE_TITLE}>
      official rlms{version ? ` ${version}` : ""}
    </Badge>
  );
}
