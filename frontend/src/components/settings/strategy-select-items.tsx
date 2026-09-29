"use client";

/**
 * The strategy options of a Run Profile — which is what decides a Chat
 * Provider's execution mode (the server resolves `execution_mode` from the
 * profile's `strategy`). Used by the create-profile form and the profile
 * card's edit form so both offer the same list.
 *
 * The official engine is a third-party package; when `GET /api/engines`
 * reports it unavailable the option stays visible but disabled, with the
 * server's reason as the tooltip, so users learn what to install instead of
 * wondering where the option went.
 */

import { SelectItem } from "@/components/ui/select";
import { useEngineAvailability } from "@/components/shared/use-engines";
import {
  displayModeName,
  MODE_DIRECT,
  MODE_RAG,
  MODE_RLM,
  MODE_RLM_OFFICIAL,
} from "@/lib/constants";

export const OFFICIAL_STRATEGY_UNAVAILABLE_SUFFIX = " — not installed";

export function StrategySelectItems() {
  const { unavailableReasonFor } = useEngineAvailability();
  const reason = unavailableReasonFor(MODE_RLM_OFFICIAL);
  const officialLabel = displayModeName(MODE_RLM_OFFICIAL);
  return (
    <>
      <SelectItem value={MODE_DIRECT}>{displayModeName(MODE_DIRECT)}</SelectItem>
      <SelectItem value={MODE_RLM}>{displayModeName(MODE_RLM)}</SelectItem>
      <SelectItem value={MODE_RAG}>{displayModeName(MODE_RAG)}</SelectItem>
      <SelectItem
        value={MODE_RLM_OFFICIAL}
        disabled={reason !== null}
        title={reason ?? undefined}
      >
        {reason === null ? officialLabel : officialLabel + OFFICIAL_STRATEGY_UNAVAILABLE_SUFFIX}
      </SelectItem>
    </>
  );
}
