"use client";

/**
 * Availability of third-party RLM engines (``GET /api/engines``).
 *
 * Shared by the Compare mode picker, the Chat Provider mode select and the
 * engine badges, under one SWR key so every consumer sees the same answer.
 * No focus revalidation: installing the ``interop`` extra means restarting
 * the server anyway.
 *
 * While the answer is unknown (loading, or the request failed) an engine is
 * treated as selectable — the server is the authority and rejects a run with
 * the same user-facing reason, so the UI never wedges an option shut over a
 * transient fetch error.
 */

import useSWR from "swr";
import { getEngines, type EngineStatus, type EnginesResponse } from "@/lib/api";
import { MODE_RLM_OFFICIAL } from "@/lib/constants";

export const ENGINES_SWR_KEY = "engines";

/** Reason a mode cannot be selected right now, or null when it can. */
export function unavailableReasonFor(
  mode: string,
  engines: EnginesResponse | undefined,
): string | null {
  if (mode !== MODE_RLM_OFFICIAL) return null;
  const status = engines?.rlm_official;
  if (!status || status.available) return null;
  return status.reason;
}

export interface EngineAvailability {
  /** The paper authors' `rlms` engine, or undefined while unknown. */
  rlmOfficial: EngineStatus | undefined;
  /** @see unavailableReasonFor */
  unavailableReasonFor: (mode: string) => string | null;
}

export function useEngineAvailability(): EngineAvailability {
  const { data } = useSWR<EnginesResponse>(ENGINES_SWR_KEY, getEngines, {
    revalidateOnFocus: false,
  });
  return {
    rlmOfficial: data?.rlm_official,
    unavailableReasonFor: (mode) => unavailableReasonFor(mode, data),
  };
}
