/**
 * Shared constants — single source of truth for mode identifiers,
 * backend keys, and related groupings.  Mirrors the Python-side
 * constants in `src/rlmstudio/application/sandbox_vars.py`.
 */

// ---------------------------------------------------------------------------
// Execution modes
// ---------------------------------------------------------------------------

export const MODE_DIRECT = "direct" as const;
export const MODE_RLM = "rlm" as const;
export const MODE_RAG = "rag" as const;
export const MODE_COMPARE = "compare" as const;
export const MODE_AUTO = "auto" as const;
/** The paper authors' reference implementation (`rlms`) run as an engine. */
export const MODE_RLM_OFFICIAL = "rlm_official" as const;

/** All execution modes the user can select in the UI. */
export const ALL_EXECUTION_MODES = [MODE_DIRECT, MODE_RLM, MODE_RAG, MODE_RLM_OFFICIAL] as const;

/** Default mode for new profiles and Chat Providers. */
export const DEFAULT_MODE = MODE_DIRECT;

/** Default enabled modes for ModeConfig. */
export const DEFAULT_ENABLED_MODES = [MODE_DIRECT, MODE_RLM] as const;

/**
 * Human-readable names for execution modes — the single source of truth for
 * how a mode is spelled in the UI.  Raw identifiers must never be rendered:
 * `rlm_official` would read as "Rlm_official" under a `capitalize` class, and
 * the acronyms need casing no CSS transform can produce.
 */
export const MODE_DISPLAY_NAMES: Record<string, string> = {
  [MODE_DIRECT]: "Direct",
  [MODE_RLM]: "RLM",
  [MODE_RAG]: "RAG",
  [MODE_RLM_OFFICIAL]: "Official RLM (rlms)",
};

// ---------------------------------------------------------------------------
// LLM backend identifiers
// ---------------------------------------------------------------------------

export const BACKEND_OPENAI = "openai" as const;
export const BACKEND_ANTHROPIC = "anthropic" as const;
export const BACKEND_OLLAMA = "ollama" as const;
export const BACKEND_LMSTUDIO = "lmstudio" as const;
export const BACKEND_VLLM = "vllm" as const;

/** Backends that require an API key (cloud providers). */
export const CLOUD_BACKENDS = [BACKEND_OPENAI, BACKEND_ANTHROPIC] as const;

/** Backends that connect to a local endpoint. */
export const LOCAL_BACKENDS = [BACKEND_OLLAMA, BACKEND_LMSTUDIO, BACKEND_VLLM] as const;

/** All available backends for the LLM Provider selector. */
export const ALL_BACKENDS = [...CLOUD_BACKENDS, ...LOCAL_BACKENDS] as const;

/** Human-readable display names for backend keys. */
export const BACKEND_DISPLAY_NAMES: Record<string, string> = {
  [BACKEND_OPENAI]: "OpenAI",
  [BACKEND_ANTHROPIC]: "Anthropic",
  [BACKEND_OLLAMA]: "Ollama",
  [BACKEND_LMSTUDIO]: "LM Studio",
  [BACKEND_VLLM]: "vLLM",
};

/**
 * Model placeholder text per backend.
 *
 * Deliberately free of concrete model IDs: an example here is a hardcoded
 * value with no owner, and the last set named models the vendors had already
 * retired.  The live model list (fetched from the provider) is the source of
 * truth; this text only explains what to type when discovery is unavailable.
 */
export const MODEL_PLACEHOLDER: Record<string, string> = {
  [BACKEND_OPENAI]: "Model ID from your account",
  [BACKEND_ANTHROPIC]: "Model ID from your account",
  [BACKEND_VLLM]: "Model ID served by your endpoint",
  [BACKEND_OLLAMA]: "Model name listed by ollama list",
  [BACKEND_LMSTUDIO]: "Model ID served by your endpoint",
};

/** Endpoint placeholder text per local backend. */
export const ENDPOINT_PLACEHOLDER: Record<string, string> = {
  [BACKEND_OLLAMA]: "http://localhost:11434",
  [BACKEND_VLLM]: "http://<dgx-spark-ip>:8000/v1",
  [BACKEND_LMSTUDIO]: "http://localhost:1234/v1",
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Map a raw backend key to a human-readable name. */
export function displayProviderName(key: string): string {
  return BACKEND_DISPLAY_NAMES[key] ?? key;
}

/**
 * Map a raw mode identifier to its human-readable name.  Unknown modes fall
 * back to the identifier so a server that grows a mode ahead of the UI still
 * renders something.
 */
export function displayModeName(key: string): string {
  return MODE_DISPLAY_NAMES[key] ?? key;
}

// ---------------------------------------------------------------------------
// Prompt refinement (#78) — mirrors src/rlmstudio/application/refinement_constants.py
// ---------------------------------------------------------------------------

export const REFINEMENT_STATUS_PROPOSAL_READY = "proposal_ready" as const;
export const REFINEMENT_STATUS_EVALUATING = "evaluating" as const;
export const REFINEMENT_STATUS_EVALUATED = "evaluated" as const;
export const REFINEMENT_STATUS_INTERRUPTED = "interrupted" as const;
export const REFINEMENT_STATUS_FAILED = "failed" as const;
export const REFINEMENT_STATUS_ACCEPTED = "accepted" as const;
export const REFINEMENT_STATUS_REJECTED = "rejected" as const;
export const REFINEMENT_STATUS_ROLLED_BACK = "rolled_back" as const;

export type RefinementStatus =
  | typeof REFINEMENT_STATUS_PROPOSAL_READY
  | typeof REFINEMENT_STATUS_EVALUATING
  | typeof REFINEMENT_STATUS_EVALUATED
  | typeof REFINEMENT_STATUS_INTERRUPTED
  | typeof REFINEMENT_STATUS_FAILED
  | typeof REFINEMENT_STATUS_ACCEPTED
  | typeof REFINEMENT_STATUS_REJECTED
  | typeof REFINEMENT_STATUS_ROLLED_BACK;

/** Statuses a reject is allowed from (server: REJECTABLE_STATUSES). */
export const REFINEMENT_REJECTABLE_STATUSES: readonly RefinementStatus[] = [
  REFINEMENT_STATUS_PROPOSAL_READY,
  REFINEMENT_STATUS_EVALUATED,
  REFINEMENT_STATUS_INTERRUPTED,
  REFINEMENT_STATUS_FAILED,
];

export const REFINEMENT_STATUS_LABELS: Record<RefinementStatus, string> = {
  [REFINEMENT_STATUS_PROPOSAL_READY]: "Proposal ready",
  [REFINEMENT_STATUS_EVALUATING]: "Evaluating",
  [REFINEMENT_STATUS_EVALUATED]: "Evaluated",
  [REFINEMENT_STATUS_INTERRUPTED]: "Interrupted",
  [REFINEMENT_STATUS_FAILED]: "Failed",
  [REFINEMENT_STATUS_ACCEPTED]: "Accepted",
  [REFINEMENT_STATUS_REJECTED]: "Rejected",
  [REFINEMENT_STATUS_ROLLED_BACK]: "Rolled back",
};

/** Repetitions per case an evaluation may run (server: REFINEMENT_MIN/MAX_REPS). */
export const REFINEMENT_MIN_REPS = 2;
export const REFINEMENT_MAX_REPS = 5;

/** Longest accept/reject reason (server: REFINEMENT_DECISION_REASON_MAX_CHARS). */
export const REFINEMENT_DECISION_REASON_MAX_CHARS = 2_000;

/** Polling interval while an evaluation is running. */
export const REFINEMENT_POLL_MS = 2_000;

/** How often the header re-checks the backend while it is reachable. */
export const HEALTH_POLL_MS = 10_000;
/** How often it retries while the backend is unreachable, so it recovers on its own. */
export const HEALTH_RETRY_MS = 5_000;

/** SWR key for the sidebar session list; revalidate it when a session is created. */
export const SESSIONS_SWR_KEY = "sessions";
