/**
 * Bounds for the numbers the Settings forms accept, and the checks that
 * refuse a value outside them with a reason.  The server enforces the same
 * bounds on its request models (src/rlmstudio/server/models.py).
 */

/**
 * Smallest context window (tokens) an LLM Provider may be given.  Below
 * this, the output cap derived from the window leaves no room to answer.
 */
export const CONTEXT_WINDOW_MIN_TOKENS = 1_024;

/**
 * The context window an LLM Provider update sends to remove a stored value
 * (the update route treats `null` as "no change").
 */
export const CONTEXT_WINDOW_CLEAR = 0;

/** RAG chunk size. */
export const RAG_CHUNK_SIZE_MIN = 1;
export const RAG_CHUNK_SIZE_MAX = 100_000;
/** Chunks retrieved per query. */
export const RAG_TOP_K_MIN = 1;
export const RAG_TOP_K_MAX = 100;

/** Longest per-call timeout a profile may set (1 hour). */
export const RUNTIME_MAX_TIMEOUT_SECONDS = 3_600;
/** Most RLM steps a profile may allow. */
export const BUDGET_MAX_STEPS = 500;
/** Longest total run time a profile may allow (24 hours). */
export const BUDGET_MAX_TIME_SECONDS = 86_400;
/** Duplicate inspect actions RLM tolerates before it finalizes. */
export const BUDGET_MAX_REPEAT_LIMIT = 20;
/** Token budget per query (Settings → Budget). */
export const BUDGET_MIN_TOKENS = 1;
export const BUDGET_MAX_TOKENS = 500_000;
/** Dollar limit per query, for Settings → Budget and profiles alike. */
export const BUDGET_MIN_COST_USD = 0;
export const BUDGET_MAX_COST_USD = 100;

const fmt = (n: number) => n.toLocaleString();

/**
 * Why `text` is not a whole number from `min` to `max`, or null when it is.
 * A blank field is refused rather than read as 0.
 */
export function wholeNumberProblem(
  label: string,
  text: string,
  min: number,
  max: number,
): string | null {
  const t = text.trim();
  if (t === "") return `Enter a value for ${label}.`;
  if (!/^-?\d+$/.test(t)) return `${label} must be a whole number.`;
  const n = Number(t);
  if (n < min || n > max) return `${label} must be between ${fmt(min)} and ${fmt(max)}.`;
  return null;
}

/** Why `text` is not a number from `min` to `max`, or null when it is. */
export function decimalProblem(
  label: string,
  text: string,
  min: number,
  max: number,
): string | null {
  const t = text.trim();
  if (t === "") return `Enter a value for ${label}.`;
  if (!/^-?\d+(\.\d+)?$/.test(t)) return `${label} must be a number.`;
  const n = Number(t);
  if (n < min || n > max) return `${label} must be between ${fmt(min)} and ${fmt(max)}.`;
  return null;
}
