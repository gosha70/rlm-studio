/**
 * Bounds for the numbers the Settings forms accept, and the checks that
 * refuse a value outside them with a reason.  The server enforces the same
 * bounds on its request models (src/rlmstudio/server/models.py).
 */

/** RAG chunk size. */
export const RAG_CHUNK_SIZE_MIN = 1;
export const RAG_CHUNK_SIZE_MAX = 100_000;
/** Chunks retrieved per query. */
export const RAG_TOP_K_MIN = 1;
export const RAG_TOP_K_MAX = 100;

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
