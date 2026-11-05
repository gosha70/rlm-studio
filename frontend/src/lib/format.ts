/**
 * Display formatting for dollar costs in charts and tables.
 *
 * Local and embedding-only runs cost fractions of a cent, so a fixed
 * 4-decimal format reads "$0.0000" next to a visibly non-zero bar, and
 * Recharts' default axis ticks fall into scientific notation ("6e-7").
 */

/**
 * Shown in place of a cost the API marks unknown (`cost_known: false`):
 * an unpriced run's $0 is not free.
 */
export const COST_UNKNOWN_TEXT = "unknown";
/** Tooltip for a cost shown as {@link COST_UNKNOWN_TEXT}. */
export const COST_UNKNOWN_HINT = "Cost unknown: no price is known for this model, so its $0 is not free.";

/** Below this, four decimals would round a real cost to "$0.0000". */
const SMALL_COST_USD = 0.0001;
const COST_DECIMALS = 4;
const SMALL_COST_SIGNIFICANT_DIGITS = 2;
const TICK_SIGNIFICANT_DIGITS = 3;

function usdSignificant(value: number, digits: number): string {
  return value.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumSignificantDigits: digits,
  });
}

/** A cost for tooltips and tables: "$0.0360", or "$0.0000006" for tiny costs. */
export function formatUsd(value: number): string {
  if (value !== 0 && Math.abs(value) < SMALL_COST_USD) {
    return usdSignificant(value, SMALL_COST_SIGNIFICANT_DIGITS);
  }
  return `$${value.toFixed(COST_DECIMALS)}`;
}

/** Shown in place of a token count the API marks unknown (`tokens_known: false`). */
export const TOKENS_UNKNOWN_TEXT = "unknown";
/** Tooltip for a token count shown as {@link TOKENS_UNKNOWN_TEXT}. */
export const TOKENS_UNKNOWN_HINT = "Token usage unknown: the engine reported none for this run.";

/** A run's cost, or {@link COST_UNKNOWN_TEXT} when the API says it is unknown. */
export function formatCostOrUnknown(value: number, known?: boolean): string {
  return known === false ? COST_UNKNOWN_TEXT : formatUsd(value);
}

/** A run's token count, or {@link TOKENS_UNKNOWN_TEXT} when the API says it is unknown. */
export function formatTokensOrUnknown(value: number, known?: boolean): string {
  return known === false ? TOKENS_UNKNOWN_TEXT : value.toLocaleString();
}

/** A cost axis tick: short plain decimals ("$0", "$0.0000003", "$2.5"), never "6e-7". */
export function formatUsdTick(value: number): string {
  return usdSignificant(value, TICK_SIGNIFICANT_DIGITS);
}
