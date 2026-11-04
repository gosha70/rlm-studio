/**
 * Display formatting for dollar costs in charts and tables.
 *
 * Local and embedding-only runs cost fractions of a cent, so a fixed
 * 4-decimal format reads "$0.0000" next to a visibly non-zero bar, and
 * Recharts' default axis ticks fall into scientific notation ("6e-7").
 */

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

/** A cost axis tick: short plain decimals ("$0", "$0.0000003", "$2.5"), never "6e-7". */
export function formatUsdTick(value: number): string {
  return usdSignificant(value, TICK_SIGNIFICANT_DIGITS);
}
