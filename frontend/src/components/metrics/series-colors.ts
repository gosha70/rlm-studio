/**
 * Chart colours shared by the Dashboard's charts.
 *
 * Modes have fixed colours. Chat Providers get theirs from a separate
 * palette with none of the mode hues, so a provider called "Qwen RAG" is
 * never drawn in another mode's colour, and one name→colour map built for
 * the whole page keeps a provider the same colour in every chart.
 */

import { MODE_DIRECT, MODE_RAG, MODE_RLM, MODE_RLM_OFFICIAL } from "@/lib/constants";

export const MODE_COLORS: Record<string, string> = {
  [MODE_DIRECT]: "#7c3aed", // violet
  [MODE_RLM]: "#2563eb", // blue
  [MODE_RAG]: "#10b981", // emerald
  [MODE_RLM_OFFICIAL]: "#f59e0b", // amber
};

/** Chat Provider colours: no violet, blue, emerald or amber (the mode hues). */
export const PROVIDER_PALETTE = [
  "#0891b2", // cyan
  "#db2777", // pink
  "#ea580c", // orange
  "#65a30d", // lime
  "#475569", // slate
  "#c026d3", // fuchsia
];

/**
 * One colour per name. Names are de-duplicated and sorted, so the same set
 * of names always gets the same colours whichever chart lists them first.
 */
export function providerColorMap(names: Iterable<string>): Record<string, string> {
  const sorted = [...new Set(names)].sort((a, b) => a.localeCompare(b));
  return Object.fromEntries(
    sorted.map((name, i) => [name, PROVIDER_PALETTE[i % PROVIDER_PALETTE.length]]),
  );
}
