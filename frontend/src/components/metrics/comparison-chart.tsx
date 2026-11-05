"use client";

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts";
import { ChartContainer } from "./chart-container";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ModeSummary } from "@/lib/api";
import { useChartTooltipStyle } from "./use-chart-tooltip";
import { formatUsd, formatUsdTick } from "@/lib/format";
import { MODE_DIRECT, MODE_RAG, MODE_RLM, MODE_RLM_OFFICIAL } from "@/lib/constants";
import { MODE_COLORS } from "./series-colors";

interface ComparisonChartProps {
  rlmData: ModeSummary | undefined;
  directData: ModeSummary | undefined;
  ragData?: ModeSummary | undefined;
  /** Runs by the paper authors' engine (`rlm_official`). */
  officialData?: ModeSummary | undefined;
}

const OFFICIAL_SERIES = "Official RLM";

/**
 * Draw every Y tick. In these short charts Recharts' default
 * ("preserveEnd") dropped one tick, leaving uneven steps (0, 1000, 2000,
 * 4000).
 */
const ALL_TICKS = 0;

/** Room for sub-cent ticks such as "$0.00000015"; the 60px default clipped them. */
const COST_AXIS_WIDTH = 96;
const COST_TICK_FONT_SIZE = 12;

export function ComparisonChart({
  rlmData,
  directData,
  ragData,
  officialData,
}: ComparisonChartProps) {
  const tooltipStyle = useChartTooltipStyle();
  if (!rlmData && !directData && !ragData && !officialData) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Mode Comparison</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="flex h-48 items-center justify-center text-sm text-muted-foreground">
            Run queries in multiple modes to see comparison.
          </p>
        </CardContent>
      </Card>
    );
  }

  const tokenData = [
    {
      name: "Tokens",
      ...(directData ? { Direct: directData.total_tokens } : {}),
      ...(rlmData ? { RLM: rlmData.total_tokens } : {}),
      ...(ragData ? { RAG: ragData.total_tokens } : {}),
      ...(officialData ? { [OFFICIAL_SERIES]: officialData.total_tokens } : {}),
    },
  ];
  const costData = [
    {
      name: "Cost ($)",
      ...(directData ? { Direct: directData.total_cost_usd } : {}),
      ...(rlmData ? { RLM: rlmData.total_cost_usd } : {}),
      ...(ragData ? { RAG: ragData.total_cost_usd } : {}),
      ...(officialData ? { [OFFICIAL_SERIES]: officialData.total_cost_usd } : {}),
    },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Mode Comparison</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <ChartContainer className="h-48" role="img" aria-label="Bar chart comparing mode token usage">
            {(w, h) => (
              <BarChart width={w} height={h} data={tokenData} barCategoryGap="20%">
                <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                <XAxis dataKey="name" className="text-xs" tick={{ fill: "var(--color-muted-foreground)" }} />
                <YAxis className="text-xs" tick={{ fill: "var(--color-muted-foreground)" }} interval={ALL_TICKS} />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                {directData && <Bar dataKey="Direct" fill={MODE_COLORS[MODE_DIRECT]} radius={[4, 4, 0, 0]} />}
                {rlmData && <Bar dataKey="RLM" fill={MODE_COLORS[MODE_RLM]} radius={[4, 4, 0, 0]} />}
                {ragData && <Bar dataKey="RAG" fill={MODE_COLORS[MODE_RAG]} radius={[4, 4, 0, 0]} />}
                {officialData && (
                  <Bar dataKey={OFFICIAL_SERIES} fill={MODE_COLORS[MODE_RLM_OFFICIAL]} radius={[4, 4, 0, 0]} />
                )}
              </BarChart>
            )}
          </ChartContainer>
          <ChartContainer className="h-48" role="img" aria-label="Bar chart comparing mode cost">
            {(w, h) => (
              <BarChart width={w} height={h} data={costData} barCategoryGap="20%">
                <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                <XAxis dataKey="name" className="text-xs" tick={{ fill: "var(--color-muted-foreground)" }} />
                <YAxis
                  className="text-xs"
                  tick={{ fill: "var(--color-muted-foreground)", fontSize: COST_TICK_FONT_SIZE }}
                  interval={ALL_TICKS}
                  width={COST_AXIS_WIDTH}
                  tickFormatter={(v) => formatUsdTick(Number(v))}
                />
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => formatUsd(Number(v))} />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                {directData && <Bar dataKey="Direct" fill={MODE_COLORS[MODE_DIRECT]} radius={[4, 4, 0, 0]} />}
                {rlmData && <Bar dataKey="RLM" fill={MODE_COLORS[MODE_RLM]} radius={[4, 4, 0, 0]} />}
                {ragData && <Bar dataKey="RAG" fill={MODE_COLORS[MODE_RAG]} radius={[4, 4, 0, 0]} />}
                {officialData && (
                  <Bar dataKey={OFFICIAL_SERIES} fill={MODE_COLORS[MODE_RLM_OFFICIAL]} radius={[4, 4, 0, 0]} />
                )}
              </BarChart>
            )}
          </ChartContainer>
        </div>
      </CardContent>
    </Card>
  );
}
