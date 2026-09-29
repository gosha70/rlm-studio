"use client";

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts";
import { ChartContainer } from "./chart-container";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ModeSummary } from "@/lib/api";
import { useChartTooltipStyle } from "./use-chart-tooltip";

interface ComparisonChartProps {
  rlmData: ModeSummary | undefined;
  directData: ModeSummary | undefined;
  ragData?: ModeSummary | undefined;
  /** Runs by the paper authors' engine (`rlm_official`). */
  officialData?: ModeSummary | undefined;
}

const OFFICIAL_SERIES = "Official RLM";

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
                <YAxis className="text-xs" tick={{ fill: "var(--color-muted-foreground)" }} />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                {directData && <Bar dataKey="Direct" fill="#7c3aed" radius={[4, 4, 0, 0]} />}
                {rlmData && <Bar dataKey="RLM" fill="#2563eb" radius={[4, 4, 0, 0]} />}
                {ragData && <Bar dataKey="RAG" fill="#10b981" radius={[4, 4, 0, 0]} />}
                {officialData && (
                  <Bar dataKey={OFFICIAL_SERIES} fill="#f59e0b" radius={[4, 4, 0, 0]} />
                )}
              </BarChart>
            )}
          </ChartContainer>
          <ChartContainer className="h-48" role="img" aria-label="Bar chart comparing mode cost">
            {(w, h) => (
              <BarChart width={w} height={h} data={costData} barCategoryGap="20%">
                <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                <XAxis dataKey="name" className="text-xs" tick={{ fill: "var(--color-muted-foreground)" }} />
                <YAxis className="text-xs" tick={{ fill: "var(--color-muted-foreground)" }} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => `$${Number(v).toFixed(4)}`} />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                {directData && <Bar dataKey="Direct" fill="#7c3aed" radius={[4, 4, 0, 0]} />}
                {rlmData && <Bar dataKey="RLM" fill="#2563eb" radius={[4, 4, 0, 0]} />}
                {ragData && <Bar dataKey="RAG" fill="#10b981" radius={[4, 4, 0, 0]} />}
                {officialData && (
                  <Bar dataKey={OFFICIAL_SERIES} fill="#f59e0b" radius={[4, 4, 0, 0]} />
                )}
              </BarChart>
            )}
          </ChartContainer>
        </div>
      </CardContent>
    </Card>
  );
}
