"use client";

import { PieChart, Pie, Cell, Tooltip, Legend } from "recharts";
import { ChartContainer } from "./chart-container";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useChartTooltipStyle } from "./use-chart-tooltip";
import { displayProviderName } from "@/lib/constants";
import { formatUsd } from "@/lib/format";
import { providerColorMap } from "./series-colors";

interface CostBreakdownProps {
  data: Record<string, { queries: number; total_tokens: number; total_cost_usd: number; avg_latency_seconds: number }>;
  title?: string;
  /** Provider name → colour, shared across the page's charts. */
  colors?: Record<string, string>;
}

export const NO_COST_TEXT =
  "No cost recorded in this session. Local models, and models without a known price, report $0.";

export function CostBreakdown({ data, title = "Cost by Provider", colors }: CostBreakdownProps) {
  const tooltipStyle = useChartTooltipStyle();
  const chartData = Object.entries(data).map(([name, values]) => ({
    name: displayProviderName(name),
    value: values.total_cost_usd,
  }));
  const colorOf = { ...providerColorMap(chartData.map((d) => d.name)), ...colors };

  // A pie of zeros draws nothing, which left the card blank.
  const hasCost = chartData.some((d) => d.value > 0);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        {!hasCost ? (
          <p className="flex h-64 items-center justify-center px-4 text-center text-sm text-muted-foreground">
            {NO_COST_TEXT}
          </p>
        ) : (
          <ChartContainer className="h-64" role="img" aria-label="Pie chart showing cost breakdown by provider">
            {(w, h) => (
              <PieChart width={w} height={h}>
                <Pie
                  data={chartData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={80}
                  dataKey="value"
                >
                  {chartData.map((entry) => (
                    <Cell key={entry.name} fill={colorOf[entry.name]} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value) => formatUsd(Number(value))}
                  contentStyle={tooltipStyle}
                />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
              </PieChart>
            )}
          </ChartContainer>
        )}
      </CardContent>
    </Card>
  );
}
