"use client";

import { PieChart, Pie, Cell, Tooltip, Legend } from "recharts";
import { ChartContainer } from "./chart-container";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useChartTooltipStyle } from "./use-chart-tooltip";
import { displayProviderName } from "@/lib/constants";
import { formatUsd } from "@/lib/format";

interface CostBreakdownProps {
  data: Record<string, { queries: number; total_tokens: number; total_cost_usd: number; avg_latency_seconds: number }>;
  title?: string;
}

const COLORS = ["#2563eb", "#7c3aed", "#059669", "#d97706", "#dc2626", "#6366f1"];

export function CostBreakdown({ data, title = "Cost by Provider" }: CostBreakdownProps) {
  const tooltipStyle = useChartTooltipStyle();
  const chartData = Object.entries(data).map(([name, values]) => ({
    name: displayProviderName(name),
    value: values.total_cost_usd,
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent>
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
                {chartData.map((_, index) => (
                  <Cell key={index} fill={COLORS[index % COLORS.length]} />
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
      </CardContent>
    </Card>
  );
}
