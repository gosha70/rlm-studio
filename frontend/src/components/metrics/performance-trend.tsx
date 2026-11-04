"use client";

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts";
import { ChartContainer } from "./chart-container";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { TimelineEntry } from "@/lib/api";
import { useChartTooltipStyle } from "./use-chart-tooltip";
import { providerColorMap } from "./series-colors";

interface PerformanceTrendProps {
  timeline: TimelineEntry[];
  /** Provider name → colour, shared across the page's charts. */
  colors?: Record<string, string>;
}

export function PerformanceTrend({ timeline, colors }: PerformanceTrendProps) {
  const tooltipStyle = useChartTooltipStyle();

  // Group timeline entries by chat provider, then build one data point
  // per query index with separate keys per provider for latency.
  const providers = [...new Set(
    timeline.map((e) => e.chat_provider_name || e.provider || "unknown"),
  )];
  const colorOf = { ...providerColorMap(providers), ...colors };

  // Build per-provider sequences: each provider gets its own query counter
  const providerCounters: Record<string, number> = {};
  const dataPoints: Record<string, string | number>[] = [];

  for (const entry of timeline) {
    const name = entry.chat_provider_name || entry.provider || "unknown";
    providerCounters[name] = (providerCounters[name] || 0) + 1;
    const idx = providerCounters[name];

    // Find or create data point for this query index
    let point = dataPoints.find((p) => p._index === idx);
    if (!point) {
      point = { _index: idx };
      dataPoints.push(point);
    }
    point[`${name} (latency)`] = entry.latency_seconds;
    point[`${name} (tokens)`] = entry.tokens;
  }

  // Sort by index
  dataPoints.sort((a, b) => (a._index as number) - (b._index as number));

  if (dataPoints.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Performance Over Time</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="flex h-48 items-center justify-center text-sm text-muted-foreground">
            No data yet.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Performance Over Time (per Chat Provider)</CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer className="h-64" role="img" aria-label="Line chart showing latency per provider over queries">
          {(w, h) => (
            <LineChart width={w} height={h} data={dataPoints}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
              <XAxis
                dataKey="_index"
                label={{ value: "Query #", position: "insideBottomRight", offset: -5 }}
                className="text-xs"
                tick={{ fill: "var(--color-muted-foreground)" }}
              />
              <YAxis
                yAxisId="latency"
                className="text-xs"
                tick={{ fill: "var(--color-muted-foreground)" }}
                label={{ value: "Latency (s)", angle: -90, position: "insideLeft", style: { fill: "var(--color-muted-foreground)", fontSize: 11 } }}
              />
              <YAxis
                yAxisId="tokens"
                orientation="right"
                className="text-xs"
                tick={{ fill: "var(--color-muted-foreground)" }}
                label={{ value: "Tokens", angle: 90, position: "insideRight", style: { fill: "var(--color-muted-foreground)", fontSize: 11 } }}
              />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend wrapperStyle={{ fontSize: "12px" }} />
              {providers.map((name) => (
                <Line
                  key={`${name}-latency`}
                  yAxisId="latency"
                  type="monotone"
                  dataKey={`${name} (latency)`}
                  stroke={colorOf[name]}
                  strokeWidth={2}
                  dot={timeline.length < 30}
                  connectNulls
                  name={`${name} (latency)`}
                />
              ))}
              {providers.map((name) => (
                <Line
                  key={`${name}-tokens`}
                  yAxisId="tokens"
                  type="monotone"
                  dataKey={`${name} (tokens)`}
                  stroke={colorOf[name]}
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={false}
                  connectNulls
                  name={`${name} (tokens)`}
                />
              ))}
            </LineChart>
          )}
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
