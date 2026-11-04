"use client";

import Link from "next/link";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts";
import { ChartContainer } from "./chart-container";
import { CheckCircle2, Lightbulb } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useChartTooltipStyle } from "./use-chart-tooltip";
import { providerColorMap } from "./series-colors";
import type { FailureMetricsResponse, FailureCategory } from "@/lib/api";

interface FailureBreakdownProps {
  data: FailureMetricsResponse;
  /** Provider name → colour, shared across the page's charts. */
  colors?: Record<string, string>;
}

// Exported so tests can assert on category coverage without reaching
// into the recharts DOM (which is mocked in the frontend test env).
export const CATEGORY_LABELS: Record<FailureCategory, string> = {
  timeout: "Timeout",
  prefill_timeout: "Prefill Timeout",
  budget_exhausted: "Budget Exhausted",
  context_overflow: "Context Overflow",
  general_error: "General Error",
};

export const ALL_CATEGORIES: FailureCategory[] = [
  "timeout",
  "prefill_timeout",
  "budget_exhausted",
  "context_overflow",
  "general_error",
];

export function FailureBreakdown({ data, colors }: FailureBreakdownProps) {
  const tooltipStyle = useChartTooltipStyle();
  const { total_runs, total_failures, failure_rate, by_provider, by_category } =
    data;

  // Build grouped chart data: one row per error category, one bar per provider.
  // Shape: [{ category: "Timeout", "vLLM-RLM": 1, "DIRECT-ANTHROPIC": 0, ... }, ...]
  const providers = Object.keys(by_provider);
  const colorOf = { ...providerColorMap(providers), ...colors };

  const chartData = ALL_CATEGORIES
    .map((cat) => {
      const row: Record<string, string | number> = {
        category: CATEGORY_LABELS[cat],
      };
      let hasAny = false;
      for (const provider of providers) {
        const entry = by_provider[provider]?.find((c) => c.category === cat);
        const count = entry?.count ?? 0;
        row[provider] = count;
        if (count > 0) hasAny = true;
      }
      return hasAny ? row : null;
    })
    .filter(Boolean) as Record<string, string | number>[];

  // Spec §8d: surface a remediation hint when prefill_timeout appears.
  // Links to the troubleshoot entry the classifier writer added
  // (`docs/troubleshoot.yaml::prefill-timeout-enable-caching`).
  const prefillTimeoutCount =
    by_category.find((c) => c.category === "prefill_timeout")?.count ?? 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">⚠️ Failure Analysis (by Chat Provider)</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {total_failures === 0 ? (
          <div className="flex items-center gap-2 py-6 text-sm text-green-600 dark:text-green-400">
            <CheckCircle2 className="h-5 w-5" />
            <span>No failures in this session</span>
          </div>
        ) : (
          <>
            <p className="text-sm text-muted-foreground">
              {total_failures} of {total_runs} runs failed ({(failure_rate * 100).toFixed(0)}%)
            </p>

            {chartData.length > 0 && (
              <ChartContainer className="h-64" role="img" aria-label="Grouped bar chart showing failures per category per provider">
                {(w, h) => (
                  <BarChart width={w} height={h} data={chartData} barCategoryGap="20%">
                    <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                    <XAxis
                      dataKey="category"
                      className="text-xs"
                      tick={{ fill: "var(--color-muted-foreground)" }}
                    />
                    <YAxis
                      allowDecimals={false}
                      className="text-xs"
                      tick={{ fill: "var(--color-muted-foreground)" }}
                    />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Legend />
                    {providers.map((provider) => (
                      <Bar
                        key={provider}
                        dataKey={provider}
                        name={provider}
                        fill={colorOf[provider]}
                        radius={[4, 4, 0, 0]}
                      />
                    ))}
                  </BarChart>
                )}
              </ChartContainer>
            )}

            {prefillTimeoutCount > 0 && (
              <aside
                role="note"
                aria-label="Prefill-timeout remediation hint"
                className="flex items-start gap-2 rounded-md border border-amber-300/60 bg-amber-50 p-3 text-sm dark:border-amber-900/60 dark:bg-amber-950/40"
              >
                <Lightbulb className="mt-0.5 h-4 w-4 flex-shrink-0 text-amber-600 dark:text-amber-400" />
                <div className="space-y-1">
                  <p className="font-medium text-amber-900 dark:text-amber-200">
                    Prefill-dominated timeouts detected ({prefillTimeoutCount})
                  </p>
                  <p className="text-xs text-amber-800/90 dark:text-amber-200/90">
                    Most of the loop&apos;s wall-time was spent re-prefilling a
                    growing prompt with little or no prefix-cache hit.
                    Enable prefix caching on the provider or shorten
                    history replay to recover throughput.
                  </p>
                  <Link
                    href="/learn/troubleshoot#prefill-timeout-enable-caching"
                    className="inline-block text-xs font-medium text-amber-900 underline hover:text-amber-950 dark:text-amber-200 dark:hover:text-amber-100"
                  >
                    Open troubleshoot entry →
                  </Link>
                </div>
              </aside>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
