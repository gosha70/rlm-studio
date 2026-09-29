"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import useSWR, { useSWRConfig } from "swr";
import { Hash, DollarSign, Clock, Activity, Star, ThumbsUp, ThumbsDown, Trophy, Trash2 } from "lucide-react";
import { AppShell } from "@/components/shared/app-shell";
import { MetricCard } from "@/components/metrics/metric-card";
import { ComparisonChart } from "@/components/metrics/comparison-chart";
import { CostBreakdown } from "@/components/metrics/cost-breakdown";
import { ProviderPerformance } from "@/components/metrics/provider-performance";
import { PerformanceTrend } from "@/components/metrics/performance-trend";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { getSessions, getMetrics, getFailureMetrics, getEvaluationSummary, getSessionEvaluations, resetSessionEvaluations, type SessionSummary, type MetricsResponse, type FailureMetricsResponse, type EvaluationSummaryResponse, type SessionEvaluations } from "@/lib/api";
import { JudgeDimensionsChart } from "@/components/metrics/judge-dimensions-chart";
import { FailureBreakdown } from "@/components/metrics/failure-breakdown";

export default function DashboardPage() {
  const router = useRouter();
  const { mutate } = useSWRConfig();
  const [selectedSession, setSelectedSession] = useState<string>("");
  const [isResetting, setIsResetting] = useState(false);

  const { data: sessions = [] } = useSWR<SessionSummary[]>("sessions", () => getSessions());
  // Auto-select the most recent session when none is chosen
  const effectiveSession = selectedSession || (sessions.length > 0 ? sessions[0].id : "");

  const { data: metrics, isLoading } = useSWR<MetricsResponse>(
    effectiveSession ? `metrics-${effectiveSession}` : null,
    () => getMetrics(effectiveSession),
  );

  const { data: failureMetrics } = useSWR<FailureMetricsResponse>(
    effectiveSession ? `failure-metrics-${effectiveSession}` : null,
    () => getFailureMetrics(effectiveSession),
  );

  const { data: evalSummary } = useSWR<EvaluationSummaryResponse>(
    effectiveSession ? `eval-summary-${effectiveSession}` : null,
    () => getEvaluationSummary(effectiveSession),
  );

  const { data: evalDetail } = useSWR<SessionEvaluations>(
    effectiveSession ? `eval-detail-${effectiveSession}` : null,
    () => getSessionEvaluations(effectiveSession),
  );

  const handleResetEvaluations = useCallback(async () => {
    if (!effectiveSession) return;
    setIsResetting(true);
    try {
      await resetSessionEvaluations(effectiveSession);
      await mutate(`eval-summary-${effectiveSession}`);
      await mutate(`eval-detail-${effectiveSession}`);
    } finally {
      setIsResetting(false);
    }
  }, [effectiveSession, mutate]);

  const providerNames = Object.fromEntries(
    Object.values(evalSummary?.by_chat_provider ?? {}).map((p) => [p.chat_provider_id, p.chat_provider_name]),
  );

  const summary = metrics?.summary;

  return (
    <AppShell>
      <div className="mx-auto max-w-[1200px] space-y-6 p-6">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-semibold">Dashboard</h2>
          <Select value={effectiveSession} onValueChange={setSelectedSession}>
            <SelectTrigger className="w-56" aria-label="Select session">
              <SelectValue placeholder="Select session..." />
            </SelectTrigger>
            <SelectContent>
              {sessions.map((s) => (
                <SelectItem key={s.id} value={s.id}>
                  {s.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Summary cards */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            label="Total Tokens"
            value={summary ? summary.total_tokens.toLocaleString() : "0"}
            icon={Hash}
          />
          <MetricCard
            label="Total Cost"
            value={summary ? `$${summary.total_cost_usd.toFixed(2)}` : "$0.00"}
            icon={DollarSign}
          />
          <MetricCard
            label="Avg Latency"
            value={summary ? summary.avg_latency_seconds.toFixed(1) : "0.0"}
            unit="s"
            icon={Clock}
          />
          <MetricCard
            label="Token Savings"
            value={summary?.avg_token_savings_percent != null ? `${summary.avg_token_savings_percent.toFixed(0)}%` : "N/A"}
            icon={Activity}
          />
        </div>

        {isLoading && (
          <div className="grid grid-cols-2 gap-4">
            <Skeleton className="h-72" />
            <Skeleton className="h-72" />
          </div>
        )}

        {/* Charts row */}
        {metrics && (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <ComparisonChart
              rlmData={metrics.by_mode.rlm}
              directData={metrics.by_mode.direct}
              ragData={metrics.by_mode.rag}
              officialData={metrics.by_mode.rlm_official}
            />
            <CostBreakdown
              data={metrics.by_chat_provider && Object.keys(metrics.by_chat_provider).length > 0
                ? metrics.by_chat_provider
                : metrics.by_provider}
              title="Cost by Chat Provider"
            />
          </div>
        )}

        {/* Provider performance — prefer Chat Provider names when available */}
        {metrics && metrics.by_chat_provider && Object.keys(metrics.by_chat_provider).length > 0 ? (
          <ProviderPerformance data={metrics.by_chat_provider} title="Chat Provider Performance" />
        ) : metrics && Object.keys(metrics.by_provider).length > 0 ? (
          <ProviderPerformance data={metrics.by_provider} />
        ) : null}

        {/* Performance trend */}
        {metrics && metrics.timeline.length > 0 && (
          <PerformanceTrend timeline={metrics.timeline} />
        )}

        {/* Failure breakdown */}
        {failureMetrics && (
          <FailureBreakdown data={failureMetrics} />
        )}

        {/* Recent executions table */}
        {metrics && metrics.timeline.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Recent Executions</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>#</TableHead>
                    <TableHead>Mode</TableHead>
                    <TableHead>Chat Provider</TableHead>
                    <TableHead className="text-right">Tokens</TableHead>
                    <TableHead className="text-right">Cost</TableHead>
                    <TableHead className="text-right">Latency</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {metrics.timeline.map((entry, i) => (
                    <TableRow
                      key={i}
                      className={entry.execution_id ? "cursor-pointer hover:bg-muted/50" : undefined}
                      onClick={entry.execution_id ? () => router.push(`/traces?exec=${entry.execution_id}`) : undefined}
                    >
                      <TableCell>{i + 1}</TableCell>
                      <TableCell>
                        <Badge variant="outline">{entry.mode.toUpperCase()}</Badge>
                      </TableCell>
                      <TableCell className="text-muted-foreground text-sm">
                        {entry.chat_provider_name || entry.provider || "—"}
                      </TableCell>
                      <TableCell className="text-right">
                        {entry.tokens.toLocaleString()}
                      </TableCell>
                      <TableCell className="text-right">
                        ${entry.cost_usd.toFixed(4)}
                      </TableCell>
                      <TableCell className="text-right">
                        {entry.latency_seconds.toFixed(1)}s
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        )}

        {/* Judge Dimensions Chart */}
        {evalDetail && evalDetail.judge_scores.length > 0 && (
          <JudgeDimensionsChart
            judgeScores={evalDetail.judge_scores}
            providerNames={providerNames}
          />
        )}

        {/* Judge Quality Scores */}
        {evalSummary && Object.keys(evalSummary.by_chat_provider).length > 0 && (
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="flex items-center gap-2 text-base">
                <Star className="h-4 w-4" />
                Judge Quality Scores
              </CardTitle>
              <button
                onClick={handleResetEvaluations}
                disabled={isResetting}
                className="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs text-muted-foreground hover:bg-destructive/10 hover:text-destructive disabled:opacity-50"
                title="Reset all evaluations for this session"
              >
                <Trash2 className="h-3.5 w-3.5" />
                {isResetting ? "Resetting…" : "Reset"}
              </button>
            </CardHeader>
            <CardContent className="space-y-4">
              {evalSummary.recommendation && (
                <div className="flex items-center gap-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm dark:border-amber-800 dark:bg-amber-950">
                  <Trophy className="h-4 w-4 text-amber-500" />
                  <span className="font-medium">Recommendation:</span>
                  <span className="text-muted-foreground">{evalSummary.recommendation_reason}</span>
                </div>
              )}
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Provider</TableHead>
                    <TableHead className="text-right">Judge Score</TableHead>
                    <TableHead className="text-right">Quality Bar</TableHead>
                    <TableHead className="text-right">
                      <ThumbsUp className="ml-auto h-3.5 w-3.5" />
                    </TableHead>
                    <TableHead className="text-right">
                      <ThumbsDown className="ml-auto h-3.5 w-3.5" />
                    </TableHead>
                    <TableHead className="text-right">Best Picks</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {Object.values(evalSummary.by_chat_provider).map((p) => {
                    const isRecommended = p.chat_provider_id === evalSummary.recommendation;
                    const score = p.judge_avg_score;
                    const pct = score != null ? (score / 5) * 100 : null;
                    return (
                      <TableRow key={p.chat_provider_id}>
                        <TableCell className="font-medium">
                          {p.chat_provider_name}
                          {isRecommended && (
                            <Badge variant="outline" className="ml-2 border-amber-400 text-amber-600 text-xs">
                              Best
                            </Badge>
                          )}
                        </TableCell>
                        <TableCell className="text-right tabular-nums">
                          {score != null ? `${score.toFixed(2)} / 5` : "—"}
                        </TableCell>
                        <TableCell className="text-right">
                          {pct != null ? (
                            <div className="ml-auto h-2 w-24 rounded-full bg-muted">
                              <div
                                className="h-2 rounded-full bg-amber-400"
                                style={{ width: `${pct}%` }}
                              />
                            </div>
                          ) : (
                            <span className="text-muted-foreground">—</span>
                          )}
                        </TableCell>
                        <TableCell className="text-right tabular-nums">{p.thumb_up}</TableCell>
                        <TableCell className="text-right tabular-nums">{p.thumb_down}</TableCell>
                        <TableCell className="text-right tabular-nums">{p.best_picks}</TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        )}

        {!effectiveSession && (
          <div className="py-20 text-center text-muted-foreground">
            <p>Select a session to view metrics.</p>
          </div>
        )}
      </div>
    </AppShell>
  );
}
