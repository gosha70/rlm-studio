"use client";

import { useState, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import useSWR from "swr";
import { AppShell } from "@/components/shared/app-shell";
import { EngineBadge } from "@/components/shared/engine-badge";
import { PERF_UI_ENABLED } from "@/lib/branding";
import { Timeline } from "@/components/trace/timeline";
import { TraceTree } from "@/components/trace/trace-tree";
import { StepDetail } from "@/components/trace/step-detail";
import { CodeBlock } from "@/components/trace/code-block";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Trash2 } from "lucide-react";
import { toast } from "sonner";
import {
  deleteExecution,
  deleteAllExecutions,
  getExecutions,
  getSessions,
  getTrace,
  type ExecutionSummary,
  type SessionSummary,
  type TraceResponse,
  type TraceStep,
} from "@/lib/api";

interface TraceDetailPanelProps {
  trace: TraceResponse;
  selectedStep: TraceStep | null;
  onSelectStep: (step: TraceStep | null) => void;
  /** True when the parent row uses the odd stripe color (zinc-200 light,
   * zinc-800 dark). Inner components pick the OPPOSITE stripe so they
   * always pop against the parent rather than merging with it — matters
   * especially in dark mode where bg-card equals zinc-900. */
  parentIsOdd: boolean;
}

/** The per-trace summary bar + Timeline/Tree/Code/Performance tabs +
 * optional step-detail pane. Extracted so the Traces page can render
 * it inline under a clicked row (instead of below the table, which
 * forced the user to scroll and lose visual focus on the row). */
function TraceDetailPanel({
  trace,
  selectedStep,
  onSelectStep,
  parentIsOdd,
}: TraceDetailPanelProps) {
  // Inner bg = opposite of parent stripe, so Card / TabsList / active
  // tab always have visible contrast against whichever stripe is behind
  // them.
  const innerBg = parentIsOdd
    ? "bg-zinc-100 dark:bg-zinc-900"
    : "bg-zinc-200 dark:bg-zinc-800";
  // The shadcn TabsTrigger default active bg (`bg-background` = #fafafa
  // light / #0a0a0a dark) has poor contrast against the zinc-100 /
  // zinc-900 lists used here — only ~10–15 levels of lightness apart.
  // Pick an active bg per parity AND add a ring so the active tab
  // always has a visible outline independent of bg similarity.
  const triggerActive = parentIsOdd
    ? "data-[state=active]:bg-white data-[state=active]:text-foreground data-[state=active]:ring-1 data-[state=active]:ring-zinc-400 dark:data-[state=active]:bg-zinc-700 dark:data-[state=active]:ring-zinc-500"
    : "data-[state=active]:bg-white data-[state=active]:text-foreground data-[state=active]:ring-1 data-[state=active]:ring-zinc-400 dark:data-[state=active]:bg-zinc-950 dark:data-[state=active]:ring-zinc-600";
  return (
    <div className="flex flex-col gap-4">
      {/* Summary bar */}
      <Card className={innerBg}>
        <CardContent className="flex flex-wrap items-center gap-4 p-4">
          <div className="flex-1 min-w-0">
            <p className="text-xs text-muted-foreground">Query</p>
            <p className="truncate font-medium">{trace.query}</p>
          </div>
          <Badge variant="outline">{trace.mode.toUpperCase()}</Badge>
          <EngineBadge mode={trace.mode} />
          {trace.chat_provider_name && (
            <>
              <Separator orientation="vertical" className="h-8" />
              <div className="text-sm">
                <span className="text-muted-foreground">Chat Provider: </span>
                <span className="font-medium">{trace.chat_provider_name}</span>
              </div>
            </>
          )}
          <Separator orientation="vertical" className="h-8" />
          <div className="text-sm">
            <span className="text-muted-foreground">Steps: </span>
            <span className="font-medium">{trace.budget.steps_used}/{trace.budget.steps_limit}</span>
          </div>
          <Separator orientation="vertical" className="h-8" />
          <div className="text-sm">
            <span className="text-muted-foreground">Tokens: </span>
            <span className="font-medium">{trace.budget.tokens_used.toLocaleString()}</span>
          </div>
          <Separator orientation="vertical" className="h-8" />
          <div className="text-sm">
            <span className="text-muted-foreground">Cost: </span>
            <span className="font-medium">${trace.budget.cost_used.toFixed(4)}</span>
          </div>
          <Badge variant={trace.result.success ? "success" : "destructive"}>
            {trace.status}
          </Badge>
        </CardContent>
      </Card>

      {/* Tabbed views */}
      <Tabs defaultValue="timeline">
        <TabsList className={innerBg}>
          <TabsTrigger value="timeline" className={triggerActive}>Timeline</TabsTrigger>
          <TabsTrigger value="tree" className={triggerActive}>Tree</TabsTrigger>
          <TabsTrigger value="code" className={triggerActive}>Code</TabsTrigger>
          {PERF_UI_ENABLED && (
            <TabsTrigger value="performance" className={triggerActive}>Performance</TabsTrigger>
          )}
        </TabsList>

        <TabsContent value="timeline">
          <Timeline
            steps={trace.steps}
            onSelect={onSelectStep}
            selectedIndex={selectedStep?.index}
          />
        </TabsContent>

        <TabsContent value="tree">
          <TraceTree
            steps={trace.steps}
            onSelect={onSelectStep}
            selectedIndex={selectedStep?.index}
          />
        </TabsContent>

        <TabsContent value="code">
          <div className="space-y-4">
            {trace.steps
              .filter((s) => s.code)
              .map((step) => (
                <div key={step.index}>
                  <p className="mb-1 text-sm font-medium">
                    Step {step.index + 1}: {step.action_type}
                  </p>
                  <CodeBlock code={step.code!} output={step.output} />
                </div>
              ))}
            {trace.steps.filter((s) => s.code).length === 0 && (
              <p className="text-sm text-muted-foreground">No code steps in this trace.</p>
            )}
          </div>
        </TabsContent>

        {PERF_UI_ENABLED && (
          <TabsContent value="performance">
            <PerformanceTab steps={trace.steps} />
          </TabsContent>
        )}
      </Tabs>

      {/* Step detail panel */}
      {selectedStep && (
        <div>
          <StepDetail step={selectedStep} />
        </div>
      )}
    </div>
  );
}

function PerformanceTab({ steps }: { steps: TraceStep[] }) {
  const rows = steps.map((s) => ({
    index: s.index,
    action: s.action_type,
    promptTokens: s.prompt_tokens ?? s.input_tokens ?? 0,
    completionTokens: s.completion_tokens ?? s.output_tokens ?? 0,
    cachedTokens: s.cached_tokens ?? 0,
    ttftMs: s.ttft_ms ?? null,
    decodeMs: s.decode_ms ?? 0,
    durationSeconds: s.duration_seconds ?? 0,
  }));

  const anyTelemetry = rows.some(
    (r) => r.ttftMs !== null || r.decodeMs > 0 || r.cachedTokens > 0,
  );

  if (!anyTelemetry) {
    return (
      <p className="text-sm text-muted-foreground">
        No prefill/decode telemetry on this trace. Performance metrics are
        populated on runs recorded after Phase 3 of the prefill/decode
        telemetry rollout.
      </p>
    );
  }

  return (
    <Table className="text-xs">
      <TableHeader>
        <TableRow>
          <TableHead>Step</TableHead>
          <TableHead>Action</TableHead>
          <TableHead className="text-right">Prompt</TableHead>
          <TableHead className="text-right">Completion</TableHead>
          <TableHead className="text-right">Cached</TableHead>
          <TableHead className="text-right">TTFT (ms)</TableHead>
          <TableHead className="text-right">Decode (ms)</TableHead>
          <TableHead className="text-right">Duration (s)</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((r) => (
          <TableRow key={r.index}>
            <TableCell>{r.index + 1}</TableCell>
            <TableCell>
              <Badge variant="outline" className="text-xs">
                {r.action}
              </Badge>
            </TableCell>
            <TableCell className="text-right">{r.promptTokens}</TableCell>
            <TableCell className="text-right">{r.completionTokens}</TableCell>
            <TableCell className="text-right">{r.cachedTokens}</TableCell>
            <TableCell className="text-right">
              {r.ttftMs === null ? "—" : r.ttftMs}
            </TableCell>
            <TableCell className="text-right">{r.decodeMs}</TableCell>
            <TableCell className="text-right">
              {r.durationSeconds.toFixed(2)}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

function TracesPageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [trace, setTrace] = useState<TraceResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedStep, setSelectedStep] = useState<TraceStep | null>(null);
  const [filterProviderId, setFilterProviderId] = useState<string>("");
  const [filterSessionId, setFilterSessionId] = useState<string>("");
  const [limit, setLimit] = useState(20);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [deleting, setDeleting] = useState(false);

  const { data: sessions = [] } = useSWR<SessionSummary[]>("sessions", getSessions);

  // Unfiltered fetch — drives the chat-provider filter dropdown so it never disappears
  const { data: allExecutions = [] } = useSWR<ExecutionSummary[]>(
    ["executions-all", filterSessionId],
    () => getExecutions(50, undefined, filterSessionId || undefined),
    { refreshInterval: 5000 },
  );

  // Filtered fetch — drives the table
  const { data: executions = [], mutate: mutateExecutions } = useSWR<ExecutionSummary[]>(
    ["executions", filterProviderId, filterSessionId, limit],
    () => getExecutions(limit, filterProviderId || undefined, filterSessionId || undefined),
    { refreshInterval: 5000 },
  );

  // Unique chat providers from ALL executions (within selected session) for the filter dropdown
  const chatProviderOptions = Array.from(
    new Map(
      allExecutions
        .filter((e) => e.chat_provider_id && e.chat_provider_name)
        .map((e) => [e.chat_provider_id!, e.chat_provider_name!]),
    ),
  );

  const handleSelectExecution = async (executionId: string) => {
    // Toggle semantics: clicking the currently-expanded row collapses
    // the detail. Without this the user had to scroll down to find
    // the detail and had no affordance to close it from the row.
    if (trace?.execution_id === executionId) {
      setTrace(null);
      setSelectedStep(null);
      setError(null);
      return;
    }
    setError(null);
    setLoading(true);
    setSelectedStep(null);
    try {
      const data = await getTrace(executionId);
      setTrace(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load trace");
      setTrace(null);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteSelected = async () => {
    if (selectedIds.size === 0) return;
    setDeleting(true);
    try {
      await Promise.all([...selectedIds].map((id) => deleteExecution(id)));
      toast.success(`Deleted ${selectedIds.size} trace(s)`);
      setSelectedIds(new Set());
      setTrace(null);
      mutateExecutions();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Delete failed");
    } finally {
      setDeleting(false);
    }
  };

  const handleDeleteAll = async () => {
    if (!window.confirm("Delete all traces? This cannot be undone.")) return;
    setDeleting(true);
    try {
      await deleteAllExecutions();
      toast.success("All traces deleted");
      setSelectedIds(new Set());
      setTrace(null);
      mutateExecutions();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Delete failed");
    } finally {
      setDeleting(false);
    }
  };

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleSelectAll = () => {
    if (selectedIds.size === executions.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(executions.map((e) => e.execution_id)));
    }
  };

  // Auto-load trace when navigated from Dashboard with ?exec=<id>
  useEffect(() => {
    const execId = searchParams.get("exec");
    if (execId) {
      handleSelectExecution(execId);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <AppShell>
      <div className="mx-auto max-w-[1200px] space-y-6 p-6">
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <h2 className="text-2xl font-semibold">Traces</h2>
          <div className="flex items-center gap-2">
            {sessions.length > 0 && (
              <Select
                value={filterSessionId}
                onValueChange={(v) => { setFilterSessionId(v === "all" ? "" : v); setFilterProviderId(""); setLimit(20); setTrace(null); setSelectedStep(null); }}
              >
                <SelectTrigger className="w-48" aria-label="Filter by Session">
                  <SelectValue placeholder="All Sessions" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Sessions</SelectItem>
                  {sessions.map((s) => (
                    <SelectItem key={s.id} value={s.id}>{s.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
            {chatProviderOptions.length > 0 && (
              <Select
                value={filterProviderId}
                onValueChange={(v) => { setFilterProviderId(v === "all" ? "" : v); setLimit(20); setTrace(null); setSelectedStep(null); }}
              >
                <SelectTrigger className="w-56" aria-label="Filter by Chat Provider">
                  <SelectValue placeholder="All Chat Providers" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Chat Providers</SelectItem>
                  {chatProviderOptions.map(([id, name]) => (
                    <SelectItem key={id} value={id}>{name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>
        </div>

        {/* Execution list */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-3">
            <CardTitle className="text-base">Recent Executions</CardTitle>
            <div className="flex items-center gap-2">
              {selectedIds.size > 0 && (
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={handleDeleteSelected}
                  disabled={deleting}
                >
                  <Trash2 className="mr-1 h-3 w-3" />
                  Delete {selectedIds.size}
                </Button>
              )}
              {executions.length > 0 && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleDeleteAll}
                  disabled={deleting}
                >
                  Clear all
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {executions.length === 0 ? (
              <p className="py-8 text-center text-sm text-muted-foreground">
                No executions yet. Send a message in Chat to see traces here.
              </p>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-10">
                      <input
                        type="checkbox"
                        checked={executions.length > 0 && selectedIds.size === executions.length}
                        onChange={toggleSelectAll}
                        aria-label="Select all"
                        className="h-4 w-4 rounded border-input accent-primary"
                      />
                    </TableHead>
                    <TableHead>Query</TableHead>
                    <TableHead>Chat Provider</TableHead>
                    <TableHead>Mode</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">Tokens</TableHead>
                    <TableHead className="text-right">Cost</TableHead>
                    <TableHead className="w-36 text-right">
                      <span className="sr-only">Actions</span>
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {executions.flatMap((exec, idx) => {
                    const isExpanded = trace?.execution_id === exec.execution_id;
                    // Zebra-stripe by execution index (not DOM-row index),
                    // so an expanded detail row below an odd row doesn't
                    // shift the alternation for every row after it.
                    // Hover must repeat the same bg: shadcn's TableRow
                    // ships with `hover:bg-muted/50`, which would otherwise
                    // override the stripe and make the row appear to blink
                    // as the cursor moves across it.
                    const stripeClass =
                      idx % 2 === 1
                        ? "bg-zinc-200 hover:bg-zinc-200 dark:bg-zinc-800 dark:hover:bg-zinc-800"
                        : "bg-zinc-100 hover:bg-zinc-100 dark:bg-zinc-900 dark:hover:bg-zinc-900";
                    const rowEls: React.ReactNode[] = [
                      <TableRow
                        key={exec.execution_id}
                        className={`cursor-pointer ${stripeClass}`}
                        data-expanded={isExpanded ? "true" : undefined}
                        aria-expanded={isExpanded}
                        onClick={() => handleSelectExecution(exec.execution_id)}
                        role="button"
                        tabIndex={0}
                        onKeyDown={(e) => {
                          // Only handle Enter/Space when the row itself is
                          // the keyboard target. Interactive descendants
                          // (the checkbox, the Replay-in-Learn CTA) bubble
                          // keydown events up to the row; without this
                          // guard their Enter/Space activations would also
                          // trigger "open trace detail", racing the
                          // descendant's own handler.
                          if (e.target !== e.currentTarget) return;
                          if (e.key === "Enter" || e.key === " ") {
                            e.preventDefault();
                            handleSelectExecution(exec.execution_id);
                          }
                        }}
                        aria-label={
                          isExpanded
                            ? `Collapse trace for: ${exec.query}`
                            : `Expand trace for: ${exec.query}`
                        }
                      >
                      <TableCell onClick={(e) => e.stopPropagation()}>
                        <input
                          type="checkbox"
                          checked={selectedIds.has(exec.execution_id)}
                          onChange={() => toggleSelect(exec.execution_id)}
                          aria-label={`Select trace: ${exec.query}`}
                          className="h-4 w-4 rounded border-input accent-primary"
                        />
                      </TableCell>
                      <TableCell className="max-w-[300px] truncate font-medium">
                        {exec.query}
                      </TableCell>
                      <TableCell className="text-sm text-muted-foreground">
                        {exec.chat_provider_name || "—"}
                      </TableCell>
                      <TableCell>
                        <div className="flex flex-wrap items-center gap-1.5">
                          <Badge variant="outline">{exec.mode.toUpperCase()}</Badge>
                          <EngineBadge mode={exec.mode} />
                        </div>
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant={
                            exec.status === "complete"
                              ? "success"
                              : exec.status === "running"
                                ? "default"
                                : "destructive"
                          }
                        >
                          {exec.status}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        {exec.total_tokens.toLocaleString()}
                      </TableCell>
                      <TableCell className="text-right">
                        ${exec.total_cost.toFixed(4)}
                      </TableCell>
                      <TableCell
                        className="text-right"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={(e) => {
                            e.stopPropagation();
                            router.push(
                              `/learn/replay?id=${encodeURIComponent(exec.execution_id)}&from=traces`,
                            );
                          }}
                          aria-label={`Replay in Learn: ${exec.query}`}
                        >
                          Replay in Learn
                        </Button>
                      </TableCell>
                    </TableRow>,
                    ];
                    if (isExpanded) {
                      rowEls.push(
                        <TableRow
                          key={`${exec.execution_id}-detail`}
                          // Inherit the parent row's stripe so the detail
                          // pane reads as one continuous band with the
                          // clicked row. Never a selection-highlight —
                          // clicking must not repaint the parent row.
                          className={stripeClass}
                        >
                          <TableCell
                            colSpan={8}
                            className="p-4"
                            // Don't let clicks inside the detail bubble
                            // up to the row click handler and toggle
                            // the detail closed unexpectedly.
                            onClick={(e) => e.stopPropagation()}
                          >
                            {loading && !trace ? (
                              <p className="text-sm text-muted-foreground">
                                Loading trace...
                              </p>
                            ) : error ? (
                              <p className="text-sm text-destructive" role="alert">
                                {error}
                              </p>
                            ) : trace ? (
                              <TraceDetailPanel
                                trace={trace}
                                selectedStep={selectedStep}
                                onSelectStep={setSelectedStep}
                                parentIsOdd={idx % 2 === 1}
                              />
                            ) : null}
                          </TableCell>
                        </TableRow>,
                      );
                    }
                    return rowEls;
                  })}
                </TableBody>
              </Table>
            )}
            {executions.length === limit && (
              <div className="mt-3 flex justify-center">
                <Button variant="outline" size="sm" onClick={() => setLimit((l) => l + 20)}>
                  Load more
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Trace detail now renders inline under the clicked row — see
            the flatMap block above. Errors/loading that fire before
            any row is expanded still surface here as a safety net for
            anything that might load a trace without an active row
            (none today, but keeps the panel resilient). */}
        {!trace && error && (
          <p className="text-sm text-destructive" role="alert">{error}</p>
        )}
      </div>
    </AppShell>
  );
}

export default function TracesPage() {
  return (
    <Suspense>
      <TracesPageInner />
    </Suspense>
  );
}
