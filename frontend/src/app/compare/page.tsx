"use client";

/**
 * Compare page — Provider × Mode Matrix Comparison UI (SDD Bet 3 item 3.4).
 *
 * Wraps the synchronous POST /api/chat/compare-matrix endpoint in a
 * self-contained single-purpose view:
 *
 *   - Query input + content/file input
 *   - LLM Provider chip multi-select (filtered to connected/configured)
 *   - Mode multi-select (direct / rlm / rag)
 *   - Profile dropdown (pre-fills runtime_settings + budget)
 *   - Advanced settings (collapsible): runtime, budget, rag config
 *   - Ranking metric dropdown
 *   - "Run Compare" button
 *   - Result grid: one card per (provider × mode) slot with answer
 *     preview, token/cost/latency metrics, and an expand button that
 *     reveals the full answer.
 *   - Summary strip: best slot, total elapsed, comparison_group_id
 *
 * The endpoint is synchronous (waits for every slot to finish before
 * returning), so this page shows a spinner for the duration of the run
 * and renders results all at once.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import useSWR from "swr";
import { Loader2, Trophy, Hash, DollarSign, Clock, Upload, X, ChevronDown, ChevronUp, Ban, Sparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

import { AppShell } from "@/components/shared/app-shell";
import { PERF_UI_ENABLED } from "@/lib/branding";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { FieldLabel } from "@/components/settings/field-label";
import { JudgeScores } from "@/components/chat/judge-scores";
import { JudgeDimensionsChart } from "@/components/metrics/judge-dimensions-chart";
import { toast } from "sonner";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import {
  getConfig,
  getLLMProviders,
  getProfiles,
  submitCompareMatrix,
  triggerJudge,
  uploadFile,
  type AppConfig,
  type LLMProviderConfig,
  type RunProfile,
  type CompareMatrixResponse,
  type CompareMatrixSlotResponse,
  type FileUploadResponse,
  type JudgeScoreData,
  type MatrixRankingMetric,
  type MatrixSlotMode,
  type RuntimeSettings,
  type BudgetConfig,
  type RAGConfig,
  type CompareMatrixRequestV2,
} from "@/lib/api";
import { ALL_EXECUTION_MODES, MODE_DIRECT, MODE_RAG } from "@/lib/constants";
import { EngineBadge } from "@/components/shared/engine-badge";
import { useEngineAvailability } from "@/components/shared/use-engines";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ALL_MODES: MatrixSlotMode[] = [...ALL_EXECUTION_MODES];
const MODE_DESCRIPTIONS: Record<MatrixSlotMode, string> = {
  direct: "Single LLM call with full content",
  rlm: "Recursive exploration with sandbox",
  rag: "Retrieval over embedded chunks",
  rlm_official: "Paper authors' reference implementation (rlms)",
};

const RANKING_METRICS: {
  value: MatrixRankingMetric;
  label: string;
  help: string;
}[] = [
  { value: "cost", label: "Cost (lowest wins)", help: "Cheapest successful slot" },
  { value: "tokens", label: "Tokens (fewest wins)", help: "Smallest token footprint" },
  { value: "latency", label: "Latency (fastest wins)", help: "Wall-clock elapsed time" },
  {
    value: "answer_per_cost",
    label: "Answer/Cost (most wins)",
    help: "Answer length divided by cost",
  },
  {
    value: "judge_score",
    label: "Judge Score (highest wins)",
    help: "LLM-as-judge overall quality score (1-5). Run Auto-Judge first.",
  },
];

// Phase 5 — "Performance shape" ranking options. Hidden behind
// `NEXT_PUBLIC_RLM_STUDIO_PERF_UI=1` (see lib/branding.ts; the legacy
// RLMKIT name is honoured for one release) at build time. When off, the
// backend still accepts these values — the frontend just doesn't
// surface them in the dropdown.
const PERF_RANKING_METRICS: typeof RANKING_METRICS = [
  {
    value: "ttft",
    label: "TTFT (lowest wins)",
    help: "Median time-to-first-token across the slot's steps",
  },
  {
    value: "decode_tokens_per_sec",
    label: "Decode tok/s (highest wins)",
    help: "Completion tokens per decode-phase second; measures steady-state speed",
  },
  {
    value: "cache_hit_rate",
    label: "Cache hit rate (highest wins)",
    help: "Fraction of prompt tokens served from the provider's prefix cache",
  },
];

const VISIBLE_RANKING_METRICS: typeof RANKING_METRICS = PERF_UI_ENABLED
  ? [...RANKING_METRICS, ...PERF_RANKING_METRICS]
  : RANKING_METRICS;

// Matches the backend MAX_SLOTS constant in RunMatrixComparisonUseCase.
const MAX_SLOTS = 10;

/** Detect slots that failed or produced only a warning (not a real answer). */
function _isSlotFailed(slot: CompareMatrixSlotResponse): boolean {
  if (!slot.success) return true;
  // Backend wraps errors/warnings with these prefixes
  const a = slot.answer;
  return a.startsWith("\u26a0\ufe0f") || a.startsWith("Error:");
}

/** Human-friendly error message — strip raw JSON/stack traces. */
function _friendlyError(slot: CompareMatrixSlotResponse): string {
  const raw = slot.error || slot.answer || "Execution failed";
  // Extract just the user-facing message from litellm errors
  const match = raw.match(/Cannot connect to the LLM server[^.]*\./);
  if (match) return match[0];
  const timeoutMatch = raw.match(/RAG timed out[^.]*\./);
  if (timeoutMatch) return timeoutMatch[0];
  const budgetMatch = raw.match(/Step budget exhausted[^.]*\./);
  if (budgetMatch) return budgetMatch[0];
  // Generic: truncate at a reasonable length
  if (raw.length > 200) return raw.slice(0, 200) + "\u2026";
  return raw;
}

// Tooltip help text for configuration fields.  Defined as constants to
// avoid magic strings and keep them in sync with budget-config.tsx.
const FIELD_TOOLTIPS = {
  temperature:
    "Controls randomness. Lower values (0.1–0.3) produce focused, deterministic answers. Higher values (0.7–1.5) increase creativity and variety.",
  top_p:
    "Nucleus sampling: only tokens within the top-p cumulative probability are considered. Lower values narrow the vocabulary. Note: Anthropic Claude does not support temperature + top_p together — when top_p is non-default, temperature is omitted for Claude models.",
  max_output_tokens:
    "Maximum tokens the model can generate per call. Longer answers need higher values. Does not affect input size.",
  timeout:
    "Wall-clock timeout per LLM call in seconds. Increase for slow models or large documents.",
  max_steps:
    "Maximum number of LLM reasoning steps per query. Higher values allow deeper exploration in RLM mode but increase cost and latency.",
  max_time:
    "Total wall-clock budget for the entire execution (all steps combined). The run is terminated if this limit is exceeded.",
  max_cost:
    "Dollar limit per slot. The execution is terminated if this cost is exceeded. Applies per slot, not total — a 6-slot matrix with $5/slot allows up to $30 total.",
  repeat_limit:
    "How many duplicate inspect results before the RLM loop forces finalization. Lower values converge faster but may miss content.",
  nudge_at_fraction:
    "Fraction of max_steps at which a soft convergence nudge is sent (e.g., 0.4 = at 40% of budget). Encourages the model to start synthesizing.",
  chunk_size:
    "Number of characters per RAG chunk. Smaller chunks improve precision but increase the number of embeddings.",
  chunk_overlap:
    "Characters of overlap between adjacent chunks. Prevents information loss at chunk boundaries. Typically 10–15% of chunk size.",
  top_k:
    "Number of most-similar chunks retrieved for context. Higher values provide more context but increase token usage.",
  embedding_model:
    "Model used to generate vector embeddings for RAG retrieval. Must match the model used during indexing.",
} as const;

interface InlineConfig {
  // Runtime
  temperature: number;
  top_p: number;
  max_output_tokens: number;
  timeout_seconds: number;
  // Budget
  max_steps: number;
  max_time_seconds: number;
  max_cost_usd: number;
  repeat_limit: number;
  nudge_at_fraction: number;
  // RAG
  chunk_size: number;
  chunk_overlap: number;
  top_k: number;
  embedding_model: string;
}

// NOTE: These defaults mirror the backend's sandbox_vars.py / models.py
// defaults.  If the backend changes, these must be updated to match.
const DEFAULT_CONFIG: InlineConfig = {
  temperature: 0.7,
  top_p: 1.0,
  max_output_tokens: 4096,
  timeout_seconds: 120,
  max_steps: 16,
  max_time_seconds: 600,
  max_cost_usd: 5.0,
  repeat_limit: 2,
  nudge_at_fraction: 0.4,
  chunk_size: 1000,
  chunk_overlap: 150,
  top_k: 5,
  embedding_model: "text-embedding-3-small",
};

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function ComparePage() {
  // --- Data ----------------------------------------------------------------
  const { data: llmProviders = [] } = useSWR<LLMProviderConfig[]>(
    "llm-providers",
    () => getLLMProviders(),
  );
  const { data: profiles = [] } = useSWR<RunProfile[]>(
    "profiles",
    () => getProfiles(),
  );
  const { unavailableReasonFor } = useEngineAvailability();

  // --- Inputs --------------------------------------------------------------
  const [query, setQuery] = useState("");
  const [content, setContent] = useState("");
  const [uploadedFile, setUploadedFile] = useState<FileUploadResponse | null>(null);
  const [uploadingFile, setUploadingFile] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [selectedLLMProviderIds, setSelectedLLMProviderIds] = useState<string[]>([]);
  const [selectedModes, setSelectedModes] = useState<Set<MatrixSlotMode>>(
    new Set([MODE_DIRECT]),
  );
  const [rankingMetric, setRankingMetric] = useState<MatrixRankingMetric>("cost");
  const [selectedProfileId, setSelectedProfileId] = useState<string>("none");
  const [config, setConfig] = useState<InlineConfig>({ ...DEFAULT_CONFIG });
  const [advancedOpen, setAdvancedOpen] = useState(false);

  // --- Execution state -----------------------------------------------------
  const [result, setResult] = useState<CompareMatrixResponse | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [expandedSlotId, setExpandedSlotId] = useState<string | null>(null);

  // --- Judge state ----------------------------------------------------------
  const { data: appConfig } = useSWR<AppConfig>("config", () => getConfig());
  const [judgeScores, setJudgeScores] = useState<Record<string, JudgeScoreData>>({});
  const [isJudging, setIsJudging] = useState(false);
  const judgeConfigured = !!appConfig?.judge_chat_provider_id;

  // --- Auto-select first available LLM provider once (not on every SWR revalidation) ---
  const hasAutoSelected = useRef(false);
  useEffect(() => {
    if (hasAutoSelected.current) return;
    if (selectedLLMProviderIds.length === 0 && llmProviders.length > 0) {
      const firstUsable = llmProviders.find(
        (lp) => lp.status === "connected" || lp.status === "configured",
      );
      if (firstUsable) {
        setSelectedLLMProviderIds([firstUsable.id]);
        hasAutoSelected.current = true;
      }
    }
  }, [llmProviders, selectedLLMProviderIds.length]);

  // --- Drop modes whose engine turned out to be unavailable ----------------
  // Availability arrives after the first paint, so a mode can already be
  // selected by the time the server reports its engine missing.  The picker
  // button is disabled by then, leaving the user no way to deselect it, and
  // the backend rejects the *whole* matrix when any requested mode cannot
  // run — so prune the selection instead of stranding it.
  useEffect(() => {
    setSelectedModes((prev) => {
      const kept = [...prev].filter((m) => unavailableReasonFor(m) === null);
      if (kept.length === prev.size) return prev;
      // Never leave the picker empty: Direct is always runnable.
      return new Set(kept.length > 0 ? kept : [MODE_DIRECT]);
    });
  }, [unavailableReasonFor]);

  // --- Profile pre-fill ----------------------------------------------------
  const handleProfileChange = useCallback(
    (profileId: string) => {
      setSelectedProfileId(profileId);
      if (profileId === "none") {
        setConfig({ ...DEFAULT_CONFIG });
        return;
      }
      const profile = profiles.find((p) => p.id === profileId);
      if (!profile) return;
      setConfig((prev) => ({
        ...prev,
        temperature: profile.runtime_settings?.temperature ?? prev.temperature,
        top_p: profile.runtime_settings?.top_p ?? prev.top_p,
        max_output_tokens: profile.runtime_settings?.max_output_tokens ?? prev.max_output_tokens,
        timeout_seconds: profile.runtime_settings?.timeout_seconds ?? prev.timeout_seconds,
        max_steps: profile.budget?.max_steps ?? prev.max_steps,
        max_time_seconds: profile.budget?.max_time_seconds ?? prev.max_time_seconds,
        repeat_limit: profile.budget?.repeat_limit ?? prev.repeat_limit,
        nudge_at_fraction: profile.budget?.nudge_at_fraction ?? prev.nudge_at_fraction,
      }));
    },
    [profiles],
  );

  // --- Derived -------------------------------------------------------------
  const isProviderUsable = useCallback(
    (lp: LLMProviderConfig) => lp.status === "connected" || lp.status === "configured",
    [],
  );

  const totalSlots = selectedLLMProviderIds.length * selectedModes.size;
  const tooManySlots = totalSlots > MAX_SLOTS;
  const modesSorted = useMemo(
    () => ALL_MODES.filter((m) => selectedModes.has(m)),
    [selectedModes],
  );

  const canRun =
    query.trim().length > 0 &&
    selectedLLMProviderIds.length > 0 &&
    selectedModes.size > 0 &&
    !tooManySlots &&
    !isRunning &&
    (content.trim().length > 0 || uploadedFile !== null);

  // --- Handlers ------------------------------------------------------------
  const toggleMode = useCallback((mode: MatrixSlotMode) => {
    setSelectedModes((prev) => {
      const next = new Set(prev);
      if (next.has(mode)) {
        if (next.size > 1) next.delete(mode);
      } else {
        next.add(mode);
      }
      return next;
    });
  }, []);

  const toggleLLMProvider = useCallback((id: string) => {
    setSelectedLLMProviderIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id],
    );
  }, []);

  const handleFileUpload = useCallback(async (file: File) => {
    setUploadingFile(true);
    setUploadError(null);
    try {
      const rec = await uploadFile(file);
      setUploadedFile(rec);
      setContent("");
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploadingFile(false);
    }
  }, []);

  const patchConfig = useCallback(
    <K extends keyof InlineConfig>(key: K, value: InlineConfig[K]) => {
      setConfig((prev) => ({ ...prev, [key]: value }));
    },
    [],
  );

  const handleRun = useCallback(async () => {
    if (!canRun) return;
    setIsRunning(true);
    setRunError(null);
    setResult(null);
    setExpandedSlotId(null);
    setJudgeScores({});
    try {
      const req: CompareMatrixRequestV2 = {
        query: query.trim(),
        content: uploadedFile ? null : content.trim() || null,
        file_ids: uploadedFile ? [uploadedFile.id] : null,
        llm_provider_ids: selectedLLMProviderIds,
        modes: modesSorted,
        ranking_metric: rankingMetric,
        runtime_settings: {
          temperature: config.temperature,
          top_p: config.top_p,
          max_output_tokens: config.max_output_tokens,
          timeout_seconds: config.timeout_seconds,
        } satisfies RuntimeSettings,
        budget: {
          max_steps: config.max_steps,
          max_tokens: 50000,
          max_cost_usd: config.max_cost_usd,
          max_time_seconds: config.max_time_seconds,
          max_recursion_depth: 5,
          repeat_limit: config.repeat_limit,
          nudge_at_fraction: config.nudge_at_fraction,
        } satisfies BudgetConfig,
        rag_config: selectedModes.has(MODE_RAG)
          ? ({
              chunk_size: config.chunk_size,
              chunk_overlap: config.chunk_overlap,
              top_k: config.top_k,
              embedding_model: config.embedding_model,
            } satisfies RAGConfig)
          : null,
      };
      const resp = await submitCompareMatrix(req);
      setResult(resp);
    } catch (err) {
      setRunError(err instanceof Error ? err.message : "Matrix run failed");
    } finally {
      setIsRunning(false);
    }
  }, [
    canRun,
    query,
    content,
    uploadedFile,
    selectedLLMProviderIds,
    modesSorted,
    rankingMetric,
    selectedModes,
    config,
  ]);

  // --- Judge handler --------------------------------------------------------
  const handleJudge = useCallback(async () => {
    if (!result || isJudging) return;
    const usableSlots = result.slots.filter((s) => !_isSlotFailed(s));
    if (usableSlots.length === 0) {
      toast.error("No successful slots to judge");
      return;
    }
    setIsJudging(true);
    try {
      const judgeResult = await triggerJudge({
        session_id: result.session_id,
        execution_ids: usableSlots.map((s) => s.execution_id),
        mode: "pointwise",
      });
      const newScores: Record<string, JudgeScoreData> = { ...judgeScores };
      for (const s of judgeResult.pointwise) {
        newScores[s.execution_id] = s;
      }
      setJudgeScores(newScores);
      toast.success(`Judged ${judgeResult.pointwise.length} slots`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Judge evaluation failed");
    } finally {
      setIsJudging(false);
    }
  }, [result, isJudging, judgeScores]);

  // --- Render --------------------------------------------------------------
  return (
    <AppShell>
      <div className="mx-auto max-w-6xl space-y-6 p-6">
        <header>
          <h1 className="text-2xl font-bold">LLM Tuner</h1>
          <p className="text-sm text-muted-foreground">
            Run the same query across{" "}
            <span className="font-mono">N providers × M modes</span> in
            parallel and see a ranked comparison.
          </p>
        </header>

        {/* ---------- Inputs panel ---------- */}
        <Card>
          <CardHeader>
            <CardTitle>Configure run</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Query */}
            <div>
              <label className="mb-1 block text-sm font-medium">Query</label>
              <Textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="What would you like to ask?"
                rows={2}
                disabled={isRunning}
              />
            </div>

            {/* Content or file */}
            <div>
              <label className="mb-1 block text-sm font-medium">
                Content or file
              </label>
              {uploadedFile ? (
                <div className="flex items-center justify-between rounded-md border bg-muted/30 p-2 text-sm">
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary">{uploadedFile.name}</Badge>
                    <span className="text-muted-foreground">
                      {uploadedFile.token_count.toLocaleString()} tok
                    </span>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setUploadedFile(null)}
                    disabled={isRunning}
                    aria-label="Remove uploaded file"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              ) : (
                <div className="space-y-2">
                  <Textarea
                    value={content}
                    onChange={(e) => setContent(e.target.value)}
                    placeholder="Paste text content here, or upload a file below."
                    rows={4}
                    disabled={isRunning}
                  />
                  <label
                    className={`inline-flex cursor-pointer items-center gap-2 rounded-md border border-dashed px-3 py-2 text-sm text-muted-foreground hover:bg-accent ${
                      uploadingFile ? "opacity-50" : ""
                    }`}
                  >
                    {uploadingFile ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Upload className="h-4 w-4" />
                    )}
                    <span>{uploadingFile ? "Uploading…" : "Upload file"}</span>
                    <input
                      type="file"
                      hidden
                      disabled={uploadingFile || isRunning}
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) handleFileUpload(file);
                        e.target.value = "";
                      }}
                    />
                  </label>
                  {uploadError && (
                    <div className="text-sm text-destructive">{uploadError}</div>
                  )}
                </div>
              )}
            </div>

            {/* LLM Provider chip selector */}
            <div>
              <label className="mb-2 block text-sm font-medium">
                LLM providers ({selectedLLMProviderIds.length} selected)
              </label>
              {llmProviders.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  No LLM providers configured. Add one in Settings.
                </p>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {llmProviders.map((lp) => {
                    const usable = isProviderUsable(lp);
                    const active = selectedLLMProviderIds.includes(lp.id);
                    return (
                      <button
                        key={lp.id}
                        type="button"
                        onClick={() => usable && toggleLLMProvider(lp.id)}
                        disabled={isRunning || !usable}
                        className={`flex flex-col items-start rounded-md border px-3 py-1.5 text-left transition-colors ${
                          !usable
                            ? "cursor-not-allowed border-input bg-muted/30 opacity-50"
                            : active
                              ? "border-primary bg-primary text-primary-foreground"
                              : "border-input bg-background hover:bg-accent"
                        } ${isRunning ? "cursor-not-allowed opacity-50" : ""}`}
                        aria-pressed={usable ? active : undefined}
                        title={usable ? undefined : `${lp.name} is ${lp.status} — test connection in Settings`}
                      >
                        <span className="text-sm font-medium">{lp.name}</span>
                        <span
                          className={`text-xs ${
                            !usable
                              ? "text-muted-foreground/50"
                              : active
                                ? "text-primary-foreground/70"
                                : "text-muted-foreground"
                          }`}
                        >
                          {lp.model}{!usable ? ` (${lp.status})` : ""}
                        </span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Mode selector */}
            <div>
              <label className="mb-2 block text-sm font-medium">
                Modes ({selectedModes.size} selected)
              </label>
              <div className="flex flex-wrap gap-2">
                {ALL_MODES.map((mode) => {
                  const active = selectedModes.has(mode);
                  // Third-party engines can be uninstalled; the server says why.
                  const unavailableReason = unavailableReasonFor(mode);
                  const blocked = isRunning || unavailableReason !== null;
                  return (
                    <button
                      key={mode}
                      type="button"
                      onClick={() => toggleMode(mode)}
                      disabled={blocked}
                      className={`rounded-md border px-3 py-1.5 text-sm transition-colors ${
                        active
                          ? "border-primary bg-primary text-primary-foreground"
                          : "border-input bg-background hover:bg-accent"
                      } ${blocked ? "cursor-not-allowed opacity-50" : ""}`}
                      aria-pressed={active}
                      title={unavailableReason ?? MODE_DESCRIPTIONS[mode]}
                    >
                      {mode}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Profile dropdown */}
            <div>
              <label className="mb-1 block text-sm font-medium">Profile</label>
              <Select
                value={selectedProfileId}
                onValueChange={handleProfileChange}
                disabled={isRunning}
              >
                <SelectTrigger className="w-80 truncate">
                  <SelectValue placeholder="No profile (use defaults)" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">No profile (use defaults)</SelectItem>
                  {profiles.map((p) => (
                    <SelectItem key={p.id} value={p.id} textValue={p.name}>
                      <div className="flex flex-col">
                        <span>{p.name}</span>
                        {p.description && (
                          <span className="text-xs text-muted-foreground">
                            {p.description}
                          </span>
                        )}
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Advanced settings (collapsible) */}
            <div className="rounded-md border">
              <button
                type="button"
                className="flex w-full items-center justify-between px-4 py-2 text-sm font-medium hover:bg-accent"
                onClick={() => setAdvancedOpen((v) => !v)}
                disabled={isRunning}
              >
                <span>Advanced settings</span>
                {advancedOpen ? (
                  <ChevronUp className="h-4 w-4" />
                ) : (
                  <ChevronDown className="h-4 w-4" />
                )}
              </button>
              {advancedOpen && (
                <div className="space-y-5 border-t px-4 py-4">
                  {/* Runtime settings */}
                  <div>
                    <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                      Runtime
                    </p>
                    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                      <div>
                        <FieldLabel tooltip={FIELD_TOOLTIPS.temperature} className="mb-1 text-xs text-muted-foreground">
                          Temperature (0–2)
                        </FieldLabel>
                        <Input
                          type="number"
                          min={0}
                          max={2}
                          step={0.1}
                          value={config.temperature}
                          onChange={(e) =>
                            patchConfig("temperature", parseFloat(e.target.value) || 0)
                          }
                          disabled={isRunning}
                          className="h-8 text-sm"
                        />
                      </div>
                      <div>
                        <FieldLabel tooltip={FIELD_TOOLTIPS.top_p} className="mb-1 text-xs text-muted-foreground">
                          Top-p (0–1)
                        </FieldLabel>
                        <Input
                          type="number"
                          min={0}
                          max={1}
                          step={0.05}
                          value={config.top_p}
                          onChange={(e) =>
                            patchConfig("top_p", parseFloat(e.target.value) || 0)
                          }
                          disabled={isRunning}
                          className="h-8 text-sm"
                        />
                      </div>
                      <div>
                        <FieldLabel tooltip={FIELD_TOOLTIPS.max_output_tokens} className="mb-1 text-xs text-muted-foreground">
                          Max output tokens
                        </FieldLabel>
                        <Input
                          type="number"
                          min={1}
                          value={config.max_output_tokens}
                          onChange={(e) =>
                            patchConfig(
                              "max_output_tokens",
                              parseInt(e.target.value, 10) || 1,
                            )
                          }
                          disabled={isRunning}
                          className="h-8 text-sm"
                        />
                      </div>
                      <div>
                        <FieldLabel tooltip={FIELD_TOOLTIPS.timeout} className="mb-1 text-xs text-muted-foreground">
                          Timeout (s)
                        </FieldLabel>
                        <Input
                          type="number"
                          min={1}
                          value={config.timeout_seconds}
                          onChange={(e) =>
                            patchConfig(
                              "timeout_seconds",
                              parseInt(e.target.value, 10) || 1,
                            )
                          }
                          disabled={isRunning}
                          className="h-8 text-sm"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Budget settings */}
                  <div>
                    <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                      Budget
                    </p>
                    <div className="space-y-3">
                      <div>
                        <div className="mb-1 flex items-center justify-between">
                          <FieldLabel tooltip={FIELD_TOOLTIPS.max_steps} className="text-xs text-muted-foreground">
                            Max steps (1–50)
                          </FieldLabel>
                          <span className="text-xs font-mono">{config.max_steps}</span>
                        </div>
                        <Slider
                          min={1}
                          max={50}
                          step={1}
                          value={[config.max_steps]}
                          onValueChange={([v]) => patchConfig("max_steps", v)}
                          disabled={isRunning}
                        />
                      </div>
                      <div>
                        <div className="mb-1 flex items-center justify-between">
                          <FieldLabel tooltip={FIELD_TOOLTIPS.max_time} className="text-xs text-muted-foreground">
                            Max time (5–600 s)
                          </FieldLabel>
                          <span className="text-xs font-mono">
                            {config.max_time_seconds}s
                          </span>
                        </div>
                        <Slider
                          min={5}
                          max={600}
                          step={5}
                          value={[config.max_time_seconds]}
                          onValueChange={([v]) => patchConfig("max_time_seconds", v)}
                          disabled={isRunning}
                        />
                      </div>
                      <div>
                        <FieldLabel tooltip={FIELD_TOOLTIPS.max_cost} className="mb-1 text-xs text-muted-foreground">
                          Max cost per slot (USD)
                        </FieldLabel>
                        <Input
                          type="number"
                          min={0.01}
                          max={100}
                          step={0.5}
                          value={config.max_cost_usd}
                          onChange={(e) =>
                            patchConfig(
                              "max_cost_usd",
                              parseFloat(e.target.value) || 0.01,
                            )
                          }
                          disabled={isRunning}
                          className="h-8 text-sm"
                        />
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <FieldLabel tooltip={FIELD_TOOLTIPS.repeat_limit} className="mb-1 text-xs text-muted-foreground">
                            Repeat limit (1–10)
                          </FieldLabel>
                          <Input
                            type="number"
                            min={1}
                            max={10}
                            value={config.repeat_limit}
                            onChange={(e) =>
                              patchConfig(
                                "repeat_limit",
                                parseInt(e.target.value, 10) || 1,
                              )
                            }
                            disabled={isRunning}
                            className="h-8 text-sm"
                          />
                        </div>
                        <div>
                          <FieldLabel tooltip={FIELD_TOOLTIPS.nudge_at_fraction} className="mb-1 text-xs text-muted-foreground">
                            Nudge at fraction (0.1–1.0)
                          </FieldLabel>
                          <Input
                            type="number"
                            min={0.1}
                            max={1.0}
                            step={0.1}
                            value={config.nudge_at_fraction}
                            onChange={(e) =>
                              patchConfig(
                                "nudge_at_fraction",
                                parseFloat(e.target.value) || 0.1,
                              )
                            }
                            disabled={isRunning}
                            className="h-8 text-sm"
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* RAG config — only visible when rag mode is selected */}
                  {selectedModes.has(MODE_RAG) && (
                    <div>
                      <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                        RAG
                      </p>
                      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                        <div>
                          <FieldLabel tooltip={FIELD_TOOLTIPS.chunk_size} className="mb-1 text-xs text-muted-foreground">
                            Chunk size
                          </FieldLabel>
                          <Input
                            type="number"
                            min={100}
                            value={config.chunk_size}
                            onChange={(e) =>
                              patchConfig(
                                "chunk_size",
                                parseInt(e.target.value, 10) || 100,
                              )
                            }
                            disabled={isRunning}
                            className="h-8 text-sm"
                          />
                        </div>
                        <div>
                          <FieldLabel tooltip={FIELD_TOOLTIPS.chunk_overlap} className="mb-1 text-xs text-muted-foreground">
                            Chunk overlap
                          </FieldLabel>
                          <Input
                            type="number"
                            min={0}
                            value={config.chunk_overlap}
                            onChange={(e) =>
                              patchConfig(
                                "chunk_overlap",
                                parseInt(e.target.value, 10) || 0,
                              )
                            }
                            disabled={isRunning}
                            className="h-8 text-sm"
                          />
                        </div>
                        <div>
                          <FieldLabel tooltip={FIELD_TOOLTIPS.top_k} className="mb-1 text-xs text-muted-foreground">
                            Top-k
                          </FieldLabel>
                          <Input
                            type="number"
                            min={1}
                            value={config.top_k}
                            onChange={(e) =>
                              patchConfig("top_k", parseInt(e.target.value, 10) || 1)
                            }
                            disabled={isRunning}
                            className="h-8 text-sm"
                          />
                        </div>
                        <div>
                          <FieldLabel tooltip={FIELD_TOOLTIPS.embedding_model} className="mb-1 text-xs text-muted-foreground">
                            Embedding model
                          </FieldLabel>
                          <Input
                            type="text"
                            value={config.embedding_model}
                            onChange={(e) =>
                              patchConfig("embedding_model", e.target.value)
                            }
                            disabled={isRunning}
                            className="h-8 text-sm"
                          />
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Slot count + validation */}
            <div className="flex items-center gap-3 text-sm">
              <Badge
                variant={tooManySlots ? "destructive" : "secondary"}
                className="font-mono"
              >
                {totalSlots} slot{totalSlots === 1 ? "" : "s"} ({selectedLLMProviderIds.length} × {selectedModes.size})
              </Badge>
              {tooManySlots && (
                <span className="text-destructive">
                  Exceeds MAX_SLOTS={MAX_SLOTS}. Reduce providers or modes.
                </span>
              )}
            </div>

            {/* Run button */}
            <div className="flex items-center gap-3">
              <Button onClick={handleRun} disabled={!canRun} size="lg">
                {isRunning ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Running {totalSlots} slots…
                  </>
                ) : (
                  "Run Compare"
                )}
              </Button>
              {runError && (
                <span className="text-sm text-destructive">{runError}</span>
              )}
            </div>
          </CardContent>
        </Card>

        {/* ---------- Results panel ---------- */}
        {result && (
          <ResultsPanel
            result={result}
            rankingMetric={rankingMetric}
            onRankingMetricChange={setRankingMetric}
            expandedSlotId={expandedSlotId}
            setExpandedSlotId={setExpandedSlotId}
            judgeScores={judgeScores}
            judgeConfigured={judgeConfigured}
            isJudging={isJudging}
            onJudge={handleJudge}
          />
        )}
      </div>
    </AppShell>
  );
}

// ---------------------------------------------------------------------------
// Results panel
// ---------------------------------------------------------------------------

interface ResultsPanelProps {
  result: CompareMatrixResponse;
  rankingMetric: MatrixRankingMetric;
  onRankingMetricChange: (metric: MatrixRankingMetric) => void;
  expandedSlotId: string | null;
  setExpandedSlotId: (id: string | null) => void;
  judgeScores: Record<string, JudgeScoreData>;
  judgeConfigured: boolean;
  isJudging: boolean;
  onJudge: () => void;
}

/** Client-side ranking so re-sorting doesn't require a re-run.
 *
 * Exported for vitest coverage (AC-17 regression guard). Kept as a
 * pure function — the parent component re-runs it on every render
 * via ``useMemo``, so tests can exercise it directly with fabricated
 * slots and no DOM.
 */
export function _rankSlots(
  slots: CompareMatrixSlotResponse[],
  metric: MatrixRankingMetric,
  judgeScores: Record<string, JudgeScoreData>,
): number[] {
  const scored = slots.map((s, idx) => {
    if (_isSlotFailed(s)) return { idx, score: Infinity };
    let score: number;
    switch (metric) {
      case "cost":
        score = s.total_cost;
        break;
      case "tokens":
        score = s.total_tokens;
        break;
      case "latency":
        score = s.elapsed_seconds;
        break;
      case "judge_score": {
        // Higher is better → negate. Unjudged slots sort just above failed
        // (Infinity) but below any real score (negative values).
        const js = judgeScores[s.execution_id];
        score = js ? -js.overall_score : Infinity - 1;
        break;
      }
      case "answer_per_cost":
        // Zero-cost providers (local models) get -Infinity → always rank first
        score = s.total_cost > 0 ? -(s.answer.length / s.total_cost) : -Infinity;
        break;
      case "ttft":
        // Lower TTFT is better; slots without a value sort last.
        score = s.median_ttft_ms ?? Number.POSITIVE_INFINITY;
        break;
      case "decode_tokens_per_sec": {
        const decodeMs = s.total_decode_ms ?? 0;
        const completion = s.total_completion_tokens ?? 0;
        // Higher tok/s is better → negate. Slots without a decode
        // phase sort last (infinity).
        score =
          decodeMs > 0
            ? -(completion / (decodeMs / 1000))
            : Number.POSITIVE_INFINITY;
        break;
      }
      case "cache_hit_rate":
        // Higher cache_hit_rate is better → negate. Slots without the
        // field at all (legacy bundles, pre-Phase-5 responses) sort
        // last — matching the null-last discipline used by ttft and
        // decode_tokens_per_sec.
        score =
          s.cache_hit_rate == null
            ? Number.POSITIVE_INFINITY
            : -s.cache_hit_rate;
        break;
      default:
        score = s.total_cost;
    }
    return { idx, score };
  });
  scored.sort((a, b) => a.score - b.score);
  return scored.map((s) => s.idx);
}

function ResultsPanel({
  result,
  rankingMetric,
  onRankingMetricChange,
  expandedSlotId,
  setExpandedSlotId,
  judgeScores,
  judgeConfigured,
  isJudging,
  onJudge,
}: ResultsPanelProps) {
  // Re-rank client-side whenever the metric changes
  const ranking = useMemo(
    () => _rankSlots(result.slots, rankingMetric, judgeScores),
    [result.slots, rankingMetric, judgeScores],
  );

  const rankByIndex = useMemo(() => {
    const map = new Map<number, number>();
    ranking.forEach((slotIdx, rankPos) => {
      map.set(slotIdx, rankPos + 1);
    });
    return map;
  }, [ranking]);

  const bestSlot: CompareMatrixSlotResponse | null =
    ranking.length > 0 && result.slots[ranking[0]]?.success
      ? result.slots[ranking[0]]
      : null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Results</span>
          <span className="text-sm font-normal text-muted-foreground">
            {result.total_elapsed.toFixed(1)}s wall-clock
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Summary strip */}
        <div className="flex flex-wrap gap-4 rounded-md border bg-muted/30 p-3 text-sm">
          {bestSlot ? (
            <div className="flex items-center gap-2">
              <Trophy className="h-4 w-4 text-amber-500" />
              <span className="font-medium">Best:</span>
              <span>{bestSlot.label}</span>
            </div>
          ) : (
            <div className="text-destructive">No successful slots</div>
          )}
          <div className="flex items-center gap-2 text-muted-foreground">
            <span className="font-medium">Ranked by:</span>
            <Select
              value={rankingMetric}
              onValueChange={(v) => onRankingMetricChange(v as MatrixRankingMetric)}
            >
              <SelectTrigger className="h-7 w-48 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {VISIBLE_RANKING_METRICS.map((opt) => (
                  <SelectItem key={opt.value} value={opt.value}>
                    {opt.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div
            className="flex items-center gap-2 font-mono text-xs text-muted-foreground"
            title="Shared comparison_group_id — every slot's telemetry row references this"
          >
            <span>group:</span>
            <span>{result.comparison_group_id.slice(0, 12)}…</span>
          </div>
          {/* Auto-Judge button */}
          {judgeConfigured && (
            <Button
              variant="outline"
              size="sm"
              onClick={onJudge}
              disabled={isJudging}
              className="ml-auto h-7 text-xs"
            >
              {isJudging ? (
                <>
                  <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                  Judging…
                </>
              ) : (
                <>
                  <Sparkles className="mr-1 h-3 w-3" />
                  Auto-Judge
                </>
              )}
            </Button>
          )}
        </div>

        {/* Comparison chart */}
        <ComparisonChart
          slots={result.slots}
          metric={rankingMetric}
          ranking={ranking}
          judgeScores={judgeScores}
        />

        {/* Judge dimensions chart — shows after judging */}
        {Object.keys(judgeScores).length > 0 && (
          <JudgeDimensionsChart
            judgeScores={Object.values(judgeScores)}
            providerNames={Object.fromEntries(
              result.slots.map((s) => [s.chat_provider_id, s.label])
            )}
          />
        )}

        {/* Slot grid */}
        <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
          {result.slots.map((slot, idx) => {
            const rank = rankByIndex.get(idx) ?? null;
            const isExpanded = expandedSlotId === slot.slot_id;
            return (
              <SlotCard
                key={slot.slot_id}
                slot={slot}
                rank={rank}
                expanded={isExpanded}
                onToggle={() => setExpandedSlotId(isExpanded ? null : slot.slot_id)}
                judgeScore={judgeScores[slot.execution_id]}
              />
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Comparison chart
// ---------------------------------------------------------------------------

const BAR_COLORS = [
  "hsl(45, 93%, 47%)",   // gold (#1)
  "hsl(160, 60%, 45%)",  // emerald (#2)
  "hsl(210, 60%, 55%)",  // blue
  "hsl(280, 50%, 55%)",  // purple
  "hsl(30, 70%, 50%)",   // orange
  "hsl(340, 60%, 50%)",  // rose
  "hsl(180, 50%, 45%)",  // teal
  "hsl(0, 0%, 55%)",     // gray (failed)
];

interface ComparisonChartProps {
  slots: CompareMatrixSlotResponse[];
  metric: MatrixRankingMetric;
  ranking: number[];
  judgeScores: Record<string, JudgeScoreData>;
}

function ComparisonChart({ slots, metric, ranking, judgeScores }: ComparisonChartProps) {
  const chartData = useMemo(() => {
    return ranking.map((slotIdx) => {
      const s = slots[slotIdx];
      let value: number;
      switch (metric) {
        case "cost":
          value = s.total_cost;
          break;
        case "tokens":
          value = s.total_tokens;
          break;
        case "latency":
          value = s.elapsed_seconds;
          break;
        case "answer_per_cost":
          // Zero-cost providers (local models) get a very high ratio
          // so they show as the tallest bar. Use 10M as a visual cap.
          value = s.total_cost > 0 ? s.answer.length / s.total_cost : 10_000_000;
          break;
        case "judge_score": {
          const js = judgeScores[s.execution_id];
          value = js ? js.overall_score : 0;
          break;
        }
        // Phase 5 prefill/decode metrics. Previously fell through to
        // `default: value = s.total_cost`, which plotted COST bars
        // under the TTFT / decode / cache-hit label — visibly wrong
        // whenever one of these metrics was selected.
        case "ttft":
          value = s.median_ttft_ms ?? 0;
          break;
        case "decode_tokens_per_sec":
          value =
            (s.total_decode_ms ?? 0) > 0
              ? (s.total_completion_tokens ?? 0) / ((s.total_decode_ms ?? 0) / 1000)
              : 0;
          break;
        case "cache_hit_rate":
          value = s.cache_hit_rate ?? 0;
          break;
        default:
          value = s.total_cost;
      }
      const failed = _isSlotFailed(s);
      return {
        name: s.label,
        failed,
        value: failed ? 0 : value,
        success: !failed,
        slotIdx,
      };
    });
  }, [slots, metric, ranking, judgeScores]);

  const metricLabel = RANKING_METRICS.find((m) => m.value === metric)?.label ?? metric;
  // Metrics where a HIGHER number is the better outcome. TTFT is
  // deliberately NOT in this set — lower is better for time-to-first-
  // token. answer_per_cost and judge_score are higher-better.
  // decode_tokens_per_sec and cache_hit_rate are Phase 5 additions:
  // higher decode tok/s = faster steady-state, higher cache-hit rate
  // = more prefix reuse → both higher is better.
  const HIGHER_IS_BETTER_METRICS = new Set<MatrixRankingMetric>([
    "answer_per_cost",
    "judge_score",
    "decode_tokens_per_sec",
    "cache_hit_rate",
  ]);
  const isHigherBetter = HIGHER_IS_BETTER_METRICS.has(metric);

  if (chartData.length === 0) return null;

  // Dynamic YAxis width based on longest label (~6.5px per char at font-size 11)
  const maxLabelLen = Math.max(...chartData.map((d) => d.name.length));
  const yAxisWidth = Math.max(150, Math.min(maxLabelLen * 6.5 + 30, 350));

  return (
    <div className="rounded-md border bg-card p-4">
      <p className="mb-2 text-xs font-medium text-muted-foreground">
        {metricLabel} {isHigherBetter ? "(higher is better)" : "(lower is better)"}
      </p>
      <ResponsiveContainer width="100%" height={Math.max(120, chartData.length * 40)}>
        <BarChart data={chartData} layout="vertical" margin={{ left: 0, right: 20, top: 0, bottom: 0 }}>
          <XAxis type="number" tick={{ fontSize: 11 }} />
          <YAxis
            type="category"
            dataKey="name"
            width={yAxisWidth}
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            tick={(props: any) => {
              const entry = chartData[props.payload.index];
              const isFailed = entry?.failed;
              return (
                <text
                  x={props.x - 8}
                  y={props.y}
                  dy={4}
                  textAnchor="end"
                  fontSize={11}
                  fill={isFailed ? "hsl(0, 60%, 60%)" : "currentColor"}
                >
                  {isFailed ? `\u{1F6AB} ${props.payload.value}` : props.payload.value}
                </text>
              );
            }}
          />
          <RechartsTooltip
            formatter={(value) => {
              const v = Number(value ?? 0);
              return [
                metric === "cost"
                  ? `$${v.toFixed(4)}`
                  : metric === "latency"
                    ? `${v.toFixed(1)}s`
                    : metric === "judge_score"
                      ? `${v.toFixed(1)}/5`
                      : v.toLocaleString(),
                metricLabel,
              ];
            }}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {chartData.map((entry, idx) => (
              <Cell
                key={entry.name}
                // -1 reserves the last color (gray) for failed slots
                fill={entry.success ? BAR_COLORS[idx % (BAR_COLORS.length - 1)] : BAR_COLORS[BAR_COLORS.length - 1]}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Slot card
// ---------------------------------------------------------------------------

interface SlotCardProps {
  slot: CompareMatrixSlotResponse;
  rank: number | null;
  expanded: boolean;
  onToggle: () => void;
  judgeScore?: JudgeScoreData;
}

function SlotCard({ slot, rank, expanded, onToggle, judgeScore }: SlotCardProps) {
  const failed = _isSlotFailed(slot);
  const statusColor = failed
    ? "border-destructive"
    : rank === 1
      ? "border-amber-500"
      : "border-emerald-500";

  return (
    <div
      className={`space-y-2 rounded-md border-2 bg-card p-3 ${statusColor}`}
      data-testid={`matrix-slot-${slot.mode}-${slot.provider}`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {failed ? (
            <Ban className="h-4 w-4 text-destructive" />
          ) : rank === 1 ? (
            <Trophy className="h-4 w-4 text-amber-500" />
          ) : null}
          <span className="text-sm font-semibold">{slot.label}</span>
          <EngineBadge mode={slot.mode} />
        </div>
        {rank !== null && (
          <Badge variant={failed ? "destructive" : "outline"} className="text-xs">
            {failed ? "Failed" : `#${rank}`}
          </Badge>
        )}
      </div>

      {!failed && (
        <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
          <span className="flex items-center gap-1" title="Total tokens">
            <Hash className="h-3 w-3" />
            {slot.total_tokens.toLocaleString()}
          </span>
          <span className="flex items-center gap-1" title="Cost (USD)">
            <DollarSign className="h-3 w-3" />
            {slot.total_cost.toFixed(4)}
          </span>
          <span className="flex items-center gap-1" title="Elapsed seconds">
            <Clock className="h-3 w-3" />
            {slot.elapsed_seconds.toFixed(1)}s
          </span>
        </div>
      )}

      {/* Judge score inline display */}
      {!failed && judgeScore && judgeScore.overall_score > 0 && (
        <JudgeScores
          dimensions={judgeScore.dimensions}
          overallScore={judgeScore.overall_score}
        />
      )}

      {failed ? (
        <div className="max-h-32 overflow-auto break-words rounded bg-destructive/10 p-2 text-sm text-destructive">
          {_friendlyError(slot)}
        </div>
      ) : (
        <>
          <div className="prose prose-sm dark:prose-invert max-w-none overflow-x-auto rounded bg-muted/40 p-2">
            {expanded ? (
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {slot.answer}
              </ReactMarkdown>
            ) : (
              <div className="line-clamp-3">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {slot.answer}
                </ReactMarkdown>
              </div>
            )}
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={onToggle}
            className="w-full text-xs"
          >
            {expanded ? "Collapse" : "Expand full answer"}
          </Button>
        </>
      )}
    </div>
  );
}
