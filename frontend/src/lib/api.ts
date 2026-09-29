/**
 * Typed API client for the RLM Studio backend.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

// Direct backend URL — bypasses the Next.js dev-server proxy for
// long-running or large-body requests (file uploads, compare matrix).
// CORS on the backend already allows http://localhost:3000.
//
// In the bundled-UI build (NEXT_PUBLIC_BUNDLE=1, set by
// `npm run build:bundle`), the static export is served by FastAPI on
// the same origin and port as the API, so we force same-origin
// relative URLs regardless of NEXT_PUBLIC_API_URL. This is what makes
// `rlm-studio studio --port 9000` work for uploads and the compare matrix
// — without this gate the bundled UI would still talk to :8000.
const BACKEND_BASE =
  process.env.NEXT_PUBLIC_BUNDLE === "1"
    ? ""
    : process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ---------------------------------------------------------------------------
// Types matching the backend Pydantic models
// ---------------------------------------------------------------------------

export type ChatMode = "auto" | "rlm" | "direct" | "rag" | "compare" | "rlm_official";

export interface ChatRequest {
  content?: string | null;
  file_id?: string | null; // deprecated: use file_ids
  file_ids?: string[] | null;
  query: string;
  mode?: ChatMode;
  provider?: string | null;
  model?: string | null;
  session_id?: string | null;
  chat_provider_id?: string | null;
}

export interface ChatResponse {
  execution_id: string;
  session_id: string;
  status: string;
  chat_provider_id?: string | null;
}

// ---------------------------------------------------------------------------
// Matrix compare — POST /api/chat/compare-matrix
// ---------------------------------------------------------------------------

export type MatrixSlotMode = "direct" | "rlm" | "rag" | "rlm_official";

export type MatrixRankingMetric =
  | "cost"
  | "tokens"
  | "latency"
  | "answer_per_cost"
  | "judge_score"
  // Phase 5 — prefill/decode-oriented ranking options.
  | "ttft"
  | "decode_tokens_per_sec"
  | "cache_hit_rate";

export interface CompareMatrixRequest {
  content?: string | null;
  file_ids?: string[] | null;
  query: string;
  chat_provider_ids: string[];
  modes: MatrixSlotMode[];
  session_id?: string | null;
  ranking_metric?: MatrixRankingMetric;
}

export interface CompareMatrixRequestV2 {
  content?: string | null;
  file_ids?: string[] | null;
  query: string;
  llm_provider_ids: string[];
  modes: MatrixSlotMode[];
  session_id?: string | null;
  ranking_metric?: MatrixRankingMetric;
  runtime_settings?: RuntimeSettings;
  budget?: BudgetConfig;
  rag_config?: RAGConfig | null;
}

export interface CompareMatrixSlotResponse {
  slot_id: string;
  label: string;
  mode: string;
  provider: string;
  model: string;
  chat_provider_id: string;
  chat_provider_name?: string | null;
  execution_id: string;
  success: boolean;
  answer: string;
  error?: string | null;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  total_cost: number;
  elapsed_seconds: number;
  steps: number;
  // Phase 5 — per-slot prefill/decode aggregates (optional so old
  // bundled fixtures keep parsing).
  total_prompt_tokens?: number;
  total_completion_tokens?: number;
  total_cached_tokens?: number;
  total_decode_ms?: number;
  median_ttft_ms?: number | null;
  cache_hit_rate?: number;
}

export interface CompareMatrixResponse {
  comparison_group_id: string;
  session_id: string;
  slots: CompareMatrixSlotResponse[];
  ranking: number[];
  ranking_metric: string;
  total_elapsed: number;
}

export interface FileUploadResponse {
  id: string;
  name: string;
  size_bytes: number;       // raw upload size
  text_size_bytes: number;  // extracted-text size; used for per-message 50 MB guard
  type: string;
  token_count: number;
  created_at: string;
}

export interface MessageMetrics {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cost_usd: number;
  elapsed_seconds: number;
  steps: number;
}

export interface SessionMessage {
  id: string;
  role: string;
  content: string;
  file_id?: string | null; // deprecated: use file_ids
  file_ids?: string[] | null;
  mode?: ChatMode | null;
  mode_used?: ChatMode | null;
  execution_id?: string | null;
  metrics?: MessageMetrics | null;
  chat_provider_id?: string | null;
  chat_provider_name?: string | null;
  timestamp: string;
}

export interface SessionSummary {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface SessionDetail {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  messages: SessionMessage[];
  conversations?: Record<string, SessionMessage[]>;
}

export interface MetricsSummary {
  total_queries: number;
  total_tokens: number;
  total_cost_usd: number;
  avg_latency_seconds: number;
  avg_token_savings_percent: number | null;
}

export interface ModeSummary {
  queries: number;
  total_tokens: number;
  total_cost_usd: number;
  avg_latency_seconds: number;
}

export interface TimelineEntry {
  timestamp: string;
  tokens: number;
  cost_usd: number;
  latency_seconds: number;
  mode: ChatMode;
  provider: string;
  chat_provider_name?: string | null;
  execution_id?: string | null;
}

export interface MetricsResponse {
  session_id: string;
  summary: MetricsSummary;
  by_mode: Partial<Record<ChatMode, ModeSummary>>;
  by_provider: Record<string, { queries: number; total_tokens: number; total_cost_usd: number; avg_latency_seconds: number }>;
  by_chat_provider?: Record<string, { queries: number; total_tokens: number; total_cost_usd: number; avg_latency_seconds: number }>;
  timeline: TimelineEntry[];
}

export interface TraceStep {
  index: number;
  action_type: string;
  code?: string | null;
  output?: string | null;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  duration_seconds: number;
  recursion_depth: number;
  model?: string | null;
  timestamp?: string | null;
  // Prefill/decode telemetry (spec v1.7 Phase 3). Defaults cover legacy
  // traces recorded before these fields existed.
  prompt_tokens?: number;
  completion_tokens?: number;
  ttft_ms?: number | null;
  decode_ms?: number;
  cached_tokens?: number;
  cache_write_tokens?: number;
}

export interface TraceResponse {
  execution_id: string;
  session_id: string;
  query: string;
  mode: ChatMode;
  status: string;
  started_at?: string | null;
  completed_at?: string | null;
  result: { answer: string; success: boolean; input_tokens?: number; output_tokens?: number; total_cost?: number };
  budget: {
    steps_used: number;
    steps_limit: number;
    tokens_used: number;
    tokens_limit: number;
    cost_used: number;
    cost_limit: number;
    max_depth_reached: number;
    depth_limit: number;
  };
  steps: TraceStep[];
  chat_provider_id?: string | null;
  chat_provider_name?: string | null;
}

export interface ModelInfo {
  name: string;
  input_cost_per_1k: number;
  output_cost_per_1k: number;
}

export interface RuntimeSettings {
  temperature: number;
  top_p: number;
  max_output_tokens: number;
  timeout_seconds: number;
}

export interface ProviderConfigEntry {
  provider: string;
  model: string;
  endpoint: string | null;
  runtime_settings: RuntimeSettings;
  enabled: boolean;
}

export interface ProviderInfo {
  name: string;
  display_name: string;
  status: string;
  models: ModelInfo[];
  default_model: string | null;
  configured: boolean;
  requires_api_key: boolean;
  default_endpoint: string | null;
  model_input_hint: string;
  masked_api_key: string | null;
}

export interface LLMProviderConfig {
  id: string;
  name: string;
  backend: string;  // "openai" | "anthropic" | "ollama" | "lmstudio"
  model: string;
  endpoint?: string | null;
  runtime_settings: RuntimeSettings;
  context_window?: number | null;  // total tokens (input + output), e.g. 8192
  status: string;  // "connected" | "configured" | "offline" | "error" | "not_configured"
  created_at?: string | null;
  updated_at?: string | null;
  // Scheduled-connection-testing fields.  Updated by both the manual
  // test route and the background connection-test thread.
  last_tested_at?: string | null;  // ISO-8601 UTC
  last_tested_by?: "manual" | "background" | null;
  consecutive_failures?: number;
}

export interface LLMProviderCreateRequest {
  name: string;
  backend: string;
  model: string;
  api_key?: string | null;
  endpoint?: string | null;
  runtime_settings?: RuntimeSettings | null;
  context_window?: number | null;
}

export interface LLMProviderUpdateRequest {
  name?: string | null;
  model?: string | null;
  api_key?: string | null;
  endpoint?: string | null;
  runtime_settings?: RuntimeSettings | null;
  context_window?: number | null;
}

export interface ProviderTestRequest {
  provider: string;
  api_key?: string | null;
  endpoint?: string | null;
  model?: string | null;
}

export interface ProviderTestResponse {
  connected: boolean;
  latency_ms?: number | null;
  model?: string | null;
  error?: string | null;
}

export interface BudgetConfig {
  max_steps: number;
  max_tokens: number;
  max_cost_usd: number;
  max_time_seconds: number;
  max_recursion_depth: number;
  repeat_limit: number;
  nudge_at_fraction: number;
}

export interface RAGConfig {
  chunk_size: number;
  chunk_overlap: number;
  top_k: number;
  embedding_model: string;
}

export interface ModeConfig {
  enabled_modes: string[];
  default_mode: string;
  rag_config: RAGConfig;
  rlm_max_steps: number;
  rlm_timeout_seconds: number;
}

// Chat Providers
export interface ChatProviderConfig {
  id: string;
  name: string;
  llm_provider_id: string;        // UUID reference to LLMProviderConfig
  llm_provider_name?: string | null; // resolved display name
  // Deprecated legacy fields (may be present in migrated configs):
  llm_provider?: string;  // old: backend key like "openai"
  llm_model?: string;     // old: model name
  profile_id?: string | null;
  profile_name?: string | null;
  execution_mode: "direct" | "rlm" | "rag" | "rlm_official";
  runtime_settings: RuntimeSettings;
  rag_config?: RAGConfig | null;
  rlm_max_steps: number;
  rlm_timeout_seconds: number;
  rlm_repeat_limit: number;
  rlm_nudge_at_fraction: number;
  created_at: string;
  updated_at: string;
}

export interface ChatProviderCreateRequest {
  name: string;
  llm_provider_id: string;  // UUID of LLMProviderConfig
  profile_id?: string | null;
  execution_mode?: "direct" | "rlm" | "rag" | "rlm_official";
  rag_config?: RAGConfig | null;
  rlm_max_steps?: number | null;
  rlm_timeout_seconds?: number | null;
  rlm_repeat_limit?: number | null;
  rlm_nudge_at_fraction?: number | null;
  num_retries?: number | null;
}

export interface ChatProviderUpdateRequest {
  name?: string | null;
  llm_model?: string | null;
  profile_id?: string | null;
  execution_mode?: "direct" | "rlm" | "rag" | "rlm_official" | null;
  runtime_settings?: RuntimeSettings | null;
  rag_config?: RAGConfig | null;
  rlm_max_steps?: number | null;
  rlm_timeout_seconds?: number | null;
  rlm_repeat_limit?: number | null;
  rlm_nudge_at_fraction?: number | null;
}

export interface AppConfig {
  active_provider: string;
  active_model: string;
  budget: BudgetConfig;
  sandbox: { type: string; docker_image: string | null };
  appearance: { theme: string; sidebar_collapsed: boolean };
  provider_configs: ProviderConfigEntry[];
  default_runtime_settings: RuntimeSettings;
  mode_config: ModeConfig;
  chat_providers: ChatProviderConfig[];
  active_profile_id?: string | null;
  judge_chat_provider_id?: string | null;
  // How often the background thread auto-tests LLM Provider connections.
  // 0 = disabled (default).  1-1440 = interval in minutes.
  connection_test_interval_minutes?: number;
}

// Evaluations
export interface ThumbRatingRequest {
  execution_id: string;
  session_id: string;
  chat_provider_id: string;
  rating: "up" | "down";
}

export interface BestPickRequest {
  session_id: string;
  winner_execution_id: string;
}

export interface JudgeRequest {
  session_id: string;
  execution_ids: string[];
  mode?: "pointwise" | "pairwise" | "both";
}

export interface JudgeScoreData {
  id: string;
  execution_id: string;
  session_id: string;
  chat_provider_id: string;
  judge_provider_id: string;
  dimensions: Record<string, number>;
  overall_score: number;
  reasoning: string;
  created_at: string;
}

export interface ProviderQualityScore {
  chat_provider_id: string;
  chat_provider_name: string;
  thumb_up: number;
  thumb_down: number;
  thumb_score: number;
  best_picks: number;
  pick_rate: number;
  judge_avg_score: number | null;
  combined_score: number;
}

export interface EvaluationSummaryResponse {
  session_id: string;
  by_chat_provider: Record<string, ProviderQualityScore>;
  recommendation: string | null;
  recommendation_reason: string;
}

export interface SessionEvaluations {
  thumb_ratings: Array<{
    id: string;
    execution_id: string;
    session_id: string;
    chat_provider_id: string;
    rating: "up" | "down";
    created_at: string;
  }>;
  best_picks: Array<{
    id: string;
    session_id: string;
    winner_execution_id: string;
    created_at: string;
  }>;
  judge_scores: JudgeScoreData[];
  judge_pairwise: Array<{
    id: string;
    session_id: string;
    execution_id_a: string;
    execution_id_b: string;
    winner: "a" | "b" | "tie";
    judge_provider_id: string;
    reasoning: string;
    created_at: string;
  }>;
}

export interface HealthResponse {
  status: string;
  version: string;
  uptime_seconds: number;
}

// Failure metrics
export type FailureCategory =
  | "timeout"
  // Phase 4 — prefill-dominated timeouts are a distinct remediation class
  // (enable prefix caching, shorten history replay).
  | "prefill_timeout"
  | "budget_exhausted"
  | "context_overflow"
  | "general_error";

export interface FailureCategorySummary {
  category: FailureCategory;
  count: number;
}

export interface FailureMetricsResponse {
  session_id: string;
  total_runs: number;
  total_failures: number;
  failure_rate: number;
  by_category: FailureCategorySummary[];
  by_provider: Record<string, FailureCategorySummary[]>;
  by_mode: Record<string, FailureCategorySummary[]>;
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`API error ${resp.status}: ${text}`);
  }
  if (resp.status === 204) return undefined as unknown as T;
  return resp.json();
}

// Health
export const getHealth = () => fetchJSON<HealthResponse>("/health");

// ---------------------------------------------------------------------------
// Diagnostics — Learn tab persistent strip
// ---------------------------------------------------------------------------

export type DiagnosticStatus = "ok" | "warn" | "error";

export interface DiagnosticCheck {
  status: DiagnosticStatus;
  message: string;
  fixUrl?: string | null;
}

export interface DiagnosticsResponse {
  backend: DiagnosticCheck;
  provider: DiagnosticCheck;
  judge: DiagnosticCheck;
  storage: DiagnosticCheck;
}

export const getDiagnostics = () =>
  fetchJSON<DiagnosticsResponse>("/api/diagnostics");

// ---------------------------------------------------------------------------
// Engines — third-party RLM engines (GET /api/engines)
// ---------------------------------------------------------------------------

export interface EngineStatus {
  available: boolean;
  /** User-facing: the version when available, otherwise why not. */
  reason: string;
  version?: string | null;
}

export interface EnginesResponse {
  rlm_official: EngineStatus;
}

export const getEngines = () => fetchJSON<EnginesResponse>("/api/engines");

// ---------------------------------------------------------------------------
// Docs — Learn tab allowlisted markdown loader
// ---------------------------------------------------------------------------

export interface DocResponse {
  slug: string;
  content: string;
}

export const getDoc = (slug: string) =>
  fetchJSON<DocResponse>(`/api/docs/${encodeURIComponent(slug)}`);

// ---------------------------------------------------------------------------
// Troubleshoot — Learn tab searchable FAQ
// ---------------------------------------------------------------------------

export type TroubleshootCategory =
  | "Setup"
  | "Provider"
  | "Compare"
  | "Judge"
  | "Budget"
  | "Runtime";

export interface TroubleshootEntry {
  id: string;
  title: string;
  symptom: string;
  cause: string;
  category: TroubleshootCategory;
  fix: string[];
  seealso: string[];
}

export interface TroubleshootResponse {
  entries: TroubleshootEntry[];
}

export const getTroubleshoot = () =>
  fetchJSON<TroubleshootResponse>("/api/docs/troubleshoot");

// ---------------------------------------------------------------------------
// Replay — Learn tab Concepts §C (V2)
// ---------------------------------------------------------------------------

export type LearnReplayStepKind =
  | "question"
  | "plan"
  | "code"
  | "result"
  | "decision"
  | "answer";

export interface LearnReplayStepDetails {
  prompt?: string;
  code?: string;
  output?: string;
}

export interface LearnReplayStepMetrics {
  tokensIn?: number;
  tokensOut?: number;
  latencyMs?: number;
  costUsd?: number;
  ttftMs?: number | null;
  decodeMs?: number;
  cachedTokens?: number;
  cacheWriteTokens?: number;
}

export interface LearnReplayStep {
  id: string;
  kind: LearnReplayStepKind;
  title: string;
  summary: string;
  details?: LearnReplayStepDetails;
  metrics?: LearnReplayStepMetrics;
}

export interface LearnReplayMetadata {
  source: "bundled" | "trace";
  executionId?: string;
  originalStepCount?: number;
  truncated?: boolean;
  // True when the source run ended in `error`. Optional; bundled
  // replays and pre-V2b fixtures omit it without incident.
  failed?: boolean;
  convertorVersion: number;
}

export interface LearnReplay {
  id: string;
  title: string;
  description: string;
  steps: LearnReplayStep[];
  metadata: LearnReplayMetadata;
}

// Bundled replays live under frontend/public/learn/replays/<id>.json so
// they ship with the static asset pipeline.
const BUNDLED_REPLAY_PATH = "/learn/replays/bundled-rlm-demo.json";

export async function getBundledReplay(): Promise<LearnReplay> {
  const resp = await fetch(BUNDLED_REPLAY_PATH);
  if (!resp.ok) {
    throw new Error(`Bundled replay fetch failed: ${resp.status}`);
  }
  return (await resp.json()) as LearnReplay;
}

// Trace-backed replay (V2b). Returns the same LearnReplay shape as
// the bundled loader; the converter on the backend enforces the
// kind-inference + truncation contract pinned in NEXT.md §3.
export const getReplay = (executionId: string) =>
  fetchJSON<LearnReplay>(`/api/replays/${encodeURIComponent(executionId)}`);

// Chat
export const submitChat = (req: ChatRequest) =>
  fetchJSON<ChatResponse>("/api/chat", { method: "POST", body: JSON.stringify(req) });

// Matrix compare — synchronous, returns the full result when every slot
// has either completed or failed.  Unlike submitChat (which is 202-accepted
// and reports back through polling), this endpoint waits for the thread
// pool to join.  Expect latency proportional to the slowest slot.
// Bypass Next.js proxy (same as file uploads) to avoid socket timeout
// on long-running matrix runs (6+ RLM slots can take several minutes).
export async function submitCompareMatrix(
  req: CompareMatrixRequest | CompareMatrixRequestV2,
): Promise<CompareMatrixResponse> {
  const resp = await fetch(`${BACKEND_BASE}/api/chat/compare-matrix`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`API error ${resp.status}: ${text}`);
  }
  return resp.json();
}

// Files
export function uploadFile(
  file: File,
  onProgress?: (pct: number) => void,
): Promise<FileUploadResponse> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BACKEND_BASE}/api/files/upload`);
    if (onProgress) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
      };
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as FileUploadResponse);
        } catch {
          reject(new Error("Invalid response from server"));
        }
      } else {
        reject(new Error(`Upload failed: ${xhr.status}`));
      }
    };
    xhr.onerror = () => reject(new Error("Network error during upload"));
    xhr.send(form);
  });
}

// Sessions
export const getSessions = (limit = 20, offset = 0) =>
  fetchJSON<SessionSummary[]>(`/api/sessions?limit=${limit}&offset=${offset}`);

export const getSession = (id: string) => fetchJSON<SessionDetail>(`/api/sessions/${id}`);

export const deleteSession = (id: string) =>
  fetchJSON<void>(`/api/sessions/${id}`, { method: "DELETE" });

export const renameSession = (id: string, name: string) =>
  fetchJSON<SessionSummary>(`/api/sessions/${id}`, {
    method: "PUT",
    body: JSON.stringify({ name }),
  });

// Metrics
export const getMetrics = (sessionId: string) =>
  fetchJSON<MetricsResponse>(`/api/metrics/${sessionId}`);

export const getFailureMetrics = (sessionId: string) =>
  fetchJSON<FailureMetricsResponse>(`/api/metrics/failures/${sessionId}`);

// Executions & Traces
export interface ExecutionSummary {
  execution_id: string;
  session_id: string;
  query: string;
  mode: ChatMode;
  status: string;
  started_at: string | null;
  completed_at: string | null;
  total_tokens: number;
  total_cost: number;
  chat_provider_id?: string | null;
  chat_provider_name?: string | null;
}

export const getExecutions = (limit = 20, chatProviderId?: string, sessionId?: string) => {
  const params = new URLSearchParams({ limit: String(limit) });
  if (chatProviderId) params.set("chat_provider_id", chatProviderId);
  if (sessionId) params.set("session_id", sessionId);
  return fetchJSON<ExecutionSummary[]>(`/api/executions?${params}`);
};

export const getTrace = (executionId: string) =>
  fetchJSON<TraceResponse>(`/api/traces/${executionId}`);

export const deleteExecution = (executionId: string) =>
  fetchJSON<void>(`/api/executions/${executionId}`, { method: "DELETE" });

export const deleteAllExecutions = () =>
  fetchJSON<void>("/api/executions", { method: "DELETE" });

// Providers
export const getProviders = () => fetchJSON<ProviderInfo[]>("/api/providers");

export const testProvider = (req: ProviderTestRequest) =>
  fetchJSON<ProviderTestResponse>("/api/providers/test", {
    method: "POST",
    body: JSON.stringify(req),
  });

export interface ProviderSaveRequest {
  api_key?: string | null;
  model?: string | null;
  endpoint?: string | null;
  runtime_settings?: RuntimeSettings | null;
  enabled?: boolean | null;
}

export interface ProviderSaveResponse {
  saved: boolean;
  provider: string;
  env_var?: string | null;
  message: string;
}

export const saveProvider = (providerName: string, req: ProviderSaveRequest) =>
  fetchJSON<ProviderSaveResponse>(`/api/providers/${providerName}`, {
    method: "PUT",
    body: JSON.stringify(req),
  });

export const getProviderModels = (name: string, endpoint?: string) =>
  fetchJSON<ModelInfo[]>(
    `/api/providers/${name}/models${endpoint ? `?endpoint=${encodeURIComponent(endpoint)}` : ""}`,
  );

// LLM Provider instances
export const getLLMProviders = () => fetchJSON<LLMProviderConfig[]>("/api/llm-providers");

export const createLLMProvider = (req: LLMProviderCreateRequest) =>
  fetchJSON<LLMProviderConfig>("/api/llm-providers", {
    method: "POST",
    body: JSON.stringify(req),
  });

export const updateLLMProvider = (id: string, req: LLMProviderUpdateRequest) =>
  fetchJSON<LLMProviderConfig>(`/api/llm-providers/${id}`, {
    method: "PUT",
    body: JSON.stringify(req),
  });

export const deleteLLMProvider = (id: string) =>
  fetchJSON<void>(`/api/llm-providers/${id}`, { method: "DELETE" });

export const testLLMProvider = (id: string) =>
  fetchJSON<ProviderTestResponse>(`/api/llm-providers/${id}/test`, { method: "POST" });

/**
 * Test an unsaved LLM Provider config. Used by the Settings form's
 * Test Connection button in both Create and Edit modes so the test
 * runs against the CURRENT form values, not the last-saved record.
 * Nothing is persisted to state.
 *
 * In Edit mode the frontend passes ``llm_provider_id`` so the
 * backend can resolve the saved API key when the form's api_key
 * field is blank — mirroring the save path's "leave blank to keep
 * current key" contract.
 */
export interface LLMProviderTestRequest extends LLMProviderCreateRequest {
  llm_provider_id?: string | null;
}

export const testLLMProviderConfig = (req: LLMProviderTestRequest) =>
  fetchJSON<ProviderTestResponse>("/api/llm-providers/test", {
    method: "POST",
    body: JSON.stringify(req),
  });

export const getLLMProviderModels = (id: string) =>
  fetchJSON<ModelInfo[]>(`/api/llm-providers/${id}/models`);

// Config

/** Fields accepted by PUT /api/config. active_provider/active_model are
 *  intentionally absent — those are set exclusively via PUT /api/providers/{name}. */
export interface ConfigUpdateRequest {
  budget?: BudgetConfig;
  sandbox?: { type: string; docker_image?: string | null };
  appearance?: { theme: string; sidebar_collapsed: boolean };
  provider_configs?: ProviderConfigEntry[];
  default_runtime_settings?: RuntimeSettings;
  mode_config?: Partial<ModeConfig>;
  chat_providers?: ChatProviderConfig[];
  judge_chat_provider_id?: string | null;
  // Only this specific field triggers a thread restart on the backend.
  // 0 = disable scheduled testing.  1-1440 = interval in minutes.
  connection_test_interval_minutes?: number;
}

export const getConfig = () => fetchJSON<AppConfig>("/api/config");

export const updateConfig = (update: ConfigUpdateRequest) =>
  fetchJSON<AppConfig>("/api/config", { method: "PUT", body: JSON.stringify(update) });

// Profiles
export interface RunProfile {
  id: string;
  name: string;
  description: string;
  strategy: string;
  default_provider: string | null;
  providers_enabled: string[];
  runtime_settings: RuntimeSettings;
  budget: BudgetConfig;
  system_prompts: Record<string, string>;
  prompt_template_name: string | null;
  is_builtin: boolean;
}

export interface RunProfileCreate {
  name: string;
  description?: string;
  strategy?: string;
  default_provider?: string | null;
  providers_enabled?: string[];
  runtime_settings?: RuntimeSettings;
  budget?: BudgetConfig;
  system_prompts?: Record<string, string>;
  prompt_template_name?: string | null;
}

export const getProfiles = () => fetchJSON<RunProfile[]>("/api/profiles");

export const createProfile = (req: RunProfileCreate) =>
  fetchJSON<RunProfile>("/api/profiles", { method: "POST", body: JSON.stringify(req) });

export const updateProfile = (id: string, req: Partial<RunProfileCreate>) =>
  fetchJSON<RunProfile>(`/api/profiles/${id}`, { method: "PUT", body: JSON.stringify(req) });

export const deleteProfile = (id: string) =>
  fetchJSON<void>(`/api/profiles/${id}`, { method: "DELETE" });

// System Prompts
export interface SystemPrompts {
  direct: string;
  rlm: string;
  rag: string;
}

export interface SystemPromptTemplate {
  name: string;
  description: string;
  prompts: Record<string, string>;
}

export const getSystemPrompts = () => fetchJSON<SystemPrompts>("/api/system-prompts");

export const updateSystemPrompts = (prompts: SystemPrompts) =>
  fetchJSON<SystemPrompts>("/api/system-prompts", { method: "PUT", body: JSON.stringify(prompts) });

export const getPromptTemplates = () =>
  fetchJSON<SystemPromptTemplate[]>("/api/system-prompts/templates");

// Chat Providers CRUD
export const getChatProviders = () => fetchJSON<ChatProviderConfig[]>("/api/chat-providers");

export type ChatProviderSaveResponse = ChatProviderConfig & {
  context_window_warning?: string;
};

export const createChatProvider = (req: ChatProviderCreateRequest) =>
  fetchJSON<ChatProviderSaveResponse>("/api/chat-providers", {
    method: "POST",
    body: JSON.stringify(req),
  });

export const updateChatProvider = (id: string, req: ChatProviderUpdateRequest) =>
  fetchJSON<ChatProviderSaveResponse>(`/api/chat-providers/${id}`, {
    method: "PUT",
    body: JSON.stringify(req),
  });

export const deleteChatProvider = (id: string) =>
  fetchJSON<void>(`/api/chat-providers/${id}`, { method: "DELETE" });

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

export interface WSMessage {
  type: string;
  id?: string;
  data?: unknown;
  session_id?: string;
  mode?: ChatMode;
}

// Evaluations
export const submitThumbRating = (req: ThumbRatingRequest) =>
  fetchJSON<{ status: string }>("/api/evaluations/thumb", {
    method: "POST",
    body: JSON.stringify(req),
  });

export const removeThumbRating = (executionId: string) =>
  fetchJSON<{ status: string }>(`/api/evaluations/thumb/${executionId}`, {
    method: "DELETE",
  });

export const submitBestPick = (req: BestPickRequest) =>
  fetchJSON<{ status: string }>("/api/evaluations/best-pick", {
    method: "POST",
    body: JSON.stringify(req),
  });

export const getSessionEvaluations = (sessionId: string) =>
  fetchJSON<SessionEvaluations>(`/api/evaluations/${sessionId}`);

export const resetSessionEvaluations = (sessionId: string) =>
  fetchJSON<undefined>(`/api/evaluations/${sessionId}`, { method: "DELETE" });

export const getEvaluationSummary = (sessionId: string) =>
  fetchJSON<EvaluationSummaryResponse>(`/api/evaluations/${sessionId}/summary`);

// Always route through the Next.js proxy (relative URL, not API_BASE) so the
// extended JUDGE_TIMEOUT_SECONDS applies even when NEXT_PUBLIC_API_URL is set.
export async function triggerJudge(
  req: JudgeRequest,
): Promise<{ pointwise: JudgeScoreData[]; pairwise: unknown[] }> {
  const resp = await fetch("/api/evaluations/judge", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`API error ${resp.status}: ${text}`);
  }
  return resp.json();
}
