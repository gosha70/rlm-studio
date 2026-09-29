"""Pydantic v2 request/response models for the RLMKit API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from rlmstudio.application.sandbox_vars import (
    MODE_AUTO,
    MODE_DIRECT,
    RLM_DEFAULT_MAX_COST_USD,
    RLM_DEFAULT_MAX_RECURSION_DEPTH,
    RLM_DEFAULT_MAX_STEPS,
    RLM_DEFAULT_MAX_TOKENS_BUDGET,
    RLM_DEFAULT_NUDGE_AT_FRACTION,
    RLM_DEFAULT_REPEAT_LIMIT,
    RLM_DEFAULT_WALL_CLOCK_BUDGET_SECONDS,
    RUNTIME_DEFAULT_MAX_OUTPUT_TOKENS,
    RUNTIME_DEFAULT_TEMPERATURE,
    RUNTIME_DEFAULT_TIMEOUT_SECONDS,
    RUNTIME_DEFAULT_TOP_P,
    ChatMode,
    ExecutionMode,
)

# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    uptime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Diagnostics — Learn tab persistent strip
# ---------------------------------------------------------------------------


class DiagnosticCheck(BaseModel):
    """Single diagnostic result.

    status semantics follow the Learn pitch (§Diagnostics red-dot badge):
    ``error`` = hard blocker (backend unreachable, no enabled provider,
    storage unwritable); ``warn`` = soft issue (judge missing); ``ok``
    = check passed. Only ``error`` states should drive the sidebar badge.
    """

    status: Literal["ok", "warn", "error"]
    message: str
    # The JSON surface is camelCase ``fixUrl`` because the TypeScript
    # ``DiagnosticCheck`` type on the frontend expects that key. Ruff
    # N815 flags the Python-side mixedCase; suppressed here because the
    # alternative — Pydantic alias + FastAPI `by_alias=True` — would
    # ripple through every response_model and is heavier than the win.
    fixUrl: str | None = None  # noqa: N815


class DiagnosticsResponse(BaseModel):
    backend: DiagnosticCheck
    provider: DiagnosticCheck
    judge: DiagnosticCheck
    storage: DiagnosticCheck


# ---------------------------------------------------------------------------
# Engines — third-party RLM engines (``GET /api/engines``)
# ---------------------------------------------------------------------------


class EngineStatus(BaseModel):
    """Availability of one third-party engine in this process."""

    available: bool
    reason: str  # user-facing: the version when available, otherwise why not
    version: str | None = None


class EnginesResponse(BaseModel):
    rlm_official: EngineStatus


# ---------------------------------------------------------------------------
# Docs — Learn tab allowlisted markdown loader
# ---------------------------------------------------------------------------


class DocResponse(BaseModel):
    """Allowlisted markdown document served to the Learn client."""

    slug: str
    content: str


class TroubleshootEntry(BaseModel):
    """Single troubleshoot entry, shape matches docs/troubleshoot.yaml.

    See spec-learn-tab-ux.md §5 Troubleshoot Entry Shape.

    ``fix`` is required — the documented schema treats it as the
    actionable remediation list, so an entry without one is not a
    valid row. ``seealso`` is genuinely optional.
    """

    id: str
    title: str
    symptom: str
    cause: str
    category: Literal["Setup", "Provider", "Compare", "Judge", "Budget", "Runtime"]
    fix: list[str]
    seealso: list[str] = Field(default_factory=list)


class TroubleshootResponse(BaseModel):
    """Wrapper around the YAML-sourced Troubleshoot entries list.

    Returned as JSON so the Learn client can filter/render without
    parsing YAML on the browser side.
    """

    entries: list[TroubleshootEntry]


# ---------------------------------------------------------------------------
# Replay — Learn tab Concepts §C (V2b trace-backed)
# ---------------------------------------------------------------------------
#
# Mirrors the TypeScript shape shipped in V2a
# (frontend/src/lib/api.ts). JSON surface is camelCase because the
# client's `LearnReplay` type uses that; field names carry
# `# noqa: N815` the same way DiagnosticCheck.fixUrl does to keep
# the two sides identical without Pydantic alias machinery.


LearnReplayStepKind = Literal["question", "plan", "code", "result", "decision", "answer"]


class LearnReplayStepDetails(BaseModel):
    """Optional deep-content payload surfaced by the Advanced tray."""

    prompt: str | None = None
    code: str | None = None
    output: str | None = None


class LearnReplayStepMetrics(BaseModel):
    """Optional per-step metrics (tokens, latency, cost).

    The four `ttft*/decode*/cache*` fields (spec v1.7 Phase 3) surface
    on the Learn tab's advanced-details tray when populated. Optional
    with `None` defaults so legacy bundled replays keep working.
    """

    tokensIn: int | None = None  # noqa: N815
    tokensOut: int | None = None  # noqa: N815
    latencyMs: int | None = None  # noqa: N815
    costUsd: float | None = None  # noqa: N815
    ttftMs: int | None = None  # noqa: N815
    decodeMs: int | None = None  # noqa: N815
    cachedTokens: int | None = None  # noqa: N815
    cacheWriteTokens: int | None = None  # noqa: N815


class LearnReplayStep(BaseModel):
    """Single replay step rendered in the Concepts §C walkthrough."""

    id: str
    kind: LearnReplayStepKind
    title: str
    summary: str
    details: LearnReplayStepDetails | None = None
    metrics: LearnReplayStepMetrics | None = None


class LearnReplayMetadata(BaseModel):
    """Provenance and display-time flags for a LearnReplay.

    ``failed`` is optional / default ``False`` so bundled replays
    and pre-V2b fixtures omit it without incident. ``truncated`` +
    ``originalStepCount`` pair together for the walkthrough's
    "showing X of N steps" copy. ``convertorVersion`` bumps when
    the trace → replay inference rules change so clients can
    detect stale bundled assets.
    """

    source: Literal["bundled", "trace"]
    executionId: str | None = None  # noqa: N815 — only set for source="trace"
    originalStepCount: int | None = None  # noqa: N815
    truncated: bool | None = None
    failed: bool | None = None
    convertorVersion: int  # noqa: N815


class LearnReplay(BaseModel):
    """The full replay payload served by GET /api/replays/{id}.

    See ``doc_internal/specs/learn-tab/NEXT.md`` §3a–§3f for the
    contract. Returned bare (no wrapper) from the endpoint to
    mirror other single-resource GETs (traces, docs).
    """

    id: str
    title: str
    description: str
    steps: list[LearnReplayStep]
    metadata: LearnReplayMetadata


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    content: str | None = None
    file_id: str | None = None  # deprecated: use file_ids
    file_ids: list[str] | None = None
    query: str
    mode: ChatMode = MODE_AUTO
    provider: str | None = None
    model: str | None = None
    session_id: str | None = None
    chat_provider_id: str | None = None  # NEW: reference a Chat Provider
    num_retries: int | None = Field(default=None, ge=0)  # override retry count (0 = no retries)

    @model_validator(mode="before")
    @classmethod
    def _coerce_file_ids(cls, data: Any) -> Any:
        """Promote legacy file_id to file_ids list for backward compat."""
        if isinstance(data, dict):
            if not data.get("file_ids") and data.get("file_id"):
                data = dict(data, file_ids=[data["file_id"]])
        return data


class ChatResponse(BaseModel):
    execution_id: str
    session_id: str
    status: str = "running"
    chat_provider_id: str | None = None  # NEW: echo back the Chat Provider used


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


class FileUploadResponse(BaseModel):
    id: str
    name: str
    size_bytes: int  # raw upload size (bytes)
    text_size_bytes: int  # extracted-text size; matches what the per-message 50 MB guard measures
    type: str
    token_count: int = 0
    created_at: datetime


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class MessageMetrics(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    elapsed_seconds: float = 0.0
    steps: int = 0


class SessionMessage(BaseModel):
    id: str
    role: str
    content: str
    file_id: str | None = None  # deprecated: use file_ids
    file_ids: list[str] | None = None
    mode: str | None = None
    mode_used: str | None = None
    execution_id: str | None = None
    metrics: MessageMetrics | None = None
    chat_provider_id: str | None = None  # NEW: which Chat Provider generated this
    chat_provider_name: str | None = None  # NEW: display name for convenience
    timestamp: datetime


class SessionSummary(BaseModel):
    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class SessionRenameRequest(BaseModel):
    name: str


class SessionDetail(BaseModel):
    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    messages: list[SessionMessage] = Field(default_factory=list)
    conversations: dict[str, list[SessionMessage]] = Field(
        default_factory=dict
    )  # NEW: keyed by chat_provider_id


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class ModeSummary(BaseModel):
    queries: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_seconds: float = 0.0


class ProviderSummary(BaseModel):
    queries: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_seconds: float = 0.0


class MetricsSummary(BaseModel):
    total_queries: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_seconds: float = 0.0
    avg_token_savings_percent: float | None = None


class TimelineEntry(BaseModel):
    timestamp: datetime
    tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    mode: str = ""
    provider: str = ""
    chat_provider_name: str | None = None
    execution_id: str | None = None


class MetricsResponse(BaseModel):
    session_id: str
    summary: MetricsSummary = Field(default_factory=MetricsSummary)
    by_mode: dict[str, ModeSummary] = Field(default_factory=dict)
    by_provider: dict[str, ProviderSummary] = Field(default_factory=dict)
    by_chat_provider: dict[str, ProviderSummary] = Field(default_factory=dict)
    timeline: list[TimelineEntry] = Field(default_factory=list)


class FailureCategorySummary(BaseModel):
    # Keep in sync with OutcomeCategory enum in outcome_classifier.py
    category: Literal[
        "timeout",
        "prefill_timeout",
        "budget_exhausted",
        "context_overflow",
        "general_error",
    ]
    count: int = 0


class FailureMetricsResponse(BaseModel):
    session_id: str
    total_runs: int = 0
    total_failures: int = 0
    failure_rate: float = 0.0
    by_category: list[FailureCategorySummary] = Field(default_factory=list)
    by_provider: dict[str, list[FailureCategorySummary]] = Field(default_factory=dict)
    by_mode: dict[str, list[FailureCategorySummary]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------


class TraceStep(BaseModel):
    index: int
    action_type: str = "inspect"
    code: str | None = None
    output: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    recursion_depth: int = 0
    model: str | None = None
    timestamp: datetime | None = None
    # Prefill/decode telemetry (spec v1.7). Defaults cover legacy traces
    # recorded before Phase 3.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: int | None = None
    decode_ms: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0


class TraceBudget(BaseModel):
    steps_used: int = 0
    steps_limit: int = 16
    tokens_used: int = 0
    tokens_limit: int = 50000
    cost_used: float = 0.0
    cost_limit: float = 2.0
    max_depth_reached: int = 0
    depth_limit: int = 5


class TraceResult(BaseModel):
    answer: str = ""
    success: bool = True
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0


class TraceResponse(BaseModel):
    execution_id: str
    session_id: str
    query: str
    mode: str
    status: str = "complete"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: TraceResult = Field(default_factory=TraceResult)
    budget: TraceBudget = Field(default_factory=TraceBudget)
    steps: list[TraceStep] = Field(default_factory=list)
    chat_provider_id: str | None = None
    chat_provider_name: str | None = None


# ---------------------------------------------------------------------------
# Runtime Settings
# ---------------------------------------------------------------------------


class RuntimeSettings(BaseModel):
    temperature: float = RUNTIME_DEFAULT_TEMPERATURE
    top_p: float = RUNTIME_DEFAULT_TOP_P
    max_output_tokens: int = RUNTIME_DEFAULT_MAX_OUTPUT_TOKENS
    timeout_seconds: int = RUNTIME_DEFAULT_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class ModelPricing(BaseModel):
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0


class ModelInfo(BaseModel):
    name: str
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0


class ProviderInfo(BaseModel):
    name: str
    display_name: str
    status: str = "not_configured"
    models: list[ModelInfo] = Field(default_factory=list)
    default_model: str | None = None
    configured: bool = False
    requires_api_key: bool = True
    default_endpoint: str | None = None
    model_input_hint: str = ""
    masked_api_key: str | None = None


class ProviderConfig(BaseModel):
    """Per-provider configuration: which provider/model to use with what settings."""

    provider: str
    model: str
    endpoint: str | None = None
    runtime_settings: RuntimeSettings = Field(default_factory=RuntimeSettings)
    enabled: bool = False
    last_tested_status: str | None = None


class ProviderTestRequest(BaseModel):
    provider: str
    api_key: str | None = None
    endpoint: str | None = None
    model: str | None = None


class ProviderSaveRequest(BaseModel):
    api_key: str | None = None
    model: str | None = None
    endpoint: str | None = None
    runtime_settings: RuntimeSettings | None = None
    enabled: bool | None = None


class ProviderSaveResponse(BaseModel):
    saved: bool
    provider: str
    env_var: str | None = None
    message: str = ""


class ProviderTestResponse(BaseModel):
    connected: bool
    latency_ms: int | None = None
    model: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Named LLM Provider instances
# ---------------------------------------------------------------------------


class LLMProviderConfig(BaseModel):
    """A named, user-created instance of an LLM backend with full connection config."""

    id: str
    name: str  # User-defined, e.g. "Ollama-llama3.2", "GPT4-mini"
    backend: str  # Catalog key: "openai", "anthropic", "ollama", "lmstudio"
    model: str  # e.g. "gpt-4o-mini", "llama3.2"
    endpoint: str | None = None  # Custom endpoint (overrides catalog default for local providers)
    runtime_settings: RuntimeSettings = Field(default_factory=RuntimeSettings)
    # Model context window in tokens.  Auto-discovered during health check or
    # set manually.  When present, Profile max_output_tokens will be clamped to
    # (context_window - prompt_tokens) at runtime so vLLM/Ollama never rejects.
    context_window: int | None = None  # total tokens (input + output), e.g. 8192
    # "connected" | "configured" | "offline" | "error" | "not_configured"
    # "error" is distinct from "offline": error = unhandled exception, offline =
    # graceful failure (timeout, 4xx/5xx, auth).  Same "not usable" UX semantics.
    status: str = "not_configured"
    created_at: datetime | None = None
    updated_at: datetime | None = None
    # Scheduled-connection-testing fields (spec: doc_internal/specs/
    # scheduled-connection-testing.md).  Updated by both the manual test route
    # and the background connection-test thread.
    last_tested_at: datetime | None = None  # ISO-8601 UTC
    last_tested_by: Literal["manual", "background"] | None = None
    # Consecutive failure counter (connected → offline requires N≥2 to debounce
    # transient blips).  Persisted across restarts so a blip just before
    # restart does not get erased.  See §Failure Semantics and Open Q #7 of
    # the spec for the full rationale.
    consecutive_failures: int = 0


class LLMProviderCreateRequest(BaseModel):
    name: str
    backend: str  # catalog key: "openai", "anthropic", "ollama", "lmstudio"
    model: str
    api_key: str | None = None  # persisted to SecretStore, not stored in config
    endpoint: str | None = None
    runtime_settings: RuntimeSettings | None = None
    context_window: int | None = None  # manual override; auto-discovered if omitted


class LLMProviderUpdateRequest(BaseModel):
    name: str | None = None
    model: str | None = None
    api_key: str | None = None
    endpoint: str | None = None
    runtime_settings: RuntimeSettings | None = None
    context_window: int | None = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class BudgetConfig(BaseModel):
    max_steps: int = RLM_DEFAULT_MAX_STEPS
    max_tokens: int = RLM_DEFAULT_MAX_TOKENS_BUDGET
    max_cost_usd: float = RLM_DEFAULT_MAX_COST_USD
    max_time_seconds: int = RLM_DEFAULT_WALL_CLOCK_BUDGET_SECONDS
    max_recursion_depth: int = RLM_DEFAULT_MAX_RECURSION_DEPTH
    repeat_limit: int = RLM_DEFAULT_REPEAT_LIMIT
    nudge_at_fraction: float = RLM_DEFAULT_NUDGE_AT_FRACTION


class SandboxConfig(BaseModel):
    type: str = "restricted"
    docker_image: str | None = None


class AppearanceConfig(BaseModel):
    theme: str = "system"
    sidebar_collapsed: bool = False


class RAGConfig(BaseModel):
    """RAG-specific execution settings."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    embedding_model: str = "text-embedding-3-small"


class ModeConfig(BaseModel):
    """Mode selection and per-mode settings."""

    enabled_modes: list[str] = Field(default_factory=lambda: ["direct", "rlm"])
    default_mode: str = MODE_AUTO
    rag_config: RAGConfig = Field(default_factory=RAGConfig)
    rlm_max_steps: int = RLM_DEFAULT_MAX_STEPS
    rlm_timeout_seconds: int = RLM_DEFAULT_WALL_CLOCK_BUDGET_SECONDS


# ---------------------------------------------------------------------------
# Chat Providers
# ---------------------------------------------------------------------------


class ChatProviderConfig(BaseModel):
    """A named combination of LLM Provider + Execution Mode + Profile."""

    id: str
    name: str  # e.g. "DIRECT-CLAUDE", "RLM-OPENAI"
    llm_provider_id: str = ""  # UUID reference to LLMProviderConfig; empty = not yet migrated
    llm_provider_name: str | None = None  # resolved display name (read-only, populated on GET)
    # Deprecated legacy fields kept for backward-compat migration; cleared after migration
    llm_provider: str = ""  # deprecated: backend key like "anthropic"
    llm_model: str = ""  # deprecated: model name like "claude-sonnet-4-6"
    profile_id: str | None = None  # reference to a RunProfile
    profile_name: str | None = None  # resolved profile name (read-only)
    execution_mode: ExecutionMode = MODE_DIRECT
    runtime_settings: RuntimeSettings = Field(default_factory=RuntimeSettings)
    rag_config: RAGConfig | None = None  # only for RAG mode
    rlm_max_steps: int = RLM_DEFAULT_MAX_STEPS
    rlm_timeout_seconds: int = RLM_DEFAULT_WALL_CLOCK_BUDGET_SECONDS
    rlm_repeat_limit: int = RLM_DEFAULT_REPEAT_LIMIT
    rlm_nudge_at_fraction: float = RLM_DEFAULT_NUDGE_AT_FRACTION
    num_retries: int | None = Field(default=None, ge=0)
    # Per-provider conversation history (SDD_RLM_CONVERSATION_HISTORY).
    # When True (default), follow-up turns on this provider carry prior
    # context — either as an in-prompt replay (Direct/Compare) or as a
    # bound `history` REPL variable (RLM/RAG/Auto).  Set to False for a
    # stateless provider whose every turn starts from scratch.
    conversation_memory_enabled: bool = True
    # Fraction of the model's context window the in-prompt replay is
    # allowed to spend on prior turns before the budget squeezes.  Only
    # affects Direct/Compare modes; the REPL path is byte-capped, not
    # token-capped.  Accepts [0.0, 0.9].
    conversation_memory_fraction: float = 0.30
    created_at: datetime | None = None
    updated_at: datetime | None = None
    ephemeral: bool = False  # True for temporary CPs created by compare-matrix

    @field_validator("conversation_memory_fraction", mode="before")
    @classmethod
    def _reject_bool_conversation_memory_fraction(cls, v: Any) -> Any:
        # bool is a subclass of int is a subclass of float, so Pydantic
        # would otherwise coerce True/False to 1.0/0.0 silently.
        if isinstance(v, bool):
            raise ValueError("conversation_memory_fraction must be a number, not a bool")
        return v

    @field_validator("conversation_memory_fraction")
    @classmethod
    def _validate_conversation_memory_fraction(cls, v: float) -> float:
        if not 0.0 <= v <= 0.9:
            raise ValueError("conversation_memory_fraction must be between 0.0 and 0.9")
        return v


class ChatProviderCreateRequest(BaseModel):
    name: str
    llm_provider_id: str  # UUID of the LLMProviderConfig to use
    profile_id: str | None = None
    execution_mode: ExecutionMode = MODE_DIRECT
    rag_config: RAGConfig | None = None
    rlm_max_steps: int | None = None
    rlm_timeout_seconds: int | None = None
    rlm_repeat_limit: int | None = None
    rlm_nudge_at_fraction: float | None = None
    num_retries: int | None = Field(default=None, ge=0)
    conversation_memory_enabled: bool | None = None
    conversation_memory_fraction: float | None = None

    @field_validator("conversation_memory_fraction", mode="before")
    @classmethod
    def _reject_bool_conversation_memory_fraction(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("conversation_memory_fraction must be a number, not a bool")
        return v

    @field_validator("conversation_memory_fraction")
    @classmethod
    def _validate_conversation_memory_fraction(cls, v: float | None) -> float | None:
        if v is not None and not 0.0 <= v <= 0.9:
            raise ValueError("conversation_memory_fraction must be between 0.0 and 0.9")
        return v


class ChatProviderUpdateRequest(BaseModel):
    name: str | None = None
    llm_provider_id: str | None = None  # switch to a different LLM Provider
    profile_id: str | None = None
    execution_mode: ExecutionMode | None = None
    rag_config: RAGConfig | None = None
    rlm_max_steps: int | None = None
    rlm_timeout_seconds: int | None = None
    rlm_repeat_limit: int | None = None
    rlm_nudge_at_fraction: float | None = None
    num_retries: int | None = Field(default=None, ge=0)
    conversation_memory_enabled: bool | None = None
    conversation_memory_fraction: float | None = None

    @field_validator("conversation_memory_fraction", mode="before")
    @classmethod
    def _reject_bool_conversation_memory_fraction(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("conversation_memory_fraction must be a number, not a bool")
        return v

    @field_validator("conversation_memory_fraction")
    @classmethod
    def _validate_conversation_memory_fraction(cls, v: float | None) -> float | None:
        if v is not None and not 0.0 <= v <= 0.9:
            raise ValueError("conversation_memory_fraction must be between 0.0 and 0.9")
        return v


class ConfigResponse(BaseModel):
    active_provider: str = "openai"
    active_model: str = "gpt-4o"
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    appearance: AppearanceConfig = Field(default_factory=AppearanceConfig)
    provider_configs: list[ProviderConfig] = Field(default_factory=list)
    llm_providers: list[LLMProviderConfig] = Field(default_factory=list)
    default_runtime_settings: RuntimeSettings = Field(default_factory=RuntimeSettings)
    mode_config: ModeConfig = Field(default_factory=ModeConfig)
    chat_providers: list[ChatProviderConfig] = Field(default_factory=list)
    active_profile_id: str | None = None
    trajectory_dir: str | None = None
    judge_chat_provider_id: str | None = None
    # How often the background thread should auto-test LLM Provider connections.
    # 0 = disabled (default, no thread runs).  1-1440 = interval in minutes.
    # See doc_internal/specs/scheduled-connection-testing.md.
    connection_test_interval_minutes: int = Field(default=0, ge=0, le=1440)


class ConfigUpdateRequest(BaseModel):
    # Note: active_provider/active_model are set exclusively via
    # PUT /api/providers/{name} to prevent dual-save-path corruption.
    budget: BudgetConfig | None = None
    sandbox: SandboxConfig | None = None
    appearance: AppearanceConfig | None = None
    provider_configs: list[ProviderConfig] | None = None
    llm_providers: list[LLMProviderConfig] | None = None
    default_runtime_settings: RuntimeSettings | None = None
    mode_config: ModeConfig | None = None
    chat_providers: list[ChatProviderConfig] | None = None
    judge_chat_provider_id: str | None = None
    # Only this field triggers a background-thread restart in the config
    # PUT handler; other config changes do not churn the thread.
    connection_test_interval_minutes: int | None = Field(default=None, ge=0, le=1440)


# ---------------------------------------------------------------------------
# Run Profiles
# ---------------------------------------------------------------------------


class RunProfile(BaseModel):
    """A named experiment configuration preset."""

    id: str
    name: str
    description: str = ""
    strategy: str = MODE_DIRECT
    default_provider: str | None = None
    providers_enabled: list[str] = Field(default_factory=list)
    runtime_settings: RuntimeSettings = Field(default_factory=RuntimeSettings)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    system_prompts: dict[str, str] = Field(default_factory=dict)
    prompt_template_name: str | None = None
    is_builtin: bool = False


class RunProfileCreate(BaseModel):
    """Request to create a new profile."""

    name: str
    description: str = ""
    strategy: str = MODE_DIRECT
    default_provider: str | None = None
    providers_enabled: list[str] = Field(default_factory=list)
    runtime_settings: RuntimeSettings = Field(default_factory=RuntimeSettings)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    system_prompts: dict[str, str] = Field(default_factory=dict)
    prompt_template_name: str | None = None


class RunProfileUpdate(BaseModel):
    """Request to update an existing profile (all fields optional)."""

    name: str | None = None
    description: str | None = None
    strategy: str | None = None
    default_provider: str | None = None
    providers_enabled: list[str] | None = None
    runtime_settings: RuntimeSettings | None = None
    budget: BudgetConfig | None = None
    system_prompts: dict[str, str] | None = None
    prompt_template_name: str | None = None


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------


class SystemPrompts(BaseModel):
    """Per-mode system prompts."""

    direct: str = ""
    rlm: str = ""
    rag: str = ""


class SystemPromptTemplate(BaseModel):
    """A named system prompt template."""

    name: str
    description: str = ""
    prompts: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# WebSocket events
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Evaluations
# ---------------------------------------------------------------------------


class ThumbRatingRequest(BaseModel):
    execution_id: str
    session_id: str
    chat_provider_id: str
    rating: Literal["up", "down"]


class BestPickRequest(BaseModel):
    session_id: str
    winner_execution_id: str


class JudgeRequest(BaseModel):
    session_id: str
    execution_ids: list[str]
    mode: Literal["pointwise", "pairwise", "both"] = "both"


class JudgeScore(BaseModel):
    id: str
    execution_id: str
    session_id: str
    chat_provider_id: str
    judge_provider_id: str
    dimensions: dict[str, float]
    overall_score: float
    reasoning: str
    created_at: datetime


class JudgePairwise(BaseModel):
    id: str
    session_id: str
    execution_id_a: str
    execution_id_b: str
    winner: Literal["a", "b", "tie"]
    judge_provider_id: str
    reasoning: str
    created_at: datetime


class ProviderQualityScore(BaseModel):
    chat_provider_id: str
    chat_provider_name: str
    thumb_up: int = 0
    thumb_down: int = 0
    thumb_score: float = 0.0
    best_picks: int = 0
    pick_rate: float = 0.0
    judge_avg_score: float | None = None
    combined_score: float = 0.0


class EvaluationSummaryResponse(BaseModel):
    session_id: str
    by_chat_provider: dict[str, ProviderQualityScore]
    recommendation: str | None = None
    recommendation_reason: str = ""


class WSTokenEvent(BaseModel):
    type: str = "token"
    id: str
    data: str
    mode: str = ""


class WSStepEvent(BaseModel):
    type: str = "step"
    id: str
    data: TraceStep


class WSMetricsEvent(BaseModel):
    type: str = "metrics"
    id: str
    data: dict[str, Any]


class WSCompleteEvent(BaseModel):
    type: str = "complete"
    id: str
    data: dict[str, Any]


class WSErrorEvent(BaseModel):
    type: str = "error"
    id: str
    data: dict[str, Any]
