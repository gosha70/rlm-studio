# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0
"""
Data models for RLMKit UI - Chat messages, metrics, responses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4


@dataclass
class Response:
    """LLM Response from a single model."""

    content: str
    """The actual response text."""

    stop_reason: str
    """Why the model stopped: 'stop', 'length', 'error'."""

    raw_response: str | None = None
    """Raw response object for debugging."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ExecutionMetrics:
    """Metrics for single execution (RLM or Direct LLM)."""

    # Tokens
    input_tokens: int
    """Number of input tokens consumed."""

    output_tokens: int
    """Number of output tokens generated."""

    total_tokens: int
    """Total tokens (input + output)."""

    # Cost
    cost_usd: float
    """Total cost in USD for this execution."""

    cost_breakdown: dict[str, float]
    """Cost breakdown: {'input': $x, 'output': $y}."""

    # Performance
    execution_time_seconds: float
    """Total execution time in seconds."""

    steps_taken: int
    """Number of steps (0 for direct, >0 for RLM)."""

    # Memory (RLM specific)
    memory_used_mb: float
    """Current memory usage in MB."""

    memory_peak_mb: float
    """Peak memory usage in MB."""

    # Status
    success: bool
    """Whether execution was successful."""

    error: str | None = None
    """Error message if execution failed."""

    execution_type: Literal["rlm", "direct", "rag"] = "rlm"
    """Type of execution."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ComparisonMetrics:
    """Comparison metrics between RLM and Direct execution."""

    # RLM Metrics
    rlm_cost_usd: float
    """Cost of RLM execution."""

    rlm_time_seconds: float
    """Execution time for RLM."""

    rlm_tokens: int
    """Total tokens used by RLM."""

    rlm_steps: int
    """Number of RLM exploration steps."""

    # Direct Metrics
    direct_cost_usd: float
    """Cost of direct LLM execution."""

    direct_time_seconds: float
    """Execution time for direct LLM."""

    direct_tokens: int
    """Total tokens used by direct LLM."""

    direct_steps: int
    """Number of direct execution steps (always 0)."""

    # Deltas
    cost_delta_usd: float
    """Absolute cost difference (RLM - Direct)."""

    cost_delta_percent: float
    """Percentage cost difference (RLM vs Direct)."""

    time_delta_seconds: float
    """Absolute time difference (RLM - Direct)."""

    time_delta_percent: float
    """Percentage time difference (RLM vs Direct)."""

    token_delta: int
    """Absolute token difference (RLM - Direct)."""

    # Recommendation
    recommendation: str
    """Recommendation based on trade-offs."""

    reasoning: str | None = None
    """Explanation of the recommendation."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ChatMessage:
    """Single chat message with optional dual responses."""

    # Identifiers
    id: str = field(default_factory=lambda: str(uuid4()))
    """Unique message ID."""

    timestamp: datetime = field(default_factory=datetime.now)
    """When message was created."""

    # User input
    user_query: str = ""
    """The user's question/query."""

    file_context: str | None = None
    """Optional document/file content for context."""

    file_info: dict[str, Any] | None = None
    """File metadata: {'name': str, 'size_mb': float, 'tokens': int}."""

    # Execution mode
    mode: Literal["rlm_only", "direct_only", "rag_only", "compare"] = "compare"
    """Which execution path to use."""

    # RLM Response (if applicable)
    rlm_response: Response | None = None
    """Response from RLM execution."""

    rlm_metrics: ExecutionMetrics | None = None
    """Metrics from RLM execution."""

    rlm_trace: list[dict[str, Any]] | None = None
    """Step-by-step trace of RLM execution."""

    # Direct Response (if applicable)
    direct_response: Response | None = None
    """Response from direct LLM execution."""

    direct_metrics: ExecutionMetrics | None = None
    """Metrics from direct LLM execution."""

    # RAG Response (if applicable)
    rag_response: Response | None = None
    """Response from RAG execution."""

    rag_metrics: ExecutionMetrics | None = None
    """Metrics from RAG execution."""

    rag_trace: list[dict[str, Any]] | None = None
    """Retrieval trace from RAG execution."""

    # Comparison
    comparison_metrics: ComparisonMetrics | None = None
    """Metrics comparing RLM and Direct (if both were run)."""

    # Multi-provider execution (new)
    execution_results: list[ExecutionResult] | None = None
    """Results from multi-provider execution plan."""

    generalized_comparison: GeneralizedComparison | None = None
    """Generalized comparison across N execution results."""

    execution_plan: ExecutionPlan | None = None
    """The execution plan that produced these results."""

    # Status
    in_progress: bool = False
    """Whether this message is still being processed."""

    error: str | None = None
    """Error message if something went wrong."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def has_rlm_response(self) -> bool:
        """Check if RLM response exists."""
        return bool(self.rlm_response is not None and self.rlm_response.content)

    def has_direct_response(self) -> bool:
        """Check if Direct response exists."""
        return bool(self.direct_response is not None and self.direct_response.content)

    def has_rag_response(self) -> bool:
        """Check if RAG response exists."""
        return bool(self.rag_response is not None and self.rag_response.content)

    def is_complete(self) -> bool:
        """Check if all expected responses are complete."""
        if self.mode == "rlm_only":
            return self.has_rlm_response() and not self.in_progress
        elif self.mode == "direct_only":
            return self.has_direct_response() and not self.in_progress
        elif self.mode == "rag_only":
            return self.has_rag_response() and not self.in_progress
        elif self.mode == "compare":
            return self.has_rlm_response() and self.has_direct_response() and not self.in_progress
        return False


@dataclass
class RAGConfig:
    """Configuration for RAG execution parameters."""

    chunk_size: int = 1000
    """Number of characters per chunk."""

    chunk_overlap: int = 200
    """Overlap between consecutive chunks."""

    top_k: int = 5
    """Number of top chunks to retrieve."""

    embedding_model: str = "text-embedding-3-small"
    """OpenAI embedding model name."""


@dataclass
class ExecutionSlot:
    """A single execution to perform: one mode + one provider."""

    mode: Literal["rlm", "direct", "rag"]
    """Execution mode for this slot."""

    provider_name: str
    """Key into LLMConfigManager."""

    label: str = ""
    """Display label, e.g. 'RLM (GPT-4o)'. Auto-generated if empty."""

    rag_config: RAGConfig | None = None
    """RAG-specific config (only used when mode='rag')."""


@dataclass
class ExecutionPlan:
    """Describes all executions to perform for a single user message."""

    slots: list[ExecutionSlot] = field(default_factory=list)

    @property
    def is_comparison(self) -> bool:
        """True when multiple slots will produce a comparison."""
        return len(self.slots) > 1


@dataclass
class ExecutionResult:
    """Result from a single execution slot."""

    slot: ExecutionSlot
    """Which slot produced this result."""

    response: Response
    """The LLM response."""

    metrics: ExecutionMetrics
    """Execution metrics."""

    trace: list[dict[str, Any]] | None = None
    """Step-by-step trace (RLM/RAG)."""


@dataclass
class GeneralizedComparison:
    """Comparison across N execution results."""

    results: list[ExecutionResult] = field(default_factory=list)
    """All execution results being compared."""

    cheapest_label: str = ""
    """Label of the cheapest execution."""

    fastest_label: str = ""
    """Label of the fastest execution."""

    recommendation: str = ""
    """Summary recommendation."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cheapest_label": self.cheapest_label,
            "fastest_label": self.fastest_label,
            "recommendation": self.recommendation,
            "result_count": len(self.results),
        }


@dataclass
class LLMProviderConfig:
    """Configuration for a single LLM provider."""

    # Provider info
    provider: str
    """Provider name: 'openai', 'anthropic', 'ollama', 'lmstudio'."""

    enabled: bool = True
    """Whether this provider is available."""

    # Connection details
    model: str = ""
    """Model name/ID."""

    api_endpoint: str | None = None
    """Custom API endpoint (for local models)."""

    timeout_seconds: int = 30
    """Request timeout in seconds."""

    # Parameters
    temperature: float = 0.7
    """Sampling temperature (0.0-2.0)."""

    max_tokens: int = 2000
    """Maximum tokens in response."""

    top_p: float = 1.0
    """Nucleus sampling parameter."""

    # Authentication (never logged or displayed)
    api_key: str | None = None
    """API key (temporary in memory, never saved to disk)."""

    api_key_env_var: str | None = None
    """Environment variable name for API key."""

    # Pricing (for cost calculation)
    input_cost_per_1k_tokens: float = 0.0
    """Cost per 1K input tokens in USD."""

    output_cost_per_1k_tokens: float = 0.0
    """Cost per 1K output tokens in USD."""

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    """When this config was created."""

    last_used_at: datetime | None = None
    """When this provider was last used."""

    test_successful: bool = False
    """Whether last connection test succeeded."""

    def to_dict(self, include_api_key: bool = False) -> dict[str, Any]:
        """Convert to dictionary (without API key by default)."""
        data = asdict(self)
        if not include_api_key:
            data["api_key"] = None
        return data

    @property
    def is_configured(self) -> bool:
        """Check if provider has minimal configuration."""
        return bool(self.model and (self.api_key or self.api_key_env_var))

    @property
    def is_ready(self) -> bool:
        """Check if provider is ready to use."""
        return self.is_configured and self.test_successful


@dataclass
class SessionMetrics:
    """Aggregated metrics for entire session."""

    total_messages: int = 0
    """Total number of messages in session."""

    total_tokens: int = 0
    """Total tokens used in session."""

    total_cost: float = 0.0
    """Total cost in USD for session."""

    total_duration_seconds: float = 0.0
    """Total time spent in session."""

    peak_memory_mb: float = 0.0
    """Peak memory usage in MB."""

    rlm_savings_tokens: int = 0
    """Total tokens saved by using RLM."""

    rlm_savings_cost: float = 0.0
    """Total cost saved by using RLM."""

    requests_by_mode: dict[str, int] = field(
        default_factory=lambda: {
            "rlm_only": 0,
            "direct_only": 0,
            "compare": 0,
        }
    )
    """Breakdown of requests by mode."""

    avg_response_time_seconds: float = 0.0
    """Average response time per request."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
