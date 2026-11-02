"""Domain entities: core business objects for RLMKit.

All entities are plain Python dataclasses with NO external dependencies.
They represent the fundamental concepts of the RLM paradigm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Query:
    """A user query to be processed by the RLM system.

    Attributes:
        content: The document or text content to analyze.
        question: The user's question about the content.
        mode: Requested execution mode.
        metadata: Arbitrary key-value metadata attached to the query.
    """

    content: str
    question: str
    mode: Literal["direct", "rag", "rlm", "auto", "compare"] = "auto"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    """The result produced by an RLM execution.

    Attributes:
        answer: Final answer text.
        mode_used: The mode that actually produced this answer.
        success: Whether execution completed without error.
        error: Error description if execution failed.
        steps: Number of execution steps taken.
        metadata: Additional response metadata.
    """

    answer: str
    mode_used: str
    success: bool = True
    error: str | None = None
    steps: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceStep:
    """A single step in the RLM execution trace.

    Attributes:
        index: Step number (0-indexed).
        action_type: Type of action taken
            (inspect, subcall, compaction, final, error).
        code: Python code executed, if any.
        output: Result of code execution, if any.
        tokens_used: Number of tokens consumed in this step (legacy sum;
            equals prompt_tokens + completion_tokens for new traces).
        timestamp: Unix timestamp when this step occurred.
        recursion_depth: Depth in the recursion tree (0 = root).
        cost: Estimated monetary cost for this step.
        duration: Wall-clock time for this step in seconds.
        model: Model identifier used for this step.
        raw_response: Full LLM response text.
        error: Error message if this step failed.
        prompt_tokens: Input-side token count (NEW; 0 on legacy traces).
        completion_tokens: Output-side token count (NEW; 0 on legacy traces).
        ttft_ms: Time-to-first-token in milliseconds (NEW; None on legacy).
        decode_ms: Decode-phase wall time in milliseconds (NEW; 0 on legacy).
        cached_tokens: Prompt tokens served from provider's prefix cache
            (NEW; 0 when provider does not report cache activity).
        cache_write_tokens: Prompt tokens written to provider's prefix cache
            (Anthropic-style; NEW; 0 elsewhere).
    """

    index: int
    action_type: Literal["inspect", "subcall", "compaction", "final", "error"]
    code: str | None = None
    output: str | None = None
    tokens_used: int = 0
    timestamp: float = 0.0
    recursion_depth: int = 0
    cost: float = 0.0
    duration: float = 0.0
    model: str | None = None
    raw_response: str | None = None
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: int | None = None
    decode_ms: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "index": self.index,
            "action_type": self.action_type,
            "code": self.code,
            "output": self.output,
            "tokens_used": self.tokens_used,
            "timestamp": self.timestamp,
            "recursion_depth": self.recursion_depth,
            "cost": self.cost,
            "duration": self.duration,
            "model": self.model,
            "raw_response": self.raw_response,
            "error": self.error,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ttft_ms": self.ttft_ms,
            "decode_ms": self.decode_ms,
            "cached_tokens": self.cached_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TraceStep:
        """Inverse of :meth:`to_dict`. Reads domain-export keys only.

        Tolerant of missing keys (defaults applied), extra keys (ignored),
        and unknown ``action_type`` values (falls back to ``"inspect"``).
        Does **not** decode raw-DTO trace dicts — that translation lives
        at the route layer (`_translate_raw_trace_entry`).
        """
        action_type = d.get("action_type", "inspect")
        if action_type not in ("inspect", "subcall", "compaction", "final", "error"):
            action_type = "inspect"
        return cls(
            index=d.get("index", 0),
            action_type=action_type,  # type: ignore[arg-type]
            code=d.get("code"),
            output=d.get("output"),
            tokens_used=d.get("tokens_used", 0),
            timestamp=d.get("timestamp", 0.0),
            recursion_depth=d.get("recursion_depth", 0),
            cost=d.get("cost", 0.0),
            duration=d.get("duration", 0.0),
            model=d.get("model"),
            raw_response=d.get("raw_response"),
            error=d.get("error"),
            prompt_tokens=d.get("prompt_tokens", 0),
            completion_tokens=d.get("completion_tokens", 0),
            ttft_ms=d.get("ttft_ms"),
            decode_ms=d.get("decode_ms", 0),
            cached_tokens=d.get("cached_tokens", 0),
            cache_write_tokens=d.get("cache_write_tokens", 0),
        )


@dataclass
class ExecutionTrace:
    """Complete execution trace for an RLM run.

    Attributes:
        steps: Ordered list of trace steps.
        start_time: Unix timestamp when execution started.
        end_time: Unix timestamp when execution finished (None if still running).
        metadata: Additional trace-level metadata.
    """

    steps: list[TraceStep] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: TraceStep) -> None:
        """Append a step to the trace."""
        self.steps.append(step)

    @property
    def total_tokens(self) -> int:
        """Sum of tokens across all steps."""
        return sum(s.tokens_used for s in self.steps)

    @property
    def total_cost(self) -> float:
        """Sum of costs across all steps."""
        return sum(s.cost for s in self.steps)

    @property
    def max_depth(self) -> int:
        """Maximum recursion depth reached."""
        if not self.steps:
            return 0
        return max(s.recursion_depth for s in self.steps)

    @property
    def total_prompt_tokens(self) -> int:
        """Sum of prompt-side tokens across all steps."""
        return sum(s.prompt_tokens for s in self.steps)

    @property
    def total_completion_tokens(self) -> int:
        """Sum of completion-side tokens across all steps."""
        return sum(s.completion_tokens for s in self.steps)

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of prompt tokens served from the provider's prefix cache.

        Returns ``0.0`` when no prompt tokens were observed. Capped at
        ``1.0`` defensively — providers can report cached > prompt in
        rare edge cases (OpenAI reuses cache across a turn; Anthropic
        counts cache_read distinct from cache_creation). The displayed
        rate is a user-facing heuristic, not an accounting number.
        """
        total = self.total_prompt_tokens
        if total <= 0:
            return 0.0
        hits = sum(s.cached_tokens for s in self.steps)
        return min(1.0, hits / total)

    @property
    def median_ttft_ms(self) -> int | None:
        """Median TTFT across steps that recorded one; ``None`` if none did."""
        vals = sorted(s.ttft_ms for s in self.steps if s.ttft_ms is not None)
        if not vals:
            return None
        return vals[len(vals) // 2]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire trace to a plain dictionary."""
        return {
            "steps": [s.to_dict() for s in self.steps],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "max_depth": self.max_depth,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "cache_hit_rate": self.cache_hit_rate,
            "median_ttft_ms": self.median_ttft_ms,
            "metadata": self.metadata,
        }


@dataclass
class BudgetConfig:
    """Immutable budget configuration for an execution.

    Attributes:
        max_steps: Maximum number of execution steps (None = unlimited).
        max_tokens: Maximum total tokens (None = unlimited).
        max_cost: Maximum cost in USD (None = unlimited).
        max_time_seconds: Maximum wall-clock time in seconds (None = unlimited).
        max_recursion_depth: Maximum recursion depth (None = unlimited).
    """

    max_steps: int | None = None
    max_tokens: int | None = None
    max_cost: float | None = None
    max_time_seconds: float | None = None
    max_recursion_depth: int | None = None


@dataclass
class BudgetState:
    """Mutable budget consumption state.

    Attributes:
        steps: Number of steps consumed.
        input_tokens: Number of input tokens consumed.
        output_tokens: Number of output tokens consumed.
        cost: Accumulated cost in USD.
        elapsed_seconds: Accumulated wall-clock time in seconds.
        recursion_depth: Current recursion depth.
    """

    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    elapsed_seconds: float = 0.0
    recursion_depth: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed (input + output)."""
        return self.input_tokens + self.output_tokens

    def is_within(self, config: BudgetConfig) -> bool:
        """Check whether the current state is within the given budget limits."""
        if config.max_steps is not None and self.steps > config.max_steps:
            return False
        if config.max_tokens is not None and self.total_tokens >= config.max_tokens:
            return False
        if config.max_cost is not None and self.cost >= config.max_cost:
            return False
        if config.max_time_seconds is not None and self.elapsed_seconds >= config.max_time_seconds:
            return False
        if (
            config.max_recursion_depth is not None
            and self.recursion_depth >= config.max_recursion_depth
        ):
            return False
        return True
