# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""Core RLM functionality."""

from .budget import (
    BudgetLimits,
    BudgetTracker,
    CostTracker,
    TokenUsage,
    estimate_tokens,
)
from .comparison import (
    ComparisonResult,
    ExecutionMetrics,
)
from .errors import BudgetExceeded, ExecutionError, RLMError, SecurityError
from .model_config import (
    ModelConfig,
    ModelMetrics,
)
from .parsing import (
    ParsedResponse,
    extract_final_answer,
    extract_final_var,
    extract_python_code,
    format_code_for_display,
    format_result_for_llm,
    parse_response,
)
from .rlm import RLM, LLMClient, RLMResult
from .trace import (
    ActionType,
    ExecutionTrace,
    TraceStep,
    load_trace_from_json,
    load_trace_from_jsonl,
)

__all__ = [
    # RLM controller
    "RLM",
    "RLMResult",
    "LLMClient",
    # Errors
    "RLMError",
    "BudgetExceeded",
    "ExecutionError",
    "SecurityError",
    # Budget tracking
    "BudgetTracker",
    "BudgetLimits",
    "TokenUsage",
    "CostTracker",
    "estimate_tokens",
    # Comparison
    "ComparisonResult",
    "ExecutionMetrics",
    # Parsing
    "ParsedResponse",
    "parse_response",
    "extract_python_code",
    "extract_final_answer",
    "extract_final_var",
    "format_code_for_display",
    "format_result_for_llm",
    # Tracing
    "TraceStep",
    "ExecutionTrace",
    "ActionType",
    "load_trace_from_jsonl",
    "load_trace_from_json",
    # Multi-model
    "ModelConfig",
    "ModelMetrics",
]
