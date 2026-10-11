# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""
RLMKit: Recursive Language Model Toolkit

A toolkit for building LLM-based systems that can handle arbitrarily large contexts
through code generation and recursive exploration.

Based on the RLM paper (arXiv:2512.24601).
"""

from .core import (  # noqa: I001 — .core must be imported before .api to avoid circular imports
    RLM,
    BudgetExceeded,
    BudgetLimits,
    BudgetTracker,
    CostTracker,
    ExecutionError,
    LLMClient,
    RLMError,
    RLMResult,
    SecurityError,
    TokenUsage,
    estimate_tokens,
)
from .config import ExecutionConfig, MonitoringConfig, RLMConfig, SecurityConfig
from .llm import BaseLLMProvider, LLMResponse, MockLLMClient
from .llm.config import LLMConfig, LLMProviderConfig, ModelPricing
from .prompts import format_system_prompt, get_default_system_prompt
from .api import (
    InteractResult,
    MatrixCompareResult,
    MatrixSlotResult,
    compare_matrix,
    compare_matrix_async,
    complete,
    complete_async,
    interact,
    interact_async,
)

# Optional external LLM providers — assigned None when the dependency is missing,
# so the variable type is ``type[X] | None``.  mypy cannot express that cleanly
# with a try/except import, so we pre-declare with the correct union type.
OpenAIClient: type | None
try:
    from .llm import OpenAIClient
except ImportError:
    OpenAIClient = None

ClaudeClient: type | None
try:
    from .llm import ClaudeClient
except ImportError:
    ClaudeClient = None

OllamaClient: type | None
try:
    from .llm import OllamaClient
except ImportError:
    OllamaClient = None

LMStudioClient: type | None
try:
    from .llm import LMStudioClient
except ImportError:
    LMStudioClient = None

vLLMClient: type | None  # noqa: N816
try:
    from .llm import vLLMClient  # noqa: N816
except ImportError:
    vLLMClient = None  # noqa: N816

# Strategy framework
try:
    from .strategies import (  # noqa: F401
        DirectStrategy,
        EmbeddingProvider,
        EvaluationResult,
        IndexedRAGStrategy,
        LLMStrategy,
        MultiStrategyEvaluator,
        OpenAIEmbedder,
        RAGStrategy,
        RLMStrategy,
        StrategyResult,
    )
except ImportError:
    pass

# Storage layer
try:
    from .storage import ConversationStore, Database, VectorStore, get_default_db_path  # noqa: F401
except ImportError:
    pass

# Benchmark harness
try:
    from .benchmark import (  # noqa: F401
        BenchmarkCase,
        BenchmarkDataset,
        BenchmarkReport,
        BenchmarkRun,
        BenchmarkRunner,
        CaseResult,
        load_dataset,
        load_dataset_from_dict,
    )
except ImportError:
    pass

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("rlm-studio")
except Exception:
    __version__ = "0.0.0"  # fallback for un-installed / editable without metadata

__all__ = [
    # High-Level API
    "interact",
    "interact_async",
    "complete",
    "complete_async",
    "compare_matrix",
    "compare_matrix_async",
    "InteractResult",
    "MatrixCompareResult",
    "MatrixSlotResult",
    # Core RLM
    "RLM",
    "RLMResult",
    "LLMClient",
    # Configuration
    "RLMConfig",
    "SecurityConfig",
    "ExecutionConfig",
    "MonitoringConfig",
    "LLMConfig",
    "LLMProviderConfig",
    "ModelPricing",
    # Errors
    "RLMError",
    "BudgetExceeded",
    "ExecutionError",
    "SecurityError",
    # Budget Tracking
    "BudgetTracker",
    "BudgetLimits",
    "TokenUsage",
    "CostTracker",
    "estimate_tokens",
    # LLM Clients
    "MockLLMClient",
    "BaseLLMProvider",
    "LLMResponse",
    # Prompts
    "format_system_prompt",
    "get_default_system_prompt",
]

# Add optional providers to __all__ if available
if OpenAIClient is not None:
    __all__.append("OpenAIClient")
if ClaudeClient is not None:
    __all__.append("ClaudeClient")
if OllamaClient is not None:
    __all__.append("OllamaClient")
if LMStudioClient is not None:
    __all__.append("LMStudioClient")
if vLLMClient is not None:
    __all__.append("vLLMClient")
