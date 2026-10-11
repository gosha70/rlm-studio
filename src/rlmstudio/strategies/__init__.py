# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""Strategy abstraction layer for multi-strategy evaluation."""

from .base import LLMStrategy, StrategyResult
from .direct import DirectStrategy
from .embeddings import EmbeddingProvider, OpenAIEmbedder
from .evaluator import EvaluationResult, MultiStrategyEvaluator
from .indexed_rag import IndexedRAGStrategy
from .rag import RAGStrategy
from .rlm_strategy import RLMStrategy

__all__ = [
    "LLMStrategy",
    "StrategyResult",
    "DirectStrategy",
    "RLMStrategy",
    "EmbeddingProvider",
    "OpenAIEmbedder",
    "RAGStrategy",
    "IndexedRAGStrategy",
    "MultiStrategyEvaluator",
    "EvaluationResult",
]
