"""RLMKit client: high-level entry point using Clean Architecture internals.

This client provides the same functionality as the existing ``rlmstudio.interact()``
and ``rlmstudio.complete()`` but is backed by the new layered architecture.
"""

from __future__ import annotations

import warnings
from typing import Any

from rlmstudio.application.dto import RunConfigDTO, RunResultDTO
from rlmstudio.application.ports.llm_port import LLMPort
from rlmstudio.application.use_cases.run_comparison import (
    RunComparisonUseCase,
)
from rlmstudio.application.use_cases.run_direct import RunDirectUseCase
from rlmstudio.application.use_cases.run_rag import RunRAGUseCase
from rlmstudio.application.use_cases.run_rlm import RunRLMUseCase
from rlmstudio.domain.model_catalog import DEFAULT_ANTHROPIC_MODEL
from rlmstudio.infrastructure.embedding import LiteLLMEmbeddingAdapter, MockEmbeddingAdapter
from rlmstudio.infrastructure.llm.mock_adapter import MockLLMAdapter
from rlmstudio.infrastructure.sandbox.sandbox_factory import create_sandbox
from rlmstudio.infrastructure.storage.sqlite_adapter import SQLiteStorageAdapter

from .errors import ConfigError
from .types import PublicRunResult


class RLMKitClient:
    """High-level client for RLMKit using Clean Architecture.

    This is the recommended entry point for programmatic usage.
    It wires up the correct infrastructure adapters and delegates
    to the appropriate use case.

    By default, ``provider="litellm"`` routes all calls through LiteLLM,
    which supports 100+ providers via model-name prefixes (e.g.
    ``"gpt-4o"``, ``"claude-3-opus-20240229"``, ``"ollama/llama3"``).
    Set ``provider="mock"`` for testing.

    For the paper's two-model cost optimization, pass both
    ``root_model`` and ``recursive_model``.

    Args:
        provider: LLM provider name. ``"litellm"`` (default) uses the
            unified LiteLLM backend. Legacy values ``"openai"``,
            ``"anthropic"``, ``"ollama"`` use the old direct adapters.
        model: Model identifier (used when root/recursive not specified).
        root_model: Model for root-level reasoning (overrides *model*).
        recursive_model: Cheaper model for exploration subcalls.
        api_key: Optional API key override.
        api_base: Optional custom API base URL.
        sandbox_type: Sandbox type ("local", "docker").
        safe_mode: Enable sandbox restrictions.
        max_steps: Default maximum execution steps.
        max_recursion_depth: Default maximum recursion depth.
    """

    def __init__(
        self,
        provider: str = "mock",
        model: str | None = None,
        root_model: str | None = None,
        recursive_model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        sandbox_type: str = "local",
        safe_mode: bool = True,
        max_steps: int = 16,
        max_recursion_depth: int = 1,
    ) -> None:
        warnings.warn(
            "RLMKitClient is deprecated. Use rlmstudio.interact() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._provider = provider
        self._model = model
        self._root_model = root_model
        self._recursive_model = recursive_model
        self._api_key = api_key
        self._api_base = api_base
        self._max_steps = max_steps
        self._max_recursion_depth = max_recursion_depth

        # Create sandbox adapter
        self._sandbox = create_sandbox(
            sandbox_type=sandbox_type,
            safe_mode=safe_mode,
        )

        # Create LLM adapter
        self._llm = self._create_llm_adapter(
            provider,
            model,
            api_key,
            root_model=root_model,
            recursive_model=recursive_model,
            api_base=api_base,
        )

    def interact(
        self,
        content: str,
        query: str,
        mode: str = "auto",
        verbose: bool = False,
        **kwargs: Any,
    ) -> PublicRunResult:
        """Run a query against content using the specified mode.

        This mirrors the existing ``rlmstudio.interact()`` API but uses
        Clean Architecture internals.

        Args:
            content: Document text to analyze.
            query: User question.
            mode: Execution mode ("direct", "rag", "rlm", "auto", "compare").
            verbose: Print progress output.
            **kwargs: Additional configuration.

        Returns:
            PublicRunResult with the answer and metrics.
        """
        if not content:
            raise ValueError("content cannot be empty")
        if not query:
            raise ValueError("query cannot be empty")

        # Resolve auto mode
        actual_mode = mode
        if mode == "auto":
            actual_mode = self._determine_auto_mode(content)
            if verbose:
                token_est = max(1, len(content) // 4)
                print(
                    f"[Auto Mode] Selected '{actual_mode}' based on "
                    f"content size ({token_est:,} tokens)"
                )

        config = RunConfigDTO(
            mode=actual_mode,
            provider=self._provider,
            model=self._model,
            api_key=self._api_key,
            max_steps=kwargs.get("max_steps", self._max_steps),
            max_recursion_depth=kwargs.get("max_recursion_depth", self._max_recursion_depth),
            verbose=verbose,
            extra=kwargs,
        )

        if actual_mode == "direct":
            uc = RunDirectUseCase(self._llm)
            result = uc.execute(content, query, config)
        elif actual_mode == "rlm":
            uc_rlm = RunRLMUseCase(self._llm, self._sandbox)
            result = uc_rlm.execute(content, query, config)
        elif actual_mode == "rag":
            import uuid

            collection = f"rag_{uuid.uuid4().hex}"
            config.extra["collection"] = collection
            if self._provider == "mock":
                embedder: LiteLLMEmbeddingAdapter | MockEmbeddingAdapter = MockEmbeddingAdapter()
            else:
                from rlmstudio.api import _resolve_embedding_key

                emb_key = _resolve_embedding_key(
                    self._provider, self._api_key, kwargs.get("embedding_api_key")
                )
                embedder = LiteLLMEmbeddingAdapter(api_key=emb_key)
            storage = SQLiteStorageAdapter(":memory:")
            uc_rag = RunRAGUseCase(self._llm, embedder, storage)
            result = uc_rag.execute(content, query, config)
        elif actual_mode == "compare":
            uc_cmp = RunComparisonUseCase(self._llm, self._sandbox)
            cmp_result = uc_cmp.execute(content, query, config)
            # Return the best result (prefer rlm if available)
            result = (
                cmp_result.results.get("rlm")
                or cmp_result.results.get("direct")
                or RunResultDTO(answer="", mode_used="compare", success=False, error="No results")
            )
        else:
            raise ValueError(
                f"Unsupported mode: {actual_mode!r}. Use 'direct', 'rag', 'rlm', 'auto', or 'compare'."
            )

        return self._to_public_result(result)

    def complete(self, content: str, query: str, **kwargs: Any) -> str:
        """Simple completion returning just the answer string.

        Args:
            content: Document text.
            query: User question.
            **kwargs: Passed to interact().

        Returns:
            Answer string.
        """
        result = self.interact(content, query, **kwargs)
        return result.answer

    # -- Private helpers --

    @staticmethod
    def _determine_auto_mode(content: str) -> str:
        """Pick mode based on content size (mirrors api._determine_auto_mode)."""
        token_count = max(1, len(content) // 4)
        if token_count < 8000:
            return "direct"
        elif token_count < 100000:
            return "rag"
        else:
            return "rlm"

    @staticmethod
    def _create_llm_adapter(
        provider: str,
        model: str | None,
        api_key: str | None,
        root_model: str | None = None,
        recursive_model: str | None = None,
        api_base: str | None = None,
    ) -> LLMPort:
        """Create the appropriate LLM adapter for the given provider."""
        if provider == "mock":
            return MockLLMAdapter(["FINAL: Mock response"])

        # LiteLLM: the primary unified provider (100+ models)
        if provider == "litellm":
            from rlmstudio.infrastructure.llm.litellm_adapter import LiteLLMAdapter

            return LiteLLMAdapter(
                model=model or "gpt-4o",
                root_model=root_model,
                recursive_model=recursive_model,
                api_key=api_key,
                api_base=api_base,
            )

        # Legacy direct adapters (opt-in fallbacks)
        if provider == "openai":
            try:
                from rlmstudio.infrastructure.llm.openai_adapter import OpenAIAdapter
                from rlmstudio.llm.openai_client import OpenAIClient

                openai_client = OpenAIClient(model=model or "gpt-4o", api_key=api_key)
                return OpenAIAdapter(openai_client)
            except ImportError as exc:
                raise ConfigError(f"OpenAI not available: {exc}") from exc

        if provider == "anthropic":
            try:
                from rlmstudio.infrastructure.llm.anthropic_adapter import AnthropicAdapter
                from rlmstudio.llm.anthropic_client import ClaudeClient

                anthropic_client = ClaudeClient(
                    model=model or DEFAULT_ANTHROPIC_MODEL, api_key=api_key
                )
                return AnthropicAdapter(anthropic_client)
            except ImportError as exc:
                raise ConfigError(f"Anthropic not available: {exc}") from exc

        if provider == "ollama":
            try:
                from rlmstudio.infrastructure.llm.ollama_adapter import OllamaAdapter
                from rlmstudio.llm.ollama_client import OllamaClient

                if model is None:
                    raise ConfigError(
                        "provider='ollama' requires an explicit model= argument "
                        "(e.g. model='llama3.2'). Local providers have no universal default."
                    )
                ollama_client = OllamaClient(model=model)
                return OllamaAdapter(ollama_client)
            except ImportError as exc:
                raise ConfigError(f"Ollama not available: {exc}") from exc

        raise ConfigError(
            f"Unknown provider: {provider!r}. "
            "Supported: 'litellm', 'openai', 'anthropic', 'ollama', 'mock'."
        )

    @staticmethod
    def _to_public_result(dto: RunResultDTO) -> PublicRunResult:
        """Convert an internal RunResultDTO to a public result."""
        return PublicRunResult(
            answer=dto.answer,
            mode_used=dto.mode_used,
            success=dto.success,
            error=dto.error,
            total_tokens=dto.total_tokens,
            input_tokens=dto.input_tokens,
            output_tokens=dto.output_tokens,
            total_cost=dto.total_cost,
            elapsed_time=dto.elapsed_time,
            steps=dto.steps,
            trace=dto.trace,
            metadata=dto.metadata,
        )
