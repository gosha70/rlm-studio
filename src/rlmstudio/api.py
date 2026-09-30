"""Unified high-level API for RLMKit.

Provides ``interact()`` and ``complete()`` as the main programmatic entry
points, plus async variants ``interact_async()`` and ``complete_async()``.
Under the hood these delegate to Clean Architecture use cases
(``RunDirectUseCase``, ``RunRAGUseCase``, ``RunRLMUseCase``) with
``LiteLLMAdapter`` for provider-agnostic LLM access.
"""

from __future__ import annotations

import dataclasses
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from rlmstudio.application.dto import RunConfigDTO, RunResultDTO
from rlmstudio.application.sandbox_vars import MODE_DIRECT, MODE_RAG, MODE_RLM, MODE_RLM_OFFICIAL
from rlmstudio.application.use_cases.run_comparison import (
    ComparisonResultDTO,
    RunComparisonUseCase,
)
from rlmstudio.application.use_cases.run_direct import RunDirectUseCase
from rlmstudio.application.use_cases.run_matrix_comparison import (
    MatrixComparisonResultDTO,
    MatrixSlotDTO,
    RunMatrixComparisonUseCase,
)
from rlmstudio.application.use_cases.run_rag import RunRAGUseCase
from rlmstudio.application.use_cases.run_rlm import RunRLMUseCase
from rlmstudio.infrastructure.embedding.litellm_embedding_adapter import LiteLLMEmbeddingAdapter
from rlmstudio.infrastructure.engines.rlms_adapter import RlmsEngineAdapter
from rlmstudio.infrastructure.llm.litellm_adapter import LiteLLMAdapter
from rlmstudio.infrastructure.sandbox.sandbox_factory import create_sandbox
from rlmstudio.infrastructure.storage.sqlite_adapter import SQLiteStorageAdapter
from rlmstudio.ui.data.providers_catalog import PROVIDERS_BY_KEY

# Type alias for interaction modes
InteractionMode = Literal["direct", "rag", "rlm", "compare", "auto"]

# Token-count thresholds for auto mode selection.
_DIRECT_THRESHOLD = 8_000  # below → direct
_RAG_THRESHOLD = 100_000  # between direct and rlm thresholds → rag


@dataclass
class InteractResult:
    """Result from an ``interact()`` call."""

    answer: str
    mode_used: str
    metrics: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] | None = None
    raw_result: RunResultDTO | None = None

    @property
    def total_tokens(self) -> int:
        return int(self.metrics.get("total_tokens", 0))

    @property
    def total_cost(self) -> float:
        return float(self.metrics.get("total_cost", 0.0))

    @property
    def elapsed_time(self) -> float:
        return float(self.metrics.get("execution_time", 0.0))

    @property
    def steps(self) -> int:
        return int(self.metrics.get("steps", self.metrics.get("llm_calls", 0)))

    def __str__(self) -> str:
        return self.answer

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "mode_used": self.mode_used,
            "metrics": self.metrics,
            "has_trace": self.trace is not None,
        }


@dataclass
class MatrixSlotResult:
    """Public result for one slot in a :func:`compare_matrix` run.

    Attributes:
        slot_id: Stable identifier for this slot (unique within the matrix).
        label: Human-readable label (e.g. ``"openai/gpt-4o · direct"``).
        provider: Backend key (e.g. ``"openai"``, ``"anthropic"``).
        model: Model identifier (e.g. ``"gpt-4o"``).
        mode: Execution mode (``"direct"`` | ``"rag"`` | ``"rlm"``).
        answer: Final answer text.
        success: Whether the slot's execution succeeded.
        error: Error message if the slot failed, else ``None``.
        input_tokens: Input tokens consumed.
        output_tokens: Output tokens consumed.
        total_tokens: ``input_tokens + output_tokens``.
        total_cost: Cost in USD for this slot.
        elapsed_time: Wall-clock seconds for this slot.
        steps: Number of execution steps.
        trace: Serialized execution trace (may be ``None`` for failed slots).
    """

    slot_id: str
    label: str
    provider: str
    model: str
    mode: str
    answer: str
    success: bool
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    elapsed_time: float = 0.0
    steps: int = 0
    trace: list[dict[str, Any]] | None = None


@dataclass
class MatrixCompareResult:
    """Public result from a :func:`compare_matrix` call.

    Attributes:
        comparison_group_id: UUID identifying this matrix run. All slots
            share this ID so callers can cross-reference them later.
        slots: One :class:`MatrixSlotResult` per input slot, in input order.
        ranking: List of slot indices (into ``slots``) ordered best → worst
            by ``ranking_metric``. Failed slots are placed at the end.
        ranking_metric: The metric used to produce ``ranking``
            (``"cost"`` | ``"tokens"`` | ``"latency"`` | ``"answer_per_cost"``).
        total_elapsed: Wall-clock seconds from first slot start to last finish.
    """

    comparison_group_id: str
    slots: list[MatrixSlotResult] = field(default_factory=list)
    ranking: list[int] = field(default_factory=list)
    ranking_metric: str = "cost"
    total_elapsed: float = 0.0

    def get_slot(self, slot_id: str) -> MatrixSlotResult | None:
        """Return the slot with *slot_id*, or ``None`` if not present."""
        for s in self.slots:
            if s.slot_id == slot_id:
                return s
        return None

    @property
    def best(self) -> MatrixSlotResult | None:
        """Return the top-ranked *successful* slot, or ``None`` if none
        succeeded.

        The matrix ranking always places failed slots after successful ones,
        but when every slot fails the first-ranked entry is still a failure.
        This helper guards against that: callers can treat a non-``None``
        return as a genuine winner they can display to users.
        """
        if not self.ranking or not self.slots:
            return None
        top = self.slots[self.ranking[0]]
        if not top.success:
            return None
        return top


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (1 token ~ 4 chars)."""
    return max(1, len(text) // 4)


def _determine_auto_mode(content: str) -> str:
    """Pick the best mode based on content size.

    - < 8K tokens    -> direct  (full context fits comfortably)
    - 8K–100K tokens -> rag     (too large for direct; retrieval beats full scan)
    - > 100K tokens  -> rlm     (recursive exploration scales to millions of tokens)
    """
    token_count = _estimate_tokens(content)
    if token_count < _DIRECT_THRESHOLD:
        return "direct"
    if token_count < _RAG_THRESHOLD:
        return "rag"
    return "rlm"


_PROVIDER_PREFIXES: dict[str, str] = {
    "anthropic": "anthropic/",
    "ollama": "ollama/",
    "lmstudio": "openai/",
    # vLLM needs hosted_vllm/, not openai/: with LiteLLM >= 1.50 an openai/*
    # model on a vLLM upstream auto-routes to /v1/responses, which breaks
    # multi-turn agentic loops. See docs/hosts/dgx-spark-vllm.md §7 Blocker #4.
    "vllm": "hosted_vllm/",
}


def _registry_config_kwargs() -> dict[str, int]:
    """Result-registry limits from the library config (``execution:`` section)."""
    from rlmstudio.config import get_default_config

    try:
        kwargs: dict[str, int] = get_default_config().execution.registry_kwargs()
    except Exception:  # config unreadable — fall back to the DTO defaults
        return {}
    return kwargs


def _auto_detect_provider() -> str | None:
    """Detect the LLM provider from environment variables."""
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OLLAMA_BASE_URL"):
        return "ollama"
    return None


def _resolve_embedding_key(
    provider: str,
    chat_api_key: str | None,
    embedding_api_key: str | None,
) -> str | None:
    """Return the API key to use for the embedding call, or raise ValueError.

    RAG mode uses OpenAI's ``text-embedding-3-small`` by default.
    - OpenAI chat provider: the same ``api_key`` covers embeddings.
    - Any other provider: embeddings need a separate OpenAI key.
      Resolution order: ``embedding_api_key`` arg → ``OPENAI_API_KEY`` env var.
      Raises ``ValueError`` if neither is available so the caller gets a
      clear error rather than silently passing the wrong credentials.
    """
    if embedding_api_key:
        return embedding_api_key
    if provider == "openai":
        return chat_api_key  # same key covers both chat and embeddings
    # Non-OpenAI provider: require an explicit OpenAI key for embeddings.
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return openai_key
    raise ValueError(
        f"RAG mode requires an OpenAI API key for embeddings "
        f"(provider={provider!r} has no native embedding API). "
        "Set OPENAI_API_KEY in the environment or pass embedding_api_key= explicitly."
    )


_LOCAL_PROVIDERS: frozenset[str] = frozenset({"ollama", "lmstudio", "vllm"})

_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
}


def _fetch_lmstudio_model(api_base: str, timeout: float = 5.0) -> str | None:
    """Query LM Studio's /v1/models and return the first loaded model ID, or None."""
    import json
    import urllib.error
    import urllib.request

    url = api_base.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            data = json.loads(resp.read())
        models = data.get("data", [])
        if models:
            model_id = models[0].get("id")
            return str(model_id) if model_id is not None else None
    except Exception:
        pass
    return None


def _resolve_model(
    provider: str,
    model: str | None,
    api_base: str | None = None,
) -> str:
    """Apply LiteLLM provider prefix and default model if needed.

    For ``lmstudio`` without an explicit ``model=``, the loaded model is
    auto-detected from the ``/v1/models`` endpoint when ``api_base`` is given.
    For ``ollama`` a model name is always required.
    """
    if model is None and provider == "lmstudio":
        if api_base:
            model = _fetch_lmstudio_model(api_base)
        if model is None:
            location = f" at '{api_base}'" if api_base else ""
            raise ValueError(
                f"Cannot connect to LM Studio{location}. "
                "Make sure LM Studio is running and its local API server is enabled "
                "(open LM Studio → Local Server tab → Start Server). "
                "Alternatively, pass model='<model-name>' explicitly to skip auto-detection."
            )
    if model is None and provider == "ollama":
        raise ValueError(
            "provider='ollama' requires an explicit model= argument (e.g. model='llama3.2'). "
            "Ollama can serve many models simultaneously so there is no safe default."
        )
    if model is None and provider == "vllm":
        raise ValueError(
            "provider='vllm' requires an explicit model= argument "
            "(e.g. model='Qwen/Qwen2.5-7B-Instruct'). "
            "Pass the HuggingFace model ID exactly as it was given to 'vllm serve'."
        )
    m = model or _DEFAULT_MODELS.get(provider, "gpt-4o")
    prefix = _PROVIDER_PREFIXES.get(provider, "")
    if prefix and not m.startswith(prefix):
        m = f"{prefix}{m}"
    return m


def build_llm_adapter(
    spec: str,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    timeout: float | None = None,
    num_retries: int | None = None,
) -> LiteLLMAdapter:
    """Build a :class:`LiteLLMAdapter` from a ``"provider/model"`` spec.

    The same resolution :func:`compare_matrix` applies per slot (provider
    prefix, local-provider retry default), exposed for callers that need one
    adapter outside a matrix run — for example the benchmark judge.
    """
    provider, raw_model = _parse_provider_spec(spec)
    return LiteLLMAdapter(
        model=_resolve_model(provider, raw_model, api_base),
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout if timeout is not None else 120.0,
        num_retries=(
            num_retries if num_retries is not None else 0 if provider in _LOCAL_PROVIDERS else 2
        ),
    )


def _setup(
    content: str,
    query: str,
    mode: InteractionMode,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    api_base: str | None,
    max_steps: int,
    temperature: float,
    max_tokens: int | None,
    timeout: float | None,
    verbose: bool,
    root_model: str | None = None,
    recursive_model: str | None = None,
    embedding_api_key: str | None = None,
    num_retries: int | None = None,
    stall_limit: int = 3,
) -> tuple[str, str, LiteLLMAdapter, RunConfigDTO, str | None]:
    """Validate inputs, resolve mode/provider/model, build adapter and config.

    Returns (actual_mode, provider, llm_adapter, run_config, resolved_embedding_key).
    """
    if not content:
        raise ValueError("content cannot be empty")
    if not query:
        raise ValueError("query cannot be empty")

    # -- Resolve mode --
    actual_mode: str = mode
    if mode == "auto":
        actual_mode = _determine_auto_mode(content)
        if verbose:
            print(
                f"[Auto Mode] Selected '{actual_mode}' based on content size "
                f"({_estimate_tokens(content):,} tokens)"
            )

    if actual_mode not in ("direct", "rag", "rlm", "compare"):
        raise ValueError(
            f"Invalid mode: {actual_mode}. Must be 'direct', 'rag', 'rlm', 'compare', or 'auto'"
        )

    # -- Resolve provider --
    if provider is None:
        provider = _auto_detect_provider()
        if provider is None:
            raise ValueError(
                "No LLM provider configured. Set OPENAI_API_KEY, "
                "ANTHROPIC_API_KEY, or OLLAMA_BASE_URL in the environment, "
                "or pass provider= explicitly."
            )
        if verbose:
            print(f"[Auto-Detect] Using '{provider}' provider from environment")

    prefixed_model = _resolve_model(provider, model, api_base)
    resolved_root = _resolve_model(provider, root_model) if root_model else None
    resolved_recursive = _resolve_model(provider, recursive_model) if recursive_model else None

    if verbose:
        print(f"[Setup] Configuring {provider}/{prefixed_model} ...")

    llm = LiteLLMAdapter(
        model=prefixed_model,
        root_model=resolved_root,
        recursive_model=resolved_recursive,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout if timeout is not None else 120.0,
        num_retries=(
            num_retries if num_retries is not None else 0 if provider in _LOCAL_PROVIDERS else 2
        ),
    )

    config = RunConfigDTO(
        mode=actual_mode,
        provider=provider,
        model=prefixed_model,
        max_steps=max_steps,
        stall_limit=stall_limit,
        **_registry_config_kwargs(),
    )

    resolved_emb_key = (
        _resolve_embedding_key(provider, api_key, embedding_api_key)
        if actual_mode == "rag"
        else None
    )
    return actual_mode, provider, llm, config, resolved_emb_key


def _build_result(
    actual_mode: str,
    result: RunResultDTO | ComparisonResultDTO,
    verbose: bool,
) -> InteractResult:
    """Convert a RunResultDTO or ComparisonResultDTO into an InteractResult."""
    if isinstance(result, ComparisonResultDTO):
        # Pick best answer: prefer successful rlm, then successful direct, then any
        successful = {m: r for m, r in result.results.items() if r.success}
        best = (
            successful.get("rlm")
            or successful.get("direct")
            or result.results.get("rlm")
            or result.results.get("direct")
            or RunResultDTO(answer="", mode_used="compare", success=False, error="No results")
        )
        metrics: dict[str, Any] = {
            "total_tokens": sum(r.total_tokens for r in result.results.values()),
            "total_cost": sum(r.total_cost for r in result.results.values()),
            "execution_time": result.total_elapsed,
            "comparison": {
                mode: {
                    "answer_length": len(r.answer),
                    "total_tokens": r.total_tokens,
                    "total_cost": r.total_cost,
                    "elapsed_time": r.elapsed_time,
                }
                for mode, r in result.results.items()
            },
        }
        if verbose:
            print(f"[Compare] Ran {len(result.results)} modes: {result.modes_run}")
            for mode, r in result.results.items():
                print(f"  {mode}: {r.total_tokens} tokens, ${r.total_cost:.4f}")
        return InteractResult(
            answer=best.answer,
            mode_used="compare",
            metrics=metrics,
            raw_result=best,
        )

    metrics = {
        "total_tokens": result.total_tokens,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "total_cost": result.total_cost,
        "execution_time": result.elapsed_time,
        "steps": result.steps,
    }

    if verbose:
        print(f"[Complete] Generated {len(result.answer)} character response")
        print(f"  Tokens: {metrics['total_tokens']:,} | Cost: ${metrics['total_cost']:.4f}")

    return InteractResult(
        answer=result.answer,
        mode_used=actual_mode,
        metrics=metrics,
        trace=result.trace or None,
        raw_result=result,
    )


def _dispatch_sync(
    actual_mode: str,
    llm: LiteLLMAdapter,
    config: RunConfigDTO,
    content: str,
    query: str,
    embedding_api_key: str | None = None,
) -> RunResultDTO | ComparisonResultDTO:
    """Run the appropriate use case synchronously."""
    if actual_mode == "compare":
        sandbox = create_sandbox()
        uc_cmp = RunComparisonUseCase(llm, sandbox)
        return uc_cmp.execute(content, query, config)
    elif actual_mode == "rlm":
        sandbox = create_sandbox()
        uc = RunRLMUseCase(llm, sandbox)
        return uc.execute(content, query, config)
    elif actual_mode == "rag":
        collection = f"rag_{uuid.uuid4().hex}"
        config.extra["collection"] = collection
        embedder = LiteLLMEmbeddingAdapter(api_key=embedding_api_key)
        storage = SQLiteStorageAdapter(":memory:")
        uc_rag = RunRAGUseCase(llm, embedder, storage)
        return uc_rag.execute(content, query, config)
    else:
        uc_direct = RunDirectUseCase(llm)
        return uc_direct.execute(content, query, config)


async def _dispatch_async(
    actual_mode: str,
    llm: LiteLLMAdapter,
    config: RunConfigDTO,
    content: str,
    query: str,
    embedding_api_key: str | None = None,
) -> RunResultDTO | ComparisonResultDTO:
    """Run the appropriate use case asynchronously."""
    if actual_mode == "compare":
        sandbox = create_sandbox()
        uc_cmp = RunComparisonUseCase(llm, sandbox)
        return await uc_cmp.execute_async(content, query, config)
    elif actual_mode == "rlm":
        sandbox = create_sandbox()
        uc = RunRLMUseCase(llm, sandbox)
        return await uc.execute_async(content, query, config)
    elif actual_mode == "rag":
        import asyncio

        collection = f"rag_{uuid.uuid4().hex}"
        config.extra["collection"] = collection
        embedder = LiteLLMEmbeddingAdapter(api_key=embedding_api_key)
        storage = SQLiteStorageAdapter(":memory:")
        uc_rag = RunRAGUseCase(llm, embedder, storage)
        return await asyncio.to_thread(uc_rag.execute, content, query, config)
    else:
        uc_direct = RunDirectUseCase(llm)
        return await uc_direct.execute_async(content, query, config)


def interact(
    content: str,
    query: str,
    mode: InteractionMode = "auto",
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    max_steps: int = 16,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    timeout: float | None = None,
    verbose: bool = False,
    root_model: str | None = None,
    recursive_model: str | None = None,
    embedding_api_key: str | None = None,
    num_retries: int | None = None,
    stall_limit: int = 3,
    **kwargs: Any,
) -> InteractResult:
    """Interact with content using an LLM through Direct, RLM, or Compare mode.

    This is the main entry point for RLMKit.  It handles provider
    resolution, mode selection, and use-case dispatch.

    Args:
        content: Document / text content to analyse.
        query: Question or instruction for the LLM.
        mode: ``"direct"`` | ``"rlm"`` | ``"compare"`` | ``"auto"``.
        provider: Provider key (``"openai"``, ``"anthropic"``, ``"ollama"``).
            Auto-detected from env vars when *None*.
        model: Model name.  Defaults to the provider's flagship model.
        api_key: API key override (usually read from env vars).
        api_base: Custom API base URL (e.g. for Ollama on a non-default port).
        max_steps: Budget limit for RLM loop iterations.
        temperature: Sampling temperature.
        max_tokens: Max output tokens per LLM call.
        timeout: Request timeout in seconds (default 120).
        verbose: Print progress to stdout.
        root_model: Model for root-level reasoning (paper's two-model optimization).
        recursive_model: Cheaper model for RLM exploration subcalls.
        embedding_api_key: API key for the embedding model used in RAG mode.
            Defaults to ``api_key`` when ``provider="openai"``, otherwise falls
            back to ``OPENAI_API_KEY`` from the environment.
        num_retries: Number of retry attempts on transient LLM errors.
            ``None`` (default) applies automatic defaults: ``0`` for local
            providers (``ollama``, ``lmstudio``) to avoid multiplying hangs,
            ``2`` for cloud providers.  Pass ``0`` to disable retries entirely,
            or a positive integer to override.
        stall_limit: Consecutive no-progress steps before the circuit breaker
            fires (default ``3``).  When the limit is reached, if the model
            produced any plain-text response (without using the ``FINAL:``
            prefix), that text is returned as the answer.  Lower values (e.g.
            ``1``) make RLM exit sooner on models that answer directly without
            following the protocol; higher values give the model more chances to
            produce code before giving up.

    Returns:
        :class:`InteractResult` with answer, mode used, and metrics.

    Raises:
        ValueError: If inputs are invalid or no provider can be resolved.
    """
    actual_mode, _provider, llm, config, emb_key = _setup(
        content,
        query,
        mode,
        provider,
        model,
        api_key,
        api_base,
        max_steps,
        temperature,
        max_tokens,
        timeout,
        verbose,
        root_model=root_model,
        recursive_model=recursive_model,
        embedding_api_key=embedding_api_key,
        num_retries=num_retries,
        stall_limit=stall_limit,
    )

    result = _dispatch_sync(actual_mode, llm, config, content, query, emb_key)
    return _build_result(actual_mode, result, verbose)


async def interact_async(
    content: str,
    query: str,
    mode: InteractionMode = "auto",
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    max_steps: int = 16,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    timeout: float | None = None,
    verbose: bool = False,
    root_model: str | None = None,
    recursive_model: str | None = None,
    embedding_api_key: str | None = None,
    num_retries: int | None = None,
    stall_limit: int = 3,
    **kwargs: Any,
) -> InteractResult:
    """Async version of :func:`interact`.

    Same parameters and return type.  Uses the async execution path
    of each use case (``execute_async``), which enables true async
    LLM calls via ``litellm.acompletion``.
    """
    actual_mode, _provider, llm, config, emb_key = _setup(
        content,
        query,
        mode,
        provider,
        model,
        api_key,
        api_base,
        max_steps,
        temperature,
        max_tokens,
        timeout,
        verbose,
        root_model=root_model,
        recursive_model=recursive_model,
        embedding_api_key=embedding_api_key,
        num_retries=num_retries,
        stall_limit=stall_limit,
    )

    result = await _dispatch_async(actual_mode, llm, config, content, query, emb_key)
    return _build_result(actual_mode, result, verbose)


# ---------------------------------------------------------------------------
# Matrix comparison (N providers × M modes)
# ---------------------------------------------------------------------------


MATRIX_MODES: frozenset[str] = frozenset({MODE_DIRECT, MODE_RAG, MODE_RLM, MODE_RLM_OFFICIAL})
"""Every mode a Compare-matrix slot can run in."""
_MATRIX_MODE_SET = MATRIX_MODES
_MATRIX_RANKING_METRICS: frozenset[str] = frozenset(
    {"cost", "tokens", "latency", "answer_per_cost"}
)


def _parse_provider_spec(spec: str) -> tuple[str, str]:
    """Split a ``"provider/model"`` string into ``(provider, model)``.

    Splits on the *first* ``/`` only, so HuggingFace-style model IDs
    with embedded slashes are preserved in the model portion:

    - ``"openai/gpt-4o"`` → ``("openai", "gpt-4o")``
    - ``"vllm/Qwen/Qwen2.5-7B-Instruct"`` → ``("vllm", "Qwen/Qwen2.5-7B-Instruct")``

    Callers using HF-style IDs should still pass ``api_base`` (for vLLM)
    and make sure the provider prefix matches the backend, just like the
    single-provider :func:`interact` path.
    """
    if not isinstance(spec, str) or "/" not in spec:
        raise ValueError(
            f"Provider spec must be 'provider/model' (got {spec!r}). Example: 'openai/gpt-4o'."
        )
    provider, _, model = spec.partition("/")
    if not provider or not model:
        raise ValueError(f"Invalid provider spec: {spec!r}")
    return provider, model


def _default_endpoint(provider: str) -> str | None:
    """Catalog default endpoint for local providers (``None`` for cloud backends)."""
    entry = PROVIDERS_BY_KEY.get(provider)
    return entry.default_endpoint if entry else None


def build_matrix_slots(
    providers: list[str],
    modes: list[str],
    *,
    api_key: str | None,
    api_base: str | None,
    temperature: float,
    max_tokens: int | None,
    timeout: float | None,
    max_steps: int,
    num_retries: int | None,
    embedding_api_key: str | None,
    base_config: RunConfigDTO | None = None,
    sandbox_type: str | None = None,
) -> list[MatrixSlotDTO]:
    """Build per-slot :class:`MatrixSlotDTO`s from the public API args.

    Each slot gets its own :class:`LiteLLMAdapter` so one slot's streaming
    state cannot bleed into another's, a fresh sandbox for ``rlm`` slots,
    a fresh in-memory :class:`SQLiteStorageAdapter` +
    :class:`LiteLLMEmbeddingAdapter` for ``rag`` slots, and an
    :class:`RlmsEngineAdapter` (priced by the slot's LLM adapter) for
    ``rlm_official`` slots.

    ``base_config`` seeds every slot's :class:`RunConfigDTO` — the benchmark
    runner uses it to apply per-case budgets (wall-clock, cost, depth); when
    given, its ``max_steps`` wins over the ``max_steps`` argument.
    ``sandbox_type`` selects the engine's environment for ``rlm_official``
    (``docker`` → the engine's Docker environment; default → in-process).
    """
    slots: list[MatrixSlotDTO] = []
    for spec in providers:
        provider, raw_model = _parse_provider_spec(spec)
        prefixed_model = _resolve_model(provider, raw_model, api_base)

        for mode in modes:
            llm = LiteLLMAdapter(
                model=prefixed_model,
                api_key=api_key,
                api_base=api_base,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout if timeout is not None else 120.0,
                num_retries=(
                    num_retries
                    if num_retries is not None
                    else 0
                    if provider in _LOCAL_PROVIDERS
                    else 2
                ),
            )

            sandbox = create_sandbox() if mode == MODE_RLM else None

            engine = None
            if mode == MODE_RLM_OFFICIAL:
                engine = RlmsEngineAdapter(
                    backend=provider,
                    model=raw_model,
                    api_key=api_key,
                    base_url=api_base or _default_endpoint(provider),
                    cost_fn=llm.get_completion_cost,
                    **({"sandbox_type": sandbox_type} if sandbox_type else {}),
                )

            embedder: LiteLLMEmbeddingAdapter | None = None
            storage: SQLiteStorageAdapter | None = None
            extra: dict[str, Any] = {}
            if mode == MODE_RAG:
                emb_key = _resolve_embedding_key(provider, api_key, embedding_api_key)
                embedder = LiteLLMEmbeddingAdapter(api_key=emb_key)
                storage = SQLiteStorageAdapter(":memory:")
                extra = {"collection": f"rag_{uuid.uuid4().hex}"}

            base = base_config or RunConfigDTO(max_steps=max_steps, **_registry_config_kwargs())
            slot_config = dataclasses.replace(
                base,
                mode=mode,
                provider=provider,
                model=prefixed_model,
                api_key=api_key,
                extra={**base.extra, **extra},
            )

            slots.append(
                MatrixSlotDTO(
                    slot_id=uuid.uuid4().hex[:12],
                    mode=mode,  # type: ignore[arg-type]
                    llm=llm,
                    sandbox=sandbox,
                    embedder=embedder,
                    storage=storage,
                    label=f"{provider}/{raw_model} · {mode}",
                    provider=provider,
                    model=raw_model,
                    config=slot_config,
                    engine=engine,
                )
            )
    return slots


def _to_public_matrix_result(dto: MatrixComparisonResultDTO) -> MatrixCompareResult:
    """Convert an internal :class:`MatrixComparisonResultDTO` to the public type."""
    return MatrixCompareResult(
        comparison_group_id=dto.comparison_group_id,
        slots=[
            MatrixSlotResult(
                slot_id=s.slot_id,
                label=s.label,
                provider=s.provider,
                model=s.model,
                mode=s.mode,
                answer=s.result.answer,
                success=s.result.success,
                error=s.result.error,
                input_tokens=s.result.input_tokens,
                output_tokens=s.result.output_tokens,
                total_tokens=s.result.total_tokens,
                total_cost=s.result.total_cost,
                elapsed_time=s.result.elapsed_time,
                steps=s.result.steps,
                trace=s.result.trace or None,
            )
            for s in dto.slots
        ],
        ranking=list(dto.ranking),
        ranking_metric=dto.ranking_metric,
        total_elapsed=dto.total_elapsed,
    )


def _validate_matrix_inputs(
    content: str,
    query: str,
    providers: list[str],
    modes: list[str],
    ranking_metric: str,
) -> None:
    """Validate :func:`compare_matrix` inputs, raising ``ValueError`` on bad args.

    Runs before any adapters are built or slots are executed, so a typo
    in ``ranking_metric`` fails immediately instead of after spending
    tokens and money on the whole matrix.
    """
    if not content:
        raise ValueError("content cannot be empty")
    if not query:
        raise ValueError("query cannot be empty")
    if not providers:
        raise ValueError("providers must be a non-empty list")
    if not modes:
        raise ValueError("modes must be a non-empty list")
    for mode in modes:
        if mode not in _MATRIX_MODE_SET:
            valid = ", ".join(repr(m) for m in sorted(_MATRIX_MODE_SET))
            raise ValueError(f"Invalid matrix mode: {mode!r}. Valid: {valid}.")
    if ranking_metric not in _MATRIX_RANKING_METRICS:
        raise ValueError(
            f"Invalid ranking_metric: {ranking_metric!r}. Valid: {sorted(_MATRIX_RANKING_METRICS)}."
        )
    # Validate every provider spec up front so a typo in the third entry
    # doesn't fire only after the first two slots have already run.
    for spec in providers:
        _parse_provider_spec(spec)
    total_slots = len(providers) * len(modes)
    if total_slots > RunMatrixComparisonUseCase.MAX_SLOTS:
        raise ValueError(
            f"Too many slots: {total_slots} (= {len(providers)} providers × "
            f"{len(modes)} modes) exceeds MAX_SLOTS="
            f"{RunMatrixComparisonUseCase.MAX_SLOTS}"
        )


def compare_matrix(
    content: str,
    query: str,
    providers: list[str],
    modes: list[str] | None = None,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    timeout: float | None = None,
    max_steps: int = 16,
    embedding_api_key: str | None = None,
    ranking_metric: str = "cost",
    num_retries: int | None = None,
    verbose: bool = False,
) -> MatrixCompareResult:
    """Run the same query across N providers × M modes in parallel.

    Each slot builds its own :class:`LiteLLMAdapter` so provider state is
    fully isolated, then all slots fan out through a thread pool.  Slot
    failures become failed :class:`MatrixSlotResult`s — they never fail
    the whole matrix run.

    Args:
        content: Document text to analyze.
        query: User question.
        providers: Provider specs in ``"backend/model"`` format (e.g.
            ``["openai/gpt-4o", "anthropic/claude-sonnet-4-6"]``).  The
            split is on the *first* ``/`` only, so HuggingFace-style IDs
            work via ``"vllm/Qwen/Qwen2.5-7B-Instruct"`` — pass
            ``api_base`` for the vLLM server alongside.
        modes: Execution modes to run for each provider (default
            ``["direct"]``).  Valid: ``"direct"``, ``"rag"``, ``"rlm"``.
        api_key: API key applied to every slot.
        api_base: API base URL applied to every slot.
        temperature: Sampling temperature.
        max_tokens: Max output tokens per LLM call.
        timeout: Request timeout in seconds.
        max_steps: RLM budget for ``rlm`` slots.
        embedding_api_key: API key for RAG embeddings (OpenAI by default).
        ranking_metric: Metric for ranking successful slots. One of
            ``"cost"`` | ``"tokens"`` | ``"latency"`` | ``"answer_per_cost"``.
        num_retries: Retry count per LLM call.
        verbose: Print progress to stdout.

    Returns:
        :class:`MatrixCompareResult` with per-slot results and a ranking.

    Raises:
        ValueError: If inputs are invalid, a provider spec is malformed,
            or the total slot count exceeds ``MAX_SLOTS=10``.
    """
    if modes is None:
        modes = ["direct"]
    _validate_matrix_inputs(content, query, providers, modes, ranking_metric)

    slots = build_matrix_slots(
        providers,
        modes,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_steps=max_steps,
        num_retries=num_retries,
        embedding_api_key=embedding_api_key,
    )

    uc = RunMatrixComparisonUseCase()
    result = uc.execute(
        content,
        query,
        slots,
        ranking_metric=ranking_metric,  # type: ignore[arg-type]
    )

    if verbose:
        print(
            f"[Matrix] Ran {len(result.slots)} slot(s), "
            f"group={result.comparison_group_id[:8]}, "
            f"elapsed={result.total_elapsed:.2f}s"
        )
        for rank_pos, slot_idx in enumerate(result.ranking, start=1):
            s = result.slots[slot_idx]
            marker = "✓" if s.result.success else "✗"
            print(
                f"  #{rank_pos} {marker} {s.label}: "
                f"{s.result.total_tokens} tokens, ${s.result.total_cost:.4f}, "
                f"{s.result.elapsed_time:.2f}s"
            )

    return _to_public_matrix_result(result)


async def compare_matrix_async(
    content: str,
    query: str,
    providers: list[str],
    modes: list[str] | None = None,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    timeout: float | None = None,
    max_steps: int = 16,
    embedding_api_key: str | None = None,
    ranking_metric: str = "cost",
    num_retries: int | None = None,
    verbose: bool = False,
) -> MatrixCompareResult:
    """Async version of :func:`compare_matrix`.

    Internally offloads the synchronous fan-out to a worker thread via
    :func:`asyncio.to_thread`, so this coroutine is safe to call from
    async contexts without blocking the event loop.
    """
    import asyncio

    return await asyncio.to_thread(
        compare_matrix,
        content,
        query,
        providers,
        modes,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_steps=max_steps,
        embedding_api_key=embedding_api_key,
        ranking_metric=ranking_metric,
        num_retries=num_retries,
        verbose=verbose,
    )


def complete(content: str, query: str, **kwargs: Any) -> str:
    """Convenience wrapper returning just the answer string.

    Args:
        content: Document / text content.
        query: Question or instruction.
        **kwargs: Forwarded to :func:`interact`.

    Returns:
        The answer text.
    """
    return interact(content=content, query=query, **kwargs).answer


async def complete_async(content: str, query: str, **kwargs: Any) -> str:
    """Async convenience wrapper returning just the answer string.

    Args:
        content: Document / text content.
        query: Question or instruction.
        **kwargs: Forwarded to :func:`interact_async`.

    Returns:
        The answer text.
    """
    result = await interact_async(content=content, query=query, **kwargs)
    return result.answer
