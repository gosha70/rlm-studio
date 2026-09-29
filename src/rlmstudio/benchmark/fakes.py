# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Offline stand-ins for ``rlm-studio bench --dry-run`` (specs/benchmarks-v1 FR-7).

The dry run pushes every engine through the *real* matrix code path —
``RunMatrixComparisonUseCase`` and the direct / RAG / RLM use cases — with
in-memory adapters, so CI catches a broken runner, scorer or report without
network access or credentials. Each fake answers the case's expected answer
(first alternative) so the accuracy column exercises the pass path, and a
provider whose model is named ``fail`` raises, so the failed column is
exercised too.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterator
from typing import Any

from rlmstudio.application.dto import LLMResponseDTO, RunConfigDTO, RunResultDTO
from rlmstudio.application.sandbox_vars import (
    MODE_RAG,
    MODE_RLM,
    MODE_RLM_OFFICIAL,
    TRACE_KEY_CONTENT,
    TRACE_KEY_ELAPSED_SECONDS,
    TRACE_KEY_INPUT_TOKENS,
    TRACE_KEY_MODE,
    TRACE_KEY_MODEL,
    TRACE_KEY_OUTPUT_TOKENS,
    TRACE_KEY_ROLE,
    TRACE_KEY_STEP,
)
from rlmstudio.application.use_cases.run_matrix_comparison import MatrixSlotDTO
from rlmstudio.infrastructure.sandbox.sandbox_factory import create_sandbox
from rlmstudio.infrastructure.storage.sqlite_adapter import SQLiteStorageAdapter

from .dataset import BenchmarkCase
from .scoring import ANY_OF_SEPARATOR

FAILING_MODEL = "fail"
DRY_RUN_MODEL = "dry-run"
DRY_RUN_ENGINE_VERSION = "0.0.0-dry-run"
_RLM_FINAL_PREFIX = "FINAL: "  # the reply that completes Studio's RLM loop in one step
_EMBEDDING_DIM = 8
_JUDGE_VERDICT = (
    '{"dimensions": {"relevance": 4, "correctness": 4, "completeness": 3, '
    '"coherence": 4, "conciseness": 4}, "reasoning": "dry run"}'
)


def scripted_answer(case: BenchmarkCase) -> str:
    """The answer the fakes give for *case*: its expected answer, or a placeholder."""
    if case.expected_answer:
        return case.expected_answer.split(ANY_OF_SEPARATOR)[0]
    return f"Dry-run answer for {case.id}."


class ScriptedLLM:
    """LLMPort fake with deterministic token counts and a free price."""

    def __init__(self, reply: str, *, fail: bool = False, model: str = DRY_RUN_MODEL) -> None:
        self._reply = reply
        self._fail = fail
        self.model = model
        self.active_model = model

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        if self._fail:
            raise RuntimeError("dry-run provider failure")
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        return LLMResponseDTO(
            content=self._reply,
            model=self.model,
            input_tokens=max(1, prompt_chars // 4),
            output_tokens=max(1, len(self._reply) // 4),
            ttft_ms=1,
        )

    def complete_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        yield self.complete(messages).content

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def get_pricing(self) -> dict[str, float]:
        return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}

    def get_completion_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0


class ScriptedEmbedder:
    """EmbeddingPort fake: a stable hash-derived vector per text (offline RAG)."""

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i] / 255.0 for i in range(_EMBEDDING_DIM)]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return _EMBEDDING_DIM


class ScriptedEngine:
    """RLMEnginePort fake for ``rlm_official`` slots."""

    def __init__(self, reply: str, *, fail: bool = False) -> None:
        self._reply = reply
        self._fail = fail

    @property
    def version(self) -> str | None:
        return DRY_RUN_ENGINE_VERSION

    def is_available(self) -> tuple[bool, str]:
        return True, f"dry-run engine {DRY_RUN_ENGINE_VERSION}"

    def run(self, content: str, query: str, config: RunConfigDTO) -> RunResultDTO:
        if self._fail:
            raise RuntimeError("dry-run engine failure")
        entry_common = {
            TRACE_KEY_MODE: MODE_RLM_OFFICIAL,
            TRACE_KEY_MODEL: DRY_RUN_MODEL,
            TRACE_KEY_ELAPSED_SECONDS: 0.0,
        }
        return RunResultDTO(
            answer=self._reply,
            mode_used=MODE_RLM_OFFICIAL,
            success=True,
            steps=2,
            input_tokens=max(1, len(content) // 4),
            output_tokens=max(1, len(self._reply) // 4),
            total_cost=0.0,
            trace=[
                {
                    TRACE_KEY_STEP: 0,
                    TRACE_KEY_ROLE: "execution",
                    TRACE_KEY_CONTENT: "dry run",
                    TRACE_KEY_INPUT_TOKENS: 0,
                    TRACE_KEY_OUTPUT_TOKENS: 0,
                    **entry_common,
                },
                {
                    TRACE_KEY_STEP: 1,
                    TRACE_KEY_ROLE: "assistant",
                    TRACE_KEY_CONTENT: self._reply,
                    TRACE_KEY_INPUT_TOKENS: 0,
                    TRACE_KEY_OUTPUT_TOKENS: 0,
                    **entry_common,
                },
            ],
        )

    async def run_async(self, content: str, query: str, config: RunConfigDTO) -> RunResultDTO:
        return self.run(content, query, config)


def dry_run_judge_llm() -> ScriptedLLM:
    """A judge that always returns a well-formed verdict."""
    return ScriptedLLM(_JUDGE_VERDICT, model="dry-run-judge")


def dry_run_slot_builder(
    providers: list[str],
    engines: list[str],
    base_config: RunConfigDTO,
    case: BenchmarkCase,
) -> list[MatrixSlotDTO]:
    """Slots for one case with every adapter faked; same shape the real builder returns."""
    answer = scripted_answer(case)
    slots: list[MatrixSlotDTO] = []
    for spec in providers:
        provider, _, model = spec.partition("/")
        fail = model == FAILING_MODEL
        for engine in engines:
            reply = f"{_RLM_FINAL_PREFIX}{answer}" if engine == MODE_RLM else answer
            extra: dict[str, Any] = {}
            embedder = storage = None
            if engine == MODE_RAG:
                embedder = ScriptedEmbedder()
                storage = SQLiteStorageAdapter(":memory:")
                extra = {"collection": f"rag_{uuid.uuid4().hex}"}
            slots.append(
                MatrixSlotDTO(
                    slot_id=uuid.uuid4().hex[:12],
                    mode=engine,  # type: ignore[arg-type]
                    llm=ScriptedLLM(reply, fail=fail, model=model or DRY_RUN_MODEL),
                    sandbox=create_sandbox() if engine == MODE_RLM else None,
                    embedder=embedder,
                    storage=storage,
                    label=f"{spec} · {engine}",
                    provider=provider,
                    model=model,
                    config=RunConfigDTO(
                        mode=engine,
                        provider=provider,
                        model=model,
                        max_steps=base_config.max_steps,
                        max_time_seconds=base_config.max_time_seconds,
                        max_cost=base_config.max_cost,
                        max_recursion_depth=base_config.max_recursion_depth,
                        extra=extra,
                    ),
                    engine=ScriptedEngine(answer, fail=fail)
                    if engine == MODE_RLM_OFFICIAL
                    else None,
                )
            )
    return slots
