"""Use case: run the same query across N (provider × mode) slots in parallel.

Extends the single-provider compare mode to a full NxM matrix.  Each slot
brings its own LLM adapter (and optional sandbox for RLM mode), so callers
can compare different providers, models, and modes against the same query.

Fan-out uses a thread pool; each slot's execution is independent.  A slot
failure becomes a failed ``MatrixSlotResultDTO`` — it never fails the whole
matrix run.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypeAlias

from rlmstudio.application.dto import RunConfigDTO, RunResultDTO
from rlmstudio.application.ports.embedding_port import EmbeddingPort
from rlmstudio.application.ports.llm_port import LLMPort
from rlmstudio.application.ports.rlm_engine_port import RLMEnginePort
from rlmstudio.application.ports.sandbox_port import SandboxPort
from rlmstudio.application.ports.storage_port import StoragePort
from rlmstudio.application.sandbox_vars import (
    MODE_DIRECT,
    MODE_DISPLAY_NAMES,
    MODE_RAG,
    MODE_RLM,
    MODE_RLM_OFFICIAL,
    RESULT_KEY_COST_KNOWN,
    ExecutionMode,
    slot_mode_label,
)
from rlmstudio.application.use_cases.run_direct import RunDirectUseCase
from rlmstudio.application.use_cases.run_rag import RunRAGUseCase
from rlmstudio.application.use_cases.run_rlm import RunRLMUseCase
from rlmstudio.application.use_cases.run_rlm_official import RunRLMOfficialUseCase

logger = logging.getLogger(__name__)

SlotMode: TypeAlias = ExecutionMode
RankingMetric = Literal[
    "cost",
    "tokens",
    "latency",
    "answer_per_cost",
    # Judge-scored ordering is produced client-side (the frontend
    # re-ranks on every response); the backend accepts it as a
    # passthrough so the request schema matches the frontend union.
    "judge_score",
    # Phase 5 — prefill/decode-oriented ranking.
    "ttft",
    "decode_tokens_per_sec",
    "cache_hit_rate",
]

_SUPPORTED_MODES: frozenset[str] = frozenset({MODE_DIRECT, MODE_RAG, MODE_RLM, MODE_RLM_OFFICIAL})


@dataclass
class MatrixSlotDTO:
    """Describes one slot in a matrix-compare run.

    Attributes:
        slot_id: Stable identifier for this slot, unique within the matrix.
        mode: Execution mode for this slot (direct / rag / rlm).
        llm: LLM adapter used to execute this slot.
        sandbox: Sandbox adapter (required for rlm mode, ignored otherwise).
        embedder: Embedding adapter (required for rag mode).
        storage: Storage adapter with vector search (required for rag mode).
        label: Human-readable label (e.g. ``"OpenAI GPT-4o / RLM"``).
            Used in UI, telemetry, and logs.
        provider: Backend key (e.g. ``"openai"``); recorded in telemetry.
        model: Model identifier (e.g. ``"gpt-4o"``); recorded in telemetry.
        config: Optional per-slot :class:`RunConfigDTO`.  When set, it
            overrides the base config passed to :meth:`execute` for this
            slot only — useful when different slots need different profile
            settings (e.g. per-provider RLM max_steps or system prompt).
            Callers must ensure ``config.mode`` matches :attr:`mode`.
        engine: Third-party RLM engine (required for rlm_official mode,
            ignored otherwise).  The slot's ``llm`` still prices the run
            when the engine reports no cost.
    """

    slot_id: str
    mode: SlotMode
    llm: LLMPort
    label: str = ""
    provider: str = ""
    model: str = ""
    sandbox: SandboxPort | None = None
    embedder: EmbeddingPort | None = None
    storage: StoragePort | None = None
    config: RunConfigDTO | None = None
    engine: RLMEnginePort | None = None


@dataclass
class MatrixSlotResultDTO:
    """Result produced by one slot in a matrix-compare run."""

    slot_id: str
    label: str
    mode: str
    provider: str
    model: str
    result: RunResultDTO


@dataclass
class MatrixComparisonResultDTO:
    """Aggregate result from a matrix-compare run.

    Attributes:
        comparison_group_id: UUID tying all slot telemetry rows together.
        slots: One result per input slot, in input order.
        total_elapsed: Wall-clock time from first slot start to last slot finish.
        ranking: List of slot indices (into ``slots``) ordered best → worst
            by ``ranking_metric``. Failed slots are placed at the end in input
            order.
        ranking_metric: The metric used to produce ``ranking``.
    """

    comparison_group_id: str
    slots: list[MatrixSlotResultDTO] = field(default_factory=list)
    total_elapsed: float = 0.0
    ranking: list[int] = field(default_factory=list)
    ranking_metric: str = "cost"

    def get_slot(self, slot_id: str) -> MatrixSlotResultDTO | None:
        """Return the slot result with *slot_id*, or None if not present."""
        for s in self.slots:
            if s.slot_id == slot_id:
                return s
        return None

    @property
    def best(self) -> MatrixSlotResultDTO | None:
        """Return the top-ranked slot (first entry of ``ranking``)."""
        if not self.ranking or not self.slots:
            return None
        return self.slots[self.ranking[0]]


class RunMatrixComparisonUseCase:
    """Orchestrates fan-out execution across N matrix slots in parallel.

    Unlike :class:`RunComparisonUseCase` (which is single-provider and only
    runs Direct vs RLM), this use case accepts an arbitrary list of slots,
    each with its own LLM adapter and mode.  Slots run concurrently in a
    thread pool.

    Args:
        max_workers: Upper bound on the thread pool.  ``None`` (default)
            picks ``min(len(slots), 8)`` at ``execute()`` time.
        serial_modes: Modes whose slots must run one after another (while
            the other slots still run in parallel).  The official engine's
            in-process REPL admits one run at a time process-wide and a
            queued run gives up after a short wait, so callers pass
            ``{MODE_RLM_OFFICIAL}`` unless that engine runs in Docker.
    """

    #: Hard ceiling on fan-out — matches the SDD's stated max of 10 slots.
    MAX_SLOTS = 10

    def __init__(
        self, max_workers: int | None = None, serial_modes: frozenset[str] = frozenset()
    ) -> None:
        self._max_workers = max_workers
        self._serial_modes = serial_modes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        content: str,
        query: str,
        slots: list[MatrixSlotDTO],
        config: RunConfigDTO | None = None,
        ranking_metric: RankingMetric = "cost",
    ) -> MatrixComparisonResultDTO:
        """Run every slot against the same (content, query) in parallel.

        Args:
            content: Document text to analyze.
            query: User question.
            slots: List of (mode + adapters) to execute.  Must be non-empty
                and contain no duplicate ``slot_id`` values.
            config: Optional :class:`RunConfigDTO`.  Each slot receives a
                shallow copy with ``mode`` set to the slot's mode.
            ranking_metric: Metric used to rank successful slots.

        Returns:
            :class:`MatrixComparisonResultDTO` with one entry per slot and
            a ranking over successful slots.

        Raises:
            ValueError: If ``slots`` is empty, has duplicate ``slot_id``s,
                exceeds :attr:`MAX_SLOTS`, or contains an invalid mode.
        """
        self._validate(slots)
        base_config = config or RunConfigDTO(mode="compare")

        comparison_group_id = uuid.uuid4().hex
        results_by_slot_id: dict[str, MatrixSlotResultDTO] = {}
        start = time.time()

        # Slots in a serial mode run in order inside one pool task; every
        # other slot gets its own task.  A single serial slot needs no queue.
        serial = [s for s in slots if s.mode in self._serial_modes]
        if len(serial) < 2:
            serial = []
        serial_ids = {s.slot_id for s in serial}
        groups: list[list[MatrixSlotDTO]] = [[s] for s in slots if s.slot_id not in serial_ids]
        if serial:
            groups.append(serial)

        workers = self._max_workers or min(len(groups), 8)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures: dict[Future[None], list[MatrixSlotDTO]] = {
                pool.submit(
                    self._execute_group, group, content, query, base_config, results_by_slot_id
                ): group
                for group in groups
            }
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:  # pragma: no cover — defensive
                    for slot in futures[fut]:
                        if slot.slot_id not in results_by_slot_id:
                            logger.exception("Matrix slot %s crashed", slot.slot_id)
                            results_by_slot_id[slot.slot_id] = _failed_slot_result(slot, str(exc))

        # Preserve input order in the result list.
        ordered = [results_by_slot_id[s.slot_id] for s in slots]
        total_elapsed = time.time() - start

        ranking = self._rank(ordered, ranking_metric)

        return MatrixComparisonResultDTO(
            comparison_group_id=comparison_group_id,
            slots=ordered,
            total_elapsed=total_elapsed,
            ranking=ranking,
            ranking_metric=ranking_metric,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _execute_group(
        self,
        group: list[MatrixSlotDTO],
        content: str,
        query: str,
        base_config: RunConfigDTO,
        results_by_slot_id: dict[str, MatrixSlotResultDTO],
    ) -> None:
        """Run ``group``'s slots in order, recording each result as it lands."""
        for slot in group:
            results_by_slot_id[slot.slot_id] = self._execute_slot(slot, content, query, base_config)

    def _validate(self, slots: list[MatrixSlotDTO]) -> None:
        if not slots:
            raise ValueError("slots must be a non-empty list")
        if len(slots) > self.MAX_SLOTS:
            raise ValueError(f"Too many slots: {len(slots)} exceeds MAX_SLOTS={self.MAX_SLOTS}")
        seen: set[str] = set()
        for slot in slots:
            if slot.slot_id in seen:
                raise ValueError(f"Duplicate slot_id: {slot.slot_id!r}")
            seen.add(slot.slot_id)
            if slot.mode not in _SUPPORTED_MODES:
                raise ValueError(
                    f"Slot {slot.slot_id!r}: unsupported mode {slot.mode!r}. "
                    f"Supported: {sorted(_SUPPORTED_MODES)}"
                )
            if slot.mode == "rlm" and slot.sandbox is None:
                raise ValueError(f"Slot {slot.slot_id!r}: mode='rlm' requires a sandbox adapter")
            if slot.mode == "rag" and (slot.embedder is None or slot.storage is None):
                raise ValueError(f"Slot {slot.slot_id!r}: mode='rag' requires embedder and storage")
            if slot.mode == MODE_RLM_OFFICIAL and slot.engine is None:
                raise ValueError(
                    f"Slot {slot.slot_id!r}: mode={MODE_RLM_OFFICIAL!r} requires an engine adapter"
                )

    def _execute_slot(
        self,
        slot: MatrixSlotDTO,
        content: str,
        query: str,
        base_config: RunConfigDTO,
    ) -> MatrixSlotResultDTO:
        """Run a single slot, returning a :class:`MatrixSlotResultDTO`.

        Exceptions are caught here so they become failed slot results instead
        of crashing the whole matrix run.
        """
        # Per-slot config override (for provider/profile-specific settings)
        # takes precedence over the shared base config.
        if slot.config is not None:
            cfg = _copy_config_for_slot(slot.config, slot.mode)
        else:
            cfg = _copy_config_for_slot(base_config, slot.mode)
        # The official use case enforces its own wall clock; Studio's modes
        # check time only between steps (Direct and RAG not at all), so a
        # slow call ran far past the slot's max time and still counted.
        if slot.mode == MODE_RLM_OFFICIAL or not cfg.max_time_seconds:
            return self._run_slot(slot, content, query, cfg)
        return self._run_slot_within_wall_clock(slot, content, query, cfg)

    def _run_slot_within_wall_clock(
        self,
        slot: MatrixSlotDTO,
        content: str,
        query: str,
        cfg: RunConfigDTO,
    ) -> MatrixSlotResultDTO:
        """Run a Studio slot and give up waiting after ``cfg.max_time_seconds``.

        Mirrors the official engine's guard: the worker is a daemon thread
        that is not joined on timeout, so the slot gets its classified result
        on time; the run's own time budget stops the worker at its next step.
        """
        timeout = cfg.max_time_seconds or 0.0
        start = time.time()
        box: dict[str, MatrixSlotResultDTO] = {}

        def _target() -> None:
            box["result"] = self._run_slot(slot, content, query, cfg)

        worker = threading.Thread(target=_target, name=f"matrix-slot-{slot.slot_id}", daemon=True)
        worker.start()
        worker.join(timeout)
        if "result" in box:
            return box["result"]
        logger.warning(
            "Matrix slot %s exceeded its %.1fs max time; abandoning the worker thread",
            slot.slot_id,
            timeout,
        )
        return _timed_out_slot_result(slot, timeout, time.time() - start)

    def _run_slot(
        self,
        slot: MatrixSlotDTO,
        content: str,
        query: str,
        cfg: RunConfigDTO,
    ) -> MatrixSlotResultDTO:
        """Run one slot to completion; an exception becomes a failed result."""
        try:
            if slot.mode == "direct":
                result = RunDirectUseCase(slot.llm).execute(content, query, cfg)
            elif slot.mode == "rlm":
                assert slot.sandbox is not None  # validated
                result = RunRLMUseCase(slot.llm, slot.sandbox).execute(content, query, cfg)
            elif slot.mode == "rag":
                assert slot.embedder is not None  # validated
                assert slot.storage is not None  # validated
                result = RunRAGUseCase(slot.llm, slot.embedder, slot.storage).execute(
                    content, query, cfg
                )
            elif slot.mode == MODE_RLM_OFFICIAL:
                assert slot.engine is not None  # validated
                result = RunRLMOfficialUseCase(slot.engine).execute(content, query, cfg)
            else:  # pragma: no cover — validated upstream
                raise ValueError(f"Unsupported mode: {slot.mode}")
        except Exception as exc:
            logger.exception("Slot %s failed", slot.slot_id)
            return _failed_slot_result(slot, str(exc))

        return MatrixSlotResultDTO(
            slot_id=slot.slot_id,
            label=slot.label or _default_label(slot),
            mode=slot.mode,
            provider=slot.provider,
            model=slot.model,
            result=result,
        )

    @staticmethod
    def _rank(slots: list[MatrixSlotResultDTO], metric: RankingMetric) -> list[int]:
        """Return slot indices ordered best→worst by *metric*.

        Failed slots are always last, in their original input order.
        For the three Phase-5 perf metrics, slots lacking the relevant
        telemetry (e.g. ``median_ttft_ms`` is ``None``) sort last
        among successes, matching the "failed slots last" contract.
        """
        successes = [i for i, s in enumerate(slots) if s.result.success]
        failures = [i for i, s in enumerate(slots) if not s.result.success]

        def _perf_aggregates(i: int) -> dict:
            """Sum perf aggregates over the slot's raw trace dicts.

            Computed directly from the raw DTO keys (plain strings per
            spec v1.7 §3) rather than via a materialized TraceStep —
            the use-case layer cannot import the route-layer
            materializer without violating the dependency rule.
            """
            raw = slots[i].result.trace or []
            ttft_values: list[int] = []
            total_decode_ms = 0
            total_completion_tokens = 0
            total_prompt_tokens = 0
            total_cached_tokens = 0
            for entry in raw:
                if not isinstance(entry, dict):
                    continue
                ttft = entry.get("ttft_ms")
                if ttft is not None:
                    ttft_values.append(int(ttft))
                total_decode_ms += int(entry.get("decode_ms", 0) or 0)
                total_completion_tokens += int(entry.get("output_tokens", 0) or 0)
                total_prompt_tokens += int(entry.get("input_tokens", 0) or 0)
                total_cached_tokens += int(entry.get("cached_tokens", 0) or 0)
            median_ttft = None
            if ttft_values:
                ttft_values.sort()
                median_ttft = ttft_values[len(ttft_values) // 2]
            cache_hit_rate = 0.0
            if total_prompt_tokens > 0:
                cache_hit_rate = min(1.0, total_cached_tokens / total_prompt_tokens)
            return {
                "median_ttft_ms": median_ttft,
                "total_decode_ms": total_decode_ms,
                "total_completion_tokens": total_completion_tokens,
                "cache_hit_rate": cache_hit_rate,
            }

        def _key(i: int) -> tuple[float, int]:
            r = slots[i].result
            if metric == "cost":
                # Lower is better; break ties by earlier input order. A slot
                # whose engine reported no price (``cost_known`` False) is
                # not "free" — it sorts last among successes.
                if r.metadata.get(RESULT_KEY_COST_KNOWN, True) is False:
                    return (float("inf"), i)
                return (r.total_cost, i)
            if metric == "tokens":
                return (float(r.total_tokens), i)
            if metric == "latency":
                return (r.elapsed_time, i)
            if metric == "answer_per_cost":
                # Higher is better — invert so smaller sorts first.
                if r.metadata.get(RESULT_KEY_COST_KNOWN, True) is False:
                    return (float("inf"), i)
                answer_length = len(r.answer)
                if r.total_cost <= 0:
                    # Free slots are best; penalize only zero-length answers.
                    return (-float(answer_length), i)
                return (-answer_length / r.total_cost, i)
            if metric == "ttft":
                # Lower TTFT is better; slots without TTFT sort last
                # among successes (infinity sentinel).
                agg = _perf_aggregates(i)
                ttft = agg["median_ttft_ms"]
                if ttft is None:
                    return (float("inf"), i)
                return (float(ttft), i)
            if metric == "decode_tokens_per_sec":
                # Higher is better; slots with no decode phase sort last.
                agg = _perf_aggregates(i)
                decode_ms = agg["total_decode_ms"]
                if decode_ms <= 0:
                    return (float("inf"), i)
                tps = agg["total_completion_tokens"] / (decode_ms / 1000.0)
                return (-tps, i)
            if metric == "cache_hit_rate":
                # Higher is better; slots with no prompt tokens observed
                # have rate=0.0 and sort last.
                agg = _perf_aggregates(i)
                return (-agg["cache_hit_rate"], i)
            if metric == "judge_score":
                # Judge scores are produced out-of-band and re-ranked
                # client-side; the backend preserves input order here.
                return (0.0, i)
            raise ValueError(f"Unknown ranking metric: {metric!r}")  # pragma: no cover

        successes.sort(key=_key)
        return successes + failures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _copy_config_for_slot(base: RunConfigDTO, mode: str) -> RunConfigDTO:
    """Return a per-slot config copy with ``mode`` set.

    ``dataclasses.replace`` preserves every config field added in the
    future; ``extra`` is explicitly copied so slots do not share its
    mutable dict.
    """
    extra_copy: dict[str, Any] = dict(base.extra)
    return replace(base, mode=mode, extra=extra_copy)


def _default_label(slot: MatrixSlotDTO) -> str:
    parts = [p for p in (slot.provider, slot.model) if p]
    base = "/".join(parts) if parts else slot.slot_id
    return f"{base} · {slot_mode_label(slot.mode)}"


def _timed_out_slot_result(
    slot: MatrixSlotDTO, timeout: float, elapsed: float
) -> MatrixSlotResultDTO:
    """A slot abandoned at its max time: a classified timeout, cost unknown."""
    mode_name = MODE_DISPLAY_NAMES.get(slot.mode, slot.mode)
    return MatrixSlotResultDTO(
        slot_id=slot.slot_id,
        label=slot.label or _default_label(slot),
        mode=slot.mode,
        provider=slot.provider,
        model=slot.model,
        result=RunResultDTO(
            answer=(
                f"⚠️ **{mode_name} run timed out** after {timeout:.0f}s.\n\n"
                "The run did not finish within its max time.\n\n"
                "**How to fix:** raise **Max time** in the LLM Tuner (or **Max "
                "Execution Time (s)** on the Chat Provider's profile), or ask a "
                "more focused question."
            ),
            mode_used=slot.mode,
            success=False,
            error=f"{mode_name} run timed out after {timeout:.0f}s (max time)",
            elapsed_time=elapsed,
            # The abandoned run spent something this guard cannot see; a known
            # $0 would understate exactly the runs that ran longest.
            metadata={RESULT_KEY_COST_KNOWN: False},
        ),
    )


def _failed_slot_result(slot: MatrixSlotDTO, error: str) -> MatrixSlotResultDTO:
    return MatrixSlotResultDTO(
        slot_id=slot.slot_id,
        label=slot.label or _default_label(slot),
        mode=slot.mode,
        provider=slot.provider,
        model=slot.model,
        result=RunResultDTO(
            answer="",
            mode_used=slot.mode,
            success=False,
            error=error,
        ),
    )
