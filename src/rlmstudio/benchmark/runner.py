# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""Benchmark runner for executing strategies across dataset cases."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rlmstudio.strategies.base import LLMStrategy, StrategyResult
from rlmstudio.strategies.evaluator import EvaluationResult, MultiStrategyEvaluator

from .dataset import BenchmarkCase, BenchmarkDataset


@dataclass
class CaseResult:
    """Result of running all strategies on a single benchmark case."""

    case: BenchmarkCase
    evaluation: EvaluationResult
    elapsed_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.to_dict(),
            "evaluation": self.evaluation.get_summary(),
            "elapsed_time": self.elapsed_time,
        }


@dataclass
class BenchmarkRun:
    """Complete results from a benchmark run across all cases."""

    dataset_name: str
    strategy_names: list[str]
    case_results: list[CaseResult] = field(default_factory=list)
    total_elapsed_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def case_count(self) -> int:
        return len(self.case_results)

    @property
    def success_rate(self) -> dict[str, float]:
        """Per-strategy success rate across all cases."""
        rates: dict[str, float] = {}
        for name in self.strategy_names:
            total = 0
            successes = 0
            for cr in self.case_results:
                if name in cr.evaluation.results:
                    total += 1
                    if cr.evaluation.results[name].success:
                        successes += 1
            rates[name] = (successes / total) if total > 0 else 0.0
        return rates

    def get_strategy_metrics(self, strategy_name: str) -> dict[str, Any]:
        """Aggregate metrics for a single strategy across all cases."""
        results: list[StrategyResult] = []
        for cr in self.case_results:
            sr = cr.evaluation.results.get(strategy_name)
            if sr is not None:
                results.append(sr)

        if not results:
            return {"strategy": strategy_name, "cases": 0}

        successful = [r for r in results if r.success]
        total_tokens = sum(r.tokens.total_tokens for r in results)
        total_cost = sum(r.cost for r in results)
        total_time = sum(r.elapsed_time for r in results)

        return {
            "strategy": strategy_name,
            "cases": len(results),
            "successes": len(successful),
            "success_rate": len(successful) / len(results),
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "total_time": total_time,
            "avg_tokens": total_tokens / len(results),
            "avg_cost": total_cost / len(results),
            "avg_time": total_time / len(results),
            "avg_steps": sum(r.steps for r in results) / len(results),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "strategy_names": self.strategy_names,
            "case_count": self.case_count,
            "total_elapsed_time": self.total_elapsed_time,
            "success_rates": self.success_rate,
            "per_strategy": {name: self.get_strategy_metrics(name) for name in self.strategy_names},
            "cases": [cr.to_dict() for cr in self.case_results],
            "metadata": self.metadata,
        }


class BenchmarkRunner:
    """Run a benchmark dataset against a set of strategies.

    Usage::

        runner = BenchmarkRunner(strategies=[direct, rlm, rag])
        run = runner.run(dataset)
        report = run.to_dict()
    """

    def __init__(
        self,
        strategies: list[LLMStrategy],
        on_case_complete: Callable[[CaseResult], None] | None = None,
    ):
        self.strategies = strategies
        self.evaluator = MultiStrategyEvaluator(strategies)
        self.on_case_complete = on_case_complete

    def run(
        self,
        dataset: BenchmarkDataset,
        case_ids: list[str] | None = None,
    ) -> BenchmarkRun:
        """Run all strategies on each case in the dataset.

        Args:
            dataset: The benchmark dataset to evaluate.
            case_ids: Optional list of case IDs to run. If None, runs all.

        Returns:
            BenchmarkRun with results for all cases.
        """
        cases = dataset.cases
        if case_ids is not None:
            id_set = set(case_ids)
            cases = [c for c in cases if c.id in id_set]

        strategy_names = [s.name for s in self.strategies]
        bench_run = BenchmarkRun(
            dataset_name=dataset.name,
            strategy_names=strategy_names,
            metadata={"description": dataset.description},
        )

        run_start = time.time()
        for case in cases:
            case_start = time.time()
            evaluation = self.evaluator.evaluate(case.content, case.query)
            case_elapsed = time.time() - case_start

            case_result = CaseResult(
                case=case,
                evaluation=evaluation,
                elapsed_time=case_elapsed,
            )
            bench_run.case_results.append(case_result)

            if self.on_case_complete is not None:
                self.on_case_complete(case_result)

        bench_run.total_elapsed_time = time.time() - run_start
        return bench_run

    def run_case(self, case: BenchmarkCase) -> CaseResult:
        """Run all strategies on a single case."""
        start = time.time()
        evaluation = self.evaluator.evaluate(case.content, case.query)
        elapsed = time.time() - start
        return CaseResult(case=case, evaluation=evaluation, elapsed_time=elapsed)
