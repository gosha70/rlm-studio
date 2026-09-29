# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Benchmark harness: the matrix runner behind ``rlm-studio bench`` / BENCHMARKS.md,
plus the legacy strategy-based runner kept for its public API."""

from .corpus import materialise
from .dataset import BenchmarkCase, BenchmarkDataset, load_dataset, load_dataset_from_dict
from .matrix_runner import BenchmarkResults, MatrixBenchmarkRunner, SlotOutcome
from .report import BenchmarkReport, MatrixBenchmarkReport, regenerate_page
from .runner import BenchmarkRun, BenchmarkRunner, CaseResult
from .scoring import JudgeScorer, JudgeVerdict

__all__ = [
    "BenchmarkCase",
    "BenchmarkDataset",
    "load_dataset",
    "load_dataset_from_dict",
    "materialise",
    "MatrixBenchmarkRunner",
    "BenchmarkResults",
    "SlotOutcome",
    "MatrixBenchmarkReport",
    "regenerate_page",
    "JudgeScorer",
    "JudgeVerdict",
    "BenchmarkRunner",
    "BenchmarkRun",
    "CaseResult",
    "BenchmarkReport",
]
