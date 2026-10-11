# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0
"""
MemoryMonitor - Track memory usage during execution.
"""

import psutil


class MemoryMonitor:
    """
    Monitor and track memory usage during execution.

    Responsibilities:
    - Capture memory snapshots
    - Track peak memory usage
    - Calculate memory delta from baseline
    - Provide timeline of memory usage
    """

    def __init__(self) -> None:
        """Initialize MemoryMonitor with baseline memory."""
        self.process = psutil.Process()
        self.start_memory_mb = self._get_memory_mb()
        self.peak_memory_mb = self.start_memory_mb
        self.measurements: list[float] = []
        self.baseline_memory_mb = self.start_memory_mb

    def _get_memory_mb(self) -> float:
        """
        Get current process memory in MB.

        Returns:
            Memory in MB

        Implementation notes:
        - Should use RSS (resident set size)
        - Should be cross-platform (Windows/macOS/Linux)
        """
        try:
            memory_bytes = self.process.memory_info().rss
            return float(memory_bytes) / 1024.0 / 1024.0
        except Exception:
            return 0.0

    def capture(self) -> float:
        """
        Capture current memory and update peak.

        Returns:
            Current memory usage relative to start (MB)
        """
        current = self._get_memory_mb()
        self.peak_memory_mb = max(self.peak_memory_mb, current)
        self.measurements.append(current)
        return self.current_mb()

    def current_mb(self) -> float:
        """
        Get current memory usage relative to baseline.

        Returns:
            Memory delta in MB
        """
        return self._get_memory_mb() - self.baseline_memory_mb

    def peak_mb(self) -> float:
        """
        Get peak memory usage relative to baseline.

        Returns:
            Peak memory delta in MB
        """
        return self.peak_memory_mb - self.baseline_memory_mb

    def reset(self) -> None:
        """Reset baseline (call before execution)."""
        self.baseline_memory_mb = self._get_memory_mb()
        self.peak_memory_mb = self.baseline_memory_mb
        self.measurements = []

    def get_timeline(self) -> list[dict[str, float]]:
        """
        Get memory timeline for charting.

        Returns:
            List of dicts with 'time_ms' and 'memory_mb' keys

        Implementation notes:
        - Should convert to milliseconds from capture time
        - Should use relative memory (delta from baseline)
        - Should be suitable for Plotly charting
        """
        timeline = []
        for i, memory_mb in enumerate(self.measurements):
            timeline.append(
                {
                    "time_ms": i * 100,  # Assume 100ms between captures
                    "memory_mb": memory_mb - self.baseline_memory_mb,
                }
            )
        return timeline

    def get_stats(self) -> dict[str, float]:
        """
        Get memory statistics.

        Returns:
            Dict with 'current', 'peak', 'average', 'max_growth'
        """
        if not self.measurements:
            return {
                "current": 0.0,
                "peak": 0.0,
                "average": 0.0,
                "max_growth": 0.0,
            }

        baseline = self.baseline_memory_mb
        current = self._get_memory_mb() - baseline
        peak = self.peak_memory_mb - baseline
        average = sum(m - baseline for m in self.measurements) / len(self.measurements)

        return {
            "current": current,
            "peak": peak,
            "average": average,
            "max_growth": peak,
        }
