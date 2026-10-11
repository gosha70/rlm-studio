# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""
Execution tracing for RLM.

This module provides tracing capabilities to capture and visualize the execution
trajectory of RLM agents, including all actions, results, and metadata.
"""

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

ActionType = Literal["inspect", "subcall", "final", "error"]


@dataclass
class TraceStep:
    """
    A single step in the RLM execution trace.

    Attributes:
        index: Step number (0-indexed)
        action_type: Type of action taken
        code: Python code executed (for inspect/subcall)
        output: Result of code execution
        tokens_used: Number of tokens consumed
        timestamp: When this step occurred
        recursion_depth: Current depth in recursion tree (0 = root)
        cost: Estimated cost for this step
        duration: Time taken for this step (seconds)
        model: Model used for this step
        raw_response: Full LLM response text
        error: Error message if step failed
    """

    index: int
    action_type: ActionType
    code: str | None = None
    output: str | None = None
    tokens_used: int = 0
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    recursion_depth: int = 0
    cost: float = 0.0
    duration: float = 0.0
    model: str | None = None
    raw_response: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        """Human-readable string representation."""
        indent = "  " * self.recursion_depth
        result = f"{indent}[Step {self.index}] {self.action_type.upper()}"
        if self.code:
            result += f"\n{indent}  Code: {self.code[:100]}{'...' if len(self.code) > 100 else ''}"
        if self.output:
            result += (
                f"\n{indent}  Output: {self.output[:100]}{'...' if len(self.output) > 100 else ''}"
            )
        if self.error:
            result += f"\n{indent}  Error: {self.error}"
        result += f"\n{indent}  Tokens: {self.tokens_used} | Cost: ${self.cost:.4f} | Time: {self.duration:.2f}s"
        return result


class ExecutionTrace:
    """
    Complete execution trace for an RLM run.

    This captures all steps, metadata, and provides export functionality.
    """

    def __init__(self) -> None:
        self.steps: list[TraceStep] = []
        self.start_time: float = datetime.now().timestamp()
        self.end_time: float | None = None
        self.metadata: dict[str, Any] = {}

    def add_step(self, step: TraceStep) -> None:
        """Add a step to the trace."""
        self.steps.append(step)

    def finalize(self) -> None:
        """Mark the trace as complete."""
        self.end_time = datetime.now().timestamp()

    @property
    def total_duration(self) -> float:
        """Total execution time in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return datetime.now().timestamp() - self.start_time

    @property
    def total_tokens(self) -> int:
        """Total tokens used across all steps."""
        return sum(step.tokens_used for step in self.steps)

    @property
    def total_cost(self) -> float:
        """Total cost across all steps."""
        return sum(step.cost for step in self.steps)

    @property
    def max_recursion_depth(self) -> int:
        """Maximum recursion depth reached."""
        if not self.steps:
            return 0
        return max(step.recursion_depth for step in self.steps)

    def get_step(self, index: int) -> TraceStep | None:
        """Get a specific step by index."""
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None

    def filter_by_type(self, action_type: ActionType) -> list[TraceStep]:
        """Get all steps of a specific type."""
        return [step for step in self.steps if step.action_type == action_type]

    def filter_by_depth(self, depth: int) -> list[TraceStep]:
        """Get all steps at a specific recursion depth."""
        return [step for step in self.steps if step.recursion_depth == depth]

    def to_dict(self) -> dict[str, Any]:
        """Convert entire trace to dictionary."""
        return {
            "steps": [step.to_dict() for step in self.steps],
            "metadata": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "total_duration": self.total_duration,
                "total_tokens": self.total_tokens,
                "total_cost": self.total_cost,
                "total_steps": len(self.steps),
                "max_recursion_depth": self.max_recursion_depth,
                **self.metadata,
            },
        }

    def to_jsonl(self, filepath: str) -> None:
        """
        Export trace to JSONL format (one JSON object per line).

        This format is useful for streaming processing and log analysis tools.

        Args:
            filepath: Path to save JSONL file
        """
        with open(filepath, "w") as f:
            # Write metadata as first line
            f.write(json.dumps({"metadata": self.to_dict()["metadata"]}) + "\n")

            # Write each step as a separate line
            for step in self.steps:
                f.write(json.dumps(step.to_dict()) + "\n")

    def to_json(self, filepath: str, pretty: bool = True) -> None:
        """
        Export trace to JSON format.

        Args:
            filepath: Path to save JSON file
            pretty: If True, format with indentation
        """
        with open(filepath, "w") as f:
            if pretty:
                json.dump(self.to_dict(), f, indent=2)
            else:
                json.dump(self.to_dict(), f)

    def to_markdown(self, filepath: str) -> None:
        """
        Export trace to Markdown format for human reading.

        Args:
            filepath: Path to save Markdown file
        """
        lines = [
            "# RLM Execution Trace",
            "",
            "## Summary",
            "",
            f"- **Total Steps:** {len(self.steps)}",
            f"- **Total Duration:** {self.total_duration:.2f}s",
            f"- **Total Tokens:** {self.total_tokens:,}",
            f"- **Total Cost:** ${self.total_cost:.4f}",
            f"- **Max Recursion Depth:** {self.max_recursion_depth}",
            "",
            "## Execution Steps",
            "",
        ]

        for step in self.steps:
            indent = "  " * step.recursion_depth
            lines.append(f"{indent}### Step {step.index}: {step.action_type.upper()}")
            lines.append("")

            if step.code:
                lines.append(f"{indent}**Code:**")
                lines.append(f"{indent}```python")
                lines.append(f"{indent}{step.code}")
                lines.append(f"{indent}```")
                lines.append("")

            if step.output:
                lines.append(f"{indent}**Output:**")
                lines.append(f"{indent}```")
                lines.append(f"{indent}{step.output}")
                lines.append(f"{indent}```")
                lines.append("")

            if step.error:
                lines.append(f"{indent}**Error:**")
                lines.append(f"{indent}```")
                lines.append(f"{indent}{step.error}")
                lines.append(f"{indent}```")
                lines.append("")

            lines.append(f"{indent}**Metrics:**")
            lines.append(f"{indent}- Tokens: {step.tokens_used}")
            lines.append(f"{indent}- Cost: ${step.cost:.4f}")
            lines.append(f"{indent}- Duration: {step.duration:.2f}s")
            if step.model:
                lines.append(f"{indent}- Model: {step.model}")
            lines.append("")

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

    def __len__(self) -> int:
        """Number of steps in trace."""
        return len(self.steps)

    def __getitem__(self, index: int) -> TraceStep:
        """Get step by index."""
        return self.steps[index]

    def __iter__(self) -> Iterator[TraceStep]:
        """Iterate over steps."""
        return iter(self.steps)

    def __str__(self) -> str:
        """Human-readable representation."""
        return "\n".join(str(step) for step in self.steps)

    def print_summary(self) -> None:
        """Print a summary of the trace to console."""
        print("=" * 60)
        print("RLM EXECUTION TRACE SUMMARY")
        print("=" * 60)
        print(f"Total Steps: {len(self.steps)}")
        print(f"Duration: {self.total_duration:.2f}s")
        print(f"Tokens: {self.total_tokens:,}")
        print(f"Cost: ${self.total_cost:.4f}")
        print(f"Max Depth: {self.max_recursion_depth}")
        print("=" * 60)

        # Show action type breakdown
        inspect_count = len(self.filter_by_type("inspect"))
        subcall_count = len(self.filter_by_type("subcall"))
        final_count = len(self.filter_by_type("final"))
        error_count = len(self.filter_by_type("error"))

        print("\nAction Breakdown:")
        if inspect_count > 0:
            print(f"  Inspect: {inspect_count}")
        if subcall_count > 0:
            print(f"  Subcall: {subcall_count}")
        if final_count > 0:
            print(f"  Final: {final_count}")
        if error_count > 0:
            print(f"  Error: {error_count}")
        print("=" * 60)


def load_trace_from_jsonl(filepath: str) -> ExecutionTrace:
    """
    Load a trace from JSONL file.

    Args:
        filepath: Path to JSONL file

    Returns:
        ExecutionTrace object
    """
    trace = ExecutionTrace()

    with open(filepath) as f:
        lines = f.readlines()

        # First line is metadata
        if lines:
            metadata_line = json.loads(lines[0])
            if "metadata" in metadata_line:
                meta = metadata_line["metadata"]
                trace.start_time = meta.get("start_time", trace.start_time)
                trace.end_time = meta.get("end_time")
                trace.metadata = {
                    k: v for k, v in meta.items() if k not in ["start_time", "end_time"]
                }

        # Remaining lines are steps
        for line in lines[1:]:
            step_dict = json.loads(line)
            step = TraceStep(**step_dict)
            trace.add_step(step)

    return trace


def load_trace_from_json(filepath: str) -> ExecutionTrace:
    """
    Load a trace from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        ExecutionTrace object
    """
    trace = ExecutionTrace()

    with open(filepath) as f:
        data = json.load(f)

        # Load metadata
        if "metadata" in data:
            meta = data["metadata"]
            trace.start_time = meta.get("start_time", trace.start_time)
            trace.end_time = meta.get("end_time")
            trace.metadata = {k: v for k, v in meta.items() if k not in ["start_time", "end_time"]}

        # Load steps
        if "steps" in data:
            for step_dict in data["steps"]:
                step = TraceStep(**step_dict)
                trace.add_step(step)

    return trace
