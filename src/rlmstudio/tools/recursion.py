# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""Recursion tools for RLM - subcall functionality.

Based on the RLM paper, allows spawning sub-RLM instances to handle
complex subtasks within the main RLM execution.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.rlm import RLM


def create_subcall(rlm_instance: "RLM") -> "Callable[[str, str, int | None], str]":
    """
    Create a subcall function bound to an RLM instance.

    This factory function creates a subcall function that has access to
    the parent RLM's client and budget tracker.

    Args:
        rlm_instance: The parent RLM instance

    Returns:
        subcall function bound to this RLM instance
    """

    def subcall(content: str, query: str, max_steps: int | None = None) -> str:
        """
        Spawn a sub-RLM to analyze content and answer a query.

        This allows the LLM to decompose complex tasks by creating focused
        sub-analyses. The sub-RLM runs independently with its own execution
        loop but shares the parent's budget and recursion tracking.

        Args:
            content: Text content for the sub-RLM to analyze
            query: Question for the sub-RLM to answer
            max_steps: Optional max steps for sub-RLM (defaults to parent's remaining steps)

        Returns:
            str: The sub-RLM's final answer

        Raises:
            BudgetExceeded: If recursion depth limit exceeded or other budget limits hit

        Example:
            ```python
            # In main RLM analyzing a book:
            chapter_text = peek(1000, 5000)
            summary = subcall(chapter_text, "Summarize this chapter in 2 sentences")
            print(f"Chapter summary: {summary}")
            ```
        """
        # Check if we have a budget tracker and respect recursion limits
        if rlm_instance._budget_tracker:
            tracker = rlm_instance._budget_tracker

            # Enter recursion context
            tracker.enter_recursion()

            try:
                # Check recursion depth limit
                tracker.check_limits()

                # Determine max steps for sub-RLM
                if max_steps is None:
                    # Use parent's remaining steps
                    remaining_steps = (
                        tracker.limits.max_steps - tracker.steps
                        if tracker.limits.max_steps
                        else None
                    )
                    if remaining_steps is not None:
                        max_steps = max(1, remaining_steps)  # At least 1 step

                # Create sub-RLM with shared client but own environment
                from ..config import ExecutionConfig, RLMConfig
                from ..core.rlm import RLM

                # Clone parent config but update max_steps
                sub_config = RLMConfig(
                    security=rlm_instance.config.security,
                    execution=ExecutionConfig(
                        default_timeout=rlm_instance.config.execution.default_timeout,
                        max_output_chars=rlm_instance.config.execution.max_output_chars,
                        default_safe_mode=rlm_instance.config.execution.default_safe_mode,
                        max_steps=max_steps or rlm_instance.config.execution.max_steps,
                    ),
                    monitoring=rlm_instance.config.monitoring,
                )

                sub_rlm = RLM(
                    client=rlm_instance.client,
                    config=sub_config,
                )

                # Share the budget tracker
                sub_rlm._budget_tracker = tracker

                # Run sub-RLM
                result = sub_rlm.run(prompt=content, query=query)

                if result.success:
                    return result.answer
                else:
                    return f"Error in subcall: {result.error}"

            finally:
                # Exit recursion context
                tracker.exit_recursion()
        else:
            # No budget tracker, run without recursion tracking
            from ..core.rlm import RLM

            sub_rlm = RLM(
                client=rlm_instance.client,
                config=rlm_instance.config,
            )
            result = sub_rlm.run(prompt=content, query=query)
            return result.answer if result.success else f"Error: {result.error}"

    # Add metadata
    subcall.__doc__ = """Spawn a sub-RLM to analyze content and answer a query.

    Args:
        content: Text to analyze
        query: Question to answer
        max_steps: Optional max steps (defaults to parent's remaining steps)

    Returns:
        str: The answer from the sub-RLM
    """

    return subcall
