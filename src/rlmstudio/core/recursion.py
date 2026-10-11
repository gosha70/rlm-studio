# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""Recursive controller for managing nested RLM calls.

Based on the RLM paper (arXiv:2512.24601), this module implements arbitrary-depth
recursion where sub-models can spawn their own sub-models to handle focused subtasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .budget import BudgetLimits, BudgetTracker
from .errors import BudgetExceeded

if TYPE_CHECKING:
    from .rlm import RLM, RLMResult


class RecursiveController:
    """Manages nested RLM calls up to a configurable max_depth.

    Each subcall creates a child RLM controller at ``depth + 1``.  A shared
    :class:`BudgetTracker` is threaded through the entire call-tree so that
    token, cost, and step limits apply globally.

    Args:
        controller_factory: Callable that produces a new :class:`RLM` instance
            for a given depth.  Receives ``depth`` as keyword argument.
        max_depth: Maximum recursion depth (default 1, matching the paper's evaluated regime).
        budget_tracker: Shared budget tracker across all recursion levels.
            Created automatically if not provided.

    Example::

        def factory(depth: int) -> RLM:
            return RLM(client=client, config=config)

        rc = RecursiveController(factory, max_depth=3)
        answer = rc.execute_subcall(content="...", query="...", depth=0)
    """

    def __init__(
        self,
        controller_factory: Any,  # Callable[[int], RLM] — Any to avoid import
        max_depth: int = 1,
        budget_tracker: BudgetTracker | None = None,
    ) -> None:
        self.controller_factory = controller_factory
        self.max_depth = max_depth

        if budget_tracker is None:
            budget_tracker = BudgetTracker(BudgetLimits(max_recursion_depth=max_depth))
        self.budget_tracker = budget_tracker

        # Trace of recursive calls: list of dicts capturing each subcall
        self.recursion_trace: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_subcall(
        self,
        content: str,
        query: str,
        depth: int,
        max_steps: int | None = None,
    ) -> str:
        """Execute a child RLM controller at *depth + 1*.

        Args:
            content: Text content for the child controller to analyse.
            query: Question for the child controller to answer.
            depth: Current recursion depth of the caller (0 = root).
            max_steps: Optional per-child step limit.

        Returns:
            The child controller's final answer string.

        Raises:
            BudgetExceeded: If ``depth + 1`` exceeds *max_depth* or any
                other budget limit is hit.
        """
        child_depth = depth + 1

        if child_depth > self.max_depth:
            raise BudgetExceeded(
                f"Maximum recursion depth ({self.max_depth}) exceeded. Current depth: {child_depth}"
            )

        # Track recursion in the shared budget tracker
        self.budget_tracker.enter_recursion()

        try:
            self.budget_tracker.check_limits()

            # Create a child controller
            child: RLM = self.controller_factory(depth=child_depth)

            # Wire the shared budget tracker into the child
            child._budget_tracker = self.budget_tracker

            result: RLMResult = child.run(prompt=content, query=query)

            # Record in the recursion trace
            self.recursion_trace.append(
                {
                    "depth": child_depth,
                    "query": query,
                    "content_length": len(content),
                    "success": result.success,
                    "answer": result.answer if result.success else None,
                    "error": result.error,
                    "steps": result.steps,
                }
            )

            if result.success:
                return result.answer
            return f"Error in subcall at depth {child_depth}: {result.error}"

        finally:
            self.budget_tracker.exit_recursion()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def create_subcall_function(self, parent_depth: int) -> Any:
        """Return a ``subcall(content, query, ...)`` callable bound to *parent_depth*.

        The returned function is suitable for injection into a
        :class:`PyReplEnv` so that LLM-generated code can call ``subcall()``
        directly.

        Args:
            parent_depth: The depth of the caller.

        Returns:
            A callable ``subcall(content, query, max_steps=None) -> str``.
        """

        def subcall(
            content: str,
            query: str,
            max_steps: int | None = None,
        ) -> str:
            """Spawn a sub-RLM to analyse *content* and answer *query*."""
            return self.execute_subcall(
                content=content,
                query=query,
                depth=parent_depth,
                max_steps=max_steps,
            )

        return subcall

    @property
    def current_depth(self) -> int:
        """The current recursion depth tracked by the budget tracker."""
        return self.budget_tracker.recursion_depth
