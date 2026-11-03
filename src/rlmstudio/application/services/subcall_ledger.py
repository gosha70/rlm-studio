"""Budget ledger for batched RLM subcalls.

The pre-batching implementation derived each child's allowance from
*completed* siblings, which is correct only while children run serially.
A batch launches children concurrently, so the ledger pre-reserves an
equal partition of the parent's remaining cost, tokens and controller
steps before launch and folds each child's actual usage back in on
completion.  Unused headroom is returned automatically, so a later batch
(including children still queued behind the pool's worker cap) can use it.

Wall-clock time is **not** divided among siblings.  Every allowance carries
the parent's common absolute deadline; a child that starts later simply has
less remaining time, and two parallel children may each use the same
interval.  The deadline is a between-steps guarantee: an in-flight provider
call may still run to the adapter's request timeout.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class ChildAllowance:
    """One child's reserved share of the parent's remaining budget.

    Attributes:
        index: Position in the batch (results stay in input order).
        max_cost: Reserved cost in USD, or ``None`` when the parent has no
            cost limit.
        max_tokens: Reserved input+output tokens, or ``None`` when unbounded.
        max_steps: LLM calls reserved for the child.  ``0`` means the parent
            has no budget left and the child must not start.
        deadline_epoch: The parent's absolute ``time.time()`` deadline, shared
            by every child; ``None`` when the parent has no time limit.
    """

    index: int
    max_cost: float | None
    max_tokens: int | None
    max_steps: int
    deadline_epoch: float | None


@dataclass(frozen=True)
class ChildUsage:
    """What one child actually spent."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    steps: int = 0
    model: str | None = None
    """The child adapter's active model, when it is known and consistent."""
    cost_known: bool = True
    """Whether ``total_cost`` is a measurement rather than zero by ignorance."""


class SubcallLedger:
    """Lock-protected reservation ledger for one parent run."""

    def __init__(
        self,
        *,
        max_cost: float | None,
        max_tokens: int | None,
        max_steps: int,
        spent_cost: float = 0.0,
        spent_tokens: int = 0,
        spent_steps: int = 0,
        deadline_epoch: float | None = None,
    ) -> None:
        self._max_cost = max_cost
        self._max_tokens = max_tokens
        self._max_steps = max_steps
        self._spent_cost = max(0.0, spent_cost)
        self._spent_tokens = max(0, spent_tokens)
        self._spent_steps = max(0, spent_steps)
        self._reserved_cost = 0.0
        self._reserved_tokens = 0
        self._reserved_steps = 0
        self._deadline_epoch = deadline_epoch
        self._lock = threading.Lock()

    @property
    def spent_cost(self) -> float:
        return self._spent_cost

    @property
    def spent_tokens(self) -> int:
        return self._spent_tokens

    @property
    def spent_steps(self) -> int:
        return self._spent_steps

    @property
    def over_budget(self) -> bool:
        """Whether actual spend has crossed a parent limit.

        Mirrors ``BudgetState.is_within``: tokens and cost are exhausted at
        equality; steps only once they exceed the limit.
        """
        with self._lock:
            if self._max_cost is not None and self._spent_cost >= self._max_cost:
                return True
            if self._max_tokens is not None and self._spent_tokens >= self._max_tokens:
                return True
            return self._max_steps is not None and self._spent_steps > self._max_steps

    def reserve(self, n_children: int) -> list[ChildAllowance]:
        """Reserve an equal partition of the remaining budget for *n_children*.

        The sum of the allowances never exceeds what is left after spend and
        outstanding reservations.  A later ``reserve`` call sees any headroom
        returned by ``release``.
        """
        if n_children < 1:
            raise ValueError("n_children must be positive")
        with self._lock:
            share_cost: float | None = None
            if self._max_cost is not None:
                available_cost = max(0.0, self._max_cost - self._spent_cost - self._reserved_cost)
                share_cost = available_cost / n_children

            share_tokens: int | None = None
            if self._max_tokens is not None:
                available_tokens = max(
                    0, self._max_tokens - self._spent_tokens - self._reserved_tokens
                )
                share_tokens = available_tokens // n_children

            available_steps = max(0, self._max_steps - self._spent_steps - self._reserved_steps)
            share_calls = available_steps // n_children

            self._reserved_cost += (share_cost or 0.0) * n_children
            self._reserved_tokens += (share_tokens or 0) * n_children
            self._reserved_steps += share_calls * n_children

            return [
                ChildAllowance(
                    index=i,
                    max_cost=share_cost,
                    max_tokens=share_tokens,
                    max_steps=share_calls,
                    deadline_epoch=self._deadline_epoch,
                )
                for i in range(n_children)
            ]

    def release(self, allowance: ChildAllowance, usage: ChildUsage) -> None:
        """Fold actual *usage* into spend and return the unused reservation."""
        with self._lock:
            self._reserved_cost = max(0.0, self._reserved_cost - (allowance.max_cost or 0.0))
            self._reserved_tokens = max(0, self._reserved_tokens - (allowance.max_tokens or 0))
            self._reserved_steps = max(0, self._reserved_steps - max(0, allowance.max_steps))

            self._spent_cost += max(0.0, usage.total_cost)
            self._spent_tokens += max(0, usage.input_tokens) + max(0, usage.output_tokens)
            self._spent_steps += max(0, usage.steps)
