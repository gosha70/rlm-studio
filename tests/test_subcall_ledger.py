"""Tests for the batched-subcall budget ledger."""

from __future__ import annotations

import pytest

from rlmstudio.application.services.subcall_ledger import (
    ChildUsage,
    SubcallLedger,
)


class TestReservation:
    def test_equal_split_of_cost_tokens_and_steps(self) -> None:
        ledger = SubcallLedger(
            max_cost=1.0,
            max_tokens=1000,
            max_steps=10,
            spent_cost=0.2,
            spent_tokens=200,
            spent_steps=2,
            deadline_epoch=123.0,
        )
        allowances = ledger.reserve(4)

        assert len(allowances) == 4
        assert [a.index for a in allowances] == [0, 1, 2, 3]
        for allowance in allowances:
            assert allowance.max_cost == pytest.approx(0.2)
            assert allowance.max_tokens == 200
            # 8 remaining calls / 4 children = 2 calls each.
            assert allowance.max_steps == 2
            assert allowance.deadline_epoch == 123.0

    def test_release_returns_headroom_for_a_later_batch(self) -> None:
        ledger = SubcallLedger(
            max_cost=1.0,
            max_tokens=1000,
            max_steps=10,
            spent_cost=0.2,
            spent_tokens=200,
            spent_steps=2,
        )
        first = ledger.reserve(4)
        for allowance in first[:2]:
            ledger.release(
                allowance,
                ChildUsage(input_tokens=50, output_tokens=50, total_cost=0.05, steps=1),
            )

        later = ledger.reserve(2)
        assert later[0].max_cost == pytest.approx(0.15)
        assert later[0].max_tokens == 100
        assert later[0].max_steps == 1  # 2 remaining calls / 2 children

    def test_release_returns_unused_headroom(self) -> None:
        ledger = SubcallLedger(
            max_cost=1.0,
            max_tokens=1000,
            max_steps=10,
            spent_cost=0.0,
            spent_tokens=0,
            spent_steps=0,
        )
        first = ledger.reserve(2)
        ledger.release(
            first[0],
            ChildUsage(input_tokens=50, output_tokens=50, total_cost=0.1, steps=1),
        )
        # The released share minus actual spend is available again.
        later = ledger.reserve(2)
        assert later[0].max_cost == pytest.approx(0.2)
        assert later[0].max_tokens == 200
        assert later[0].max_steps == 2
        assert ledger.spent_cost == pytest.approx(0.1)
        assert ledger.spent_tokens == 100
        assert ledger.spent_steps == 1

    def test_no_limit_stays_unbounded(self) -> None:
        ledger = SubcallLedger(
            max_cost=None,
            max_tokens=None,
            max_steps=4,
        )
        allowances = ledger.reserve(2)
        assert allowances[0].max_cost is None
        assert allowances[0].max_tokens is None
        # 4 available calls / 2 children = 2 each.
        assert allowances[0].max_steps == 2

    def test_rejects_empty_batch(self) -> None:
        ledger = SubcallLedger(max_cost=None, max_tokens=None, max_steps=4)
        with pytest.raises(ValueError, match="positive"):
            ledger.reserve(0)

    def test_common_deadline_is_not_divided(self) -> None:
        ledger = SubcallLedger(
            max_cost=None,
            max_tokens=None,
            max_steps=10,
            deadline_epoch=500.0,
        )
        allowances = ledger.reserve(4)
        assert {a.deadline_epoch for a in allowances} == {500.0}
