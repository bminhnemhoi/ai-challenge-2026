"""The other channels must actually get rows, not just get appended.

Round 1 lost query-p1-19 and query-p1-22 because only the spoken channel found
their videos, the pipeline appended those candidates to the end of the 400-long
retrieval list, and then neither allocator ever spent a row on them.  These
tests pin the structural fix: reserved tail slots, bounded cost, prefix intact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.submission import (  # noqa: E402
    MAX_ROWS,
    AllocationPlan,
    Candidate,
    allocate_coverage_rows,
    allocate_hybrid_rows,
    reserve_tail_rows,
)


def _rows(n=MAX_ROWS):
    return [("L21_V001", 100 + 10 * i) for i in range(n)]


def test_extras_land_at_the_very_end():
    extras = [("L26_V153", 1770), ("L28_V018", 451)]
    out = reserve_tail_rows(_rows(), extras)
    assert len(out) == MAX_ROWS
    assert out[-2:] == extras


def test_the_prefix_is_untouched():
    """R@1/R@5/R@20/R@50 must be bit-identical — only the cheap tail is spent."""
    base = _rows()
    out = reserve_tail_rows(base, [("L26_V153", 1770)] * 1)
    assert out[:50] == base[:50]
    assert out[:99] == base[:99]


def test_cost_is_exactly_the_number_of_extras():
    base = _rows()
    extras = [(f"L2{i}_V00{i}", 500 + i) for i in range(5)]
    out = reserve_tail_rows(base, extras)
    assert out[:95] == base[:95]
    assert out[95:] == extras


def test_an_extra_already_present_is_not_reserved_twice():
    base = _rows()
    out = reserve_tail_rows(base, [base[3], ("L29_V009", 42)])
    assert len(out) == MAX_ROWS
    assert out[-1] == ("L29_V009", 42)
    assert out[:99] == base[:99], "only the one genuinely new extra costs a row"


def test_duplicate_extras_collapse():
    out = reserve_tail_rows(_rows(), [("L29_V009", 42), ("L29_V009", 42)])
    assert out.count(("L29_V009", 42)) == 1


def test_no_extras_changes_nothing():
    base = _rows()
    assert reserve_tail_rows(base, []) == base


def test_more_extras_than_budget_are_truncated_not_crashing():
    extras = [(f"L21_V{i:03d}", i) for i in range(150)]
    out = reserve_tail_rows(_rows(), extras)
    assert len(out) == MAX_ROWS
    assert out == extras[:MAX_ROWS]


def test_a_short_row_list_is_still_topped_out_at_budget():
    out = reserve_tail_rows(_rows(20), [("L29_V009", 42)])
    assert len(out) == 21
    assert out[-1] == ("L29_V009", 42)


# ------------------------------------------------- the failure it exists to fix


@pytest.mark.parametrize("allocator", ["hybrid", "coverage"])
def test_a_tail_appended_candidate_gets_no_row_without_the_reservation(allocator):
    """The bug itself, pinned: appending to the candidate list is NOT enough.

    Both allocators spend the budget on the strongest candidates, so a
    speech-found video sitting at index 400 with a sentinel score receives
    nothing at all — which is exactly how two round-1 queries were lost.
    """
    pytest.importorskip("numpy")
    cands = [Candidate("L21_V001", 1000 + 7 * i, 0.40 - 0.0005 * i, 50_000) for i in range(400)]
    cands.append(Candidate("L26_V153", 1770, -1.0, 40_000))  # found only by speech

    plan = AllocationPlan(breadth_cost=1.0, depth_cost=0.5, step=10)
    rows = (allocate_hybrid_rows(cands, n_flat=30, plan=plan) if allocator == "hybrid"
            else allocate_coverage_rows(cands, tail_n_flat=30, tail_plan=plan))[:MAX_ROWS]
    assert not any(v == "L26_V153" for v, _f in rows), (
        "if this ever passes a row to the speech candidate on its own, "
        "the reservation may be reconsidered — until then it is mandatory"
    )

    fixed = reserve_tail_rows(rows, [("L26_V153", 1770)])
    assert fixed[-1] == ("L26_V153", 1770)
    assert len(fixed) == MAX_ROWS
