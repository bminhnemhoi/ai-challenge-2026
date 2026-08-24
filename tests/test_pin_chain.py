"""Hedging a KIS query across several takes of the same action.

A real round asked for the moment a performer completes one specific action.
The video contains eight takes of that same action, minutes apart, and a still
cannot say which take the answer key means — every one matches the sentence.

R@k is a max over the first k rows, so a second and third guess at ranks 2 and 3
cost one row each out of a hundred and can only add. What must NOT change is the
single-frame path: putting the ladder immediately behind one confirmed frame was
measured at +24% and a chain must not quietly undo it.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.apply_picks import pin_plan  # noqa: E402
from src.core.submission import AllocationPlan, Candidate, allocate_hybrid_rows  # noqa: E402

LAST = 20000
OTHERS = [Candidate("L99_V001", f, 0.5, LAST) for f in (2000, 4000, 6000, 8000)]


def _rows(cands, n_flat, budget=50):
    return allocate_hybrid_rows(
        cands, n_flat=n_flat,
        plan=AllocationPlan(breadth_cost=1.0, depth_cost=0.5, step=10, budget=budget),
    )


def test_one_confirmed_frame_still_gets_its_ladder_immediately():
    cands, n_flat = pin_plan("L99_V001", LAST, None, 16263, OTHERS, 30)
    assert n_flat == 1
    rows = _rows(cands, n_flat)
    assert rows[0] == ("L99_V001", 16263)
    # the rows right behind it are the ladder, not other keyframes 4000 away
    assert {f for _v, f in rows[1:3]} == {16253, 16273}


def test_every_take_in_a_chain_gets_one_of_the_leading_rows():
    chain = [16263, 17534, 18240, 14732]
    cands, n_flat = pin_plan("L99_V001", LAST, chain, 16263, OTHERS, 30)
    assert n_flat == 4
    rows = _rows(cands, n_flat)
    assert [f for _v, f in rows[:4]] == chain, "order given is the order submitted"


def test_the_first_take_keeps_the_deep_ladder():
    chain = [16263, 17534, 18240]
    cands, n_flat = pin_plan("L99_V001", LAST, chain, 16263, OTHERS, 30)
    rows = _rows(cands, n_flat, budget=60)
    around_first = [f for v, f in rows if abs(f - 16263) <= 120]
    around_second = [f for v, f in rows if abs(f - 17534) <= 120]
    assert len(around_first) > len(around_second) >= 1


def test_a_chain_never_repeats_a_frame():
    cands, n_flat = pin_plan("L99_V001", LAST, [16263, 16263, 17534], 16263, OTHERS, 30)
    assert n_flat == 2
    assert [c.frame_idx for c in cands[:2]] == [16263, 17534]


def test_a_frame_named_in_the_chain_is_not_also_kept_as_an_ordinary_candidate():
    others = OTHERS + [Candidate("L99_V001", 17534, 0.9, LAST)]
    cands, _ = pin_plan("L99_V001", LAST, [16263, 17534], 16263, others, 30)
    assert [c.frame_idx for c in cands].count(17534) == 1


def test_no_frame_given_falls_back_to_breadth_over_the_video():
    cands, n_flat = pin_plan("L99_V001", LAST, None, None, OTHERS, 30)
    assert cands == OTHERS
    assert n_flat == len(OTHERS)
