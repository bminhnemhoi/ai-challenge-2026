"""Contract tests for the probability-coverage allocator.

The +15.3% TEST result (docs/SHIP_PHU_XAC_SUAT.md) belongs to a specific
deterministic procedure.  These tests pin the parts of that procedure a later
"harmless" edit could silently break: the row budget, determinism, the
4-decimal score-rounding boundary shared with the review page, and the
mandatory tail fill that keeps every submitted file at 100 rows.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.submission import (  # noqa: E402
    MAX_ROWS,
    AllocationPlan,
    Candidate,
    CoveragePlan,
    allocate_coverage_rows,
    allocate_hybrid_rows,
)

np = pytest.importorskip("numpy")


def _pool(seed: int, n: int = 60, n_videos: int = 12):
    """A retrieval-shaped pool: clustered per video, scores decaying by rank."""
    rng = random.Random(seed)
    videos = [
        (f"L{rng.randint(21, 30)}_V{rng.randint(1, 400):03d}", rng.randint(5_000, 60_000))
        for _ in range(n_videos)
    ]
    out = []
    for i in range(n):
        vid, last = videos[rng.randrange(n_videos)]
        centre = (hash(vid) % last) or 1000
        f = min(last, max(0, int(rng.gauss(centre, 800))))
        out.append(Candidate(vid, f, 0.30 - 0.002 * i + rng.random() * 0.001, last))
    return out


# ------------------------------------------------------------- row budget


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_returns_exactly_the_full_budget_on_a_realistic_pool(seed):
    rows = allocate_coverage_rows(_pool(seed))
    assert len(rows) == MAX_ROWS
    assert len(set(rows)) == MAX_ROWS, "duplicate (video, frame) rows"


def test_rows_stay_inside_the_video():
    cands = _pool(7)
    last_of = {c.video_id: c.video_last_frame for c in cands}
    for vid, frame in allocate_coverage_rows(cands):
        assert 0 <= frame <= last_of[vid], (vid, frame)


def test_a_smaller_budget_is_respected():
    rows = allocate_coverage_rows(_pool(1), plan=CoveragePlan(budget=50))
    assert len(rows) == 50


def test_empty_pool_yields_no_rows():
    assert allocate_coverage_rows([]) == []


# ------------------------------------------------------- mandatory tail fill


def test_tiny_pool_is_topped_up_by_the_hybrid_tail():
    """Greedy coverage on one candidate runs out of mass after a handful of
    rows; the verifier treats a short file as a truncation accident, so the
    hybrid ladder must fill in behind it (as far as one candidate allows)."""
    cands = [Candidate("L21_V001", 5_000, 0.9, 50_000)]
    rows = allocate_coverage_rows(cands)
    hybrid = allocate_hybrid_rows(cands, n_flat=20)
    assert len(rows) >= len(hybrid)
    assert len(set(rows)) == len(rows)


def test_pinned_short_video_still_fills_the_budget():
    """The measured failure shape: a pin on a short video gave coverage only
    13-21 rows before the tail fill existed."""
    cands = [Candidate("L21_V002", f, 0.5 - 0.01 * i, 900) for i, f in enumerate(range(0, 900, 60))]
    cands += [Candidate("L22_V003", 4_000 + 70 * i, 0.3 - 0.001 * i, 40_000) for i in range(45)]
    rows = allocate_coverage_rows(cands)
    assert len(rows) == MAX_ROWS
    assert len(set(rows)) == MAX_ROWS


# ------------------------------------------------------------ determinism


def test_identical_input_gives_identical_rows():
    cands = _pool(11)
    assert allocate_coverage_rows(cands) == allocate_coverage_rows(list(cands))


def test_scores_equal_after_4dp_rounding_give_identical_rows():
    """The review page embeds scores at 4 decimals; the Python side rounds at
    the allocator entry so both sides see the same numbers.  Any input whose
    scores round to the same 4-decimal values must therefore produce the very
    same 100 rows — this is the contract the JS port relies on."""
    cands = _pool(5)
    jittered = [
        Candidate(c.video_id, c.frame_idx, round(c.score, 4) + 3e-6, c.video_last_frame)
        for c in cands
    ]
    assert allocate_coverage_rows(cands) == allocate_coverage_rows(jittered)


def test_candidate_order_is_the_tiebreak_not_dict_hashing():
    """Videos are visited in candidate insertion order and every argmax keeps
    the first maximum, so a symmetric pool must resolve ties toward the
    earlier candidate."""
    a = [Candidate("L21_V010", 1_000, 0.5, 50_000), Candidate("L21_V011", 1_000, 0.5, 50_000)]
    rows = allocate_coverage_rows(a)
    assert rows[0][0] == "L21_V010"


# ------------------------------------------------------------- behaviour


def test_a_dominant_candidate_owns_row_one():
    """At nhiet 0.02 a 0.1 score gap is a ~150x prior ratio; row 1 must land
    on the dominant candidate's video, near its frame."""
    cands = [Candidate("L25_V100", 12_345, 0.45, 50_000)] + [
        Candidate(f"L26_V{i:03d}", 3_000 + 100 * i, 0.30, 40_000) for i in range(30)
    ]
    vid, frame = allocate_coverage_rows(cands)[0]
    assert vid == "L25_V100"
    assert abs(frame - 12_345) <= 30  # within one sigma of the prior peak


def test_flat_scores_spread_across_videos():
    """With an uninformative prior the greedy must hedge across videos rather
    than drilling one — that is the whole point of coverage."""
    cands = [
        Candidate(f"L27_V{i:03d}", 10_000, 0.30, 50_000) for i in range(10)
    ]
    rows = allocate_coverage_rows(cands, plan=CoveragePlan(budget=20))
    assert len({v for v, _f in rows[:10]}) == 10


def test_none_video_last_frame_means_unclamped():
    cands = [Candidate("L21_V001", 5_000, 0.9, None)]
    rows = allocate_coverage_rows(cands)
    assert rows, "a pool with unknown video end must still allocate"
    assert all(f >= 0 for _v, f in rows)


# ------------------------------------------------------------- dispatcher


def test_make_submission_dispatcher_switches_on_the_flag():
    from scripts.make_submission import allocate_rows

    cands = _pool(3)
    plan = AllocationPlan(breadth_cost=1.0, depth_cost=0.5, step=10)
    hybrid = allocate_rows(cands, "hybrid", 30, plan)
    coverage = allocate_rows(cands, "coverage", 30, plan)
    assert hybrid == allocate_hybrid_rows(cands, n_flat=30, plan=plan)
    assert coverage == allocate_coverage_rows(
        cands, plan=CoveragePlan(budget=plan.budget), tail_n_flat=30, tail_plan=plan
    )
    assert hybrid != coverage, "the flag must actually change the allocation"


def test_dispatcher_coverage_respects_the_plan_budget():
    """apply_picks caps pinned-video rows at --pin-budget through the same
    AllocationPlan; coverage must inherit that cap, not MAX_ROWS."""
    from scripts.make_submission import allocate_rows

    plan = AllocationPlan(breadth_cost=1.0, depth_cost=0.5, step=10, budget=40)
    rows = allocate_rows(_pool(9), "coverage", 30, plan)
    assert len(rows) == 40
