"""The four ways of spending the 100 submission lines compared in the paper,
and the construction of the production candidate pool from the raw pool.

Nothing here is new code. Each function is a copy of the code that produced
the paper's numbers, reduced to what runs without the video collection:

  allocate_rows ........ scripts/make_submission.py::allocate_rows (verbatim logic)
  alloc_A .. alloc_D ... scripts/e2_allocation_158.py::alloc_A .. alloc_D
  permute_by_scene_b ... scripts/make_submission.py::hoan_vi_theo_canh_b; the only
                         change is that the scene-B similarity of a keyframe is
                         looked up in a small per-item table shipped in
                         pools/scene_b.jsonl instead of in a similarity vector
                         over all 177,321 keyframes of the collection
  build_production_pool  scripts/e2_allocation_158.py::build_production_pools,
                         same change

The allocators themselves (ranking-based "hybrid" and coverage) and the
official scorer live unmodified in ``submission.py``.

    A  ranking-based (the production allocator of rounds 1-2), n_flat=30,
       AllocationPlan(breadth 1.0, depth 0.5, step 10)
    B  coverage-based (round 3), CoveragePlan(nhiet 0.02, sigma 30,
       nua_cua_so 6, luoi 5), tail filled by the A plan
    C  the first 100 distinct (video, keyframe) pairs of the candidate list
    D  one frame ladder per video, videos in order of first appearance
"""

from __future__ import annotations

from submission import (
    MAX_ROWS,
    AllocationPlan,
    Candidate,
    CoveragePlan,
    allocate_coverage_rows,
    allocate_hybrid_rows,
    frame_ladder,
)

DEFAULT_N_FLAT = 30
DEFAULT_DEPTH_COST = 0.5
LADDER_STEP = 10
PLAN_HYBRID = AllocationPlan(breadth_cost=1.0, depth_cost=DEFAULT_DEPTH_COST, step=LADDER_STEP)
ALLOCS = ("A", "B", "C", "D")
ALLOC_LABEL = {
    "A": "Ranking-based (rounds 1-2)",
    "B": "Coverage-based (round 3)",
    "C": "Top-100 candidates, one line each",
    "D": "One frame ladder per video",
}


def allocate_rows(cands, allocator: str, n_flat: int, plan: AllocationPlan):
    """One dispatch point, exactly as in the production pipeline."""
    if allocator == "coverage":
        return allocate_coverage_rows(
            cands,
            plan=CoveragePlan(budget=plan.budget),
            tail_n_flat=n_flat,
            tail_plan=plan,
        )
    return allocate_hybrid_rows(cands, n_flat=n_flat, plan=plan)


def alloc_A(cands):
    return allocate_rows(cands, "hybrid", DEFAULT_N_FLAT, PLAN_HYBRID)[:MAX_ROWS]


def alloc_B(cands):
    return allocate_rows(cands, "coverage", DEFAULT_N_FLAT, PLAN_HYBRID)[:MAX_ROWS]


def alloc_C(cands, budget: int = MAX_ROWS):
    rows, seen = [], set()
    for c in cands:
        key = (c.video_id, int(c.frame_idx))
        if key in seen:
            continue
        seen.add(key)
        rows.append(key)
        if len(rows) >= budget:
            break
    return rows


def alloc_D(cands, budget: int = MAX_ROWS, step: int = 10, rungs: int = 5):
    order, best = [], {}
    for c in cands:
        v = c.video_id
        if v not in best:
            order.append(v)
            best[v] = c
        elif float(c.score) > float(best[v].score):
            best[v] = c
    rows, seen = [], set()
    width = 1
    while len(rows) < budget:
        added = False
        for v in order:
            c = best[v]
            for f in frame_ladder(int(c.frame_idx), rungs * width, step, lo=0, hi=c.video_last_frame):
                key = (v, int(f))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(key)
                added = True
                if len(rows) >= budget:
                    return rows
        if not added:
            break
        width += 1
    return rows


ALLOC_FN = {"A": alloc_A, "B": alloc_B, "C": alloc_C, "D": alloc_D}


def run(fn, cands):
    return [(str(v), int(f)) for v, f in fn(cands)]


# ---------------------------------------------------------------------------
# production pool = raw pool + scene-B candidates + within-video permutation
# ---------------------------------------------------------------------------


def permute_by_scene_b(cands, sim_b, w: float = 1.0, alpha: float = 0.5,
                       so_video: int = 3, so_khung: int = 12):
    """Permute scores WITHIN each of the top videos by scene-B similarity.

    The multiset of scores of every video is unchanged; only which frame
    carries which score moves. ``sim_b`` maps (video_id, frame_idx) to the
    scene-B similarity of that keyframe (a missing key reads as 0.0, as a
    missing row did in production).
    """
    if sim_b is None or w <= 0 or len(cands) < 2:
        return cands

    thu_tu, theo_video = [], {}
    for i, c in enumerate(cands):
        if c.video_id not in theo_video:
            thu_tu.append(c.video_id)
            theo_video[c.video_id] = []
        theo_video[c.video_id].append(i)

    key_of: dict = {}
    for vid in thu_tu[:so_video]:
        pos = sorted(theo_video[vid], key=lambda i: -float(cands[i].score))[:so_khung]
        if len(pos) < 2:
            continue
        gia = []
        for i in pos:
            s = sim_b.get((cands[i].video_id, int(cands[i].frame_idx)))
            gia.append(float(s) if s is not None else 0.0)
        lo, hi = min(gia), max(gia)
        chuan = [(g - lo) / (hi - lo) for g in gia] if hi > lo else [0.0] * len(gia)
        thu = sorted(range(len(pos)), key=lambda k: int(cands[pos[k]].frame_idx))
        truoc = 0.0
        for k in thu:
            b = chuan[k]
            key_of[pos[k]] = float(cands[pos[k]].score) + w * b * (1.0 - alpha * truoc)
            truoc = b

    if len(key_of) < 2:
        return cands

    diem_moi = [float(c.score) for c in cands]
    for _vid, pos in theo_video.items():
        co = [i for i in pos if i in key_of]
        if len(co) < 2:
            continue
        cac_diem = sorted((float(cands[i].score) for i in co), reverse=True)
        for i, d in zip(sorted(co, key=lambda i: (-key_of[i], i)), cac_diem):
            diem_moi[i] = d
    return [Candidate(c.video_id, c.frame_idx, diem_moi[i], c.video_last_frame)
            for i, c in enumerate(cands)]


def build_production_pool(raw, scene_b_rec):
    """raw: list[Candidate]; scene_b_rec: one line of pools/scene_b.jsonl."""
    if not scene_b_rec or not scene_b_rec.get("scene_b_applied"):
        return list(raw)
    have = {(c.video_id, int(c.frame_idx)) for c in raw}
    extra = []
    for v, f, s, last in scene_b_rec["top100"]:
        key = (str(v), int(f))
        if key in have:
            continue
        have.add(key)
        extra.append(Candidate(key[0], key[1], float(s), last))
    sim_b = {(str(v), int(f)): float(s) for v, f, s in scene_b_rec["sim_b"]}
    return permute_by_scene_b(list(raw) + extra, sim_b)
