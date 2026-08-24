"""Sweep the 100-row allocation against the official score.

The rulebook scores a KIS row only when the video matches AND the frame falls
inside the ground-truth window, and R@k is a max over the first k rows.  So the
only question that matters is: given a fixed budget of 100 rows, how should
they be split between *breadth* (more candidate videos, one keyframe each) and
*depth* (more frames around a candidate, walking the ±step ladder)?

Breadth wins when retrieval is unsure which video it is.  Depth wins when the
video is right but the keyframe is far from the window — and keyframes here sit
a median 62 frames apart while the window is ~10 frames wide, so a keyframe-only
row usually misses even on the correct video.  The balance is an empirical
question, and this measures it instead of guessing.

    python scripts/experiment_allocation.py            # default sweep
    python scripts/experiment_allocation.py --windows 6,10,20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.make_submission import ranked_hits  # noqa: E402
from src.core.submission import (  # noqa: E402
    MAX_ROWS,
    AllocationPlan,
    Candidate,
    allocate_hybrid_rows,
    final_score,
    r_score_kis,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--windows", default="6,10,20", help="half-widths to score at")
    ap.add_argument("--n-flat", default="10,20,30,40,50,60")
    ap.add_argument("--depth-cost", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--step", default="6,10,14")
    ap.add_argument(
        "--snapped",
        action="store_true",
        help="score against the raw ground truth, which is snapped to keyframes and "
        "therefore cannot measure depth at all (kept only to show the bias)",
    )
    ap.add_argument("--draws", type=int, default=8, help="random re-draws of the true instant")
    args = ap.parse_args()

    from src.core.kis_engine import KISEngine

    print("loading index ...", flush=True)
    eng = KISEngine(args.data).load()
    gt = json.loads((Path(args.data) / "ground_truth.json").read_text(encoding="utf-8"))
    gt = [g for g in gt if g.get("video_id") in eng.last_frame]
    print(f"{len(gt)} ground-truth queries usable\n", flush=True)

    # 93% of the recorded ground-truth frames ARE keyframes, because they were
    # annotated by picking one out of this very index.  The real answer key is a
    # moment a human marked in the source video, which lands anywhere in the ~60
    # frame gap between keyframes.  Scored against the snapped copy, a row placed
    # off-keyframe can only ever be wrong, so depth measures as worthless and the
    # sweep would recommend pure breadth — the exact mistake that produced a good
    # internal number and a bad leaderboard.  So the true instant is re-drawn
    # uniformly inside its keyframe gap, several times, and averaged.
    import numpy as np

    kf_by_video: dict = {}
    for m in eng.metadata:
        kf_by_video.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
    for v in kf_by_video:
        kf_by_video[v] = np.array(sorted(kf_by_video[v]))

    print("retrieving ...", flush=True)
    cached = []
    for g in gt:
        hits = ranked_hits(eng, g["kis_query_vi"], g.get("kis_query_en"))
        cands = [Candidate(h.video_id, h.frame_idx, h.score, h.video_last_frame) for h in hits]
        cached.append((g, cands))

    def truth_frames(seed: int):
        """One plausible answer key: the marked instant, un-snapped."""
        if args.snapped:
            return [int(g["frame_idx"]) for g, _ in cached]
        rng = np.random.default_rng(seed)
        out = []
        for g, _ in cached:
            a = kf_by_video[g["video_id"]]
            f = int(g["frame_idx"])
            i = int(np.argmin(np.abs(a - f)))
            lo = (a[i] + a[i - 1]) // 2 if i > 0 else a[i] - 30
            hi = (a[i] + a[i + 1]) // 2 if i + 1 < len(a) else a[i] + 30
            out.append(int(rng.integers(lo, max(lo + 1, hi))))
        return out

    draws = [truth_frames(1000 + s) for s in range(1 if args.snapped else args.draws)]

    windows = [int(x) for x in args.windows.split(",")]
    n_flats = [int(x) for x in args.n_flat.split(",")]
    depth_costs = [float(x) for x in args.depth_cost.split(",")]
    steps = [int(x) for x in args.step.split(",")]

    def score_config(n_flat: int, depth_cost: float, step: int, half: int) -> float:
        plan = AllocationPlan(breadth_cost=1.0, depth_cost=depth_cost, step=step, budget=MAX_ROWS)
        total = 0.0
        for i, (g, cands) in enumerate(cached):
            rows = allocate_hybrid_rows(cands, n_flat=n_flat, plan=plan)[:MAX_ROWS]
            gv = g["video_id"]
            for truth in draws:
                gf = truth[i]
                span = (gf - half, gf + half)
                total += final_score([r_score_kis(v, f, gv, span) for v, f in rows])
        return total / (len(cached) * len(draws))

    print(f"{'n_flat':>7} {'depth':>6} {'step':>5} " + "".join(f"{'W=' + str(w):>9}" for w in windows) + "     mean")
    print("-" * (20 + 9 * len(windows) + 9))
    results = []
    for step in steps:
        for n_flat in n_flats:
            for dc in depth_costs:
                per_w = [score_config(n_flat, dc, step, w) for w in windows]
                mean = sum(per_w) / len(per_w)
                results.append((mean, n_flat, dc, step, per_w))
                print(
                    f"{n_flat:7d} {dc:6.2f} {step:5d} "
                    + "".join(f"{v:9.3f}" for v in per_w)
                    + f"{mean:9.3f}",
                    flush=True,
                )

    results.sort(reverse=True)
    best = results[0]
    print("\nbest: n_flat=%d depth_cost=%.2f step=%d  mean %.3f" % (best[1], best[2], best[3], best[0]))
    from scripts.make_submission import DEFAULT_DEPTH_COST, DEFAULT_N_FLAT

    cur = next(
        (r for r in results if r[1] == DEFAULT_N_FLAT and r[2] == DEFAULT_DEPTH_COST and r[3] == 10),
        None,
    )
    if cur:
        print("shipping default: n_flat=%d depth_cost=%.2f step=10  mean %.3f  (%+.1f%%)"
              % (DEFAULT_N_FLAT, DEFAULT_DEPTH_COST, cur[0], 100 * (best[0] / cur[0] - 1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
