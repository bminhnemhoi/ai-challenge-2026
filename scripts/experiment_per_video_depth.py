"""Should the 100 rows ladder MORE keyframes per video, or FEWER more deeply?

The loss decomposition says 60% of the recoverable score is lost to frame
position, not to picking the wrong video. Then this, measured on the ground
truth:

    the keyframe NEAREST the true instant is rank 1 inside the right video
    only 48% of the time, but it is in that video's top 5 76% of the time.

    laddering around the video's rank-1 keyframe covers 55% of queries
    laddering around the truly nearest one would cover 98%

So the miss is a within-video ranking problem, and the fix does not need a
better model — it needs the budget spread over a video's top few keyframes
instead of concentrated on its best one.

The current allocator ranks candidates GLOBALLY: cost(i, d) = i + 0.5*d over a
flat list, so a video whose keyframes land at global ranks 1, 7 and 12 gets a
deep ladder on the first and progressively shallower ones after. This tries a
three-term cost that names the structure explicitly:

    cost(v, m, d) = A*v + B*m + C*d

        v  which video (0 = best)
        m  which keyframe inside that video (0 = its best)
        d  how far along that keyframe's offset ladder

A/B/C = 1/1/0.5 reproduces something close to the current behaviour. Lowering B
buys more keyframes per video; lowering C buys deeper ladders; lowering A buys
more videos. The three are swept against each other.

    python scripts/experiment_per_video_depth.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.make_submission import DEFAULT_DEPTH_COST, DEFAULT_N_FLAT, ranked_hits  # noqa: E402
from src.core.submission import (  # noqa: E402
    MAX_ROWS,
    AllocationPlan,
    Candidate,
    allocate_hybrid_rows,
    final_score,
    frame_ladder,
    r_score_kis,
)


def allocate_by_video(
    candidates: Sequence[Candidate],
    a: float = 1.0,
    b: float = 1.0,
    c: float = 0.5,
    step: int = 10,
    budget: int = MAX_ROWS,
    max_depth: int = 24,
    max_per_video: int = 8,
) -> List[Tuple[str, int]]:
    """Rows ordered by cost(v, m, d) = a*v + b*m + c*d.

    Videos keep the order their best keyframe had; keyframes keep the order they
    had inside their video. Only the SPENDING changes, never the ranking.
    """
    by_video: dict = defaultdict(list)
    order: List[str] = []
    for cand in candidates:
        if cand.video_id not in by_video:
            order.append(cand.video_id)
        by_video[cand.video_id].append(cand)

    slots = []
    for v, vid in enumerate(order):
        for m, cand in enumerate(by_video[vid][:max_per_video]):
            ladder = frame_ladder(
                cand.frame_idx, max_depth, step, lo=0, hi=cand.video_last_frame
            )
            for d, f in enumerate(ladder):
                slots.append((a * v + b * m + c * d, v, m, d, vid, f))
    slots.sort(key=lambda t: (t[0], t[1], t[2], t[3]))

    rows: List[Tuple[str, int]] = []
    seen = set()
    for _cost, _v, _m, _d, vid, f in slots:
        key = (vid, int(f))
        if key in seen:
            continue
        seen.add(key)
        rows.append(key)
        if len(rows) >= budget:
            break
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--draws", type=int, default=48)
    ap.add_argument("--step", type=int, default=10)
    args = ap.parse_args()

    from src.core.kis_engine import KISEngine

    print("loading index ...", flush=True)
    eng = KISEngine(args.data).load()
    gt = json.loads((Path(args.data) / "ground_truth.json").read_text(encoding="utf-8"))
    gt = [g for g in gt if g.get("video_id") in eng.last_frame]

    print("retrieving ...", flush=True)
    cands_of = []
    for g in gt:
        hits = ranked_hits(eng, g["kis_query_vi"], g.get("kis_query_en"))
        cands_of.append(
            [Candidate(h.video_id, h.frame_idx, h.score, h.video_last_frame) for h in hits]
        )

    kf: dict = {}
    for m in eng.metadata:
        kf.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
    for v in kf:
        kf[v] = np.array(sorted(kf[v]))

    def draw(seed):
        rng = np.random.default_rng(seed)
        out = []
        for g in gt:
            arr = kf[g["video_id"]]
            i = int(np.argmin(np.abs(arr - int(g["frame_idx"]))))
            lo = (arr[i] + arr[i - 1]) // 2 if i > 0 else arr[i] - 30
            hi = (arr[i] + arr[i + 1]) // 2 if i + 1 < len(arr) else arr[i] + 30
            out.append(int(rng.integers(lo, max(lo + 1, hi))))
        return out

    draws = [draw(9000 + s) for s in range(args.draws)]
    windows = [int(w) for w in args.windows.split(",")]

    def evaluate(rows_of):
        per_w = []
        for half in windows:
            tot = 0.0
            for qi, g in enumerate(gt):
                gv = g["video_id"]
                for truth in draws:
                    span = (truth[qi] - half, truth[qi] + half)
                    tot += final_score([r_score_kis(v, f, gv, span) for v, f in rows_of[qi]])
            per_w.append(tot / (len(gt) * len(draws)))
        return per_w

    plan = AllocationPlan(breadth_cost=1.0, depth_cost=DEFAULT_DEPTH_COST, step=args.step)
    base_rows = [
        allocate_hybrid_rows(c, n_flat=DEFAULT_N_FLAT, plan=plan)[:MAX_ROWS] for c in cands_of
    ]
    base_w = evaluate(base_rows)
    base = sum(base_w) / len(base_w)

    head = (f"{'A(video)':>9}{'B(kf)':>7}{'C(depth)':>9}"
            + "".join(f"{'W=' + str(w):>9}" for w in windows) + "     mean    delta")
    print("\n" + head)
    print("-" * len(head))
    print(f"{'hien tai (n_flat=30)':>25}" + "".join(f"{v:9.3f}" for v in base_w)
          + f"{base:9.3f}        -")

    best = (base, "hien tai")
    grid = [
        (1.0, 1.0, 0.5), (1.0, 1.0, 0.25), (1.0, 1.0, 1.0),
        (1.0, 0.5, 0.5), (1.0, 0.5, 0.25), (1.0, 0.5, 1.0),
        (1.0, 0.25, 0.5), (1.0, 0.25, 0.25),
        (1.0, 2.0, 0.5), (1.0, 2.0, 0.25),
        (0.5, 1.0, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.25),
        (2.0, 1.0, 0.5), (2.0, 0.5, 0.5), (2.0, 0.5, 0.25), (2.0, 0.25, 0.25),
        (3.0, 0.5, 0.25), (3.0, 1.0, 0.5),
    ]
    for a, b, c in grid:
        rows = [
            allocate_by_video(cs, a=a, b=b, c=c, step=args.step)[:MAX_ROWS] for cs in cands_of
        ]
        per_w = evaluate(rows)
        mean = sum(per_w) / len(per_w)
        if mean > best[0]:
            best = (mean, f"A={a} B={b} C={c}")
        print(f"{a:9.2f}{b:7.2f}{c:9.2f}" + "".join(f"{v:9.3f}" for v in per_w)
              + f"{mean:9.3f}  {100 * (mean / base - 1):+6.1f}%", flush=True)

    print(f"\nTot nhat: {best[1]} -> {best[0]:.3f}  ({100 * (best[0] / base - 1):+.1f}%)")
    if best[1] == "hien tai":
        print("Cach chia hien tai da toi uu; phan mat vi tri frame khong sua duoc bang phan bo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
