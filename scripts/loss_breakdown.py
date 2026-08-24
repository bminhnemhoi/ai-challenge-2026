"""Where exactly is the score being lost?

Optimising before measuring is how this project got to 5.8 with a good encoder.
The per-query score is a mean of R@k, and a query can fail in three separable
ways:

    A. the right VIDEO is never in the 100 rows at all      -> 0, nothing helps
    B. the right video IS there but never at a good rank    -> capped by rank
    C. the right video is at a good rank but every frame
       of it misses the answer window                       -> capped by frames

Those need completely different fixes. (A) is a retrieval problem — more
candidates, another modality. (B) is a reranking problem — the VLM, the operator.
(C) is a budget problem — the frame ladder, denser sampling, a human scrubbing
the video. Spending on the wrong one buys nothing.

This decomposes the loss on the ground truth, and reports the score each failure
mode would give if the OTHER two were fixed perfectly — which is the honest way
to size a fix before building it.

    python scripts/loss_breakdown.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.make_submission import DEFAULT_DEPTH_COST, DEFAULT_N_FLAT, ranked_hits  # noqa: E402
from src.core.submission import (  # noqa: E402
    MAX_ROWS,
    RANK_THRESHOLDS,
    AllocationPlan,
    Candidate,
    allocate_hybrid_rows,
    final_score,
    r_score_kis,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--draws", type=int, default=48)
    args = ap.parse_args()

    from src.core.kis_engine import KISEngine

    print("loading index ...", flush=True)
    eng = KISEngine(args.data).load()
    gt = json.loads((Path(args.data) / "ground_truth.json").read_text(encoding="utf-8"))
    gt = [g for g in gt if g.get("video_id") in eng.last_frame]

    print("retrieving ...", flush=True)
    plan = AllocationPlan(breadth_cost=1.0, depth_cost=DEFAULT_DEPTH_COST, step=10)
    rows_of, hits_of = [], []
    for g in gt:
        hits = ranked_hits(eng, g["kis_query_vi"], g.get("kis_query_en"))
        cands = [Candidate(h.video_id, h.frame_idx, h.score, h.video_last_frame) for h in hits]
        rows_of.append(allocate_hybrid_rows(cands, n_flat=DEFAULT_N_FLAT, plan=plan)[:MAX_ROWS])
        hits_of.append(hits)

    kf: dict = {}
    for m in eng.metadata:
        kf.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
    for v in kf:
        kf[v] = np.array(sorted(kf[v]))

    def draw(seed):
        rng = np.random.default_rng(seed)
        out = []
        for g in gt:
            a = kf[g["video_id"]]
            i = int(np.argmin(np.abs(a - int(g["frame_idx"]))))
            lo = (a[i] + a[i - 1]) // 2 if i > 0 else a[i] - 30
            hi = (a[i] + a[i + 1]) // 2 if i + 1 < len(a) else a[i] + 30
            out.append(int(rng.integers(lo, max(lo + 1, hi))))
        return out

    draws = [draw(8000 + s) for s in range(args.draws)]
    windows = [int(w) for w in args.windows.split(",")]

    # -------------------------------------------------------- classification
    n_missing = n_deep = 0
    video_rank = []
    for qi, g in enumerate(gt):
        rows = rows_of[qi]
        gv = g["video_id"]
        first = next((i for i, (v, _f) in enumerate(rows) if v == gv), None)
        video_rank.append(first)
        if first is None:
            n_missing += 1
        elif first >= 20:
            n_deep += 1

    print(f"\n{len(gt)} cau ground truth\n")
    print("=== A. VIDEO DUNG CO TRONG 100 DONG KHONG ===")
    at1 = sum(1 for r in video_rank if r == 0)
    at5 = sum(1 for r in video_rank if r is not None and r < 5)
    at20 = sum(1 for r in video_rank if r is not None and r < 20)
    at100 = sum(1 for r in video_rank if r is not None)
    print(f"  video dung o dong 1      : {at1:3d}/{len(gt)}  ({100*at1/len(gt):4.0f}%)  -> he so 1.0")
    print(f"  trong 5 dong dau         : {at5:3d}/{len(gt)}  ({100*at5/len(gt):4.0f}%)  -> he so 0.8")
    print(f"  trong 20 dong dau        : {at20:3d}/{len(gt)}  ({100*at20/len(gt):4.0f}%)  -> he so 0.6")
    print(f"  co mat dau do trong 100  : {at100:3d}/{len(gt)}  ({100*at100/len(gt):4.0f}%)")
    print(f"  KHONG CO -> chac chan 0  : {n_missing:3d}/{len(gt)}  ({100*n_missing/len(gt):4.0f}%)")

    # ------------------------------------------------------------ the score
    def score_all(row_lists):
        per_w = []
        for half in windows:
            tot = 0.0
            for qi, g in enumerate(gt):
                gv = g["video_id"]
                for truth in draws:
                    span = (truth[qi] - half, truth[qi] + half)
                    tot += final_score(
                        [r_score_kis(v, f, gv, span) for v, f in row_lists[qi]]
                    )
            per_w.append(tot / (len(gt) * len(draws)))
        return per_w

    base = score_all(rows_of)
    print(f"\n=== B. DIEM HIEN TAI ===")
    print("  " + "  ".join(f"W={w}: {s:.3f}" for w, s in zip(windows, base))
          + f"   TB {sum(base)/len(base):.3f}")

    # ------------------------------------------------- counterfactual ceilings
    print(f"\n=== C. NEU SUA TUNG THU MOT (tran ly thuyet) ===")

    # C1 perfect video ranking: the right video's rows moved to the front,
    # frames untouched. Isolates how much is lost to ranking alone.
    perfect_rank = []
    for qi, g in enumerate(gt):
        gv = g["video_id"]
        rows = rows_of[qi]
        mine = [r for r in rows if r[0] == gv]
        rest = [r for r in rows if r[0] != gv]
        perfect_rank.append((mine + rest)[:MAX_ROWS])
    s1 = score_all(perfect_rank)
    print(f"  xep hang video hoan hao   : {sum(s1)/len(s1):.3f}  "
          f"({100*(sum(s1)/sum(base)-1):+.0f}%)")

    # C2 perfect frames: keep the ranking, but replace each row of the right
    # video with the true instant. Isolates the frame-precision loss.
    def perfect_frames_for(truth_idx):
        out = []
        for qi, g in enumerate(gt):
            gv = g["video_id"]
            t = draws[truth_idx][qi]
            out.append([(v, t if v == gv else f) for v, f in rows_of[qi]])
        return out

    tot = 0.0
    for half in windows:
        for ti in range(len(draws)):
            rl = perfect_frames_for(ti)
            for qi, g in enumerate(gt):
                span = (draws[ti][qi] - half, draws[ti][qi] + half)
                tot += final_score(
                    [r_score_kis(v, f, g["video_id"], span) for v, f in rl[qi]]
                )
    s2 = tot / (len(gt) * len(draws) * len(windows))
    print(f"  frame hoan hao            : {s2:.3f}  ({100*(s2/(sum(base)/len(base))-1):+.0f}%)")

    # C3 both
    tot = 0.0
    for half in windows:
        for ti in range(len(draws)):
            for qi, g in enumerate(gt):
                t = draws[ti][qi]
                span = (t - half, t + half)
                rows = [(g["video_id"], t)] + [
                    r for r in rows_of[qi] if r[0] != g["video_id"]
                ]
                tot += final_score(
                    [r_score_kis(v, f, g["video_id"], span) for v, f in rows[:MAX_ROWS]]
                )
    s3 = tot / (len(gt) * len(draws) * len(windows))
    print(f"  ca hai (tran tuyet doi)   : {s3:.3f}")

    b = sum(base) / len(base)
    lost_rank = sum(s1) / len(s1) - b
    lost_frame = s2 - b
    print(f"\n=== D. PHAN BO PHAN DIEM DANG MAT ===")
    print(f"  do XEP HANG video : {lost_rank:.3f}  ({100*lost_rank/(s3-b):3.0f}% phan mat duoc)")
    print(f"  do VI TRI frame   : {lost_frame:.3f}  ({100*lost_frame/(s3-b):3.0f}% phan mat duoc)")
    print(f"  (hai phan chong lan nhau; tong khong bang 100%)")

    # ------------------------------------------------ per-query worst offenders
    print(f"\n=== E. 10 CAU MAT DIEM NHAT ===")
    per_q = []
    for qi, g in enumerate(gt):
        gv = g["video_id"]
        tot = 0.0
        for half in windows:
            for truth in draws:
                span = (truth[qi] - half, truth[qi] + half)
                tot += final_score([r_score_kis(v, f, gv, span) for v, f in rows_of[qi]])
        per_q.append((tot / (len(draws) * len(windows)), qi))
    per_q.sort()
    print(f"  {'diem':>6} {'hang video':>11}  cau")
    for s, qi in per_q[:10]:
        r = video_rank[qi]
        rr = "KHONG CO" if r is None else f"{r+1}"
        print(f"  {s:6.3f} {rr:>11}  {gt[qi]['kis_query_vi'][:62]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
