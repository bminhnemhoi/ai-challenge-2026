"""Promote the right KEYFRAME inside the right video, so it gets the deep ladder.

The loss decomposition, then the failure autopsy, point at one mechanism:

    22 of 60 ground-truth queries miss. Of those,
        14  the right keyframe IS among the candidates — median rank 5 inside
            its own video — but the ladder never reaches it
         5  the video is not in the 100 rows at all
         3  the right keyframe is not in the 400 candidates

The allocator spends cost(i, d) = i + 0.5*d, so a candidate at global rank 25
gets exactly one flat row and no ladder: it scores only if the answer lands
within +-5 of that one keyframe, about 18% of the time. A candidate at rank 1
gets a ladder reaching +-120.

So the fix is not more rows and not a different split — three sweeps have now
shown the current allocation is optimal. It is to get the RIGHT keyframe to the
front, where the depth already is.

Three ways to try that, all measured against the official formula:

    vlm-frame   ask the VLM to score each keyframe of the leading videos and
                reorder within each video by its verdict
    peak        prefer keyframes that are a local maximum of the similarity
                curve over their video's timeline, on the theory that a moment
                is a peak and a plateau is a scene
    spread      force the leading video's keyframes to be temporally spread,
                so the ladders cover different parts of the video instead of
                three of them piling onto the same two seconds

    python scripts/experiment_frame_promote.py --vlm
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

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
    r_score_kis,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--draws", type=int, default=48)
    ap.add_argument("--vlm", action="store_true", help="also try the VLM reordering (costs API calls)")
    ap.add_argument("--vlm-videos", type=int, default=3, help="leading videos whose frames the VLM scores")
    ap.add_argument("--vlm-frames", type=int, default=8, help="frames per video shown to the VLM")
    args = ap.parse_args()

    from src.core.kis_engine import KISEngine

    print("loading index ...", flush=True)
    eng = KISEngine(args.data).load()
    meta = {(m["video_id"], m["frame_idx"]): m for m in eng.metadata}
    gt = json.loads((Path(args.data) / "ground_truth.json").read_text(encoding="utf-8"))
    gt = [g for g in gt if g.get("video_id") in eng.last_frame]

    print("retrieving ...", flush=True)
    hits_of = [ranked_hits(eng, g["kis_query_vi"], g.get("kis_query_en")) for g in gt]

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

    draws = [draw(11000 + s) for s in range(args.draws)]
    windows = [int(w) for w in args.windows.split(",")]
    plan = AllocationPlan(breadth_cost=1.0, depth_cost=DEFAULT_DEPTH_COST, step=10)

    def evaluate(order_of):
        per_w = []
        for half in windows:
            tot = 0.0
            for qi, g in enumerate(gt):
                cs = [Candidate(h.video_id, h.frame_idx, h.score, h.video_last_frame)
                      for h in order_of[qi]]
                rows = allocate_hybrid_rows(cs, n_flat=DEFAULT_N_FLAT, plan=plan)[:MAX_ROWS]
                gv = g["video_id"]
                for truth in draws:
                    span = (truth[qi] - half, truth[qi] + half)
                    tot += final_score([r_score_kis(v, f, gv, span) for v, f in rows])
            per_w.append(tot / (len(gt) * len(draws)))
        return per_w

    base_w = evaluate(hits_of)
    base = sum(base_w) / len(base_w)
    head = f"{'variant':26s}" + "".join(f"{'W=' + str(w):>9}" for w in windows) + "     mean    delta"
    print("\n" + head)
    print("-" * len(head))
    print(f"{'hien tai':26s}" + "".join(f"{v:9.3f}" for v in base_w) + f"{base:9.3f}        -")

    results = [(base, "hien tai")]

    def report(name, order_of):
        per_w = evaluate(order_of)
        mean = sum(per_w) / len(per_w)
        results.append((mean, name))
        print(f"{name:26s}" + "".join(f"{v:9.3f}" for v in per_w)
              + f"{mean:9.3f}  {100 * (mean / base - 1):+6.1f}%", flush=True)

    # ---- spread: stop three ladders piling onto the same two seconds --------
    for min_gap in (30, 60, 120):
        out = []
        for hits in hits_of:
            kept, taken = [], defaultdict(list)
            for h in hits:
                if any(abs(h.frame_idx - f) < min_gap for f in taken[h.video_id]):
                    continue
                taken[h.video_id].append(h.frame_idx)
                kept.append(h)
            out.append(kept + [h for h in hits if h not in kept])
        report(f"spread >= {min_gap} frame", out)

    # ---- peak: a moment is a local maximum, a scene is a plateau ------------
    for w in (0.002, 0.005, 0.01):
        out = []
        for qi, hits in enumerate(hits_of):
            by_v = defaultdict(dict)
            for h in hits:
                by_v[h.video_id][int(h.frame_idx)] = h.score
            scored = []
            for i, h in enumerate(hits):
                arr = kf[h.video_id]
                j = int(np.searchsorted(arr, h.frame_idx))
                neigh = [by_v[h.video_id].get(int(arr[k]))
                         for k in (j - 1, j + 1) if 0 <= k < len(arr)]
                neigh = [x for x in neigh if x is not None]
                # how much this keyframe stands out from the ones beside it
                bump = (h.score - max(neigh)) if neigh else 0.0
                scored.append((h.score + w * max(bump, 0.0) / max(abs(h.score), 1e-6), i, h))
            scored.sort(key=lambda t: (-t[0], t[1]))
            out.append([t[2] for t in scored])
        report(f"peak w={w}", out)

    # ---- vlm: reorder each leading video's frames by what the model sees ----
    if args.vlm:
        from src.core.vlm import VLMJudge

        judge = VLMJudge(args.data)
        if not judge.ready:
            print("\n(bo qua VLM: khong co GEMINI_API_KEY)")
        else:
            print("\njudging frames ...", flush=True)
            for wgt in (0.02, 0.05):
                out = []
                for qi, (g, hits) in enumerate(zip(gt, hits_of)):
                    lead = []
                    seen_v: list = []
                    for h in hits:
                        if h.video_id not in seen_v:
                            if len(seen_v) >= args.vlm_videos:
                                continue
                            seen_v.append(h.video_id)
                        if sum(1 for x in lead if x.video_id == h.video_id) < args.vlm_frames:
                            lead.append(h)
                    cands = [(h.video_id, h.frame_idx, meta[(h.video_id, h.frame_idx)]["frame_filename"])
                             for h in lead if (h.video_id, h.frame_idx) in meta]
                    sc = judge.score(g["kis_query_vi"], cands)
                    scored = [
                        (h.score + wgt * sc.get((h.video_id, h.frame_idx), (0.0, ""))[0], i, h)
                        for i, h in enumerate(hits)
                    ]
                    scored.sort(key=lambda t: (-t[0], t[1]))
                    out.append([t[2] for t in scored])
                    if (qi + 1) % 20 == 0:
                        print(f"  {qi+1}/{len(gt)}  {judge.cost_note()}", flush=True)
                report(f"vlm-frame w={wgt}", out)
            print(judge.cost_note())

    results.sort(reverse=True)
    print(f"\nTot nhat: {results[0][1]} -> {results[0][0]:.3f} "
          f"({100*(results[0][0]/base-1):+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
