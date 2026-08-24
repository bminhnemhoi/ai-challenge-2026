"""Measure the OFFICIAL Final Score on the 60-sample ground truth.

`scripts/evaluate_official_pipeline.py` reports how highly the right VIDEO is
ranked.  That is not the competition metric: rules 2.1.1 require the video AND
a frame inside the answer interval [s, e], and rules 2.2 average R@k over
k in {1, 5, 20, 50, 100}.  A video-level number can look excellent while the
official score is low, which is exactly the gap this script exists to close.

The answer-window width is not published, so the score is reported across a
range of plausible widths.  The window is also PLACED realistically: the GT
file stores a keyframe index, but the organisers annotate the moment on the
original video timeline, so centring the window on the stored keyframe would
flatter any strategy that submits keyframe indices.  Instead the moment is
drawn uniformly within half a local keyframe gap, averaged over many draws.

    python scripts/evaluate_official.py                # ship-ready config
    python scripts/evaluate_official.py --compare      # against the old strategy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.make_submission import ranked_hits  # noqa: E402

from src.core.submission import (  # noqa: E402
    MAX_ROWS,
    AllocationPlan,
    Candidate,
    allocate_hybrid_rows,
    final_score,
    r_score_kis,
    score_bucket,
)

WINDOWS = (10, 20, 50, 100, 200)


def load(data_dir: Path):
    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    gt = json.loads((data_dir / "ground_truth.json").read_text(encoding="utf-8"))
    return meta, gt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--draws", type=int, default=24, help="window placements per query")
    ap.add_argument("--n-flat", type=int, default=30)
    ap.add_argument("--depth-cost", type=float, default=0.5)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--compare", action="store_true", help="also score the keyframes-only strategy")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data_dir = Path(args.data)
    meta, gt = load(data_dir)
    if args.limit:
        gt = gt[: args.limit]

    print("loading index + model ...", flush=True)
    from src.core.kis_engine import KISEngine

    engine = KISEngine(data_dir).load()
    print(f"  {len(meta):,} keyframes, {len(engine.last_frame)} videos\n")

    by_video = defaultdict(list)
    for i, m in enumerate(meta):
        by_video[m["video_id"]].append(i)
    frame = engine.frame_idx

    # realistic window placement (see module docstring)
    rng = np.random.default_rng(0)
    spans = {w: [] for w in WINDOWS}
    for g in gt:
        fr = np.sort(frame[by_video[g["video_id"]]])
        d = np.diff(fr)
        gap = int(np.median(d)) if len(d) else 50
        half = max(1, gap // 2)
        offs = rng.integers(-half, half + 1, size=args.draws)
        for w in WINDOWS:
            spans[w].append(
                [(max(0, g["frame_idx"] + int(o) - w // 2), max(0, g["frame_idx"] + int(o) - w // 2) + w)
                 for o in offs]
            )

    plan = AllocationPlan(breadth_cost=1.0, depth_cost=args.depth_cost, step=args.step)

    strategies = {
        f"SHIPPED  hybrid flat-{args.n_flat} + ladder": (
            lambda c: allocate_hybrid_rows(c, n_flat=args.n_flat, plan=plan)
        )
    }
    if args.compare:
        strategies["OLD      keyframes only, cap 2/video"] = _capped_only

    totals = {name: {w: 0.0 for w in WINDOWS} for name in strategies}
    video_r1 = 0
    lat = []
    per_query_rank = {name: [] for name in strategies}

    for qi, g in enumerate(gt):
        t0 = time.time()
        hits = ranked_hits(engine, g["kis_query_vi"], g.get("kis_query_en"))
        lat.append((time.time() - t0) * 1000)
        if hits and hits[0].video_id == g["video_id"]:
            video_r1 += 1
        cands = [Candidate(h.video_id, h.frame_idx, h.score, h.video_last_frame) for h in hits]

        for name, alloc in strategies.items():
            rows = alloc(cands)[:MAX_ROWS]
            best_rank = None
            for w in WINDOWS:
                for sp in spans[w][qi]:
                    per = [r_score_kis(v, f, g["video_id"], sp) for v, f in rows]
                    totals[name][w] += final_score(per)
                    if w == 50 and best_rank is None:
                        best_rank = next((i + 1 for i, s in enumerate(per) if s > 0), None)
            per_query_rank[name].append(best_rank)

    n = len(gt) * args.draws
    print("=" * 92)
    print(f"OFFICIAL Final Score  ({len(gt)} queries x {args.draws} window placements)")
    print("=" * 92)
    print(f"  {'strategy':40s} " + "  ".join(f"W={w:<4d}" for w in WINDOWS) + "     mean")
    print("  " + "-" * 88)
    for name in strategies:
        v = [totals[name][w] / n for w in WINDOWS]
        print(f"  {name:40s} " + "  ".join(f"{x:6.3f}" for x in v) + f"  {np.mean(v):7.3f}")

    print(f"\n  video R@1 (for reference only): {video_r1}/{len(gt)} = {video_r1 / len(gt):.1%}")
    print(f"  retrieval latency: median {np.median(lat):.0f} ms, p95 {np.percentile(lat, 95):.0f} ms")

    print("\n  where the first correct row lands (at W=50), and what each bucket is worth:")
    for name in strategies:
        ranks = per_query_rank[name]
        buckets = defaultdict(int)
        for r in ranks:
            if r is None:
                buckets["miss"] += 1
            elif r == 1:
                buckets["1"] += 1
            elif r <= 5:
                buckets["2-5"] += 1
            elif r <= 20:
                buckets["6-20"] += 1
            elif r <= 50:
                buckets["21-50"] += 1
            else:
                buckets["51-100"] += 1
        line = "  ".join(
            f"{k}:{buckets[k]}" for k in ("1", "2-5", "6-20", "21-50", "51-100", "miss") if buckets[k]
        )
        est = sum(score_bucket(r) for r in ranks) / len(ranks)
        print(f"    {name:40s} {line}   -> {est:.3f}/query")

    print("\n  A hit at rank 1 is worth 1.0; 2-5 is 0.8; 6-20 is 0.6; 21-50 is 0.4; 51-100 is 0.2.")
    print("  Moving a hit from rank 15 to 10 is worth nothing; from 6 to 5 is worth 0.2.")
    return 0


def _capped_only(cands, per_video: int = 2):
    """The previous strategy: <=2 keyframes per video, no frame ladder."""
    seen = defaultdict(int)
    out = []
    for c in cands:
        if seen[c.video_id] >= per_video:
            continue
        seen[c.video_id] += 1
        out.append((c.video_id, c.frame_idx))
        if len(out) >= MAX_ROWS:
            break
    return out


if __name__ == "__main__":
    raise SystemExit(main())
