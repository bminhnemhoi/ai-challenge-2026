"""Is merging the automatic and the hand-written reading actually better?

`merged_hits` keeps BOTH the automatic translation's candidates and a human
English rewrite's candidates, taking the better score per frame.  The argument
for it is that R@k is a maximum over a prefix, so an extra candidate can only
help.  But the merge also *reorders* — a frame the rewrite scores highly gets
promoted past one the automatic reading found — so it can push a correct frame
down, and "extra rows are free" does not cover that.

This settles it on all 60 ground-truth queries, which carry a human English
rendering alongside the Vietnamese, scored with the official formula against a
non-snapped answer key (see experiment_allocation.py for why the recorded
ground-truth frames cannot be used as-is).

    python scripts/experiment_merge.py
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

from scripts.make_submission import (  # noqa: E402
    DEFAULT_DEPTH_COST,
    DEFAULT_N_FLAT,
    ranked_hits,
)


def old_merged_hits(engine, query_text, query_en, top_n=400):
    """The candidate-list merge that used to ship, kept so the comparison stands.

    Calling the current ranked_hits here instead would silently compare the new
    behaviour against itself once the default changed, and the table would look
    like the merge had always been fine.
    """
    hits = engine.search(query_text, top_n=top_n)
    if not query_en:
        return hits
    extra = engine.search(query_en, query_en=query_en, top_n=top_n)
    best = {}
    for h in hits + extra:
        key = (h.video_id, h.frame_idx)
        if key not in best or h.score > best[key].score:
            best[key] = h
    return sorted(best.values(), key=lambda h: -h.score)
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
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--draws", type=int, default=24)
    args = ap.parse_args()

    from src.core.kis_engine import KISEngine

    print("loading index ...", flush=True)
    eng = KISEngine(args.data).load()
    gt = json.loads((Path(args.data) / "ground_truth.json").read_text(encoding="utf-8"))
    gt = [g for g in gt if g.get("video_id") in eng.last_frame and g.get("kis_query_en")]
    print(f"{len(gt)} ground-truth queries with both readings\n", flush=True)

    kf: dict = {}
    for m in eng.metadata:
        kf.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
    for v in kf:
        kf[v] = np.array(sorted(kf[v]))

    VARIANTS = {
        "auto only  (search(vi))": lambda g: eng.search(g["kis_query_vi"], top_n=400),
        "human EN only": lambda g: eng.search(
            g["kis_query_en"], query_en=g["kis_query_en"], top_n=400
        ),
        "merged lists (was shipping)": lambda g: old_merged_hits(
            eng, g["kis_query_vi"], g["kis_query_en"], top_n=400
        ),
        "ranked_hits (SHIPPING NOW)": lambda g: ranked_hits(
            eng, g["kis_query_vi"], g["kis_query_en"], top_n=400
        ),
    }

    print("retrieving each variant once ...", flush=True)
    rows_by_variant = {}
    plan = AllocationPlan(breadth_cost=1.0, depth_cost=DEFAULT_DEPTH_COST, step=10)
    for name, fn in VARIANTS.items():
        per_query = []
        for g in gt:
            hits = fn(g)
            cands = [Candidate(h.video_id, h.frame_idx, h.score, h.video_last_frame) for h in hits]
            per_query.append(allocate_hybrid_rows(cands, n_flat=DEFAULT_N_FLAT, plan=plan)[:MAX_ROWS])
        rows_by_variant[name] = per_query
        print(f"  {name}", flush=True)

    # the answer key: the marked instant, re-drawn inside its own keyframe gap
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

    draws = [draw(2000 + s) for s in range(args.draws)]
    windows = [int(w) for w in args.windows.split(",")]

    print(f"\n{'variant':30s}" + "".join(f"{'W=' + str(w):>9}" for w in windows) + "     mean   videoR@1")
    print("-" * (30 + 9 * len(windows) + 20))
    base = None
    for name, per_query in rows_by_variant.items():
        per_w = []
        for half in windows:
            tot = 0.0
            for qi, (g, rows) in enumerate(zip(gt, per_query)):
                gv = g["video_id"]
                for truth in draws:
                    span = (truth[qi] - half, truth[qi] + half)
                    tot += final_score([r_score_kis(v, f, gv, span) for v, f in rows])
            per_w.append(tot / (len(gt) * len(draws)))
        mean = sum(per_w) / len(per_w)
        r1 = sum(1 for g, rows in zip(gt, per_query) if rows and rows[0][0] == g["video_id"])
        if base is None:
            base = mean
        delta = f"{100 * (mean / base - 1):+.1f}%" if base else ""
        print(
            f"{name:30s}" + "".join(f"{v:9.3f}" for v in per_w)
            + f"{mean:9.3f}  {r1:3d}/{len(gt)}  {delta}"
        )

    print(
        "\nThe merge is only worth keeping if it beats BOTH single readings.\n"
        "If it does not, make_submission should stop merging."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
