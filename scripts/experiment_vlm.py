"""Does a vision model looking at the frames raise the contest score?

Everything else this project measured — video metadata, per-frame object bonuses,
transcripts at video level, transcripts at frame level — came out negative or
inside the noise, and each one had looked obviously useful beforehand. So this
gets the same treatment before it goes anywhere near a submission: the official
formula, a non-snapped answer key, many re-draws, and the honest negative
reported if that is what comes out.

What makes this one different in principle is that the VLM is not a second
opinion from the same kind of model. SigLIP-2 compresses a frame to one vector,
which is why it cannot tell a yellow lion from a red one. A VLM is asked the
question in words and answers about the specific details the query names.

    python scripts/experiment_vlm.py --limit 30
    python scripts/experiment_vlm.py --model gemini-3.7-flash
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from src.core.vlm import DEFAULT_MODEL, VLMJudge  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="first N ground-truth queries")
    ap.add_argument("--judge", type=int, default=24, help="candidates the VLM looks at per query")
    ap.add_argument("--windows", default="6,10,20,40")
    ap.add_argument("--draws", type=int, default=32)
    args = ap.parse_args()

    judge = VLMJudge(args.data, model=args.model)
    if not judge.ready:
        print("Khong co GEMINI_API_KEY (dat trong .env hoac bien moi truong).")
        return 2

    from src.core.kis_engine import KISEngine

    print(f"model: {args.model}\nloading index ...", flush=True)
    eng = KISEngine(args.data).load()
    meta = {(m["video_id"], m["frame_idx"]): m for m in eng.metadata}
    gt = json.loads((Path(args.data) / "ground_truth.json").read_text(encoding="utf-8"))
    gt = [g for g in gt if g.get("video_id") in eng.last_frame]
    if args.limit:
        gt = gt[: args.limit]
    print(f"{len(gt)} cau ground truth\n", flush=True)

    print("retrieving ...", flush=True)
    cached = []
    for g in gt:
        hits = ranked_hits(eng, g["kis_query_vi"], g.get("kis_query_en"))[:200]
        cached.append((g, hits))

    print(f"judging top-{args.judge} per query ...", flush=True)
    t0 = time.time()
    vlm_scores = []
    for qi, (g, hits) in enumerate(cached, 1):
        top = hits[: args.judge]
        cands = [
            (h.video_id, h.frame_idx, meta[(h.video_id, h.frame_idx)]["frame_filename"])
            for h in top
            if (h.video_id, h.frame_idx) in meta
        ]
        vlm_scores.append(judge.score(g["kis_query_vi"], cands))
        if qi % 5 == 0 or qi == len(cached):
            el = time.time() - t0
            print(f"  {qi}/{len(cached)}  ({el/qi:.1f}s/cau, con ~{(len(cached)-qi)*el/qi/60:.0f} phut)  "
                  f"{judge.cost_note()}", flush=True)
    if judge.errors:
        print(f"  ! {len(judge.errors)} loi goi API, vi du: {judge.errors[0]}")

    kf: dict = {}
    for m in eng.metadata:
        kf.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
    for v in kf:
        kf[v] = np.array(sorted(kf[v]))

    def draw(seed):
        rng = np.random.default_rng(seed)
        out = []
        for g, _h in cached:
            a = kf[g["video_id"]]
            i = int(np.argmin(np.abs(a - int(g["frame_idx"]))))
            lo = (a[i] + a[i - 1]) // 2 if i > 0 else a[i] - 30
            hi = (a[i] + a[i + 1]) // 2 if i + 1 < len(a) else a[i] + 30
            out.append(int(rng.integers(lo, max(lo + 1, hi))))
        return out

    draws = [draw(7000 + s) for s in range(args.draws)]
    windows = [int(w) for w in args.windows.split(",")]
    plan = AllocationPlan(breadth_cost=1.0, depth_cost=DEFAULT_DEPTH_COST, step=10)

    def rescore(mode: str, w: float):
        per_query = []
        for qi, (g, hits) in enumerate(cached):
            vs = vlm_scores[qi]
            if not vs:
                per_query.append(hits)
                continue
            if mode == "video":
                # best VLM score per video, applied to all its frames — the shape
                # that worked for object detections and preserves the embedding's
                # within-video ordering, which is the one that knows about timing
                best: dict = {}
                for (v, f), (s, _why) in vs.items():
                    best[v] = max(best.get(v, 0.0), s)
                bonus = lambda h: w * best.get(h.video_id, 0.0)  # noqa: E731
            else:
                bonus = lambda h: w * vs.get((h.video_id, h.frame_idx), (0.0, ""))[0]  # noqa: E731
            scored = [(h.score + bonus(h), i, h) for i, h in enumerate(hits)]
            scored.sort(key=lambda t: (-t[0], t[1]))
            per_query.append([t[2] for t in scored])
        return per_query

    def evaluate(per_query):
        per_w = []
        for half in windows:
            tot = 0.0
            for qi, ((g, _), hits) in enumerate(zip(cached, per_query)):
                cands = [Candidate(h.video_id, h.frame_idx, h.score, h.video_last_frame) for h in hits]
                rows = allocate_hybrid_rows(cands, n_flat=DEFAULT_N_FLAT, plan=plan)[:MAX_ROWS]
                gv = g["video_id"]
                for truth in draws:
                    span = (truth[qi] - half, truth[qi] + half)
                    tot += final_score([r_score_kis(v, f, gv, span) for v, f in rows])
            per_w.append(tot / (len(cached) * len(draws)))
        r1 = sum(1 for (g, _), h in zip(cached, per_query) if h and h[0].video_id == g["video_id"])
        return per_w, r1

    base_w, base_r1 = evaluate([h for _g, h in cached])
    base = sum(base_w) / len(base_w)
    head = f"{'variant':22s}" + "".join(f"{'W=' + str(w):>9}" for w in windows) + "     mean  vR@1   delta"
    print("\n" + head)
    print("-" * len(head))
    print(f"{'khong dung VLM':22s}" + "".join(f"{v:9.3f}" for v in base_w)
          + f"{base:9.3f}  {base_r1:3d}       -")

    best = (base, "baseline")
    for mode in ("frame", "video"):
        for w in (0.01, 0.02, 0.05, 0.10, 0.20):
            per_w, r1 = evaluate(rescore(mode, w))
            mean = sum(per_w) / len(per_w)
            if mean > best[0]:
                best = (mean, f"{mode} w={w}")
            print(f"{mode + ' w=' + str(w):22s}" + "".join(f"{v:9.3f}" for v in per_w)
                  + f"{mean:9.3f}  {r1:3d}  {100 * (mean / base - 1):+6.1f}%", flush=True)

    print(f"\n{judge.cost_note()}")
    if best[1] == "baseline":
        print("Khong cach nao thang baseline — khong dua VLM vao duong cham diem.")
    else:
        print(f"Tot nhat: {best[1]} -> {best[0]:.3f}  ({100 * (best[0] / base - 1):+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
