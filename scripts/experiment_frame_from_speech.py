"""Can the spoken word pick the FRAME, not just the video?

The earlier experiment folded transcripts in at video level and measured
negative. That is a different question from this one. A transcript cue carries a
TIMESTAMP, so where the words match, they say *when* — and "when" is the whole
difficulty here: keyframes sit 55 frames apart and the answer window is under 10.

So this scores each candidate keyframe by how well the speech AROUND ITS OWN
TIMESTAMP matches the query, and adds that to the visual score. A video-level
bonus cannot reorder frames within a video; this can, which is the point.

Measured the same way as everything else: official formula, non-snapped answer
key, many re-draws, honest negative if that is the result.

    python scripts/experiment_frame_from_speech.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
from src.core.transcripts import TranscriptIndex, tokenise  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--transcripts", default=str(ROOT.parent / "transcripts_full"))
    ap.add_argument("--windows", default="6,10,20,40")
    ap.add_argument("--draws", type=int, default=32)
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--halflife", type=float, default=8.0,
                    help="seconds; how fast the spoken bonus decays away from a matching cue")
    args = ap.parse_args()

    from src.core.kis_engine import KISEngine

    print("loading index ...", flush=True)
    eng = KISEngine(args.data).load()
    tx = TranscriptIndex().load_dir(
        *[Path(d) for d in args.transcripts.split(",") if d.strip()], Path(args.data) / "captions"
    )
    print(f"{tx.n_videos} video co loi thoai", flush=True)

    gt = json.loads((Path(args.data) / "ground_truth.json").read_text(encoding="utf-8"))
    gt = [g for g in gt if g.get("video_id") in eng.last_frame]

    print("retrieving ...", flush=True)
    cached = []
    for g in gt:
        hits = ranked_hits(eng, g["kis_query_vi"], g.get("kis_query_en"))[: args.top_n]
        cached.append((g, hits))

    # A per-second speech-match profile for every candidate video, so a frame can
    # be scored by what was being said at its own timestamp.
    print("building per-second speech profiles ...", flush=True)
    profiles = []
    for g, hits in cached:
        terms = Counter(tokenise(g["kis_query_vi"]))
        idfs = {t: tx.idf_of(t) for t in terms}
        prof = {}
        for vid in {h.video_id for h in hits}:
            segs = tx.segments.get(vid)
            if not segs:
                continue
            pts = []
            for start, text in segs:
                tf = Counter(tokenise(text))
                s = sum(idfs.get(t, 0.0) * min(tf.get(t, 0), 2) * q for t, q in terms.items())
                if s > 0:
                    pts.append((start, s))
            if pts:
                top = max(s for _t, s in pts)
                prof[vid] = ([t for t, _s in pts], [s / top for _t, s in pts])
        profiles.append(prof)

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

    draws = [draw(6000 + s) for s in range(args.draws)]
    windows = [int(w) for w in args.windows.split(",")]
    plan = AllocationPlan(breadth_cost=1.0, depth_cost=DEFAULT_DEPTH_COST, step=10)

    def rescore(w: float, halflife: float):
        per_query = []
        for qi, (g, hits) in enumerate(cached):
            prof = profiles[qi]
            if not prof:
                per_query.append(hits)
                continue
            scored = []
            for i, h in enumerate(hits):
                bonus = 0.0
                p = prof.get(h.video_id)
                if p:
                    ts, ss = p
                    # nearest matching cue, decayed by distance in time
                    j = int(np.argmin(np.abs(np.asarray(ts) - h.pts_time)))
                    dt = abs(ts[j] - h.pts_time)
                    bonus = w * ss[j] * float(np.exp(-dt / halflife))
                scored.append((h.score + bonus, i, h))
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
    covered = sum(1 for p in profiles if p)
    print(f"\n{covered}/{len(cached)} cau co it nhat mot video ung vien co loi thoai khop")
    head = f"{'variant':22s}" + "".join(f"{'W=' + str(w):>9}" for w in windows) + "     mean  vR@1   delta"
    print("\n" + head)
    print("-" * len(head))
    print(f"{'khong dung loi thoai':22s}" + "".join(f"{v:9.3f}" for v in base_w)
          + f"{base:9.3f}  {base_r1:3d}       -")

    best = (base, "baseline")
    for hl in (4.0, 8.0, 20.0):
        for w in (0.005, 0.01, 0.02, 0.04):
            per_w, r1 = evaluate(rescore(w, hl))
            mean = sum(per_w) / len(per_w)
            if mean > best[0]:
                best = (mean, f"w={w} halflife={hl}s")
            print(f"{'w=' + str(w) + ' hl=' + str(hl) + 's':22s}"
                  + "".join(f"{v:9.3f}" for v in per_w)
                  + f"{mean:9.3f}  {r1:3d}  {100 * (mean / base - 1):+6.1f}%", flush=True)

    print()
    print(f"Tot nhat: {best[1]} -> {best[0]:.3f}  ({100 * (best[0] / base - 1):+.1f}%)"
          if best[1] != "baseline" else "Khong cach nao thang baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
