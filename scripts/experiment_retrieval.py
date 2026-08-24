"""Can the retrieval itself be improved, without a new model?

Strategy work (see experiment_strategies.py) decides how to spend 100 rows on
whatever the retriever returns.  This script asks the other half of the
question: can we make the retriever rank the right frame higher in the first
place, using only the SigLIP-2 index that already exists?

Three ideas are tested, all pure NumPy and all sub-second per query:
  1. prompt-ensemble weights  — is 0.45/0.35/0.10/0.10 actually the best mix?
  2. temporal smoothing       — a frame whose neighbours also match is more
                                likely to be the real moment than an isolated
                                spike, which is usually a thumbnail or a caption
  3. per-video score standardisation — some videos are globally "closer" to all
                                text; subtracting a video's own baseline stops
                                those videos from crowding out every query

Run:  python scripts/experiment_retrieval.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.submission import (  # noqa: E402
    AllocationPlan,
    Candidate,
    allocate_hybrid_rows,
    final_score,
    r_score_kis,
)

DATA = ROOT / "data"
QVECS = Path(
    r"C:\Users\Admin\AppData\Local\Temp\claude\d--AI2026TOP1SV"
    r"\fd167841-c759-4109-9879-e7661bba812c\scratchpad\qvecs.npz"
)
WINDOWS = [10, 20, 50, 100, 200]
N_DRAWS = 24


def main():
    meta = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))
    gt = json.loads((DATA / "ground_truth.json").read_text(encoding="utf-8"))
    E = np.asarray(np.load(DATA / "embeddings_siglip2_384.npy", mmap_mode="r"), dtype=np.float32)
    q = np.load(QVECS)

    vid = np.array([m["video_id"] for m in meta], dtype=object)
    frame = np.array([m["frame_idx"] for m in meta], dtype=np.int64)
    nkf = np.array([m.get("n", 1) for m in meta], dtype=np.int64)
    by_video = defaultdict(list)
    for i, v in enumerate(vid):
        by_video[v].append(i)
    order_in_video = {v: np.array(sorted(ix, key=lambda i: frame[i])) for v, ix in by_video.items()}
    last_frame = {v: int(frame[ix].max()) for v, ix in by_video.items()}
    valid = nkf > 2

    local_gap = []
    for g in gt:
        fr = np.sort(frame[by_video[g["video_id"]]])
        d = np.diff(fr)
        local_gap.append(int(np.median(d)) if len(d) else 50)

    rng = np.random.default_rng(0)
    spans = {}
    for w in WINDOWS:
        spans[w] = []
        for qi, g in enumerate(gt):
            hg = max(1, local_gap[qi] // 2)
            offs = rng.integers(-hg, hg + 1, size=N_DRAWS)
            spans[w].append(
                [(max(0, g["frame_idx"] + int(o) - w // 2), max(0, g["frame_idx"] + int(o) - w // 2) + w) for o in offs]
            )

    plan = AllocationPlan(breadth_cost=1.0, depth_cost=0.5)

    def score_matrix(S, label, n_flat=30):
        tot = {w: 0.0 for w in WINDOWS}
        r1 = 0
        for qi, g in enumerate(gt):
            s = np.where(valid, S[qi], -np.inf)
            top = np.argsort(-s)[:150]
            if str(vid[int(top[0])]) == g["video_id"]:
                r1 += 1
            cands = [
                Candidate(str(vid[i]), int(frame[i]), float(s[i]), last_frame[str(vid[i])])
                for i in top
            ]
            rows = allocate_hybrid_rows(cands, n_flat=n_flat, plan=plan)
            for w in WINDOWS:
                for sp in spans[w][qi]:
                    tot[w] += final_score([r_score_kis(v, f, g["video_id"], sp) for v, f in rows])
        denom = len(gt) * N_DRAWS
        vals = {w: tot[w] / denom for w in WINDOWS}
        cells = "  ".join(f"{vals[w]:6.3f}" for w in WINDOWS)
        print(f"  {label:44s} R@1={r1 / len(gt):5.1%}  {cells}  mean={np.mean(list(vals.values())):6.3f}")
        return vals

    print("=" * 108)
    print("1. PROMPT ENSEMBLE WEIGHTS   (en, vi, keyframe-prefix, photo-prefix)")
    print("=" * 108)
    print(f"  {'weights':44s} {'R@1':>9}  " + "  ".join(f"W={w:<4d}" for w in WINDOWS) + "     mean")
    mixes = {
        "0.45/0.35/0.10/0.10  [CURRENT]": (0.45, 0.35, 0.10, 0.10),
        "1.00/0.00/0.00/0.00  (en only)": (1.0, 0.0, 0.0, 0.0),
        "0.00/1.00/0.00/0.00  (vi only)": (0.0, 1.0, 0.0, 0.0),
        "0.50/0.50/0.00/0.00": (0.5, 0.5, 0.0, 0.0),
        "0.60/0.40/0.00/0.00": (0.6, 0.4, 0.0, 0.0),
        "0.40/0.30/0.15/0.15": (0.4, 0.3, 0.15, 0.15),
        "0.35/0.25/0.20/0.20": (0.35, 0.25, 0.2, 0.2),
        "0.25/0.25/0.25/0.25": (0.25, 0.25, 0.25, 0.25),
    }
    best_mix, best_S, best_mean = None, None, -1.0
    for label, (a, b, c, d) in mixes.items():
        Q = a * q["en"] + b * q["vi"] + c * q["en_keyframe"] + d * q["en_photo"]
        Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
        S = (Q.astype(np.float32) @ E.T).astype(np.float32)
        vals = score_matrix(S, label)
        m = float(np.mean(list(vals.values())))
        if m > best_mean:
            best_mix, best_S, best_mean = label, S, m
    print(f"\n  best mix by mean: {best_mix}")

    print("\n" + "=" * 108)
    print("2. TEMPORAL SMOOTHING   S'[i] = (1-a)*S[i] + a*mean(S over +/-k neighbours in the video)")
    print("=" * 108)
    print(f"  {'setting':44s} {'R@1':>9}  " + "  ".join(f"W={w:<4d}" for w in WINDOWS) + "     mean")
    score_matrix(best_S, "no smoothing  [baseline]")
    for k in (1, 2):
        for a in (0.2, 0.35):
            Sm = best_S.copy()
            for v, ix in order_in_video.items():
                sub = best_S[:, ix]  # (Q, L) in temporal order
                L = sub.shape[1]
                acc = np.zeros_like(sub)
                cnt = np.zeros(L, dtype=np.float32)
                for off in range(-k, k + 1):
                    # neighbour at position p+off contributes to position p
                    src_lo, src_hi = max(0, off), min(L, L + off)
                    dst_lo, dst_hi = max(0, -off), min(L, L - off)
                    acc[:, dst_lo:dst_hi] += sub[:, src_lo:src_hi]
                    cnt[dst_lo:dst_hi] += 1
                cnt[cnt == 0] = 1
                Sm[:, ix] = (1 - a) * sub + a * (acc / cnt)
            score_matrix(Sm, f"k={k}, alpha={a}")

    print("\n" + "=" * 108)
    print("3. PER-VIDEO STANDARDISATION   S'[i] = S[i] - beta * mean(S over that video)")
    print("=" * 108)
    print(f"  {'setting':44s} {'R@1':>9}  " + "  ".join(f"W={w:<4d}" for w in WINDOWS) + "     mean")
    score_matrix(best_S, "beta=0  [baseline]")
    for beta in (0.25, 0.5, 1.0):
        Sm = best_S.copy()
        for v, ix in by_video.items():
            mu = best_S[:, ix].mean(axis=1, keepdims=True)
            Sm[:, ix] = best_S[:, ix] - beta * mu
        score_matrix(Sm, f"beta={beta}")


if __name__ == "__main__":
    main()
