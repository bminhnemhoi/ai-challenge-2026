"""Does chunking actually rescue a query whose detail sits past the token limit?

The 60-sample ground truth has short queries, so it cannot measure this on its
own.  The mechanism is therefore tested directly: each ground-truth query is
padded with generic scene-setting text so the distinguishing description is
pushed past SigLIP-2's 64-token window, exactly as the real contest queries do
(13 of the 24 round-1 queries exceed it).

    truncating  = what the encoder sees today: the first 64 tokens only
    chunked     = every clause encoded, similarities combined

    python scripts/experiment_long_query.py
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
    MAX_ROWS,
    AllocationPlan,
    Candidate,
    allocate_hybrid_rows,
    final_score,
    r_score_kis,
)

WINDOWS = (10, 50, 200)
N_DRAWS = 12

# Generic scene-setting of the kind a describer writes before the specifics.
# It carries no information that identifies any particular video.
PAD = (
    "This is a scene captured in a Vietnamese television programme. "
    "The footage was recorded during the daytime with natural lighting and a "
    "steady camera. The overall atmosphere of the report is calm and ordinary, "
    "with people going about their usual activities in the background. "
)


def main() -> int:
    data = ROOT / "data"
    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    gt = json.loads((data / "ground_truth.json").read_text(encoding="utf-8"))

    from src.core.kis_engine import KISEngine

    print("loading index ...", flush=True)
    eng = KISEngine(data).load()
    tok = eng.processor.tokenizer

    by_video = defaultdict(list)
    for i, m in enumerate(meta):
        by_video[m["video_id"]].append(i)

    rng = np.random.default_rng(0)
    spans = {w: [] for w in WINDOWS}
    for g in gt:
        fr = np.sort(eng.frame_idx[by_video[g["video_id"]]])
        d = np.diff(fr)
        half = max(1, (int(np.median(d)) if len(d) else 50) // 2)
        offs = rng.integers(-half, half + 1, size=N_DRAWS)
        for w in WINDOWS:
            spans[w].append(
                [(max(0, g["frame_idx"] + int(o) - w // 2), max(0, g["frame_idx"] + int(o) - w // 2) + w)
                 for o in offs]
            )

    plan = AllocationPlan(depth_cost=0.5)

    def run(label, sims_fn):
        tot = {w: 0.0 for w in WINDOWS}
        r1 = 0
        for qi, g in enumerate(gt):
            s = np.where(eng.valid, sims_fn(qi, g), -np.inf)
            top = np.argsort(-s)[:200]
            if str(eng.video_id[int(top[0])]) == g["video_id"]:
                r1 += 1
            cands = [
                Candidate(str(eng.video_id[i]), int(eng.frame_idx[i]), float(s[i]),
                          eng.last_frame[str(eng.video_id[i])])
                for i in top
            ]
            rows = allocate_hybrid_rows(cands, n_flat=30, plan=plan)[:MAX_ROWS]
            for w in WINDOWS:
                for sp in spans[w][qi]:
                    tot[w] += final_score([r_score_kis(v, f, g["video_id"], sp) for v, f in rows])
        n = len(gt) * N_DRAWS
        vals = [tot[w] / n for w in WINDOWS]
        print(
            f"  {label:34s} R@1={r1 / len(gt):5.1%}   "
            + "  ".join(f"{v:6.3f}" for v in vals)
            + f"   mean={np.mean(vals):6.3f}"
        )
        return r1 / len(gt), float(np.mean(vals))

    # padded English queries: scene-setting first, identifying detail last
    padded = [PAD + g["kis_query_en"] for g in gt]
    n_over = sum(1 for p in padded if len(tok(p)["input_ids"]) > 64)
    print(f"\n{n_over}/{len(gt)} padded queries exceed the 64-token window\n")

    print("=" * 96)
    print("SHORT queries (as the ground truth ships) — chunking should change nothing")
    print("=" * 96)
    print(f"  {'setting':34s} {'R@1':>9}   " + "  ".join(f"W={w:<4d}" for w in WINDOWS) + "      mean")
    run("plain", lambda qi, g: eng.similarities(eng.query_vector(g["kis_query_vi"], g["kis_query_en"])))
    run("chunked", lambda qi, g: eng.query_similarities(g["kis_query_vi"], g["kis_query_en"]))

    print("\n" + "=" * 96)
    print("PADDED queries — the identifying detail now sits past the token window")
    print("=" * 96)
    print(f"  {'setting':34s} {'R@1':>9}   " + "  ".join(f"W={w:<4d}" for w in WINDOWS) + "      mean")
    base = run(
        "truncating (current behaviour)",
        lambda qi, g: eng.similarities(eng.query_vector(padded[qi], padded[qi])),
    )
    chunk_mean = run(
        "chunked, combine=mean",
        lambda qi, g: eng.query_similarities(padded[qi], padded[qi], combine="mean"),
    )
    chunk_max = run(
        "chunked, combine=max",
        lambda qi, g: eng.query_similarities(padded[qi], padded[qi], combine="max"),
    )

    print(
        f"\n  chunking recovers {chunk_mean[0] - base[0]:+.1%} of video R@1 and "
        f"{chunk_mean[1] - base[1]:+.3f} of official score on long queries."
    )
    if chunk_max[1] > chunk_mean[1]:
        print("  NOTE: combine=max scored higher here — re-check before changing the default.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
