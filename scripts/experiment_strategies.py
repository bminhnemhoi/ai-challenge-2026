"""Measure candidate submission strategies against the OFFICIAL score.

The existing benchmark (scripts/evaluate_official_pipeline.py) reports
video-level rank and never looks at the frame index, so it cannot tell us what
the organisers would actually award.  This script scores the real thing:

    R-Score(r) = I(video == GT_video and frame_id in [s, e])
    Final Score = mean of R@k for k in {1, 5, 20, 50, 100}

The answer-window width [s, e] is not published.  The rules' worked example
uses 11 frames and TRAKE events are "usually under 10 frames", so every result
is reported across a range of assumed widths and the narrow end is treated as
the one to design for.

Run:  python scripts/experiment_strategies.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.submission import (  # noqa: E402
    MAX_ROWS,
    AllocationPlan,
    Candidate,
    allocate_kis_rows,
    final_score,
    r_score_kis,
)

DATA = ROOT / "data"
QVECS = Path(
    r"C:\Users\Admin\AppData\Local\Temp\claude\d--AI2026TOP1SV"
    r"\fd167841-c759-4109-9879-e7661bba812c\scratchpad\qvecs.npz"
)
WINDOWS = [10, 20, 50, 100, 200]


# ---------------------------------------------------------------- data


def load():
    meta = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))
    gt = json.loads((DATA / "ground_truth.json").read_text(encoding="utf-8"))
    E = np.load(DATA / "embeddings_siglip2_384.npy", mmap_mode="r")
    q = np.load(QVECS)
    blank = set()
    p = DATA / "blank_frame_indices.json"
    if p.exists():
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            blank = set(raw if isinstance(raw, list) else raw.get("blank_frame_indices", []))
        except Exception:
            pass
    return meta, gt, E, q, blank


def build_views(meta):
    vid = np.array([m["video_id"] for m in meta], dtype=object)
    frame = np.array([m["frame_idx"] for m in meta], dtype=np.int64)
    n = np.array([m.get("n", 1) for m in meta], dtype=np.int64)
    by_video = defaultdict(list)
    for i, v in enumerate(vid):
        by_video[v].append(i)
    last_frame = {v: int(frame[idxs].max()) for v, idxs in by_video.items()}
    return vid, frame, n, by_video, last_frame


# ------------------------------------------------------- candidate builders


def cands_flat(sims, vid, frame, last_frame, valid, n_cands=120):
    """Plain top-N keyframes, no per-video cap. The simplest possible baseline."""
    order = np.argsort(-np.where(valid, sims, -np.inf))[:n_cands]
    return [
        Candidate(str(vid[i]), int(frame[i]), float(sims[i]), last_frame[str(vid[i])])
        for i in order
    ]


def cands_capped(sims, vid, frame, last_frame, valid, per_video=2, n_cands=120):
    """Top-N keyframes with a per-video cap — what the current pipeline does."""
    order = np.argsort(-np.where(valid, sims, -np.inf))
    seen = defaultdict(int)
    out = []
    for i in order:
        v = str(vid[i])
        if seen[v] >= per_video:
            continue
        seen[v] += 1
        out.append(Candidate(v, int(frame[i]), float(sims[i]), last_frame[v]))
        if len(out) >= n_cands:
            break
    return out


def cands_video_pooled(sims, vid, frame, by_video, last_frame, valid, n_cands=120):
    """Rank videos by convex pooling of their best frames, then take each one's best.

    This mirrors the shipped retriever's aggregation
    S(v) = 0.75*s1 + 0.20*s2 + 0.05*s3.
    """
    scored = []
    for v, idxs in by_video.items():
        s = np.array([sims[i] if valid[i] else -np.inf for i in idxs])
        k = np.argsort(-s)[:3]
        top = s[k]
        if not np.isfinite(top[0]):
            continue
        w = [0.75, 0.20, 0.05][: len(top)]
        agg = float(sum(a * b for a, b in zip(w, top) if np.isfinite(b)))
        scored.append((agg, v, int(idxs[int(k[0])])))
    scored.sort(key=lambda t: -t[0])
    return [
        Candidate(v, int(frame[i]), agg, last_frame[v]) for agg, v, i in scored[:n_cands]
    ]


# ------------------------------------------------------------------ scoring


def score_rows(rows, gt_video, span):
    per_row = [r_score_kis(v, f, gt_video, span) for v, f in rows]
    return final_score(per_row), per_row


def gt_spans(gt, local_gap, window, n_draws, rng):
    """Where the organisers' answer window plausibly sits, per query.

    data/ground_truth.json stores a single frame index that the team picked
    FROM THE KEYFRAME SET, so placing the window symmetrically on it would
    guarantee that submitting that keyframe scores — and would make any frame
    ladder look useless.  That is an artefact of how the file was built, not a
    property of the competition.

    The organisers annotate the semantic moment on the ORIGINAL video timeline,
    which has no reason to coincide with a sampled keyframe.  So the moment is
    modelled as falling uniformly within half a local keyframe gap of the
    labelled keyframe, and every strategy is averaged over ``n_draws`` such
    placements.  ``n_draws=1`` with offset 0 reproduces the optimistic
    (biased) view for comparison.
    """
    out = []
    for qi, g in enumerate(gt):
        half_gap = max(1, local_gap[qi] // 2)
        offs = rng.integers(-half_gap, half_gap + 1, size=n_draws)
        spans = []
        for o in offs:
            centre = int(g["frame_idx"]) + int(o)
            s = max(0, centre - window // 2)
            spans.append((s, s + window))
        out.append(spans)
    return out


def evaluate(name, rows_fn, gt, sims_all, spans_by_window, verbose=True):
    totals = {w: 0.0 for w in WINDOWS}
    anyhit = {w: 0.0 for w in WINDOWS}
    n_draws = len(next(iter(spans_by_window.values()))[0])
    for qi, _g in enumerate(gt):
        rows = rows_fn(qi, sims_all[qi])
        for w in WINDOWS:
            for span in spans_by_window[w][qi]:
                fs, per_row = score_rows(rows, gt[qi]["video_id"], span)
                totals[w] += fs
                anyhit[w] += 1.0 if any(s > 0 for s in per_row) else 0.0
    denom = len(gt) * n_draws
    if verbose:
        cells = "  ".join(f"{totals[w] / denom:6.3f}" for w in WINDOWS)
        hits = "  ".join(f"{anyhit[w] / denom:5.0%}" for w in WINDOWS)
        print(f"  {name:40s} {cells}   |  {hits}")
    return {w: totals[w] / denom for w in WINDOWS}


def main():
    print("loading corpus ...", flush=True)
    meta, gt, E, q, blank = load()
    vid, frame, n_kf, by_video, last_frame = build_views(meta)
    E = np.asarray(E, dtype=np.float32)
    valid = np.ones(len(meta), dtype=bool)
    if blank:
        idx = np.array(sorted(i for i in blank if 0 <= i < len(meta)), dtype=np.int64)
        valid[idx] = False
    valid &= n_kf > 2  # the shipped retriever drops the first two keyframes
    print(f"  {len(meta):,} keyframes, {len(by_video)} videos, {(~valid).sum():,} filtered\n")

    # the shipped 4-prompt ensemble, and the plain single-prompt baselines
    ens = 0.45 * q["en"] + 0.35 * q["vi"] + 0.10 * q["en_keyframe"] + 0.10 * q["en_photo"]
    ens /= np.linalg.norm(ens, axis=1, keepdims=True)
    query_sets = {"ensemble(4-prompt)": ens, "en only": q["en"], "vi only": q["vi"]}

    print("=" * 100)
    print("PART 1 — which query representation retrieves best (video-level R@1, for reference)")
    print("=" * 100)
    sims_cache = {}
    for qname, Q in query_sets.items():
        t0 = time.time()
        S = (Q.astype(np.float32) @ E.T).astype(np.float32)
        sims_cache[qname] = S
        r1 = 0
        for qi, g in enumerate(gt):
            s = np.where(valid, S[qi], -np.inf)
            best_v = str(vid[int(np.argmax(s))])
            r1 += best_v == g["video_id"]
        print(f"  {qname:24s} video R@1 = {r1}/{len(gt)} = {r1 / len(gt):5.1%}   ({time.time() - t0:.1f}s)")

    best_q = "ensemble(4-prompt)"
    S = sims_cache[best_q]
    print(f"\nusing {best_q} for the rest\n")

    # local keyframe spacing at each ground-truth frame, for the window model
    local_gap = []
    for g in gt:
        idxs = by_video[g["video_id"]]
        fr = np.sort(frame[idxs])
        d = np.diff(fr)
        local_gap.append(int(np.median(d)) if len(d) else 50)
    print(f"median local keyframe gap at the GT frames: {int(np.median(local_gap))} frames\n")

    rng = np.random.default_rng(0)
    n_draws = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    spans = {w: gt_spans(gt, local_gap, w, n_draws, rng) for w in WINDOWS}

    print("=" * 104)
    print("PART 2 — OFFICIAL Final Score by submission strategy")
    print(f"        the answer window is placed realistically (see gt_spans), {n_draws} draws/query")
    print("        columns = assumed answer-window width in frames")
    print("=" * 104)
    print(f"  {'strategy':40s} " + "  ".join(f"W={w:<4d}" for w in WINDOWS) + "   |  any-hit rate")
    print("  " + "-" * 100)

    def mk(fn, plan=None):
        def rows_fn(qi, sims):
            c = fn(sims)
            return allocate_kis_rows(c, plan) if plan is not None else [
                (x.video_id, x.frame_idx) for x in c[:MAX_ROWS]
            ]

        return rows_fn

    # --- what the pipeline does today: keyframe indices only, per-video cap ---
    evaluate(
        "A. capped(2), keyframes only  [CURRENT]",
        mk(lambda s: cands_capped(s, vid, frame, last_frame, valid)),
        gt, S, spans,
    )
    evaluate(
        "B. video-pooled, keyframes only",
        mk(lambda s: cands_video_pooled(s, vid, frame, by_video, last_frame, valid)),
        gt, S, spans,
    )
    evaluate(
        "C. flat top-100, keyframes only",
        mk(lambda s: cands_flat(s, vid, frame, last_frame, valid)),
        gt, S, spans,
    )

    print("  " + "-" * 96)
    for depth_cost in (0.75, 0.5, 0.35, 0.2):
        plan = AllocationPlan(breadth_cost=1.0, depth_cost=depth_cost)
        evaluate(
            f"D. capped(2) + ladder (depth_cost={depth_cost})",
            mk(lambda s: cands_capped(s, vid, frame, last_frame, valid), plan),
            gt, S, spans,
        )
    print("  " + "-" * 96)
    for depth_cost in (0.5, 0.35, 0.2):
        plan = AllocationPlan(breadth_cost=1.0, depth_cost=depth_cost)
        evaluate(
            f"E. flat + ladder (depth_cost={depth_cost})",
            mk(lambda s: cands_flat(s, vid, frame, last_frame, valid), plan),
            gt, S, spans,
        )
    print("  " + "-" * 96)
    for depth_cost in (0.5, 0.35, 0.2):
        plan = AllocationPlan(breadth_cost=1.0, depth_cost=depth_cost)
        evaluate(
            f"F. video-pooled + ladder (depth_cost={depth_cost})",
            mk(
                lambda s: cands_video_pooled(s, vid, frame, by_video, last_frame, valid),
                plan,
            ),
            gt, S, spans,
        )

    print("  " + "-" * 100)
    from src.core.submission import allocate_hybrid_rows  # noqa: E402

    def mk_hybrid(fn, n_flat, dc):
        p = AllocationPlan(breadth_cost=1.0, depth_cost=dc)

        def rows_fn(qi, sims):
            return allocate_hybrid_rows(fn(sims), n_flat=n_flat, plan=p)

        return rows_fn

    results = {}
    for n_flat in (5, 10, 20, 30, 50):
        for dc in (0.5, 0.25):
            key = f"G. HYBRID flat-{n_flat} then ladder (dc={dc})"
            results[key] = evaluate(
                key,
                mk_hybrid(lambda s: cands_flat(s, vid, frame, last_frame, valid), n_flat, dc),
                gt, S, spans,
            )

    print("\n" + "=" * 104)
    print("PART 3 — which strategy to actually ship")
    print("=" * 104)
    print("The window width is unpublished, so compare strategies by their WORST case over")
    print("the plausible range, not by their best. A strategy that wins at W=200 and scores")
    print("0.10 at W=10 is a gamble; one that never drops below ~0.28 is not.\n")
    pure = {
        "A. capped(2) keyframes [CURRENT]": evaluate(
            "", mk(lambda s: cands_capped(s, vid, frame, last_frame, valid)), gt, S, spans, False
        ),
        "C. flat keyframes": evaluate(
            "", mk(lambda s: cands_flat(s, vid, frame, last_frame, valid)), gt, S, spans, False
        ),
        "E. flat + ladder (dc=0.5)": evaluate(
            "",
            mk(lambda s: cands_flat(s, vid, frame, last_frame, valid),
               AllocationPlan(depth_cost=0.5)),
            gt, S, spans, False,
        ),
    }
    pure.update(results)
    print(f"  {'strategy':44s} {'worst':>7} {'mean':>7} {'W=10':>7} {'W=200':>7}")
    print("  " + "-" * 78)
    for k, v in sorted(pure.items(), key=lambda kv: -min(kv[1].values())):
        vals = list(v.values())
        print(
            f"  {k:44s} {min(vals):7.3f} {sum(vals) / len(vals):7.3f} "
            f"{v[10]:7.3f} {v[200]:7.3f}"
        )

    print("\nReading: 'any-hit rate' is the fraction of queries with at least one correct")
    print("row anywhere in the 100 — the ceiling the Final Score is working against.")


if __name__ == "__main__":
    main()
