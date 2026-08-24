"""Let a vision model re-rank the candidate videos — and measure whether it helps.

SigLIP-2 scores a query against every keyframe with one dot product, which is
what makes searching 177k frames instant, but it also means the top of the list
is decided by a single embedding comparison.  A vision-language model looking at
ten actual images can tell "three cyclists" from "two cyclists" in a way a
single vector cannot.  Under the official rules that is worth a lot: the right
video at rank 6-20 scores 0.6, at rank 1 it scores 1.0.

This never writes to a submission by itself.  It emits the same pick string the
review page produces, so the operator confirms it and `apply_picks.py` applies
it — a model that silently reorders rows is a model nobody can check.

Measure it before trusting it:

    python scripts/rerank_vlm.py --evaluate --limit 30

Then use it on a round:

    python scripts/rerank_vlm.py --queries round_p1/queries
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.make_submission import (  # noqa: E402
    detect_task,
    ranked_hits,
    read_en_override,
    read_query_text,
    split_qa,
)
from src.core.submission import final_score, r_score_kis  # noqa: E402


def best_per_video(hits, n_videos: int):
    """One representative frame per video — the reranker judges videos, not frames."""
    seen, out = set(), []
    for h in hits:
        if h.video_id in seen:
            continue
        seen.add(h.video_id)
        out.append(h)
        if len(out) >= n_videos:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", default=None, help="round query folder")
    ap.add_argument("--evaluate", action="store_true", help="score on ground truth instead")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--top-n", type=int, default=10, help="how many videos the VLM looks at")
    ap.add_argument("--limit", type=int, default=0, help="only the first N ground-truth queries")
    ap.add_argument("--window", type=int, default=10, help="assumed answer half-window, evaluate only")
    args = ap.parse_args()

    if not args.evaluate and not args.queries:
        print("give --queries for a round, or --evaluate to measure on ground truth")
        return 2

    from src.core.gemini_engine import GeminiAIOptimizer

    g = GeminiAIOptimizer()
    if not getattr(g, "is_ready", False):
        print("GEMINI_API_KEY is not set, so there is nothing to run.")
        print("Get a free key at https://aistudio.google.com/apikey then:")
        print('    $env:GEMINI_API_KEY="..."     (PowerShell)')
        return 2

    from src.core.kis_engine import KISEngine

    print("loading index ...", flush=True)
    eng = KISEngine(args.data).load()

    def rerank(query_vi: str, query_en, hits):
        """Returns the reranked video list, best first."""
        pool = best_per_video(hits, args.top_n)
        items = [
            {"video_id": h.video_id, "n": h.n, "frame_idx": h.frame_idx, "score": h.score}
            for h in pool
        ]
        out = g.rerank_candidates(query_en or query_vi, items, top_n=args.top_n)
        return out, {h.video_id: h for h in pool}

    if args.evaluate:
        gt = json.loads((Path(args.data) / "ground_truth.json").read_text(encoding="utf-8"))
        gt = [x for x in gt if x.get("video_id") in eng.last_frame]
        if args.limit:
            gt = gt[: args.limit]
        base_rank, new_rank = [], []
        for i, x in enumerate(gt, 1):
            hits = ranked_hits(eng, x["kis_query_vi"], x.get("kis_query_en"))
            pool = best_per_video(hits, args.top_n)
            order_before = [h.video_id for h in pool]
            out, _ = rerank(x["kis_query_vi"], x.get("kis_query_en"), hits)
            order_after = [it["video_id"] for it in out][: args.top_n]
            gv = x["video_id"]
            base_rank.append(order_before.index(gv) + 1 if gv in order_before else 999)
            new_rank.append(order_after.index(gv) + 1 if gv in order_after else 999)
            print(f"  {i:3d}/{len(gt)}  {gv:12s} rank {base_rank[-1]:3d} -> {new_rank[-1]:3d}", flush=True)

        def bucket(r):
            # rules 2.2 read as a rank bucket, which is what reordering moves
            return 1.0 if r == 1 else 0.8 if r <= 5 else 0.6 if r <= 20 else 0.4 if r <= 50 else 0.2 if r <= 100 else 0.0

        b = sum(map(bucket, base_rank)) / len(base_rank)
        n = sum(map(bucket, new_rank)) / len(new_rank)
        better = sum(1 for x, y in zip(base_rank, new_rank) if y < x)
        worse = sum(1 for x, y in zip(base_rank, new_rank) if y > x)
        print(f"\n  video R@1 : {sum(r == 1 for r in base_rank)}/{len(gt)} -> {sum(r == 1 for r in new_rank)}/{len(gt)}")
        print(f"  rank score: {b:.3f} -> {n:.3f}   ({100 * (n / b - 1):+.1f}%)")
        print(f"  moved up {better}, moved down {worse}")
        if n <= b:
            print("\n  NOT an improvement — do not put this on the submission path.")
        else:
            print("\n  Improvement. Run without --evaluate to get a pick string for the round.")
        return 0

    qdir = Path(args.queries)
    qfiles = sorted(
        p for p in qdir.glob("*.txt") if not p.name.lower().endswith((".en.txt", ".vi.txt"))
    )
    picks = []
    for qf in qfiles:
        task = detect_task(qf.name)
        if task == "trake":
            print(f"  {qf.stem:24s} skipped (a chain is judged in review.html, not here)")
            continue
        text = read_query_text(qf) or ""
        probe = split_qa(text)[0] if task == "qa" else text
        # same ranking make_submission produced, so a reranked #1 is
        # comparable to the #1 the operator sees in review.html
        hits = ranked_hits(eng, probe, read_en_override(qf))
        if not hits:
            continue
        out, by_id = rerank(probe, None, hits)
        top = out[0]
        was = hits[0].video_id
        mark = "" if top["video_id"] == was else f"   CHANGED from {was}"
        picks.append(f'{qf.stem}={top["video_id"]}:{top["frame_idx"]}')
        print(f'  {qf.stem:24s} {top["video_id"]:12s} vlm={top.get("vlm_score", "?")}{mark}')

    print("\nCheck these in review.html first, then apply:\n")
    print(f'python scripts/apply_picks.py --queries {qdir.as_posix()} --out {qdir.parent.as_posix()}/run1 \\')
    print(f'  --picks "{";".join(picks)}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
