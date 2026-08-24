"""Settle a named hypothesis: is the answer in THIS video, and at WHICH keyframe?

The retrieval engine ranks 177k keyframes with one static-image encoder, so it
answers "which frame looks most like the sentence" — not "which video is the
report about". Speech and titles answer the second question far better, and on
this round they disagreed with the engine on several queries.

A disagreement is not evidence. This resolves one by looking: every keyframe of
the named video is shown to the VLM against the query, so the verdict rests on
the pixels rather than on either ranking. It reports the best frame and the
whole profile, which is what tells a real hit (a sharp peak where the moment is)
from a topical near-miss (a flat, mediocre plateau).

    python scripts/verify_hypotheses.py --pairs "query-p1-19-kis=L99_V001,L99_V002"
    python scripts/verify_hypotheses.py --file round1/hypotheses.txt --model gemini-3.5-flash

Writes nothing. Print it, read it, then record the decision as a pick.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.make_submission import read_query_text  # noqa: E402
from src.core.vlm import DEFAULT_MODEL, VLMJudge  # noqa: E402


def parse_pairs(text: str) -> list[tuple[str, list[str]]]:
    """`query=V1,V2; query2=V3` -> [(query, [V1, V2]), ...], `#` comments out a line."""
    lines = [ln.split("#", 1)[0] for ln in text.splitlines()]
    out: list[tuple[str, list[str]]] = []
    for chunk in ";".join(lines).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        stem, _, vids = chunk.partition("=")
        vids = [v.strip() for v in vids.split(",") if v.strip()]
        if stem.strip() and vids:
            out.append((stem.strip(), vids))
    return out


def evenly(items: list, cap: int) -> list:
    """Keep at most `cap`, spread across the whole list rather than the head.

    A video's keyframes are chronological, so slicing the head would ask about
    the first two minutes and call the rest absent.
    """
    if len(items) <= cap:
        return items
    step = len(items) / cap
    return [items[int(i * step)] for i in range(cap)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--queries", default=str(ROOT / "round1" / "queries"))
    ap.add_argument("--pairs", default="", help="query=VIDEO[,VIDEO]; ...")
    ap.add_argument("--file", default="", help="same syntax, one per line")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-frames", type=int, default=40, help="keyframes judged per video")
    ap.add_argument(
        "--range",
        dest="frange",
        default="",
        help="LO-HI: judge only keyframes in this frame window, every one of them. "
             "A transcript timestamp names the window; this reads it densely.",
    )
    ap.add_argument(
        "--head",
        type=int,
        default=0,
        help="judge only the FIRST n keyframes instead of spreading over the whole video. "
             "For a query that says what the clip OPENS with, this turns a 43-video sweep "
             "into a few hundred images.",
    )
    ap.add_argument("--show", type=int, default=6, help="best frames printed per video")
    ap.add_argument("--question", default="", help="override the query text (e.g. ask about one event)")
    ap.add_argument(
        "--questions-json",
        default="",
        help="{query-stem: sharp question} — the frame-pinning question, which has to "
             "differ from the retrieval query or every frame of a topical video scores high",
    )
    args = ap.parse_args()

    sharp: dict[str, str] = {}
    if args.questions_json:
        sharp = {
            k: v
            for k, v in json.loads(Path(args.questions_json).read_text(encoding="utf-8")).items()
            if not k.startswith("_") and isinstance(v, str)
        }

    text = args.pairs
    if args.file:
        text = (text + ";" + Path(args.file).read_text(encoding="utf-8")).strip(";")
    pairs = parse_pairs(text)
    if not pairs:
        print("Khong co cap nao de kiem chung.")
        return 2

    judge = VLMJudge(args.data, model=args.model)
    if not judge.ready:
        print("Khong co GEMINI_API_KEY trong .env")
        return 2

    meta = json.loads((Path(args.data) / "metadata.json").read_text(encoding="utf-8"))
    by_video: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for m in meta:
        by_video[m["video_id"]].append((int(m["frame_idx"]), m["frame_filename"]))
    for v in by_video:
        by_video[v].sort()

    qdir = Path(args.queries)
    for stem, vids in pairs:
        qpath = qdir / f"{stem}.txt"
        query = (
            args.question
            or sharp.get(stem)
            or (read_query_text(qpath) if qpath.exists() else stem)
        )
        print(f"\n{'=' * 78}\n{stem}\n  {query[:200]}")
        table = []
        for vid in vids:
            frames = by_video.get(vid)
            if not frames:
                print(f"  {vid}: KHONG CO trong metadata")
                continue
            if args.frange:
                lo, _, hi = args.frange.partition("-")
                lo, hi = int(lo), int(hi)
                picked = [(f, fn) for f, fn in frames if lo <= f <= hi]
            elif args.head:
                picked = frames[: args.head]
            else:
                picked = evenly(frames, args.max_frames)
            cands = [(vid, f, fn) for f, fn in picked]
            scores = judge.score(query, cands)
            # A frame with no verdict was never looked at. Folding it in as 0.0
            # would let a dead network read as "the model saw it and said no",
            # which is how an empty run once passed for a decisive one.
            got = sorted(
                ((scores[(vid, f)][0], f, scores[(vid, f)][1])
                 for f, _fn in picked if (vid, f) in scores),
                reverse=True,
            )
            missing = len(picked) - len(got)
            if not got:
                print(f"\n  --- {vid}: KHONG CO KET QUA cho ca {len(picked)} khung hinh"
                      f" — chua xet duoc, dung ket luan gi tu day.")
                continue
            best = got[0]
            hi = sum(1 for s, _f, _w in got if s >= 0.6)
            table.append((best[0], vid, best[1], hi, len(got)))
            print(f"\n  --- {vid}  ({len(frames)} keyframe, xet {len(got)}"
                  + (f", THIEU {missing}" if missing else "")
                  + f")  cao nhat {best[0]:.2f}  so frame >=0.60: {hi}")
            for s, f, why in got[: args.show]:
                print(f"      {s:5.2f}  frame {f:<7d} {str(why)[:96]}")
        if len(table) > 1:
            table.sort(reverse=True)
            win = table[0]
            gap = win[0] - table[1][0]
            print(f"\n  >>> DAN DAU: {win[1]} frame {win[2]}  ({win[0]:.2f}, hon ke sau {gap:+.2f})")
    print(f"\n{judge.cost_note()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
