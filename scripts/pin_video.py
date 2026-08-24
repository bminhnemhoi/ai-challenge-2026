"""Pin a human-verified video to the top of one query's answers.

This is where the biggest single gain in a round comes from.  A person
recognises the right shot in a grid of thumbnails in about two seconds; the
scoring rules then pay 1.0 for rank 1 versus 0.6 for rank 6-20, so every
confirmed video is +0.4 on that query for almost no time spent.

    python scripts/pin_video.py --queries round_p1/queries --out round_p1/run1 \
        --query query-p1-24-kis --video L23_V013
    python scripts/repackage.py --out round_p1/run1

The confirmed video takes the first ``--pin-budget`` rows (50 by default) with
a frame ladder built around it; the original ranking keeps the remaining rows.
Because the score is a step function of the first hit's rank — 1.0 at rank 1,
0.8 at 2-5, 0.6 at 6-20, 0.4 at 21-50, 0.2 at 51-100 — that split costs almost
nothing when the human is right and still leaves a scoring chance when they are
not.  Handing every row to the confirmed video would turn one misidentification
into a zero.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.make_submission import (  # noqa: E402
    ranked_hits,
    DEFAULT_DEPTH_COST,
    DEFAULT_N_FLAT,
    detect_task,
    read_en_override,
    read_query_text,
    split_events,
    split_qa,
)
from src.core.submission import (  # noqa: E402
    MAX_ROWS,
    AllocationPlan,
    Candidate,
    allocate_hybrid_rows,
    allocate_trake_rows,
    csv_name_for_query,
    write_query_csv,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True, help="the make_submission --out directory")
    ap.add_argument("--query", required=True, help="query stem, e.g. query-p1-24-kis")
    ap.add_argument("--video", required=True, help="the video you confirmed, e.g. L23_V013")
    ap.add_argument(
        "--frame",
        type=int,
        default=None,
        help="the exact frame you saw (from the review page). Rank 1 becomes that "
        "frame with the ladder built around it — the strongest thing a human can do.",
    )
    ap.add_argument("--answer", default=None, help="Q&A only: the answer to stamp on every row")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--n-flat", type=int, default=DEFAULT_N_FLAT)
    ap.add_argument("--depth-cost", type=float, default=DEFAULT_DEPTH_COST)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument(
        "--pin-budget",
        type=int,
        default=50,
        help="rows reserved for the confirmed video; the rest keep the original "
        "ranking as insurance against a misidentification (default 50)",
    )
    args = ap.parse_args()

    qdir = Path(args.queries)
    qfile = next((p for p in qdir.glob("*.txt") if p.stem == args.query), None)
    if qfile is None:
        print(f"ERROR: no query file named {args.query}.txt in {qdir}")
        return 2

    csv_dir = Path(args.out) / "csv"
    if not csv_dir.is_dir():
        print(f"ERROR: {csv_dir} does not exist — run make_submission.py first")
        return 2

    text = read_query_text(qfile) or ""
    task = detect_task(qfile.name)
    en = read_en_override(qfile)

    from src.core.kis_engine import KISEngine

    print("loading index ...", flush=True)
    eng = KISEngine(args.data).load()

    if args.video not in eng.last_frame:
        print(f"ERROR: unknown video id {args.video!r}")
        return 2

    csv_path = csv_dir / csv_name_for_query(qfile.name)

    if task == "trake":
        from src.task3_trake import TRAKEEngine

        events = split_events(en or text)
        trake = TRAKEEngine(engine=eng).load_index()
        first = bool(re.search(r"đầu tiên|lần đầu|first", text, re.IGNORECASE))
        # align INSIDE the confirmed video: the rules score a wrong video 0, so
        # once a human has identified it there is no reason to consider others
        res = trake.align_sequence(
            events, video_id=args.video, first_occurrence=first, top_k=1
        )
        if not res:
            print("ERROR: alignment produced nothing inside that video")
            return 1
        rows = allocate_trake_rows(
            args.video, res[0]["sequence_frames"], budget=MAX_ROWS, step=args.step,
            video_last_frame=eng.last_frame[args.video],
        )
        flat = [(v, *f) for v, f in rows]
        print(f"  aligned inside {args.video}: frames {res[0]['sequence_frames']}")
    else:
        probe = text
        answer = args.answer
        if task == "qa":
            probe, _q = split_qa(text)
            if answer is None:
                # keep whatever answer the previous run wrote
                try:
                    first_line = csv_path.read_text(encoding="utf-8").splitlines()[0]
                    parts = first_line.split(",")
                    answer = parts[2].strip() if len(parts) > 2 else ""
                except Exception:
                    answer = ""

        hits = ranked_hits(eng, probe or text, en)
        pinned = [h for h in hits if h.video_id == args.video]
        rest = [h for h in hits if h.video_id != args.video]
        if not pinned:
            # the confirmed video did not make the top-400: score its frames directly
            sims = eng.query_similarities(probe or text, en)
            idx = [i for i, v in enumerate(eng.video_id) if v == args.video and eng.valid[i]]
            idx.sort(key=lambda i: -sims[i])
            from src.core.kis_engine import Hit

            pinned = [
                Hit(args.video, int(eng.frame_idx[i]), float(sims[i]), int(eng.n_in_video[i]),
                    float(eng.pts_time[i]), eng.last_frame[args.video])
                for i in idx[:40]
            ]
            print(f"  {args.video} was outside the top-400; scored its own frames instead")

        plan = AllocationPlan(breadth_cost=1.0, depth_cost=args.depth_cost, step=args.step)

        pin_cands = [
            Candidate(h.video_id, h.frame_idx, h.score, h.video_last_frame) for h in pinned
        ]
        if args.frame is not None:
            # a human looked at this exact frame and said yes; nothing the
            # retriever believes should outrank that
            exact = Candidate(args.video, int(args.frame), 1e9, eng.last_frame[args.video])
            pin_cands = [exact] + [
                c for c in pin_cands if c.frame_idx != int(args.frame)
            ]

        # The confirmed video takes the expensive ranks (1-50, worth 1.0 down to
        # 0.4) and the original ranking keeps 51-100 (worth 0.2).  That split is
        # nearly free when the human is right and is the only thing standing
        # between a misidentification and a zero, so it is not optional.
        # With --frame the operator has confirmed the instant, not just the
        # video, so breadth inside that video is spent. Measured on the ground
        # truth for exactly this case, laddering straight off the confirmed
        # frame scores 0.810 against 0.654 for n_flat=30 (+24%): the other
        # keyframes sit 55+ frames away and cannot land in a ~10 frame window,
        # while F±10 / F±20 can.
        pin_n_flat = 1 if args.frame is not None else min(args.n_flat, len(pin_cands))
        pin_rows = allocate_hybrid_rows(
            pin_cands,
            n_flat=pin_n_flat,
            plan=AllocationPlan(
                breadth_cost=plan.breadth_cost, depth_cost=plan.depth_cost,
                step=plan.step, budget=args.pin_budget,
            ),
        )
        rest_rows = allocate_hybrid_rows(
            [Candidate(h.video_id, h.frame_idx, h.score, h.video_last_frame) for h in rest],
            n_flat=args.n_flat,
            plan=plan,
        )
        seen = set(pin_rows)
        rows = pin_rows + [r for r in rest_rows if r not in seen]
        flat = rows if task != "qa" else [(v, f, answer or "") for v, f in rows]
        print(f"  {args.video} now holds {sum(1 for r in rows if r[0] == args.video)} of the top rows")
        if task == "qa" and not (answer or "").strip():
            print("  ! the answer column is still blank — pass --answer \"...\"")

    n = write_query_csv(csv_path, flat[:MAX_ROWS])
    print(f"  rewrote {csv_path.name}: {n} rows, rank 1 = {flat[0]}")
    print("\nWhen every correction is done: python scripts/repackage.py --out " + args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
