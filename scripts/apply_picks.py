"""Apply every human-confirmed frame at once, then repackage and verify.

The review page produces one line like

    query-p1-1-kis=L21_V015:25605;query-p1-6-kis=L26_V056:12480

and this applies all of it in a single pass over one loaded index, instead of
re-loading the 780 MB matrix once per query.  With a dozen corrections that is
the difference between ten seconds and three minutes — which matters, because
this runs inside a three-hour window.

    python scripts/apply_picks.py --queries round_p1/queries --out round_p1/run1 \
        --picks "query-p1-1-kis=L21_V015:25605;query-p1-6-kis=L26_V056:12480"

Answers for Q&A queries go in the same string after a second colon:

    query-p1-15-qa=L30_V072:5376:Xã Vạn Thắng
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
    DEFAULT_DEPTH_COST,
    DEFAULT_N_FLAT,
    allocate_rows,
    detect_task,
    ranked_hits,
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
    package_submission,
    verify_submission_zip,
    write_query_csv,
)


def parse_picks(s: str):
    """`query=VIDEO:FRAME[:answer]` separated by `;` or newlines.

    A TRAKE pick carries the whole chain the operator saw and approved,
    `query=VIDEO:F1|F2|F3`, so the confirmed frames go through untouched
    instead of being re-derived.  ``frame`` is then a list.
    """
    # A picks FILE exists so the reasoning can sit next to the correction —
    # "why is this the right video" is the part worth keeping. Comments are
    # stripped LINE BY LINE, before splitting on ';', because a sentence
    # explaining a pick will contain semicolons and colons of its own.
    body = "\n".join(
        ln for ln in (s or "").splitlines() if not ln.strip().startswith("#")
    )

    out = []
    for chunk in re.split(r"[;\n]+", body):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"bad pick {chunk!r}; expected query=VIDEO:FRAME")
        q, rest = chunk.split("=", 1)
        parts = rest.split(":", 2)
        video = parts[0].strip()
        raw = parts[1].strip() if len(parts) > 1 else ""
        if "|" in raw:
            frame = [int(x) for x in raw.split("|") if x.strip()]
        else:
            frame = int(raw) if raw else None
        answer = parts[2].strip() if len(parts) > 2 else None
        out.append((q.strip(), video, frame, answer))
    return out


def pin_plan(video, last_frame, frames_chain, frame, others, default_n_flat):
    """Which candidates lead the file, and how many of them get a flat row.

    Three cases, and they want different spending:

      no frame given   the video is confirmed but not the instant, so breadth
                       over its keyframes is still the best use of the budget
      one frame        the instant is confirmed; breadth has done its job and
                       every remaining row should go into the ladder around it
      several frames   the video does the action more than once and stills
                       cannot separate the takes. Each take gets a flat row, in
                       the order given, and the first still gets the ladder.
                       R@k is a max over the first k rows, so takes 2 and 3 at
                       ranks 2 and 3 cost one row each and can only add.

    Returns (candidates, n_flat).
    """
    if frames_chain:
        wanted = list(dict.fromkeys(int(f) for f in frames_chain))
        lead = [Candidate(video, f, 1e9 - i, last_frame) for i, f in enumerate(wanted)]
        rest = [c for c in others if c.frame_idx not in set(wanted)]
        return lead + rest, min(len(wanted), len(lead) + len(rest))
    if frame is not None:
        lead = [Candidate(video, int(frame), 1e9, last_frame)]
        rest = [c for c in others if c.frame_idx != int(frame)]
        return lead + rest, 1
    return list(others), min(default_n_flat, len(others))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--picks", default=None, help="query=VIDEO:FRAME[:answer]; ...")
    ap.add_argument("--picks-file", default=None, help="same, one per line")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--n-flat", type=int, default=DEFAULT_N_FLAT)
    ap.add_argument("--depth-cost", type=float, default=DEFAULT_DEPTH_COST)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--pin-budget", type=int, default=50)
    ap.add_argument(
        "--allocator",
        choices=("hybrid", "coverage"),
        default=None,
        help="allocator for the NON-pinned rows. Default: whatever "
        "make_submission wrote into <out>/csv/allocator.txt, else hybrid — a "
        "repack must not silently switch allocator mid-contest. Rows for a "
        "pick WITH a confirmed frame always use hybrid: a human-confirmed "
        "instant has ~zero uncertainty, so the dense ladder around it IS the "
        "optimal coverage of a known point.",
    )
    ap.add_argument("--allow-blank-answers", action="store_true")
    args = ap.parse_args()

    raw = args.picks or ""
    if args.picks_file:
        # utf-8-sig: a BOM would survive .strip() and glue itself to the first
        # query name, so the first correction in the file would be silently
        # skipped as "no such query file"
        raw += "\n" + Path(args.picks_file).read_text(encoding="utf-8-sig")
    picks = parse_picks(raw)
    if not picks:
        print("ERROR: no picks given (use --picks or --picks-file)")
        return 2

    qdir = Path(args.queries)
    csv_dir = Path(args.out) / "csv"
    if not csv_dir.is_dir():
        print(f"ERROR: {csv_dir} does not exist — run make_submission.py first")
        return 2

    if args.allocator is None:
        marker = csv_dir / "allocator.txt"
        args.allocator = (
            marker.read_text(encoding="utf-8").strip() if marker.is_file() else "hybrid"
        )
        if args.allocator not in ("hybrid", "coverage"):
            print(f"ERROR: {marker} says {args.allocator!r} — not an allocator this build knows")
            return 2

    from src.core.kis_engine import KISEngine

    print(f"{len(picks)} correction(s); allocator={args.allocator}; loading the index once ...",
          flush=True)
    eng = KISEngine(args.data).load()
    plan = AllocationPlan(breadth_cost=1.0, depth_cost=args.depth_cost, step=args.step)
    trake = None
    failed = []

    for stem, video, frame, answer in picks:
        qfile = next((p for p in qdir.glob("*.txt") if p.stem == stem), None)
        if qfile is None:
            print(f"  {stem:26s} SKIP: no such query file")
            failed.append(stem)
            continue
        if video not in eng.last_frame:
            print(f"  {stem:26s} SKIP: unknown video {video!r}")
            failed.append(stem)
            continue

        text = read_query_text(qfile) or ""
        task = detect_task(qfile.name)
        en = read_en_override(qfile)
        csv_path = csv_dir / csv_name_for_query(qfile.name)

        if task == "trake":
            events = split_events(en or text)
            if isinstance(frame, list) and len(frame) == len(events):
                # the operator approved this exact chain on the review page
                frames = list(frame)
            else:
                from src.task3_trake import TRAKEEngine

                if trake is None:
                    trake = TRAKEEngine(engine=eng).load_index()
                first = bool(re.search(r"đầu tiên|lần đầu|first", text, re.IGNORECASE))
                res = trake.align_sequence(
                    events, video_id=video, first_occurrence=first, top_k=1
                )
                if not res:
                    print(f"  {stem:26s} SKIP: alignment found nothing inside {video}")
                    failed.append(stem)
                    continue
                frames = res[0]["sequence_frames"]
                if isinstance(frame, list) and frame:
                    print(
                        f"  {stem:26s} ! {len(frame)} frames given but query has "
                        f"{len(events)} events — realigned instead"
                    )
                elif frame is not None:
                    # the human confirmed event 1; shift the whole chain to match
                    shift = frame - frames[0]
                    frames = [max(0, f + shift) for f in frames]
            rows = allocate_trake_rows(
                video, frames, budget=MAX_ROWS, step=args.step,
                video_last_frame=eng.last_frame[video],
            )
            flat = [(v, *f) for v, f in rows]
            print(f"  {stem:26s} {video} frames {frames}")
        else:
            # A KIS chain is a hedge, not a mistake. Some queries describe an
            # action the video performs several times — the lion takes the
            # pumpkin at 6:37, 8:11, 8:55 and 9:44 — and stills cannot say which
            # one the key means. R@k is a max over the first k rows, so a second
            # and third instant at ranks 2 and 3 cost almost nothing and turn a
            # coin-flip into three chances. Only the first gets the deep ladder.
            frames_chain = [int(x) for x in frame] if isinstance(frame, list) else None
            frame = frames_chain[0] if frames_chain else frame
            probe = text
            if task == "qa":
                probe, _ = split_qa(text)
                if answer is None:
                    try:
                        parts = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
                        answer = parts[2].strip() if len(parts) > 2 else ""
                    except Exception:
                        answer = ""

            # ranked_hits, not eng.search(..., query_en=en) — the two rank
            # differently, and only ranked_hits is what make_submission wrote
            # and what the operator saw on the review page
            hits = ranked_hits(eng, probe or text, en)
            pinned = [h for h in hits if h.video_id == video]
            rest = [h for h in hits if h.video_id != video]
            if not pinned:
                sims = eng.query_similarities(probe or text, en)
                idx = [i for i, v in enumerate(eng.video_id) if v == video and eng.valid[i]]
                idx.sort(key=lambda i: -sims[i])
                from src.core.kis_engine import Hit

                pinned = [
                    Hit(video, int(eng.frame_idx[i]), float(sims[i]), int(eng.n_in_video[i]),
                        float(eng.pts_time[i]), eng.last_frame[video])
                    for i in idx[:40]
                ]

            pin_cands = [Candidate(h.video_id, h.frame_idx, h.score, h.video_last_frame)
                         for h in pinned]
            pin_cands, pin_n_flat = pin_plan(
                video,
                eng.last_frame[video],
                frames_chain,
                frame,
                pin_cands,
                min(args.n_flat, len(pin_cands)),
            )

            pin_alloc_plan = AllocationPlan(
                breadth_cost=plan.breadth_cost, depth_cost=plan.depth_cost,
                step=plan.step, budget=args.pin_budget,
            )
            if frames_chain or frame is not None:
                # A confirmed frame must lead the file literally.  Coverage's
                # softmax at nhiet 0.02 would collapse the prior onto GRID
                # cells near the pin, not the pinned frame itself — row 1
                # would drift off the instant the human approved.  The score
                # 1e9 sentinel from pin_plan also has no meaning to a softmax.
                pin_rows = allocate_hybrid_rows(pin_cands, n_flat=pin_n_flat, plan=pin_alloc_plan)
            else:
                # Video confirmed but not the instant: the candidates keep
                # their real retrieval scores, so the chosen allocator runs
                # exactly as it would in make_submission.
                pin_rows = allocate_rows(pin_cands, args.allocator, pin_n_flat, pin_alloc_plan)
            rest_rows = allocate_rows(
                [Candidate(h.video_id, h.frame_idx, h.score, h.video_last_frame) for h in rest],
                args.allocator, args.n_flat, plan,
            )
            seen = set(pin_rows)
            rows = pin_rows + [r for r in rest_rows if r not in seen]
            flat = rows if task != "qa" else [(v, f, answer or "") for v, f in rows]
            note = f" answer={answer!r}" if task == "qa" else ""
            shown = (
                "|".join(str(f) for f in frames_chain)
                if frames_chain
                else (frame if frame is not None else "(best)")
            )
            print(f"  {stem:26s} {video} frame {shown}{note}")
            if task == "qa" and not (answer or "").strip():
                print("      ! answer still blank — these rows score 0")

        write_query_csv(csv_path, flat[:MAX_ROWS])

    zip_path = Path(args.out) / "submission.zip"
    package_submission(csv_dir, zip_path)
    expect = {
        csv_name_for_query(p.name)
        for p in qdir.glob("*.txt")
        if not p.name.lower().endswith((".en.txt", ".vi.txt"))
    }
    problems = verify_submission_zip(
        zip_path, expect_names=expect, allow_blank_answers=args.allow_blank_answers
    )

    print(f"\nrepacked -> {zip_path} ({zip_path.stat().st_size / 1024:.0f} KB)")
    if failed:
        print(f"  {len(failed)} pick(s) skipped: {', '.join(failed)}")
    if problems:
        print("\nFORMAT PROBLEMS — do not upload this:")
        for p in problems:
            print("  -", p)
        return 1
    print("format check passed. This zip is safe to upload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
