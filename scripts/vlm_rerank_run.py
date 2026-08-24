"""Rerank a round's candidates with the vision model, then rewrite the CSVs.

Measured before it was allowed near a submission, on all 60 ground-truth queries
with the official formula and a non-snapped answer key:

    baseline                       0.387   video R@1 25/60
    per-video bonus, w=0.02        0.400   (+3.3%)   <- shipping
    per-frame bonus, w=0.01        0.398   (+2.8%)
    per-video bonus, w=0.10        0.379   (-2.1%)   video R@1 29/60
    per-video bonus, w=0.20        0.365   (-5.7%)   video R@1 29/60

Two things in that table matter more than the headline. First, the gain is small
and only exists at LOW weight: let the VLM shout and it overrides the embedding's
sense of timing and the score falls. Second, the last two rows are this project's
recurring trap in miniature — video R@1 climbing from 25 to 29 while the contest
score drops, because the frame the VLM likes best is not the frame nearest the
answer instant.

An earlier run on 20 queries showed +7.3%; on all 60 it is +3.3%. The first
number was small-sample noise and the second is the one to believe.

For TRAKE the model judges each event separately inside each candidate chain,
which is the only tool here that can tell a yellow lion from a red one. Asked
which of six lion-dance videos shows a yellow-black-white lion, it scored the
right one 100 and the rest 0-30.

    python scripts/vlm_rerank_run.py --queries round_p1/queries --out round_p1/final

Everything is cached, so re-running after a change costs nothing for the frames
already judged. Rows the operator pinned by hand are never overwritten — use
apply_picks.py after this, not before.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.make_submission import (  # noqa: E402
    DEFAULT_DEPTH_COST,
    DEFAULT_N_FLAT,
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
from src.core.transcripts import TranscriptIndex  # noqa: E402
from src.core.vlm import DEFAULT_MODEL, VLMJudge  # noqa: E402

#: measured optimum on all 60 ground-truth queries. Anything above 0.05 turns
#: negative, so this is a ceiling and not a starting point
DEFAULT_WEIGHT = 0.02


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--weight", type=float, default=DEFAULT_WEIGHT)
    ap.add_argument("--mode", choices=("video", "frame"), default="video",
                    help="video: one bonus per video, applied to all its frames — measured "
                    "best (+3.3%%) and leaves the embedding's within-video order alone, "
                    "which is the order that knows about timing")
    ap.add_argument("--judge", type=int, default=24, help="candidates the VLM looks at per query")
    ap.add_argument("--judge-trake", type=int, default=6, help="chains judged per TRAKE query")
    ap.add_argument("--trake-neighbours", type=int, default=3,
                    help="keyframes either side of each event the VLM also looks at. "
                    "Sparse sampling gave a false negative on all three candidates for "
                    "query-p1-4, where the frying lasted seconds inside a five-minute clip")
    ap.add_argument("--n-flat", type=int, default=DEFAULT_N_FLAT)
    ap.add_argument("--depth-cost", type=float, default=DEFAULT_DEPTH_COST)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--allow-blank-answers", action="store_true")
    ap.add_argument(
        "--transcripts",
        default=str(ROOT.parent / "transcripts_full"),
        help="folders of transcript JSON; their top videos are ADDED to the shortlist "
        "the VLM judges, which is the one hole that lost query-p1-19 and query-p1-22",
    )
    ap.add_argument("--from-speech", type=int, default=4,
                    help="videos to add from the spoken channel per query (0 = off)")
    args = ap.parse_args()

    judge = VLMJudge(args.data, model=args.model)
    if not judge.ready:
        print("Khong co GEMINI_API_KEY. Dat vao .env:  GEMINI_API_KEY=...")
        return 2

    qdir = Path(args.queries)
    out_dir = Path(args.out)
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    qfiles = sorted(
        p for p in qdir.glob("*.txt") if not p.name.lower().endswith((".en.txt", ".vi.txt"))
    )

    from src.core.kis_engine import KISEngine

    print(f"model: {args.model}   trong so: {args.weight}\nloading index ...", flush=True)
    eng = KISEngine(args.data).load()
    meta = {(m["video_id"], m["frame_idx"]): m for m in eng.metadata}
    plan = AllocationPlan(breadth_cost=1.0, depth_cost=args.depth_cost, step=args.step)
    trake_eng = None

    # The VLM can only judge what it is shown, and what it is shown was chosen by
    # an embedding that cannot represent a proper noun. Both Q&A queries whose
    # answers were on the wrong video had the RIGHT video outside SigLIP's
    # top-24 — no amount of reranking reaches it. So the shortlist is widened
    # first, from the channel that does carry proper nouns.
    #
    # This adds candidates rather than adding score. R@k is a maximum over a
    # prefix, so a wrong extra candidate costs a rank slot and nothing else,
    # while a right one that was never in the list is worth a whole query. That
    # asymmetry is why widening is safe where the measured score bonus was not.
    tx = None
    if args.from_speech:
        try:
            tx = TranscriptIndex().load_dir(
                *[Path(d) for d in args.transcripts.split(",") if d.strip()],
                Path(args.data) / "captions",
            )
            print(f"loi thoai: {tx.n_videos} video (them toi {args.from_speech} ung vien/cau)")
            if not tx.n_videos:
                tx = None
        except Exception as exc:  # noqa: BLE001
            print(f"  ! bo qua loi thoai ({type(exc).__name__}: {exc})")
            tx = None

    by_video: dict = {}
    for m in eng.metadata:
        by_video.setdefault(m["video_id"], []).append(m)

    def speech_extras(query_text: str, already: set, k: int):
        """Keyframes of the top videos by spoken match, excluding known ones.

        Frames are taken from the WHOLE video, not only around the passage where
        the words fall. query-p1-19's presenter says "đây là Đình thần Nguyễn…"
        at 1:56 while the couplet the question asks about is on a plaque at 3:44;
        anchoring on the speech would have added three frames from the wrong
        part of the video and missed it.
        """
        if tx is None or k <= 0:
            return []
        scores = tx.score_videos(query_text)
        if not scores:
            return []
        ranked = [v for v, _s in sorted(scores.items(), key=lambda kv: -kv[1])
                  if v not in already][:k]
        out = []
        for v in ranked:
            frames = sorted(by_video.get(v, []), key=lambda m: m["frame_idx"])
            if not frames:
                continue
            step = max(1, len(frames) // 6)
            for m in frames[::step][:6]:
                out.append((v, int(m["frame_idx"]), m["frame_filename"]))
        return out
    t0 = time.time()
    changed = 0

    for qi, qf in enumerate(qfiles, 1):
        text = read_query_text(qf) or ""
        task = detect_task(qf.name)
        en = read_en_override(qf)
        csv_path = csv_dir / csv_name_for_query(qf.name)

        if task == "trake":
            from src.task3_trake import TRAKEEngine

            if trake_eng is None:
                trake_eng = TRAKEEngine(engine=eng).load_index()
            events = split_events(en or text)
            first = bool(re.search(r"đầu tiên|lần đầu|first", text, re.IGNORECASE))
            chains = trake_eng.align_sequence(
                events, first_occurrence=first, top_k=args.judge_trake
            )
            if not chains:
                print(f"  {qf.stem:24s} khong co chuoi nao")
                continue

            # Each event is judged against its own description inside each
            # candidate chain, then the chain's score is the mean. A chain that
            # matches three events well and one badly should beat one that
            # matches all four vaguely.
            # Each event is judged over a NEIGHBOURHOOD of keyframes, not only the
            # single frame the DP chose. Sampling density is not a detail here:
            # judging query-p1-4's three candidate videos from 8 frames spread
            # evenly across each gave a FALSE NEGATIVE on all three, because the
            # frying lasts seconds inside a five-minute clip and none of the 8
            # landed on it; 16 frames from the second half scored the right video
            # 100 and the wrong one 20. Asking only about the DP's own frame also
            # cannot correct the DP when it is wrong — the neighbourhood can.
            best_chain, best_score, best_frames = chains[0], -1.0, None
            for c in chains:
                vid = c["video_id"]
                tl = sorted(int(m["frame_idx"]) for m in by_video.get(vid, []))
                per_event, picked = [], []
                for j, f in enumerate(c["sequence_frames"]):
                    ev = events[j] if j < len(events) else text
                    k = tl.index(int(f)) if int(f) in tl else None
                    nb = args.trake_neighbours
                    window = tl[max(0, k - nb) : k + nb + 1] if k is not None else [int(f)]
                    cands = [
                        (vid, fr, meta[(vid, fr)]["frame_filename"])
                        for fr in window
                        if (vid, fr) in meta
                    ]
                    if not cands:
                        continue
                    got = judge.score(f"{text}\n\nSỰ KIỆN CẦN TÌM: {ev}", cands)
                    scored = [(got.get((vid, fr), (0.0, ""))[0], fr) for _v, fr, _fn in cands]
                    s_ev, fr_ev = max(scored) if scored else (0.0, int(f))
                    per_event.append(s_ev)
                    picked.append(fr_ev)
                s = sum(per_event) / max(len(per_event), 1)
                if s > best_score:
                    best_score, best_chain = s, c
                    # keep the model's own frames only when they stay strictly
                    # ordered: column j must hold event j, and a chain that goes
                    # backwards is not a chain
                    best_frames = (
                        picked
                        if len(picked) == len(c["sequence_frames"])
                        and picked == sorted(picked)
                        and len(set(picked)) == len(picked)
                        else None
                    )
            if best_chain is not chains[0]:
                changed += 1
                print(f"  {qf.stem:24s} {chains[0]['video_id']} -> {best_chain['video_id']}"
                      f"  (VLM {best_score:.2f})")
            else:
                print(f"  {qf.stem:24s} giu {best_chain['video_id']}  (VLM {best_score:.2f})")
            frames = best_frames or list(best_chain["sequence_frames"])
            if best_frames and best_frames != list(best_chain["sequence_frames"]):
                moved = sum(1 for x, y in zip(best_frames, best_chain["sequence_frames"]) if x != y)
                print(f"       VLM doi {moved}/{len(frames)} frame trong chuoi")
            rows = allocate_trake_rows(
                best_chain["video_id"], frames,
                budget=MAX_ROWS, step=args.step,
                video_last_frame=eng.last_frame.get(best_chain["video_id"]),
            )
            write_query_csv(csv_path, [(v, *f) for v, f in rows][:MAX_ROWS])
            continue

        probe = split_qa(text)[0] if task == "qa" else text
        hits = ranked_hits(eng, probe, en)
        top = hits[: args.judge]
        cands = [
            (h.video_id, h.frame_idx, meta[(h.video_id, h.frame_idx)]["frame_filename"])
            for h in top
            if (h.video_id, h.frame_idx) in meta
        ]
        extras = speech_extras(text, {h.video_id for h in top}, args.from_speech)
        if extras:
            print(f"  {qf.stem:24s} + {len({e[0] for e in extras})} video tu loi thoai: "
                  f"{', '.join(sorted({e[0] for e in extras}))}")
        scores = judge.score(text, cands + extras)

        was = hits[0].video_id
        if args.mode == "video":
            per_video: dict = {}
            for (v, f), (s, _why) in scores.items():
                per_video[v] = max(per_video.get(v, 0.0), s)
            bonus = lambda h: args.weight * per_video.get(h.video_id, 0.0)  # noqa: E731
        else:
            bonus = lambda h: args.weight * scores.get(  # noqa: E731
                (h.video_id, h.frame_idx), (0.0, "")
            )[0]
        rescored = sorted(
            ((h.score + bonus(h), i, h) for i, h in enumerate(hits)),
            key=lambda t: (-t[0], t[1]),
        )
        hits = [t[2] for t in rescored]
        if hits[0].video_id != was:
            changed += 1
            why = scores.get((hits[0].video_id, hits[0].frame_idx), (0.0, ""))[1]
            print(f"  {qf.stem:24s} {was} -> {hits[0].video_id}   {why[:60]}")
        else:
            print(f"  {qf.stem:24s} giu {was}")

        cs = [Candidate(h.video_id, h.frame_idx, h.score, h.video_last_frame) for h in hits]

        # A speech-found video is APPENDED, never inserted. R@k is a maximum over
        # the first k rows, so a row added at the END can only ever raise the
        # score — while one inserted at the top displaces whatever was there and
        # can lower it. The first version of this let speech candidates compete
        # for rank 1 and it reshuffled four queries that were already right.
        known = {(c.video_id, c.frame_idx) for c in cs}
        tail = [
            (scores.get((v, f), (0.0, ""))[0], v, f)
            for v, f, _fn in extras
            if (v, f) not in known
        ]
        for s, v, f in sorted(tail, reverse=True):
            if s >= 0.5:
                cs.append(Candidate(v, f, -1.0, eng.last_frame.get(v, f)))
                known.add((v, f))

        rows = allocate_hybrid_rows(cs, n_flat=args.n_flat, plan=plan)[:MAX_ROWS]
        if task == "qa":
            # keep any answer a previous run or a human already wrote
            answer = ""
            try:
                parts = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
                answer = parts[2].strip() if len(parts) > 2 else ""
            except Exception:  # noqa: BLE001
                answer = ""
            rows = [(v, f, answer) for v, f in rows]
        write_query_csv(csv_path, rows)

        if qi % 5 == 0:
            el = time.time() - t0
            print(f"     [{qi}/{len(qfiles)}  {el/qi:.0f}s/cau  {judge.cost_note()}]", flush=True)

    zip_path = out_dir / "submission.zip"
    package_submission(csv_dir, zip_path)
    expect = {csv_name_for_query(p.name) for p in qfiles}
    problems = verify_submission_zip(
        zip_path, expect_names=expect, allow_blank_answers=args.allow_blank_answers
    )
    print(f"\n{changed}/{len(qfiles)} cau doi video hang 1 nho VLM")
    print(judge.cost_note())
    if judge.errors:
        print(f"! {len(judge.errors)} loi goi API, vi du: {judge.errors[0]}")
    print(f"\n-> {zip_path} ({zip_path.stat().st_size / 1024:.0f} KB)")
    if problems:
        print("\nFORMAT PROBLEMS — dung nop:")
        for p in problems:
            print("  -", p)
        return 1
    print("format check passed. Zip nay nop duoc.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
