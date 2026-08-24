"""Look at what a submission actually retrieved, and how sure it is.

Without ground truth the only honest signals are (a) how strongly the top
candidates score, and (b) whether they agree with each other.  A query whose
top-30 rows come from one video with a clear score margin is probably solved;
one whose rows are scattered over 25 videos at a flat score is a coin flip and
deserves human eyeballs first.

    python scripts/inspect_run.py --queries round_p1/queries
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.make_submission import (  # noqa: E402
    detect_task,
    ranked_hits,
    read_en_override,
    read_query_text,
    split_events,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out", default=None, help="the run directory, to check Q&A answers are filled in")
    args = ap.parse_args()

    qdir = Path(args.queries)
    qfiles = sorted(
        p for p in qdir.glob("*.txt") if not p.name.lower().endswith((".en.txt", ".vi.txt"))
    )

    from src.core.kis_engine import KISEngine

    print("loading index ...", flush=True)
    engine = KISEngine(args.data).load()
    print()

    print("=" * 108)
    print(f"{'query':24s} {'task':6s} {'top video':12s} {'margin':>7} {'conc':>5} {'spread':>7}  english rendering")
    print("=" * 108)

    rows_out = []
    trake_eng = None
    for qf in qfiles:
        text = read_query_text(qf) or ""
        task = detect_task(qf.name)
        en = read_en_override(qf)  # must match make_submission, or this reports a fiction

        if task == "trake":
            # Judging a TRAKE query by its first event alone names a video that
            # is usually NOT the one the submission picks, because the CSV comes
            # from aligning the whole ordered chain.  Report the real thing.
            from src.task3_trake import TRAKEEngine

            if trake_eng is None:
                trake_eng = TRAKEEngine(engine=engine).load_index()
            events = split_events(en or text)
            # same earliness flag make_submission uses, or this reports a
            # different winner than the one actually written to the CSV
            first = bool(re.search(r"đầu tiên|lần đầu|first", text, re.IGNORECASE))
            chains = trake_eng.align_sequence(events, first_occurrence=first, top_k=8)
            if not chains:
                print(f"{qf.stem:24s} {task:6s} (no chain)")
                continue
            best = chains[0]["score"]
            rest = [c["score"] for c in chains[1:]] or [0.0]
            sd = float(np.std([c["score"] for c in chains])) + 1e-6
            margin = (best - max(rest)) / sd
            conc = 0.0  # a chain already commits to one video; breadth has no meaning
            flag = "  " if margin > 0.8 else "??"
            en_shown = en or engine.translate(events[0] if events else text)
            print(
                f"{qf.stem:24s} {task:6s} {chains[0]['video_id']:12s} {margin:7.2f} "
                f"{'chain':>5} {best - min(c['score'] for c in chains):7.3f}  {flag} {en_shown[:38]}"
            )
            rows_out.append((qf.stem, task, chains[0]["video_id"], margin, 1.0 if margin > 0.8 else 0.0))
            continue

        hits = ranked_hits(engine, text, en)
        en_shown = en or engine.translate(text)
        if not hits:
            print(f"{qf.stem:24s} {task:6s} (no hits)")
            continue

        top = hits[: args.top]
        scores = np.array([h.score for h in top])
        vids = Counter(h.video_id for h in top)
        best_v, best_n = vids.most_common(1)[0]

        # margin: how far the best video's best frame is above the best frame
        # of any OTHER video, in units of the score spread among the top-30
        other = [h.score for h in top if h.video_id != top[0].video_id]
        margin = (top[0].score - max(other)) / (scores.std() + 1e-6) if other else 9.9
        conc = best_n / len(top)  # concentration of the top-N in one video

        # margin alone: a near-tie between the top two videos is a coin flip
        # however concentrated the frames are (see build_review_page.py)
        flag = "  " if margin > 0.8 else "??"
        print(
            f"{qf.stem:24s} {task:6s} {top[0].video_id:12s} {margin:7.2f} {conc:5.0%} "
            f"{scores.max() - scores.min():7.3f}  {flag} {en_shown[:38]}"
        )
        rows_out.append((qf.stem, task, top[0].video_id, margin, conc))

    print()
    weak = [r for r in rows_out if r[3] <= 0.8]
    print(f"{len(weak)}/{len(rows_out)} queries look uncertain (flat scores, scattered videos):")
    for name, task, v, m, c in weak:
        print(f"  {name:24s} top={v:12s} margin={m:5.2f} concentration={c:.0%}")
    print("\nThose are the ones to eyeball first — human verification turns a rank-6-20")
    print("hit (0.6) into a rank-1 hit (1.0), which is the cheapest 0.4 available.")

    # A Q&A query with no answer text scores zero on every row, so it is a
    # certain loss rather than an uncertain one — worth more attention than
    # anything on the list above.
    if args.out:
        from src.core.submission import csv_name_for_query

        blanks = []
        for name, task, *_ in rows_out:
            if task != "qa":
                continue
            csv_path = Path(args.out) / "csv" / csv_name_for_query(name + ".txt")
            try:
                first = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
            except Exception:
                continue
            if len(first) < 3 or not first[2].strip():
                blanks.append(name)
        if blanks:
            print(f"\n!! {len(blanks)} Q&A quer{'y' if len(blanks) == 1 else 'ies'} "
                  f"still have a BLANK answer — those score 0 no matter how good the frame is:")
            for b in blanks:
                print(f"  {b}")
            print("  Fix by typing the answer in review.html, or set GEMINI_API_KEY and re-run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
