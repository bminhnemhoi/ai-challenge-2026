"""Does the spoken channel actually raise the contest score?

The transcripts obviously *contain* things the pixels do not — searching them by
hand found MĂNG TÂY CHIÊN BIA for query-p1-4 at rank 1 and CỦ NĂNG OM NẤM CHAY
for query-p1-18, neither of which the visual model ranked. That is not the same
as helping, and this project has already measured one signal (object detections,
per frame) that improved video Recall@1 while LOWERING the score.

So the same discipline as everywhere else: the official formula, a non-snapped
answer key, many re-draws, and the honest negative reported if that is what comes
out. See docs/WHAT_CHANGED.md for why the recorded ground-truth frames cannot be
used as they are.

Four ways to fold the text in, each swept over weights:

    video     one BM25 bonus per VIDEO, added to all of its frames equally.
              This is the shape that worked for object detections (+3.3%);
              per-frame bonuses were worthless there.
    gate      only keep candidates from videos the transcript does not rule out
    time      bonus decays with distance from the best-matching spoken passage,
              which is the one thing a title cannot give: a TIMESTAMP
    both      video + time

    python scripts/experiment_transcripts.py
"""

from __future__ import annotations

import argparse
import json
import sys
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
from src.core.transcripts import TranscriptIndex  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--transcripts", default=str(ROOT.parent / "transcripts_full"))
    ap.add_argument("--windows", default="6,10,20,40")
    ap.add_argument("--draws", type=int, default=32)
    ap.add_argument("--top-n", type=int, default=200)
    args = ap.parse_args()

    from src.core.kis_engine import KISEngine

    print("loading index ...", flush=True)
    eng = KISEngine(args.data).load()
    tx = TranscriptIndex().load_dir(Path(args.transcripts), Path(args.data) / "captions")
    print(f"{tx.n_videos} video co loi thoai", flush=True)

    gt = json.loads((Path(args.data) / "ground_truth.json").read_text(encoding="utf-8"))
    gt = [g for g in gt if g.get("video_id") in eng.last_frame]
    covered = sum(1 for g in gt if g["video_id"] in tx.docs)
    print(f"{len(gt)} cau ground truth, {covered} co loi thoai cho video dung\n", flush=True)

    print("retrieving ...", flush=True)
    cached = []
    for g in gt:
        hits = ranked_hits(eng, g["kis_query_vi"], g.get("kis_query_en"))[: args.top_n]
        cached.append((g, hits))

    # BM25 over the candidate videos only, then squashed to [0,1]. The raw score
    # is unbounded and length-dependent, so adding it to a cosine similarity in
    # its natural units would let one long bulletin dominate every query.
    print("scoring transcripts ...", flush=True)
    text_norm = []
    for g, hits in cached:
        vids = {h.video_id for h in hits}
        raw = tx.score_videos(g["kis_query_vi"], restrict=vids)
        top = max(raw.values()) if raw else 0.0
        text_norm.append({v: (s / top if top > 0 else 0.0) for v, s in raw.items()})

    seg_time = []
    for (g, hits), tn in zip(cached, text_norm):
        best = max(tn, key=tn.get) if tn else None
        out = {}
        for v in tn:
            seg = tx.best_segment(g["kis_query_vi"], v) if tn[v] > 0.3 else None
            if seg:
                out[v] = seg[0]
        seg_time.append(out)
        del best

    kf: dict = {}
    fps: dict = {}
    for m in eng.metadata:
        kf.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
        fps[m["video_id"]] = float(m["fps"])
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

    draws = [draw(5000 + s) for s in range(args.draws)]
    windows = [int(w) for w in args.windows.split(",")]
    plan = AllocationPlan(breadth_cost=1.0, depth_cost=DEFAULT_DEPTH_COST, step=10)

    # How much does the transcript SINGLE OUT one video? Measured on the two
    # query styles this competition actually contains:
    #   ground truth  — pure visual-scene descriptions ("a dark red sedan with a
    #                   rear spoiler"). Nobody says those out loud, so the BM25
    #                   surface is flat: spreads of 0-18%, i.e. noise.
    #   round-1       — several are topical or named-entity ("research in
    #                   Lausanne", "a poem praising Nguyễn Trung Trực"). Spreads
    #                   of 21% and 33%, i.e. one video really is singled out.
    # So the bonus is gated on that spread instead of applied unconditionally.
    spread = []
    for tn in text_norm:
        vals = sorted(tn.values(), reverse=True)
        spread.append((vals[0] - vals[1]) / vals[0] if len(vals) > 1 and vals[0] > 0 else 0.0)

    def rescore(mode: str, w: float, gate_at: float = 0.0):
        per_query = []
        for qi, (g, hits) in enumerate(cached):
            tn, st = text_norm[qi], seg_time[qi]
            if not tn or mode == "none" or spread[qi] < gate_at:
                per_query.append(hits)
                continue
            scored = []
            for i, h in enumerate(hits):
                bonus = 0.0
                if mode in ("video", "both"):
                    bonus += w * tn.get(h.video_id, 0.0)
                if mode == "gate":
                    bonus += w * (1.0 if tn.get(h.video_id, 0.0) > 0.5 else 0.0)
                if mode in ("time", "both") and h.video_id in st:
                    # a spoken passage localises a moment; decay over 20 seconds,
                    # which is roughly how long one news item or one cooking step runs
                    dt = abs(h.pts_time - st[h.video_id])
                    bonus += w * tn.get(h.video_id, 0.0) * float(np.exp(-dt / 20.0))
                scored.append((h.score + bonus, i, h))
            # stable on the original position: within a video the embedding
            # order knows about timing and must not be shuffled by a tie
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
    head = f"{'variant':20s}" + "".join(f"{'W=' + str(w):>9}" for w in windows) + "     mean  vR@1   delta"
    print("\n" + head)
    print("-" * len(head))
    print(f"{'baseline (khong loi thoai)':20s}"[:20] + "".join(f"{v:9.3f}" for v in base_w)
          + f"{base:9.3f}  {base_r1:3d}       -")

    best = (base, "baseline")
    for mode in ("video", "gate", "time", "both"):
        for w in (0.005, 0.01, 0.02, 0.04, 0.08):
            per_w, r1 = evaluate(rescore(mode, w))
            mean = sum(per_w) / len(per_w)
            if mean > best[0]:
                best = (mean, f"{mode} w={w}")
            print(f"{mode + ' w=' + str(w):20s}" + "".join(f"{v:9.3f}" for v in per_w)
                  + f"{mean:9.3f}  {r1:3d}  {100 * (mean / base - 1):+6.1f}%", flush=True)

    print(f"\n--- gated: bonus chi ap dung khi loi thoai CHI RO mot video ---")
    print(f"{'so cau vuot cong':20s}" + "  ".join(
        f"g={g}: {sum(1 for s in spread if s >= g)}/{len(spread)}" for g in (0.2, 0.3, 0.4, 0.5)))
    for gate_at in (0.2, 0.3, 0.4, 0.5):
        for w in (0.02, 0.04, 0.08, 0.15):
            per_w, r1 = evaluate(rescore("video", w, gate_at))
            mean = sum(per_w) / len(per_w)
            if mean > best[0]:
                best = (mean, f"video w={w} gate={gate_at}")
            print(f"{'gated ' + str(gate_at) + ' w=' + str(w):20s}"
                  + "".join(f"{v:9.3f}" for v in per_w)
                  + f"{mean:9.3f}  {r1:3d}  {100 * (mean / base - 1):+6.1f}%", flush=True)

    print()
    if best[1] == "baseline":
        print("Khong cach nao thang duoc baseline tren ground truth.")
        print("Luu y: 60 cau ground truth deu la MO TA CANH NHIN THAY, khong ai NOI ra chung,")
        print("nen phep do nay khong noi gi ve cac cau chu de/danh tu rieng cua vong thi that.")
    else:
        print(f"Tot nhat: {best[1]} -> {best[0]:.3f}  ({100 * (best[0] / base - 1):+.1f}%)")
        print("Chi dua vao duong cham diem neu bien do lon hon nhieu chay-lai;")
        print("chay lai voi --draws 64 truoc khi doi mac dinh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
