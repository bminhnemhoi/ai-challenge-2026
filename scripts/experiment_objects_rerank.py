"""Do the BTC object detections actually improve the ranking?

Every thumbnail in review.html already shows what the detector found —
"Woman x2, Boy x2, Girl". The question is whether that belongs in the *score*
as well as on screen. It looks obviously useful: half the round-1 queries name
a thing and often a count ("three cyclists", "four children"), and the
detections give an exact count.

The catch is that a detector class is a very blunt instrument. "Person" fires
on 57% of all keyframes, so "contains a person" is a constant, not a filter.
And SigLIP-2 has already seen the same pixels — anything the detector knows
about a frame is plausibly already in the embedding.

So this measures four ways of folding detections into the score, against the
official formula on a non-snapped answer key (see experiment_allocation.py for
why the recorded ground-truth frames cannot be used as they are):

    match     + bonus per query noun that the frame's detections contain
    count     + bonus only when a number in the query matches the class count
    filter    - penalty when a noun named in the query is absent
    both      match + count

    python scripts/experiment_objects_rerank.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
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

OBJ_CONF = 0.4
#: classes so common they carry no information (measured over the whole corpus)
UNINFORMATIVE = {
    "clothing", "human face", "human body", "human head", "human arm", "human leg",
    "human hand", "human nose", "human hair", "human eye", "human mouth", "human ear",
    "footwear", "sports equipment",
}
#: English number words the organisers actually use in query text
NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def load_detections(data_dir: Path, wanted: set) -> dict:
    """(video_id, frame_stem) -> Counter of class -> count, for the frames we score."""
    out: dict = {}
    unpacked = data_dir / "objects"
    zpath = data_dir / "objects-aic25-b1.zip"

    def absorb(key, payload):
        try:
            j = json.loads(payload)
        except Exception:
            return
        c = Counter()
        for cls, sc in zip(j.get("detection_class_entities", []), j.get("detection_scores", [])):
            cls = str(cls).strip().lower()
            if float(sc) > OBJ_CONF and cls not in UNINFORMATIVE:
                c[cls] += 1
        out[key] = c

    if unpacked.is_dir():
        for vid, stem in wanted:
            f = unpacked / vid / f"{stem}.json"
            if f.exists():
                absorb((vid, stem), f.read_bytes())
        if out:
            return out
    if not zpath.exists():
        return out
    with zipfile.ZipFile(zpath) as zf:
        for n in zf.namelist():
            if not n.endswith(".json"):
                continue
            parts = Path(n).parts
            if len(parts) < 2:
                continue
            key = (parts[-2], Path(n).stem)
            if key in wanted:
                absorb(key, zf.read(n))
    return out


def query_terms(text: str, vocab: set):
    """Detector classes named in the query, and any count asked for.

    Matching is on the class name appearing as a whole phrase in the query, so
    "bicycle" matches "Bicycle" and "bicycle wheel" only matches the longer
    phrase. Counts come from a digit or number word immediately before it.
    """
    low = " " + re.sub(r"[^a-z0-9 ]+", " ", text.lower()) + " "
    low = re.sub(r"\s+", " ", low)
    found = {}
    for cls in vocab:
        if f" {cls} " not in low and f" {cls}s " not in low:
            continue
        want = None
        m = re.search(rf"(\d+|{'|'.join(NUMBERS)})\s+(?:\w+\s+){{0,2}}{re.escape(cls)}s?\b", low)
        if m:
            tok = m.group(1)
            want = int(tok) if tok.isdigit() else NUMBERS[tok]
        found[cls] = want
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--draws", type=int, default=24)
    ap.add_argument("--top-n", type=int, default=120, help="candidates rescored per query")
    args = ap.parse_args()

    from src.core.kis_engine import KISEngine

    print("loading index ...", flush=True)
    eng = KISEngine(args.data).load()
    meta_by_key = {(m["video_id"], m["frame_idx"]): m for m in eng.metadata}
    gt = json.loads((Path(args.data) / "ground_truth.json").read_text(encoding="utf-8"))
    gt = [g for g in gt if g.get("video_id") in eng.last_frame]
    print(f"{len(gt)} ground-truth queries\n", flush=True)

    print("retrieving ...", flush=True)
    cached = []
    for g in gt:
        hits = ranked_hits(eng, g["kis_query_vi"], g.get("kis_query_en"), top_n=400)
        cached.append((g, hits[: args.top_n]))

    wanted = set()
    for _g, hits in cached:
        for h in hits:
            m = meta_by_key.get((h.video_id, h.frame_idx))
            if m:
                wanted.add((h.video_id, Path(m["frame_filename"]).stem))
    print(f"reading detections for {len(wanted):,} frames ...", flush=True)
    det = load_detections(Path(args.data), wanted)
    if not det:
        print("no detections found in data/ — nothing to measure")
        return 2
    vocab = {c for counter in det.values() for c in counter}
    print(f"{len(det):,} frames carry detections, {len(vocab)} distinct classes\n")

    def counter_for(h):
        m = meta_by_key.get((h.video_id, h.frame_idx))
        return det.get((h.video_id, Path(m["frame_filename"]).stem)) if m else None

    # how often is there anything to work with at all?
    named = 0
    for g, _h in cached:
        if query_terms(g.get("kis_query_en") or "", vocab):
            named += 1
    print(f"{named}/{len(gt)} queries name at least one detector class\n")

    kf: dict = {}
    for m in eng.metadata:
        kf.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
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

    draws = [draw(4000 + s) for s in range(args.draws)]
    windows = [int(w) for w in args.windows.split(",")]

    def rescore(mode: str, w: float):
        """New candidate order for every query under one weighting."""
        per_query = []
        for g, hits in cached:
            terms = query_terms(g.get("kis_query_en") or "", vocab) if mode != "none" else {}
            if not terms:
                per_query.append(hits)
                continue
            def frame_bonus(h):
                c = counter_for(h) or Counter()
                b = 0.0
                for cls, want in terms.items():
                    have = c.get(cls, 0)
                    if mode in ("match", "both", "video") and have:
                        b += w
                    if mode in ("count", "both") and want is not None and have == want:
                        b += w
                    if mode == "filter" and not have:
                        b -= w
                return b

            if mode == "video":
                # Detections identify the right VIDEO better than the embedding
                # does, but promoting one frame over another inside a video is
                # harmful: the frame with the matching object is not the frame
                # nearest the answer instant. So the bonus is computed per video
                # and applied to all of its frames equally, which reorders
                # videos while leaving each video's internal order untouched.
                by_video = defaultdict(float)
                for h in hits:
                    by_video[h.video_id] = max(by_video[h.video_id], frame_bonus(h))
                order = {}
                for pos, h in enumerate(hits):
                    order.setdefault(h.video_id, pos)
                scored = [
                    (h.score + by_video[h.video_id], order[h.video_id], i, h)
                    for i, h in enumerate(hits)
                ]
                scored.sort(key=lambda t: (-t[0], t[2]))
                per_query.append([t[3] for t in scored])
                continue

            scored = [(h.score + frame_bonus(h), h) for h in hits]
            scored.sort(key=lambda t: -t[0])
            per_query.append([h for _s, h in scored])
        return per_query

    plan = AllocationPlan(breadth_cost=1.0, depth_cost=DEFAULT_DEPTH_COST, step=10)

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
        r1 = sum(
            1
            for (g, _), hits in zip(cached, per_query)
            if hits and hits[0].video_id == g["video_id"]
        )
        return per_w, r1

    base_w, base_r1 = evaluate([h for _g, h in cached])
    base = sum(base_w) / len(base_w)
    print(f"{'variant':22s}" + "".join(f"{'W=' + str(w):>9}" for w in windows) + "     mean  vR@1   delta")
    print("-" * (22 + 9 * len(windows) + 24))
    print(f"{'baseline (no objects)':22s}" + "".join(f"{v:9.3f}" for v in base_w)
          + f"{base:9.3f}  {base_r1:3d}       -")

    best = (base, "baseline")
    for mode in ("match", "count", "filter", "both", "video"):
        for w in (0.005, 0.01, 0.02, 0.05):
            per_w, r1 = evaluate(rescore(mode, w))
            mean = sum(per_w) / len(per_w)
            if mean > best[0]:
                best = (mean, f"{mode} w={w}")
            print(f"{mode + ' w=' + str(w):22s}" + "".join(f"{v:9.3f}" for v in per_w)
                  + f"{mean:9.3f}  {r1:3d}  {100 * (mean / base - 1):+6.1f}%", flush=True)

    print()
    if best[1] == "baseline":
        print("Nothing beats the baseline. Detections stay on screen for the operator")
        print("and OUT of the scoring path — which is where they already are.")
    else:
        print(f"Best: {best[1]} at {best[0]:.3f} ({100 * (best[0] / base - 1):+.1f}%).")
        print("Only worth shipping if the margin is bigger than the run-to-run noise;")
        print("re-run with --draws 48 before changing anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
