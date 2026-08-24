"""Do the BTC object detections help, on the official metric?

`objects-aic25-b1.zip` holds Faster R-CNN (OpenImages V4) detections for every
keyframe: class labels, boxes and scores.  Unlike the video metadata, this is a
FRAME-level signal, so unlike metadata it *can* move the right frame up — which
is what the official score actually rewards.

The specific hope is counting.  SigLIP is weak at "three cyclists", "two women",
"four children", and several round-1 queries turn on exactly that.  Object
counts are exact.

Two things are measured, honestly:
  1. whether an object-count agreement bonus improves the official score
     overall on the 60-sample ground truth, and
  2. whether it improves it on the SUBSET of queries that mention a count,
     which is the only place the mechanism claims to help.

    python scripts/experiment_objects.py
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
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
CONF = 0.4  # the confidence cut the BTC baseline notebook itself uses

# Vietnamese / English number words that signal a count in a query
NUMS = {
    "một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5, "sáu": 6,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
}
# query noun -> OpenImages class names that would satisfy it
NOUN2CLASS = {
    "người": {"Person", "Man", "Woman", "Boy", "Girl"},
    "phụ nữ": {"Woman", "Person"}, "đàn ông": {"Man", "Person"},
    "em nhỏ": {"Boy", "Girl", "Person"}, "cô bé": {"Girl", "Person"},
    "tay đua": {"Person", "Bicycle"}, "phi hành gia": {"Person"},
    "chim": {"Bird"}, "dê": {"Goat", "Sheep"}, "hổ": {"Tiger"}, "mèo": {"Cat"},
    "xe đạp": {"Bicycle"}, "ô tô": {"Car"}, "xe tải": {"Truck"},
}


def load_objects(data: Path, wanted_keys):
    """(video_id, frame_filename_stem) -> Counter of detected classes."""
    z = data / "objects-aic25-b1.zip"
    if not z.exists():
        return None
    out = {}
    with zipfile.ZipFile(z) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        print(f"objects package: {len(names):,} json files", flush=True)
        for i, n in enumerate(names):
            parts = Path(n).parts
            if len(parts) < 2:
                continue
            key = (parts[-2], Path(n).stem)
            if key not in wanted_keys:
                continue
            try:
                j = json.loads(zf.read(n))
            except Exception:
                continue
            c = Counter()
            for cls, score in zip(
                j.get("detection_class_entities", []), j.get("detection_scores", [])
            ):
                if float(score) > CONF:
                    c[cls] += 1
            out[key] = c
            if i % 40000 == 0:
                print(f"  ...{i:,}", flush=True)
    return out


def query_demands(text: str):
    """[(count, {classes})] the query explicitly asks for."""
    low = text.lower()
    out = []
    for noun, classes in NOUN2CLASS.items():
        for m in re.finditer(rf"(\d+|{'|'.join(NUMS)})\s+{re.escape(noun)}", low):
            tok = m.group(1)
            n = int(tok) if tok.isdigit() else NUMS.get(tok, 0)
            if 1 <= n <= 8:
                out.append((n, classes))
    return out


def main() -> int:
    data = ROOT / "data"
    if not (data / "objects-aic25-b1.zip").exists():
        print("data/objects-aic25-b1.zip not present — download it first")
        return 2

    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    gt = json.loads((data / "ground_truth.json").read_text(encoding="utf-8"))

    from src.core.kis_engine import KISEngine

    print("loading index ...", flush=True)
    eng = KISEngine(data).load()

    by_video = defaultdict(list)
    for i, m in enumerate(meta):
        by_video[m["video_id"]].append(i)

    demands = [query_demands(g["kis_query_vi"] + " " + g["kis_query_en"]) for g in gt]
    counted = [i for i, d in enumerate(demands) if d]
    print(f"{len(counted)}/{len(gt)} ground-truth queries mention a count")
    if not counted:
        print("\nThe ground-truth set has no counting queries, so this mechanism")
        print("cannot be validated here. Shipping it would be unmeasured.")

    print("scoring queries ...", flush=True)
    sims = [eng.query_similarities(g["kis_query_vi"], g["kis_query_en"]) for g in gt]

    # only the candidate frames need object data; loading all 177k is wasteful
    key_of = {i: (m["video_id"], Path(m["frame_filename"]).stem) for i, m in enumerate(meta)}
    wanted = set()
    tops = []
    for qi in range(len(gt)):
        s = np.where(eng.valid, sims[qi], -np.inf)
        t = np.argsort(-s)[:200]
        tops.append(t)
        wanted.update(key_of[int(i)] for i in t)
    print(f"need object data for {len(wanted):,} candidate keyframes", flush=True)
    objs = load_objects(data, wanted)
    if not objs:
        return 2
    print(f"loaded detections for {len(objs):,} of them\n")

    rng = np.random.default_rng(0)
    spans = {w: [] for w in WINDOWS}
    for g in gt:
        fr = np.sort(eng.frame_idx[by_video[g["video_id"]]])
        d = np.diff(fr)
        half = max(1, (int(np.median(d)) if len(d) else 50) // 2)
        offs = rng.integers(-half, half + 1, size=N_DRAWS)
        for w in WINDOWS:
            spans[w].append(
                [(max(0, g["frame_idx"] + int(o) - w // 2),
                  max(0, g["frame_idx"] + int(o) - w // 2) + w) for o in offs]
            )

    plan = AllocationPlan(depth_cost=0.5)

    def evaluate(bonus, subset=None):
        idxs = subset if subset is not None else range(len(gt))
        tot = {w: 0.0 for w in WINDOWS}
        r1 = 0
        for qi in idxs:
            g = gt[qi]
            t = tops[qi]
            s = sims[qi][t].astype(np.float64).copy()
            s = (s - s.mean()) / (s.std() + 1e-6)
            if bonus and demands[qi]:
                for j, fi in enumerate(t):
                    c = objs.get(key_of[int(fi)])
                    if not c:
                        continue
                    ok = 0
                    for want_n, classes in demands[qi]:
                        got = sum(c.get(cl, 0) for cl in classes)
                        if got >= want_n:
                            ok += 1
                    if ok:
                        s[j] += bonus * (ok / len(demands[qi]))
            order = np.argsort(-s)
            t = t[order]
            if str(eng.video_id[int(t[0])]) == g["video_id"]:
                r1 += 1
            cands = [
                Candidate(str(eng.video_id[i]), int(eng.frame_idx[i]), 0.0,
                          eng.last_frame[str(eng.video_id[i])])
                for i in t
            ]
            rows = allocate_hybrid_rows(cands, n_flat=30, plan=plan)[:MAX_ROWS]
            for w in WINDOWS:
                for sp in spans[w][qi]:
                    tot[w] += final_score([r_score_kis(v, f, g["video_id"], sp) for v, f in rows])
        n = len(list(idxs)) * N_DRAWS
        return r1 / len(list(idxs)), [tot[w] / n for w in WINDOWS]

    for label, subset in (("ALL 60 queries", None), (f"the {len(counted)} counting queries", counted)):
        if subset is not None and not subset:
            continue
        print("=" * 88)
        print(label)
        print("=" * 88)
        print(f"  {'object bonus':>13} {'R@1':>8}   " + "  ".join(f"W={w:<4d}" for w in WINDOWS) + "      mean")
        for bonus in (0.0, 0.25, 0.5, 1.0):
            r1, vals = evaluate(bonus, subset)
            tag = "  [visual only]" if bonus == 0 else ""
            print(
                f"  {bonus:13.2f} {r1:7.1%}   "
                + "  ".join(f"{v:6.3f}" for v in vals)
                + f"   {np.mean(vals):6.3f}{tag}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
