"""Does the BTC-provided video metadata add anything on the official metric?

`media-info-aic25-b1.zip` carries the YouTube title, description, keywords and
channel for all 873 videos. The shipped retriever fuses a bounded BM25-ish
signal from it; this script measures whether that is worth anything once the
score is computed the way the organisers compute it.

The question matters because the signal is video-level while the metric is
frame-level: a metadata boost can only reorder VIDEOS, and getting the video
right is necessary but far from sufficient.

    python scripts/experiment_metadata.py
"""

from __future__ import annotations

import json
import math
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
STOP = {
    "và", "là", "có", "trong", "đang", "của", "cho", "với", "những", "các", "trên",
    "tại", "một", "ở", "ra", "vào", "được", "bị", "khi", "để", "màu", "cảnh", "video",
    "đoạn", "clip", "người", "tìm", "này", "cần", "the", "a", "of", "in", "on", "is",
}


def load_media_info(data: Path):
    """video_id -> searchable text blob from the BTC media-info package."""
    z = data / "media-info-aic25-b1.zip"
    if not z.exists():
        return {}
    out = {}
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            if not n.endswith(".json"):
                continue
            vid = Path(n).stem
            try:
                j = json.loads(zf.read(n).decode("utf-8"))
            except Exception:
                continue
            kw = j.get("keywords") or []
            out[vid] = " ".join(
                [
                    str(j.get("title", "")),
                    " ".join(kw) if isinstance(kw, list) else str(kw),
                    str(j.get("description", ""))[:1200],
                ]
            ).lower()
    return out


def tokenize(s: str):
    return [w for w in re.split(r"[^0-9a-zà-ỹ]+", s.lower()) if len(w) >= 2 and w not in STOP]


def build_idf(blobs):
    df = Counter()
    for text in blobs.values():
        for w in set(tokenize(text)):
            df[w] += 1
    n = max(len(blobs), 1)
    return {w: math.log(1 + n / (1 + c)) for w, c in df.items()}


def meta_scores(query: str, blobs, idf):
    """Bounded per-video lexical match, in [0, 1]."""
    q = set(tokenize(query))
    if not q:
        return {}
    out = {}
    for vid, text in blobs.items():
        toks = set(tokenize(text))
        hit = sum(idf.get(w, 0.0) for w in q & toks)
        out[vid] = math.tanh(hit * 0.15)
    return out


def main() -> int:
    data = ROOT / "data"
    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    gt = json.loads((data / "ground_truth.json").read_text(encoding="utf-8"))

    blobs = load_media_info(data)
    if not blobs:
        print("media-info-aic25-b1.zip not found in data/ — nothing to measure")
        return 2
    print(f"media-info: {len(blobs)} videos")
    idf = build_idf(blobs)

    from src.core.kis_engine import KISEngine

    print("loading index ...", flush=True)
    eng = KISEngine(data).load()

    by_video = defaultdict(list)
    for i, m in enumerate(meta):
        by_video[m["video_id"]].append(i)

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

    # cache the visual similarities once; only the metadata weight varies
    print("scoring queries ...", flush=True)
    sims_cache = [eng.query_similarities(g["kis_query_vi"], g["kis_query_en"]) for g in gt]
    meta_cache = [meta_scores(g["kis_query_vi"] + " " + g["kis_query_en"], blobs, idf) for g in gt]
    vid_arr = eng.video_id
    plan = AllocationPlan(depth_cost=0.5)

    print("\n" + "=" * 90)
    print("metadata weight  ->  OFFICIAL Final Score        (visual score is z-scored per query)")
    print("=" * 90)
    print(f"  {'weight':>8} {'R@1':>8}   " + "  ".join(f"W={w:<4d}" for w in WINDOWS) + "      mean")
    print("  " + "-" * 84)

    for weight in (0.0, 0.05, 0.1, 0.2, 0.4):
        tot = {w: 0.0 for w in WINDOWS}
        r1 = 0
        for qi, g in enumerate(gt):
            s = sims_cache[qi].astype(np.float32).copy()
            s = (s - s.mean()) / (s.std() + 1e-6)  # comparable scale for the mix
            if weight:
                boost = np.zeros_like(s)
                ms = meta_cache[qi]
                for vid, val in ms.items():
                    if val > 0:
                        boost[by_video[vid]] = val
                s = s + weight * boost
            s = np.where(eng.valid, s, -np.inf)
            top = np.argsort(-s)[:200]
            if str(vid_arr[int(top[0])]) == g["video_id"]:
                r1 += 1
            cands = [
                Candidate(str(vid_arr[i]), int(eng.frame_idx[i]), float(s[i]),
                          eng.last_frame[str(vid_arr[i])])
                for i in top
            ]
            rows = allocate_hybrid_rows(cands, n_flat=30, plan=plan)[:MAX_ROWS]
            for w in WINDOWS:
                for sp in spans[w][qi]:
                    tot[w] += final_score([r_score_kis(v, f, g["video_id"], sp) for v, f in rows])
        n = len(gt) * N_DRAWS
        vals = [tot[w] / n for w in WINDOWS]
        tag = "  [visual only]" if weight == 0 else ""
        print(
            f"  {weight:8.2f} {r1 / len(gt):7.1%}   "
            + "  ".join(f"{v:6.3f}" for v in vals)
            + f"   {np.mean(vals):6.3f}{tag}"
        )

    print("\nThe metadata signal is video-level, so it can only reorder videos —")
    print("it cannot move the right FRAME up, which is what the official metric needs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
