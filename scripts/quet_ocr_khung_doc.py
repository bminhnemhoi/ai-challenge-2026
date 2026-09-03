"""OCR dich danh cac khung ma duong tra loi Q&A THAT SU doc (top diem +-1)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._console import safe_console
safe_console()
from src.core.submission import Candidate

moi = json.loads((ROOT / "data" / "ground_truth_moi.json").read_text(encoding="utf-8"))
uv = json.loads((ROOT / "data" / "cache_bo_do_moi" / "uv_moi.json").read_text(encoding="utf-8"))
giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]
meta = json.loads((ROOT / "data" / "metadata.json").read_text(encoding="utf-8"))
ten_of = {(m["video_id"], int(m["frame_idx"])): m["frame_filename"] for m in meta}
kf = {}
for m in meta:
    kf.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
kf = {v: np.array(sorted(a)) for v, a in kf.items()}
del meta

dich = set()
for i in giu:
    cands = sorted([Candidate(v, f, s, lf) for v, f, s, lf in uv[i]],
                   key=lambda c: -float(c.score))[:4]  # neo=2 + du phong
    for c in cands:
        a = kf[c.video_id]
        j = int(np.argmin(np.abs(a - int(c.frame_idx))))
        for x in range(max(0, j - 1), min(len(a), j + 2)):
            dich.add((c.video_id, int(a[x])))

from src.core.ocr import OCRIndex
idx = OCRIndex(str(ROOT / "data"), langs=["vi", "en"])
thieu = [(v, f, ten_of[(v, f)]) for v, f in sorted(dich)
         if (v, f) in ten_of and str(f) not in idx._video(v)]
print(f"khung doc: {len(dich)} | can OCR them: {len(thieu)}", flush=True)
idx.read_frames(thieu)
print("XONG")
