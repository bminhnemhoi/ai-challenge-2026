"""Dem tat dinh: nang pool 400 -> 800 co them bao nhieu cau co dap an trong pool?

Tang SINH ung vien = 17% tran; 41% cau hai canh khong co ung vien nao <=20
frame quanh dap an trong pool 400. Phep dem nay 0 API (dung engine + sims),
tat dinh, khong doc diem.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._console import safe_console
safe_console()
from scripts.do_cap_thoi_gian_moi import canh_cua
from scripts.make_submission import ranked_hits

moi = json.loads((ROOT / "data" / "ground_truth_moi.json").read_text(encoding="utf-8"))
giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]

from src.core.kis_engine import KISEngine
eng = KISEngine(str(ROOT / "data")).load()

ket = {False: {400: 0, 800: 0, "n": 0}, True: {400: 0, 800: 0, "n": 0}}
for k, i in enumerate(giu):
    g = moi[i]
    hits = ranked_hits(eng, g["kis_query_vi"], None, top_n=800)
    dap, vid = int(g["frame_idx"]), g["video_id"]
    co400 = any(h.video_id == vid and abs(int(h.frame_idx) - dap) <= 20 for h in hits[:400])
    co800 = any(h.video_id == vid and abs(int(h.frame_idx) - dap) <= 20 for h in hits)
    nh = bool(canh_cua(g))
    ket[nh]["n"] += 1
    ket[nh][400] += co400
    ket[nh][800] += co800
    if (k + 1) % 40 == 0:
        print(f"  {k+1}/{len(giu)}", flush=True)

for nh, ten in ((False, "MOT canh"), (True, "HAI canh")):
    d = ket[nh]
    print(f"{ten}: n={d['n']} | dap an trong pool<=20: 400={d[400]} ({100*d[400]/d['n']:.0f}%) "
          f"-> 800={d[800]} ({100*d[800]/d['n']:.0f}%)  (+{d[800]-d[400]})")
print("\nLuu y: canh-B augmentation (da ship) con them ung vien NGOAI pool nay"
      " cho cau hai canh — so nay la pool GOC.")
