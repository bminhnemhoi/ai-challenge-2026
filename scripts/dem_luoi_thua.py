"""R2 — chan doan luoi keyframe thua (0 dong, tat dinh).

Tren cau TUNE dang TRUOT (video dung co dong nhung khong dong nao trong ±20
o 100 dong): khoang cach thoi gian tu moc GT den keyframe GAN NHAT trong chi
muc (fps tinh tu chinh muc: frame_idx/pts_time). Nguong dang ky truoc HAI PHIA:
>=30% cau truot co GT cach keyframe >2s => luoi thua la nut nghen that;
<10% => bo ca nhanh lam min cuc bo.
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
from scripts.experiment_cap_thoi_gian import _plan
from scripts.make_submission import DEFAULT_N_FLAT, allocate_rows
from src.core.submission import MAX_ROWS, Candidate

moi = json.loads((ROOT / "data" / "ground_truth_moi.json").read_text(encoding="utf-8"))
uv = json.loads((ROOT / "data" / "cache_bo_do_moi" / "uv_moi.json").read_text(encoding="utf-8"))
giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]
meta = json.loads((ROOT / "data" / "metadata.json").read_text(encoding="utf-8"))
kf = {}
for m in meta:
    kf.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
kf = {v: np.array(sorted(a)) for v, a in kf.items()}
del meta

# cung phep chia TUNE (seed 20260903, phan tang theo canh) nhu cac phep dem R
ket = []
for i in giu:
    g = moi[i]
    cands = [Candidate(v, f, s, lf) for v, f, s, lf in uv[i]]
    rows = allocate_rows(cands, "coverage", DEFAULT_N_FLAT, _plan())[:MAX_ROWS]
    d = int(g["frame_idx"])
    hit = any(v == g["video_id"] and abs(int(f) - d) <= 20 for v, f in rows)
    vd = any(v == g["video_id"] for v, _f in rows)
    fps = d / float(g["pts_time"]) if g.get("pts_time") else 25.0
    a = kf.get(g["video_id"])
    lech_f = int(np.min(np.abs(a - d))) if a is not None and len(a) else 10**9
    ket.append({"hai": bool(canh_cua(g)), "truot": (not hit) and vd,
                "lech_s": lech_f / max(1.0, fps)})

rng = np.random.default_rng(20260903)
mot = [q for q, k in enumerate(ket) if not k["hai"]]
hai = [q for q, k in enumerate(ket) if k["hai"]]
tune = set()
for nhom in (mot, hai):
    x = rng.permutation(len(nhom))
    tune |= {nhom[j] for j in x[: len(x) // 2]}

for ten, chon in (("MOT canh", mot), ("HAI canh (phu)", hai)):
    tr = [ket[q] for q in chon if q in tune and ket[q]["truot"]]
    if not tr:
        print(f"{ten}: 0 cau truot trong TUNE"); continue
    xa = sum(1 for k in tr if k["lech_s"] > 2.0)
    ls = sorted(k["lech_s"] for k in tr)
    print(f"{ten}: {len(tr)} cau TUNE truot | GT cach keyframe gan nhat >2s: "
          f"{xa}/{len(tr)} = {100*xa/len(tr):.0f}% | trung vi {ls[len(ls)//2]:.2f}s"
          f" | p90 {ls[int(len(ls)*0.9)]:.2f}s")
print("\nNguong dang ky truoc (MOT canh): >=30% lam; <10% bo ca nhanh.")
