"""Bac DIEM cho pool 800: dung lai uv-800, cham y san xuat, TUNE chon TEST doc mot lan."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._console import safe_console
safe_console()
from scripts.do_bo_do_moi import GOC_HAT
from scripts.do_cap_thoi_gian_moi import canh_cua
from scripts.experiment_cap_thoi_gian import _plan
from scripts.experiment_phu_quet_luoi import cac_lan_boc, cham_nhanh, ma_tran_dong
from scripts.make_submission import DEFAULT_N_FLAT, allocate_rows, ranked_hits
from src.core.submission import MAX_ROWS, Candidate

moi = json.loads((ROOT / "data" / "ground_truth_moi.json").read_text(encoding="utf-8"))
uv400 = json.loads((ROOT / "data" / "cache_bo_do_moi" / "uv_moi.json").read_text(encoding="utf-8"))
giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]
gt_sub = [moi[i] for i in giu]

cache8 = ROOT / "data" / "cache_bo_do_moi" / "uv800.json"
if cache8.exists():
    uv800 = json.loads(cache8.read_text(encoding="utf-8"))
else:
    from src.core.kis_engine import KISEngine
    eng = KISEngine(str(ROOT / "data")).load()
    uv800 = {}
    for k, i in enumerate(giu):
        hits = ranked_hits(eng, moi[i]["kis_query_vi"],
                           moi[i].get("kis_query_en"), top_n=800)
        uv800[str(i)] = [[h.video_id, int(h.frame_idx), float(h.score),
                          int(h.video_last_frame)] for h in hits]
        if (k + 1) % 40 == 0:
            print(f"  truy xuat 800: {k+1}/{len(giu)}", flush=True)
    cache8.write_text(json.dumps(uv800), encoding="utf-8")

meta = json.loads((ROOT / "data" / "metadata.json").read_text(encoding="utf-8"))
kf = {}
for m in meta:
    kf.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in kf.items()}
del meta

def dong(uv_map, tu_moi):
    ra = []
    for i in giu:
        q = uv_map[str(i)] if tu_moi else uv_map[i]
        cands = [Candidate(v, f, s, lf) for v, f, s, lf in q]
        ra.append(allocate_rows(cands, "coverage", DEFAULT_N_FLAT, _plan())[:MAX_ROWS])
    return ra

rows4, rows8 = dong(uv400, False), dong(uv800, True)
ho = cac_lan_boc(GOC_HAT, 4, 48, gt_sub, kf)

def diem_cau(rows_of):
    mats = ma_tran_dong(rows_of, gt_sub)
    d = np.zeros(len(gt_sub))
    for qi in range(len(gt_sub)):
        d[qi] = float(np.mean([cham_nhanh([mats[qi]], [b[qi]], [6, 10, 20]) for b in ho]))
    return d

d4, d8 = diem_cau(rows4), diem_cau(rows8)
rng = np.random.default_rng(20260903)
nhom = {}
for q in range(len(gt_sub)):
    nhom.setdefault(bool(canh_cua(gt_sub[q])), []).append(q)
tune = set()
for _c, idxs in sorted(nhom.items()):
    x = rng.permutation(len(idxs)); tune |= {idxs[j] for j in x[:len(x)//2]}
test = [q for q in range(len(gt_sub)) if q not in tune]

for ten, cs in (("TUNE", sorted(tune)), ("TEST (doc MOT lan)", test)):
    a, b = d4[cs], d8[cs]
    ch = b.mean() - a.mean()
    r2 = np.random.default_rng(4242)
    lay = r2.integers(0, len(cs), size=(4000, len(cs)))
    dd = b[lay].mean(axis=1) - a[lay].mean(axis=1)
    lo, hi = np.percentile(dd, [2.5, 97.5])
    print(f"{ten}: 400={a.mean():.4f} -> 800={b.mean():.4f} ({ch:+.4f}, "
          f"{100*ch/a.mean():+.1f}%) KTC [{lo:+.4f},{hi:+.4f}] P(<=0)={(dd<=0).mean():.1%}")
    if ten.startswith("TUNE") and ch <= 0:
        print("AM/hoa tren TUNE — DUNG, khong doc TEST."); sys.exit(0)
