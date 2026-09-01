# -*- coding: utf-8 -*-
"""PHAN BIEN: gia dinh MO HINH BOC cua bo cham dang lai ket luan cua ca bon lane.

``experiment_phu_quet_luoi.boc_khoanh_khac`` boc khoanh khac that DEU tren O
KEYFRAME chua khung neo. Do la mot LUA CHON MO HINH, khong phai du lieu — va no
lam rat nhieu viec:

  * O keyframe cua khung neo rong trung vi 72 frame, trong khi cua so cham rong
    nhat chi +-20. Mot dong dat DUNG khung neo, hang 1, chi trung ~34% so lan.
  * Vi the chien luoc toi uu duoi mo hinh nay la RAI RONG, khong phai dat dung.
  * Truc sigma cua CoveragePlan — thu ma lane phan-bo-sau quet ca luoi va lane
    tin-hieu dung lam DOI CHUNG (e)/(f) — DOI CHIEU khi doi mo hinh boc.

Ba phan, khong phan nao goi model, khong phan nao doc TEST cua ai:
  1. be rong o keyframe cua khung neo (dem tat dinh)
  2. tran cua "mot dong dat dung khung neo" va cua thang dong +-X
  3. truc sigma duoi ba mo hinh boc: DEU (hien tai) / TAM GIAC / GAUSS hep

    python -u scripts/phan_bien_mo_hinh_boc.py
"""
from __future__ import annotations
import collections, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._console import safe_console  # noqa: E402
safe_console()
from scripts.experiment_phu_quet_luoi import cham_nhanh, ma_tran_dong  # noqa: E402
from scripts.make_submission import DEFAULT_DEPTH_COST, DEFAULT_N_FLAT  # noqa: E402
from src.core.submission import (  # noqa: E402
    MAX_ROWS, AllocationPlan, Candidate, CoveragePlan, allocate_coverage_rows)

PLAN = AllocationPlan(breadth_cost=1.0, depth_cost=DEFAULT_DEPTH_COST, step=10)
WINDOWS = [6, 10, 20]


def nap():
    d = ROOT / "data"
    moi = json.loads((d / "ground_truth_moi.json").read_text(encoding="utf-8"))
    uv = json.loads((d / "cache_bo_do_moi" / "uv_moi.json").read_text(encoding="utf-8"))
    giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]
    sach = [moi[i] for i in giu]
    cands = [[Candidate(v, f, s, lf) for v, f, s, lf in uv[i]] for i in giu]
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    kfl: dict = {}
    for m in meta:
        kfl.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
    kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in kfl.items()}
    hai = [i for i, g in enumerate(sach) if g.get("co_2_canh")]
    return sach, cands, kf, hai


def o_cua_neo(g, kf):
    a = kf[g["video_id"]]
    i = int(np.argmin(np.abs(a - int(g["frame_idx"]))))
    lo = (a[i] + a[i - 1]) // 2 if i > 0 else a[i] - 30
    hi = (a[i] + a[i + 1]) // 2 if i + 1 < len(a) else a[i] + 30
    return int(lo), int(hi), int(a[i])


def main() -> int:
    sach, cands, kf, hai = nap()
    hs = set(hai)
    print(f"bo SACH {len(sach)} | HAI canh {len(hai)}")

    print("\n" + "=" * 76)
    print("1. BE RONG O KEYFRAME cua khung neo (dem tat dinh)")
    print("=" * 76)
    for co2 in (True, False):
        sub = [g for i, g in enumerate(sach) if (i in hs) == co2]
        W = np.array([o_cua_neo(g, kf)[1] - o_cua_neo(g, kf)[0] for g in sub])
        ten = "HAI canh" if co2 else "MOT canh"
        print(f"  {ten}: trung vi {np.median(W):.0f} frame  p75 {np.percentile(W,75):.0f}"
              f"  p90 {np.percentile(W,90):.0f}  max {W.max():.0f}")
        for th in (40, 60, 100):
            print(f"     so cau co o rong > {th:>3} frame: {(W>th).sum()}/{len(W)}")

    print("\n" + "=" * 76)
    print("2. TRAN cua mot dong dat DUNG khung neo (hang 1), duoi mo hinh boc DEU")
    print("=" * 76)

    def P(g, w, extra=0):
        lo, hi, c = o_cua_neo(g, kf)
        a, b = max(lo, c - w - extra), min(hi, c + w + extra)
        return max(0.0, b - a) / max(hi - lo, 1)

    for co2 in (True, False):
        sub = [g for i, g in enumerate(sach) if (i in hs) == co2]
        ten = "HAI canh" if co2 else "MOT canh"
        r1 = np.mean([np.mean([P(g, w) for w in WINDOWS]) for g in sub])
        print(f"  {ten}: MOT dong dat dung khung neo -> xac suat trung TB = {r1:.3f}")
        for ex in (10, 20, 30, 50, 100):
            r = np.mean([np.mean([P(g, w, ex) for w in WINDOWS]) for g in sub])
            print(f"     thang dong phu +-{ex:>3} frame quanh neo -> {r:.3f}")

    print("\n" + "=" * 76)
    print("3. TRUC SIGMA duoi ba MO HINH BOC khac nhau (ca 132 muc)")
    print("=" * 76)

    def boc(goc, kieu, nlan=288):
        rng = np.random.default_rng(goc)
        out = []
        for g in sach:
            lo, hi, c = o_cua_neo(g, kf)
            hi = max(hi, lo + 1)
            if kieu == "deu":
                t = rng.integers(lo, hi, size=nlan)
            elif kieu == "tam_giac":
                t = np.clip(rng.triangular(lo, min(max(c, lo), hi), hi, size=nlan)
                            .astype(int), lo, hi - 1)
            else:
                t = np.clip((c + rng.normal(0, 12, size=nlan)).astype(int), lo, hi - 1)
            out.append(t)
        return out

    def sinh(sig, nhiet=0.02):
        return [allocate_coverage_rows(
            c, plan=CoveragePlan(nhiet=nhiet, sigma=sig, nua_cua_so=6, luoi=5,
                                 budget=MAX_ROWS),
            tail_n_flat=DEFAULT_N_FLAT, tail_plan=PLAN)[:MAX_ROWS] for c in cands]

    sigmas = (15.0, 30.0, 45.0, 60.0)
    rows_of = {s: sinh(s) for s in sigmas}
    print(f"  {'mo hinh boc':<14}" + "".join(f"{'sig=' + str(int(s)):>10}" for s in sigmas)
          + "   huong tot nhat")
    for kieu, mo in (("deu", "DEU trong o (bo cham HIEN TAI)"),
                     ("tam_giac", "TAM GIAC, dinh o khung neo"),
                     ("gauss12", "GAUSS sd=12 frame quanh neo")):
        dr = boc(999000, kieu)
        tong, hai_l = [], []
        for s in sigmas:
            mats = ma_tran_dong(rows_of[s], sach)
            d = np.array([cham_nhanh([mats[q]], [dr[q]], WINDOWS) for q in range(len(sach))])
            tong.append(d.mean())
            hai_l.append(d[hai].mean())
        print(f"  {kieu:<14}" + "".join(f"{x:>10.4f}" for x in tong)
              + f"   sigma={int(sigmas[int(np.argmax(tong))])}   [{mo}]")
        print(f"  {'  HAI canh':<14}" + "".join(f"{x:>10.4f}" for x in hai_l)
              + f"   sigma={int(sigmas[int(np.argmax(hai_l))])}")
    print("\n  DOC: 'sigma lon hon tot hon' DOI CHIEU khi doi mo hinh boc. No la ket qua")
    print("  cua GIA DINH boc deu tren o keyframe, khong phai mot tinh chat cua bo cau hoi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
