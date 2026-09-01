# -*- coding: utf-8 -*-
"""CONG TRUOC SHIP: cau hinh CHOT cua lever hoan vi diem theo sim(canh B) co song
duoc khi DOI MO HINH BOC cua bo cham khong?

Vi sao phai co cong nay. Phan bien 1 da chi ra rang truc sigma cua CoveragePlan
DOI CHIEU khi doi gia dinh boc (DEU trong o keyframe -> sigma lon thang;
GAUSS quanh neo -> sigma nho thang). Neu lever nay cung dao dau theo mo hinh boc
thi no khong phai mot cai tien dinh vi, no chi la mot danh doi be-rong/do-sau
khac duoc doi ten. Neu no giu dau va giu do lon o CA BON mo hinh thi ket luan
"day la lever chon dung O" duoc chung minh doc lap voi thiet bi do.

Bon mo hinh boc, tren dung o keyframe chua khung neo [lo, hi):
  DEU       — boc deu trong o (bo cham HIEN TAI)
  TAM_GIAC  — tam giac, dinh o khung neo
  GAUSS12   — chuan sd=12 frame quanh neo, cat vao o
  SAU_NEO   — deu tren [neo, hi): voi cau HAI canh, khung neo la khung DAU TIEN
              cua canh B, tuc dung mot cu cat; khoanh khac that nam SAU cu cat
              chu khong the nam truoc no. Mo hinh DEU dat nua khoi luong xac suat
              vao canh A — day la mo hinh sai ro nhat cho nhom nay.

KHONG doc TEST cua ai: chay cau hinh DA CHOT tu truoc tren CA 66 cau hai canh,
ho hat giong doc lap 667000. Day khong phai mot lan chon nua.

    python -u scripts/cong_do_ben_mo_hinh_boc.py
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.experiment_cap_thoi_gian import _plan  # noqa: E402
from scripts.experiment_phu_quet_luoi import cham_nhanh, ma_tran_dong  # noqa: E402
from scripts.make_submission import DEFAULT_N_FLAT, allocate_rows  # noqa: E402
from src.core.submission import MAX_ROWS  # noqa: E402

import scripts.do_vlm_noi_video_moi as L  # noqa: E402

GOC = 667000
WINDOWS = [6, 10, 20]
ALPHA, W = 0.5, 1.0


def o_cua_neo(g, kf):
    a = kf[g["video_id"]]
    i = int(np.argmin(np.abs(a - int(g["frame_idx"]))))
    lo = (a[i] + a[i - 1]) // 2 if i > 0 else a[i] - 30
    hi = (a[i] + a[i + 1]) // 2 if i + 1 < len(a) else a[i] + 30
    return int(lo), int(max(lo + 1, hi)), int(a[i])


def boc_kieu(seed, gt_sub, kf, kieu):
    rng = np.random.default_rng(seed)
    out = []
    for g in gt_sub:
        lo, hi, c = o_cua_neo(g, kf)
        if kieu == "DEU":
            x = int(rng.integers(lo, hi))
        elif kieu == "TAM_GIAC":
            x = int(np.clip(int(rng.triangular(lo, min(max(c, lo), hi - 1), hi)), lo, hi - 1))
        elif kieu == "GAUSS12":
            x = int(np.clip(int(c + rng.normal(0, 12)), lo, hi - 1))
        elif kieu == "SAU_NEO":
            a0 = min(max(c, lo), hi - 1)
            x = int(rng.integers(a0, max(a0 + 1, hi)))
        else:
            raise ValueError(kieu)
        out.append(x)
    return out


def cac_lan_boc_kieu(goc, so_ho, so_boc, gt_sub, kf, kieu):
    ho = []
    for s in range(so_ho):
        draws = [boc_kieu(goc + s * 1000 + t, gt_sub, kf, kieu) for t in range(so_boc)]
        ho.append([np.array([d[q] for d in draws]) for q in range(len(gt_sub))])
    return ho


def main() -> int:
    a = argparse.Namespace(
        data=str(ROOT / "data"),
        moi=str(ROOT / "data" / "ground_truth_moi.json"),
        cache=str(ROOT / "data" / "cache_bo_do_moi"),
        canh_b=100,
        allocator="coverage",
        videos=3,
        frames=12,
        windows="6,10,20",
    )
    sach, nhan, bat, cands, kf, _ten, hang_of, kho = L.nap(a)
    print(f"bo SACH {len(sach)} | qua cong HAI canh {len(bat)}")

    rows_nen = [allocate_rows(c, "coverage", DEFAULT_N_FLAT, _plan())[:MAX_ROWS] for c in cands]
    khung_of = {i: L.chon_khung_de_cham(cands[i], 3, 12) for i in bat}

    # ---- tin hieu 0 dong: sim(canh B), chuan min-max trong TUNG video --------
    rows = list(rows_nen)
    for i in bat:
        s = kho.lay(nhan[i]["canh_B_vi"], nhan[i]["canh_B_en"])
        tv = defaultdict(list)
        for c in khung_of[i]:
            tv[c.video_id].append(int(c.frame_idx))
        dB = {}
        for v, fs in tv.items():
            gia = np.array([float(s[hang_of[(v, f)]]) for f in fs])
            lo, hi = float(gia.min()), float(gia.max())
            ch = (gia - lo) / (hi - lo) if hi > lo else np.zeros_like(gia)
            for f, x in zip(fs, ch):
                dB[(v, f)] = float(x)
        loc = {}
        for v, fs in tv.items():
            sub = L.suy_ra_loc({f: dB[(v, f)] for f in fs}, {v: fs}, ALPHA)
            for f, x in sub.items():
                loc[(v, f)] = x
        key = L.khoa_theo_chi_so(cands[i], loc, W)
        if len(key) < 2:
            continue
        rows[i] = allocate_rows(
            L.hoan_vi_diem(cands[i], key), "coverage", DEFAULT_N_FLAT, _plan()
        )[:MAX_ROWS]

    ngoai = [i for i in range(len(sach)) if i not in bat]
    for i in ngoai:
        assert rows[i] == rows_nen[i]
    print(f"bat bien: {len(ngoai)} cau MOT canh ra 100 dong giong het nen (assert OK)\n")

    gt = [sach[i] for i in bat]
    mn = ma_tran_dong([rows_nen[i] for i in bat], gt)
    mc = ma_tran_dong([rows[i] for i in bat], gt)
    rng = np.random.default_rng(2026)

    print(f"{'mo hinh boc':<10}{'nen':>9}{'chot':>9}{'%':>9}   {'KTC 95% theo cau':>24}"
          f"{'P(<=0)':>9}   tot/xau/khong")
    for kieu in ("DEU", "TAM_GIAC", "GAUSS12", "SAU_NEO"):
        ho = cac_lan_boc_kieu(GOC, 6, 48, gt, kf, kieu)

        def per(m, ho=ho):
            r = np.zeros(len(gt))
            for dr in ho:
                for q in range(len(gt)):
                    r[q] += cham_nhanh([m[q]], [dr[q]], WINDOWS)
            return r / len(ho)

        dn, dc = per(mn), per(mc)
        h = dc - dn
        lay = rng.integers(0, len(h), size=(4000, len(h)))
        dd = dc[lay].mean(1) - dn[lay].mean(1)
        print(f"{kieu:<10}{dn.mean():>9.4f}{dc.mean():>9.4f}"
              f"{100*(dc.mean()/dn.mean()-1):>+8.1f}%"
              f"   [{np.percentile(dd,2.5):+.4f}, {np.percentile(dd,97.5):+.4f}]"
              f"{float((dd<=0).mean()):>8.1%}   "
              f"{int((h>1e-12).sum())}/{int((h<-1e-12).sum())}/{int((abs(h)<=1e-12).sum())}")
    print("\nDOC: neu ca bon dong deu duong va KTC tach khoi 0 thi lever nay KHONG phai")
    print("mot danh doi be-rong/do-sau — no la phep chon O, doc lap voi gia dinh boc.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
