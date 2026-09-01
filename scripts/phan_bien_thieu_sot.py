"""PHAN BIEN: do nhung thu ca bon lane deu bo lo.

Bon lane vua roi (phan-bo-sau, tin-hieu-noi-video, paper-2026, vlm-noi-video)
deu do CUNG MOT THU: diem KIS cua 100 dong, tren cung mot bo 132 muc, voi cung
mot gia dinh la "so dong danh cho moi video da co dinh". Script nay do bon thu
NAM NGOAI vong tron do:

  §1  TRAN THEO NGAN SACH DONG (goi y a) — tran oracle +126% giu nguyen SO DONG
      moi video. Neu duoc doi ca ngan sach thi tran con cao bao nhieu? Do bang
      hai bien the: (i) ORACLE-VIDEO (chi giu ung vien cua video dung, chay
      allocator that) va (ii) hoan vi thuan tuy (giu nguyen TAP dong, chi doi
      thu tu) — hoan vi khong the doi R@100 nen no la lever RUI RO CHAN CUNG.

  §2  BE RONG-TRUOC (breadth-first) — mot chien luoc THAT (khong oracle), chi la
      mot hoan vi cua 100 dong hien co: dua dong TOT NHAT cua B video dau len
      hang 1..B. Do cho B = 1 (nen), 2, 3, 5, 8, 12, 20.

  §3  KENH Q&A — 16/59 cau de that (27%; rieng vong 2 la 9/25 = 36%) la Q&A.
      Diem Q&A = diem dinh vi NHAN voi 1[dap an dung]. Dap an sinh tu
      ``hits[:5]`` cua merged_hits bang BIEU QUYET DA SO. Dem tat dinh: trong 5
      khung do, bao nhieu khung nam dung khoanh khac phai tra loi?

  §4  DO THUA KEYFRAME — bo cu vs bo moi. Tham so ``sigma`` cua allocator la mot
      hang so toan cuc, trong khi khe keyframe thay doi tu 1s toi 8s. Kiem xem
      hai bo do co phan bo khe khac nhau khong (neu co, quet sigma toan cuc tren
      hai bo PHAI cho ket qua nguoc chieu — dung nhu lane phan-bo-sau da thay).

Khong sua bat ky file san xuat nao. Khong nap KISEngine (dung metadata.json).

    python -u scripts/phan_bien_thieu_sot.py
    python -u scripts/phan_bien_thieu_sot.py --phan 1,2
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

from scripts.experiment_phu_quet_luoi import (  # noqa: E402
    BUCKET,
    cac_lan_boc,
    cham_nhanh,
    ma_tran_dong,
)
from scripts.make_submission import (  # noqa: E402
    DEFAULT_DEPTH_COST,
    DEFAULT_N_FLAT,
    allocate_rows,
)
from src.core.submission import MAX_ROWS, AllocationPlan, Candidate  # noqa: E402

GOC_HAT = 91000  # goc rieng cua lane phan bien, tach khoi 77000/81000/310000/...
CUA_SO = [6, 10, 20]
SO_HO = 4
SO_BOC = 48

PLAN = AllocationPlan(breadth_cost=1.0, depth_cost=DEFAULT_DEPTH_COST, step=10)


# --------------------------------------------------------------------- nap


def nap():
    gt_all = json.loads((ROOT / "data" / "ground_truth_moi.json").read_text(encoding="utf-8"))
    uv_all = json.loads((ROOT / "data" / "cache_bo_do_moi" / "uv_moi.json").read_text(encoding="utf-8"))
    assert len(gt_all) == len(uv_all), (len(gt_all), len(uv_all))

    md = json.loads((ROOT / "data" / "metadata.json").read_text(encoding="utf-8"))
    kf: dict = {}
    for m in md:
        kf.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
    kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in kf.items()}

    gt, uv = [], []
    for g, q in zip(gt_all, uv_all):
        if str(g.get("lan_truc")) == "True":      # bo shard c (lan truc)
            continue
        if g["video_id"] not in kf:
            continue
        gt.append({
            "video_id": g["video_id"],
            "frame_idx": int(g["frame_idx"]),
            "hai_canh": str(g.get("co_2_canh")) == "True",
            "vqa_answer": g.get("vqa_answer", ""),
            "vqa_question": g.get("vqa_question", ""),
        })
        uv.append([Candidate(v, int(f), float(s), int(lf)) for v, f, s, lf in q])
    return gt, uv, kf, md


def nhom(gt):
    hai = np.array([g["hai_canh"] for g in gt])
    return {"TAT CA": np.ones(len(gt), bool), "MOT canh": ~hai, "HAI canh": hai}


def cham(rows_of, gt, ho, chi_so=None):
    """Diem trung binh tren tap con ``chi_so`` (mac dinh: tat ca)."""
    if chi_so is None:
        chi_so = np.arange(len(gt))
    chi_so = np.asarray(chi_so)
    rs = [rows_of[i] for i in chi_so]
    gs = [gt[i] for i in chi_so]
    mats = ma_tran_dong(rs, gs)
    return float(np.mean([cham_nhanh(mats, [d[i] for i in chi_so], CUA_SO) for d in ho]))


def bang(ten, rows_of, gt, ho, nen=None):
    ns = nhom(gt)
    ra = {}
    for k, mask in ns.items():
        ra[k] = cham(rows_of, gt, ho, np.flatnonzero(mask))
    s = f"  {ten:<34}"
    for k in ("TAT CA", "MOT canh", "HAI canh"):
        if nen is None:
            s += f"  {k}={ra[k]:.4f}"
        else:
            d = 100.0 * (ra[k] / nen[k] - 1.0) if nen[k] > 0 else float("nan")
            s += f"  {k}={ra[k]:.4f} ({d:+.1f}%)"
    print(s, flush=True)
    return ra


# --------------------------------------------------- §1 tran theo ngan sach


def phan_1(gt, uv, kf, ho, nen_rows, nen):
    print("\n" + "=" * 100)
    print("§1  TRAN THEO NGAN SACH DONG — cai ma tran oracle +126% da DONG BANG")
    print("=" * 100)
    print("  Tran oracle cu: giu nguyen VIDEO va THU HANG tung dong, chi dat lai frame id.")
    print("  Tuc no khoa cung 'video dung duoc bao nhieu dong' va 'chung nam o hang nao'.")
    print("  Ba phep duoi day mo lan luot tung rang buoc do.\n")

    # (1a) hoan vi thuan tuy: dua MOI dong dung len hang 1. Khong doi TAP dong,
    #      nen R@100 bat bien -> day dung bang R@100 cua nen.
    hv = []
    for rows, g in zip(nen_rows, gt):
        hv.append(list(rows))
    mats = ma_tran_dong(nen_rows, gt)
    ns = nhom(gt)
    r100 = {}
    for k, mask in ns.items():
        idx = np.flatnonzero(mask)
        tot = 0.0
        for d in ho:
            per = 0.0
            for i in idx:
                f, m = mats[i]
                t = np.asarray(d[i], dtype=np.int64)
                dd = np.abs(f[None, :] - t[:, None])
                acc = 0.0
                for half in CUA_SO:
                    acc += (m[None, :] & (dd <= half)).any(axis=1).mean()
                per += acc / len(CUA_SO)
            tot += per / len(idx)
        r100[k] = tot / len(ho)
    print(f"  (1a) HOAN VI oracle (dua dong dung ve hang 1; TAP dong khong doi)")
    for k in ("TAT CA", "MOT canh", "HAI canh"):
        print(f"       {k:<9} {nen[k]:.4f} -> {r100[k]:.4f}  = {100*(r100[k]/nen[k]-1):+.1f}%"
              f"   [= R@100 cua chinh nen]")

    # (1b) ORACLE-VIDEO: chi giu ung vien cua video dung, chay allocator THAT.
    #      Ngan sach 100 dong duoc phan bo lai HOAN TOAN cho mot video.
    ov = []
    for cands, g in zip(uv, gt):
        con = [c for c in cands if c.video_id == g["video_id"]]
        ov.append(allocate_rows(con, "coverage", DEFAULT_N_FLAT, PLAN)[:MAX_ROWS] if con else [])
    print(f"\n  (1b) ORACLE-VIDEO (chi ung vien cua video dung; allocator that, 100 dong)")
    b = bang("oracle-video", ov, gt, ho, nen)

    co_video = sum(1 for c, g in zip(uv, gt) if any(x.video_id == g["video_id"] for x in c))
    print(f"       video dung co mat trong 400 ung vien: {co_video}/{len(gt)}")
    print(f"       (tran tuyet doi neu chon dung video VA dat dung dong = {co_video/len(gt):.4f})")

    # (1c) so dong that su danh cho video dung, so voi so dong CAN de phu
    n_dong = []
    for rows, g in zip(nen_rows, gt):
        n_dong.append(sum(1 for v, _ in rows if v == g["video_id"]))
    n_dong = np.array(n_dong)
    hai = np.array([g["hai_canh"] for g in gt])
    print(f"\n  (1c) so dong roi vao video DUNG (trung vi): tat ca {np.median(n_dong):.0f}"
          f" | MOT {np.median(n_dong[~hai]):.0f} | HAI {np.median(n_dong[hai]):.0f}")
    print(f"       so dong roi vao video dung o cau TRUOT hoan toan: "
          f"{np.median([n for n, r in zip(n_dong, nen_rows) if n > 0]):.0f} (khi > 0)")
    return b, r100


# ---------------------------------------------------- §2 be rong-truoc


def hoan_vi_be_rong(rows, B):
    """Dua dong DAU TIEN cua B video khac nhau len hang 1..B, giu nguyen phan con lai.

    Day la mot HOAN VI: tap 100 dong khong doi mot dong nao, nen R@100 bat bien.
    """
    if B <= 1:
        return list(rows)
    dau, con, thay = [], [], set()
    for r in rows:
        if r[0] not in thay and len(dau) < B:
            thay.add(r[0])
            dau.append(r)
        else:
            con.append(r)
    return dau + con


def hv_gom_video(rows, cands):
    """Gom moi video thanh mot khoi lien, video theo thu tu xuat hien dau tien."""
    thu_tu, khoi = [], {}
    for r in rows:
        if r[0] not in khoi:
            khoi[r[0]] = []
            thu_tu.append(r[0])
        khoi[r[0]].append(r)
    return [r for v in thu_tu for r in khoi[v]]


def _mat_do_tien_nghiem(cands, nhiet=0.02, sigma=30.0, luoi=5, nua_cua_so=6):
    """Ban sao mat do tien nghiem cua allocate_coverage_rows, KHONG tru phan da phu."""
    diem = np.array([round(float(c.score), 4) for c in cands], dtype=np.float64)
    w = np.exp((diem - diem.max()) / nhiet)
    w /= w.sum()
    theo_video: dict = {}
    for c, wi in zip(cands, w):
        last = int(c.video_last_frame) if c.video_last_frame is not None else 1 << 31
        theo_video.setdefault(c.video_id, []).append((int(c.frame_idx), float(wi), last))
    khoi = {}
    nua = max(1, nua_cua_so // luoi)
    for vid, items in theo_video.items():
        last = max(x[2] for x in items)
        lo = max(0, min(x[0] for x in items) - 4 * int(sigma))
        hi = max(lo, min(last, max(x[0] for x in items) + 4 * int(sigma)))
        truc = np.arange(lo, hi + 1, luoi, dtype=np.int64)
        if truc.size == 0:
            continue
        mass = np.zeros(truc.size)
        for f, wi, _ in items:
            mass += wi * np.exp(-0.5 * ((truc - f) / sigma) ** 2)
        tich = np.cumsum(np.concatenate(([0.0], mass)))
        a = np.maximum(0, np.arange(mass.size) - nua)
        b = np.minimum(mass.size, np.arange(mass.size) + nua + 1)
        khoi[vid] = (truc, tich[b] - tich[a])
    return khoi


def hv_mat_do(rows, cands):
    """Sap 100 dong theo MAT DO TIEN NGHIEM ma dong do phu (khong tru phan da phu).

    Greedy phat dong theo mat do CON LAI, nen sau dong dau cua moi video no nhay
    sang video khac. Sap lai theo mat do GOC dua cac dong xac suat cao nhat len
    truoc — dung thu tu, khong dung mot dong nao.
    """
    khoi = _mat_do_tien_nghiem(cands)
    gia = []
    for v, f in rows:
        if v in khoi:
            truc, g = khoi[v]
            j = int(np.argmin(np.abs(truc - f)))
            gia.append(float(g[j]))
        else:
            gia.append(0.0)
    thu = sorted(range(len(rows)), key=lambda i: (-gia[i], i))
    return [rows[i] for i in thu]


def hv_diem_gan(rows, cands):
    """Sap 100 dong theo DIEM cua ung vien gan nhat trong cung video (0 tham so)."""
    theo_v: dict = {}
    for c in cands:
        theo_v.setdefault(c.video_id, []).append((int(c.frame_idx), float(c.score)))
    gia = []
    for v, f in rows:
        it = theo_v.get(v)
        if not it:
            gia.append(-1e9)
            continue
        gia.append(max(s - abs(fr - f) * 1e-9 for fr, s in it))
    thu = sorted(range(len(rows)), key=lambda i: (-gia[i], i))
    return [rows[i] for i in thu]


def phan_2(gt, uv, kf, ho, nen_rows, nen):
    print("\n" + "=" * 100)
    print("§2  BE RONG-TRUOC — mot HOAN VI thuan tuy cua 100 dong hien co (0 GPU, 0 truy xuat)")
    print("=" * 100)
    print("  Diem = bucket(hang cua dong DUNG dau tien). Hoan vi khong doi TAP dong")
    print("  => R@100 BAT BIEN => rui ro bi chan cung o cac bucket 1/5/20/50.\n")
    for B in (1, 2, 3, 5, 8, 12, 20):
        rw = [hoan_vi_be_rong(r, B) for r in nen_rows]
        for a, b in zip(rw, nen_rows):
            assert sorted(a) == sorted(b), "hoan vi lam doi TAP dong"
        bang(f"B = {B:<3} video dau len hang 1..B", rw, gt, ho, nen)

    print("\n  --- cac hoan vi khac (cung bat bien: TAP dong khong doi) ---")
    for ten, fn in (
        ("gom theo VIDEO (khoi lien)", hv_gom_video),
        ("sap theo MAT DO tien nghiem", None),
        ("sap theo DIEM ung vien gan nhat", None),
        ("dao nguoc 100 dong (doi chung)", lambda r, u: list(reversed(r))),
    ):
        if ten.startswith("sap theo MAT DO"):
            rw = [hv_mat_do(r, u) for r, u in zip(nen_rows, uv)]
        elif ten.startswith("sap theo DIEM"):
            rw = [hv_diem_gan(r, u) for r, u in zip(nen_rows, uv)]
        else:
            rw = [fn(r, u) for r, u in zip(nen_rows, uv)]
        for a, b in zip(rw, nen_rows):
            assert sorted(a) == sorted(b), f"{ten}: hoan vi lam doi TAP dong"
        bang(ten, rw, gt, ho, nen)


# ------------------------------------------------------------- §3 kenh Q&A


def phan_3(gt, uv, kf):
    print("\n" + "=" * 100)
    print("§3  KENH Q&A — 16/59 cau de that (27%), rieng VONG 2 la 9/25 = 36%")
    print("=" * 100)
    print("  make_submission.build_qa_rows:  answer = answerer(hits[:5], ...)")
    print("  -> dap an sinh tu 5 khung DAU cua merged_hits, bang BIEU QUYET DA SO,")
    print("     va MOT chuoi dap an do duoc dan len CA 100 DONG.")
    print("  -> calculate_vqa_r_score: dap an sai => CA 100 dong = 0, bat ke dinh vi dung.")
    print("  -> them_ung_vien_canh_b tra ve `cands`, KHONG dung vao `hits` => lever canh B")
    print("     da ship KHONG cham vao duong sinh dap an.\n")

    def o_neo(g):
        """Chi so keyframe cua neo trong video, va mang keyframe cua video do."""
        a = kf[g["video_id"]]
        return a, int(np.argmin(np.abs(a - g["frame_idx"])))

    def cach_o(g, c):
        """Khoang cach tinh theo O KEYFRAME giua ung vien c va neo; None neu khac video."""
        if c.video_id != g["video_id"]:
            return None
        a, i = o_neo(g)
        j = int(np.argmin(np.abs(a - c.frame_idx)))
        return abs(j - i)

    # cac tap 5 khung ung vien khac nhau, cung mot ngan sach 5 lan goi VLM
    def tap_hien_tai(cands, g):
        return cands[:5]

    def tap_be_rong(cands, g):
        ra, thay = [], set()
        for c in cands:
            if c.video_id in thay:
                continue
            thay.add(c.video_id)
            ra.append(c)
            if len(ra) == 5:
                break
        return ra

    def tap_sau_video1(cands, g):
        v = cands[0].video_id
        return [c for c in cands if c.video_id == v][:5]

    def tap_oracle_video(cands, g):
        return [c for c in cands if c.video_id == g["video_id"]][:5]

    TAPS = (("hits[:5]  (HIEN TAI)", tap_hien_tai),
            ("top-1 cua 5 video dau", tap_be_rong),
            ("top-5 trong video #1", tap_sau_video1),
            ("[oracle] top-5 video dung", tap_oracle_video))

    hai = np.array([g["hai_canh"] for g in gt])
    print("  DEM TAT DINH — trong 5 khung ma VLM duoc nhin, bao nhieu khung o gan neo?")
    print(f"  {'tap 5 khung':<26}{'nhom':<10}{'0 khung dung o':>16}{'>=1 dung o':>12}"
          f"{'>=1 cach<=1 o':>15}{'>=3 cach<=2 o':>15}{'TB dung video':>15}")
    for ten, fn in TAPS:
        for k, mask in (("MOT canh", ~hai), ("HAI canh", hai)):
            n0 = n1 = n1b = n3 = 0
            tbv = 0.0
            idx = np.flatnonzero(mask)
            for i in idx:
                g, cands = gt[i], uv[i]
                t5 = fn(cands, g)
                d = [cach_o(g, c) for c in t5]
                dd = [x for x in d if x is not None]
                n0 += int(sum(1 for x in dd if x == 0) == 0)
                n1 += int(any(x == 0 for x in dd))
                n1b += int(any(x <= 1 for x in dd))
                n3 += int(sum(1 for x in dd if x <= 2) >= 3)
                tbv += len(dd)
            n = len(idx)
            print(f"  {ten:<26}{k:<10}{n0:>10}/{n:<5}{n1:>12}{n1b:>15}{n3:>15}{tbv/n:>15.2f}")

    print(f"\n  Bieu quyet DA SO (Counter.most_common) can >=3/5 phieu trung nhau moi thang.")
    print(f"  Voi tap hien tai, khong cau nao co >=3 khung trong ban kinh 2 o keyframe")
    print(f"  o nhom HAI canh => phieu cua khung dung (neu co) LUON bi phieu cua canh A de bep.")
    print(f"\n  He qua nhan: diem Q&A = diem dinh vi x 1[dap an dung].")
    print(f"  Chua ai do P(dap an dung) tren bat ky bo do nao, du:")
    print(f"    - ground_truth_moi.json co san vqa_question/vqa_answer cho ca 132 muc;")
    print(f"    - ground_truth_de_that.json co 8 muc Q&A DE THAT da nguoi kiem chung dap an.")


# ----------------------------------------------------- §4 do thua keyframe


def phan_4(gt, kf, md):
    print("\n" + "=" * 100)
    print("§4  KHE KEYFRAME — vi sao quet sigma toan cuc PHAI cho ket qua nguoc chieu")
    print("=" * 100)
    cu = json.loads((ROOT / "data" / "ground_truth.json").read_text(encoding="utf-8"))
    cu = [g for g in cu if g.get("video_id") in kf]

    def khe(items):
        ra = []
        for g in items:
            a = kf[g["video_id"]]
            i = int(np.argmin(np.abs(a - int(g["frame_idx"]))))
            lo = (a[i] + a[i - 1]) // 2 if i > 0 else a[i] - 30
            hi = (a[i] + a[i + 1]) // 2 if i + 1 < len(a) else a[i] + 30
            ra.append(int(hi - lo))
        return np.array(ra)

    hai = np.array([g["hai_canh"] for g in gt])
    bo = {
        "bo CU 60 cau": khe(cu),
        "bo MOI - MOT canh": khe([g for g, h in zip(gt, hai) if not h]),
        "bo MOI - HAI canh": khe([g for g, h in zip(gt, hai) if h]),
    }
    print(f"  Be rong khe keyframe chua dap an (frame; harness boc DEU trong khe nay):")
    print(f"  {'':<22}{'n':>5}{'p25':>8}{'trung vi':>10}{'p75':>8}{'p90':>8}{'max':>8}")
    for k, v in bo.items():
        print(f"  {k:<22}{len(v):>5}{np.percentile(v,25):>8.0f}{np.median(v):>10.0f}"
              f"{np.percentile(v,75):>8.0f}{np.percentile(v,90):>8.0f}{v.max():>8.0f}")
    print(f"\n  Tham so san xuat: sigma = 30 frame (hang so TOAN CUC), nua_cua_so = 6, luoi = 5")
    print(f"  -> moi dong phu 3 o luoi = 15 frame. Khe trung vi rong "
          f"{np.median(bo['bo MOI - MOT canh']):.0f}-{np.median(bo['bo MOI - HAI canh']):.0f} frame.")
    print(f"  -> so DONG can de phu tron mot khe trung vi = {np.median(bo['bo MOI - HAI canh'])/15:.1f}")
    print(f"     dong; de phu khe p90 = {np.percentile(bo['bo MOI - HAI canh'],90)/15:.1f} dong.")

    # ty le cau roi vao khe RONG (khong the phu bang ngan sach hien co)
    for k, v in bo.items():
        print(f"  {k:<22} ty le khe > 15*10=150 frame (can >10 dong): "
              f"{100*(v > 150).mean():.0f}%   > 300 frame: {100*(v > 300).mean():.0f}%")


# -------------------------------------------------------------------- main


def hv_gom_k(K):
    def f(rows, cands):
        thu_tu, khoi = [], {}
        for r in rows:
            if r[0] not in khoi:
                khoi[r[0]] = []
                thu_tu.append(r[0])
            khoi[r[0]].append(r)
        gom = [r for v in thu_tu[:K] for r in khoi[v]]
        con = [r for r in rows if r[0] not in set(thu_tu[:K])]
        return gom + con
    return f


def hv_gom_roi_diem(rows, cands):
    """Gom theo video, va trong moi khoi sap theo diem ung vien gan nhat."""
    thu_tu, khoi = [], {}
    for r in rows:
        if r[0] not in khoi:
            khoi[r[0]] = []
            thu_tu.append(r[0])
        khoi[r[0]].append(r)
    ra = []
    for v in thu_tu:
        ra.extend(hv_diem_gan(khoi[v], cands))
    return ra


CAU_HINH = {
    "gom_video": hv_gom_video,
    "gom_video_K5": hv_gom_k(5),
    "gom_video_K10": hv_gom_k(10),
    "gom_roi_diem": hv_gom_roi_diem,
    "diem_gan": hv_diem_gan,
    "mat_do": hv_mat_do,
}


def phan_5(gt, uv, kf, nen_rows):
    print("\n" + "=" * 100)
    print("§5  GIAO THUC CHOT cho ho HOAN VI — TUNE/TEST phan tang + bootstrap theo CAU")
    print("=" * 100)
    hai = np.array([g["hai_canh"] for g in gt])
    rng = np.random.default_rng(910001)
    i_tune, i_test = [], []
    for mask in (~hai, hai):                       # phan tang theo truc bi tac dong
        idx = np.flatnonzero(mask)
        idx = idx[rng.permutation(len(idx))]       # NGAU NHIEN trong tang, khong chan/le
        i_tune.extend(idx[: len(idx) // 2])
        i_test.extend(idx[len(idx) // 2:])
    i_tune, i_test = np.array(sorted(i_tune)), np.array(sorted(i_test))
    print(f"  TUNE {len(i_tune)} muc ({int(hai[i_tune].sum())} hai canh) | "
          f"TEST {len(i_test)} muc ({int(hai[i_test].sum())} hai canh)")

    ho_tune = cac_lan_boc(GOC_HAT + 100, 3, 32, [gt[i] for i in i_tune], kf)
    ho_test = cac_lan_boc(GOC_HAT + 200, 4, 48, [gt[i] for i in i_test], kf)

    def diem(rw, idx, ho):
        mats = ma_tran_dong([rw[i] for i in idx], [gt[i] for i in idx])
        return float(np.mean([cham_nhanh(mats, [d[k] for k in range(len(idx))], CUA_SO)
                              for d in ho]))

    rows_hv = {}
    for ten, fn in CAU_HINH.items():
        rw = [fn(r, u) for r, u in zip(nen_rows, uv)]
        for a, b in zip(rw, nen_rows):
            assert sorted(a) == sorted(b), f"{ten}: doi TAP dong"
        rows_hv[ten] = rw

    nen_t = diem(nen_rows, i_tune, ho_tune)
    print(f"\n  TUNE nen = {nen_t:.4f}")
    bang_tune = {}
    for ten, rw in rows_hv.items():
        d = diem(rw, i_tune, ho_tune)
        bang_tune[ten] = d
        print(f"    {ten:<16} {d:.4f}  ({100*(d/nen_t-1):+.1f}%)")
    chot = max(bang_tune, key=bang_tune.get)
    print(f"  CHOT tren TUNE: {chot}")

    print("\n  TEST — doc DUNG MOT LAN")
    gt_t = [gt[i] for i in i_test]
    mn = ma_tran_dong([nen_rows[i] for i in i_test], gt_t)
    mc = ma_tran_dong([rows_hv[chot][i] for i in i_test], gt_t)

    def tung_cau(mats):
        ra = np.zeros(len(gt_t))
        for draws in ho_test:
            for q in range(len(gt_t)):
                ra[q] += cham_nhanh([mats[q]], [draws[q]], CUA_SO)
        return ra / len(ho_test)

    dn, dc = tung_cau(mn), tung_cau(mc)
    hai_t = np.array([g["hai_canh"] for g in gt_t])
    rb = np.random.default_rng(4242)
    for ten, sel in (("toan bo TEST", np.ones(len(gt_t), bool)),
                     ("nhom HAI canh", hai_t), ("nhom MOT canh", ~hai_t)):
        a, b = dn[sel], dc[sel]
        lay = rb.integers(0, len(a), size=(4000, len(a)))
        dd = b[lay].mean(axis=1) - a[lay].mean(axis=1)
        lo, hi = np.percentile(dd, [2.5, 97.5])
        print(f"    {ten:<16} ({sel.sum():>2}) {a.mean():.4f} -> {b.mean():.4f} = "
              f"{100*(b.mean()/a.mean()-1):+.1f}%  KTC95 [{lo:+.4f}, {hi:+.4f}]  "
              f"P(<=0) = {(dd <= 0).mean():.1%}")
    doi = int(((dc - dn) != 0).sum())
    print(f"    phan ra tat dinh: {int((dc > dn).sum())} cau tot len, "
          f"{int((dc < dn).sum())} cau xau di, {len(gt_t)-doi} cau khong doi")


def phan_7(gt, kf, ho, nen_rows):
    """Phan bo HANG cua dong DUNG dau tien — bang chia diem cho tung bucket."""
    print("\n" + "=" * 100)
    print("§7  DONG DUNG DAU TIEN nam o HANG NAO — chia deficit thanh 'khong co' vs 'chon sai cho'")
    print("=" * 100)
    mats = ma_tran_dong(nen_rows, gt)
    hai = np.array([g["hai_canh"] for g in gt])
    khoang = [(1, 1), (2, 5), (6, 20), (21, 50), (51, 100)]
    for ten, mask in (("MOT canh", ~hai), ("HAI canh", hai)):
        cnt = np.zeros(6)
        for d in ho:
            for i in np.flatnonzero(mask):
                f, m = mats[i]
                t = np.asarray(d[i])
                dd = np.abs(f[None, :] - t[:, None])
                for half in CUA_SO:
                    hit = m[None, :] & (dd <= half)
                    co = hit.any(axis=1)
                    r = hit.argmax(axis=1) + 1
                    for bi, (a, b) in enumerate(khoang):
                        cnt[bi] += (co & (r >= a) & (r <= b)).mean()
                    cnt[5] += (~co).mean()
        cnt /= (len(ho) * len(CUA_SO) * mask.sum())
        print(f"  {ten} (n={int(mask.sum())}): " +
              " | ".join(f"hang {a}-{b}: {100*c:.1f}%" for (a, b), c in zip(khoang, cnt[:5])) +
              f" | KHONG co dong dung: {100*cnt[5]:.1f}%")
        chon_sai = cnt[2] + cnt[3] + cnt[4]
        print(f"    -> co dong dung: {100*(1-cnt[5]):.1f}%; trong do nam o hang >=6: "
              f"{100*chon_sai:.1f} diem pt")
        print(f"    -> deficit do THU TU (dua het ve hang 1) = {1 - cnt[5] - sum(c*w for c,w in zip(cnt[:5],[1.0,0.8,0.6,0.4,0.2])):.4f}")


def phan_6(kf):
    """Kiem cheo tren bo 60 cau CU — tap doc lap, doc mot lan."""
    print("\n" + "=" * 100)
    print("§6  KIEM CHEO bo 60 cau CU (tap doc lap) — hoan vi gom_video")
    print("=" * 100)
    f = ROOT / "data" / "cache_phu_quet_luoi" / "ung_vien.json"
    raw = json.loads(f.read_text(encoding="utf-8"))
    gt = [{"video_id": g["video_id"], "frame_idx": int(g["frame_idx"]), "hai_canh": False}
          for g in raw["gt"]]
    uv = [[Candidate(v, int(fi), float(sc), int(lf)) for v, fi, sc, lf in q] for q in raw["cands"]]
    rows = [allocate_rows(c, "coverage", DEFAULT_N_FLAT, PLAN)[:MAX_ROWS] for c in uv]
    ho = cac_lan_boc(GOC_HAT + 300, 4, 48, gt, kf)

    def diem(rw):
        mats = ma_tran_dong(rw, gt)
        return float(np.mean([cham_nhanh(mats, d, CUA_SO) for d in ho]))

    a = diem(rows)
    for ten, fn in (("gom_video", hv_gom_video), ("diem_gan", hv_diem_gan),
                    ("mat_do", hv_mat_do)):
        rw = [fn(r, u) for r, u in zip(rows, uv)]
        for x, y in zip(rw, rows):
            assert sorted(x) == sorted(y)
        b = diem(rw)
        print(f"  {ten:<12} {a:.4f} -> {b:.4f} = {100*(b/a-1):+.1f}%")

    # bootstrap theo cau cho gom_video
    rw = [hv_gom_video(r, u) for r, u in zip(rows, uv)]
    mn, mc = ma_tran_dong(rows, gt), ma_tran_dong(rw, gt)

    def tung_cau(mats):
        ra = np.zeros(len(gt))
        for draws in ho:
            for q in range(len(gt)):
                ra[q] += cham_nhanh([mats[q]], [draws[q]], CUA_SO)
        return ra / len(ho)

    dn, dc = tung_cau(mn), tung_cau(mc)
    rb = np.random.default_rng(4243)
    lay = rb.integers(0, len(gt), size=(4000, len(gt)))
    dd = dc[lay].mean(axis=1) - dn[lay].mean(axis=1)
    lo, hi = np.percentile(dd, [2.5, 97.5])
    print(f"  bootstrap gom_video: chenh {dc.mean()-dn.mean():+.4f}; "
          f"KTC95 [{lo:+.4f}, {hi:+.4f}]; P(<=0) = {(dd <= 0).mean():.1%}")
    print(f"  phan ra: {int((dc > dn).sum())} tot len / {int((dc < dn).sum())} xau di / "
          f"{int((dc == dn).sum())} khong doi")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phan", default="1,2,3,4,5,6,7")
    args = ap.parse_args()
    phan = {int(x) for x in args.phan.split(",") if x.strip()}

    gt, uv, kf, md = nap()
    hai = sum(g["hai_canh"] for g in gt)
    print(f"bo do sach: {len(gt)} muc  ({len(gt)-hai} MOT canh / {hai} HAI canh)")
    print(f"goc hat {GOC_HAT}, {SO_HO} ho x {SO_BOC} boc, cua so {CUA_SO}")

    nen_rows = [allocate_rows(c, "coverage", DEFAULT_N_FLAT, PLAN)[:MAX_ROWS] for c in uv]
    ho = cac_lan_boc(GOC_HAT, SO_HO, SO_BOC, gt, kf)
    print("\nNEN (duong san xuat, allocator coverage, khong canh-B):")
    nen = bang("nen", nen_rows, gt, ho)

    if 1 in phan:
        phan_1(gt, uv, kf, ho, nen_rows, nen)
    if 2 in phan:
        phan_2(gt, uv, kf, ho, nen_rows, nen)
    if 3 in phan:
        phan_3(gt, uv, kf)
    if 7 in phan:
        phan_7(gt, kf, ho, nen_rows)
    if 6 in phan:
        phan_6(kf)
    if 5 in phan:
        phan_5(gt, uv, kf, nen_rows)
    if 4 in phan:
        phan_4(gt, kf, md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
