"""Quét lưới tham số cho bộ phân bổ PHỦ XÁC SUẤT, có chia TUNE/TEST chống overfit.

Bối cảnh: ``experiment_phu_xac_suat.py`` đã đo được +10% trên cả 60 câu, nhưng
mới quét 4 tổ hợp và nhiệt 0,05 đã ÂM 5,6% — tham số NHẠY, mà chấm cả 60 câu
rồi chọn tham số trên chính 60 câu đó là tự chấm bài mình. Script này sửa cả hai:

* Lưới 48 tổ hợp: nhiệt {0,01, 0,015, 0,02, 0,03} × sigma {20, 30, 40, 55}
  × nửa_cửa_sổ {6, 10, 15} × lưới {5}.
* Chia 60 câu theo CHỈ SỐ chẵn/lẻ: 30 câu TUNE (chẵn) để chọn tham số,
  30 câu TEST (lẻ) chỉ chấm ĐÚNG MỘT tổ hợp đã chốt (và nền). Hai phía dùng
  họ hạt giống tách rời (TUNE gốc 50000, TEST gốc 90000) nên không rò rỉ.
* TUNE: 3 họ hạt giống × 32 lần bốc. TEST: 4 họ × 48 lần bốc.
* Chấm bằng đúng ``final_score``/``r_score_kis`` của src/core/submission.py —
  bản vector hoá được ĐỐI CHIẾU từng giá trị với bản gốc trước khi dùng.
* Ứng viên (đường sản xuất ``ranked_hits``), các bộ 100 dòng, và điểm từng họ
  đều cache xuống đĩa — chạy lại chỉ mất vài giây.

    python scripts/experiment_phu_quet_luoi.py            # chạy đủ
    python scripts/experiment_phu_quet_luoi.py --refresh  # bỏ cache, tính lại
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.experiment_phu_xac_suat import phu_xac_suat  # noqa: E402  (bản tham chiếu)
from scripts.make_submission import (  # noqa: E402
    DEFAULT_DEPTH_COST,
    DEFAULT_N_FLAT,
    RETRIEVE_TOP_N,
    ranked_hits,
)
from src.core.submission import (  # noqa: E402
    MAX_ROWS,
    RANK_THRESHOLDS,
    AllocationPlan,
    Candidate,
    allocate_hybrid_rows,
    final_score,
    r_score_kis,
)

# ---------------------------------------------------------------------------
# Lưới tham số và hạt giống
# ---------------------------------------------------------------------------

LUOI_NHIET = (0.01, 0.015, 0.02, 0.03)
LUOI_SIGMA = (20.0, 30.0, 40.0, 55.0)
LUOI_NUA = (6, 10, 15)
LUOI_BUOC = (5,)

#: gốc hạt giống tách rời nhau và tách khỏi experiment_phu_xac_suat (30000)
GOC_TUNE = 50000
GOC_TEST = 90000


def cau_hinh_luoi():
    return [
        (nhiet, sigma, nua, luoi)
        for nhiet in LUOI_NHIET
        for sigma in LUOI_SIGMA
        for nua in LUOI_NUA
        for luoi in LUOI_BUOC
    ]


def ten_cau_hinh(c) -> str:
    return f"n{c[0]:g}_s{c[1]:g}_w{c[2]}_l{c[3]}"


# ---------------------------------------------------------------------------
# Bộ phân bổ phủ xác suất — bản nhanh, tương đương từng dòng với bản gốc
# ---------------------------------------------------------------------------


def phu_xac_suat_nhanh(candidates, nhiet, sigma, nua_cua_so, budget=MAX_ROWS, luoi=5):
    """Cùng thuật toán với ``phu_xac_suat`` nhưng mỗi bước chỉ tính lại video
    vừa bị đục lỗ, thay vì quét cumsum toàn bộ video mỗi bước (nhanh ~30 lần).

    Tính tương đương được khẳng định bằng ``_doi_chieu_bo_phan_bo`` lúc chạy.
    """
    if not candidates:
        return []

    diem = np.array([c.score for c in candidates], dtype=np.float64)
    w = np.exp((diem - diem.max()) / max(nhiet, 1e-9))
    w /= w.sum()

    theo_video: dict[str, list[tuple[int, float, int]]] = defaultdict(list)
    for c, wi in zip(candidates, w):
        theo_video[c.video_id].append((int(c.frame_idx), float(wi), int(c.video_last_frame)))

    khoi: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for vid, items in theo_video.items():
        last = max(x[2] for x in items)
        lo = max(0, min(x[0] for x in items) - 4 * int(sigma))
        hi = min(last, max(x[0] for x in items) + 4 * int(sigma))
        truc = np.arange(lo, hi + 1, luoi, dtype=np.int64)
        if truc.size == 0:
            continue
        mass = np.zeros(truc.size, dtype=np.float64)
        for f, wi, _ in items:
            mass += wi * np.exp(-0.5 * ((truc - f) / sigma) ** 2)
        khoi[vid] = (truc, mass)

    nua = max(1, nua_cua_so // luoi)
    chua_phu = {v: m.copy() for v, (_t, m) in khoi.items()}
    bien = {}
    for vid, (_t, m) in khoi.items():
        n = m.size
        idx = np.arange(n)
        bien[vid] = (np.maximum(0, idx - nua), np.minimum(n, idx + nua + 1))

    def gia_tot_nhat(vid):
        con = chua_phu[vid]
        tich = np.cumsum(np.concatenate(([0.0], con)))
        lo_i, hi_i = bien[vid]
        gia = tich[hi_i] - tich[lo_i]
        j = int(np.argmax(gia))
        return float(gia[j]), j

    vids = list(khoi)
    tot = {vid: gia_tot_nhat(vid) for vid in vids}

    rows: list[tuple[str, int]] = []
    while len(rows) < budget:
        tot_v, tot_gia, tot_i = None, 0.0, -1
        for vid in vids:  # cùng thứ tự duyệt (thứ tự chèn) với bản gốc
            g, j = tot[vid]
            if g > tot_gia:
                tot_v, tot_gia, tot_i = vid, g, j
        if tot_v is None:
            break
        truc, _ = khoi[tot_v]
        rows.append((tot_v, int(truc[tot_i])))
        lo = max(0, tot_i - nua)
        chua_phu[tot_v][lo : tot_i + nua + 1] = 0.0
        tot[tot_v] = gia_tot_nhat(tot_v)
    return rows[:budget]


def _doi_chieu_bo_phan_bo(cands_of, cauhinh):
    """Bản nhanh phải cho ra ĐÚNG các dòng của bản gốc trên vài mẫu, nếu không thì dừng."""
    mau = [q for q in cands_of[:4] if q]
    for c in cauhinh:
        for cs in mau:
            goc = phu_xac_suat(cs, nhiet=c[0], sigma=c[1], nua_cua_so=c[2], luoi=c[3])
            nhanh = phu_xac_suat_nhanh(cs, nhiet=c[0], sigma=c[1], nua_cua_so=c[2], luoi=c[3])
            if goc != nhanh:
                raise AssertionError(f"bản nhanh lệch bản gốc tại cấu hình {c}: "
                                     f"{len(goc)} vs {len(nhanh)} dòng, khác từ dòng "
                                     f"{next(i for i,(a,b) in enumerate(zip(goc,nhanh)) if a!=b)}")


# ---------------------------------------------------------------------------
# Chấm điểm — vector hoá, đối chiếu với final_score/r_score_kis của sản xuất
# ---------------------------------------------------------------------------

#: giá trị Final Score khi dòng ĐÚNG đầu tiên nằm ở hạng r (1-based); [0] = không trúng
BUCKET = np.array(
    [0.0] + [sum(1 for k in RANK_THRESHOLDS if k >= r) / len(RANK_THRESHOLDS) for r in range(1, MAX_ROWS + 1)]
)


def ma_tran_dong(rows_of, gt_sub):
    """Đổi 100 dòng mỗi câu thành (frames[100], đúng_video[100]) đệm sẵn."""
    ra = []
    for rows, g in zip(rows_of, gt_sub):
        f = np.full(MAX_ROWS, -(10**9), dtype=np.int64)
        m = np.zeros(MAX_ROWS, dtype=bool)
        for i, (v, fr) in enumerate(rows[:MAX_ROWS]):
            f[i] = int(fr)
            m[i] = v == g["video_id"]
        ra.append((f, m))
    return ra


def cham_nhanh(mats, draws_theo_cau, windows):
    """Điểm trung bình (cửa sổ × câu × lần bốc) — thuần numpy."""
    tong = 0.0
    for (f, m), truths in zip(mats, draws_theo_cau):
        t = np.asarray(truths, dtype=np.int64)
        d = np.abs(f[None, :] - t[:, None])  # (bốc, 100)
        per_q = 0.0
        for half in windows:
            hit = m[None, :] & (d <= half)
            co = hit.any(axis=1)
            hang = hit.argmax(axis=1) + 1
            per_q += np.where(co, BUCKET[hang], 0.0).mean()
        tong += per_q / len(windows)
    return tong / len(mats)


def cham_goc(rows_of, gt_sub, draws_theo_cau, windows):
    """Bản gốc, đúng từng chữ theo src/core/submission.py — chỉ dùng để đối chiếu."""
    per_w = []
    for half in windows:
        tot = 0.0
        n = 0
        for rows, g, truths in zip(rows_of, gt_sub, draws_theo_cau):
            for t in truths:
                span = (t - half, t + half)
                tot += final_score([r_score_kis(v, f, g["video_id"], span) for v, f in rows])
                n += 1
        per_w.append(tot / n)
    return sum(per_w) / len(per_w)


def _doi_chieu_bo_cham(rows_of, gt_sub, draws_theo_cau, windows):
    goc = cham_goc(rows_of, gt_sub, draws_theo_cau, windows)
    nhanh = cham_nhanh(ma_tran_dong(rows_of, gt_sub), draws_theo_cau, windows)
    if abs(goc - nhanh) > 1e-12:
        raise AssertionError(f"bộ chấm vector hoá lệch bản sản xuất: {goc!r} vs {nhanh!r}")


# ---------------------------------------------------------------------------
# Bốc khoảnh khắc thật — đúng hệt experiment_phu_xac_suat (bọc trong khe keyframe)
# ---------------------------------------------------------------------------


def boc_khoanh_khac(seed, gt_sub, kf):
    rng = np.random.default_rng(seed)
    out = []
    for g in gt_sub:
        a = kf[g["video_id"]]
        i = int(np.argmin(np.abs(a - int(g["frame_idx"]))))
        lo = (a[i] + a[i - 1]) // 2 if i > 0 else a[i] - 30
        hi = (a[i] + a[i + 1]) // 2 if i + 1 < len(a) else a[i] + 30
        out.append(int(rng.integers(lo, max(lo + 1, hi))))
    return out


def cac_lan_boc(goc, so_ho, so_boc, gt_sub, kf):
    """[họ][câu] -> mảng vị trí thật của ``so_boc`` lần bốc."""
    ho = []
    for s in range(so_ho):
        draws = [boc_khoanh_khac(goc + s * 1000 + t, gt_sub, kf) for t in range(so_boc)]
        # chuyển thành theo-câu: mỗi câu một mảng (so_boc,)
        ho.append([np.array([d[qi] for d in draws]) for qi in range(len(gt_sub))])
    return ho


# ---------------------------------------------------------------------------
# Cache xuống đĩa
# ---------------------------------------------------------------------------


def nap_ung_vien(data_dir: str, cache_dir: Path, refresh: bool):
    """Ứng viên đường sản xuất + bản đồ keyframe, cache 1 lần vì đây là phần đắt."""
    f = cache_dir / "ung_vien.json"
    if f.exists() and not refresh:
        raw = json.loads(f.read_text(encoding="utf-8"))
        if raw.get("version") == 1 and raw.get("top_n") == RETRIEVE_TOP_N:
            gt = raw["gt"]
            cands_of = [
                [Candidate(v, fi, sc, lf) for v, fi, sc, lf in q] for q in raw["cands"]
            ]
            kf = {v: np.array(a, dtype=np.int64) for v, a in raw["kf"].items()}
            print(f"  cache ứng viên: {f} ({len(gt)} câu)", flush=True)
            return gt, cands_of, kf

    print("  nạp chỉ mục (chỉ lần đầu; các lần sau đọc cache) ...", flush=True)
    from src.core.kis_engine import KISEngine

    eng = KISEngine(data_dir).load()
    gt_full = json.loads((Path(data_dir) / "ground_truth.json").read_text(encoding="utf-8"))
    gt_full = [g for g in gt_full if g.get("video_id") in eng.last_frame]

    cands_of, gt = [], []
    for g in gt_full:
        hits = ranked_hits(eng, g["kis_query_vi"], g.get("kis_query_en"))
        cands_of.append([Candidate(h.video_id, h.frame_idx, h.score, h.video_last_frame) for h in hits])
        gt.append({"n": g.get("n"), "video_id": g["video_id"], "frame_idx": int(g["frame_idx"])})

    kf_list: dict[str, list[int]] = {}
    for m in eng.metadata:
        kf_list.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
    kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in kf_list.items()}

    cache_dir.mkdir(parents=True, exist_ok=True)
    f.write_text(
        json.dumps(
            {
                "version": 1,
                "top_n": RETRIEVE_TOP_N,
                "gt": gt,
                "cands": [
                    [[c.video_id, int(c.frame_idx), float(c.score), int(c.video_last_frame)] for c in q]
                    for q in cands_of
                ],
                "kf": {v: [int(x) for x in a] for v, a in kf.items()},
            }
        ),
        encoding="utf-8",
    )
    return gt, cands_of, kf


def nap_dong(cache_dir: Path, tag: str, refresh: bool, tinh):
    """Cache 100 dòng/câu của một (cấu hình × tập câu)."""
    d = cache_dir / "dong"
    f = d / f"{tag}.json"
    if f.exists() and not refresh:
        return [[(v, int(fr)) for v, fr in q] for q in json.loads(f.read_text(encoding="utf-8"))]
    rows_of = tinh()
    d.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps([[[v, int(fr)] for v, fr in q] for q in rows_of]), encoding="utf-8")
    return rows_of


class CacheDiem:
    """Điểm từng (nhãn cấu hình × họ hạt giống), giữ trên đĩa để chạy lại nhanh."""

    def __init__(self, path: Path, harness: dict, refresh: bool):
        self.path = path
        self.data = {"harness": harness, "diem": {}}
        if path.exists() and not refresh:
            cu = json.loads(path.read_text(encoding="utf-8"))
            if cu.get("harness") == harness:
                self.data = cu

    def lay(self, nhan: str, tinh):
        if nhan not in self.data["diem"]:
            self.data["diem"][nhan] = tinh()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data), encoding="utf-8")
        return self.data["diem"][nhan]


# ---------------------------------------------------------------------------
# Chạy
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache_phu_quet_luoi"))
    ap.add_argument("--windows", default="6,10,20", help="các nửa-bề-rộng cửa sổ chấm, lấy trung bình")
    ap.add_argument("--tune-seeds", type=int, default=3)
    ap.add_argument("--tune-draws", type=int, default=32)
    ap.add_argument("--test-seeds", type=int, default=4)
    ap.add_argument("--test-draws", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="giới hạn số câu (chỉ để chạy thử)")
    ap.add_argument("--refresh", action="store_true", help="bỏ mọi cache, tính lại từ đầu")
    ap.add_argument("--tune-phia", choices=("chan", "le"), default="chan",
                    help="nửa nào dùng để CHỌN tham số (mặc định: chỉ số chẵn); "
                         "'le' chạy fold đảo chiều để kiểm tra chéo")
    args = ap.parse_args()

    cache_dir = Path(args.cache)
    windows = [int(w) for w in args.windows.split(",")]
    cauhinh = cau_hinh_luoi()
    t0 = time.time()

    print("=== 1) Ứng viên đường sản xuất (ranked_hits) ===", flush=True)
    gt, cands_of, kf = nap_ung_vien(args.data, cache_dir, args.refresh)
    if args.limit:
        gt, cands_of = gt[: args.limit], cands_of[: args.limit]

    # chia chẵn/lẻ theo CHỈ SỐ trong danh sách ground truth
    du_tune = 0 if args.tune_phia == "chan" else 1
    pt, px = ("chan", "le") if du_tune == 0 else ("le", "chan")
    i_tune = [i for i in range(len(gt)) if i % 2 == du_tune]
    i_test = [i for i in range(len(gt)) if i % 2 != du_tune]
    gt_tune, c_tune = [gt[i] for i in i_tune], [cands_of[i] for i in i_tune]
    gt_test, c_test = [gt[i] for i in i_test], [cands_of[i] for i in i_test]
    print(f"  {len(gt)} câu -> TUNE {len(gt_tune)} (chỉ số {pt}) / TEST {len(gt_test)} (chỉ số {px})")

    print("=== 2) Đối chiếu bản nhanh với bản gốc phu_xac_suat ===", flush=True)
    _doi_chieu_bo_phan_bo(c_tune, [cauhinh[0], cauhinh[-1], (0.02, 20.0, 10, 5)])
    print("  khớp từng dòng trên 4 câu × 3 cấu hình.")

    # ---- nền: bộ phân bổ đang nộp -------------------------------------------
    plan = AllocationPlan(breadth_cost=1.0, depth_cost=DEFAULT_DEPTH_COST, step=10)

    def dong_nen(cs_list):
        return [allocate_hybrid_rows(c, n_flat=DEFAULT_N_FLAT, plan=plan)[:MAX_ROWS] for c in cs_list]

    nen_tune = nap_dong(cache_dir, f"nen_{pt}{len(gt_tune)}", args.refresh, lambda: dong_nen(c_tune))
    nen_test = nap_dong(cache_dir, f"nen_{px}{len(gt_test)}", args.refresh, lambda: dong_nen(c_test))

    # ---- lần bốc ------------------------------------------------------------
    ho_tune = cac_lan_boc(GOC_TUNE, args.tune_seeds, args.tune_draws, gt_tune, kf)
    ho_test = cac_lan_boc(GOC_TEST, args.test_seeds, args.test_draws, gt_test, kf)

    print("=== 3) Đối chiếu bộ chấm vector hoá với final_score/r_score_kis ===", flush=True)
    mau_draws = [t[:3] for t in ho_tune[0]]
    _doi_chieu_bo_cham(nen_tune, gt_tune, mau_draws, windows)
    print("  khớp tuyệt đối trên 30 câu × 3 lần bốc × 3 cửa sổ (nền).")

    harness_tune = {
        "windows": windows, "seeds": args.tune_seeds, "draws": args.tune_draws,
        "goc": GOC_TUNE, "n_cau": len(gt_tune), "phia": pt,
    }
    harness_test = {
        "windows": windows, "seeds": args.test_seeds, "draws": args.test_draws,
        "goc": GOC_TEST, "n_cau": len(gt_test), "phia": px,
    }
    diem_tune = CacheDiem(cache_dir / f"diem_tune_{pt}.json", harness_tune, args.refresh)
    diem_test = CacheDiem(cache_dir / f"diem_test_{px}.json", harness_test, args.refresh)

    def diem_cac_ho(rows_of, gt_sub, ho):
        mats = ma_tran_dong(rows_of, gt_sub)
        return [cham_nhanh(mats, draws, windows) for draws in ho]

    # ---- 4) quét TUNE -------------------------------------------------------
    print(f"=== 4) TUNE: {len(cauhinh)} tổ hợp × {len(gt_tune)} câu × "
          f"{args.tune_seeds} họ × {args.tune_draws} bốc ===", flush=True)
    nen_tune_ho = diem_tune.lay("nen", lambda: diem_cac_ho(nen_tune, gt_tune, ho_tune))
    nen_tune_m, nen_tune_sd = float(np.mean(nen_tune_ho)), float(np.std(nen_tune_ho))

    ket_qua = {}
    for ci, c in enumerate(cauhinh):
        tag = ten_cau_hinh(c)

        def tinh_diem(c=c, tag=tag):
            rows = nap_dong(
                cache_dir, f"{tag}_{pt}{len(gt_tune)}", args.refresh,
                lambda: [phu_xac_suat_nhanh(cs, nhiet=c[0], sigma=c[1], nua_cua_so=c[2], luoi=c[3])
                         for cs in c_tune],
            )
            return diem_cac_ho(rows, gt_tune, ho_tune)

        ho = diem_tune.lay(tag, tinh_diem)
        ket_qua[c] = (float(np.mean(ho)), float(np.std(ho)))
        if (ci + 1) % 12 == 0:
            print(f"  ... {ci + 1}/{len(cauhinh)} tổ hợp, {time.time() - t0:.0f}s", flush=True)

    print(f"\nNỀN trên TUNE: {nen_tune_m:.4f} ±{nen_tune_sd:.4f}")
    print(f"\n{'nhiệt':>7}{'sigma':>7}{'nửa_cs':>8}{'lưới':>6}{'điểm':>9}{'±':>8}{'so nền':>9}")
    print("-" * 56)
    for c in cauhinh:
        m, sd = ket_qua[c]
        dau = " <-- TỐT NHẤT" if c == max(ket_qua, key=lambda k: ket_qua[k][0]) else ""
        print(f"{c[0]:7.3f}{c[1]:7.1f}{c[2]:8d}{c[3]:6d}{m:9.4f}{sd:8.4f}{100*(m/nen_tune_m-1):+8.1f}%{dau}")

    chot = max(ket_qua, key=lambda k: ket_qua[k][0])
    chot_m, chot_sd = ket_qua[chot]
    print(f"\nCHỐT trên TUNE: nhiệt={chot[0]} sigma={chot[1]} nửa_cửa_sổ={chot[2]} lưới={chot[3]}"
          f" -> {chot_m:.4f} ({100*(chot_m/nen_tune_m-1):+.1f}% so nền)")

    # ---- 5) độ nhạy quanh đỉnh ---------------------------------------------
    print("\n=== 5) Độ nhạy quanh đỉnh (lân cận 1 bước lưới, điểm TUNE) ===")
    truc_ten = ["nhiệt", "sigma", "nửa_cửa_sổ"]
    truc_gia_tri = [LUOI_NHIET, LUOI_SIGMA, LUOI_NUA]
    for ax, (ten, gia_tri) in enumerate(zip(truc_ten, truc_gia_tri)):
        i = gia_tri.index(chot[ax])
        for di in (-1, +1):
            j = i + di
            if not (0 <= j < len(gia_tri)):
                continue
            lan_can = list(chot)
            lan_can[ax] = gia_tri[j]
            lan_can = tuple(lan_can)
            m, _sd = ket_qua[lan_can]
            print(f"  {ten} {chot[ax]:g} -> {gia_tri[j]:g}: {m:.4f} ({100*(m/chot_m-1):+.2f}% so đỉnh)")

    # ---- 6) TEST: chỉ tổ hợp đã chốt và nền ---------------------------------
    print(f"\n=== 6) TEST ({len(gt_test)} câu chỉ số {px}, {args.test_seeds} họ × {args.test_draws} bốc,"
          f" hạt giống tách rời) ===", flush=True)
    nen_test_ho = diem_test.lay("nen", lambda: diem_cac_ho(nen_test, gt_test, ho_test))

    tag = ten_cau_hinh(chot)
    rows_chot_test = nap_dong(
        cache_dir, f"{tag}_{px}{len(gt_test)}", args.refresh,
        lambda: [phu_xac_suat_nhanh(cs, nhiet=chot[0], sigma=chot[1], nua_cua_so=chot[2], luoi=chot[3])
                 for cs in c_test],
    )
    chot_test_ho = diem_test.lay(tag, lambda: diem_cac_ho(rows_chot_test, gt_test, ho_test))

    nen_t_m, nen_t_sd = float(np.mean(nen_test_ho)), float(np.std(nen_test_ho))
    chot_t_m, chot_t_sd = float(np.mean(chot_test_ho)), float(np.std(chot_test_ho))
    up = 100 * (chot_t_m / nen_t_m - 1)
    print(f"  nền  : {nen_t_m:.4f} ±{nen_t_sd:.4f}")
    print(f"  chốt : {chot_t_m:.4f} ±{chot_t_sd:.4f}   ({up:+.1f}% so nền)")

    def co_video(rows_of, gt_sub):
        return sum(1 for rows, g in zip(rows_of, gt_sub) if any(v == g["video_id"] for v, _f in rows))

    print(f"  số câu có VIDEO ĐÚNG trong 100 dòng (TEST): nền {co_video(nen_test, gt_test)}"
          f"/{len(gt_test)}, chốt {co_video(rows_chot_test, gt_test)}/{len(gt_test)}")

    bien = max(nen_t_sd, 0.0005)
    print("\n=== KẾT LUẬN ===")
    if (chot_t_m - nen_t_m) < 2 * bien:
        print(f"  HOÀ: chênh {chot_t_m - nen_t_m:.4f} < 2 lần sai số ({bien:.4f}) trên TEST.")
    elif up < 5.0:
        print(f"  OVERFIT: TEST chỉ giữ {up:+.1f}% (< +5%) dù TUNE cho {100*(chot_m/nen_tune_m-1):+.1f}%.")
    else:
        print(f"  GIỮ ĐƯỢC: TEST {up:+.1f}% (>= +5%) và vượt 2 sigma — tham số đáng đưa vào sản xuất.")
    print(f"\nXong sau {time.time() - t0:.0f}s. Cache: {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
