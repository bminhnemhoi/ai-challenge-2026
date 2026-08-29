"""Cổng TUNE/TEST cho BẢN SHIP của bộ phân bổ phủ xác suất (bước 3 + 5.1).

Con số +15,3% trong docs/SHIP_PHU_XAC_SUAT.md thuộc về BẢN THÍ NGHIỆM
(``phu_xac_suat_nhanh``).  Bản ship (``allocate_coverage_rows``) thêm ba thứ
mà thí nghiệm chưa có: làm tròn score 4 chữ số tại cửa vào, lượng tử hoá khối
lượng 1e-9, và đuôi lấp hybrid.  Script này trả lời câu hỏi duy nhất còn mở
trước khi được phép đổi mặc định: **con số có giữ nguyên trên đường mã sẽ
chạy thật trong trận không?**

Hai chế độ:

* Mặc định — cổng trên 60 câu ground truth: phân bổ bằng ĐÚNG hàm điều phối
  ``allocate_rows`` của make_submission (cả hai allocator), chấm theo đúng luật
  harness (họ hạt giống TEST gốc 90000, tách khỏi mọi lần chọn tham số; cửa sổ
  6/10/20; luật hoà 2 sigma), in bảng từng câu + cột "video dòng 1 có đổi
  không" + độ trễ thật của allocator.  Tham số coverage ĐÃ CHỐT từ trước nên
  đọc cả hai nửa không phải là rò rỉ — không có lựa chọn nào xảy ra ở đây nữa.

* ``--queries <dir> --out <dir>`` — diff cấu trúc trên đề thật (không GT):
  truy xuất MỘT lần, phân bổ HAI lần, xuất hai bộ csv cạnh nhau và bảng
  từng câu (video dòng 1 giống/khác, số video được phủ, số dòng thuộc video
  dẫn đầu).  Dùng trước ngày thi để biết coverage đổi những gì trên phân bố
  đề thật, khi không thể biết đúng/sai.

    python scripts/so_sanh_allocator.py                       # cổng GT
    python scripts/so_sanh_allocator.py --queries round2/de/queries --out /tmp/ss
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.experiment_phu_quet_luoi import (  # noqa: E402
    BUCKET,
    GOC_TEST,
    CacheDiem,
    _doi_chieu_bo_cham,
    cac_lan_boc,
    cham_nhanh,
    ma_tran_dong,
    nap_dong,
    nap_ung_vien,
)
from scripts.make_submission import (  # noqa: E402
    DEFAULT_DEPTH_COST,
    DEFAULT_N_FLAT,
    allocate_rows,
)
from src.core.submission import (  # noqa: E402
    MAX_ROWS,
    AllocationPlan,
    Candidate,
)


def _plan() -> AllocationPlan:
    return AllocationPlan(breadth_cost=1.0, depth_cost=DEFAULT_DEPTH_COST, step=10)


def dong_ship(cands_list, allocator: str):
    """100 dòng/câu bằng đúng đường mã make_submission sẽ chạy trong trận."""
    return [allocate_rows(c, allocator, DEFAULT_N_FLAT, _plan())[:MAX_ROWS] for c in cands_list]


def diem_tung_cau(rows_of, gt_sub, ho, windows):
    """Điểm final_score trung bình từng câu (họ × bốc × cửa sổ)."""
    mats = ma_tran_dong(rows_of, gt_sub)
    out = np.zeros(len(gt_sub))
    for draws in ho:
        for qi, ((f, m), truths) in enumerate(zip(mats, draws)):
            t = np.asarray(truths, dtype=np.int64)
            d = np.abs(f[None, :] - t[:, None])
            per_q = 0.0
            for half in windows:
                hit = m[None, :] & (d <= half)
                co = hit.any(axis=1)
                hang = hit.argmax(axis=1) + 1
                per_q += np.where(co, BUCKET[hang], 0.0).mean()
            out[qi] += per_q / len(windows)
    return out / len(ho)


# ---------------------------------------------------------------------------
# Chế độ 1: cổng trên ground truth
# ---------------------------------------------------------------------------


def cong_gt(args) -> int:
    cache_dir = Path(args.cache)
    windows = [int(w) for w in args.windows.split(",")]
    t0 = time.time()

    print("=== 1) Ứng viên đường sản xuất (cache của experiment_phu_quet_luoi) ===", flush=True)
    gt, cands_of, kf = nap_ung_vien(args.data, cache_dir, args.refresh)
    if args.limit:
        gt, cands_of = gt[: args.limit], cands_of[: args.limit]

    print("=== 2) Phân bổ bằng BẢN SHIP (allocate_rows của make_submission) ===", flush=True)
    t1 = time.time()
    do_tre = []
    rows_cov = []
    for c in cands_of:
        t2 = time.perf_counter()
        rows_cov.append(allocate_rows(c, "coverage", DEFAULT_N_FLAT, _plan())[:MAX_ROWS])
        do_tre.append(time.perf_counter() - t2)
    rows_nen = nap_dong(cache_dir, f"ship_nen_{len(gt)}", args.refresh,
                        lambda: dong_ship(cands_of, "hybrid"))
    do_tre = np.array(do_tre)
    print(f"  coverage: {do_tre.mean()*1000:.0f} ms/câu trung bình, "
          f"p95 {np.percentile(do_tre, 95)*1000:.0f} ms, max {do_tre.max()*1000:.0f} ms "
          f"(tổng {len(gt)} câu: {time.time()-t1:.1f}s)")

    thieu = [i for i, r in enumerate(rows_cov) if len(r) < MAX_ROWS and len(cands_of[i]) >= 40]
    if thieu:
        print(f"  ! {len(thieu)} câu pool đầy đủ mà < {MAX_ROWS} dòng — đuôi lấp hỏng: {thieu}")
        return 1

    # ---- chấm theo luật harness: hai nửa, họ hạt giống TEST tách rời --------
    print("=== 3) Chấm (họ hạt giống TEST gốc 90000 — chưa từng dùng để chọn gì) ===",
          flush=True)
    ket = {}
    for phia, du in (("chan", 0), ("le", 1)):
        idx = [i for i in range(len(gt)) if i % 2 == du]
        gt_h = [gt[i] for i in idx]
        nen_h = [rows_nen[i] for i in idx]
        cov_h = [rows_cov[i] for i in idx]
        ho = cac_lan_boc(GOC_TEST, args.seeds, args.draws, gt_h, kf)

        if phia == "chan":
            _doi_chieu_bo_cham(nen_h[:5], gt_h[:5], [t[:3] for t in ho[0][:5]], windows)
            print("  bộ chấm vector hoá khớp tuyệt đối bản sản xuất (5 câu × 3 bốc).")

        harness = {"build": "ship-v1", "windows": windows, "seeds": args.seeds,
                   "draws": args.draws, "goc": GOC_TEST, "phia": phia, "n_cau": len(gt_h)}
        cache = CacheDiem(cache_dir / f"diem_ship_{phia}.json", harness, args.refresh)
        nen_ho = cache.lay("nen", lambda: [
            cham_nhanh(ma_tran_dong(nen_h, gt_h), draws, windows) for draws in ho])
        cov_ho = cache.lay("coverage", lambda: [
            cham_nhanh(ma_tran_dong(cov_h, gt_h), draws, windows) for draws in ho])
        ket[phia] = (idx, gt_h, nen_h, cov_h, ho,
                     float(np.mean(nen_ho)), float(np.std(nen_ho)),
                     float(np.mean(cov_ho)), float(np.std(cov_ho)))

    # ---- bảng từng câu ------------------------------------------------------
    print("\n=== 4) Từng câu (điểm TB họ×bốc×cửa sổ; Δ = coverage − hybrid) ===")
    print(f"{'câu':>4} {'nửa':>5} {'hybrid':>8} {'coverage':>9} {'Δ':>8}  dòng-1")
    doi_dong1 = 0
    for phia in ("chan", "le"):
        idx, gt_h, nen_h, cov_h, ho, *_ = ket[phia]
        d_nen = diem_tung_cau(nen_h, gt_h, ho, windows)
        d_cov = diem_tung_cau(cov_h, gt_h, ho, windows)
        for k, i in enumerate(idx):
            v_nen = nen_h[k][0][0] if nen_h[k] else "?"
            v_cov = cov_h[k][0][0] if cov_h[k] else "?"
            doi = "" if v_nen == v_cov else f"{v_nen} -> {v_cov}"
            doi_dong1 += bool(doi)
            delta = d_cov[k] - d_nen[k]
            danh_dau = "" if abs(delta) < 0.02 else ("  <<<" if delta < 0 else "  +++")
            print(f"{i:>4} {phia:>5} {d_nen[k]:8.4f} {d_cov[k]:9.4f} {delta:+8.4f}  {doi}{danh_dau}")
    print(f"  video dòng 1 đổi ở {doi_dong1}/{len(gt)} câu")

    # ---- kết luận theo luật 2 sigma -----------------------------------------
    print("\n=== KẾT LUẬN (bản ship, từng nửa) ===")
    dau_hong = False
    for phia in ("chan", "le"):
        _i, _g, _n, _c, _h, nen_m, nen_sd, cov_m, cov_sd = ket[phia]
        up = 100 * (cov_m / nen_m - 1)
        bien = max(nen_sd, 0.0005)
        if (cov_m - nen_m) < 2 * bien:
            loi = f"HOÀ (chênh {cov_m-nen_m:.4f} < 2σ={2*bien:.4f})"
            dau_hong = True
        elif up < 5.0:
            loi = f"YẾU: chỉ {up:+.1f}% (< +5%)"
            dau_hong = True
        else:
            loi = f"GIỮ ĐƯỢC: {up:+.1f}% (>= +5%, > 2σ)"
        print(f"  nửa {phia:4s}: nền {nen_m:.4f} ±{nen_sd:.4f}  "
              f"coverage {cov_m:.4f} ±{cov_sd:.4f}  -> {loi}")
    if dau_hong:
        print("\n  => CHƯA đủ điều kiện đổi mặc định. Coverage ở lại dạng cờ opt-in;")
        print("     đối chiếu với số thí nghiệm trong docs/SHIP_PHU_XAC_SUAT.md để tìm khác biệt.")
    else:
        print("\n  => Cổng bước 3 XANH trên bản ship: được phép đổi mặc định make_submission")
        print("     sang coverage (ghi số này vào docs/SHIP_PHU_XAC_SUAT.md trước).")
    print(f"\nXong sau {time.time()-t0:.0f}s. Cache: {cache_dir}")
    return 1 if dau_hong else 0


# ---------------------------------------------------------------------------
# Chế độ 2: diff cấu trúc trên đề thật (không có GT)
# ---------------------------------------------------------------------------


def diff_de_that(args) -> int:
    from scripts.make_submission import (
        detect_task,
        ranked_hits,
        read_en_override,
        read_query_text,
        split_qa,
    )
    from src.core.kis_engine import KISEngine
    from src.core.submission import write_query_csv

    qdir = Path(args.queries)
    qfiles = sorted(p for p in qdir.glob("*.txt")
                    if not p.name.lower().endswith((".en.txt", ".vi.txt")))
    if not qfiles:
        print(f"ERROR: không có .txt nào trong {qdir}")
        return 2

    print(f"{len(qfiles)} câu; nạp chỉ mục một lần ...", flush=True)
    eng = KISEngine(args.data).load()
    out = Path(args.out)

    print(f"\n{'câu':<28}{'dòng-1 hybrid':>16}{'dòng-1 coverage':>17}"
          f"{'video phủ':>11}{'dòng video top':>15}")
    print("-" * 88)
    giong = 0
    n_kis = 0
    for qf in qfiles:
        task = detect_task(qf.name)
        if task == "trake":
            continue  # TRAKE giữ nguyên đường cũ, ngoài phạm vi allocator
        n_kis += 1
        text = read_query_text(qf) or ""
        if task == "qa":
            probe = split_qa(text)[0] or text
        else:
            probe = text
        hits = ranked_hits(eng, probe, read_en_override(qf))
        cands = [Candidate(h.video_id, h.frame_idx, h.score, h.video_last_frame) for h in hits]

        hai_bo = {}
        for alloc in ("hybrid", "coverage"):
            rows = allocate_rows(cands, alloc, DEFAULT_N_FLAT, _plan())[:MAX_ROWS]
            hai_bo[alloc] = rows
            d = out / alloc / "csv"
            write_query_csv(d / (qf.stem + ".csv"), rows)

        h, c = hai_bo["hybrid"], hai_bo["coverage"]
        v_h = h[0][0] if h else "?"
        v_c = c[0][0] if c else "?"
        giong += v_h == v_c
        top_v = v_c
        print(f"{qf.stem:<28}{v_h + ':' + str(h[0][1]):>16}{v_c + ':' + str(c[0][1]):>17}"
              f"{len({v for v, _ in c}):>11}{sum(1 for v, _ in c if v == top_v):>15}"
              f"{'' if v_h == v_c else '   ĐỔI VIDEO'}")

    print(f"\nvideo dòng 1 giống nhau: {giong}/{n_kis} câu KIS/QA")
    print(f"hai bộ csv nằm cạnh nhau trong {out}\\hybrid\\csv và {out}\\coverage\\csv")
    print("nhớ: đây là diff CẤU TRÚC — không nói bên nào đúng, chỉ nói coverage đổi gì.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache_phu_quet_luoi"))
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--draws", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--queries", default=None, help="thư mục đề thật -> chế độ diff cấu trúc")
    ap.add_argument("--out", default=None, help="nơi ghi hai bộ csv (chế độ diff)")
    args = ap.parse_args()

    if args.queries:
        if not args.out:
            print("ERROR: chế độ --queries cần --out")
            return 2
        return diff_de_that(args)
    return cong_gt(args)


if __name__ == "__main__":
    raise SystemExit(main())
