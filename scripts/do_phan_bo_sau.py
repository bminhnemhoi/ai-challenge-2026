"""RỘNG vs SÂU — mở lại cánh bậc của bộ phân bổ, trên bộ đo khớp phân bố đề thật.

Bảng tín hiệu có một dòng đóng cửa việc này: "chia lại ngân sách theo (video,
keyframe, độ sâu) | ÂM 15–30% | KHÔNG DÙNG". Dòng đó đo trên bộ 60 câu CŨ, nơi
SigLIP đã đặt keyframe đáp án ở hạng nội-video trung vị 1,0 — tức đúng chỗ rồi,
độ sâu không mua được gì và MỌI cấu hình đều chỉ có thể làm tệ đi.

Trên bộ đo mới, oracle định vị nội-video nói +126% (``tran_dinh_vi_noi_video.py``).
Oracle đó KHÔNG đổi việc chọn video và KHÔNG đổi thứ hạng — nó chỉ đặt lại frame
id thành một thang DÀY quanh khoảnh khắc thật. Nghĩa là một phần lớn headroom nằm
đúng ở thứ mà tham số phân bổ điều khiển: **rải bao nhiêu dòng, rải rộng ra bao
nhiêu video, và rải dày cỡ nào quanh mỗi keyframe**.

Script này làm ba việc, theo đúng thứ tự:

1. **Chẩn đoán rẻ, tất định** — trong 100 dòng hiện tại, bao nhiêu dòng rơi vào
   video ĐÚNG, và bao nhiêu trong số đó nằm trong ±50/±100/±200 frame quanh
   khoảnh khắc thật. Tách riêng nhóm một cảnh / hai cảnh. Đây là phép đếm, không
   có khoảng tin cậy nào để bàn.
2. **Quét lưới** cả hai họ phân bổ trên bộ đo mới: CoveragePlan (nhiệt × sigma ×
   nửa_cửa_sổ × lưới = 200 tổ hợp) và hybrid (n_flat × depth_cost = 16 tổ hợp).
3. **TUNE/TEST phân tầng** theo trục bị tác động (một cảnh / hai cảnh), chọn trên
   TUNE, đọc TEST đúng một lần, bootstrap THEO CÂU, báo cáo riêng nhóm hai cảnh.
4. Nếu tham số mới thắng: **kiểm lại trên bộ 60 câu CŨ**. Nếu nó làm tệ đi ở đó
   thì đây là đánh đổi giữa hai phân bố, phải nói ra chứ không được giấu.

Bất biến: đường sinh dòng ở đây phải cho ra ĐÚNG các dòng của
``make_submission.allocate_rows`` khi truyền tham số mặc định — kiểm bằng assert
trên toàn bộ 132 mục trước khi quét. Không có cổng bật/tắt nào ở lane này: tham
số phân bổ tác động lên MỌI câu, nên bất biến phải là bất biến trùng-khớp-nền.

    python -u scripts/do_phan_bo_sau.py
    python -u scripts/do_phan_bo_sau.py --refresh     # bỏ cache lưới, tính lại
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.experiment_phu_quet_luoi import (  # noqa: E402
    cac_lan_boc,
    cham_nhanh,
    ma_tran_dong,
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
    CoveragePlan,
    allocate_coverage_rows,
    allocate_hybrid_rows,
)

# --------------------------------------------------------------------------
# Lưới tham số và gốc hạt giống (tách khỏi mọi gốc đã dùng ở lane khác)
# --------------------------------------------------------------------------

LUOI_NHIET = (0.01, 0.015, 0.02, 0.03, 0.05)
LUOI_SIGMA = (15.0, 20.0, 30.0, 45.0, 60.0)
LUOI_NUA = (6, 10, 15, 25)
LUOI_BUOC = (5, 10)

LUOI_NFLAT = (10, 20, 30, 50)
LUOI_DEPTH = (0.25, 0.5, 0.75, 1.0)

GOC_TUNE_PB = 310000
GOC_TEST_PB = 320000
GOC_CU_PB = 330000  # bộ 60 câu CŨ, kiểm chéo — tách hẳn khỏi 50000/90000 đã dùng

NEN_CP = (0.02, 30.0, 6, 5)  # tham số sản xuất hiện tại
NEN_HY = (DEFAULT_N_FLAT, DEFAULT_DEPTH_COST)


def _plan(depth_cost: float = DEFAULT_DEPTH_COST) -> AllocationPlan:
    return AllocationPlan(breadth_cost=1.0, depth_cost=depth_cost, step=10)


def sinh_dong(cands, ho: str, tham):
    """Đúng thân của ``make_submission.allocate_rows``, chỉ mở tham số ra để quét.

    Nhánh ``coverage`` gọi ``allocate_coverage_rows`` với đúng các đối số sản
    xuất (tail_n_flat = DEFAULT_N_FLAT, tail_plan = plan hybrid), khác duy nhất
    ở chỗ CoveragePlan mang tham số đang quét thay vì mặc định. Nhánh ``hybrid``
    gọi thẳng ``allocate_hybrid_rows`` — đúng hàm sản xuất, không có bản sao.
    """
    if ho == "coverage":
        nhiet, sigma, nua, luoi = tham
        return allocate_coverage_rows(
            cands,
            plan=CoveragePlan(nhiet=nhiet, sigma=sigma, nua_cua_so=nua, luoi=luoi,
                              budget=MAX_ROWS),
            tail_n_flat=DEFAULT_N_FLAT,
            tail_plan=_plan(),
        )[:MAX_ROWS]
    n_flat, depth = tham
    return allocate_hybrid_rows(cands, n_flat=n_flat, plan=_plan(depth))[:MAX_ROWS]


def cham_tung_cau(rows_sub, gt_sub, ho, windows):
    """(điểm trung bình từng CÂU, điểm trung bình từng HỌ hạt giống)."""
    mats = ma_tran_dong(rows_sub, gt_sub)
    per_q = np.zeros(len(gt_sub))
    per_ho = []
    for draws in ho:
        per_ho.append(cham_nhanh(mats, draws, windows))
        for q in range(len(gt_sub)):
            per_q[q] += cham_nhanh([mats[q]], [draws[q]], windows)
    return per_q / len(ho), np.array(per_ho)


# --------------------------------------------------------------------------
# Worker song song — mỗi tiến trình tự nạp ứng viên một lần
# --------------------------------------------------------------------------

_W: dict = {}


def _khoi_tao(uv_path, gt_path, idx_tune, windows, goc, so_ho, so_boc, kf_raw):
    moi = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    uv = json.loads(Path(uv_path).read_text(encoding="utf-8"))
    giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]
    gt_sub = [moi[giu[i]] for i in idx_tune]
    # chỉ giữ ứng viên của các câu TUNE — 9 tiến trình mỗi cái ôm cả 148 câu là
    # cách nhanh nhất để bị OOM giết ngang, và pool chết thì mất sạch lưới
    _W["cands"] = [[Candidate(v, f, s, lf) for v, f, s, lf in uv[giu[i]]] for i in idx_tune]
    del uv, moi
    kf = {v: np.array(a, dtype=np.int64) for v, a in kf_raw.items()}
    _W["gt"] = gt_sub
    _W["ho"] = cac_lan_boc(goc, so_ho, so_boc, gt_sub, kf)
    _W["windows"] = windows


def _chay(job):
    ho, tham = job
    rows = [sinh_dong(c, ho, tham) for c in _W["cands"]]
    per_q, per_ho = cham_tung_cau(rows, _W["gt"], _W["ho"], _W["windows"])
    return ho, tham, per_q.tolist(), float(per_ho.mean()), float(per_ho.std())


# --------------------------------------------------------------------------


def chan_doan(rows_of, gt_sub, ten, out):
    """Đếm tất định: 100 dòng rơi vào đâu so với khoảnh khắc thật."""
    n = len(gt_sub)
    co_video, trong = 0, {50: 0, 100: 0, 200: 0}
    tong_dong_video, tong_dong_gan = 0.0, {50: 0.0, 100: 0.0, 200: 0.0}
    hang_dau, so_video_100, so_video_20 = [], [], []
    trai_rong = []
    for rows, g in zip(rows_of, gt_sub):
        vid, fr = g["video_id"], int(g["frame_idx"])
        dung = [f for v, f in rows if v == vid]
        tong_dong_video += len(dung)
        so_video_100.append(len({v for v, _ in rows}))
        so_video_20.append(len({v for v, _ in rows[:20]}))
        if dung:
            co_video += 1
            trai_rong.append(max(dung) - min(dung))
            for k, (v, f) in enumerate(rows, 1):
                if v == vid:
                    hang_dau.append(k)
                    break
        for w in (50, 100, 200):
            c = sum(1 for f in dung if abs(f - fr) <= w)
            tong_dong_gan[w] += c
            if c:
                trong[w] += 1
    out.append(
        f"{ten:<16}{n:>4}{co_video:>10}{tong_dong_video/n:>10.1f}"
        + "".join(f"{trong[w]:>7}{tong_dong_gan[w]/n:>8.1f}" for w in (50, 100, 200))
        + f"{np.median(hang_dau) if hang_dau else float('nan'):>8.1f}"
        f"{np.mean(so_video_100):>9.1f}{np.mean(so_video_20):>7.1f}"
        f"{np.median(trai_rong) if trai_rong else float('nan'):>10.0f}"
    )
    return {
        "n": n, "co_video": co_video,
        "dong_video_tb": tong_dong_video / n,
        "cau_co_dong_gan": {str(w): trong[w] for w in (50, 100, 200)},
        "dong_gan_tb": {str(w): tong_dong_gan[w] / n for w in (50, 100, 200)},
    }


def boot(dn, dc, rng, nlan=4000):
    m = len(dn)
    lay = rng.integers(0, m, size=(nlan, m))
    d = dc[lay].mean(axis=1) - dn[lay].mean(axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi), float((d <= 0).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache_bo_do_moi"))
    ap.add_argument("--cu-cache", default=str(ROOT / "data" / "cache_phu_quet_luoi"))
    ap.add_argument("--ket", default=str(ROOT / "data" / "cache_phan_bo_sau"))
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--tune-seeds", type=int, default=3)
    ap.add_argument("--tune-draws", type=int, default=32)
    ap.add_argument("--test-seeds", type=int, default=4)
    ap.add_argument("--test-draws", type=int, default=48)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--refresh", action="store_true")
    # máy này có lane khác chiếm ~2,4 GB; tiến trình nền bị giết ngang hai lần
    # giữa chừng. Nên lưới chạy ĐƯỢC CẮT LÁT và cache ghi sau mỗi 10 tổ hợp:
    # --chi-quet N làm tối đa N tổ hợp còn thiếu rồi thoát, gọi lại là chạy tiếp.
    ap.add_argument("--chi-quet", type=int, default=0,
                    help="chi quet toi da N to hop con thieu roi thoat (0 = chay du)")
    args = ap.parse_args()

    data = Path(args.data)
    windows = [int(w) for w in args.windows.split(",")]
    ket_dir = Path(args.ket)
    ket_dir.mkdir(parents=True, exist_ok=True)

    moi = json.loads(Path(args.moi).read_text(encoding="utf-8"))
    uv_path = Path(args.cache) / "uv_moi.json"
    uv = json.loads(uv_path.read_text(encoding="utf-8"))
    assert len(uv) == len(moi), "cache ứng viên lệch số mục — chạy lại do_bo_do_moi.py"

    giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]
    sach = [moi[i] for i in giu]
    cands = [[Candidate(v, f, s, lf) for v, f, s, lf in uv[i]] for i in giu]
    hai = [i for i, g in enumerate(sach) if g.get("co_2_canh")]
    mot = [i for i, g in enumerate(sach) if not g.get("co_2_canh")]
    print(f"bo SACH {len(sach)} muc | MOT canh {len(mot)} | HAI canh {len(hai)}")

    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    kf_list: dict = {}
    for m in meta:
        kf_list.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
    kf_raw = {v: sorted(a) for v, a in kf_list.items()}
    kf = {v: np.array(a, dtype=np.int64) for v, a in kf_raw.items()}
    del meta, kf_list

    # ---- BẤT BIẾN: đường sinh dòng ở đây == allocate_rows sản xuất ----------
    rows_nen = [sinh_dong(c, "coverage", NEN_CP) for c in cands]
    if not args.chi_quet:
        for i, c in enumerate(cands):
            b = allocate_rows(c, "coverage", DEFAULT_N_FLAT, _plan())[:MAX_ROWS]
            assert rows_nen[i] == b, f"muc {i}: duong sinh dong lech allocate_rows san xuat"
        print(f"bat bien OK: {len(cands)} muc, sinh_dong(coverage, mac dinh) == allocate_rows()")

    # ================= 1. CHẨN ĐOÁN =========================================
    cd = {}
    if args.chi_quet:
        print("(che do cat lat: bo qua chan doan + TEST, chi chay tiep luoi TUNE)")
    else:
        print("\n" + "=" * 108)
        print("1. CHAN DOAN — 100 dong hien tai roi vao dau "
              "(dem tat dinh, khong co nhieu thong ke)")
        print("=" * 108)
        print(f"{'nhom':<16}{'n':>4}{'co video':>10}{'dong/video':>10}"
              f"{'q<=50':>7}{'dong<=50':>8}{'q<=100':>7}{'d<=100':>8}{'q<=200':>7}{'d<=200':>8}"
              f"{'hang1':>8}{'nvideo':>9}{'top20':>7}{'trairong':>10}")
        print("-" * 108)
        out: list = []
        cd["tat_ca"] = chan_doan(rows_nen, sach, "bo SACH", out)
        print(out[-1])
        cd["mot"] = chan_doan([rows_nen[i] for i in mot], [sach[i] for i in mot],
                              "  |- MOT canh", out)
        print(out[-1])
        cd["hai"] = chan_doan([rows_nen[i] for i in hai], [sach[i] for i in hai],
                              "  |- HAI canh", out)
        print(out[-1])
        print("\n  co video  = so CAU co it nhat 1 dong roi vao video dung")
        print("  dong/video= so DONG trung binh roi vao video dung (tren 100 dong)")
        print("  q<=W      = so CAU co it nhat 1 dong trong +-W frame quanh khoanh khac that")
        print("  dong<=W   = so DONG trung binh nam trong +-W frame quanh khoanh khac that")
        print("  hang1     = hang trung vi cua dong dau tien thuoc video dung")
        print("  nvideo    = so video khac nhau trong 100 dong; top20 = trong 20 dong dau")
        print("  trairong  = be rong (frame) tu dong dau den dong cuoi trong video dung, trung vi")

    # ================= 2. CHIA TUNE/TEST PHÂN TẦNG ==========================
    i_tune = sorted(hai[0::2] + mot[0::2])
    i_test = sorted(hai[1::2] + mot[1::2])
    assert not (set(i_tune) & set(i_test)) and len(i_tune) + len(i_test) == len(sach)
    n_hai_tune = sum(1 for i in i_tune if i in set(hai))
    n_hai_test = sum(1 for i in i_test if i in set(hai))
    print(f"\nTUNE {len(i_tune)} muc ({n_hai_tune} hai canh) | "
          f"TEST {len(i_test)} muc ({n_hai_test} hai canh)  — phan tang theo truc bi tac dong")

    # ================= 3. QUÉT LƯỚI TRÊN TUNE ===============================
    jobs = [("coverage", (n, s, w, l)) for n in LUOI_NHIET for s in LUOI_SIGMA
            for w in LUOI_NUA for l in LUOI_BUOC]
    jobs += [("hybrid", (nf, dc)) for nf in LUOI_NFLAT for dc in LUOI_DEPTH]
    f_cache = ket_dir / "tune.json"
    if f_cache.is_file() and not args.refresh:
        raw = json.loads(f_cache.read_text(encoding="utf-8"))
        kq = {(r["ho"], tuple(r["tham"])): r for r in raw}
        print(f"\ncache luoi TUNE: {f_cache} ({len(kq)} to hop)")
    else:
        kq = {}
    thieu = [j for j in jobs if (j[0], tuple(j[1])) not in kq]
    con_lai = len(thieu)
    if args.chi_quet:
        thieu = thieu[: args.chi_quet]
    if thieu:
        import multiprocessing as mp
        print(f"\nquet {len(thieu)} to hop tren TUNE, {args.workers} tien trinh ...", flush=True)
        t0 = time.time()
        with mp.Pool(args.workers, initializer=_khoi_tao,
                     initargs=(str(uv_path), args.moi, i_tune, windows,
                               GOC_TUNE_PB, args.tune_seeds, args.tune_draws, kf_raw)) as pool:
            for k, (ho, tham, per_q, m, sd) in enumerate(pool.imap_unordered(_chay, thieu), 1):
                kq[(ho, tuple(tham))] = {"ho": ho, "tham": list(tham), "per_q": per_q,
                                         "diem": m, "sd": sd}
                # ghi sau MỖI kết quả: một lần bị giết ngang (OOM) đã mất trọn
                # 10 phút lưới vì cache chỉ ghi ở cuối. Không lặp lại.
                if k % 10 == 0 or k == len(thieu):
                    f_cache.write_text(json.dumps(list(kq.values())), encoding="utf-8")
                    print(f"  {k}/{len(thieu)}  ({time.time()-t0:.0f}s)", flush=True)
        f_cache.write_text(json.dumps(list(kq.values())), encoding="utf-8")

    if args.chi_quet:
        print(f"\nlat xong. Con thieu {con_lai - len(thieu)}/{len(jobs)} to hop — "
              "goi lai lenh nay de chay tiep, hoac bo --chi-quet de chay du.")
        return 0

    nen_key = ("coverage", NEN_CP)
    m_nen = kq[nen_key]["diem"]
    hai_t = set(hai)
    vt_hai_tune = [k for k, i in enumerate(i_tune) if i in hai_t]

    def diem_hai(r):
        return float(np.array(r["per_q"])[vt_hai_tune].mean())

    print(f"\nNEN (coverage {NEN_CP[0]:g}/{NEN_CP[1]:g}/{NEN_CP[2]}/{NEN_CP[3]}) "
          f"tren TUNE: {m_nen:.4f} +-{kq[nen_key]['sd']:.4f} | "
          f"nhom HAI canh {diem_hai(kq[nen_key]):.4f}")

    for ho, ten in (("coverage", "CoveragePlan"), ("hybrid", "hybrid")):
        r = sorted([v for v in kq.values() if v["ho"] == ho],
                   key=lambda v: -v["diem"])
        print(f"\n--- {ten}: 12 to hop TOT nhat tren TUNE (tren {len(r)}) ---")
        print(f"{'tham so':<28}{'TUNE':>9}{'+-':>8}{'so nen':>9}{'HAI canh':>10}{'so nen':>9}")
        nen_hai = diem_hai(kq[nen_key])
        for v in r[:12]:
            th = v["tham"]
            ten_th = (f"nhiet={th[0]:g} sig={th[1]:g} nua={th[2]} luoi={th[3]}"
                      if ho == "coverage" else f"n_flat={th[0]} depth={th[1]:g}")
            dh = diem_hai(v)
            print(f"{ten_th:<28}{v['diem']:>9.4f}{v['sd']:>8.4f}"
                  f"{100*(v['diem']/m_nen-1):>+8.1f}%{dh:>10.4f}{100*(dh/nen_hai-1):>+8.1f}%")
        print(f"    ... {len(r)-12} to hop con lai thap hon" if len(r) > 12 else "")

    # chốt: tổ hợp cao nhất trên TUNE, mỗi họ một
    chot_cp = max((v for v in kq.values() if v["ho"] == "coverage"), key=lambda v: v["diem"])
    chot_hy = max((v for v in kq.values() if v["ho"] == "hybrid"), key=lambda v: v["diem"])
    chot = chot_cp if chot_cp["diem"] >= chot_hy["diem"] else chot_hy
    print(f"\nCHOT tren TUNE: ho={chot['ho']} tham={chot['tham']} "
          f"({100*(chot['diem']/m_nen-1):+.1f}% so nen)")

    # ================= 4. TEST — ĐỌC ĐÚNG MỘT LẦN ===========================
    print("\n" + "=" * 84)
    print("2. TEST — doc DUNG MOT LAN")
    print("=" * 84)
    gt_test = [sach[i] for i in i_test]
    ho_test = cac_lan_boc(GOC_TEST_PB, args.test_seeds, args.test_draws, gt_test, kf)
    r_nen = [rows_nen[i] for i in i_test]
    dn, hn = cham_tung_cau(r_nen, gt_test, ho_test, windows)

    rng = np.random.default_rng(90210)
    vt_hai = [k for k, i in enumerate(i_test) if i in hai_t]
    vt_mot = [k for k, i in enumerate(i_test) if i not in hai_t]

    def bao_cao(ten, dc_, hc):
        print(f"\n{ten}")
        print(f"  toan bo TEST ({len(gt_test)}): {dn.mean():.4f} -> {dc_.mean():.4f} "
              f"= {100*(dc_.mean()/dn.mean()-1):+.1f}%   (sigma hat giong "
              f"nen {hn.std():.4f} / moi {hc.std():.4f})")
        lo, hi, p = boot(dn, dc_, rng)
        print(f"    bootstrap theo CAU: KTC 95% [{lo:+.4f}, {hi:+.4f}]  P(<=0) = {p:.1%}")
        for ten_n, vt in (("HAI canh", vt_hai), ("MOT canh", vt_mot)):
            a, b = dn[vt], dc_[vt]
            lo, hi, p = boot(a, b, rng)
            print(f"  {ten_n} ({len(vt)}): {a.mean():.4f} -> {b.mean():.4f} "
                  f"= {100*(b.mean()/a.mean()-1):+.1f}%; "
                  f"KTC 95% [{lo:+.4f}, {hi:+.4f}]  P(<=0) = {p:.1%}")
        return lo, hi, p

    kq_test = {}
    for nhan, v in (("CHOT CoveragePlan", chot_cp), ("CHOT hybrid", chot_hy)):
        tham = tuple(v["tham"])
        r = [sinh_dong(cands[i], v["ho"], tham) for i in i_test]
        dc_, hc = cham_tung_cau(r, gt_test, ho_test, windows)
        ten = f"{nhan}  {v['ho']} {tham}"
        bao_cao(ten, dc_, hc)
        kq_test[nhan] = (v, dc_, hc)

    # ================= 5. KIỂM CHÉO TRÊN BỘ 60 CÂU CŨ =======================
    print("\n" + "=" * 84)
    print("3. KIEM CHEO tren bo 60 cau CU (phan bo khac han) — doc mot lan")
    print("=" * 84)
    cu_raw = json.loads((Path(args.cu_cache) / "ung_vien.json").read_text(encoding="utf-8"))
    gt_cu = cu_raw["gt"]
    cands_cu = [[Candidate(v, f, s, lf) for v, f, s, lf in q] for q in cu_raw["cands"]]
    kf_cu = {v: np.array(a, dtype=np.int64) for v, a in cu_raw["kf"].items()}
    ho_cu = cac_lan_boc(GOC_CU_PB, args.test_seeds, args.test_draws, gt_cu, kf_cu)
    r0 = [sinh_dong(c, "coverage", NEN_CP) for c in cands_cu]
    d0, h0 = cham_tung_cau(r0, gt_cu, ho_cu, windows)
    print(f"  nen (san xuat)      : {d0.mean():.4f} +-{h0.std():.4f}")
    for nhan, (v, _dc, _hc) in kq_test.items():
        tham = tuple(v["tham"])
        r = [sinh_dong(c, v["ho"], tham) for c in cands_cu]
        d1, h1 = cham_tung_cau(r, gt_cu, ho_cu, windows)
        lo, hi, p = boot(d0, d1, rng)
        print(f"  {nhan:<20}: {d1.mean():.4f} +-{h1.std():.4f}  "
              f"({100*(d1.mean()/d0.mean()-1):+.1f}%)  "
              f"KTC 95% [{lo:+.4f}, {hi:+.4f}]  P(<=0) = {p:.1%}")

    json.dump({"chan_doan": cd}, open(ket_dir / "chan_doan.json", "w"), indent=1)
    print(f"\n(chan doan da ghi: {ket_dir / 'chan_doan.json'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
