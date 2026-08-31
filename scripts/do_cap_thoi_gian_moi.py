"""Lever ③ — truy vấn CẶP THỜI GIAN, đo trên bộ đo KHỚP PHÂN BỐ.

Lần đo trước kết luận **CHƯA ĐO ĐƯỢC**, và lý do đắt hơn phép đo: cổng "câu này
mô tả hai cảnh nối tiếp" bật **0/60** trên bộ ground truth cũ, nên mọi cấu hình
là phép đồng nhất — không TUNE, không TEST, không luật 2σ nào áp được.

`data/ground_truth_moi.json` sửa đúng chỗ đó: câu sinh từ ĐOẠN video thật, 50%
có cấu trúc hai cảnh (đề thật BTC: 51%), khung neo đã xác minh bằng một-ảnh-một-
request (`docs/BO_DO_KHOP_PHAN_BO.md`). Trên bộ SẠCH 48 mục, cổng bật **24/48**.

Và ta đã biết nhóm đó là nhóm yếu nhất: câu hai cảnh đạt 0,1213 so với 0,2158
của câu một cảnh (**−43,8%**, vượt 2σ), với chất lượng mô tả hai nhóm y hệt nhau.
Đây chính là chỗ lever ③ được thiết kế để chữa.

Kỷ luật giữ nguyên như mọi phép đo đã ship:
  * chia TUNE/TEST 24/24 theo chỉ số chẵn/lẻ TRONG bộ sạch, chọn trên TUNE, đọc
    TEST đúng một lần, luật hoà 2σ;
  * chấm qua ``allocate_rows`` THẬT của make_submission, không chấm tắt;
  * **bất biến**: câu cổng TẮT phải ra 100 dòng giống hệt nền — assert, không
    nhìn bằng mắt;
  * shard c bị lẫn trục (hai cảnh ≡ một dải video) nên bị loại khỏi mọi phép so.

    python -u scripts/do_cap_thoi_gian_moi.py
    python -u scripts/do_cap_thoi_gian_moi.py --allocator hybrid   # đối chứng
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

from scripts.do_bo_do_moi import GOC_HAT  # noqa: E402,F401
from scripts.experiment_cap_thoi_gian import (  # noqa: E402
    LUOI_GOP,
    LUOI_LAMBDA,
    LUOI_W,
    KhoSims,
    _plan,
    ap_dung,
    nap_truc_video,
)
from scripts.experiment_phu_quet_luoi import cac_lan_boc, cham_nhanh, ma_tran_dong  # noqa: E402
from scripts.make_submission import DEFAULT_N_FLAT, allocate_rows  # noqa: E402
from src.core.submission import MAX_ROWS, Candidate  # noqa: E402

GOC_TUNE_MOI = 61000
GOC_TEST_MOI = 62000


def canh_cua(muc: dict):
    """(cảnh_A_vi, cảnh_A_en, cảnh_B_vi, cảnh_B_en) hoặc None nếu chưa tách được.

    Bước sinh của shard a/d ghi sẵn hai mệnh đề; shard b chỉ khai ``co_2_canh``
    mà không tách, nên những mục đó không có gì để đối chiếu với ảnh và bị coi
    là **cổng tắt** ở đây — thà bỏ sót còn hơn bịa ra hai mệnh đề rồi đo chính
    thứ mình bịa.
    """
    if not muc.get("co_2_canh"):
        return None

    def lay(x):
        if isinstance(x, dict):
            return x.get("vi") or x.get("mo_ta") or "", x.get("en") or ""
        return (str(x) if x else ""), ""

    a_vi, a_en = lay(muc.get("canh_A") or muc.get("canh_A_vi"))
    b_vi, b_en = lay(muc.get("canh_B") or muc.get("canh_B_vi"))
    if not muc.get("canh_A") and muc.get("canh_A_vi"):
        a_vi, a_en = muc.get("canh_A_vi", ""), muc.get("canh_A_en", "")
        b_vi, b_en = muc.get("canh_B_vi", ""), muc.get("canh_B_en", "")
    if not (a_vi.strip() and b_vi.strip()):
        return None
    return a_vi, a_en, b_vi, b_en


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache_bo_do_moi"))
    ap.add_argument("--allocator", default="coverage", choices=("coverage", "hybrid"))
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--tune-seeds", type=int, default=3)
    ap.add_argument("--tune-draws", type=int, default=32)
    ap.add_argument("--test-seeds", type=int, default=4)
    ap.add_argument("--test-draws", type=int, default=48)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    data = Path(args.data)
    windows = [int(w) for w in args.windows.split(",")]
    t0 = time.time()

    moi = json.loads(Path(args.moi).read_text(encoding="utf-8"))
    sach = [m for m in moi if not m.get("lan_truc")]
    print(f"bộ mới: {len(moi)} mục, bộ SẠCH (bỏ shard c lẫn trục): {len(sach)}")

    uv_f = Path(args.cache) / "uv_moi.json"
    if not uv_f.is_file():
        print(f"ERROR: chưa có {uv_f} — chạy scripts/do_bo_do_moi.py trước")
        return 2
    uv_all = json.loads(uv_f.read_text(encoding="utf-8"))
    if len(uv_all) != len(moi):
        print(f"ERROR: cache ứng viên {len(uv_all)} != {len(moi)} mục — chạy lại do_bo_do_moi.py")
        return 2
    giu = [i for i, m in enumerate(moi) if not m.get("lan_truc")]
    cands_of = [[Candidate(v, fi, s, lf) for v, fi, s, lf in uv_all[i]] for i in giu]

    # nhãn theo đúng định dạng ap_dung() mong đợi
    nhan = []
    for m in sach:
        c = canh_cua(m)
        nhan.append({"co_2_canh": bool(c), "canh_A_vi": c[0] if c else "",
                     "canh_A_en": c[1] if c else "", "canh_B_vi": c[2] if c else "",
                     "canh_B_en": c[3] if c else ""} if True else None)
    bat = [i for i, d in enumerate(nhan) if d["co_2_canh"]]
    khai = sum(1 for m in sach if m.get("co_2_canh"))
    print(f"cổng BẬT {len(bat)}/{len(sach)} mục "
          f"(tự khai hai cảnh: {khai}; {khai - len(bat)} mục thiếu mệnh đề A/B nên coi như tắt)")
    if len(bat) < 8:
        print("!! quá ít mục qua cổng — không đủ lực thống kê, dừng.")
        return 1

    # Bản đồ keyframe dựng thẳng từ metadata.json, KHÔNG nạp KISEngine: KhoSims
    # đã giữ một bản chỉ mục SigLIP (177k × 1152 float32 ≈ 780 MB) và nạp thêm
    # một bản nữa làm tiến trình chết vì hết bộ nhớ (segfault, không phải lỗi
    # Python nên không có traceback để lần).
    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    kf_list: dict = {}
    for m in meta:
        kf_list.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
    kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in kf_list.items()}
    del meta, kf_list

    print("nạp chỉ mục (một lần, qua KhoSims) ...", flush=True)
    kho = KhoSims(args.data, args.refresh)
    truc = nap_truc_video(data, args.refresh)
    simsA = {i: kho.lay(nhan[i]["canh_A_vi"], nhan[i]["canh_A_en"]) for i in bat}
    simsB = {i: kho.lay(nhan[i]["canh_B_vi"], nhan[i]["canh_B_en"]) for i in bat}

    rows_nen = [allocate_rows(c, args.allocator, DEFAULT_N_FLAT, _plan())[:MAX_ROWS]
                for c in cands_of]

    # ---- bất biến: câu cổng TẮT không được đổi một dòng nào -----------------
    print("\n=== bất biến: cổng tắt ⇒ dòng không đổi ===", flush=True)
    tat = [i for i in range(len(sach)) if i not in bat]
    rows_thu = ap_dung(cands_of, nhan, simsA, simsB, 5, "hm", 1.0, args.allocator, truc=truc)
    for i in tat:
        assert rows_nen[i] == rows_thu[i], f"mục {i} đổi dòng dù cổng TẮT"
    print(f"  OK: {len(tat)} mục cổng tắt ra dòng GIỐNG HỆT nền (assert).")

    # ---- TUNE/TEST ---------------------------------------------------------
    # Chia PHÂN TẦNG, không chia theo chẵn/lẻ thô. Bước sinh đặt câu hai cảnh
    # vào đúng các chỉ số chẵn trong mỗi shard, nên chia chẵn/lẻ CHÍNH LÀ chia
    # hai-cảnh/một-cảnh: lần chạy đầu cho TUNE 16/16 câu qua cổng và TEST 0/24,
    # khiến số TEST là phép đồng nhất và hoàn toàn vô nghĩa. Chia trong TỪNG
    # nhóm mới giữ được cả hai nửa cùng tỷ lệ cổng bật.
    bat_set = set(bat)
    g_bat = [i for i in range(len(sach)) if i in bat_set]
    g_tat = [i for i in range(len(sach)) if i not in bat_set]
    i_tune = sorted(g_bat[0::2] + g_tat[0::2])
    i_test = sorted(g_bat[1::2] + g_tat[1::2])
    print(f"\nTUNE {len(i_tune)} mục ({sum(1 for i in i_tune if i in bat)} qua cổng) | "
          f"TEST {len(i_test)} mục ({sum(1 for i in i_test if i in bat)} qua cổng)")

    ho_tune = cac_lan_boc(GOC_TUNE_MOI, args.tune_seeds, args.tune_draws,
                          [sach[i] for i in i_tune], kf)
    ho_test = cac_lan_boc(GOC_TEST_MOI, args.test_seeds, args.test_draws,
                          [sach[i] for i in i_test], kf)

    def cham(idx, cfg, ho):
        gt_s = [sach[i] for i in idx]
        rows = ap_dung(
            [cands_of[i] for i in idx], [nhan[i] for i in idx],
            {k: simsA[i] for k, i in enumerate(idx) if i in simsA},
            {k: simsB[i] for k, i in enumerate(idx) if i in simsB},
            *cfg, args.allocator, truc=truc)
        mats = ma_tran_dong(rows, gt_s)
        return [cham_nhanh(mats, d, windows) for d in ho]

    def cham_nen(idx, ho):
        gt_s = [sach[i] for i in idx]
        mats = ma_tran_dong([rows_nen[i] for i in idx], gt_s)
        return [cham_nhanh(mats, d, windows) for d in ho]

    nen_tune = cham_nen(i_tune, ho_tune)
    m_nen_tune = float(np.mean(nen_tune))
    print(f"\nNỀN trên TUNE: {m_nen_tune:.4f} ±{np.std(nen_tune):.4f}")
    print(f"\n{'W':>3}{'gộp':>7}{'λ':>6}{'điểm':>9}{'±':>8}{'so nền':>9}")
    print("-" * 44)
    ket = {}
    for W in LUOI_W:
        for gop in LUOI_GOP:
            for lam in LUOI_LAMBDA:
                d = cham(i_tune, (W, gop, lam), ho_tune)
                m, sd = float(np.mean(d)), float(np.std(d))
                ket[(W, gop, lam)] = m
                print(f"{W:>3}{gop:>7}{lam:>6.2f}{m:>9.4f}{sd:>8.4f}"
                      f"{100*(m/m_nen_tune-1):>+8.1f}%", flush=True)

    chot = max(ket, key=lambda k: ket[k])
    print(f"\nCHỐT trên TUNE: W={chot[0]} gộp={chot[1]} λ={chot[2]} "
          f"→ {ket[chot]:.4f} ({100*(ket[chot]/m_nen_tune-1):+.1f}%)")

    # ---- TEST: đúng một lần -------------------------------------------------
    print("\n=== TEST (đọc ĐÚNG MỘT LẦN) ===")
    nen_test = cham_nen(i_test, ho_test)
    chot_test = cham(i_test, chot, ho_test)
    a, sa = float(np.mean(nen_test)), float(np.std(nen_test))
    b, sb = float(np.mean(chot_test)), float(np.std(chot_test))
    bien = max(sa, 0.0005)
    up = 100 * (b / a - 1) if a else 0.0
    print(f"  nền : {a:.4f} ±{sa:.4f}")
    print(f"  chốt: {b:.4f} ±{sb:.4f}   ({up:+.1f}%)")
    # ---- thanh sai số ĐÚNG: bootstrap theo CÂU, không phải theo hạt giống ----
    #
    # ±0,00xx ở trên là độ lệch giữa các họ hạt giống — nó đo nhiễu BỐC THĂM
    # đáp án, và với cùng một tập câu thì nó nhỏ tuỳ ý khi tăng số lần bốc.
    # Thứ thật sự chưa biết là: nếu bốc một tập câu KHÁC thì kết quả có giữ
    # không. Với 12 câu qua cổng, đó mới là nguồn bất định chính. Bootstrap
    # lấy lại mẫu chính các câu TEST để trả lời đúng câu hỏi đó.
    print("\n=== bootstrap theo CÂU (nguồn bất định thật khi n nhỏ) ===")
    gt_test = [sach[i] for i in i_test]
    rows_nen_t = [rows_nen[i] for i in i_test]
    rows_chot_t = ap_dung(
        [cands_of[i] for i in i_test], [nhan[i] for i in i_test],
        {k: simsA[i] for k, i in enumerate(i_test) if i in simsA},
        {k: simsB[i] for k, i in enumerate(i_test) if i in simsB},
        *chot, args.allocator, truc=truc)
    mats_nen = ma_tran_dong(rows_nen_t, gt_test)
    mats_chot = ma_tran_dong(rows_chot_t, gt_test)

    def diem_tung_cau(mats):
        """Điểm trung bình mỗi câu (qua mọi họ × bốc × cửa sổ)."""
        ra = np.zeros(len(gt_test))
        for draws in ho_test:
            for qi in range(len(gt_test)):
                ra[qi] += cham_nhanh([mats[qi]], [draws[qi]], windows)
        return ra / len(ho_test)

    d_nen = diem_tung_cau(mats_nen)
    d_chot = diem_tung_cau(mats_chot)
    rng = np.random.default_rng(4242)
    lay = rng.integers(0, len(gt_test), size=(4000, len(gt_test)))
    delta = d_chot[lay].mean(axis=1) - d_nen[lay].mean(axis=1)
    lo, hi = np.percentile(delta, [2.5, 97.5])
    p_am = float((delta <= 0).mean())
    print(f"  chênh trung bình {d_chot.mean()-d_nen.mean():+.4f}; "
          f"khoảng tin cậy 95% theo câu: [{lo:+.4f}, {hi:+.4f}]")
    print(f"  xác suất chênh ≤ 0 khi bốc lại tập câu: {p_am:.1%}")
    vung = "KHÔNG chứa 0 — hiệu ứng vững trước việc đổi tập câu" if lo > 0 else \
           "CHỨA 0 — với cỡ mẫu này chưa loại được khả năng hoà"
    print(f"  → khoảng tin cậy {vung}")

    # Phán quyết đi theo BOOTSTRAP THEO CÂU, không theo 2σ hạt giống.
    #
    # Luật 2σ dùng cho bộ phân bổ (docs/SHIP_PHU_XAC_SUAT.md) lấy σ giữa các họ
    # hạt giống. Ở đó nó đúng vì tập câu cố định 30 mục và hiệu ứng lớn. Ở đây
    # chỉ 12 mục qua cổng, nên nguồn bất định chính không phải bốc thăm đáp án
    # mà là ĐỔI TẬP CÂU — và σ hạt giống mù hoàn toàn với nguồn đó: tăng số lần
    # bốc là nó nhỏ đi, dù chẳng biết thêm gì về câu hỏi. Dùng nó ở đây sẽ tuyên
    # bố "vượt 2σ" cho một hiệu ứng mà bootstrap nói còn 14% khả năng là hoà.
    print("\n=== KẾT LUẬN ===")
    if lo > 0 and up >= 5.0:
        print(f"  GIỮ ĐƯỢC: TEST {up:+.1f}% và khoảng tin cậy theo câu "
              f"[{lo:+.4f}, {hi:+.4f}] không chứa 0.")
    elif lo > 0:
        print(f"  DƯƠNG nhưng YẾU: chỉ {up:+.1f}% (< +5%). Chưa đáng ship.")
    else:
        print(f"  CHƯA KẾT LUẬN ĐƯỢC: điểm trung bình {up:+.1f}% nhưng khoảng tin cậy "
              f"theo câu [{lo:+.4f}, {hi:+.4f}] CHỨA 0 ({p_am:.0%} khả năng hoà).")
        print("  KHÔNG ship. Hiệu ứng có vẻ thật nhưng cỡ mẫu chưa loại được may rủi.")
        # bao nhiêu câu nữa thì đủ: bề rộng khoảng tin cậy co theo 1/sqrt(n)
        nua = (hi - lo) / 2
        d = d_chot.mean() - d_nen.mean()
        if d > 0:
            can = int(np.ceil(len(i_test) * (nua / d) ** 2))
            print(f"  Cần khoảng {can} mục TEST (hiện {len(i_test)}) — tức sinh thêm "
                  f"~{max(0, can - len(i_test)) * 2} mục cho cả hai nửa — để khoảng tin cậy "
                  "tách khỏi 0 nếu hiệu ứng giữ nguyên độ lớn.")
    print(f"  Cỡ mẫu: TEST {len(i_test)} mục, "
          f"{sum(1 for i in i_test if i in bat)} qua cổng.")
    print(f"  (2σ hạt giống nói {'GIỮ ĐƯỢC' if (b-a) >= 2*bien else 'HOÀ'} — "
          "ghi ra để thấy hai thước đo khác nhau, phán quyết lấy theo bootstrap.)")
    print(f"\nXong sau {time.time()-t0:.0f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
