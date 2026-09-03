"""R1 — chi lại phần ĐUÔI 51..100, phép-đếm-trước 0 đồng tất định.

`docs/NGHIEN_CUU_DA_NGUON_0309.md` R1 (từ IA-Select WSDM'09: mục tiêu
P(≥1 hit) là submodular — hit thứ hai không cộng gì dưới prefix-max, nên đuôi
danh sách nên PHỦ vùng chưa phủ thay vì đào sâu vùng đã phủ).

Vì sao an toàn tuyệt đối: dòng 1..50 KHOÁ CỨNG ⇒ R@1, R@5, R@20, R@50 bất biến
theo xây dựng (assert). Chỉ số hạng R@100 (1/5 trọng số) có thể đổi.

Cơ chế đuôi mới (đăng ký trước, không tham số quét):
  * V50 = các video đã có mặt trong 50 dòng đầu (hệ đã tin chúng);
  * ứng viên pool thuộc V50, chưa được phủ (cách mọi dòng-đã-khoá cùng video
    >20 frame), xếp theo điểm giảm dần, mỗi ứng viên MỘT dòng tại đúng keyframe,
    khử trùng lặp trong video (cách dòng-mới-đã-chọn >20 frame);
  * còn chỗ thì giữ lại các dòng đuôi cũ chưa trùng.

Phép đếm tất định (cửa sổ ±20 frame quanh khoảnh khắc thật, như
`chan_doan_dat_dong.py`):
  (i)  nhóm "MẤT DO ĐẶT DÒNG" (video đúng có dòng + pool có keyframe ≤20 mà
       không dòng nào trúng): bao nhiêu mục trượt→trúng ở ≤100;
  (ii) nhóm "đang trúng CHỈ ở hạng 51..100": bao nhiêu mục mất hit.

NGƯỠNG TIỀN-ĐĂNG-KÝ: net = (i) − (ii) ≥ +4 mục trên nửa TUNE phân tầng
VÀ 0 thay đổi ở mọi hạng ≤50 (assert). Qua mới sang bậc chấm điểm + TEST.

    python -u scripts/dem_chi_lai_duoi.py
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

from scripts.do_cap_thoi_gian_moi import canh_cua  # noqa: E402
from scripts.experiment_cap_thoi_gian import _plan  # noqa: E402
from scripts.make_submission import DEFAULT_N_FLAT, allocate_rows  # noqa: E402
from src.core.submission import MAX_ROWS, Candidate  # noqa: E402

CUA_SO = 20
KHOA = 50


def duoi_moi(rows, cands):
    """Giữ rows[:KHOA]; chi lại phần còn lại theo sàn phủ đều trên V50."""
    dau = rows[:KHOA]
    v50 = []
    for v, _f in dau:
        if v not in v50:
            v50.append(v)
    phu = {}  # video -> list frame đã phủ (dòng khoá + dòng mới)
    for v, f in dau:
        phu.setdefault(v, []).append(int(f))

    moi = []
    for c in sorted(cands, key=lambda c: -float(c.score)):
        if len(moi) >= MAX_ROWS - KHOA:
            break
        v, f = c.video_id, int(c.frame_idx)
        if v not in v50:
            continue
        if any(abs(f - g) <= CUA_SO for g in phu.get(v, ())):
            continue
        moi.append((v, f))
        phu.setdefault(v, []).append(f)

    # còn chỗ: giữ dòng đuôi cũ chưa trùng (video, frame)
    da_co = set(dau) | set(moi)
    for r in rows[KHOA:]:
        if len(moi) >= MAX_ROWS - KHOA:
            break
        if r in da_co:
            continue
        moi.append(r)
        da_co.add(r)
    return dau + moi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    ap.add_argument("--cache-uv", default=str(ROOT / "data" / "cache_bo_do_moi"))
    args = ap.parse_args()

    moi = json.loads(Path(args.moi).read_text(encoding="utf-8"))
    uv = json.loads((Path(args.cache_uv) / "uv_moi.json").read_text(encoding="utf-8"))
    giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]

    def hit_tai(rows, g, toi):
        d = int(g["frame_idx"])
        return any(v == g["video_id"] and abs(int(f) - d) <= CUA_SO
                   for v, f in rows[:toi])

    ket = []
    for i in giu:
        g = moi[i]
        cands = [Candidate(v, f, s, lf) for v, f, s, lf in uv[i]]
        rows = allocate_rows(cands, "coverage", DEFAULT_N_FLAT, _plan())[:MAX_ROWS]
        r2 = duoi_moi(rows, cands)
        assert r2[:KHOA] == rows[:KHOA], "50 dòng đầu phải bất biến"
        assert len(r2) <= MAX_ROWS
        pool_gan = any(v == g["video_id"] and abs(int(f) - int(g["frame_idx"])) <= CUA_SO
                       for v, f, _s, _lf in uv[i])
        video_co_dong = any(v == g["video_id"] for v, _f in rows)
        ket.append({
            "hai_canh": bool(canh_cua(g)),
            "mat_dat_dong": (not hit_tai(rows, g, 100)) and pool_gan and video_co_dong,
            "trung_chi_duoi": hit_tai(rows, g, 100) and not hit_tai(rows, g, KHOA),
            "cuu": (not hit_tai(rows, g, 100)) and hit_tai(r2, g, 100),
            "mat": hit_tai(rows, g, 100) and not hit_tai(r2, g, 100),
        })

    # TUNE/TEST phân tầng theo nhóm cảnh, seed cố định (mục mới chưa ai đọc ở TEST)
    rng = np.random.default_rng(20260903)
    idx_mot = [q for q, k in enumerate(ket) if not k["hai_canh"]]
    idx_hai = [q for q, k in enumerate(ket) if k["hai_canh"]]
    tune = set()
    for nhom in (idx_mot, idx_hai):
        x = rng.permutation(len(nhom))
        tune |= {nhom[j] for j in x[: len(x) // 2]}

    def dem(chi_so):
        ks = [ket[q] for q in chi_so]
        return {
            "n": len(ks),
            "nhom_mat_dat_dong": sum(k["mat_dat_dong"] for k in ks),
            "trung_chi_duoi (rui ro)": sum(k["trung_chi_duoi"] for k in ks),
            "cuu (truot->trung)": sum(k["cuu"] for k in ks),
            "mat (trung->truot)": sum(k["mat"] for k in ks),
        }

    for ten, chi_so in (("TUNE", sorted(tune)),
                        ("TEST (chỉ in sau khi TUNE qua ngưỡng)",
                         [q for q in range(len(ket)) if q not in tune])):
        d = dem(chi_so)
        net = d["cuu (truot->trung)"] - d["mat (trung->truot)"]
        print(f"\n=== {ten} (n={d['n']}) ===")
        for k, v in d.items():
            if k != "n":
                print(f"  {k}: {v}")
        print(f"  NET: {net:+d} mục (ngưỡng tiền-đăng-ký: ≥ +4 trên TUNE)")
        if ten == "TUNE" and net < 4:
            print("\n=== KẾT LUẬN: ÂM/CHƯA ĐỦ trên TUNE — dừng, KHÔNG đọc TEST"
                  " theo nghĩa quyết định (số TEST in dưới chỉ để lưu trữ trung"
                  " thực vì phép đếm tất định đã chạy cả bộ). ===")
    print("\nGhi chú: đây là BẬC ĐẾM. Qua ngưỡng mới sang bậc chấm điểm đầy đủ"
          " (bootstrap theo câu + smoke sản xuất).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
