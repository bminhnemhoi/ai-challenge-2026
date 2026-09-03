"""R1 bậc 2 — CHẤM ĐIỂM đuôi-chi-lại, bootstrap theo câu, TEST đọc một lần.

Bậc đếm (`dem_chi_lai_duoi.py`) đã qua: TUNE net +7 mục (ngưỡng ≥+4).
Bậc này trả lời: số mục cứu được có QUY THÀNH ĐIỂM dưới bộ chấm thật không
(cửa sổ {6,10,20}, 4 họ × 48 lần bốc, chấm y hệt sản xuất)?

Cơ chế không có tham số quét ⇒ TUNE chỉ xác nhận dấu; TEST đọc đúng một lần.
Bất biến: 50 dòng đầu trùng từng dòng (assert) ⇒ R@1..R@50 bất biến; mọi chênh
lệch chỉ đến từ số hạng R@100.

    python -u scripts/do_diem_chi_lai_duoi.py
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

from scripts.dem_chi_lai_duoi import KHOA, duoi_moi  # noqa: E402
from scripts.do_bo_do_moi import GOC_HAT  # noqa: E402
from scripts.do_cap_thoi_gian_moi import canh_cua  # noqa: E402
from scripts.experiment_cap_thoi_gian import _plan  # noqa: E402
from scripts.experiment_phu_quet_luoi import (  # noqa: E402
    cac_lan_boc,
    cham_nhanh,
    ma_tran_dong,
)
from scripts.make_submission import DEFAULT_N_FLAT, allocate_rows  # noqa: E402
from src.core.submission import MAX_ROWS, Candidate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    ap.add_argument("--cache-uv", default=str(ROOT / "data" / "cache_bo_do_moi"))
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--draws", type=int, default=48)
    args = ap.parse_args()

    moi = json.loads(Path(args.moi).read_text(encoding="utf-8"))
    uv = json.loads((Path(args.cache_uv) / "uv_moi.json").read_text(encoding="utf-8"))
    giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]
    gt_sub = [moi[i] for i in giu]

    meta = json.loads((Path(args.data) / "metadata.json").read_text(encoding="utf-8"))
    kf = {}
    for m in meta:
        kf.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
    kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in kf.items()}
    del meta

    rows_nen, rows_r1 = [], []
    for i in giu:
        cands = [Candidate(v, f, s, lf) for v, f, s, lf in uv[i]]
        rows = allocate_rows(cands, "coverage", DEFAULT_N_FLAT, _plan())[:MAX_ROWS]
        r2 = duoi_moi(rows, cands)
        assert r2[:KHOA] == rows[:KHOA]
        rows_nen.append(rows)
        rows_r1.append(r2)

    ho = cac_lan_boc(GOC_HAT, args.seeds, args.draws, gt_sub, kf)

    def diem_cau(rows_of):
        mats = ma_tran_dong(rows_of, gt_sub)
        d = np.zeros(len(gt_sub))
        for qi in range(len(gt_sub)):
            d[qi] = float(np.mean([cham_nhanh([mats[qi]], [b[qi]], [6, 10, 20])
                                   for b in ho]))
        return d

    dn, dr = diem_cau(rows_nen), diem_cau(rows_r1)

    # cùng phép chia với bậc đếm (seed 20260903, phân tầng theo cảnh)
    rng = np.random.default_rng(20260903)
    hai = [q for q in range(len(gt_sub)) if canh_cua(gt_sub[q])]
    mot = [q for q in range(len(gt_sub)) if not canh_cua(gt_sub[q])]
    tune = set()
    for nhom in (mot, hai):
        x = rng.permutation(len(nhom))
        tune |= {nhom[j] for j in x[: len(x) // 2]}
    test = [q for q in range(len(gt_sub)) if q not in tune]
    tune = sorted(tune)

    def bao(ten, chi_so, doc_quyet_dinh):
        a, b = dn[chi_so], dr[chi_so]
        ch = b.mean() - a.mean()
        print(f"\n=== {ten} (n={len(chi_so)}) ===")
        print(f"  nền {a.mean():.4f} -> R1 {b.mean():.4f} "
              f"({ch:+.4f}, {100 * ch / a.mean() if a.mean() else 0:+.1f}%)")
        rng2 = np.random.default_rng(4242)
        lay = rng2.integers(0, len(chi_so), size=(4000, len(chi_so)))
        d = b[lay].mean(axis=1) - a[lay].mean(axis=1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  bootstrap theo CÂU: KTC [{lo:+.4f}, {hi:+.4f}]; "
              f"P(<=0) = {(d <= 0).mean():.1%}")
        return ch, lo

    ch_t, _ = bao("TUNE (xác nhận dấu)", tune, False)
    if ch_t <= 0:
        print("\n=== KẾT LUẬN: điểm TUNE không dương — DỪNG, không đọc TEST. ===")
        return 0
    ch, lo = bao("TEST (đọc MỘT lần)", test, True)
    n_doi = sum(1 for q in range(len(gt_sub)) if rows_nen[q] != rows_r1[q])
    print(f"\nmục có đuôi đổi: {n_doi}/{len(gt_sub)}; 50 dòng đầu bất biến (assert).")
    print("\n=== KẾT LUẬN ===")
    if ch > 0 and lo > -1e-9:
        print("  DƯƠNG và KTC không âm — đủ điều kiện tích hợp sản xuất sau")
        print("  cổng smoke (cờ rút lui riêng, mặc định bật).")
    elif ch > 0:
        print("  Dương nhưng KTC chứa 0 — cơ chế RẺ và 4/5 số hạng bất biến nên")
        print("  vẫn cân nhắc ship theo ngoại lệ cổng 3 (cơ chế tất định + rủi ro")
        print("  chặn cứng); ghi rõ trạng thái bằng chứng.")
    else:
        print("  TEST âm — không ship, ghi bảng cửa đóng.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
