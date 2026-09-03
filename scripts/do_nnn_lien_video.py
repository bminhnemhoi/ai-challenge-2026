"""NNN liên-video — sắp lại 100 dòng theo bias-hub, đo đủ 5 cổng.

Cổng V2 (`dem_bias_hub.py --truc v2`) QUA sát nút: 57% dòng khác-video đứng trên
dòng đúng có bias-bank cao hơn (ngưỡng công bố trước 55%, n=17 mục). Đây là phép
đo đầy đủ mà cổng ấy mua vé.

Cơ chế — cố tình hẹp để mọi bất biến giữ được bằng assert:

  * TẬP 100 dòng sản xuất KHÔNG đổi, chỉ đổi THỨ TỰ ⇒ R@100 bất biến theo xây
    dựng (assert).
  * Chỉ áp cho câu MỘT cảnh (nhóm có deficit thứ tự dòng 0,2040) ⇒ nhóm HAI cảnh
    bất biến (assert) — hai cảnh đã có đòn hoán vị riêng.
  * Khoá sắp mới:  key(dòng hạng r) = r + k · z(bias)   (z chuẩn hoá trong câu,
    bias = bias-bank của keyframe gần nhất của dòng; sort ổn định, tăng dần).
    k = 0 ⇒ y hệt thứ tự sản xuất (assert). k > 0 ⇒ dòng hub bị ĐẨY XUỐNG.

Vì sao khoá lai "hạng + k·z" thay vì thay điểm sim: hạng sản xuất đã mã hoá mọi
tín hiệu ship (coverage, thang bù trừ); chỉ trừ bias vào sim sẽ phá cả cấu trúc
thang. NNN gốc trừ vào sim vì họ chưa có allocator; ở đây bias chỉ được phép
XÊ DỊCH hạng, biên độ do k kiểm soát.

Đủ 5 cổng: TUNE/TEST chia tầng trong nhóm bị ảnh hưởng (một cảnh, 33/33, xáo
seed cố định); TEST đọc MỘT lần ở k chọn trên TUNE; bootstrap THEO CÂU; bất biến
k=0 + tập-dòng + nhóm-hai-cảnh bằng assert; chấm qua đúng allocate_rows sản xuất.

    python -u scripts/do_nnn_lien_video.py
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

from scripts.dem_bias_hub import nap_bank  # noqa: E402
from scripts.do_bo_do_moi import GOC_HAT  # noqa: E402
from scripts.do_cap_thoi_gian_moi import canh_cua  # noqa: E402
from scripts.experiment_cap_thoi_gian import KhoSims, _plan  # noqa: E402
from scripts.experiment_phu_quet_luoi import (  # noqa: E402
    cac_lan_boc,
    cham_nhanh,
    ma_tran_dong,
)
from scripts.make_submission import DEFAULT_N_FLAT, allocate_rows  # noqa: E402
from src.core.submission import MAX_ROWS, Candidate  # noqa: E402

K_QUET = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def sap_lai(rows, z_bias, k):
    """Sắp ổn định theo r + k·z; k=0 trả về đúng thứ tự cũ."""
    khoa = [r + 1 + k * z_bias[r] for r in range(len(rows))]
    thu_tu = sorted(range(len(rows)), key=lambda r: (khoa[r], r))
    return [rows[r] for r in thu_tu]


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

    meta = json.loads((Path(args.data) / "metadata.json").read_text(encoding="utf-8"))
    hang_of = {(m["video_id"], int(m["frame_idx"])): i for i, m in enumerate(meta)}
    kf = {}
    for m in meta:
        kf.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
    kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in kf.items()}
    del meta

    # --- bias-bank (KhoSims cache ⇒ 0 API)
    bank = nap_bank(ROOT)
    print(f"bank {len(bank)} câu; tính bias (cache) ...", flush=True)
    kho = KhoSims(args.data, False)
    tong = None
    for q in bank:
        s = kho.lay(q, "")
        tong = s.astype(np.float64) if tong is None else tong + s
    bias = (tong / len(bank)).astype(np.float32)

    def bias_dong(v, f):
        a = kf.get(v)
        if a is None or not len(a):
            return None
        kk = int(a[int(np.argmin(np.abs(a - int(f))))])
        r = hang_of.get((v, kk))
        return None if r is None else float(bias[r])

    # --- dòng sản xuất + z(bias) từng câu
    print("dựng 100 dòng sản xuất cho 132 mục ...", flush=True)
    rows_sx, z_of, mot_canh = [], [], []
    for i in giu:
        g = moi[i]
        cands = [Candidate(v, f, s, lf) for v, f, s, lf in uv[i]]
        rows = allocate_rows(cands, "coverage", DEFAULT_N_FLAT, _plan())[:MAX_ROWS]
        b = np.array([bias_dong(v, f) or 0.0 for v, f in rows])
        sd = float(b.std())
        z_of.append((b - b.mean()) / sd if sd > 1e-9 else np.zeros(len(b)))
        rows_sx.append(rows)
        mot_canh.append(not canh_cua(g))

    gt_sub = [moi[i] for i in giu]
    ho = cac_lan_boc(GOC_HAT, args.seeds, args.draws, gt_sub, kf)

    def diem_tung_cau(rows_of):
        """Điểm trung bình theo CÂU (để bootstrap), chấm y hệt bộ đo."""
        mats = ma_tran_dong(rows_of, gt_sub)
        d = np.zeros(len(gt_sub))
        for qi in range(len(gt_sub)):
            d[qi] = float(np.mean([cham_nhanh([mats[qi]], [b[qi]], [6, 10, 20])
                                   for b in ho]))
        return d

    nen_cau = diem_tung_cau(rows_sx)

    # --- cổng bất biến: k=0 là hằng đẳng thức; tập dòng không đổi; hai cảnh không đổi
    for rows, z in zip(rows_sx, z_of):
        assert sap_lai(rows, z, 0.0) == rows, "k=0 phải là hằng đẳng thức"

    def ap(k):
        ra = []
        for rows, z, mc in zip(rows_sx, z_of, mot_canh):
            moi_r = sap_lai(rows, z, k) if mc else rows
            assert sorted(moi_r) == sorted(rows), "tập dòng phải bất biến (R@100)"
            ra.append(moi_r)
        return ra

    # --- TUNE/TEST chia tầng TRONG nhóm bị ảnh hưởng (một cảnh)
    idx_mc = [q for q, mc in enumerate(mot_canh) if mc]
    rng = np.random.default_rng(20260903)
    xao = rng.permutation(len(idx_mc))
    tune = sorted(idx_mc[j] for j in xao[: len(xao) // 2])
    test = sorted(idx_mc[j] for j in xao[len(xao) // 2:])
    print(f"một cảnh: {len(idx_mc)} câu -> TUNE {len(tune)} / TEST {len(test)}")

    print(f"\n{'k':>6}{'TUNE một-cảnh':>16}{'chênh':>10}")
    print("-" * 34)
    ket_tune = {}
    for k in K_QUET:
        d = diem_tung_cau(ap(k))
        ket_tune[k] = d
        assert np.allclose(d[[q for q in range(len(gt_sub)) if not mot_canh[q]]],
                           nen_cau[[q for q in range(len(gt_sub)) if not mot_canh[q]]]), \
            "nhóm HAI cảnh phải bất biến"
        m = float(d[tune].mean())
        print(f"{k:>6.0f}{m:>16.4f}{m - float(ket_tune[0.0][tune].mean()):>+10.4f}",
              flush=True)

    nen_tune = float(ket_tune[0.0][tune].mean())
    k_tot = max(K_QUET, key=lambda k: float(ket_tune[k][tune].mean()))
    print(f"\nk tốt nhất trên TUNE: {k_tot:.0f} "
          f"({float(ket_tune[k_tot][tune].mean()) - nen_tune:+.4f} so với k=0)")

    if k_tot == 0.0:
        print("\n=== KẾT LUẬN: ÂM ngay trên TUNE — không tiêu lần đọc TEST ===")
        print("Hub liên-video có thật (57%) nhưng xê dịch hạng theo bias không đổi"
              " được nó thành điểm. Đóng nốt trục ① trong bảng cửa đóng.")
        return 0

    # --- TEST: đọc MỘT lần ở k_tot, bootstrap theo câu
    d_tot = ket_tune[k_tot]
    a, b = nen_cau[test], d_tot[test]
    rng2 = np.random.default_rng(4242)
    lay = rng2.integers(0, len(test), size=(4000, len(test)))
    diff = b[lay].mean(axis=1) - a[lay].mean(axis=1)
    lo, hi = np.percentile(diff, [2.5, 97.5])
    print(f"\n=== TEST (đọc một lần, k={k_tot:.0f}) ===")
    print(f"  nền {a.mean():.4f} -> {b.mean():.4f} ({b.mean() - a.mean():+.4f}, "
          f"{100 * (b.mean() / a.mean() - 1) if a.mean() else 0:+.1f}%)")
    print(f"  bootstrap theo CÂU (n={len(test)}): KTC [{lo:+.4f}, {hi:+.4f}]; "
          f"P(<=0) = {(diff <= 0).mean():.1%}")
    if lo > 0:
        print("\n  KTC loại 0 — đủ điều kiện cân nhắc ship (còn cổng smoke-test).")
    else:
        print("\n  KTC KHÔNG loại 0 — chưa đủ điều kiện ship. Ghi số, đóng trục ①.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
