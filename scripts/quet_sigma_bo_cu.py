"""Trục sigma trên bộ 60 câu CŨ — kiểm chéo hướng mà lưới TUNE của bộ mới chỉ ra.

Lưới trên bộ mới cho một biên **đơn điệu**: sigma 15 → 60 kéo điểm TUNE từ
0,0973 lên 0,1498. Cám dỗ ở đây là đọc nó thành "sigma lớn hơn thì tốt hơn".
Nhưng cả 200 tổ hợp dùng CHUNG 66 câu TUNE, nên biên đơn điệu nói về **hình dạng
mặt tham số**, không nói gì về khả năng khái quát — bốn mươi tổ hợp mỗi mức
không phải bốn mươi phép đo độc lập.

Bộ 60 câu CŨ là tập dữ liệu ĐỘC LẬP và có phân bố khác hẳn (0% câu hai cảnh).
Quét đúng trục sigma trên đó tách được hai khả năng:

  * sigma lớn tốt ở cả hai bộ  -> tham số sản xuất đang đặt sai, đáng đo tiếp;
  * sigma lớn chỉ tốt ở bộ mới -> đây là ĐÁNH ĐỔI giữa hai phân bố, và phải nói
    ra chứ không được giấu.

    python -u scripts/quet_sigma_bo_cu.py
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

from scripts.do_phan_bo_sau import GOC_CU_PB, NEN_CP, cham_tung_cau, sinh_dong  # noqa: E402
from scripts.experiment_phu_quet_luoi import cac_lan_boc  # noqa: E402
from src.core.submission import Candidate  # noqa: E402

SIGMA = (15.0, 20.0, 30.0, 45.0, 60.0)
NHIET = (0.01, 0.02)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cu-cache", default=str(ROOT / "data" / "cache_phu_quet_luoi"))
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--draws", type=int, default=48)
    args = ap.parse_args()

    windows = [int(w) for w in args.windows.split(",")]
    raw = json.loads((Path(args.cu_cache) / "ung_vien.json").read_text(encoding="utf-8"))
    gt = raw["gt"]
    cands = [[Candidate(v, f, s, lf) for v, f, s, lf in q] for q in raw["cands"]]
    kf = {v: np.array(a, dtype=np.int64) for v, a in raw["kf"].items()}
    ho = cac_lan_boc(GOC_CU_PB, args.seeds, args.draws, gt, kf)
    print(f"bo CU: {len(gt)} cau | goc hat {GOC_CU_PB} | {args.seeds} ho x {args.draws} boc")

    d_nen = None
    print(f"\n{'nhiet':>8}{'sigma':>8}{'diem':>10}{'+-':>8}{'so nen':>9}")
    print("-" * 43)
    for nhiet in NHIET:
        for sg in SIGMA:
            rows = [sinh_dong(c, "coverage", (nhiet, sg, 6, 5)) for c in cands]
            per_q, per_ho = cham_tung_cau(rows, gt, ho, windows)
            if (nhiet, sg) == (NEN_CP[0], NEN_CP[1]):
                d_nen = per_q
            m = per_q.mean()
            so = f"{100*(m/d_nen.mean()-1):+7.1f}%" if d_nen is not None else f"{'':>8}"
            danh = "   <- nen san xuat" if (nhiet, sg) == (NEN_CP[0], NEN_CP[1]) else ""
            print(f"{nhiet:>8g}{sg:>8g}{m:>10.4f}{per_ho.std():>8.4f}{so}{danh}", flush=True)

    print("\nDoc: neu cot 'so nen' AM o sigma 45/60 tren bo CU trong khi bien TUNE cua")
    print("bo MOI di len o dung hai muc do, thi huong 'sigma lon hon' KHONG phai mot")
    print("cai tien chung — no la danh doi giua hai phan bo de.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
