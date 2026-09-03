"""Độ BÃO HOÀ nội-video — hạng-1 của SigLIP đã nằm trong cửa sổ chấm chưa?

Câu hỏi mở §9.1 của `docs/PAPER_XEP_HANG_NOI_VIDEO.md`: nếu trên phần lớn câu,
keyframe hạng-1 nội-video (theo sim, trong video ĐÚNG) đã rơi vào cửa sổ chấm
quanh đáp án, thì trục "xếp lại nội-video" đã BÃO HOÀ cho MỌI tín hiệu — VLM,
encoder mới, làm mượt — vì không còn chỗ để leo.

Phép đếm 0 API: với mỗi mục sạch, lấy ứng viên cùng video-đúng trong pool sản
xuất, xếp theo sim; hỏi keyframe hạng-1 có |frame − đáp án| ≤ {6,10,20} không.
Đồng thời in phân vị của HẠNG mà keyframe-đáp-án đứng, tách nhóm một/hai cảnh.

Số này KHÔNG thay cổng tiền-đăng-ký của PE-Core (docs/PRETEST_ENCODER.md §2) —
nó chỉ cho biết trần còn lại của trục này nằm ở nhóm nào.

    python -u scripts/dem_bao_hoa_noi_video.py
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    ap.add_argument("--cache-uv", default=str(ROOT / "data" / "cache_bo_do_moi"))
    args = ap.parse_args()

    moi = json.loads(Path(args.moi).read_text(encoding="utf-8"))
    uv = json.loads((Path(args.cache_uv) / "uv_moi.json").read_text(encoding="utf-8"))
    giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]

    ket = {"mot": [], "hai": []}
    for i in giu:
        g = moi[i]
        vid, dap = g["video_id"], int(g["frame_idx"])
        trong = sorted(((float(s), int(f)) for v, f, s, _lf in uv[i] if v == vid),
                       key=lambda t: -t[0])
        if not trong:
            continue
        lech1 = abs(trong[0][1] - dap)  # hạng-1 cách đáp án bao xa
        # hạng của keyframe GẦN đáp án nhất trong pool
        hang_dap = 1 + int(np.argmin([abs(f - dap) for _s, f in trong]))
        ket["hai" if canh_cua(g) else "mot"].append((lech1, hang_dap, len(trong)))

    print("=== BÃO HOÀ NỘI-VIDEO (hạng-1 theo sim, trong video ĐÚNG) ===")
    for nhom, ten in (("mot", "MỘT cảnh"), ("hai", "HAI cảnh")):
        a = ket[nhom]
        if not a:
            continue
        lech = np.array([x[0] for x in a])
        hang = np.array([x[1] for x in a])
        n = len(a)
        print(f"\n{ten} (n={n}, pool cùng-video trung vị "
              f"{int(np.median([x[2] for x in a]))} keyframe):")
        for w in (6, 10, 20):
            print(f"  hạng-1 nằm trong ±{w:>2}: {int((lech <= w).sum()):>2}/{n}"
                  f" = {100 * (lech <= w).mean():.0f}%")
        print(f"  độ lệch hạng-1 → đáp án: trung vị {int(np.median(lech))} frame; "
              f"p75 {int(np.percentile(lech, 75))}; p90 {int(np.percentile(lech, 90))}")
        print(f"  hạng của keyframe-gần-đáp-án: trung vị {int(np.median(hang))}; "
              f"≤3: {100 * (hang <= 3).mean():.0f}%; ≤10: {100 * (hang <= 10).mean():.0f}%")

    print("\nĐọc số: nhóm nào có 'hạng-1 trong ±20' đã cao thì trục xếp-lại-nội-video"
          "\nBÃO HOÀ ở đó cho MỌI tín hiệu; chỗ ăn còn lại nằm ở nhóm thấp.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
