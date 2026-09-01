# -*- coding: utf-8 -*-
"""Thống kê MÔ TẢ hình học 100 dòng trên bộ đo khớp phân bố — KHÔNG phải thí nghiệm.

Lane `paper-2026` là lane đọc và đánh giá. Script này không đề xuất, không chọn,
không chấm cải tiến nào: nó chỉ **đếm tất định** để định giá các đề xuất trong
`docs/PAPER_DINH_VI_2026.md` bằng số thay vì bằng cảm tính.

Câu hỏi nó trả lời:
  1. Trên video ĐÚNG, 100 dòng của đường sản xuất rơi cách khung neo bao xa —
     tính bằng FRAME và tính bằng KEYFRAME?
  2. Con số đó khác nhau thế nào giữa câu MỘT cảnh và câu HAI cảnh?
  3. Trước allocator, 400 ứng viên thô đã đứng cách khung neo bao nhiêu keyframe?

Ý nghĩa: nếu khoảng cách đo bằng KEYFRAME là 0–2 thì nghẽn định vị nội-video là
bài toán "chọn đúng ô ~2 giây", không phải bài toán "dò cả video" — và đó là hai
họ công nghệ hoàn toàn khác nhau.

    python -u scripts/khao_sat_hinh_hoc_dong.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.make_submission import DEFAULT_DEPTH_COST, DEFAULT_N_FLAT, allocate_rows  # noqa: E402
from src.core.submission import MAX_ROWS, AllocationPlan, Candidate  # noqa: E402


def nap():
    gt = json.loads((ROOT / "data" / "ground_truth_moi.json").read_text(encoding="utf-8"))
    uv = json.loads((ROOT / "data" / "cache_bo_do_moi" / "uv_moi.json").read_text(encoding="utf-8"))
    assert len(gt) == len(uv), "cache ứng viên lệch số mục — chạy lại do_bo_do_moi.py"
    meta = json.loads((ROOT / "data" / "metadata.json").read_text(encoding="utf-8"))
    kf = collections.defaultdict(list)
    for m in meta:
        kf[m["video_id"]].append(int(m["frame_idx"]))
    kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in kf.items()}
    return gt, uv, kf


def main() -> int:
    gt, uv, kf = nap()
    # bộ SẠCH: bỏ shard lẫn trục (xem docs/BO_DO_KHOP_PHAN_BO.md §1)
    sach = [i for i, g in enumerate(gt) if not g.get("lan_truc")]
    plan = AllocationPlan(breadth_cost=1.0, depth_cost=DEFAULT_DEPTH_COST, step=10)
    print(f"bộ đo: {len(gt)} mục | bộ sạch: {len(sach)} mục")

    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in sach:
        g = gt[i]
        vid, a = g["video_id"], kf[g["video_id"]]
        i_gt = int(np.argmin(np.abs(a - int(g["frame_idx"]))))
        grp = "HAI" if g.get("co_2_canh") else "MOT"

        cands = [Candidate(v, f, s, lf) for v, f, s, lf in uv[i]]
        rows = allocate_rows(cands, "coverage", DEFAULT_N_FLAT, plan)[:MAX_ROWS]
        idx = [r for r, (v, _f) in enumerate(rows) if v == vid]

        acc[grp]["co_dong"].append(bool(idx))
        if idx:
            acc[grp]["so_dong"].append(len(idx))
            acc[grp]["hang_dau"].append(idx[0] + 1)
            fr = np.array([rows[r][1] for r in idx])
            acc[grp]["gan_frame"].append(int(np.min(np.abs(fr - int(g["frame_idx"])))))
            o = set(int(np.argmin(np.abs(a - int(rows[r][1])))) for r in idx)
            acc[grp]["gan_kf"].append(min(abs(k - i_gt) for k in o))
            acc[grp]["so_o"].append(len(o))
            acc[grp]["tong_o"].append(len(a))
        ck = [int(np.argmin(np.abs(a - int(f)))) for v, f, _s, _lf in uv[i] if v == vid]
        if ck:
            acc[grp]["uv_gan_kf"].append(min(abs(k - i_gt) for k in ck))

    for grp in ("MOT", "HAI"):
        d = acc[grp]
        n = len(d["co_dong"])
        print(f"\n=== câu {grp} cảnh (n={n}) ===")
        print(f"  có ≥1 dòng trên video đúng      : {sum(d['co_dong'])}/{n}")
        sd = np.array(d["so_dong"])
        print(f"  số dòng trên video đúng          : trung vị {np.median(sd):.0f}"
              f"  p25 {np.percentile(sd,25):.0f}  p75 {np.percentile(sd,75):.0f}")
        hd = np.array(d["hang_dau"])
        print(f"  hạng dòng ĐẦU TIÊN của video đúng: trung vị {np.median(hd):.0f}"
              f"  ≤1: {(hd<=1).sum()}  ≤5: {(hd<=5).sum()}  ≤20: {(hd<=20).sum()}")
        gf = np.array(d["gan_frame"])
        print(f"  dòng gần nhất → khung neo (FRAME): trung vị {np.median(gf):.0f}"
              f"  ≤20: {(gf<=20).sum()}  ≤6: {(gf<=6).sum()}")
        gk = np.array(d["gan_kf"])
        print(f"  dòng gần nhất → khung neo (Ô KEYFRAME): =0: {(gk==0).sum()}"
              f"  ≤1: {(gk<=1).sum()}  ≤2: {(gk<=2).sum()}  ≤5: {(gk<=5).sum()}"
              f" | trung vị {np.median(gk):.0f}")
        so, tong = np.array(d["so_o"]), np.array(d["tong_o"])
        print(f"  số ô keyframe được phủ           : trung vị {np.median(so):.0f}"
              f" / {np.median(tong):.0f} ô của video ({100*np.median(so/tong):.1f}%)")
        uk = np.array(d["uv_gan_kf"])
        print(f"  [400 ứng viên THÔ] ứng viên gần nhất → khung neo (Ô KEYFRAME),"
              f" trên {len(uk)} câu có ứng viên ở video đúng:")
        print(f"     =0: {(uk==0).sum()}  ≤1: {(uk<=1).sum()}  ≤2: {(uk<=2).sum()}"
              f"  ≤5: {(uk<=5).sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
