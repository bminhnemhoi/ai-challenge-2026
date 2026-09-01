"""Phân rã trần +126% thành ba tầng — điểm còn lại nằm ở đâu, bằng số.

`tran_dinh_vi_noi_video.py` cho biết định vị nội-video đáng **+126%**, nhưng
"định vị nội-video" gộp ba việc rất khác nhau, và biết việc nào chiếm bao nhiêu
quyết định nên đầu tư vào đâu:

    tầng 1  SINH ỨNG VIÊN  — keyframe đáp án có lọt vào 400 ứng viên không?
    tầng 2  XẾP HẠNG       — nếu đã lọt, nó có được xếp hạng-1 nội-video không?
    tầng 3  PHÂN BỔ DÒNG   — nếu đã hạng-1, 100 dòng có dồn quanh nó không?

Mỗi tầng được đo bằng một oracle **cộng dồn**: tầng k giả định mọi tầng trước đã
hoàn hảo. Chênh lệch giữa hai tầng liên tiếp chính là phần thuộc về tầng sau.

Vì sao phân rã này đáng làm: đã đo được cảnh B (một biện pháp ở **tầng 1**) mua
+23,3% ở nhóm câu hai cảnh. Nếu tầng 1 chỉ chiếm một phần nhỏ của trần thì phần
lớn công sức tiếp theo phải dồn sang tầng 2 và 3 — và ngược lại. Không có phép
phân rã này thì mọi lựa chọn đầu tư đều là phỏng đoán.

    python -u scripts/phan_ra_tran.py
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

from scripts.do_bo_do_moi import GOC_HAT  # noqa: E402
from scripts.experiment_phu_quet_luoi import cac_lan_boc, cham_nhanh, ma_tran_dong  # noqa: E402
from scripts.make_submission import DEFAULT_N_FLAT, allocate_rows  # noqa: E402
from scripts.tran_dinh_vi_noi_video import oracle_rows  # noqa: E402
from src.core.submission import MAX_ROWS, AllocationPlan, Candidate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache_bo_do_moi"))
    ap.add_argument("--allocator", default="coverage")
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--draws", type=int, default=48)
    args = ap.parse_args()

    data = Path(args.data)
    windows = [int(w) for w in args.windows.split(",")]
    moi = json.loads(Path(args.moi).read_text(encoding="utf-8"))
    uv = json.loads((Path(args.cache) / "uv_moi.json").read_text(encoding="utf-8"))

    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    kf_list, last_of, meta_by_key = {}, {}, {}
    for m in meta:
        kf_list.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
        last_of[m["video_id"]] = max(last_of.get(m["video_id"], 0), int(m["frame_idx"]))
        meta_by_key[(m["video_id"], int(m["frame_idx"]))] = m
    kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in kf_list.items()}
    del meta, kf_list

    plan = AllocationPlan(breadth_cost=1.0, depth_cost=0.5, step=10)

    def kf_gan_nhat(g):
        a = kf.get(g["video_id"])
        if a is None or not len(a):
            return None
        return int(a[int(np.argmin(np.abs(a - int(g["frame_idx"]))))])

    def bon_tang(idx, ten):
        gt = [moi[i] for i in idx]
        c0 = [[Candidate(v, f, s, lf) for v, f, s, lf in uv[i]] for i in idx]

        # --- nền
        r_nen = [allocate_rows(c, args.allocator, DEFAULT_N_FLAT, plan)[:MAX_ROWS] for c in c0]

        # --- tầng 1: keyframe đáp án CHẮC CHẮN có trong pool.
        # Điểm gán cho nó là điểm cao nhất mà video đó đang có — tức "được truy
        # xuất ngang ứng viên tốt nhất của chính video mình", KHÔNG phải điểm
        # tuyệt đối cao nhất; như thế oracle này chỉ vá khâu SINH, không lén vá
        # luôn khâu xếp hạng của tầng 2.
        c1, co_san = [], 0
        for cc, g in zip(c0, gt):
            k = kf_gan_nhat(g)
            if k is None:
                c1.append(cc)
                continue
            if any(c.video_id == g["video_id"] and int(c.frame_idx) == k for c in cc):
                co_san += 1
                c1.append(cc)
                continue
            cung = [c.score for c in cc if c.video_id == g["video_id"]]
            diem = max(cung) if cung else float(np.median([c.score for c in cc]))
            c1.append(list(cc) + [Candidate(g["video_id"], k, diem,
                                            last_of.get(g["video_id"], k + 1000))])
        r1 = [allocate_rows(c, args.allocator, DEFAULT_N_FLAT, plan)[:MAX_ROWS] for c in c1]

        # --- tầng 2: keyframe đáp án đứng HẠNG 1 nội-video (điểm vượt mọi ứng
        # viên cùng video một khoảng nhỏ, không đụng tới video khác)
        c2 = []
        for cc, g in zip(c1, gt):
            k = kf_gan_nhat(g)
            if k is None:
                c2.append(cc)
                continue
            cung = [c.score for c in cc if c.video_id == g["video_id"]]
            dinh = (max(cung) if cung else 0.0) + 1e-4
            c2.append([Candidate(c.video_id, c.frame_idx,
                                 dinh if (c.video_id == g["video_id"] and int(c.frame_idx) == k)
                                 else c.score, c.video_last_frame) for c in cc])
        r2 = [allocate_rows(c, args.allocator, DEFAULT_N_FLAT, plan)[:MAX_ROWS] for c in c2]

        # --- tầng 3: phân bổ hoàn hảo (oracle đầy đủ)
        r3 = [oracle_rows(r, g["video_id"], int(g["frame_idx"]),
                          last_of.get(g["video_id"], int(g["frame_idx"]) + 1000))
              for r, g in zip(r2, gt)]

        ho = cac_lan_boc(GOC_HAT, args.seeds, args.draws, gt, kf)

        def diem(rows):
            mats = ma_tran_dong(rows, gt)
            return float(np.mean([cham_nhanh(mats, d, windows) for d in ho]))

        d0, d1, d2, d3 = diem(r_nen), diem(r1), diem(r2), diem(r3)
        tong = d3 - d0
        print(f"\n=== {ten} (n={len(idx)}; {co_san}/{len(idx)} câu đã có keyframe đáp án trong pool) ===")
        print(f"{'tầng':<34}{'điểm':>9}{'thêm':>9}{'% của trần':>12}")
        print("-" * 64)
        print(f"{'nền (hệ thống hiện tại)':<34}{d0:>9.4f}{'—':>9}{'—':>12}")
        for ten_t, d, tr in (("1. SINH ứng viên hoàn hảo", d1, d1 - d0),
                             ("2. + XẾP HẠNG nội-video hoàn hảo", d2, d2 - d1),
                             ("3. + PHÂN BỔ dòng hoàn hảo", d3, d3 - d2)):
            print(f"{ten_t:<34}{d:>9.4f}{tr:>+9.4f}"
                  f"{(100*tr/tong if tong else 0):>11.0f}%")
        print(f"{'TRẦN (cả ba tầng)':<34}{d3:>9.4f}{tong:>+9.4f}"
              f"{100*(d3/d0-1) if d0 else 0:>+11.1f}%")
        return d0, d1, d2, d3

    sach = [i for i, g in enumerate(moi) if not g.get("lan_truc")]
    bon_tang(sach, "bộ MỚI sạch")
    bon_tang([i for i in sach if not moi[i].get("co_2_canh")], "câu MỘT cảnh")
    bon_tang([i for i in sach if moi[i].get("co_2_canh")], "câu HAI cảnh")

    print("\nĐọc bảng này thế nào: cột cuối chia trần thành ba phần rời nhau.")
    print("Tầng nào chiếm phần lớn thì công sức tiếp theo phải dồn vào đó.")
    print("Oracle tầng 1 cố tình chỉ cho keyframe đáp án điểm CAO NHẤT TRONG VIDEO")
    print("của nó, không phải cao nhất tuyệt đối — nếu không nó sẽ lén vá luôn")
    print("tầng 2 và bảng phân rã mất hết ý nghĩa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
