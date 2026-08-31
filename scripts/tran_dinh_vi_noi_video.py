"""Trần lý thuyết của MỌI tín hiệu định vị nội-video — đo bằng oracle, không cần model.

Chẩn đoán trên bộ đo khớp phân bố cho thấy một điều sắc: câu HAI cảnh và câu MỘT
cảnh tìm đúng VIDEO ngang nhau (53/66 vs 51/66) nhưng điểm chênh −62,5%. Toàn bộ
khoảng cách nằm ở **định vị khoảnh khắc trong video**.

Trước khi tiêu 2–4 giờ GPU dựng chỉ mục encoder thứ hai (lever ④), câu hỏi phải
trả lời là: **giỏi tuyệt đối ở định vị nội-video thì đáng bao nhiêu điểm?**

Oracle ở đây rất chặt, và cố tình chặt:
  * KHÔNG đổi việc chọn video — video nào đang có dòng thì vẫn có đúng ngần ấy dòng;
  * KHÔNG đổi thứ hạng — dòng thứ r vẫn thuộc đúng video cũ;
  * chỉ đổi **frame id** của các dòng thuộc video ĐÚNG, đặt lại thành thang bao
    quanh khoảnh khắc thật.

Nên con số nó cho là **cận trên của mọi bộ định vị nội-video** — PE-Core, FG-CLIP,
làm mượt thời gian, cặp thời gian, bất cứ thứ gì. Không tín hiệu nào vượt được
oracle. Nếu trần này nhỏ thì lever ④ không đáng tiền, và đó là kết luận rẻ nhất
có thể mua được.

    python -u scripts/tran_dinh_vi_noi_video.py
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
from src.core.submission import (  # noqa: E402
    MAX_ROWS,
    AllocationPlan,
    Candidate,
    frame_ladder,
)


def oracle_rows(rows, gt_video: str, gt_frame: int, last: int, step: int = 10):
    """Giữ nguyên video và thứ hạng; chỉ đặt lại frame id của video ĐÚNG."""
    idx = [i for i, (v, _f) in enumerate(rows) if v == gt_video]
    if not idx:
        return list(rows)
    thang = frame_ladder(int(gt_frame), len(idx), step, lo=0, hi=last)
    ra = list(rows)
    for k, i in enumerate(idx):
        ra[i] = (gt_video, int(thang[k]) if k < len(thang) else int(gt_frame))
    return ra


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
    assert len(uv) == len(moi), "cache ung vien lech so muc — chay lai do_bo_do_moi.py"

    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    kf_list: dict = {}
    last_of: dict = {}
    for m in meta:
        kf_list.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
        last_of[m["video_id"]] = max(last_of.get(m["video_id"], 0), int(m["frame_idx"]))
    kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in kf_list.items()}
    del meta, kf_list

    plan = AllocationPlan(breadth_cost=1.0, depth_cost=0.5, step=10)
    sach = [i for i, g in enumerate(moi) if not g.get("lan_truc")]

    def do(idx, ten):
        gt = [moi[i] for i in idx]
        cands = [[Candidate(v, f, s, lf) for v, f, s, lf in uv[i]] for i in idx]
        nen = [allocate_rows(c, args.allocator, DEFAULT_N_FLAT, plan)[:MAX_ROWS] for c in cands]
        orc = [oracle_rows(r, g["video_id"], int(g["frame_idx"]),
                           last_of.get(g["video_id"], int(g["frame_idx"]) + 1000))
               for r, g in zip(nen, gt)]
        ho = cac_lan_boc(GOC_HAT, args.seeds, args.draws, gt, kf)
        a = float(np.mean([cham_nhanh(ma_tran_dong(nen, gt), d, windows) for d in ho]))
        b = float(np.mean([cham_nhanh(ma_tran_dong(orc, gt), d, windows) for d in ho]))
        co = sum(1 for r, g in zip(nen, gt) if any(v == g["video_id"] for v, _ in r))
        print(f"{ten:<34}{len(idx):>4}{a:>9.4f}{b:>10.4f}{100*(b/a-1) if a else 0:>+9.1f}%"
              f"{co:>8}/{len(idx)}")
        return a, b

    print(f"{'nhom':<34}{'n':>4}{'nen':>9}{'ORACLE':>10}{'tran':>10}{'video dung':>16}")
    print("-" * 84)
    do(sach, "bo MOI sach")
    do([i for i in sach if not moi[i].get("co_2_canh")], "  |- MOT canh")
    do([i for i in sach if moi[i].get("co_2_canh")], "  |- HAI canh")
    do(list(range(len(moi))), "bo MOI (ca 4 shard)")

    print("\nORACLE = giu nguyen VIDEO va THU HANG cua tung dong, chi dat lai frame id")
    print("cua cac dong thuoc video DUNG thanh thang bao quanh khoanh khac that.")
    print("=> day la CAN TREN cua moi bo dinh vi noi-video (PE-Core, FG-CLIP, lam muot,")
    print("   cap thoi gian...). Khong tin hieu nao vuot duoc no.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
