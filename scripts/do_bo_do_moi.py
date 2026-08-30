"""Chấm hệ thống SẢN XUẤT trên bộ đo mới, và lượng hoá điểm mù về cấu trúc thời gian.

Bộ đo 60 câu cũ có **0/60** câu mô tả hai cảnh nối tiếp, trong khi đề thật của
BTC có **28/55 = 51%**. Bộ mới (``data/ground_truth_moi.json``) được sinh từ
ĐOẠN video thật và có 50% câu hai cảnh, tức khớp phân bố ra đề.

Câu hỏi phép đo này trả lời, bằng số:

1. Hệ thống làm tệ hơn bao nhiêu trên câu HAI CẢNH so với câu MỘT CẢNH?
2. Khác biệt đó có phải do cấu trúc thời gian, hay chỉ vì câu do máy sinh khác
   câu do người viết? — tách bằng cách so nhóm MỘT CẢNH của bộ mới với bộ cũ.
   Nếu hai nhóm đó ngang nhau thì khác biệt đến từ cấu trúc; nếu bộ mới dễ hơn
   hẳn thì câu máy sinh dễ hơn và phải nói rõ.

Chấm bằng đúng đường sản xuất: ``ranked_hits`` -> ``allocate_rows(..., allocator)``
-> ``final_score``/``r_score_kis`` (bản vector hoá đã đối chiếu tuyệt đối trong
``experiment_phu_quet_luoi.py``). Đáp án bốc đều trong khe giữa hai keyframe,
không snap; cửa sổ chấm {6, 10, 20}; nhiều họ hạt giống để có sai số thật.

    python -u scripts/do_bo_do_moi.py
    python -u scripts/do_bo_do_moi.py --allocator hybrid    # đối chứng
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
    ranked_hits,
)
from src.core.submission import MAX_ROWS, AllocationPlan, Candidate  # noqa: E402

GOC_HAT = 77000  # tách khỏi mọi gốc đã dùng (30000 / 50000 / 70000 / 90000 / 123450)


def nap_kf(eng):
    kf: dict = {}
    for m in eng.metadata:
        kf.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
    return {v: np.array(sorted(a), dtype=np.int64) for v, a in kf.items()}


def cham_bo(gt, cands_of, kf, windows, seeds, draws, allocator):
    plan = AllocationPlan(breadth_cost=1.0, depth_cost=DEFAULT_DEPTH_COST, step=10)
    rows_of = [allocate_rows(c, allocator, DEFAULT_N_FLAT, plan)[:MAX_ROWS] for c in cands_of]
    ho = cac_lan_boc(GOC_HAT, seeds, draws, gt, kf)
    mats = ma_tran_dong(rows_of, gt)
    diem = [cham_nhanh(mats, d, windows) for d in ho]
    co_video = sum(1 for r, g in zip(rows_of, gt) if any(v == g["video_id"] for v, _ in r))
    return float(np.mean(diem)), float(np.std(diem)), co_video, rows_of


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    ap.add_argument("--allocator", choices=("coverage", "hybrid"), default="coverage")
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--draws", type=int, default=48)
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache_bo_do_moi"))
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    data = Path(args.data)
    windows = [int(w) for w in args.windows.split(",")]
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    moi = json.loads(Path(args.moi).read_text(encoding="utf-8"))
    cu = json.loads((data / "ground_truth.json").read_text(encoding="utf-8"))

    print("nap chi muc (cham, mot lan) ...", flush=True)
    from src.core.kis_engine import KISEngine

    eng = KISEngine(args.data).load()
    kf = nap_kf(eng)
    moi = [g for g in moi if g["video_id"] in eng.last_frame]
    cu = [g for g in cu if g["video_id"] in eng.last_frame]

    def ung_vien(gt, tag):
        f = cache / f"uv_{tag}.json"
        if f.is_file() and not args.refresh:
            raw = json.loads(f.read_text(encoding="utf-8"))
            if len(raw) == len(gt):
                return [[Candidate(v, fi, s, lf) for v, fi, s, lf in q] for q in raw]
        t0 = time.time()
        out = []
        for i, g in enumerate(gt, 1):
            hits = ranked_hits(eng, g["kis_query_vi"], g.get("kis_query_en"))
            out.append([Candidate(h.video_id, h.frame_idx, h.score, h.video_last_frame)
                        for h in hits])
            if i % 20 == 0:
                print(f"  {tag}: {i}/{len(gt)}  ({time.time()-t0:.0f}s)", flush=True)
        f.write_text(json.dumps(
            [[[c.video_id, int(c.frame_idx), float(c.score), int(c.video_last_frame)] for c in q]
             for q in out]), encoding="utf-8")
        return out

    print(f"truy xuat duong san xuat: {len(cu)} cau cu + {len(moi)} cau moi", flush=True)
    uv_cu = ung_vien(cu, "cu")
    uv_moi = ung_vien(moi, "moi")

    # bộ SẠCH: loại shard c (trục hai-cảnh trùng khít trục dải video)
    sach = [i for i, g in enumerate(moi) if not g.get("lan_truc")]
    nhom = {
        "cu (60 cau nguoi viet, 0% hai canh)": list(range(len(cu))),
    }
    ket = {}
    m, sd, cv, _ = cham_bo(cu, uv_cu, kf, windows, args.seeds, args.draws, args.allocator)
    ket["cu"] = (m, sd, cv, len(cu))

    def con(idx, ten):
        if not idx:
            return
        g = [moi[i] for i in idx]
        c = [uv_moi[i] for i in idx]
        m, sd, cv, _ = cham_bo(g, c, kf, windows, args.seeds, args.draws, args.allocator)
        ket[ten] = (m, sd, cv, len(g))

    con(sach, "moi_sach")
    con([i for i in sach if moi[i].get("co_2_canh")], "moi_hai_canh")
    con([i for i in sach if not moi[i].get("co_2_canh")], "moi_mot_canh")
    con(list(range(len(moi))), "moi_tat_ca")

    print(f"\nallocator = {args.allocator} | cua so {windows} | "
          f"{args.seeds} ho x {args.draws} boc | goc hat {GOC_HAT}")
    print(f"\n{'nhom':<38}{'n':>4}{'diem':>9}{'+-':>8}{'video dung/100 dong':>22}")
    print("-" * 82)
    ten_hien = {
        "cu": "bo CU (nguoi viet, 0% hai canh)",
        "moi_tat_ca": "bo MOI (ca 4 shard)",
        "moi_sach": "bo MOI sach (bo shard c lan truc)",
        "moi_mot_canh": "  |- MOT canh",
        "moi_hai_canh": "  |- HAI canh",
    }
    for k in ("cu", "moi_tat_ca", "moi_sach", "moi_mot_canh", "moi_hai_canh"):
        if k not in ket:
            continue
        m, sd, cv, n = ket[k]
        print(f"{ten_hien[k]:<38}{n:>4}{m:>9.4f}{sd:>8.4f}{cv:>13}/{n:<8}")

    print("\n=== DOC KET QUA ===")
    if "moi_mot_canh" in ket and "moi_hai_canh" in ket:
        a, sa, _, na = ket["moi_mot_canh"]
        b, sb, _, nb = ket["moi_hai_canh"]
        bien = max(sa, sb, 0.0005)
        chenh = 100 * (b / a - 1) if a else 0.0
        print(f"1) HAI canh vs MOT canh (cung bo, cung cach sinh): {b:.4f} vs {a:.4f} "
              f"= {chenh:+.1f}%")
        if abs(b - a) < 2 * bien:
            print(f"   -> HOA (chenh {abs(b-a):.4f} < 2 sigma = {2*bien:.4f}): "
                  "chua chung minh duoc cau hai canh kho hon.")
        else:
            print(f"   -> KHAC BIET THAT (> 2 sigma = {2*bien:.4f}).")
        print(f"   Luu y: n = {na} va {nb}, con nho — day la tin hieu, chua phai ket luan chac.")
    if "moi_mot_canh" in ket:
        a = ket["moi_mot_canh"][0]
        c = ket["cu"][0]
        print(f"2) TACH NGUYEN NHAN — MOT canh cua bo moi vs bo cu: {a:.4f} vs {c:.4f} "
              f"= {100*(a/c-1):+.1f}%")
        print("   Neu ngang nhau: khac biet o (1) den tu CAU TRUC THOI GIAN.")
        print("   Neu bo moi cao hon han: cau may sinh DE HON cau nguoi viet, va moi")
        print("   so sanh voi bo cu deu phai tru di phan de hon nay.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
