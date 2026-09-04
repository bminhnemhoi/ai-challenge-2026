"""Quét MỌI keyframe của một video với một câu hỏi — đấu pháp #2, hết quét thưa.

Vòng 2 suýt mất TRAKE p2-8 vì quét 30/334 khung rồi kết luận "video không có
trái cây". Luật từ đó: câu 'đầu tiên / tất cả / lần đầu' phải quét ĐỦ khung —
nhưng đang làm tay. Công cụ này quét TOÀN BỘ keyframe của video (hoặc dải chỉ
định), hỏi MỘT câu chấm-điểm-nhiều-lớp (đấu pháp: 100=thấy rõ X, 50=chỉ thấy Y,
0=không có gì), in dải điểm theo frame để thấy NGAY khoảnh khắc đầu tiên/các
cụm xuất hiện.

Một-ảnh-một-request (giao thức đã kiểm — không có chỉ số ảnh để nhầm), xoay
vòng model, cache theo (video, frame, hash câu) — chạy lại 0 đồng, dừng giữa
chừng không mất gì.

    python scripts/quet_video_hoi.py --video Lxx_Vyyy \
        --hoi "Khung hình có TRÁI CÂY thật không? 100=có trái cây thấy rõ, 50=chỉ có tranh vẽ trái cây, 0=không có"
    # dải hẹp + bước nhảy: --tu 3000 --den 9000 --buoc 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.kiem_neo_don_anh import hoi_mot_anh  # noqa: E402
from src.core.vlm import DEFAULT_MODEL, VLMJudge  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--hoi", required=True,
                    help="câu chấm điểm nhiều lớp (100=... 50=... 0=...)")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--tu", type=int, default=None)
    ap.add_argument("--den", type=int, default=None)
    ap.add_argument("--buoc", type=int, default=1, help="lấy 1 mỗi N keyframe")
    ap.add_argument("--nguong", type=int, default=60,
                    help="điểm >= ngưỡng thì đánh dấu ★")
    args = ap.parse_args()

    meta = json.loads((Path(args.data) / "metadata.json").read_text(encoding="utf-8"))
    khung = sorted(
        ((int(m["frame_idx"]), m["frame_filename"]) for m in meta
         if m["video_id"] == args.video),
    )
    del meta
    if not khung:
        print(f"KHÔNG thấy video {args.video}")
        return 2
    if args.tu is not None:
        khung = [k for k in khung if k[0] >= args.tu]
    if args.den is not None:
        khung = [k for k in khung if k[0] <= args.den]
    khung = khung[:: max(1, args.buoc)]
    print(f"{args.video}: quét {len(khung)} keyframe "
          f"({khung[0][0]}..{khung[-1][0]}, bước {args.buoc})")
    print(f"câu hỏi: {args.hoi[:120]}\n")

    cdir = Path(args.data) / "cache_quet_hoi"
    cdir.mkdir(exist_ok=True)
    h_cau = hashlib.sha1(args.hoi.strip().encode()).hexdigest()[:10]
    f_cache = cdir / f"{args.video}_{h_cau}.json"
    cache = json.loads(f_cache.read_text(encoding="utf-8")) if f_cache.exists() else {}

    judge = VLMJudge(args.data, model=DEFAULT_MODEL)
    diem = {}
    t0 = time.time()
    for n, (f, fn) in enumerate(khung, 1):
        k = str(f)
        if k in cache:
            diem[f] = cache[k]
        else:
            try:
                blob = judge._fetch(args.video, fn)
                r = hoi_mot_anh(judge, blob, args.hoi)
                diem[f] = int(r.get("diem", -1))
            except Exception as exc:  # noqa: BLE001
                print(f"  frame {f}: LOI {type(exc).__name__} — dừng được, chạy"
                      f" lại sẽ tiếp từ đây (cache giữ)")
                break
            cache[k] = diem[f]
            if n % 10 == 0:
                f_cache.write_text(json.dumps(cache), encoding="utf-8")
                toc = n / max(1.0, time.time() - t0)
                con = (len(khung) - n) / max(0.1, toc)
                print(f"  {n}/{len(khung)} ({toc:.1f} khung/s, ~{con/60:.0f} phút nữa)",
                      flush=True)
    f_cache.write_text(json.dumps(cache), encoding="utf-8")

    # dải điểm — nhìn một phát ra khoảnh khắc đầu tiên và các cụm
    print(f"\n{'frame':>8}  điểm  ")
    dau_tien = None
    for f in sorted(diem):
        d = diem[f]
        sao = "★" if d >= args.nguong else " "
        if sao == "★" and dau_tien is None:
            dau_tien = f
        thanh = "#" * max(0, min(20, d // 5))
        print(f"{f:>8}  {d:>4} {sao} {thanh}")
    if dau_tien is not None:
        print(f"\n>>> KHOẢNH KHẮC ĐẦU TIÊN ≥{args.nguong}: frame {dau_tien}")
        print(">>> Soát MẮT khung đó trước khi chốt (review/CDN), rồi apply_picks.")
    else:
        print(f"\n>>> KHÔNG khung nào ≥{args.nguong} trong dải đã quét"
              f" ({len(diem)}/{len(khung)} khung có điểm).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
