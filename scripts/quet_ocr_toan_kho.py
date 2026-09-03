"""OCR TOÀN KHO — quét 177k keyframe qua đêm, nhiều tiến trình, resumable từng video.

`docs/KIEN_TRUC_VA_HUONG_CAI_THIEN.md` §4.2: 48% khung hình có chữ và chữ đó là
dòng tiêu đề tin tức — thứ trả lời trực tiếp nhiều câu Q&A. Hiện chỉ OCR ứng
viên từng vòng (~564 khung); có toàn kho thì `search_ocr.py` thành công cụ TÌM
KIẾM thật và §4.1 (gộp 3 nguồn ứng viên) có nguồn thứ ba để gộp.

Thiết kế:
  * chia THEO VIDEO cho N tiến trình — cache của `OCRIndex` ghi mỗi video một
    file nên các tiến trình không bao giờ đụng file của nhau;
  * video đã cache đủ thì bỏ qua ⇒ giết/chạy lại thoải mái, tiếp đúng chỗ;
  * dải ưu tiên tin tức trước (mặc định L21,L22,L23,L27,L24,L28,L29,L30,L25,L26
    — L26 79k khung để cuối, chạy được đến đâu hay đến đó);
  * mỗi tiến trình một EasyOCR reader (~1GB RAM) — mặc định 4 worker ≈ 4GB.

    python -u scripts/quet_ocr_toan_kho.py --workers 4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

DAI_MAC_DINH = "L21,L22,L23,L27,L24,L28,L29,L30,L25,L26"

_idx = None


def _khoi_tao(data_dir: str, langs: str):
    global _idx
    from src.core.ocr import OCRIndex
    _idx = OCRIndex(data_dir, langs=langs.split(","))


def _quet_video(job):
    vid, triples = job
    t0 = time.time()
    try:
        n = _idx.read_frames(triples)
        return vid, n, time.time() - t0, ""
    except Exception as exc:  # noqa: BLE001
        return vid, -1, time.time() - t0, f"{type(exc).__name__}: {str(exc)[:80]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--dai", default=DAI_MAC_DINH)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--langs", default="vi,en")
    args = ap.parse_args()

    meta = json.loads((Path(args.data) / "metadata.json").read_text(encoding="utf-8"))
    theo_video = {}
    for m in meta:
        theo_video.setdefault(m["video_id"], []).append(
            (m["video_id"], int(m["frame_idx"]), m["frame_filename"]))
    del meta

    uu_tien = {d: i for i, d in enumerate(args.dai.split(","))}
    videos = sorted((v for v in theo_video if v[:3] in uu_tien),
                    key=lambda v: (uu_tien[v[:3]], v))

    # bỏ video đã cache đủ (đọc file cache trực tiếp, rẻ hơn dựng OCRIndex)
    cache_dir = Path(args.data) / "ocr"
    con_lai = []
    da_xong = 0
    for v in videos:
        p = cache_dir / f"{v}.json"
        if p.exists():
            try:
                da = json.loads(p.read_text(encoding="utf-8"))
                if len(da) >= len(theo_video[v]):
                    da_xong += 1
                    continue
            except Exception:  # noqa: BLE001
                pass
        con_lai.append(v)

    tong_khung = sum(len(theo_video[v]) for v in con_lai)
    print(f"video: {len(videos)} trong dải ưu tiên | đã xong {da_xong} | "
          f"còn {len(con_lai)} ({tong_khung} khung) | {args.workers} worker",
          flush=True)

    t0 = time.time()
    xong_khung = 0
    with mp.Pool(args.workers, initializer=_khoi_tao,
                 initargs=(args.data, args.langs)) as pool:
        for i, (vid, n, giay, loi) in enumerate(
                pool.imap_unordered(_quet_video,
                                    ((v, theo_video[v]) for v in con_lai)), 1):
            if loi:
                print(f"  [{i}/{len(con_lai)}] {vid}: LOI {loi}", flush=True)
                continue
            xong_khung += n
            if i % 10 == 0 or n > 200:
                toc_do = xong_khung / max(1.0, time.time() - t0)
                print(f"  [{i}/{len(con_lai)}] {vid}: {n} khung/{giay:.0f}s | "
                      f"tổng {xong_khung} khung, {toc_do:.1f} khung/s toàn cục",
                      flush=True)

    print(f"\nXONG đợt này: {xong_khung} khung mới / {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
