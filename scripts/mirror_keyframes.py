"""Mirror toàn bộ keyframe về đĩa, làm đường lui khi CDN chết giữa vòng thi.

Vì sao đáng một script riêng: khảo sát ngân sách vòng thi cho thấy trang
review.html — nơi tiêu 55 phút soát mắt, bước ăn điểm nhất — trước đây phụ
thuộc 100% vào CDN Hugging Face, trong khi tầng VLM (ít giá trị hơn) lại có
sẵn fallback đĩa. Mạng phòng thi chập một nhịp là công cụ đắt nhất chết trước.

Mirror dùng lại đúng cơ chế cache của ``VLMJudge._fetch``: ảnh 512px JPEG lưu
tại ``data/frames/<video>/NNN.jpg`` — trùng cấu trúc đường dẫn CDN, và cũng là
thứ review.html (cờ ``--local-mirror``) lẫn VLM cùng đọc. Chạy dở, Ctrl+C, chạy
lại: ảnh đã có trên đĩa được bỏ qua ngay, không tốn request nào.

Kích thước: ~35 KB/ảnh × 177.321 ảnh ≈ 6 GB. KHÔNG phải 30 GB — đó là cỡ ảnh
gốc; mirror lưu bản 512px đã đủ cho soát mắt và VLM.

    python scripts/mirror_keyframes.py                # cả kho, tiếp tục từ chỗ dở
    python scripts/mirror_keyframes.py --prefix L24   # một nhóm video
    python scripts/mirror_keyframes.py --videos-of round1/queries  # chỉ video có
                                                      # trong CSV của một vòng
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from src.core.vlm import VLMJudge  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--workers", type=int, default=12,
                    help="luồng tải song song; CDN chịu tốt, nghẽn thật là ~172k request HTTP")
    ap.add_argument("--prefix", default="", help="chỉ các video có mã bắt đầu bằng chuỗi này")
    ap.add_argument("--videos-of", default="",
                    help="thư mục CSV của một vòng — chỉ mirror các video xuất hiện trong đó")
    ap.add_argument("--limit", type=int, default=0, help="dừng sau n ảnh (để đo tốc độ)")
    args = ap.parse_args()

    judge = VLMJudge(args.data)          # không cần API key: chỉ dùng _fetch
    meta = json.loads((Path(args.data) / "metadata.json").read_text(encoding="utf-8"))

    keep_videos: set[str] | None = None
    if args.videos_of:
        keep_videos = set()
        for csv in Path(args.videos_of).glob("*.csv"):
            for line in csv.read_text(encoding="utf-8").splitlines():
                v = line.split(",", 1)[0].strip()
                if v:
                    keep_videos.add(v)
        print(f"gioi han theo {args.videos_of}: {len(keep_videos)} video")

    todo = []
    have = 0
    for m in meta:
        vid = m["video_id"]
        if args.prefix and not vid.startswith(args.prefix):
            continue
        if keep_videos is not None and vid not in keep_videos:
            continue
        thumb = judge.frame_dir / vid / f"{Path(m['frame_filename']).stem}.jpg"
        if thumb.is_file():
            have += 1
        else:
            todo.append((vid, m["frame_filename"]))
    if args.limit:
        todo = todo[: args.limit]

    print(f"da co {have:,} anh tren dia; can tai {len(todo):,}")
    if not todo:
        print("mirror day du — khong phai tai gi.")
        return 0

    t0 = time.time()
    ok = 0

    def one(job):
        nonlocal ok
        if judge._fetch(*job) is not None:
            ok += 1

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, _ in enumerate(ex.map(one, todo), 1):
            if i % 2000 == 0:
                rate = i / max(time.time() - t0, 1e-9)
                eta = (len(todo) - i) / max(rate, 1e-9)
                print(f"  {i:,}/{len(todo):,}  {rate:5.1f} anh/s  con ~{eta/60:.0f} phut"
                      f"  hong {judge.fetch_failures}", flush=True)

    print(f"\nxong: {ok:,}/{len(todo):,} anh trong {(time.time()-t0)/60:.1f} phut; "
          f"{judge.fetch_failures} anh khong tai duoc"
          + (f" ({judge.last_fetch_error})" if judge.fetch_failures else ""))
    # Mot mirror thieu vai anh van tot hon khong co mirror; nhung phai NOI ro,
    # vi nguoi ta se tin no trong phong thi.
    return 0 if judge.fetch_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
