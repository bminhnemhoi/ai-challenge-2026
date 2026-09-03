"""Nửa cửa §4.5 — điểm-cắt tương đối trong video, đo bằng thước trận tay đôi.

`docs/KE_HOACH_DINH_VI.md` §4.5: trên thang tương đối trong từng video, cú cắt
thật nằm ở phân vị 0,24 của cosine-liền-kề, đối chứng một cảnh ở 0,44 — có tín
hiệu, yếu. Nửa cửa còn lại: dùng nó làm đặc trưng xếp hạng nội-video cho nhóm
MỘT cảnh. Từ 03/09 mọi phép-đếm-trước của trục này đo bằng MỘT thước
(`QUYET_DINH_ENCODER_TRAKE.md` §3): trận tay đôi hạng-1 ↔ đáp-án, thắng ≥62%.

GIẢ THUYẾT ĐĂNG KÝ TRƯỚC (một chiều): keyframe-đáp-án GẦN CÚ CẮT hơn khung
hạng-1 cũ — tức cut(fd) > cut(f1), với cut(f) = 1 − phân_vị_trong_video của
cosine(f, keyframe liền trước). Căn cứ chọn chiều: khoảnh khắc BTC hỏi thường
mở đầu một diễn biến; với câu hai cảnh khung neo *được định nghĩa* là cú cắt.
Chiều ngược lại chỉ in để minh bạch, KHÔNG được dùng làm kết quả.

0 API, 0 GPU — chỉ embeddings mmap. Không đọc điểm, không đọc TEST.

    python -u scripts/dem_cat_tran_tay_doi.py
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

NGUONG = 62.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    ap.add_argument("--cache-uv", default=str(ROOT / "data" / "cache_bo_do_moi"))
    ap.add_argument("--hang-toi-da", type=int, default=3)
    args = ap.parse_args()

    data = Path(args.data)
    moi = json.loads(Path(args.moi).read_text(encoding="utf-8"))
    uv = json.loads((Path(args.cache_uv) / "uv_moi.json").read_text(encoding="utf-8"))
    giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]

    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    hang_of = {(m["video_id"], int(m["frame_idx"])): r for r, m in enumerate(meta)}
    kf = {}
    for m in meta:
        kf.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
    kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in kf.items()}
    del meta

    emb = np.load(data / "embeddings_siglip2_384.npy", mmap_mode="r")

    def cut_score(vid):
        """cut(f) theo phân vị trong video: cao = càng giống cú cắt."""
        a = kf[vid]
        rows = [hang_of.get((vid, int(f))) for f in a]
        ok = [i for i, r in enumerate(rows) if r is not None]
        e = np.asarray(emb[[rows[i] for i in ok]], dtype=np.float64)
        cos = (e[1:] * e[:-1]).sum(axis=1)  # cosine với keyframe liền trước
        # phân vị của cosine trong video; cut = 1 − phân_vị (cosine thấp = cắt)
        thu = cos.argsort().argsort() / max(1, len(cos) - 1)
        ra = {}
        for j, i in enumerate(ok[1:]):
            ra[int(a[i])] = 1.0 - float(thu[j])
        return ra  # keyframe đầu video không có "liền trước" — bỏ

    thang, nguoc, bo = 0, 0, 0
    n = 0
    for i in giu:
        g = moi[i]
        if canh_cua(g):
            continue
        vid, dap = g["video_id"], int(g["frame_idx"])
        a = kf.get(vid)
        if a is None or not len(a):
            continue
        fd = int(a[int(np.argmin(np.abs(a - dap)))])
        trong = sorted(((float(s), int(f)) for v, f, s, _lf in uv[i] if v == vid),
                       key=lambda t: -t[0])
        hang = next((r for r, (_s, f) in enumerate(trong, 1) if f == fd), None)
        if hang is None or not (2 <= hang <= args.hang_toi_da):
            continue
        f1 = trong[0][1]
        cs = cut_score(vid)
        if f1 not in cs or fd not in cs:
            bo += 1
            continue
        n += 1
        if cs[fd] > cs[f1]:
            thang += 1
        elif cs[fd] < cs[f1]:
            nguoc += 1

    print(f"số trận: {n} (bỏ {bo} vì keyframe đầu video không có cosine liền trước)")
    print(f"\n  GIẢ THUYẾT ĐĂNG KÝ TRƯỚC — đáp án GẦN cú cắt hơn: "
          f"{thang}/{n} = {100*thang/max(1,n):.0f}%")
    print(f"  (chiều ngược, chỉ để minh bạch:                    "
          f"{nguoc}/{n} = {100*nguoc/max(1,n):.0f}%)")
    ty = 100 * thang / max(1, n)
    print(f"\n=== KẾT LUẬN (ngưỡng ≥{NGUONG:.0f}%, một chiều, đăng ký trước) ===")
    if ty >= NGUONG:
        print("  ĐI TIẾP: đo đầy đủ 5 cổng với đặc trưng cut-score.")
    else:
        print("  ÂM, DỪNG: đóng nốt nửa cửa §4.5. Tín hiệu cắt tương-đối có thật")
        print("  ở mức phân bố (0,24 vs 0,44) nhưng không phân xử được trận tay đôi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
