"""Phép-đếm-trước ③ PRF Rocchio — thước TRẬN TAY ĐÔI hạng-1 ↔ đáp-án, 0 API 0 GPU.

Đặc tả: `docs/QUYET_DINH_ENCODER_TRAKE.md` §3.2. Cơ chế: kéo truy vấn về phía
ảnh — `q' = q + λ·c` với `c` = trung bình embedding ảnh của top-m ứng viên
toàn cục thuộc video KHÁC (leave-one-video-out bắt buộc: video đang xếp lại
không được góp vào `c`, nếu không phép đo thành vòng tự xác nhận).

Vì embedding đã chuẩn hoá L2 và mẫu số |q'| chung cho mọi khung trong video,
xếp theo cos(q', e_i) tương đương xếp theo  s(i) + λ·(c·e_i)  với s(i) là điểm
sản xuất (chính là thứ đã định nghĩa hạng trong trận đấu). Đây cũng đúng là
biến thể HẸP sẽ ship nếu đạt: giữ điểm sản xuất, chỉ cộng chứng cứ ảnh-với-ảnh.

Thước quyết định (công bố TRƯỚC, docs/QUYET_DINH_ENCODER_TRAKE.md §3):
trên các câu MỘT cảnh mà keyframe-đáp-án đứng hạng 2–3 nội-video, tín hiệu
phải cho đáp án THẮNG khung hạng-1 cũ ở **≥62%** số trận thì mới đi tiếp.
λ=0 là đối chứng bắt buộc: thắng đúng 0% (đáp án hạng ≥2 thua theo định nghĩa).
KHÔNG đọc điểm ở phép đếm này — chỉ đếm trận.

    python -u scripts/dem_prf_rocchio.py
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

LAM = (0.0, 0.25, 0.5, 1.0)
M = (3, 5, 10)
NGUONG = 62.0  # % trận thắng, công bố trước


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    ap.add_argument("--cache-uv", default=str(ROOT / "data" / "cache_bo_do_moi"))
    ap.add_argument("--hang-toi-da", type=int, default=3,
                    help="lấy câu có đáp án đứng hạng 2..N nội-video (mặc định 3)")
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

    # --- gom trận: câu một cảnh, đáp án đứng hạng 2..N nội-video
    tran = []  # (i, row_f1, s_f1, row_fd, s_fd, hang_dap)
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
        s1, f1 = trong[0]
        sd = next(s for s, f in trong if f == fd)
        r1, rd = hang_of.get((vid, f1)), hang_of.get((vid, fd))
        if r1 is None or rd is None:
            continue
        # top-m toàn cục thuộc video KHÁC (leave-one-video-out)
        khac = [(v, int(f)) for v, f, s, _lf in
                sorted(uv[i], key=lambda t: -float(t[2])) if v != vid]
        rows_khac = [hang_of.get(vf) for vf in khac]
        rows_khac = [r for r in rows_khac if r is not None]
        tran.append((i, r1, s1, rd, sd, hang, rows_khac))

    print(f"số trận (một cảnh, đáp án hạng 2..{args.hang_toi_da} nội-video): "
          f"{len(tran)}")
    if len(tran) < 10:
        print("QUÁ ÍT trận — không kết luận được, dừng.")
        return 0

    print(f"\n{'λ':>6}{'m':>4}{'đáp án thắng':>15}{'tỷ lệ':>8}")
    print("-" * 35)
    dat = []
    for m_top in M:
        for lam in LAM:
            thang = 0
            for _i, r1, s1, rd, sd, _h, rows_khac in tran:
                c = np.asarray(emb[sorted(rows_khac[:m_top])], dtype=np.float64).mean(axis=0)
                d1 = s1 + lam * float(c @ np.asarray(emb[r1], dtype=np.float64))
                dd = sd + lam * float(c @ np.asarray(emb[rd], dtype=np.float64))
                thang += dd > d1
            ty = 100 * thang / len(tran)
            if lam == 0.0:
                assert thang == 0, "đối chứng λ=0 phải thắng 0 trận"
                continue
            print(f"{lam:>6.2f}{m_top:>4}{thang:>10}/{len(tran):<4}{ty:>7.0f}%",
                  flush=True)
            dat.append((lam, m_top, ty))

    tot = max(dat, key=lambda t: t[2])
    print(f"\nđối chứng λ=0: thắng 0/{len(tran)} (đúng định nghĩa) ✓")
    print(f"tốt nhất: λ={tot[0]}, m={tot[1]} -> {tot[2]:.0f}% "
          f"(ngưỡng công bố trước: ≥{NGUONG:.0f}%)")
    print("\n=== KẾT LUẬN PHÉP ĐẾM ===")
    if tot[2] >= NGUONG:
        print("  ĐI TIẾP: đo đầy đủ 5 cổng, biến thể hoán-vị-giữ-điểm.")
    else:
        print("  ÂM, DỪNG: PRF ảnh-với-ảnh không phân xử được trận tay đôi.")
        print("  Ghi bảng cửa đóng; ②GQE là đề xuất sống cuối của trục nội-video.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
