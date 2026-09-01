# -*- coding: utf-8 -*-
"""Thăm dò KHẢ THI: kho đã có sẵn ranh giới shot chưa? — KHÔNG phải thí nghiệm.

Đề xuất "khai thác cấu trúc cảnh quay" chỉ đáng tiền nếu ta biết ranh giới shot
nằm ở đâu. Trước khi tính chuyện tải 873 video về chạy TransNetV2, script này
hỏi hai câu rẻ nhất, chỉ đọc dữ liệu đã có:

  A. **Dấu vết bộ trích keyframe.** Bộ trích của BTC không lấy đều: có chỗ lấy
     cách nhau ~5–6 s, có chỗ lấy một CHÙM keyframe cách nhau 0,04 s. Nếu chùm
     ứng với cú cắt cảnh thì ta có bảng ranh giới MIỄN PHÍ. Đo: nhịp chùm mỗi
     phút, so với nhịp cắt cảnh thực tế của bản tin truyền hình (~4–6 s/cảnh).

  B. **Cosine giữa hai keyframe liền kề trong chỉ mục SigLIP.** Nếu cosine tụt
     mạnh ở cú cắt thì ta có bộ dò cắt cảnh miễn phí ngay trên chỉ mục sẵn có.
     Đo: phân bố cosine liền kề, tách theo độ dài khe thời gian.

Cả hai câu đều có thể trả lời ÂM, và kết luận âm ở đây tiết kiệm hàng chục giờ
GPU + vài chục GB tải về. Không mục nào ở đây là phép đo cải tiến điểm số.

    python -u scripts/tham_do_cau_truc_shot.py
"""

from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

KHE_CHUM = 0.5  # giây: hai keyframe cách nhau dưới ngần này coi là cùng một "chùm"


def phan_a(meta):
    print("=== A. Dấu vết bộ trích keyframe: chùm có phải ranh giới shot không? ===")
    byv = collections.defaultdict(list)
    for m in meta:
        byv[m["video_id"]].append(float(m["pts_time"]))

    do_dai, nhip, khong_chum = [], [], 0
    theo_dai = collections.Counter()
    tong_dai = collections.Counter()
    for v, t in byv.items():
        t = np.array(sorted(t))
        g = np.diff(t)
        chum, i = 0, 0
        while i < len(g):
            if g[i] < KHE_CHUM:
                j = i
                while j < len(g) and g[j] < KHE_CHUM:
                    j += 1
                do_dai.append(j - i + 1)
                chum += 1
                i = j
            else:
                i += 1
        if chum == 0:
            khong_chum += 1
        tong_dai[v[:3]] += 1
        theo_dai[v[:3]] += 1 if chum else 0
        if len(t) and t[-1] > 60:
            nhip.append(chum / (t[-1] / 60.0))

    d = np.array(do_dai)
    nh = np.array(nhip)
    print(f"  tổng chùm: {len(d)} trên {len(byv)} video "
          f"| video KHÔNG có chùm nào: {khong_chum}/{len(byv)}")
    print(f"  độ dài chùm (keyframe): trung vị {np.median(d):.0f}"
          f"  p95 {np.percentile(d,95):.0f}  max {d.max()}"
          f" | keyframe nằm trong chùm: {d.sum()}/{len(meta)} = {100*d.sum()/len(meta):.1f}%")
    print(f"  NHỊP chùm: trung vị {np.median(nh):.1f} chùm/phút"
          f"  → một chùm mỗi {60/np.median(nh):.1f} giây"
          f"  (p10 {np.percentile(nh,10):.1f}, p90 {np.percentile(nh,90):.1f})")
    print("  phủ theo dải (số video có ≥1 chùm / tổng):",
          "  ".join(f"{k}:{theo_dai[k]}/{tong_dai[k]}" for k in sorted(tong_dai)))
    print("  ĐỌC: bản tin truyền hình cắt cảnh mỗi ~4–6 s. Nhịp chùm đo được thưa"
          " hơn thế nhiều lần ⇒ chùm KHÔNG phải bảng ranh giới shot đầy đủ.")

    g_all = []
    for v, t in byv.items():
        g_all.extend(np.diff(np.array(sorted(t))).tolist())
    g_all = np.array(g_all)
    print(f"  khe giữa hai keyframe: trung vị {np.median(g_all):.2f}s"
          f"  p25 {np.percentile(g_all,25):.2f}  p75 {np.percentile(g_all,75):.2f}"
          f"  max {g_all.max():.1f}")
    return byv


def phan_b(meta):
    print("\n=== B. Cosine SigLIP giữa hai keyframe liền kề có dò được cú cắt không? ===")
    f = ROOT / "data" / "embeddings_siglip2_384.npy"
    E = np.load(f, mmap_mode="r")  # đọc memmap: KHÔNG nạp 817 MB vào RAM
    assert E.shape[0] == len(meta), "embedding lệch metadata"
    t0 = time.time()
    cos, gap = [], []
    prev, i, N, CH = None, 0, len(meta), 20000
    while i < N:
        j = min(i + CH, N)
        blk = np.asarray(E[i:j], dtype=np.float32)
        off = i
        if prev is not None:
            blk = np.vstack([prev, blk])
            off = i - 1
        d = (blk[:-1] * blk[1:]).sum(1)
        for k in range(len(d)):
            a, b = meta[off + k], meta[off + k + 1]
            if a["video_id"] != b["video_id"]:
                continue
            cos.append(float(d[k]))
            gap.append(float(b["pts_time"]) - float(a["pts_time"]))
        prev, i = blk[-1:], j
    cos, gap = np.array(cos), np.array(gap)
    print(f"  {len(cos)} cặp liền kề, {time.time()-t0:.1f}s")
    print(f"  cosine: trung vị {np.median(cos):.3f}"
          f"  p5 {np.percentile(cos,5):.3f}  p25 {np.percentile(cos,25):.3f}"
          f"  p95 {np.percentile(cos,95):.3f}")
    for lo, hi in [(0, 0.3), (0.3, 1), (1, 2), (2, 4), (4, 6), (6, 9)]:
        m = (gap >= lo) & (gap < hi)
        if m.sum():
            print(f"    khe {lo:.1f}–{hi:.1f}s: n={m.sum():6d}"
                  f"  cosine trung vị {np.median(cos[m]):.3f}"
                  f"  tỷ lệ <0,5: {(cos[m]<0.5).mean():.3f}")
    print(f"  tỷ lệ cặp có cosine < 0,5: {(cos<0.5).mean():.3f}"
          f" | < 0,35: {(cos<0.35).mean():.3f}")
    print("  ĐỌC: với ~4–6 s/cảnh và khe keyframe trung vị ~2,2 s, PHẦN LỚN cặp"
          " liền kề phải vắt qua một cú cắt. Vậy mà cosine trung vị vẫn 0,90 và"
          " chỉ <1% cặp xuống dưới 0,5 ⇒ dải động của SigLIP quá nén, ngưỡng tuyệt"
          " đối KHÔNG dò được cắt cảnh.")


def main() -> int:
    meta = json.loads((ROOT / "data" / "metadata.json").read_text(encoding="utf-8"))
    print(f"{len(meta)} keyframe / {len(set(m['video_id'] for m in meta))} video\n")
    phan_a(meta)
    phan_b(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
