"""GQE bước 2 — đếm trận tay đôi hạng-1 ↔ đáp-án với ensemble paraphrase.

Đặc tả: `docs/QUYET_DINH_ENCODER_TRAKE.md` §3.1. Trận lấy Y HỆT phép đếm PRF
(`dem_prf_rocchio.py`): câu MỘT cảnh sạch, keyframe-đáp-án đứng hạng 2–3
nội-video theo điểm sản xuất; trận = (khung hạng-1 cũ f1, keyframe-đáp-án fd).

Ba lá phiếu: câu gốc + p1 + p2, mỗi lá xếp theo sim SigLIP của chính nó tại
hai khung. Câu gốc về bản chất bầu cho f1 (chính nó tạo ra hạng), nên:

  * THƯỚC CHÍNH (công bố trước, majority-vote k=2): thắng ⟺ CẢ p1 VÀ p2 đều
    cho sim(fd) > sim(f1). Ngưỡng đi tiếp ≥62%.
  * chẩn đoán: tỷ lệ thắng từng paraphrase riêng lẻ; ensemble trung-bình-sim
    (cơ chế gốc của GQE, arXiv:2408.07249).

Paraphrase đã soi tay 20 mẫu (giữ chi tiết định danh, không drift) trước khi
chạy phép đếm này. 0 lần đọc TEST — chỉ đếm trận, không đọc điểm.

RAM: PHẢI chạy qua wrapper khi tiến trình encode PE còn sống:
    python -u scripts/chay_gon_ram.py scripts/dem_gqe_tran.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.do_cap_thoi_gian_moi import canh_cua  # noqa: E402
from scripts.experiment_cap_thoi_gian import KhoSims  # noqa: E402

NGUONG = 62.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    ap.add_argument("--cache-uv", default=str(ROOT / "data" / "cache_bo_do_moi"))
    ap.add_argument("--cache-gqe", default=str(ROOT / "data" / "cache_gqe"))
    ap.add_argument("--hang-toi-da", type=int, default=3)
    args = ap.parse_args()

    data = Path(args.data)
    moi = json.loads(Path(args.moi).read_text(encoding="utf-8"))
    uv = json.loads((Path(args.cache_uv) / "uv_moi.json").read_text(encoding="utf-8"))
    para = json.loads((Path(args.cache_gqe) / "paraphrase.json").read_text(encoding="utf-8"))
    giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]

    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    hang_of = {(m["video_id"], int(m["frame_idx"])): r for r, m in enumerate(meta)}
    kf = {}
    for m in meta:
        kf.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
    kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in kf.items()}
    del meta

    # --- gom trận Y HỆT dem_prf_rocchio.py
    tran = []
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
        r1 = hang_of.get((vid, trong[0][1]))
        rd = hang_of.get((vid, fd))
        h = hashlib.sha1(g["kis_query_vi"].strip().encode()).hexdigest()[:16]
        if r1 is None or rd is None or h not in para:
            continue
        tran.append((g["kis_query_vi"], para[h]["p1"], para[h]["p2"], r1, rd))

    print(f"số trận: {len(tran)} (phải khớp phép đếm PRF: 16)")

    kho = KhoSims(args.data, False)
    thang_ca_hai, thang_p1, thang_p2, thang_mean, thang_goc = 0, 0, 0, 0, 0
    for goc, p1, p2, r1, rd in tran:
        vg, v1, v2 = kho.lay(goc, ""), kho.lay(p1, ""), kho.lay(p2, "")
        b_g = float(vg[rd]) > float(vg[r1])
        b_1 = float(v1[rd]) > float(v1[r1])
        b_2 = float(v2[rd]) > float(v2[r1])
        thang_goc += b_g
        thang_p1 += b_1
        thang_p2 += b_2
        thang_ca_hai += b_1 and b_2
        m_d = (float(vg[rd]) + float(v1[rd]) + float(v2[rd])) / 3
        m_1 = (float(vg[r1]) + float(v1[r1]) + float(v2[r1])) / 3
        thang_mean += m_d > m_1

    n = len(tran)
    print(f"\n=== KẾT QUẢ (n={n} trận) ===")
    print(f"  câu GỐC (sim thô, đối chiếu):        {thang_goc}/{n} = {100*thang_goc/n:.0f}%")
    print(f"  p1 riêng lẻ:                         {thang_p1}/{n} = {100*thang_p1/n:.0f}%")
    print(f"  p2 riêng lẻ:                         {thang_p2}/{n} = {100*thang_p2/n:.0f}%")
    print(f"  THƯỚC CHÍNH — cả p1 VÀ p2 (k=2):     {thang_ca_hai}/{n} = "
          f"{100*thang_ca_hai/n:.0f}%")
    print(f"  chẩn đoán — ensemble trung-bình-sim: {thang_mean}/{n} = "
          f"{100*thang_mean/n:.0f}%")
    ty = 100 * thang_ca_hai / n
    print(f"\n=== KẾT LUẬN (ngưỡng công bố trước ≥{NGUONG:.0f}%) ===")
    if ty >= NGUONG:
        print("  ĐI TIẾP: đo đầy đủ 5 cổng, biến thể hẹp (xếp lại nội-video).")
    else:
        print(f"  ÂM, DỪNG: {ty:.0f}% < {NGUONG:.0f}%. Ghi bảng cửa đóng.")
        print("  Theo §10d PAPER_XEP_HANG_NOI_VIDEO.md: không gian tín-hiệu")
        print("  training-free cho trục nội-video một cảnh cạn theo văn liệu.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
