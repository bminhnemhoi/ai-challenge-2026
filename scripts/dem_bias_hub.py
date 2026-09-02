"""Đếm bias HUB — cổng tất định 0 đồng cho đề xuất NNN, làm TRƯỚC mọi lần đọc TEST.

Đề xuất đầu bảng của `docs/PAPER_XEP_HANG_NOI_VIDEO.md`: khử bias-hub theo bank
câu hỏi (NNN, EMNLP 2024). Trực giác: một số khung là "hub" — chúng khớp với
*mọi* câu hỏi (phát thanh viên trong trường quay, đồ hoạ hiệu chương trình) nên
chiếm hạng cao bất kể câu hỏi là gì, đè lên keyframe đáp án thật. NNN trừ đi
phần "khớp với mọi thứ" ấy: điểm_mới(q, f) = sim(q, f) − k · mean_{b∈bank} sim(b, f).

Cơ chế này **trực giao** với chuẩn-hoá-theo-video đã ÂM: cái đã âm chuẩn hoá
trục KHUNG trong từng câu; cái này chuẩn hoá trục CÂU trên từng khung.

Trước khi tiêu bất kỳ lần đọc TEST nào, phép đếm này trả lời: **hub có thật
không?** Với mỗi mục mà keyframe đáp án KHÔNG đứng hạng-1 nội-video, so
bias-bank của các khung đứng TRÊN nó với bias-bank của chính nó:

    hub thật  ⟺  khung đứng trên có bias-bank CAO HƠN ở đa số mục

Ngưỡng công bố TRƯỚC khi chạy (theo lane paper): nếu ≤55% số mục cho thấy khung
đứng trên có bias cao hơn ⇒ **ÂM, dừng**, không tiêu lần đọc TEST nào cho NNN.

Bank: 141 câu — 25 round1 + 26 round_p1 + 30 round2 (đề thật) + 60 GT cũ.
KHÔNG chứa câu nào của bộ đo 132 mục (câu máy sinh) ⇒ không rò rỉ.

    python -u scripts/dem_bias_hub.py
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
from scripts.experiment_cap_thoi_gian import KhoSims  # noqa: E402
from scripts.make_submission import read_query_text  # noqa: E402


def nap_bank(root: Path):
    """141 câu văn bản: đề thật ba vòng + 60 GT cũ. Không đụng bộ đo 132 mục."""
    cau = []
    for d in ("round1/queries", "round_p1/queries", "round2/queries"):
        dd = root / d
        if not dd.is_dir():
            continue
        for p in sorted(dd.glob("*.txt")):
            if p.name.lower().endswith((".en.txt", ".vi.txt")):
                continue
            t = read_query_text(p)
            if t and t.strip():
                cau.append(t.strip())
    gt = json.loads((root / "data" / "ground_truth.json").read_text(encoding="utf-8"))
    cau += [g["kis_query_vi"] for g in gt if g.get("kis_query_vi")]
    return cau


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    ap.add_argument("--cache-uv", default=str(ROOT / "data" / "cache_bo_do_moi"))
    args = ap.parse_args()

    data = Path(args.data)
    moi = json.loads(Path(args.moi).read_text(encoding="utf-8"))
    uv = json.loads((Path(args.cache_uv) / "uv_moi.json").read_text(encoding="utf-8"))
    giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]
    sach = [moi[i] for i in giu]

    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    hang_of = {(m["video_id"], int(m["frame_idx"])): i for i, m in enumerate(meta)}
    kf = {}
    for m in meta:
        kf.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
    for v in kf:
        kf[v] = np.array(sorted(kf[v]))
    del meta

    bank = nap_bank(ROOT)
    print(f"bank: {len(bank)} câu (đề thật 3 vòng + 60 GT cũ; không giao bộ đo 132)")

    # bias-bank của TOÀN BỘ 177k khung: trung bình sim với 141 câu bank.
    # KhoSims cache theo văn bản nên chạy lại miễn phí.
    print("tính sim của 141 câu bank (KhoSims, cache) ...", flush=True)
    kho = KhoSims(args.data, False)
    tong = None
    for i, q in enumerate(bank, 1):
        s = kho.lay(q, "")
        tong = s.astype(np.float64) if tong is None else tong + s
        if i % 30 == 0:
            print(f"  {i}/{len(bank)}", flush=True)
    bias = (tong / len(bank)).astype(np.float32)
    print(f"bias-bank: mean {bias.mean():.4f}, sd {bias.std():.4f}, "
          f"p99 {np.percentile(bias, 99):.4f}")

    # --- phép đếm: khung đứng TRÊN keyframe đáp án có bias cao hơn không?
    ket = {"mot": [], "hai": []}
    for k, i in enumerate(giu):
        g = moi[i]
        vid = g["video_id"]
        a = kf.get(vid)
        if a is None or not len(a):
            continue
        kf_dung = int(a[int(np.argmin(np.abs(a - int(g["frame_idx"]))))])
        # ứng viên CÙNG video trong pool sản xuất, xếp theo điểm
        trong = sorted(((float(s), int(f)) for v, f, s, _lf in uv[i] if v == vid),
                       key=lambda t: -t[0])
        if not trong:
            continue
        hang = next((r for r, (_s, f) in enumerate(trong, 1) if f == kf_dung), None)
        if hang is None or hang == 1:
            continue  # đáp án vắng mặt hoặc đã hạng-1: NNN không có gì để sửa
        r_dap = hang_of.get((vid, kf_dung))
        tren = [hang_of.get((vid, f)) for _s, f in trong[: hang - 1]]
        tren = [r for r in tren if r is not None]
        if r_dap is None or not tren:
            continue
        hon = float(np.median([bias[r] for r in tren]) - bias[r_dap])
        nhom = "hai" if canh_cua(g) else "mot"
        ket[nhom].append(hon)

    print("\n=== PHÉP ĐẾM (chỉ mục mà đáp án có mặt nhưng KHÔNG hạng-1 nội-video) ===")
    tat_ca = []
    for nhom, ten in (("mot", "MỘT cảnh"), ("hai", "HAI cảnh")):
        a = np.array(ket[nhom])
        tat_ca += ket[nhom]
        if not len(a):
            print(f"  {ten}: 0 mục đủ điều kiện")
            continue
        duong = int((a > 0).sum())
        print(f"  {ten}: n={len(a)} | khung-trên có bias CAO hơn đáp án: "
              f"{duong}/{len(a)} = {100*duong/len(a):.0f}% | chênh bias trung vị "
              f"{np.median(a):+.4f}")
    a = np.array(tat_ca)
    duong = int((a > 0).sum())
    ty = 100 * duong / max(1, len(a))
    print(f"\n  TỔNG: {duong}/{len(a)} = {ty:.0f}% "
          f"(ngưỡng công bố trước: >55% mới đi tiếp)")
    print("\n=== KẾT LUẬN CỔNG ===")
    if ty > 55:
        print(f"  ĐI TIẾP: hub là có thật ở {ty:.0f}% số mục — NNN có mục tiêu để sửa.")
        print("  Bước sau: cài NNN (điểm − k·bias), quét k trên TUNE, đọc TEST một lần,")
        print("  bootstrap theo câu, bất biến k=0 bằng assert.")
    else:
        print(f"  ÂM, DỪNG: chỉ {ty:.0f}% ≤ 55% — khung đứng trên đáp án không phải hub.")
        print("  Ghi vào bảng cửa đóng, không tiêu lần đọc TEST nào cho NNN.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
