"""R3 — đáp án GT có nằm trong chữ OCR quanh keyframe đáp án không? (0 đồng sau OCR)

`docs/NGHIEN_CUU_DA_NGUON_0309.md` R3 (KOCRBench: Gemini +12 điểm % khi chèn
OCR vào prompt; kênh ghép OCR+ASR của MMMORRF). Phép đếm cận-trên rẻ nhất:
nếu đáp án chuẩn xuất hiện trong OCR của keyframe-đáp-án ±1 lân cận thì kênh
"chèn chữ OCR vào prompt trả lời" có cửa; đặc biệt đếm riêng nhóm câu ĐANG SAI
(`trang_thai_nen.json` — NEN 77/158 đúng).

Khớp: (a) chuỗi con nguyên văn (thường hoá chữ thường), hoặc (b) MỌI token của
đáp án đều có trong tập token OCR (đáp án ngắn: tên/số/biển hiệu).

NGƯỠNG tiền-đăng-ký: ≥1 câu đang SAI có đáp án nằm sẵn trong OCR → làm A/B
chèn "Chữ xuất hiện trên hình: ..." vào prompt; ship khi net ≥ +1 câu và KHÔNG
câu đúng nào bị lật sai.

    python -u scripts/dem_dap_an_trong_ocr.py            # dùng cache OCR có gì đếm nấy
    python -u scripts/dem_dap_an_trong_ocr.py --quet     # OCR nốt khung còn thiếu
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.do_cap_thoi_gian_moi import canh_cua  # noqa: E402


def chuan_hoa(t: str) -> str:
    t = unicodedata.normalize("NFC", t or "").lower()
    return re.sub(r"\s+", " ", t).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    ap.add_argument("--quet", action="store_true", help="OCR nốt khung còn thiếu")
    ap.add_argument("--lan_can", type=int, default=1)
    args = ap.parse_args()

    data = Path(args.data)
    moi = json.loads(Path(args.moi).read_text(encoding="utf-8"))
    giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]
    tt = json.loads((data / "cache_nhan_tu_qa" / "trang_thai_nen.json")
                    .read_text(encoding="utf-8"))
    dung_cua = {r["chi_so_sach"]: r["dung"] for r in tt}

    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    ten_of, kf = {}, {}
    for m in meta:
        ten_of[(m["video_id"], int(m["frame_idx"]))] = m["frame_filename"]
        kf.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
    kf = {v: np.array(sorted(a)) for v, a in kf.items()}
    del meta

    # khung đích: keyframe-đáp-án ±lan_can
    dich = []  # (k_sach, video, [frame...])
    for k, i in enumerate(giu):
        g = moi[i]
        a = kf.get(g["video_id"])
        if a is None or not len(a) or not g.get("vqa_answer"):
            continue
        j = int(np.argmin(np.abs(a - int(g["frame_idx"]))))
        fs = [int(a[x]) for x in range(max(0, j - args.lan_can),
                                       min(len(a), j + args.lan_can + 1))]
        dich.append((k, g["video_id"], fs, chuan_hoa(g["vqa_answer"])))

    from src.core.ocr import OCRIndex
    idx = OCRIndex(str(data), langs=["vi", "en"])

    if args.quet:
        thieu = [(v, f, ten_of[(v, f)]) for _k, v, fs, _c in dich for f in fs
                 if str(f) not in idx._video(v) and (v, f) in ten_of]
        print(f"OCR nốt {len(thieu)} khung còn thiếu ...", flush=True)
        idx.read_frames(thieu)

    co_ocr, thieu_ocr = 0, 0
    hit_dung, hit_sai, n_dung, n_sai = 0, 0, 0, 0
    vi_du = []
    for k, v, fs, chuan in dich:
        texts = []
        cache_v = idx._video(v)
        for f in fs:
            r = cache_v.get(str(f))
            if r:
                texts += [x[1] if isinstance(x, (list, tuple)) else str(x) for x in r]
        if not any(str(f) in cache_v for f in fs):
            thieu_ocr += 1
            continue
        co_ocr += 1
        ocr = chuan_hoa(" ".join(map(str, texts)))
        tk_ocr = set(ocr.split())
        tk_ans = [t for t in chuan.split() if t]
        khop = bool(tk_ans) and (chuan in ocr or all(t in tk_ocr for t in tk_ans))
        d = dung_cua.get(k)
        if d:
            n_dung += 1
            hit_dung += khop
        else:
            n_sai += 1
            hit_sai += khop
            if khop and len(vi_du) < 6:
                vi_du.append((k, chuan[:60]))

    print(f"\nmục có OCR quanh keyframe-đáp-án: {co_ocr} | chưa OCR: {thieu_ocr}")
    print(f"  nhóm đang ĐÚNG (n={n_dung}): đáp án trong OCR ở {hit_dung} mục")
    print(f"  nhóm đang SAI  (n={n_sai}): đáp án trong OCR ở {hit_sai} mục"
          f"  <-- NGƯỠNG: ≥1 là kênh có cửa")
    for k, c in vi_du:
        print(f"    ví dụ mục sách #{k}: đáp án '{c}' nằm sẵn trong OCR")
    print("\nKẾT LUẬN:", "ĐI TIẾP — làm A/B chèn OCR vào prompt trả lời."
          if hit_sai >= 1 else
          ("ÂM trên phần đã OCR — đợi sweep phủ thêm rồi đếm lại." if thieu_ocr
           else "ÂM, cửa đóng."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
