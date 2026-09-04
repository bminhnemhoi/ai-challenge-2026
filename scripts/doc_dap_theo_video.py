"""Đọc đáp án Q&A từ MỘT VIDEO người chỉ định — vá đúng lỗi 57/158 "sai vì nhầm video".

Đo được (04/09): 57/81 câu Q&A sai của bộ đo sai vì đường tự động đọc từ video
SAI; khi người soát xác định được video đúng (cheat-sheet `goi_y.txt`, lời
thoại, title), cần một lệnh chạy ĐÚNG đường-trả-lời-đã-đo (ảnh gốc + lời thoại
+ OCR-prompt + ép đoán) trên video đó thay vì đọc tay từng khung.

Chọn khung trong video: (a) ứng viên pool của câu (nếu video nằm trong 400);
(b) không có → đoạn lời thoại khớp nhất với câu hỏi (timestamp → keyframe);
(c) không có thoại → 3 keyframe rải đều. Luôn kèm 2 lân cận mỗi khung neo.

    python scripts/doc_dap_theo_video.py --query round3/queries/query-1-qa.txt --video Lxx_Vyyy
    # nhiều video ứng viên: --video Lxx_Vyyy,Lxx_Vzzz  (mỗi video một đáp án để so)
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

from scripts.answer_qa import nap_loi_thoai, tra_loi_tu_ung_vien  # noqa: E402
from scripts.make_submission import read_query_text  # noqa: E402
from src.core.submission import Candidate  # noqa: E402
from src.core.transcripts import TranscriptIndex, tokenise  # noqa: E402
from src.core.vlm import DEFAULT_MODEL, VLMJudge  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query", required=True, help="file .txt của câu hỏi")
    ap.add_argument("--video", required=True, help="Lxx_Vyyy hoặc danh sách phẩy")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--khung", default=None,
                    help="ép frame cụ thể, vd 3400,5100 (bỏ qua tự chọn)")
    ap.add_argument("--ocr", type=int, default=1, help="1 = chèn chữ OCR (như sản xuất)")
    args = ap.parse_args()

    data = Path(args.data)
    cau = read_query_text(Path(args.query)) or ""
    print(f"đề: {cau[:160]}\n")

    meta_raw = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    meta = {(m["video_id"], int(m["frame_idx"])): m for m in meta_raw}
    by_n = {(m["video_id"], int(m["n"])): m for m in meta_raw}
    kf, fps_of = {}, {}
    for m in meta_raw:
        kf.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
        pt = float(m.get("pts_time") or 0)
        if pt > 1:
            fps_of[m["video_id"]] = int(m["frame_idx"]) / pt
    kf = {v: np.array(sorted(a)) for v, a in kf.items()}
    del meta_raw

    ti = TranscriptIndex().load_dir(ROOT.parent / "transcripts_full",
                                    data / "captions")
    caps = nap_loi_thoai(data)
    judge = VLMJudge(str(data), model=DEFAULT_MODEL)

    from scripts.make_submission import _DocDapAn  # dùng chu_ocr đã ship

    for vid in [v.strip() for v in args.video.split(",") if v.strip()]:
        a = kf.get(vid)
        if a is None or not len(a):
            print(f"[{vid}] KHÔNG có trong chỉ mục — kiểm lại mã video")
            continue
        if args.khung:
            neo = [int(x) for x in args.khung.split(",")]
        else:
            neo = []
            qt = set(t for t in tokenise(cau) if len(t) > 1)
            tot = []
            for t0, txt in (ti.segments.get(vid) or []):
                h = len(qt & set(tokenise(txt)))
                if h:
                    tot.append((h, t0))
            tot.sort(reverse=True)
            for _h, t0 in tot[:2]:
                f = int(t0 * fps_of.get(vid, 25.0))
                neo.append(int(a[int(np.argmin(np.abs(a - f)))]))
            if not neo:
                neo = [int(a[len(a) // 4]), int(a[len(a) // 2]), int(a[3 * len(a) // 4])]
        # dựng ứng viên: neo được điểm cao dần để giữ thứ tự
        cands = [Candidate(vid, f, 1.0 - 0.01 * i, 0) for i, f in enumerate(neo)]
        cau_day_du = cau
        if args.ocr:
            class _Gia:  # đủ thuộc tính cho chu_ocr
                pass
            g = _Gia()
            g.meta, g.by_n, g.data_dir, g._ocr_idx = meta, by_n, str(data), None
            ocr_t = _DocDapAn.chu_ocr(g, cands)
            if ocr_t:
                cau_day_du = (cau + "\nChữ đọc được trên các khung hình (OCR tự"
                              " động, có thể sai chính tả, chỉ dùng làm gợi ý): "
                              + ocr_t)
                print(f"[{vid}] + chèn OCR ({len(ocr_t)} ký tự)")
        try:
            dap, ghi = tra_loi_tu_ung_vien(judge, DEFAULT_MODEL, cands, meta,
                                           by_n, caps, cau_day_du)
        except Exception as exc:  # noqa: BLE001
            print(f"[{vid}] LỖI {type(exc).__name__}: {str(exc)[:80]}")
            continue
        print(f"[{vid}] khung neo {neo}")
        print(f"[{vid}] ĐÁP ÁN: {dap}")
        print(f"[{vid}] ghi chú: {ghi}\n")
    print("Nhớ: cập nhật CSV qua apply_picks (--picks \"<query>=video:frame\"),"
          " KHÔNG sửa tay.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
