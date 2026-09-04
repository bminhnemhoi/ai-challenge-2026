"""RUNBOOK vòng thi — một lệnh chạy trọn chuỗi máy, rồi in checklist người.

Đấu pháp đã kiểm chứng (memory `aic-playbook-vong-thi`, vòng 2: 8,6 → 10,0):
máy làm xong phần nền trong ~15 phút, TOÀN BỘ thời gian còn lại dành cho vòng
người-soát — nơi thực sự ăn điểm. Script này bảo đảm phần máy không quên bước
nào và phần người có checklist trước mặt.

    python scripts/chay_vong_thi.py --queries round3/queries --out round3/out

Từng giai đoạn có cờ bỏ qua nếu đã chạy (an toàn chạy lại — mọi thứ cache).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

CHECKLIST = """
================== PHẦN NGƯỜI — CHECKLIST ĂN ĐIỂM (vòng 2 đã kiểm chứng) ==================

NGAY BÂY GIỜ, theo thứ tự:
 1. LỜI THOẠI TRƯỚC TIÊN — với TỪNG câu Q&A, dò BM25 ngay (5/9 đáp án vòng 2 lộ nguyên văn):
      python scripts/search_transcripts.py "<cụm từ nghi là đáp án/bối cảnh>" -n 5
    Title video cũng là bằng chứng (MỰC QUE, tên đội...). Link YouTube in kèm — đưa người soát.
 2. Mở review.html — soát từng câu, ưu tiên: (a) câu Q&A, (b) câu hai cảnh, (c) TRAKE.
 2b. Câu Q&A mà video hệ chọn SAI (soát bằng goi_y.txt/lời thoại) → chạy lại đáp án
    trên ĐÚNG video người tìm ra (57/158 câu sai của bộ đo là vì nhầm video):
      python scripts/doc_dap_theo_video.py --query <file đề> --video Lxx_Vyyy
 3. Câu SỐ/CHỮ trên màn hình → ĐỌC ẢNH GỐC bằng gpt-5.2, bắt chép nguyên văn:
      python scripts/read_answer.py --video Lxx_Vyyy --frames <f> --provider openai --max-side 1900 --question "..."
    (answer_qa tự động chỉ là NHÁP — vòng 2 nó sai 4/9.)
 4. HEDGE THEO DÒNG — hai giả thuyết (video / đáp án / cách hiểu đề TRAKE) cùng nằm top-5.
    TRAKE không ràng buộc thứ tự cột: nộp song song "đầu-tiên-toàn-video" và "chuỗi-liên-tiếp".
 5. Câu "đầu tiên / tất cả": quét ĐỦ khung của video, cấm lấy mẫu thưa (suýt mất p2-8).
    Đã có công cụ quét TỰ ĐỘNG toàn bộ keyframe + mã điểm nhiều lớp (~15-20 phút/video,
    dừng/chạy lại thoải mái, in dải điểm thấy ngay khoảnh khắc đầu tiên):
      python scripts/quet_video_hoi.py --video Lxx_Vyyy --hoi "...? 100=... 50=... 0=..."
 6. MỌI CHỈNH SỬA qua apply_picks (CẤM sửa CSV tay — vòng 2 từng làm hỏng file):
      python scripts/apply_picks.py --queries <q> --out <out> --picks "<query>=Lxx_Vyyy:frame,..."
 7. Trước lượt nộp cuối: python scripts/verify_zip.py <out>/submission.zip
    và đối chiếu chéo với người soát thứ hai. LƯỢT CUỐI MỚI ĐƯỢC TÍNH. Có 3 lượt — dùng sớm 1 lượt
    để giữ bản an toàn trên hệ thống BTC ngay khi máy chạy xong.

 8. Câu nào truy xuất ra RÁC (top toàn video lạc đề): VIẾT TAY bản dịch tiếng Anh
    <tên câu>.en.txt đặt cạnh file đề rồi chạy lại RIÊNG câu đó (lever đã đo — bản
    dịch tay thắng dịch máy; giúp TRAKE ít nhất ngang KIS):
      python scripts/make_submission.py --queries <thư mục chỉ chứa câu đó> --out <out2>
 9. Rút lui khẩn cấp (đã diễn tập chiều 04/09, chạy sạch): thêm cờ
      --ocr-prompt 0 --canh-b 0 --hoan-vi-canh-b 0 --allocator hybrid
    để về đúng cấu hình vòng 2 cho một câu/một lượt chạy bất kỳ.

BẪY ĐÃ TỪNG MẤT ĐIỂM: title bị cắt ngắn rồi suy diễn; hai bản tin sáng/chiều HTV phát cùng
clip (phân xử bằng chi tiết ĐẾM ĐƯỢC); kết luận "video không có X" khi mới xem vài khung.
===========================================================================================
"""


def chay(ten, lenh, log_dir):
    t0 = time.time()
    print(f"\n=== {ten}: {' '.join(lenh)}", flush=True)
    log = log_dir / f"{ten}.log"
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run([sys.executable, "-u"] + lenh, cwd=ROOT, stdout=f,
                           stderr=subprocess.STDOUT)
    duoi = log.read_text(encoding="utf-8", errors="replace").splitlines()[-4:]
    for d in duoi:
        print(f"    {d}")
    print(f"    -> {'OK' if r.returncode == 0 else 'LOI ' + str(r.returncode)}"
          f" ({time.time() - t0:.0f}s), log: {log}")
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-ocr", type=int, default=24)
    ap.add_argument("--bo-ocr", action="store_true", help="bỏ giai đoạn OCR")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = out / "logs"
    log_dir.mkdir(exist_ok=True)

    # 1. đường ống chính — mọi lever tốt nhất đã là mặc định
    if not chay("make_submission",
                ["scripts/make_submission.py", "--queries", args.queries,
                 "--out", str(out)], log_dir):
        print("\nDUNG: make_submission loi — doc log roi chay lai (cache giu).")
        return 1

    # 2. kiểm zip ngay — có bản nộp an toàn sớm nhất có thể
    chay("verify_zip", ["scripts/verify_zip.py", str(out / "submission.zip")],
         log_dir)

    # 3. OCR ứng viên của vòng (~25 phút nền các vòng trước; cache nên chạy lại rẻ)
    if not args.bo_ocr:
        chay("run_ocr", ["scripts/run_ocr.py", "--queries", args.queries,
                         "--top", str(args.top_ocr)], log_dir)

    # 4. trang soát tay
    chay("review_page", ["scripts/build_review_page.py", "--queries",
                         args.queries, "--run-out", str(out),
                         "--out", str(out / "review.html")], log_dir)

    # 5. cheat-sheet lời thoại + OCR cho từng câu (đấu pháp #1, tự động hoá)
    chay("goi_y", ["scripts/do_tim_goi_y.py", "--queries", args.queries,
                   "--ra", str(out / "goi_y.txt")], log_dir)

    print(CHECKLIST)
    print(f"cheat-sheet lời thoại+OCR từng câu: {out / 'goi_y.txt'}")
    print(f"zip để nộp SỚM (giữ bản an toàn): {out / 'submission.zip'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
