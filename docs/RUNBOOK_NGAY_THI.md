# RUNBOOK NGÀY THI — vòng sau

Bản này là trình tự thao tác, không phải chiến thuật (chiến thuật: `docs/TOI_NAY.md`
và bảng đấu pháp trong memory). Mọi lệnh chạy từ gốc repo, PowerShell hay bash đều được.

---

## 0. TRƯỚC ngày thi (làm xong từ hôm trước, không để tới giờ G)

```
git pull                                  # bản mới nhất
python -m pytest -q tests src/task3_trake/tests src/task2_vqa   # phải xanh 100%
python scripts/so_sanh_allocator.py       # cổng allocator phải in "GIỮ ĐƯỢC" cả 2 nửa
```

- [ ] **Chỉ mục batch 2** đã dựng và nằm trong `data/` (nếu BTC đã phát dữ liệu mới).
      Video không có vector = 0 tuyệt đối. Dựng GIỐNG HỆT cách cũ:
      `notebooks/index-siglip2.ipynb` trên Colab. CẤM nhân dịp đổi encoder.
      Sau khi thay chỉ mục: `python scripts/experiment_phu_quet_luoi.py --refresh`
      và `--tune-phia le --refresh` (luật sau merge — sigma/nhiệt tính trên pool cũ).
- [ ] Mirror ảnh đủ: `python scripts/mirror_keyframes.py --workers 24` chạy tới khi
      "hong" ngừng giảm (script resumable, chạy lại bao nhiêu lần cũng được).
- [ ] `.env` có key Gemini + OpenAI, `python scripts/read_answer.py --help` chạy được.
- [ ] Đề phòng: in sẵn trang này.

## 1. Khi nhận đề (mỗi lượt)

```
# 1. Giải nén đề vào round<N>/de/queries (mỗi câu 1 file .txt)
# 2. Sinh bản nộp chính — allocator theo cờ ĐÃ CHỐT trong docs/SHIP_PHU_XAC_SUAT.md:
python scripts/make_submission.py --queries round<N>/de/queries --out round<N>/run1
# (allocator mặc định là cái đã qua cổng; make_submission in allocator ra dòng 2 —
#  NHÌN dòng đó, đừng đoán)

# 3. Dựng trang duyệt:
python scripts/build_review_page.py --run round<N>/run1 --out round<N>/review.html --local-mirror
```

**Lệnh rút lui** (nếu giữa trận nghi allocator mới có vấn đề trên phân bố đề thật):

```
python scripts/make_submission.py --queries round<N>/de/queries --out round<N>/run1b --allocator hybrid
```

Hybrid cách đúng MỘT cờ. Không sửa code giữa trận, không chỉnh tham số coverage
giữa trận (vùng an toàn nếu bị ép: nhiệt 0,015–0,02, sigma 20–30, nửa 6–10 —
nhưng mặc định đã là đỉnh của vùng đó rồi).

## 2. Trước MỖI lần upload (30 giây, không bỏ qua)

```
python -m pytest tests/test_page_export_matches_pipeline.py -q   # trang = pipeline
```

- Zip phải qua `verify_submission_zip` sạch (make_submission/apply_picks tự chạy —
  đọc dòng cuối: "format check passed" mới được nộp).
- File nào bị verifier kêu "only N rows" là có thứ vừa cắt cụt nó — dừng, tìm nguyên nhân,
  KHÔNG nộp file thiếu dòng.
- KHÔNG BAO GIỜ sửa CSV bằng tay/Excel. Mọi sửa chữa đi qua `apply_picks.py`
  (nó giữ nguyên allocator của run nhờ `allocator.txt`).

## 2b. Điền đáp án Q&A (làm NGAY sau make_submission, trước khi soát)

```
python scripts/answer_qa.py --queries round<N>/de/queries --out round<N>/run1 --repackage
```

Mặc định đã là cấu hình **đã đo**: ảnh gốc 1900px + 4 keyframe lân cận + 2 video
dự phòng + lời thoại ±30 s + prompt cấm bỏ trống — **86,7%** độ chính xác đáp án
trên TEST so với 63,3% của đường cũ (`docs/NGHIEN_CUU_SOTA.md` §1①).

- Câu nào model trả `confidence` thấp → ưu tiên soát tay, nó vẫn đoán chứ không
  bỏ trống (bỏ trống là 0 điểm chắc chắn).
- **Không bao giờ để trống ô đáp án**, kể cả khi không chắc. Sai và trống đều 0
  điểm, nhưng đoán thì còn cơ hội.
- Câu nào cần đọc chữ nhỏ (biển số, bảng hiệu, con số thập phân) → chốt lại bằng
  `python scripts/read_answer.py --video <V> --frames <f> --question "..." --provider openai`.
- Đường lui nếu nghi ngờ: `--neo 0` quay về đường cũ (12 thumbnail 512px).

## 3. Sửa theo góp ý người xem (giữa các lượt)

```
python scripts/apply_picks.py --queries round<N>/de/queries --out round<N>/run1 \
    --picks-file round<N>/picks.txt
```

- Pick có frame → frame người chốt đứng đầu file, nguyên trạng (đường hybrid pin).
- Pick chỉ video → allocator đang chốt chạy trong video đó.
- Link YouTube của bạn bè → đổi ra video_id bằng transcripts trước
  (`scripts/verify_hypotheses.py` các lần trước có sẵn mẫu), fps lấy THEO VIDEO.

## 4. Sự cố đã từng gặp và thuốc

| Triệu chứng | Thuốc |
|---|---|
| File CSV < 100 dòng | verifier giờ tự chặn; tìm ai cắt nó, dựng lại bằng apply_picks |
| CRLF trong csv | chỉ dùng write_query_csv (newline="") — đừng Path.write_text |
| Đáp án QA "lỗi font" trên Excel | UTF-8 đúng rồi — artifact hiển thị Excel, KHÔNG sửa file |
| gpt trả rỗng | max_completion_tokens ≥ 2000 (ngân sách thinking) |
| Gemini 429 | phân biệt per-minute (đợi) vs per-day (xoay model — VLMJudge tự làm) |
| Câu "đầu tiên xuất hiện X" | full-scan video, KHÔNG quét mẫu thưa (bài học p2-8) |

## 5. Sau trận

- Ghi điểm từng lượt + góp ý vào `round<N>/KET_QUA.md` ngay khi còn nhớ.
- Chạy `python scripts/so_sanh_allocator.py --queries round<N>/de/queries --out round<N>/ss`
  để lưu diff cấu trúc hai allocator trên đề thật — dữ liệu cho lần chốt tham số sau.
