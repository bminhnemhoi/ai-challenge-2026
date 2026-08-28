# TỐI NAY 19:30 — trình tự tác chiến (in ra / mở cạnh terminal)

Đo nháp lúc 15h hôm nay, cùng máy: make_submission 12 câu = **112s**, trang review = **46s**,
verify_zip = **1s**. Quota Gemini: **6/7 model còn sống** (probe 15h). Mọi lệnh dưới đây đã chạy thật hôm nay.

## 18:45 — trước giờ thi (5 phút, BẮT BUỘC)

```powershell
# 1) DỪNG mirror đang tải nền — nó ăn băng thông và quota CDN trong giờ thi
Get-Process python | Where-Object {$_.CPU -gt 0} | Format-Table Id,CPU   # tìm PID mirror
Stop-Process -Id <PID_mirror>          # chạy lại sau giờ thi: python scripts/mirror_keyframes.py --workers 24

# 2) kiểm mạng + quota (10 giây)
python -c "import urllib.request;urllib.request.urlopen('https://huggingface.co',timeout=10);print('CDN OK')"
```

Mở sẵn: **3 terminal** tại gốc repo + trình duyệt + trang nộp của BTC + file này.
Phân vai như vòng 1: 1 người terminal, 2-3 người soát trang review, 1 người ghi `round2/picks_verified.txt`.

## Phút 0–8: nền + NỘP LƯỢT 1 ngay

```bash
mkdir -p round2/queries        # giải nén đề của BTC vào đây
ls round2/queries | wc -l      # đếm đủ câu chưa

python scripts/make_submission.py --queries round2/queries --out round2/base --allow-blank-answers
python scripts/verify_zip.py round2/base/submission.zip --queries round2/queries
# format check passed -> NỘP LUÔN lượt 1. Điểm nền về sớm = mốc chẩn đoán.
cp -r round2/base round2/final
```

## Phút 8–35: ba việc SONG SONG (3 terminal)

```bash
# T1 — VLM xếp lại + trả lời Q&A (xoay 6 model tự động, hết quota sẽ TỰ BÁO chứ không im lặng)
python scripts/vlm_rerank_run.py --queries round2/queries --out round2/final --allow-blank-answers
python scripts/answer_qa.py --queries round2/queries --out round2/final --overwrite --repackage

# T2 — OCR ứng viên (chạy nền ~25', KHÔNG chặn việc khác)
python scripts/run_ocr.py --queries round2/queries --top 24

# T3 — dò lời thoại cho MỌI câu có tên riêng/món ăn/địa danh (kênh đã cứu nguyên câu ở vòng 1)
python scripts/search_transcripts.py --query "<vài từ khoá đặc trưng của câu>"
```

Người rảnh: viết `.en.txt` cho câu khó (dịch tay ăn +10,5% — đừng bỏ).

## Phút 35–40: dựng trang soát

```bash
python scripts/build_review_page.py --queries round2/queries --run-out round2/final
# Trang TỰ ĐỘNG rơi về ảnh mirror trên đĩa nếu CDN chết — không cần làm gì.
```

## Phút 40–105: soát mắt — nguồn điểm lớn nhất (55')

Thứ tự soát: **câu gắn cờ ⚠ trước** (bộ lọc bắt 5/5 câu phải sửa ở vòng 1). Sổ tay quyết định
nhanh: `docs/VI_DU_LUAN_CHUNG.md` (bảng cuối). Ghi mọi quyết định vào `round2/picks_verified.txt`
theo dạng `query-...=VIDEO:F1|F2|F3[:đáp án]` — **chuỗi nhiều frame khi hành động lặp lại**.

Hai kênh cãi nhau → phân xử bằng mắt máy (đừng đoán):

```bash
python scripts/verify_hypotheses.py --pairs "query-X=VID_A,VID_B" --max-frames 36
# Q&A đọc số/chữ -> ảnh GỐC, đừng tin bản 512px:
python scripts/read_answer.py --video <VID> --frames <F> --neighbours 2 --max-side 1900 --question "..."
```

## Phút 105–110: áp picks + NỘP LƯỢT 2 (lấy điểm chẩn đoán)

```bash
python scripts/apply_picks.py --queries round2/queries --out round2/final --picks-file round2/picks_verified.txt
python scripts/verify_zip.py round2/final/submission.zip --queries round2/queries   # BẮT BUỘC trước mỗi lần nộp
```

**NHẬT KÝ** — điền ngay khi nộp, đây là cách đọc lỗi từ điểm số:

| lượt | phút | điểm | những câu ĐÃ ĐỔI so với lượt trước |
|---|---|---|---|
| 1 | ~8 | ___ | (nền) |
| 2 | ~110 | ___ | ___ |
| 3 | ~172 | ___ | ___ |

Đọc delta: điểm lượt 2 − lượt 1 ≈ tổng đóng góp của các câu đã đổi. Tăng ít hơn kỳ vọng
→ nhóm câu vừa đổi có câu SAI — soát lại đúng nhóm đó ở phút 110–165, đừng soát lan man.

## Phút 110–165: đào sâu nhóm nghi vấn → Phút 165–172: NỘP LƯỢT CUỐI

Chỉ lượt cuối tính điểm. `verify_zip` xong mới bấm nộp. **Đệm 8 phút** — vòng 1 suýt trễ.

## Khi hỏng — 4 kịch bản đã có thuốc

| sự cố | thuốc |
|---|---|
| Mất mạng / CDN chết | trang review tự rơi về mirror đĩa; VLM có cache; cứ soát tiếp |
| VLM báo HẾT QUOTA | nó tự xoay 6 model; nếu cạn cả 6: bỏ VLM, soát tay + lời thoại vẫn đủ sống |
| Engine/terminal chết giữa chừng | CSV đã ghi còn nguyên: `python scripts/repackage.py --out round2/final` (20 giây, không cần index) |
| Sửa CSV tay xong | đừng zip tay — `repackage.py` rồi `verify_zip.py` |

## Tuyệt đối không

- Nộp mà chưa `verify_zip` (sai định dạng = **mất trắng câu**, TRAKE = số sự kiện + 1 cột).
- Đáp án Q&A bỏ trống (= 0 điểm dù frame đúng) hoặc chứa **dấu phẩy** (viết `2.15` không viết `2,15`).
- Sửa file bằng PowerShell `Set-Content` (hỏng UTF-8).
- Tin điểm "cấp video" — BTC chấm video **và** frame trong cửa sổ.
