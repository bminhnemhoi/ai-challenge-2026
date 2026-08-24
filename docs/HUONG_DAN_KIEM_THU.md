# Hướng dẫn kiểm thử một vòng đề — làm đúng thứ tự

Bộ đề 24 câu vòng 1 đã được đặt sẵn ở `round_p1/queries/`. Quy trình dưới đây dùng lại được cho mọi vòng.

---

## Bước −1 — Tải dữ liệu phụ trợ của BTC (làm một lần)

```bash
python scripts/download_data.py
```

Tải cả `objects-aic25-b1.zip` (610 MB) — nhãn vật thể cho từng keyframe. Trang duyệt ở Bước 3 dùng nó để hiện số lượng vật thể dưới mỗi ảnh.

**Đã đo và KHÔNG dùng để chấm điểm tự động:**

| Dữ liệu | Kết quả đo trên điểm chính thức |
|---|---|
| `media-info` (tiêu đề/mô tả video) | 0.405 → 0.408, **trong nhiễu**; R@1 còn giảm 43.3% → 40.0%. Là tín hiệu **cấp video**, không đẩy được frame đúng lên. |
| `objects` (đếm vật thể mỗi frame) | **Không kết luận được**: ground truth chỉ có 2/60 câu đếm, mà cả hai đều đòi *"≥1 người"* — thoả mãn bởi **57.4% toàn bộ keyframe**, tức cộng điểm cho nó là cộng hằng số. |

Nên objects chỉ dùng để **hỗ trợ mắt người**, không đưa vào công thức xếp hạng. Đưa một cơ chế chưa đo được vào đường chấm điểm là đúng cái sai mà cả loạt thử nghiệm trước đã cho thấy: làm mượt thời gian −0.023, chuẩn hoá theo video −0.129, metadata 0, viết lại câu hỏi 3/4 tệ đi.

---

## Bước 0 — Lấy đề

Tải file zip của BTC, giải nén vào một thư mục. **Giữ nguyên tên file** — tên CSV nộp bài phải trùng tên file đề, sai là bị loại.

```
round_p1/queries/query-p1-1-kis.txt
round_p1/queries/query-p1-15-qa.txt
round_p1/queries/query-p1-4-trake.txt
...
```

---

## Bước 1 — Sinh bài nộp (≈1 phút cho 24 câu)

```bash
python scripts/make_submission.py --queries round_p1/queries --out round_p1/run1
```

Bỏ `--no-answer` để Gemini trả lời 3 câu Q&A. Không có API key thì thêm `--no-answer`, rồi gõ tay đáp án ở bước 4.

Đầu ra: `round_p1/run1/submission.zip`, mỗi câu đủ 100 dòng, đã tự kiểm tra định dạng.

> Nếu báo `X/100 rows have a blank answer` — đúng như thiết kế. Công cụ **từ chối** cho nộp bài mà Task 2 chắc chắn 0 điểm.

---

## Bước 2 — Xem hệ thống chắc chắn tới đâu (≈1 phút)

```bash
python scripts/inspect_run.py --queries round_p1/queries
```

Cột `margin` là khoảng cách giữa video top-1 và video tốt nhất kế tiếp, tính theo độ lệch chuẩn. Cột `conc` là tỉ lệ top-24 thuộc cùng một video. Câu nào bị đánh `??` là hệ thống **không chắc** — xem những câu đó trước.

Trên bộ đề vòng 1: **6/24 câu không chắc chắn**.

---

## Bước 3 — Xác minh bằng mắt ⭐ (bước ăn điểm nhất)

```bash
python scripts/build_review_page.py --queries round_p1/queries --run-out round_p1/run1
```

Mở `round_p1/review.html` bằng trình duyệt. Mỗi câu hiện 24 khung hình đầu, tải thẳng từ CDN.

Mắt người nhận ra cảnh trong 2 giây. Luật trả **1.0 cho hạng 1** nhưng chỉ **0.6 cho hạng 6–20** — nên mỗi câu bạn xác nhận được là **+0.4**, rẻ hơn mọi cải tiến mô hình.

Dưới mỗi ảnh có **nhãn vật thể** do BTC cung cấp (Faster R-CNN), ví dụ `Woman×2, Bookcase×2, Man`. Rất hữu ích cho **12/24 câu vòng này có yếu tố đếm** — "3 tay đua", "bốn em nhỏ", "ba người".

Bấm vào khung hình đúng → bấm nút → trang in ra lệnh cần chạy:

```bash
python scripts/pin_video.py --queries round_p1/queries --out round_p1/run1 \
    --query query-p1-24-kis --video L01_V004 --frame 12625
```

`--frame` là **đúng khung hình bạn nhìn thấy** — nó thành hạng 1, thang frame dựng quanh nó.

Chia dòng: **50 dòng đầu** cho video bạn xác nhận (phủ các bậc 1.0 / 0.8 / 0.6 / 0.4), **50 dòng sau** giữ nguyên xếp hạng gốc làm bảo hiểm. Nếu bạn nhìn nhầm thì vẫn còn cơ hội ăn bậc 0.2; nếu dồn cả 100 dòng cho một video thì nhìn nhầm là mất trắng.

**Chia việc:** 4 người × 6 câu = mỗi người ~3 phút.

---

## Bước 4 — Điền đáp án Q&A

Ba câu Q&A vòng này đều hỏi **chữ hiện trên màn hình**, không phải nội dung hình ảnh:

| Câu | Hỏi gì | Cách trả lời |
|---|---|---|
| `query-p1-15-qa` | Tên xã ở Khánh Hòa | Đọc chữ trên băng-rôn / phụ đề trong khung hình |
| `query-p1-19-qa` | 2 câu thơ trong đình | Đọc hoành phi / câu đối trong khung hình |
| `query-p1-22-qa` | Tên món ăn trên công thức | Đọc tờ công thức trong khung hình |

Mở `review.html`, phóng to khung hình, đọc chữ, rồi:

```bash
python scripts/pin_video.py --queries round_p1/queries --out round_p1/run1 \
    --query query-p1-15-qa --video L30_V072 --answer "Xã Vạn Thắng"
```

Đáp án được dán lên **cả 100 dòng** — bỏ trống từ dòng 6 là vứt 3/5 thành phần điểm.

---

## Bước 5 — Đóng gói lại và nộp

```bash
python scripts/repackage.py --out round_p1/run1 --queries round_p1/queries
```

Phải thấy `format check passed`. Chưa thấy dòng đó thì **đừng nộp** — sai định dạng vẫn mất 1 trong 3 lượt.

---

## Chiến thuật 3 lượt nộp

| Lượt | Khi nào | Nộp gì |
|---|---|---|
| 1 | Phút thứ 10 | Bản thô từ Bước 1. Mục đích là **thử định dạng**, không phải điểm. |
| 2 | Phút thứ 100 | Sau khi xác minh bằng mắt và điền Q&A. |
| 3 | Phút thứ 170 | Bản tốt nhất. **Không đổi gì về định dạng ở lượt cuối.** |

---

## Viết lại câu hỏi bằng tiếng Anh — cách dùng cho đúng

Đặt file cạnh đề, công cụ tự nhận:

```
round_p1/queries/query-p1-7-kis.txt      ← đề gốc
round_p1/queries/query-p1-7-kis.en.txt   ← bạn viết
```

**Quan trọng — đo được, không phải suy đoán:** bản viết lại **không chắc chắn tốt hơn** bản dịch tự động. Trên 4 câu vòng 1 tôi thử: 1 câu tốt lên rõ (câu chim, margin 1.41 → 2.26 và top-5 dồn về một video), 3 câu còn lại độ chắc chắn **giảm**.

Nên công cụ **không thay thế** mà **gộp cả hai** danh sách ứng viên. Vì R@k lấy max, thêm giả thuyết chỉ có thể giúp:

- Câu `query-p1-7-kis` sau khi gộp: rank 1 là `L29_V023` (video bản viết lại tin chắc), **nhưng** `L29_V003` (đáp án tự động) vẫn nằm trong top-20.
- Cả hai khả năng đều được phủ. Chọn một trong hai mới là rủi ro.

Cách viết: như caption ảnh, **chi tiết phân biệt để lên đầu**, và **giải mã tham chiếu gián tiếp** thành thứ nhìn thấy được.

---

## Ba câu cần kiến thức ngoài hình ảnh

SigLIP không suy luận được những thứ này. Viết `.en.txt` mô tả **thứ nhìn thấy được**, rồi xác minh bằng mắt:

| Câu | Ẩn ý | `.en.txt` nên viết |
|---|---|---|
| `query-p1-23-kis` | phim Spielberg 1975 = **Jaws** = cá mập | `a coastal seaside town where tourists watch great white sharks` |
| `query-p1-1-kis` | phi hành gia + cực quang = nhiệm vụ SpaceX | `four astronauts in black flight suits, private spacecraft launch` |
| `query-p1-21-kis` | Đại học ở Lausanne = **EPFL** | `laboratory studying insect flight to build a small flying robot` |

Hai file `.en.txt` cho câu 7 và 23 đã được tạo sẵn làm mẫu — mở ra xem cách viết.

---

## Một điều đã sửa mà bạn nên biết

SigLIP-2 chỉ đọc được **64 token**, phần sau bị cắt âm thầm. Trên bộ đề này **13/24 câu vượt giới hạn**, và thứ bị cắt đúng là chi tiết phân biệt:

- `query-p1-24-kis` mất *"nón đỏ… nón đen"* — chính là thứ phân biệt 3 tay đua
- `query-p1-7-kis` mất *"loài chim thường thấy ở Nam Bộ"*
- `query-p1-11-kis` mất *"tạo thành hình chân dung một người đàn ông"* — trọng tâm cả clip

Hệ thống giờ chia câu dài thành từng đoạn và cộng điểm lại. Đo trên bộ ground truth có chèn thêm chữ: **R@1 từ 3.3% lên 33.3%**, điểm chính thức từ 0.029 lên 0.323. Câu `query-p1-24-kis` từ margin 0.00 (đoán mò) lên 1.00.
