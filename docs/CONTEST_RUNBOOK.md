# RUNBOOK — 3 tiếng thi, làm đúng thứ tự này

> Mỗi đợt chỉ mở **3 tiếng** và chỉ được nộp **3 lần**, **lần cuối mới tính điểm**.
> In trang này ra. Đừng đọc lần đầu vào lúc 19:30.

---

## Luật chấm — 4 điều phải thuộc

| | |
|---|---|
| **KIS đúng khi nào** | Đúng video **VÀ** `frame_id` nằm trong đoạn đáp án `[s,e]`. Đúng video mà sai frame = **0 điểm**. |
| **Điểm mỗi câu** | `Final = ⅕ × (R@1 + R@5 + R@20 + R@50 + R@100)`, mà **R@k là điểm CAO NHẤT trong k dòng đầu**. |
| **Bậc thang điểm** | Trúng ở hạng 1 → **1.0** · hạng 2–5 → **0.8** · 6–20 → **0.6** · 21–50 → **0.4** · 51–100 → **0.2** · trượt → 0 |
| **Dòng thừa** | **Không bao giờ làm giảm điểm.** Luôn nộp đủ 100 dòng. |

Hệ quả trực tiếp: chuyển một câu trúng từ hạng 15 lên hạng 10 **không được gì**; từ hạng 6 lên hạng 5 được **+0.2**. Chỉ tối ưu quanh 5 mốc đó.

---

## Chuẩn bị TRƯỚC ngày thi (làm ngay hôm nay, ~30 phút)

```bash
git clone <repo> && cd ai-challenge-2026
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
python scripts/download_data.py                     # ~830 MB
python -m pytest src/task3_trake/tests tests -q     # phải xanh hết
```

Rồi **diễn tập trọn vẹn một lần**:

```bash
python scripts/make_submission.py --queries rehearsal/queries --out rehearsal/check --no-answer
```

Phải thấy dòng cuối: `format check passed`. Nếu chưa từng chạy được lệnh này thì **chưa sẵn sàng thi**.

Ba việc cuối:
1. Đặt `GEMINI_API_KEY` vào file `.env` rồi chạy lại **không** có `--no-answer`, xác nhận cột đáp án có chữ.
2. Chạy 1 query bất kỳ để **nạp sẵn model SigLIP-2 vào cache** (lần đầu tải ~3.5 GB — đừng để rơi vào giờ thi).
3. Mở `https://sotuyenaic.oj.io.vn/`, đăng nhập sẵn, thử nộp 1 file zip diễn tập trong **vòng thử nghiệm**.

---

## Trong giờ thi

### Phút 0–5 — nhận đề
Tải bộ query, giải nén vào một thư mục, ví dụ `round1/queries/`. Giữ **nguyên tên file** BTC đặt (`query-1-kis.txt`, …) — tên file CSV nộp bài phải khớp đúng.

### Phút 5–10 — nộp lần 1 (kiểm tra format)
```bash
python scripts/make_submission.py --queries round1/queries --out round1/a --no-answer
```
Nộp `round1/a/submission.zip` ngay. Mục đích **không phải** điểm cao mà là xác nhận hệ thống chấm chấp nhận định dạng. Còn 2 lần nộp.

### Phút 10–100 — làm chất lượng

**Việc đáng giá nhất: dịch tay sang tiếng Anh.** Dịch tự động hay bị chặn giữa chừng, mà bản tiếng Anh tốt đáng giá ~8 điểm phần trăm video R@1. Với mỗi query, tạo file cạnh nó:

```
round1/queries/query-1-kis.txt      ← đề gốc của BTC
round1/queries/query-1-kis.en.txt   ← bản tiếng Anh bạn tự viết
```

Viết như caption ảnh: *"a dark red sedan with a rear spoiler driving on a city street behind a truck"*. Chia nhau mỗi người vài câu, 15 phút là xong.

Chạy lại và đối chiếu bằng mắt:
```bash
python scripts/make_submission.py --queries round1/queries --out round1/b
```
Dòng nào có `[manual EN]` là đã dùng bản dịch tay.

**Kiểm tra bằng mắt — đây là việc đáng giá nhất trong cả 3 tiếng.** Mắt người nhìn 24 ảnh nhanh và chính xác hơn mọi mô hình, và luật chấm trả rất hậu: kéo đáp án đúng từ hạng 6–20 (0.6 điểm) lên hạng 1 (1.0 điểm) là **+0.4 cho mỗi câu**, chỉ tốn vài giây nhìn.

```bash
python scripts/build_review_page.py --queries round1/queries --run-out round1/b
```

Mở `round1/review.html`. Cách dùng:

| Thao tác | Ý nghĩa |
|---|---|
| **Kéo thả khung hình** | Đổi thứ hạng. Vị trí **#1 = dòng đầu tiên** của bài nộp câu đó |
| Bấm vào khung hình | Đưa thẳng nó lên #1 (nhanh hơn kéo khi bạn đã chắc) |
| Phím `1`–`9` | Đưa khung hình thứ n lên #1 |
| Phím `j` / `k` | Nhảy sang câu sau / câu trước |
| Nút 🔍 hoặc `Shift`+bấm | **Phóng to** — đọc chữ cháy trên hình (tên xã, tên món, nhãn nguyên liệu) |
| Câu Q&A | Ô nhập đáp án ngay dưới đề. **Bỏ trống là 0 điểm** dù frame đúng |
| Câu TRAKE | Mỗi khối là **một video trọn chuỗi** E1→EN. Kéo cả khối, không kéo lẻ từng frame |
| **`▶ xem` / `▶ Xem video`** | Mở **video gốc trên YouTube tại đúng giây đó** — xem mục dưới |
| **`⬇ Tải submission.zip`** | Tạo file nộp **ngay trong trình duyệt**, đúng thứ tự bạn vừa sắp |

Mọi thay đổi lưu trong trình duyệt, đóng mở lại vẫn còn.

### Xem video gốc — dùng cho TRAKE, Q&A và câu không chắc

Ảnh thu nhỏ 158 pixel không đủ để quyết một chuỗi TRAKE, không đọc nổi đáp án Q&A, và không phân xử được hai video gần bằng điểm nhau. Cả **873 video đều có link YouTube** trong `media-info` của BTC, nên trang tự mở đúng khoảnh khắc.

Bấm nút **`Chỉ hiện câu cần soi video`** trên thanh công cụ để lọc còn đúng những câu đáng bỏ công: toàn bộ TRAKE, toàn bộ Q&A, và những câu mà hai video đầu bảng gần như hoà. Vòng thử vừa rồi là **11/24 câu**.

Trong cửa sổ xem:

| Thao tác | Ý nghĩa |
|---|---|
| Dải keyframe phía dưới | Bấm để nhảy tới, phím `←` `→` để đi từng keyframe |
| Ô **Giây trên video** | Tạm dừng YouTube đúng khoảnh khắc, đọc số giây, gõ vào, bấm `Nhảy tới` |
| **`Đặt frame này lên #1`** | Chốt **đúng frame đó** làm dòng 1, kể cả khi nó không có trong danh sách gợi ý |
| Tab `E1` `E2` … (TRAKE) | Chốt riêng từng sự kiện — xem đến đâu đánh dấu đến đó |
| Đề bài hiện ngay trên player | Để vừa xem vừa gõ đáp án Q&A, không phải cuộn lên |

**Vì sao việc chốt tay đáng giá:** cửa sổ đáp án chỉ rộng khoảng 10 frame, còn keyframe cách nhau ~55 frame. Một frame do bạn xem video mà chỉ ra thường **chính xác hơn** mọi ứng viên hệ thống đưa ra — và hệ thống sẽ tự rải thang ±10, ±20 quanh nó.

Nếu bạn chốt một sự kiện TRAKE thuộc video khác với chuỗi đang chọn, trang **từ chối** và báo lý do: các sự kiện phải cùng một video, sai video là 0 điểm.

**Cách nhanh nhất (không cần terminal):** sắp xong → bấm `⬇ Tải submission.zip` → kiểm tra rồi nộp:

```bash
python scripts/verify_zip.py ~/Downloads/submission.zip --queries round1/queries
```

Lệnh này chạy trong một giây (không nạp index) và dùng **đúng** bộ kiểm tra mà `make_submission` dùng. Chỉ nộp khi nó in `format check passed`.

> File zip do trình duyệt tạo được kiểm chứng bằng test: `tests/test_page_export_matches_pipeline.py` chạy chính đoạn JS trong trang qua `node` và bắt buộc từng dòng CSV phải **trùng khít** với bản `make_submission` sinh ra.

**Cách cũ (nếu bạn thích chạy Python):** nút `Lệnh sửa` sinh một lệnh `apply_picks.py`. Lưu ý nó **chỉ** áp dụng lựa chọn #1 của mỗi câu, không giữ toàn bộ thứ tự bạn đã kéo:

```bash
python scripts/apply_picks.py --queries round1/queries --out round1/b \
  --picks "query-1-kis=L21_V025:12480;query-15-qa=L30_V072:5376:Xã Vạn Thắng"
```

Xem câu nào nên soi trước:
```bash
python scripts/inspect_run.py --queries round1/queries --out round1/b
```
Nó xếp hạng độ chắc chắn và **cảnh báo riêng câu Q&A còn trống đáp án**.

### Phút 100–170 — nộp lần 2, rồi tinh chỉnh
Nộp `round1/b/submission.zip`. Sau đó chỉ sửa những câu bạn **biết chắc** là sai.

### Phút 170–180 — nộp lần cuối
Nộp bản tốt nhất. **Không đổi bất cứ thứ gì về định dạng ở lần nộp cuối** — chỉ nộp thứ đã chạy qua `format check passed`.

---

## Sự cố — xử lý theo thứ tự

| Triệu chứng | Xử lý |
|---|---|
| `ModuleNotFoundError` bất kỳ | `pip install -r requirements.txt`. Không sửa code trong giờ thi. |
| Dịch báo *translation unavailable* | Không sao, vẫn chạy bằng tiếng Việt. Viết file `.en.txt` tay để bù. |
| Không có `GEMINI_API_KEY` | Chạy `--no-answer` để lấy frame trước, rồi **gõ đáp án vào ô trong `review.html`** (phóng to bằng 🔍 để đọc chữ trên hình). Task 2 rất ít câu nên làm tay hoàn toàn kịp. |
| Báo `X/100 rows have a blank answer` | Đúng như thiết kế — công cụ **từ chối** cho nộp bài mà Task 2 chắc chắn 0 điểm. Điền đáp án trong `review.html` rồi chạy `apply_picks.py`. |
| `format check` báo lỗi | **Đừng nộp.** Đọc dòng lỗi, sửa, chạy lại. Nộp sai định dạng vẫn mất 1 lượt. |
| Một query lỗi | Câu đó được ghi **một dòng giữ chỗ** để không chặn cả gói; tên nó hiện trong danh sách "fix these by hand". Các câu khác không ảnh hưởng. |
| Đề TRAKE đánh số `(1) … (2) …` | Đã hỗ trợ. Kiểm tra nhanh: file `*-trake.csv` phải có **số cột = số sự kiện + 1**. |
| Máy hết RAM | Đóng web app. `make_submission.py` chỉ cần ~2 GB. |
| Model SigLIP tải lại từ đầu | Đã hỏng cache. Copy `~/.cache/huggingface` từ máy đồng đội. |

---

## Ba lỗi khiến mất điểm oan — đừng lặp lại

1. **Nộp ít hơn 100 dòng.** Dòng 51–100 là 0.2 điểm miễn phí. Công cụ đã tự điền đủ.
2. **Mỗi video chỉ nộp 1–2 frame.** Vì R@k lấy max, nộp nhiều frame của cùng một video **không bao giờ hại**. Đây chính là lỗi của phiên bản cũ.
3. **Để trống hoặc ghi "Không xác định" ở cột đáp án Q&A từ dòng 6 trở đi.** Làm vậy là vứt đi 3 trong 5 thành phần điểm. Công cụ đã dán cùng một đáp án lên mọi dòng.
