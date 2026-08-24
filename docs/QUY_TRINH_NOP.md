# QUY TRÌNH NỘP — cẩm nang bấm giờ trong ngày thi

> Tài liệu này **không giải thích tại sao**. Nó chỉ trả lời ba câu cho mỗi bước:
> **gõ lệnh gì**, **mất bao lâu**, **làm sao biết bước đó đã xong đúng**.
>
> - Muốn biết *tại sao* làm vậy → `docs/CONTEST_RUNBOOK.md` (luật chấm, chiến thuật 3 lượt, cách dùng `review.html`).
> - Muốn *thử một vòng đề từ đầu tới cuối* lúc rảnh → `docs/HUONG_DAN_KIEM_THU.md`.
> - Muốn hiểu kiến trúc → `docs/KIEN_TRUC_VA_HUONG_CAI_THIEN.md`.
>
> In trang này ra giấy. Trong giờ thi không ai có thời gian cuộn màn hình.

Quy ước trong tài liệu: thư mục đề là `round2/queries`, thư mục làm việc là
`round2/a`, `round2/b`, `round2/c`. Đổi tên theo vòng thật của bạn, nhưng
**giữ nguyên cấu trúc** — mọi lệnh dưới đây đều nhận `--queries` (thư mục đề)
và `--out` (thư mục làm việc), và `--out` luôn chứa `csv/` + `submission.zip`.

---

## 0. Bảng tổng thời gian (vòng 3 tiếng)

| Phút | Việc | Ai làm | Máy chạy |
|---|---|---|---|
| 0–3 | Nhận đề, giải nén | 1 người | — |
| 3–6 | Sinh bản gốc `--no-answer` | 1 người | ~90 giây |
| 6–10 | Kiểm zip → **NỘP LẦN 1** | 1 người | ~1 giây |
| 6–35 | OCR ứng viên (chạy nền, mở terminal riêng) | tự chạy | ~25 phút |
| 10–25 | Viết `.en.txt` cho các câu khó | cả đội chia nhau | — |
| 25–40 | Chạy VLM rerank | 1 người | ~10 phút |
| 40–45 | Điền đáp án Q&A | 1 người | ~1 phút |
| 45–50 | Dựng `review.html` | 1 người | ~2 phút (ước lượng) |
| 50–105 | **Soát bằng mắt, chốt frame** ⭐ | cả đội chia câu | — |
| 105–115 | Áp picks, kiểm zip → **NỘP LẦN 2** | 1 người | ~1–3 phút |
| 115–165 | Đào sâu các câu còn nghi | cả đội | — |
| 165–175 | Kiểm zip lần cuối → **NỘP LẦN CUỐI** | 1 người | ~1 giây |
| 175–180 | Đệm. Không làm gì mới. | — | — |

Số phút máy chạy lấy từ bảng chi phí đã đo trong
`docs/KIEN_TRUC_VA_HUONG_CAI_THIEN.md:227-231` (dựng bài nộp 90 giây, VLM 24 câu
10 phút, OCR 25 phút chạy nền, trả lời Q&A 1 phút). Riêng thời gian dựng
`review.html` **tôi chưa thấy con số đo nào trong repo** — nó nạp lại index
SigLIP-2 giống `make_submission` nên tôi ước lượng cùng cỡ; hãy tự bấm giờ
trong buổi diễn tập rồi sửa lại con số này.

---

## 1. TRƯỚC NGÀY THI — làm xong hết, không để rơi vào giờ thi

Không có bước nào ở mục này được phép làm lần đầu trong lúc thi.

```bash
python -m pytest tests src/task3_trake/tests -q
```
**Xong khi:** không có `F`, không có `E`. Có một cái đỏ nào là dừng, sửa xong mới thi.

```bash
python scripts/verify_zip.py round_p1/best/submission.zip --queries round_p1/queries
```
**Xong khi:** in ra `format check passed. This zip is safe to upload.`
(chuỗi này ở `scripts/verify_zip.py:97`). Lệnh này không nạp index nên chạy
dưới 1 giây — đây là cách rẻ nhất để chắc bộ kiểm tra còn hoạt động.

Checklist còn lại, mỗi mục một dòng:

| Việc | Cách biết đã xong |
|---|---|
| `.env` ở gốc repo có `GEMINI_API_KEY=...` | `python scripts/vlm_rerank_run.py --queries round_p1/queries --out /tmp/x` **không** in `Khong co GEMINI_API_KEY` (`scripts/vlm_rerank_run.py:110-112`) |
| Model SigLIP-2 đã nằm trong cache HuggingFace | Chạy `make_submission` một lần, dòng `ready in ...s` dưới 3 phút, không thấy thanh tải model |
| Đã đăng nhập sẵn https://sotuyenaic.oj.io.vn/ | Mở tab, thấy trang nộp, **để nguyên tab đó mở suốt buổi thi** |
| Đã nộp thử 1 zip ở vòng thử nghiệm | Hệ thống BTC nhận, không báo lỗi định dạng |
| Biết mình còn bao nhiêu quota Gemini | Xem mục **hết quota** ở phần 12 |

> **Cảnh báo quota (`src/core/vlm.py:44-56`):** bậc miễn phí tính **500 request
> mỗi ngày cho mỗi tên model**. Danh sách dự phòng đã lỗi thời một lần rồi —
> `gemini-2.0-flash` và `gemini-2.0-flash-lite` trả 404 và đã bị loại ngày
> 21/08/2026. **Sáng ngày thi, chạy thử một lệnh VLM nhỏ** để biết tên model nào
> còn sống. Mỗi tên chết đi là mất 500 request khỏi ngân sách cả ngày.

---

## 2. Nhận đề và giải nén — 3 phút

```bash
mkdir -p round2/queries
# giải nén file zip đề của BTC vào đó
ls round2/queries
```

**Xong khi:** thấy đúng số file `.txt` mà BTC nói, **tên file y nguyên**
(`query-p2-1-kis.txt`, `query-p2-15-qa.txt`, …).

**Không được đổi tên file.** Bộ chấm ghép CSV với câu hỏi theo tên
(`csv_name_for_query` ở `src/core/submission.py:407-409` chỉ đổi đuôi `.txt` →
`.csv`). Đổi tên là câu đó không được chấm.

**Không mở file `.txt` bằng Notepad rồi Save.** Xem mục *sai định dạng* ở phần 12.

---

## 3. Sinh bản gốc — ~90 giây

```bash
python scripts/make_submission.py --queries round2/queries --out round2/a --no-answer
```

Dùng `--no-answer` cho lượt này **cố ý**: lượt 1 chỉ để thử định dạng, không cần
tốn quota Gemini.

**Đọc gì trên màn hình:**

```
24 query files in round2/queries
loading the SigLIP-2 index (this is the slow part; it happens once) ...
  ready in 42.1s

  query-p2-1-kis.txt         kis    100 rows    2.3s
  query-p2-4-trake.txt       trake  100 rows    3.1s
  ...
wrote round2/a/submission.zip  (118 KB)
  kis=20  qa=3  trake=1  of 24 queries

format check passed: submission/ folder, every row parsed, <=100 rows, no BOM, no .mp4
```

**Xong đúng khi cả bốn điều sau đều đúng:**

1. Dòng cuối là `format check passed: ...` (`scripts/make_submission.py:589`).
2. `kis + qa + trake` **cộng lại bằng** tổng số câu (dòng `of N queries`).
3. Mọi câu đều `100 rows`. Ít hơn 100 là đang vứt điểm miễn phí.
4. **Không có dòng nào** ghi `placeholder row written — ANSWER THIS ONE BY HAND`
   (`scripts/make_submission.py:562`). Có nghĩa là câu đó đã crash và chỉ được
   ghi một dòng giữ chỗ — nó không chặn cả gói, nhưng nó chắc chắn 0 điểm.

**Nếu thấy `FORMAT PROBLEMS — do not upload this:`** (`:585`) → **đừng nộp**.
Đọc từng dòng gạch đầu dòng, sửa, chạy lại.

**Kiểm tra riêng cho câu TRAKE:** số cột của file `*-trake.csv` phải bằng
**số sự kiện + 1** (cột 1 là `video_id`, mỗi cột sau là frame của một sự kiện).
Sai số cột thì cả câu 0 điểm và **mọi kiểm tra khác vẫn pass**:

```bash
head -1 round2/a/csv/query-p2-4-trake.csv | tr ',' '\n' | wc -l
```

Nếu `make_submission` in cảnh báo về số sự kiện tách được, xử lý ngay — đừng bỏ qua.

> **Lưu ý về thư mục `csv/`:** `make_submission.py:508-510` **tự xoá** mọi
> `*.csv` cũ trong `<out>/csv/` trước khi ghi. Nhưng `repackage.py`, `apply_picks.py`
> và `pin_video.py` thì **không** — chúng đóng gói mọi thứ đang có trong thư mục
> đó. Nên đừng bao giờ trỏ `--out` của vòng mới vào thư mục của vòng cũ.

---

## 4. Kiểm zip rồi NỘP LẦN 1 — 4 phút

Dù `make_submission` đã tự kiểm, vẫn chạy lại cửa chốt độc lập:

```bash
python scripts/verify_zip.py round2/a/submission.zip --queries round2/queries
```

**Xong khi thấy:**

```
round2/a/submission.zip  (118 KB)
  24 CSV, 2400 rows total, 100 per query
  expected 24 queries from round2/queries

format check passed. This zip is safe to upload.
```

Ba con số phải khớp: số CSV = số câu, `rows total` = số câu × 100, `100 per query`.

Nếu thấy dòng `note: N of the M available rank slots are unused`
(`scripts/verify_zip.py:92-96`) thì có câu chưa đủ 100 dòng — không phải lỗi
định dạng, nhưng là điểm miễn phí đang bị vứt.

Bây giờ **upload `round2/a/submission.zip`**.

**Xong khi:** trang BTC báo nhận thành công. Ghi lại giờ nộp lên giấy.
Còn **2 lượt**.

> Mục đích lượt này **không phải điểm**. Là để biết chắc hệ thống chấm chấp nhận
> định dạng của mình, trong khi vẫn còn 2 tiếng rưỡi để sửa nếu nó không chấp nhận.

---

## 5. Bật OCR chạy nền — mở terminal thứ hai, ~25 phút

Chạy **ngay sau khi nộp lượt 1**, trong một cửa sổ terminal riêng, rồi quên nó đi.

```bash
python scripts/run_ocr.py --queries round2/queries --top 24
```

Nó đọc chữ cháy trên đúng những khung hình mà bài nộp đang đề cử, và đo màu chủ
thể. Kết quả nằm ở `data/ocr/<video_id>.json` và `data/colours/<video_id>.json`.

**Xong khi:** tiến trình thoát, và `python scripts/search_ocr.py --list-videos`
đếm ra số video/khung hình lớn hơn trước.

**Nếu nó chưa xong lúc bạn muốn dựng `review.html`:** cứ dựng. Trang vẫn chạy,
chỉ là ít khung hình có dòng 🔤 hơn. Dựng lại sau khi OCR xong thì đầy đủ.

> Cache OCR có trước khi phần đo màu ra đời, nên nếu `data/colours/` rỗng mà
> `data/ocr/` đã đầy, chạy `python scripts/run_ocr.py --queries round2/queries --colours-only`
> — không nạp model, ~10 khung/giây.

---

## 6. Viết `.en.txt` — 15 phút, cả đội chia nhau

Đây là việc rẻ nhất mà con người làm được cho máy. Với mỗi câu khó, tạo một file
**cùng tên, thêm đuôi `.en.txt`**, cạnh file đề:

```
round2/queries/query-p2-7-kis.txt      ← đề gốc BTC, không sửa
round2/queries/query-p2-7-kis.en.txt   ← bản tiếng Anh bạn viết
```

Viết như caption ảnh, **chi tiết phân biệt để lên đầu**:
`a dark red sedan with a rear spoiler driving on a city street behind a truck`.

**Riêng câu TRAKE:** phải giữ đúng dấu phân tách sự kiện (`E1:` / `(1)` / `;`).
Bản dịch viết dạng văn xuôi sẽ co cả chuỗi về **1 sự kiện** → sai số cột → 0 điểm.
`scripts/make_submission.py:425-446` có đối chiếu chéo với bản gốc và giữ bên tách
được nhiều sự kiện hơn, nhưng đừng dựa hoàn toàn vào đó.

**Xong khi:** chạy lại `make_submission` và thấy hậu tố `[manual EN]` ở đúng
những dòng bạn vừa viết file (`scripts/make_submission.py:568`).

---

## 7. Chạy VLM rerank — ~10 phút

Bước này để mô hình thị giác đọc lại danh sách ngắn và sửa những chỗ mà embedding
không phân biệt được (màu sắc, số lượng, hành động, chữ trên hình).

```bash
python scripts/vlm_rerank_run.py --queries round2/queries --out round2/b
```

Nó tự đóng gói `round2/b/submission.zip` và tự kiểm định dạng ở cuối
(`scripts/vlm_rerank_run.py:341-357`).

**Xong đúng khi cả ba điều sau đều đúng:**

1. Dòng cuối là `format check passed. Zip nay nop duoc.`
2. Dòng `N/24 cau doi video hang 1 nho VLM` — N thường nhỏ, vài câu. N = 0 nghĩa
   là VLM không đổi gì; N gần bằng tổng số câu là dấu hiệu **bất thường**, hãy đọc kỹ.
3. **Dòng `cost_note` KHÔNG chứa `!!`** — xem ngay mục dưới.

### ⚠️ Đọc `cost_note` trước khi tin kết quả

`src/core/vlm.py:365-382` in ra một dòng chi phí và **hét lên bằng `!!`** khi có
vấn đề. Bốn chuỗi phải thuộc lòng:

| Thấy chuỗi này | Nghĩa là |
|---|---|
| `!! HET QUOTA: <tên model>` | Model đó đã hết hạn mức **ngày**, bị gạch tên khỏi lượt chạy |
| `!! N lô lỗi, gần nhất: ...` | N lô ảnh gọi API thất bại |
| `!! N khung hình KHÔNG TẢI ĐƯỢC` | Mạng/CDN hỏng. Chúng **bị bỏ qua, không phải bị chấm 0** |
| `!! KHONG CHAM DUOC KHUNG HINH NAO — dung coi ket qua nay la da xet.` | **VLM chưa hề nhìn cái gì.** Bài nộp sinh ra từ lượt này là bản CHƯA được xét |

Chuỗi cuối cùng là sự cố có thật của vòng 1: quota cạn → mọi lời gọi 429 →
phán quyết rỗng → đường ống vẫn vui vẻ đóng gói một bản nộp trông hoàn chỉnh.
Đã có test ghim lại (`tests/test_vlm_quota.py:68-75`). **Lượt chạy tốt thì tuyệt
đối không có dấu `!!` nào** — đó là lý do dấu đó giữ được sức nặng.

### Đừng chỉnh trọng số

`--weight` mặc định `0.02`. Đó là **trần**, không phải điểm khởi đầu
(`scripts/vlm_rerank_run.py:73-75`). Đo trên 60 câu ground truth: w=0.02 → +3,3%,
w=0.10 → **−2,1%**, w=0.20 → **−5,7%**. Ở w cao, video R@1 *tăng* trong khi điểm
thi *giảm* — đừng dùng R@1 làm thước đo ở đây.

### Thứ tự bắt buộc

`vlm_rerank_run.py` **trước**, `apply_picks.py` **sau**
(`scripts/vlm_rerank_run.py:29-31`). Ngược lại là VLM ghi đè lên lựa chọn tay của
người soát.

---

## 8. Điền đáp án Q&A — ~1 phút

```bash
python scripts/answer_qa.py --queries round2/queries --out round2/b
```

Mặc định **không ghi đè** đáp án đã có; thêm `--overwrite` nếu muốn thay.
Thêm `--repackage` để nó đóng gói + kiểm lại luôn (nhưng chỉ chạy khi có ít nhất
một câu được điền — `scripts/answer_qa.py:213`).

**Xong khi:** mỗi câu Q&A in ra một dòng dạng
`query-p2-15-qa  'Xã ...'  tin cay 100%  (tu L01_V006 frame 1745)`.

**Chưa xong nếu thấy:** `model khong doc duoc dap an tu N khung hinh`
(`scripts/answer_qa.py:194-197`). Đó là mô hình **trung thực** báo nó không đọc
được — không phải lỗi. Bạn phải tự đọc bằng mắt (mục 10) hoặc dùng `read_answer.py`
ở độ phân giải gốc:

```bash
python scripts/read_answer.py --video L01_V003 --frames 561 --neighbours 2 \
  --question "Bien bao o dau cau ghi so bao nhieu?"
```

`read_answer.py` gửi ảnh tối đa 1536px (thay vì 512px của bước chấm) nên đọc được
chữ nhỏ trên biển báo, mặt cân, bản đồ. Nó **không có cache, không xoay vòng
model, không xử lý 429** — mỗi lần chạy là một lần gọi thật, dùng tiết kiệm.

> **Đáp án Q&A để trống = 0 điểm chắc chắn**, dù frame đúng đến đâu (luật 2.1.2).
> Bộ kiểm tra sẽ **từ chối** cả gói nếu có đáp án rỗng
> (`src/core/submission.py:596-604`). Cờ `--allow-blank-answers` chỉ dành cho lần
> nộp thử định dạng có chủ ý — **đừng dùng nó để "cho qua" rồi upload thật**.

---

## 9. Dựng trang review — ~2 phút

```bash
python scripts/build_review_page.py --queries round2/queries --run-out round2/b
```

**File ra ở đâu:** mặc định là `review.html` **cạnh thư mục queries**, tức
`round2/review.html` (`scripts/build_review_page.py:1019`). Muốn chỗ khác thì
truyền `--out`.

**Xong khi thấy hai dòng cuối:**

```
wrote round2/review.html   (24 queries, 6 flagged uncertain, 812 KB)
Mở file đó, kéo thả khung hình đúng lên #1, điền đáp án Q&A,
```

Con số `N flagged uncertain` là số câu hệ thống tự nhận **không chắc** — soát
những câu đó trước.

> `--run-out` phải trỏ đúng thư mục bạn vừa chạy VLM (`round2/b`), nếu không
> trang sẽ minh hoạ cho một bài nộp khác với bài bạn định sửa. Mặc định của nó là
> `<thư mục cha của queries>/run1` (`scripts/build_review_page.py:1020`) — hầu như
> chắc chắn không phải cái bạn muốn, nên **luôn truyền `--run-out` tường minh**.

Trang này cũng có sẵn 🔤 chữ OCR và 🎨 màu chủ thể dưới mỗi ảnh (nếu bước 5 đã
xong), và bảng 🎙 lời thoại cho từng câu.

---

## 10. Chốt frame bằng mắt — 55 phút, đây là bước ăn điểm nhất

Mở `round2/review.html` bằng trình duyệt. Chia câu cho từng người
(4 người × 6 câu ≈ 3 phút/người/câu).

Cách dùng chi tiết đã có ở `docs/CONTEST_RUNBOOK.md:80-115` — **đọc mục đó, không
lặp lại ở đây**. Chỉ nhắc ba điều dễ quên:

1. Bấm **`Chỉ hiện câu cần soi video`** (`scripts/build_review_page.py:290`) để lọc
   còn đúng những câu đáng bỏ công: toàn bộ TRAKE, toàn bộ Q&A, và những câu hai
   video đầu bảng gần hoà.
2. Với câu không chắc, **mở video gốc** (`▶ xem`) và chốt đúng giây. Cửa sổ đáp án
   chỉ rộng ~10 frame còn keyframe cách nhau ~55 frame, nên một frame do bạn xem
   video mà chỉ ra thường **chính xác hơn** mọi ứng viên hệ thống đưa ra.
3. Câu TRAKE: mọi sự kiện phải **cùng một video**. Trang sẽ từ chối nếu bạn chốt
   lệch video, và nó đúng — sai video là 0 điểm tuyệt đối.

### Ba công cụ tra cứu khi nghi hệ thống bỏ sót cả video

Chạy trong terminal, song song với việc soát:

```bash
# tìm theo LỜI NÓI trong video (mili-giây mỗi truy vấn, không cần model)
python scripts/search_transcripts.py "măng tây chiên bột"
python scripts/search_transcripts.py "củ năng" --videos L26

# tìm theo CHỮ TRÊN HÌNH (chỉ tìm được trong khung hình run_ocr đã đọc)
python scripts/search_ocr.py "Nguyễn Trung Trực"
python scripts/search_ocr.py --list-videos

# hỏi thẳng mô hình thị giác: đáp án có nằm trong ĐÚNG video này không, ở frame nào?
python scripts/verify_hypotheses.py --pairs "query-p2-19-qa=L01_V001,L01_V002"
python scripts/verify_hypotheses.py --pairs "query-p2-4-kis=L01_V001" --range 3000-4200
```

`verify_hypotheses.py` **không ghi gì ra đĩa** — nó in ra để bạn đọc rồi tự quyết.
Hai dòng phải để ý:

- `KHONG CO KET QUA cho ca N khung hinh — chua xet duoc, dung ket luan gi tu day`
  → **đừng kết luận gì**, mạng hoặc quota hỏng.
- `THIEU N` trong bảng → N khung hình đó **bị loại khỏi bảng**, không phải bị chấm 0.

> Ở bước chốt **frame**, đừng đưa nguyên văn câu truy vấn cho mô hình — nó sẽ chấm
> theo bối cảnh và cho gần như mọi khung hình của đúng video ấy điểm cao (một video
> từng có 72/193 keyframe ≥ 0.60). Hỏi một **chi tiết thoáng qua** dưới dạng có/không:
> `--question "Trong khung hinh nay co bien so xe mau vang khong?"`

---

## 11. Áp dụng lựa chọn rồi NỘP LẦN 2 — 10 phút

Có **hai đường**, chọn một. Đường A nhanh hơn và giữ được toàn bộ thứ tự bạn đã kéo.

### Đường A — xuất zip ngay trong trình duyệt (khuyên dùng)

Trên `review.html`, bấm **`⬇ Tải submission.zip`**
(`scripts/build_review_page.py:289`). File rơi vào thư mục Downloads.

Rồi **bắt buộc** kiểm lại — đây là một đường đi tới BTC **không hề đi qua**
`make_submission.py`:

```bash
python scripts/verify_zip.py ~/Downloads/submission.zip --queries round2/queries
```

**Xong khi:** `format check passed. This zip is safe to upload.`

### Đường B — chạy qua Python

Trên `review.html` bấm `Lệnh sửa (cách cũ)` để lấy chuỗi picks, rồi:

```bash
python scripts/apply_picks.py --queries round2/queries --out round2/b \
  --picks "query-p2-1-kis=L01_V005:<frame>;query-p2-15-qa=L01_V006:<frame>:<đáp án>
```

Cú pháp: `query=VIDEO:FRAME[:đáp án]`, ngăn nhau bằng `;`.
`apply_picks.py` **tự đóng gói và tự kiểm** ở cuối (`scripts/apply_picks.py:294-302`).

**Lưu ý:** đường B **chỉ** áp lựa chọn #1 của mỗi câu, không giữ toàn bộ thứ tự
bạn đã kéo trên trang. Mặc định nó dành **50 dòng đầu** cho video bạn xác nhận và
giữ 50 dòng sau theo xếp hạng gốc làm bảo hiểm (`--pin-budget`, mặc định 50 ở
`scripts/apply_picks.py:131`). Nếu bạn nhìn nhầm thì vẫn còn cơ hội ăn bậc 0.2.

Sửa từng câu một thì dùng `pin_video.py`:

```bash
python scripts/pin_video.py --queries round2/queries --out round2/b \
  --query query-p2-24-kis --video L01_V004 --frame 12625
```

Nó **không** tự đóng gói — sau khi sửa hết phải chạy:

```bash
python scripts/repackage.py --out round2/b --queries round2/queries
```

**Xong khi:** `format check passed. This zip is safe to upload.`
(`scripts/repackage.py:68`).

### Nộp

Upload zip vừa kiểm. **Xong khi** trang BTC báo nhận. Ghi giờ lên giấy. Còn **1 lượt**.

---

## 12. Nộp lần cuối — phút 165–175

```bash
python scripts/verify_zip.py <file zip cuối cùng> --queries round2/queries
```

**Chỉ nộp khi thấy `format check passed`.** Không đổi bất cứ thứ gì về định dạng ở
lượt cuối — chỉ nộp thứ đã chạy qua cửa kiểm.

**Xong khi:** trang BTC báo nhận, và bạn còn ≥ 5 phút trên đồng hồ.

> Chỉ **lượt cuối** được tính điểm. Nếu lượt 2 tốt hơn những gì bạn đang có lúc
> phút 165, thì **nộp lại chính file của lượt 2**. Đừng nộp một bản mới chỉ vì nó mới.

---

## 13. KHI HỎNG THÌ LÀM GÌ

### 13.1 Hết quota Gemini

**Triệu chứng:** `cost_note` in `!! HET QUOTA: gemini-3.5-flash-lite, ...`, hoặc tệ
hơn — `!! KHONG CHAM DUOC KHUNG HINH NAO`.

**Xử lý theo thứ tự:**

1. **Đổi model.** Quota tính riêng từng tên model, mỗi tên 500 request/ngày:
   ```bash
   python scripts/vlm_rerank_run.py --queries round2/queries --out round2/b --model gemini-2.5-flash-lite
   ```
   Danh sách còn dùng được, theo `src/core/vlm.py:44-56`:
   `gemini-3.5-flash-lite` (mặc định) · `gemini-2.5-flash-lite` ·
   `gemini-flash-lite-latest` · `gemini-3.1-flash-lite` · `gemini-2.5-flash` ·
   `gemini-flash-latest`.

2. **Đừng chạy lại cùng model.** `self.exhausted` là trạng thái **trong bộ nhớ** —
   khởi động lại tiến trình là nó quên, và bạn chỉ đâm vào tường một lần nữa, chậm
   hơn (vì phải chờ `RETRY_WAIT = 8s, 22s, 40s`).

3. **Phân biệt hai loại 429.** Code tự làm việc này qua `quotaId`
   (`src/core/vlm.py:70-81`): 429-theo-**phút** thì nó ngủ rồi thử lại; 429-theo-**ngày**
   thì nó gạch tên model và đi tiếp. Nếu chỉ thấy chậm mà không thấy `!! HET QUOTA`
   thì đó là giới hạn phút — **cứ để nó chạy**, đừng Ctrl+C.

4. **Hết sạch mọi model.** Bỏ hẳn bước VLM. Chạy đường không cần API:
   ```bash
   python scripts/make_submission.py --queries round2/queries --out round2/c --no-answer
   python scripts/build_review_page.py --queries round2/queries --run-out round2/c
   ```
   Rồi soát bằng mắt và **gõ đáp án Q&A tay** vào ô trên `review.html`. Vòng 1 chỉ
   có 3 câu Q&A — làm tay hoàn toàn kịp.

5. **Đừng đổi `--model` chỉ để thử.** Khoá cache băm **cả tên model**
   (`src/core/vlm.py:137-139`), nên đổi model là mất sạch cache của mọi câu hỏi và
   phải chấm lại từ đầu — tốn thêm quota. Chỉ đổi khi model hiện tại thật sự chết.

### 13.2 Mất mạng

Chia làm bốn mức, xử lý khác nhau:

| Thứ cần mạng | Hỏng thì sao | Làm gì |
|---|---|---|
| **Index SigLIP-2** (`data/*.npy`) | Không cần mạng — nằm trên đĩa | Vẫn chạy `make_submission` được |
| **Dịch tự động VI→EN** | In `! automatic translation unavailable; falling back to the Vietnamese text` (`src/core/kis_engine.py:196-201`) | Không crash. Nhưng video R@1 tụt từ ~43% xuống ~35%. **Bù bằng cách viết `.en.txt` tay** |
| **Ảnh thumbnail** (CDN HuggingFace) | `review.html` hiện ô trống; VLM in `!! N khung hình KHÔNG TẢI ĐƯỢC` | Ảnh đã tải nằm ở `data/frames/` và dùng lại được. Nếu mạng chết hẳn → bỏ bước VLM, soát bằng nhãn vật thể + OCR đã cache |
| **Gemini API** | Xem 13.1 mục 4 | |
| **Trang nộp BTC** | Không upload được | **Ưu tiên số một.** Dùng 4G điện thoại phát wifi. Chuẩn bị sẵn phương án này TRƯỚC ngày thi |

**Quy tắc chung khi mạng chập chờn:** đừng chạy lại `run_ocr.py` — một thumbnail
chết bị nuốt lặng lẽ và ghi cache rỗng cho khung đó, tức là cache sẽ đầy những
khung "đã đọc, không có chữ" **giả**.

### 13.3 Sai định dạng

**Nguyên tắc:** một lần bị BTC từ chối vì định dạng **vẫn tính là một lượt nộp**.
Nên không bao giờ upload thứ chưa qua `verify_zip.py`.

Đọc thẳng dòng lỗi mà `verify_submission_zip` in ra — nó nói rõ file nào, dòng nào:

| Dòng lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `... has Windows CRLF line endings` | Ai đó mở CSV bằng **Notepad hoặc Excel rồi Save** | `python scripts/repackage.py --out round2/b --queries round2/queries`. **Đừng bao giờ Save CSV từ Notepad/Excel nữa** (`src/core/submission.py:537-546`) |
| `... starts with a UTF-8 BOM` | Cùng nguyên nhân trên | Như trên |
| `... row N field 2 is not an integer frame id (...) — a header row is not allowed` | Có dòng tiêu đề, hoặc frame id không phải số | Sửa CSV rồi `repackage.py` |
| `... rows have a blank answer` | Câu Q&A chưa có đáp án | Điền đáp án (mục 8), rồi `repackage.py`. **Đừng** dùng `--allow-blank-answers` |
| `missing` / `unexpected` tên file | CSV thừa từ vòng trước, hoặc thiếu một câu | Dùng thư mục `--out` **mới tinh**, chạy lại từ bước 3 |
| `... row N has fewer than 2 fields` | Dòng hỏng | Sửa rồi `repackage.py` |

**Hai điều dễ quên:**

- `verify_submission_zip` **dừng ở dòng hỏng đầu tiên của mỗi file**
  (`src/core/submission.py:589-596`). Sửa xong phải **chạy lại** — lần đầu chưa
  chắc đã thấy hết vấn đề.
- Archive **bắt buộc** có thư mục tên đúng `submission/` bên trong. Tự zip các file
  CSV rời sẽ bị từ chối (`src/core/submission.py:450-467`). Luôn dùng
  `repackage.py`, đừng tự nén bằng chuột phải.

**Nếu terminal Windows crash `UnicodeEncodeError`:** phần lớn script đã gọi `safe_console()`
(27/40 — `evaluate_official.py`, `experiment_retrieval.py`, `experiment_strategies.py`,
`experiment_objects.py`, `experiment_metadata.py`, `experiment_long_query.py`,
`download_data.py` và vài script build thì **chưa**; với chúng hãy đặt
`set PYTHONIOENCODING=utf-8` trước khi chạy)
(`scripts/_console.py`) nên không nên xảy ra. Nếu vẫn xảy ra ở script bạn tự viết,
thêm hai dòng import đó vào đầu file — crash này có thể rơi **giữa** lúc ghi CSV và
đóng gói zip, để lại một bài nộp nửa vời.

### 13.4 Hết giờ

Đây là kịch bản có thật nhất. Quy tắc: **luôn có sẵn một zip hợp lệ trên đĩa**.

| Còn bao lâu | Làm gì |
|---|---|
| **30 phút** | Dừng mọi việc mới. Chạy `verify_zip.py` trên bản tốt nhất hiện có. Nếu pass → sẵn sàng nộp bất cứ lúc nào |
| **15 phút** | **Nộp ngay bản tốt nhất hiện có.** Đừng đợi bước nào chạy xong. Một bản 90% đã nộp hơn một bản 100% chưa nộp |
| **5 phút** | Không chạy lệnh nào cần nạp index (mất ~40 giây chỉ để load). Chỉ `verify_zip.py` (< 1 giây) rồi upload |
| **Đang chạy VLM mà hết giờ** | Ctrl+C. CSV các câu đã xong **đã được ghi**. Chạy `repackage.py --out round2/b --queries round2/queries` để đóng gói phần đã có, kiểm, nộp |
| **Người soát chưa xong** | Ai xong câu nào thì áp câu đó. `apply_picks.py` nhận nhiều câu trong một chuỗi, các câu không được nhắc tới giữ nguyên |

**Lệnh cấp cứu 20 giây** — khi cần một zip hợp lệ ngay lập tức từ những CSV đang có:

```bash
python scripts/repackage.py --out round2/b --queries round2/queries \
  && python scripts/verify_zip.py round2/b/submission.zip --queries round2/queries
```

Không nạp index, không gọi mạng, không gọi API.

### 13.5 Các sự cố khác

| Triệu chứng | Xử lý |
|---|---|
| `ModuleNotFoundError` bất kỳ | `pip install -r requirements.txt`. **Không sửa code trong giờ thi** |
| `FileNotFoundError` về `metadata.json` / `embeddings_siglip2_384.npy` | `python scripts/download_data.py`. Nếu không kịp, mượn thư mục `data/` từ máy đồng đội bằng USB |
| Model SigLIP tải lại từ đầu (~3.5 GB) | Cache HuggingFace hỏng. Copy `~/.cache/huggingface` từ máy đồng đội. **Đừng ngồi đợi tải** |
| Một câu crash, in `placeholder row written` | Câu đó có một dòng giữ chỗ nên không chặn cả gói. Tên nó hiện trong danh sách `fix these by hand`. Xử lý tay bằng `pin_video.py` |
| Máy hết RAM | Đóng trình duyệt và mọi app khác. `make_submission.py` chỉ cần ~2 GB |
| `objects-aic25-b1.zip` làm chậm máy | Thêm `--no-objects`. Mất +3,3% đã đo, nhưng nhanh hơn |
| Trang `review.html` không phản hồi khi bấm | Mở Console của trình duyệt (F12) xem lỗi JS. Chuyển sang đường B (`apply_picks.py`) |

---

## 14. Ba lỗi mất điểm oan — kiểm lại trước mỗi lần nộp

1. **Câu nào đó không đủ 100 dòng.** R@k là **max** trên k dòng đầu nên một dòng
   sai không bao giờ làm giảm điểm; dòng 51–100 là 0.2 điểm miễn phí.
   → `verify_zip.py` in `note: N of the M available rank slots are unused` nếu thiếu.

2. **Đáp án Q&A trống, hoặc chỉ điền vài dòng đầu.** Bỏ trống từ dòng 6 là vứt
   R@20 / R@50 / R@100 — ba trong năm thành phần của Final Score.
   → Bộ kiểm tra sẽ chặn, đừng dùng `--allow-blank-answers` để lách.

3. **File TRAKE sai số cột.** Số cột = số sự kiện + 1. Sai là **cả câu 0 điểm**, và
   sai kiểu này **vô hình** vì mọi kiểm tra khác đều pass trên một dòng 2 cột.
   → `head -1 <file>-trake.csv | tr ',' '\n' | wc -l`

---

## 15. Bảng tra lệnh nhanh (cắt ra dán lên màn hình)

```bash
# sinh bản gốc, không gọi VQA
python scripts/make_submission.py --queries round2/queries --out round2/a --no-answer

# sinh bản đầy đủ (cần GEMINI_API_KEY)
python scripts/make_submission.py --queries round2/queries --out round2/b

# OCR + màu cho ứng viên của vòng (chạy nền, ~25 phút)
python scripts/run_ocr.py --queries round2/queries --top 24

# VLM chấm lại + ghi lại CSV + đóng gói (~10 phút)
python scripts/vlm_rerank_run.py --queries round2/queries --out round2/b

# điền đáp án Q&A
python scripts/answer_qa.py --queries round2/queries --out round2/b

# dựng trang soát bằng mắt -> round2/review.html
python scripts/build_review_page.py --queries round2/queries --run-out round2/b

# áp lựa chọn của người soát (tự đóng gói + tự kiểm)
python scripts/apply_picks.py --queries round2/queries --out round2/b --picks "..."

# chốt một câu
python scripts/pin_video.py --queries round2/queries --out round2/b \
  --query query-p2-24-kis --video L01_V004 --frame 12625

# đóng gói lại sau khi sửa tay (BẮT BUỘC sau pin_video)
python scripts/repackage.py --out round2/b --queries round2/queries

# CỬA CHỐT CUỐI — chạy trước MỌI lần upload, kể cả zip do trình duyệt tạo
python scripts/verify_zip.py <đường dẫn zip> --queries round2/queries
```

**Chỉ một câu cần nhớ nếu quên hết:**
đừng bấm nút upload khi màn hình chưa in `format check passed`.
