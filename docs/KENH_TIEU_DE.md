# Kênh TIÊU ĐỀ / METADATA làm nguồn ứng viên — đo ngày 30/08/2026

**KẾT LUẬN: HOÀ, không ship.** Cắm ứng viên do BM25 tiêu đề tìm ra vào các dòng
CUỐI của danh sách 100 đo được **+0,0000 trên TEST** (0,4090 ± 0,0014 cả hai
bên, 2σ = 0,0028). Không hại — nhưng cũng không giúp, và chỉ số chẩn đoán mà
tiêu chí nghiệm thu đòi phải TĂNG thì **đứng yên** (28/30 và 29/30 câu có video
đúng trong 100 dòng, trước và sau y hệt).

Đây là hạng mục ⑤ trong `docs/NGHIEN_CUU_SOTA.md` (phần "kênh tiêu đề", hướng
phản biện số 3 mục §3). Cửa này đóng lại **cho đường phân bổ dòng**. Nó chưa
đóng cho đường mở rộng tầm nhìn VLM — xem §5, chỗ duy nhất còn giá trị thật.

Script: `scripts/experiment_kenh_tieu_de.py` (~1 giây CPU, không gọi API).
Cache: `data/cache_kenh_tieu_de/` — `chan_doan.json`, `tom_tat.json`,
`diem_tune.json`, `diem_test.json`, `dong/` (37 bộ 100 dòng × 60 câu).

```
python scripts/experiment_kenh_tieu_de.py                 # (a) + (b) + (c)
python scripts/experiment_kenh_tieu_de.py --chi-chan-doan # chỉ (a)
```

---

## 1. Cách làm — và vì sao nó KHÁC phép đo metadata cũ đã ÂM

Bảng tín hiệu (`docs/KIEN_TRUC_VA_HUONG_CAI_THIEN.md`) đã có một dòng
`metadata video (tiêu đề, mô tả) | video | −7,6% | ❌`. Kết quả này **không mâu
thuẫn** với dòng đó, vì hai phép đo làm hai việc khác nhau:

| | `experiment_metadata.py` (cũ, −7,6%) | script này |
|---|---|---|
| cách dùng tín hiệu | **cộng điểm**: `s_siglip + w·tanh(BM25)` rồi xếp lại | **hợp danh sách**: cắm vào dòng cuối, không đụng điểm |
| hại được không | có — xáo sai thứ tự video là cướp thang phân bổ sâu | không — `R@k` là *max* trên tiền tố |
| nguồn mẫu | — | U-CESE, arXiv:2605.23274v1: *"merges them into a single list"* |

Chỉ mục BM25 dựng bằng chính `TranscriptIndex` của `src/core/transcripts.py`
(unigram + bigram, NFC, chuẩn hoá độ dài tài liệu, tiêu đề lặp ×3) — không viết
lại bộ tách từ, để không tạo ra một nhánh thứ hai lệch với kênh lời thoại.
873/873 video, mô tả cắt 1.200 ký tự (đuôi mô tả là khối "Đăng ký KÊNH…" giống
nhau ở mọi video; idf dập được nội dung nhưng **không** dập được ảnh hưởng của
nó lên chuẩn hoá độ dài BM25).

Ứng viên cắm vào bằng `src.core.submission.reserve_tail_rows`, chấm qua đúng
`allocate_rows` của `make_submission` (allocator `coverage` bản ship), luật
harness quen thuộc: TUNE = 30 câu chỉ số chẵn, TEST = 30 câu lẻ, họ hạt giống
tách rời (TUNE gốc 50000 × 3 họ × 32 bốc; TEST gốc 90000 × 4 họ × 48 bốc),
cửa sổ {6, 10, 20}, luật hoà 2σ. Bộ chấm vector hoá được assert khớp tuyệt đối
`final_score`/`r_score_kis` lúc chạy.

---

## 2. (a) CHẨN ĐOÁN — con số quyết định là **1/60**, không phải 2/60

Trên 60 câu GT:

* video đúng **ngoài top-24 ứng viên SigLIP**: **8/60** câu (5, 12, 13, 17, 22, 23, 40, 48)
* video đúng **ngoài cả pool 400 ứng viên**: **0/60** câu

| biến thể metadata | hạng-1 | top-3 | top-10 | trung vị hạng | CỨU top-3 | **CỨU thật** | CỨU pool |
|---|---|---|---|---|---|---|---|
| tiêu đề | 5 | 6 | 12 | 20 | 2 | **1** | 0 |
| tiêu đề + mô tả | 6 | 9 | 15 | 73 | 2 | **1** | 0 |
| tiêu đề + mô tả + từ khoá | 6 | 8 | 14 | 76 | 1 | **0** | 0 |

*CỨU top-3* = video đúng ở top-3 tiêu đề mà không có trong top-24 SigLIP.
*CỨU thật* = như trên nhưng dùng **hạng bi quan**. *CỨU pool* = cứu thật và
video còn không có ở bất kỳ đâu trong pool 400.

### Cái bẫy khiến 2 thành 1 — phải nói rõ vì suýt nữa báo cáo sai

Đếm thô cho **2** câu được cứu: câu 17 và câu 40. Câu 40 là **giả**.

```
câu 17  L30_V010  hạng lạc quan 1, bi quan 1   -> CỨU THẬT
        truy vấn: "bàn tay học sinh nắn nót cầm bút chì viết chữ..."
        tiêu đề : "Lan tỏa năng lượng tích cực 2024 'Gieo chữ' cho học sinh nghèo"

câu 40  L21_V001  hạng lạc quan 1, bi quan 60  -> HOÀ 60 VIDEO
        truy vấn: "Đồ họa 3D chuyển cảnh logo 60 Giây... kênh HTV9 HD"
        tiêu đề : "60 Giây Sáng - Ngày 01082024 - HTV Tin Tức Mới Nhất 2024"
```

Kho có 60 bản tin "60 Giây Sáng – Ngày …" mà tiêu đề chỉ khác nhau đúng con số
ngày tháng. Truy vấn nói "logo 60 Giây" khớp **cả 60 video ở đúng cùng một
điểm BM25**. Xếp theo `(-điểm, video_id)` thì L21_V001 đứng đầu — không phải vì
tiêu đề nó đúng hơn 59 cái kia mà vì **bảng chữ cái**. Ground truth tình cờ là
L21_V001, nên đếm thô ghi "hạng 1, cứu một câu", trong khi cơ hội thật là 1/60.

Vì vậy script đo **hai** hạng cho mọi video và kết luận bằng hạng **bi quan**
(mọi video hoà điểm tính là đứng trước). Hạng bi quan là hạng duy nhất một kênh
ứng viên thật sự bảo đảm được.

---

## 3. (b) PHÉP ĐO — TEST +0,0000

Quét 36 tổ hợp trên TUNE (3 biến thể × {1,2,3,5} video × {1,2,3} frame/video):

| tổ hợp | dòng thêm | điểm TUNE | so nền | video/100 | **mất video** |
|---|---|---|---|---|---|
| nền (coverage bản ship) | — | 0,3894 ± 0,0017 | — | 28/30 | — |
| tiêu đề, 1 video × 1 frame | 1,0 | 0,3894 ± 0,0017 | +0,0% | 28/30 | 0 |
| tiêu đề, 3 video × 1 frame | 3,0 | 0,3894 ± 0,0017 | +0,0% | 27/30 | 1 |
| tiêu đề, 3 video × 3 frame | 9,0 | 0,3891 ± 0,0018 | −0,1% | 26/30 | 2 |
| tiêu đề, 5 video × 3 frame | 15,0 | 0,3874 ± 0,0016 | **−0,5%** | 26/30 | 2 |

Ba biến thể metadata cho **bảng giống hệt nhau tới 4 chữ số** — tín hiệu tiêu đề
không đủ mạnh để khác biệt giữa các trường metadata chạm được tới điểm.

Chốt trên TUNE: `tiêu đề, 1 video × 1 frame`. Đọc TEST **đúng một lần**:

| TEST (30 câu lẻ) | điểm | so nền |
|---|---|---|
| nền (coverage bản ship) | 0,4090 ± 0,0014 | — |
| + ứng viên tiêu đề | 0,4090 ± 0,0014 | **+0,00%** |

Chênh **+0,0000**, 2σ = 0,0028 → **HOÀ**.

Phép đo có thật, không phải cắm hụt: kiểm lại file dòng cho thấy **cả 30/30 câu
TEST đều bị đổi**, tổng cộng 30 dòng tiêu đề được chèn vào hạng 100 — và trúng
đúng **0** lần. Bằng chứng vật lý cho luật đọc-TEST-một-lần:
`data/cache_kenh_tieu_de/diem_test.json` chứa **đúng 2 mục** (`nen` và
`tieu_de_v1_f1`), trong khi `diem_tune.json` chứa 37.

### Phát hiện phụ: `reserve_tail_rows` KHÔNG miễn phí

Cột "mất video" là số câu mà video đúng **biến mất khỏi 100 dòng** vì chính việc
dành chỗ đuôi. Hàm cắt `rows[:100−len(extras)]`, nên khi video đúng chỉ có dòng
ở cuối bảng thì nó bị đẩy ra ngoài. Docstring của hàm nói trade-off này "gần như
miễn phí và bị chặn chặt" — đúng về mặt điểm số (dòng 96..100 chỉ đáng 0,2),
nhưng **chỉ số phủ video thì tụt thật**, 1–2/30 câu ngay từ 2 dòng cắm vào.
Ai dùng hàm này về sau cần biết: ngân sách đuôi là ngân sách **có giá**.

---

## 4. (c) Chỉ số về tính, và vì sao bộ 60 câu này không thể thưởng cho kênh này

| | nền | + tiêu đề |
|---|---|---|
| TUNE: số câu có VIDEO ĐÚNG trong 100 dòng | 28/30 | 28/30 |
| TEST: số câu có VIDEO ĐÚNG trong 100 dòng | 29/30 | 29/30 |

Đứng yên. Tiêu chí nghiệm thu là *KHÔNG ÂM* **cộng** *chỉ số chẩn đoán (a) tăng*
— vế thứ hai trượt, nên không nghiệm thu.

**Cơ hội duy nhất còn lại trên bộ này là 3 câu** (22, 23, 48): video đúng nằm
TRONG pool 400 mà bộ phân bổ không cấp cho nó dòng nào. Kênh tiêu đề xếp chúng
ở hạng bi quan 23/5/5, 9/17/36, 10/5/5 — **không câu nào vào nổi top-3**, và
nới lên top-5 thì phần thắng thêm bị phần "mất video" ăn hết (bảng §3).

### Trần lý thuyết của một dòng cắm mù — số học, không phải phép đo

Đây là lập luận **không phụ thuộc bộ 60 câu**, nên nó là phần đáng tin nhất của
tài liệu này. Kênh tiêu đề trả lời "video NÀO"; luật chấm hỏi "FRAME nào". Khi
đã biết video mà không biết khoảnh khắc, dòng cắm vào chỉ là một lần bốc mù trên
trục thời gian.

873 video: độ dài trung vị **7.965 frame**, 163 keyframe/video.

| cửa sổ | 1 dòng rải đều | 3 dòng | 6 dòng |
|---|---|---|---|
| ±6 | 0,15% | 0,46% | 0,93% |
| ±10 | 0,25% | 0,75% | 1,50% |
| ±20 | 0,49% | 1,46% | 2,92% |

Dòng cắm nằm ở hạng 96..100 nên kể cả khi trúng nó chỉ đáng **0,2** điểm câu
(chỉ số hạng k=100 tính tới nó). Kỳ vọng mỗi câu khi cắm 1 dòng mù:
**0,0005 điểm** — bằng 0,13% của nền 0,39, tức **nhỏ hơn 1σ của harness (0,0014)
gần ba lần**. Không harness nào ta dựng nổi trong vòng này nhìn thấy được hiệu
ứng cỡ đó, kể cả bộ 120 câu của hạng mục ②.

Nói cách khác: kết quả HOÀ ở §3 **không phải vì phép đo yếu**. Nó là con số đúng
mà số học dự đoán trước.

---

## 5. Chỗ DUY NHẤT còn giá trị: mở rộng tầm nhìn VLM, không phải cắm dòng

Một câu — **câu 17** — có video đúng ngoài top-24 SigLIP mà tiêu đề xếp hạng-1
thật (không hoà, không nhờ bảng chữ cái). Với câu đó, giá trị của kênh tiêu đề
**không phải** một dòng ở hạng 100; nó là việc VLM được nhìn thấy keyframe của
video đúng để chấm, rồi cả cỗ máy phân bổ 100 dòng làm việc trên video đúng —
tức 1,0 điểm chứ không phải 0,2 × 0,25%.

Đó là đường của mục 4.1 trong `KIEN_TRUC_VA_HUONG_CAI_THIEN.md` ("pool VLM =
top-24 SigLIP ∪ top-5 lời thoại ∪ top-5 OCR ∪ top-3 tiêu đề") và nó **chưa
được đo ở đây** — phép đo đó phải chấm qua `vlm_rerank_run.py`, tốn quota API,
và thuộc lane khác. Số liệu bàn giao cho lane đó:

* thêm kênh tiêu đề vào pool VLM chạm tới **1/60 câu** (1,7%) trên bộ GT hiện có;
* chi phí: 3 video × 6 khung = 18 ảnh/câu thêm vào lượt chấm VLM;
* **cảnh báo bắt buộc**: phải phá hoà điểm BM25 bằng thứ gì đó **không phải
  `video_id`**, nếu không 60 bản tin "60 Giây" sẽ luôn nộp L21_V001 và một
  phép đo naive sẽ ghi nhận nó như thành công.

---

## 6. Giới hạn của kết luận này — đọc trước khi trích

1. **Bộ 60 câu GT về cấu trúc KHÔNG THỂ thưởng cho một kênh mở pool.** 0/60 câu
   có video đúng nằm ngoài pool 400. Phép đo §3 vì thế **chặn được cận trên của
   phần HẠI**, chứ không kiểm được phần LỢI. Nếu đề thi thật có câu mà SigLIP
   trượt hoàn toàn khỏi 400 ứng viên (vòng 1 đã có: p1-19, p1-22 — chỉ kênh lời
   thoại tìm ra), kênh tiêu đề vẫn có thể là thứ duy nhất tìm ra video. Kết luận
   đúng là *"không đáng ship vào đường phân bổ dòng"*, **không phải** *"metadata
   vô dụng"*.
2. **Trần lý thuyết §4 mới là lập luận tổng quát**, và nó nói: kể cả khi kênh
   tiêu đề tìm đúng video, **cắm dòng mù không quy ra điểm được**. Muốn ăn điểm
   thì video tìm được phải đi vào **đường truy xuất/VLM** để được cấp thang phân
   bổ đầy đủ, chứ không phải xin vài dòng ở đuôi.
3. Phép đo chạy trên allocator `coverage` (mặc định hiện tại). Chưa đo lại trên
   `hybrid`; không kỳ vọng khác vì cơ chế cắt đuôi giống nhau.
4. Truy vấn đưa vào BM25 là `kis_query_vi`. Cờ `--truy-van vi_en` có sẵn nhưng
   chưa chạy — metadata là tiếng Việt nên thêm bản tiếng Anh chỉ pha loãng.
5. **Chưa đo trên đề thật.** 60 câu GT là proxy duy nhất, đúng như mọi hạng mục
   khác trong dự án.

---

## 7. Dòng cho bảng tín hiệu

| tín hiệu | tầng | kết quả | có dùng? |
|---|---|---|---|
| kênh tiêu đề/metadata làm **nguồn ứng viên**, cắm đuôi bằng `reserve_tail_rows` | dòng | **TEST +0,0000 (HOÀ)**; chỉ số phủ video đứng yên 28/30, 29/30; trần lý thuyết của 1 dòng mù = 0,0005 điểm/câu < 1σ | ❌ |
| ↳ nới lên 5 video × 3 frame | dòng | −0,5% TUNE — dòng cắm đuôi **đẩy văng** video đúng ở 2/30 câu | ❌ |
| ↳ kênh tiêu đề mở rộng **pool VLM** (top-3 tiêu đề) | video | chạm 1/60 câu (câu 17, hạng-1 thật); **chưa đo qua VLM** | ⏳ lane VLM |
