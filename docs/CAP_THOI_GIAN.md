# Truy vấn CẶP THỜI GIAN (lever ③) — kết quả đo

Chốt ngày 30/08/2026. Lane `cap-thoi-gian`, đề xuất ③ trong `docs/NGHIEN_CUU_SOTA.md`.
Nguồn kỹ thuật: vibro (vô địch VBS 2022+2023), vitrivr — công thức
`s_temp(i) = HM(s_A(i), max_{j cùng video, i < j <= i+W} s_B(j))`.

---

## 0. Kết luận một dòng

**CHƯA ĐO ĐƯỢC — và lý do là một phát hiện đắt giá hơn phép đo.**

Cổng của lane bật ở **0 trên 60 câu** ground truth, nhưng bật ở **28 trên 55 câu
đề thật** của BTC (51%). Bộ đo 60 câu **mù hoàn toàn** với đúng thứ mà lane này
nhắm tới. Không có TUNE, không có TEST, không có luật 2 sigma nào áp được —
mọi cấu hình đều là phép đồng nhất trên tập đo.

Hai lát cắt thay thế đều quá nhỏ, và chúng **mâu thuẫn về dấu**:

| lát cắt | n | chất lượng sự thật | công thức nguyên bản (λ=1) |
|---|---|---|---|
| picks vòng 1 (§2a) | 16 | chưa BTC xác nhận | **âm** (3 tốt / 7 tệ) |
| GT đã kiểm chứng (§2b) | 6 | `nguoi_kiem_chung` | **hơi dương** (3 tốt / 1 tệ) |

Nên **không kết luận được lever này dương hay âm** — chỉ kết luận được rằng bộ
đo hiện tại không trả lời được câu hỏi. Thứ duy nhất chắc chắn: **0/60 câu GT
qua cổng**, nên mọi cấu hình là phép đồng nhất trên bộ đo chính thức.

**Không ship gì.** Việc đáng mang đi là một sửa đặc tả cho lane harness (§8.3):
câu GT phải được viết bằng cách xem **một đoạn video**, không phải một keyframe —
nếu không, bộ 120 câu mới sẽ mù đúng như bộ 60 câu hiện tại.

---

## 1. Con số quyết định: cổng bật bao nhiêu câu

| tập câu | có cấu trúc hai cảnh | tỉ lệ |
|---|---|---|
| **60 câu ground truth** (đội tự viết) | **0 / 60** | **0%** |
| đề thật vòng 1 (`round1/queries`) | 16 / 25 | 64% |
| đề thật vòng 2 (`round2/queries`) | 12 / 30 | 40% |
| **đề thật, cả hai vòng** | **28 / 55** | **51%** |

Cách đo: một request Gemini text mỗi câu, `temperature=0`, prompt cố định
(`PROMPT_VERSION = 2`), cache từng câu xuống đĩa.
Script: `scripts/gan_nhan_hai_canh.py`.
Cache: `data/cache_cap_thoi_gian/nhan/` (60 file), `.../nhan_de/` (55 file).

### Vì sao tin được con số 0/60

Ba chốt chặn độc lập, vì một nhãn "0" rất dễ là lỗi prompt quá chặt:

1. **Kiểm từ vựng, không dùng LLM.** Quét 60 câu GT tìm mọi dấu hiệu trình tự
   thời gian (`rồi`, `sau đó`, `tiếp theo`, `trước khi`, `sau khi`,
   `chuyển cảnh`, `lia sang`, `cắt sang`, `bắt đầu bằng`, `kết thúc bằng`,
   `lần lượt`, `cuối cùng`, …): **1/60 câu** trúng, là câu 40 —
   *"Đồ họa 3D **chuyển cảnh** logo 60 Giây trên nền sọc chéo đỏ trắng"*.
   Ở đây "chuyển cảnh" là **danh từ** chỉ loại đồ hoạ (bumper), không phải hai
   cảnh nối tiếp. Gemini bác đúng. Vậy kể cả cận trên từ vựng cũng là 1/60.
2. **Đối chứng nội tại — chốt chặn quan trọng nhất.** Cùng prompt đó, cùng
   nhiệt 0, chạy trên đề thật thì bật **51%**. Prompt không hề "từ chối mọi
   thứ": nó phân biệt được hai phân bố. Nếu 0/60 do prompt chặt thì 28/55 đã
   không xảy ra.
3. **Đọc tay.** 60 câu GT đều là mô tả MỘT khung hình: chủ thể + thuộc tính +
   quan hệ không gian ("xe đỏ có cánh gió **phía sau** xe tải", "mèo mướp
   **đứng trên** sân bê tông **trước** bức tường gạch"). Đề thật thì đầy
   "*Đoạn clip **bắt đầu bằng** … **Đoạn clip kết thúc với** …*",
   "***Sau đó** chuyển sang cảnh …*", "***Tiếp theo là** cảnh …*".

### Đây là lỗ hổng của HARNESS, không phải của lever

Phần ĐO được là hai tỉ lệ ở bảng trên. Phần SUY RA — nói rõ để ai cũng bác được
— là nguyên nhân: 60 câu GT trông đúng như được viết bằng cách **nhìn MỘT
keyframe rồi tả lại nó** (mỗi mục có `frame_filename` + `cdn_url` của đúng một
ảnh, và không câu nào nhắc tới thứ xảy ra trước/sau khung đó), nên theo cấu tạo
chúng không thể có cấu trúc hai cảnh. Đề của BTC thì tả **một ĐOẠN**. Hai quy
trình viết khác nhau sinh hai phân bố khác nhau, và bộ đo đang nằm ở phía không
có tín hiệu. Suy luận này có thể sai về chi tiết quy trình, nhưng hai tỉ lệ
0/60 và 28/55 thì không đổi.

Hệ quả vượt ra ngoài lane này: **mọi đề xuất nhắm vào cấu trúc thời gian
(③, ⑨ viết lại truy vấn + hợp nhất hạng, TRAKE neo hai biên) đều không thể
đo được trên bộ 60 câu hiện tại.** Đây là lý do định lượng, cụ thể, để đẩy
đề xuất ② (mở rộng harness GT) lên trước ③ — và khi mở rộng thì
**phải viết câu bằng cách xem đoạn video, không phải xem một keyframe**,
nếu không bộ 120 câu mới sẽ mù y hệt.

---

## 2a. Lát cắt thứ nhất: proxy picks vòng 1 (n=16) — ÂM

Vì cổng không bật câu GT nào, phép đo duy nhất còn lại là chạy trên đề thật
vòng 1 và so với **lựa chọn của người soát** (`round1/picks_verified.txt`).

> **CẢNH BÁO — bảng này KHÔNG phải ground truth.** Đó là lựa chọn của người
> soát trong trận, có bằng chứng ba kênh nhưng chưa bao giờ được BTC xác nhận.
> Vòng 2 chỉ được 10,0/30 nên một phần đáng kể lựa chọn cùng dạng này là SAI.
> Số dưới đây dùng để **chẩn đoán hướng**, tuyệt đối không phải để chốt tham số.

Số đo: **hạng của video người soát đã chốt** trong 100 dòng đã phân bổ, qua
đúng `allocate_rows(..., "coverage", ...)` của `make_submission` (không chấm tắt).
16 câu vòng 1 có cổng bật. Nền = đường sản xuất hiện tại, không đổi theo W/λ:
**video chốt ở dòng 1 ở 5/16 câu**.

| W | λ | dòng-1 khớp | tốt hơn nền | tệ hơn nền | hoà |
|---|---|---|---|---|---|
| 2 | 1,00 | 3/16 | 2 | **8** | 6 |
| 3 | 1,00 | 4/16 | 2 | **8** | 6 |
| 5 | 1,00 | 4/16 | 3 | **7** | 6 |
| 8 | 1,00 | 4/16 | 3 | **7** | 6 |
| 2 | 0,50 | 5/16 | 3 | 3 | 10 |
| 3 | 0,50 | 5/16 | 2 | 5 | 9 |
| 5 | 0,50 | 5/16 | 2 | 4 | 10 |
| 8 | 0,50 | 5/16 | 2 | 4 | 10 |
| 2 | 0,25 | 5/16 | 4 | 3 | 9 |
| 3 | 0,25 | **6/16** | 3 | 3 | 10 |
| 5 | 0,25 | 5/16 | 4 | 3 | 9 |
| 8 | 0,25 | 5/16 | 4 | 3 | 9 |

Đọc bảng:

- **λ = 1,0 (đúng công thức nguyên bản, thay hẳn điểm gốc) là ÂM rõ ràng** ở cả
  4 giá trị W: 7–8 câu tệ đi so với 2–3 câu tốt lên, và dòng-1 khớp tụt từ 5/16
  xuống 3–4/16.
- **λ ≤ 0,5 (pha loãng) là HOÀ**, mọi ô. Ô cao nhất (W=3, λ=0,25) hơn nền đúng
  **1 câu trên 16** — trên n=16 đó là nhiễu, và nó được chọn sau khi nhìn cả
  bảng nên không phải một phép đo.
- **Không ô nào đáng ship.** Cận trên lạc quan nhất là hoà.

### Đối chứng tách nguyên nhân: phần GHÉP CẶP mới là phần gây hại

Câu hỏi tự nhiên: λ=1 âm vì **ghép cặp thời gian sai**, hay chỉ vì **truy vấn
con ngắn thì nhiễu hơn truy vấn đầy đủ**? Đối chứng `--gop chiA` trả lời: nó bỏ
hẳn cảnh B, chỉ dùng `s_A(i)` thay điểm gốc. (Theo cấu tạo, `chiA` không phụ
thuộc W.)

| biến thể (λ=1) | dòng-1 khớp | tốt hơn | tệ hơn | hoà |
|---|---|---|---|---|
| nền (đường sản xuất) | 5/16 | — | — | — |
| **`chiA` — chỉ cảnh A, KHÔNG ghép** | **7/16** | 3 | 4 | 9 |
| `hm` W=5 — có ghép cặp | 4/16 | 3 | **7** | 6 |
| `hm` W=2 — có ghép cặp | 3/16 | 2 | **8** | 6 |

Và ở mọi λ, `chiA` ≥ `hm`: λ=0,5 → 6/16 vs 5/16; λ=0,25 → 6/16 vs 5/16.

**Kết luận của đối chứng: cái âm KHÔNG phải do truy vấn con ngắn — nó do chính
số hạng ghép cặp.** Bỏ `max_j s_B(j)` đi thì kết quả tốt lên, đều đặn.

Chốt hạ điểm cuối cùng: ca cứu được duy nhất, `query-p1-8-kis` (hạng 67 → 1),
**xảy ra y hệt với `chiA`** — không cần ghép cặp gì cả. Nó không phải bằng
chứng cho cặp thời gian; nó là bằng chứng cho việc **cắt truy vấn dài thành
mệnh đề đầu ngắn hơn**. Tương tự `query-p1-6-kis`: `chiA` đưa 4 → **1**, còn
ghép cặp đẩy ngược 4 → 10.

Nói cách khác, **trên lát cắt này lever ③ thua chính bản ablation của nó** — và
so sánh hai biến thể trên cùng thước đo thì không phụ thuộc picks đúng hay sai.

**Nhưng đừng dừng ở đây: lát cắt thứ hai nói ngược lại.** Xem §2b.

---

## 2b. Lát cắt thứ hai: 6 mục có ground truth ĐÃ KIỂM CHỨNG — hướng NGƯỢC LẠI

Lane harness (`docs/HARNESS_DE_THAT.md`) vừa sinh
`data/ground_truth_de_that.json` với **15 mục đạt `nguoi_kiem_chung`** — bằng
chứng mạnh hơn hẳn picks thô. Giao với 28 câu có cổng bật được đúng **6 mục**.
Đây là tập tin cậy nhất hiện có, và nó **nhỏ tới mức không kết luận được gì**.

Hạng của video đúng, W=5, λ=1 (nền: dòng-1 khớp 1/6):

| câu | video đúng | nền | `hm` (có ghép cặp) | `chiA` (bỏ ghép cặp) |
|---|---|---|---|---|
| p1-12-kis | `L22_V029` | 12 | 27 ↓ | 18 ↓ |
| p1-17-qa | `L22_V027` | ∞ | ∞ | ∞ |
| p1-19-kis | `L24_V035` | ∞ | **99** ↑ | ∞ |
| p2-22-kis | `L26_V470` | ∞ | **48** ↑ | **20** ↑ |
| p2-26-kis | `L25_V062` | 25 | **12** ↑ | **16** ↑ |
| p2-7-qa | `L21_V009` | 1 | 1 | 1 |
| **tổng** | | — | **3 tốt / 1 tệ** | **2 tốt / 1 tệ** |

Ở λ=0,5: `hm` 1 tốt / 2 tệ, `chiA` 2 tốt / 0 tệ.

Trên lát cắt này `hm` **hơi dương** so với nền, và **không** thua `chiA`. Ngược
hẳn dấu của §2a.

## 2c. Hai lát cắt mâu thuẫn — và đó là kết quả

| | n | chất lượng sự thật | `hm` λ=1 so nền |
|---|---|---|---|
| §2a picks vòng 1 | 16 | người soát, chưa BTC xác nhận, vòng 2 cùng dạng chỉ 10,0/30 | **âm** (3 tốt / 7 tệ) |
| §2b GT đã kiểm chứng | 6 | `nguoi_kiem_chung`, bằng chứng nhiều kênh | **hơi dương** (3 tốt / 1 tệ) |

Hai tập **chồng nhau 3 câu** (p1-12, p1-17, p1-19) nên không độc lập; chênh dấu
đến từ hai câu vòng 2 (p2-22, p2-26) chỉ có ở tập tin cậy, nơi ghép cặp kéo
video đúng từ **ngoài 100 dòng** vào hạng 48 và từ 25 lên 12.

**Kết luận trung thực: không đủ dữ liệu để nói lever này dương hay âm.** Tập
đáng tin có n=6 (một phép tung đồng xu cho 3–1 rất thường xuyên); tập lớn hơn
thì sự thật của nó chưa được xác nhận và có thể đang phạt s_temp vì **bất đồng
với người soát** chứ không vì sai. Ai trích tài liệu này theo một trong hai
chiều mà bỏ chiều kia là đang đọc sai nó.

### Ca dương rõ nhất của §2a, và vì sao nó không chứng minh được ghép cặp

`query-p1-8-kis` — *"Người đầu bếp lần lượt đặt các miếng nguyên liệu dạng
thanh … **Sau đó**, đầu bếp …"*: video người soát chốt (`L26_V171`, xác nhận
bằng lời thoại 3:31 + món hấp) đứng **hạng 67** ở đường nền và lên **hạng 1**
với s_temp — ở λ=1 với **cả bốn** giá trị W, và vẫn lên rõ (67 → 5–11) ở λ=0,5.
Thoạt nhìn đây đúng là ca mà lý thuyết vibro mô tả. **Nhưng đối chứng ở cuối §2a
cho thấy không phải**: cùng ca đó lên hạng 1 kể cả khi bỏ hẳn cảnh B.

Nhưng một ca không phải một phép đo. Cùng cấu hình (W=5, λ=1) thì
`query-p1-3-qa` tụt **5 → 54** và `query-p1-12-kis` tụt **12 → 27**. Lane này
ăn ở đuôi tốt và mất ở giữa; trên 16 câu thì tổng vẫn âm.

Tệ hơn: các ca hỏng **không ổn định theo W**. `query-p1-3-qa` ở λ=1 cho
5 → 89 (W=2), 5 → **∞** (W=3, rơi hẳn khỏi 100 dòng), 5 → 54 (W=5), 5 → 9
(W=8). Một tín hiệu mà đổi W một nấc làm video tụt từ hạng 9 ra ngoài danh
sách thì chưa đủ ổn định để ship, kể cả nếu trung bình có dương.

Script: `scripts/experiment_cap_thoi_gian.py --de round1/queries --picks round1/picks_verified.txt --W 2,3,5,8 --lam 1.0,0.5,0.25`.
Cache: `data/cache_cap_thoi_gian/diff_de_W*_hm_l*.json` (12 file, kèm top-10 từng câu).

### Diff cấu trúc trên cả 55 câu đề thật

Chạy trên cả hai vòng (28 câu có cổng bật), W=5, HM, λ=1: **video dòng 1 đổi ở
12/28 câu**. Không có ground truth nên đây chỉ nói lane này **đụng nhiều**, và
vì bảng proxy ở trên nói hướng đụng là xấu, "đụng nhiều" ở đây là rủi ro chứ
không phải hứa hẹn.

---

## 3. Chẩn đoán riêng 6 câu nghẽn (5, 9, 12, 15, 40, 41)

Sáu câu này **không câu nào** có cấu trúc hai cảnh, nên công thức cặp thời gian
theo nghĩa đen không áp được. Thay vào đó ta ép tách mỗi câu thành hai **mệnh đề**
hình ảnh và dùng cửa sổ **ĐỐI XỨNG** (`|i-j| <= W`) — tức đang thăm dò một giả
thuyết KHÁC ("chấm theo độ phủ sub-query", mục #10 bảng 33 đề xuất), chỉ mượn
cùng bộ máy. **n = 6, thăm dò, không có kết luận thống kê nào rút ra từ đây.**

Hạng NỘI-VIDEO của keyframe đáp án (nhỏ hơn = tốt hơn), gộp HM:

| câu | #kf | SigLIP thô | W=2 | W=3 | W=5 | W=8 |
|---|---|---|---|---|---|---|
| 5 | 152 | **16** | 24 | 27 | 29 | 30 |
| 9 | 147 | 95 | 66 | 68 | 76 | 78 |
| 12 | 383 | 10 | 9 | 9 | 9 | 9 |
| 15 | 472 | 17 | 61 | 60 | **6** | 7 |
| 40 | 307 | 227 | 166 | 185 | 208 | 195 |
| 41 | 262 | 152 | 170 | 172 | 176 | 134 |

Gộp bằng tích cho bức tranh giống hệt (câu 15: 6 ở W=5; câu 40 vẫn 173–230;
câu 41 vẫn 154–210) — `data/cache_cap_thoi_gian/nghen_tich.json`.

**Kết luận: cửa vẫn đóng.** Không câu nào đi từ "không với tới" sang "với tới
được":

- 9, 40, 41 là ba câu nghẽn thật (hạng 95/227/152). Sau khi tách mệnh đề chúng
  còn **78 / 195 / 134** — cải thiện có thật nhưng vô dụng: vẫn nằm ngoài xa
  mọi cửa sổ mà 100 dòng với tới.
- 12 gần như không đổi (10 → 9).
- 15 tốt lên thật (17 → 6) nhưng vốn đã trong tầm với.
- 5 **tệ đi gấp đôi** (16 → 30), từ trong tầm với ra sát biên.

Con số "cải thiện 5/6" mà script in ra là **cận trên kiểu oracle** (chọn W tốt
nhất cho từng câu SAU khi nhìn kết quả) — không phải thứ ship được. Ở một W cố
định thì tốt nhất là 5 tốt / 1 tệ (W=8) nhưng ba câu nghẽn thật vẫn ở
78/195/134.

Điều này khớp và củng cố bảng đã có trong `SHIP_PHU_XAC_SUAT.md` §4.5 (làm mượt
thời gian, CLIP z-blend, OCR đều không nổ). Nhóm 6 câu nghẽn vẫn chỉ còn đường
**quét VLM tại trận trên top 3–5 video**.

*Chốt chéo:* cột "SigLIP thô" ở đây (16, 95, 10, 17, 227, 152) khớp bảng cũ
(16, 95, 10, 17, 225, 150) với lệch đúng +2 ở hai câu cuối, và `#kf` lệch đúng
+2 ở cả sáu câu — chênh do bảng cũ bỏ 2 keyframe đầu mỗi video (`skip_first_n`)
còn ở đây đếm hết. Hai phép đo độc lập cho cùng một kết quả.

---

## 4. Bất biến: câu không có cổng phải ra dòng y HỆT

Đòi hỏi của lane: câu không có cấu trúc hai cảnh **giữ nguyên 100%** đường cũ,
kiểm bằng assert chứ không bằng mắt.

**Phép kiểm rỗng (đã chạy, nhưng phải nói rõ là rỗng):** trên 60 câu GT, 24 cấu
hình `(W ∈ {2,3,5,8}) × (HM, tích, chiA) × (λ ∈ {1; 0,5})` đều cho 100 dòng
giống hệt nền, từng dòng một. Nhưng vì **không câu nào qua cổng**, phép assert
này chỉ chứng minh mã không chạy — nó không chứng minh cổng đúng.

**Phép kiểm thật (`--tu-kiem`):** ép bật cổng ở đúng 6 câu (dùng bản tách mệnh
đề đã cache) rồi đòi HAI điều cùng lúc trên cả 24 cấu hình:

| | kết quả |
|---|---|
| 54 câu cổng TẮT ra dòng giống hệt nền, từng dòng | **OK — 24/24 cấu hình** |
| 6 câu cổng BẬT thật sự có dòng đổi | **6/6 — 24/24 cấu hình** |

Nếu chỉ chạy phép kiểm rỗng thì một cổng hỏng thành no-op sẽ lọt lưới và mọi
kết quả "hoà" về sau đều là hoà giả. Đó là lý do `--tu-kiem` tồn tại.

---

## 5. Một quyết định thiết kế phải nói ra: chuẩn hoá thang điểm

`s_temp` sống trên thang [0,1]; điểm gốc của ứng viên trải **0,02–0,11**. Bộ
phân bổ phủ xác suất lấy `softmax` **nhiệt 0,02** trên điểm. Ném thẳng `s_temp`
vào allocator là đo **ĐỘ TRẢI** chứ không đo cấu trúc thời gian: softmax sẽ suy
biến gần như về argmax và coverage sụp còn một video — kết quả sẽ âm nặng vì
một lý do chẳng liên quan gì tới lever.

Nên `ve_thang_diem_goc()` đưa `s_temp` về đúng trung bình/độ lệch chuẩn của
điểm gốc trước khi đưa vào allocator. **Đây không phải một tham số để quét —
là điều kiện bắt buộc của phép so sánh**, cố định, không nằm trong lưới. Phần
chênh còn lại đúng là do THỨ TỰ và KHOẢNG CÁCH tương đối giữa các ứng viên.

Tương tự, cosine SigLIP chạy từ −0,21 tới +0,29 trên kho này, mà HM và tích chỉ
có nghĩa với số dương — nên hai điểm được chuẩn hoá về [0,1] **trên giá đỡ đang
xét** (keyframe của các video đang có ứng viên), không phải trên toàn kho: toàn
kho bị đuôi âm kéo, làm mọi ứng viên dồn vào dải hẹp và HM suy biến thành trung
bình cộng.

---

## 6. Điều chưa biết / cạm bẫy cho người đọc sau

1. **Chỉ MỘT khẳng định trong tài liệu này là cứng:** 0/60 câu GT qua cổng, nên
   lever không đo được trên bộ đo chính thức. Mọi thứ khác (§2a, §2b, §3) là
   chẩn đoán trên n nhỏ với sự thật chưa chắc chắn.
2. **Đừng trích thành "đã bị bác bỏ".** §2b — lát cắt tin cậy nhất — hơi dương.
3. **Đừng trích thành "có tiềm năng".** §2a — lát cắt lớn hơn — âm. Hai chiều
   trích dẫn đối lập đều sai như nhau; tài liệu này nói *chưa biết*.
4. **Đừng trích thành "cấu trúc thời gian là vô dụng".** Thứ được thử là **một
   công thức cụ thể** (`HM(s_A, max_j s_B)` trên keyframe, cửa sổ W). 51% đề
   thật có cấu trúc hai cảnh; công thức khác vẫn có thể ăn — xem mục 7 dưới.
5. **Nhãn hai cảnh là của một LLM**, chưa ai soát tay 55 câu đề thật. Kiểm từ
   vựng độc lập chỉ chống được chiều "0/60 là giả", không kiểm được chiều
   "28/55 có bị bật thừa không".
6. **Proxy §2a nhiễu theo hướng chưa biết.** `picks_verified.txt` chưa từng được
   BTC chấm; nếu tỉ lệ pick sai cao thì bảng đó có thể đang phạt s_temp vì nó
   **bất đồng với người soát**, chứ không phải vì nó sai.
7. **Vòng 2 chỉ 40% câu có cổng bật, vòng 1 là 64%.** Hai vòng, hai người ra đề,
   chênh 24 điểm phần trăm. Đừng coi 51% là hằng số của giải.
8. **Đối chứng `chiA` chỉ hơn `hm` trên §2a, không hơn trên §2b.** Nó vẫn là
   nhánh rẻ đáng đo tiếp, nhưng câu "ghép cặp là phần gây hại" **chỉ đúng trên
   lát cắt §2a** và không được nâng lên thành kết luận chung.
9. **Chưa thử chấm ở tầng VIDEO.** Toàn bộ đo đạc ở đây đổi điểm từng keyframe.
   Một biến thể chưa chạm tới: dùng cặp (A trước, B sau) chỉ để **nâng cả
   video** rồi giữ nguyên thứ tự keyframe trong video. Bảng ④ trong
   `NGHIEN_CUU_SOTA.md` nói tầng video và tầng keyframe phản ứng NGƯỢC nhau với
   cùng tín hiệu, nên kết quả ở tầng keyframe không suy ra được tầng video.
   Ghi chú ủng hộ: hai ca dương rõ nhất của §2b (p2-22 `∞ → 48`,
   p1-19 `∞ → 99`) đều là **kéo cả video vào danh sách**, đúng dạng tác dụng
   mà tầng video hứa.

---

## 7. Tái lập

```bash
# 1) gắn nhãn (đã cache; --refresh mới tốn quota)
python scripts/gan_nhan_hai_canh.py                                # 60 câu GT
python scripts/gan_nhan_hai_canh.py --de round1/queries round2/queries

# 2) cổng TUNE/TEST trên GT (in ra: 0 câu qua cổng + assert bất biến rỗng)
python scripts/experiment_cap_thoi_gian.py

# 3) bất biến THẬT (ép bật cổng 6 câu) — phép kiểm đáng tin duy nhất
python scripts/experiment_cap_thoi_gian.py --tu-kiem

# 4) proxy vòng 1 + quét lưới (bảng §2)
python scripts/experiment_cap_thoi_gian.py --de round1/queries \
       --picks round1/picks_verified.txt --W 2,3,5,8 --lam 1.0,0.5,0.25

# 4b) ĐỐI CHỨNG bỏ ghép cặp — bảng tách nguyên nhân trong §2a
python scripts/experiment_cap_thoi_gian.py --de round1/queries \
       --picks round1/picks_verified.txt --W 5 --lam 1.0,0.5,0.25 --gop chiA

# 4c) lát cắt GT ĐÃ KIỂM CHỨNG (bảng §2b) — chạy cả hai gộp
python scripts/experiment_cap_thoi_gian.py --de round1/queries round2/queries \
       --gt-de data/ground_truth_de_that.json --W 5 --lam 1.0,0.5 --gop hm
python scripts/experiment_cap_thoi_gian.py --de round1/queries round2/queries \
       --gt-de data/ground_truth_de_that.json --W 5 --lam 1.0,0.5 --gop chiA

# 5) diff cấu trúc cả 55 câu đề thật (số 12/28 ở §2c nằm trong cache của 4c)
python scripts/experiment_cap_thoi_gian.py --de round1/queries round2/queries

# 6) chẩn đoán 6 câu nghẽn (bảng §3)
python scripts/experiment_cap_thoi_gian.py --nghen --gop hm
python scripts/experiment_cap_thoi_gian.py --nghen --gop tich
```

Windows: chạy với `PYTHONIOENCODING=utf-8`.

**File mới của lane này (không sửa file sản xuất nào):**

| file | vai trò |
|---|---|
| `scripts/gan_nhan_hai_canh.py` | gắn nhãn hai cảnh, GT + đề thật, cache từng câu |
| `scripts/experiment_cap_thoi_gian.py` | 4 chế độ: cổng GT, `--tu-kiem`, `--de`, `--nghen` |
| `data/cache_cap_thoi_gian/picks_r1.json` | picks vòng 1 đã parse (§2a) |
| `docs/CAP_THOI_GIAN.md` | tài liệu này |
| `data/cache_cap_thoi_gian/nhan/` | 60 nhãn GT |
| `data/cache_cap_thoi_gian/nhan_de/` | 55 nhãn đề thật |
| `data/cache_cap_thoi_gian/sims/` | 74 vector tương đồng theo cảnh (float16) |
| `data/cache_cap_thoi_gian/tach_menh_de/` | 6 bản tách mệnh đề của câu nghẽn |
| `data/cache_cap_thoi_gian/diff_round1_picks_*.json` | 15 ô lưới của §2a (12 `hm` + 3 `chiA`) |
| `data/cache_cap_thoi_gian/diff_round1-round2_gtde_*.json` | 4 ô lưới của §2b |
| `data/cache_cap_thoi_gian/nghen_*.json` | bảng §3 |
| `data/cache_cap_thoi_gian/truc_video.npz` | dòng thời gian keyframe theo video |

Mọi con số trong tài liệu này có file cache đứng sau.

---

## 8. Đề nghị cho người điều phối

1. **KHÔNG tích hợp gì vào sản xuất từ lane này.** Không có cấu hình nào qua
   được cổng đo, và hai lát cắt thay thế mâu thuẫn về dấu. Không phải vì lever
   đã bị bác — vì chưa đo được.
2. **Ghi vào `KIEN_TRUC_VA_HUONG_CAI_THIEN.md`** hàng tín hiệu: *cặp thời gian
   (HM trên keyframe, cửa sổ W) — **CHƯA ĐO ĐƯỢC**: 0/60 câu GT qua cổng; hai
   lát cắt thay thế (picks vòng 1 n=16: âm; GT đã kiểm chứng n=6: hơi dương)
   mâu thuẫn nhau.* Đừng ghi vào cột "cửa đã đóng" — cửa này chưa đóng, nó chưa
   được mở.
3. **Việc đáng mang đi nhất: SỬA ĐẶC TẢ của lane harness ② TRƯỚC KHI họ viết
   thêm câu.** Câu GT mới phải được viết bằng cách **xem một ĐOẠN video**, không
   phải nhìn một keyframe rồi tả lại. Bằng chứng: 0/60 câu viết theo lối cũ có
   cấu trúc hai cảnh, so với 28/55 câu của BTC. Bộ 120 câu viết theo lối cũ sẽ
   mù y hệt bộ 60 câu hiện tại, và khi đó ba nhánh (③, ⑨, TRAKE neo hai biên)
   vẫn đứng nguyên chỗ cũ. **Chi phí sửa đặc tả bằng 0 nếu sửa trước khi viết;
   bằng cả bộ câu nếu phát hiện sau.** Đây là việc cần làm ngay, không chờ.
4. **Nếu quay lại lever này, hai nhánh rẻ nên thử trước công thức hiện tại:**
   (a) chấm ở **tầng video** thay vì tầng keyframe (§6.9 — hai ca dương rõ nhất
   đều là kéo cả video vào danh sách); (b) đối chứng `chiA` (cắt truy vấn dài
   lấy mệnh đề đầu), rẻ và đã có mã.
