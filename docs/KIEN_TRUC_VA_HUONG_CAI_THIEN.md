# Hệ thống đang làm gì, và chỗ nào còn cải thiện được

Viết ở mốc **8,6/24 điểm** (5,8 → 7,2 → 7,8 → 8,6).

---

## 1. Đường đi của một câu hỏi

```
đề bài tiếng Việt
   │
   ├─ dịch sang tiếng Anh (3 lớp dự phòng + cache) ─┐
   │                                                 │
   │  bản dịch tay .en.txt (nếu có) ─────────────────┤
   │                                                 ▼
   │                                    bộ 4 prompt → 1 vector 1152 chiều
   │                                                 │
   ▼                                                 ▼
SigLIP-2 SO400M-384  ×  177.321 keyframe  →  400 ứng viên  (một phép nhân ma trận)
                                                   │
                          ┌────────────────────────┤
                          ▼                        ▼
             điểm cộng nhận dạng đối tượng   xếp hạng lại bằng VLM
                (tầng VIDEO, w=0.01)          (tầng VIDEO, w=0.02)
                    ĐO ĐƯỢC +3,3%                ĐO ĐƯỢC +3,3%
                          └────────────┬───────────┘
                                       ▼
                          phân bổ 100 dòng  (n_flat=30, ±10/±20)
                                       │
                    KIS/Q&A ───────────┼─────────── TRAKE
                    thang frame        │      quy hoạch động CHRONOS
                                       │      + lưới bù trừ 4 chiều
                                       ▼
                              submission.zip  →  bộ kiểm tra định dạng
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
            review.html (mắt người)              apply_picks.py
       kéo thả · trình phát YouTube ·          chốt tay đè lên máy
       chốt frame bằng phím C · OCR ·
       màu · lời thoại · xuất zip
```

**Nguyên tắc xuyên suốt:** mọi tín hiệu chỉ được vào đường chấm điểm sau khi
**đo trên 60 câu ground truth bằng công thức chính thức, với đáp án rút thăm
không snap keyframe**. Không đo thì đi tới mắt người, không đi tới bảng xếp hạng.

---

## 2. Bảng tổng kết mọi thứ đã đo

| tín hiệu | tầng | kết quả | có dùng? |
|---|---|---|---|
| thang frame ±10/±20 thay vì chỉ keyframe | frame | **+26% TB, +120% cửa sổ hẹp** | ✅ |
| dịch tay vào cùng một vector (không gộp 2 danh sách) | truy vấn | **+10,5%** | ✅ |
| nhận dạng đối tượng | **video** | **+3,3%** | ✅ |
| nhận dạng đối tượng | frame | +0,4% (nhiễu) | ❌ |
| khớp **số lượng** đối tượng | frame | **±0,0%** (trơ hoàn toàn) | ❌ |
| VLM Gemini | **video**, w=0.02 | **+3,3%** | ✅ |
| VLM Gemini | video, w=0.20 | **−5,7%** (vR@1 tăng 25→30!) | ❌ |
| lời thoại | video | −0,4% | ❌ |
| ↳ *(phủ sóng thật là **849/873 = 97%**, không phải 24,9% — con số kia là riêng `data/captions`)* | | | |
| ↳ *(mốc thời gian lệch trung vị **2.850 frame** ⇒ vĩnh viễn chỉ là bộ lọc **cấp video**)* | | | |
| lời thoại có cổng chặn | video | +0,5% (nhiễu) | ❌ |
| lời thoại theo mốc thời gian | **frame** | −1,5% | ❌ |
| metadata video (tiêu đề, mô tả) | video | −7,6% | ❌ |
| làm mượt theo thời gian | frame | −0,023 | ❌ |
| chuẩn hoá theo video | frame | −0,129 | ❌ |
| viết lại câu hỏi | truy vấn | 1 tốt / 3 xấu | ❌ |

Ba dòng in đậm ở nhóm "❌" là bài học đắt nhất của dự án: **video R@1 tăng trong
khi điểm thi giảm**. Đã gặp hai lần, với hai tín hiệu khác nhau.

| **ưu tiên đỉnh cục bộ** (keyframe nổi hơn hai bên) | **frame** | **+2,2%** | ✅ |
| VLM xếp lại frame *bên trong* video | frame | −4,6% | ❌ |
| ↳ đo lại bằng **câu hỏi phân biệt**, trộn kiểu gom-khối | frame | ~~−34,7%~~ **ĐO NHẦM** | ⚠ |
| ↳ đo lần ba, hoán vị **giữ-slot** (sạch): nguyên văn đề | frame | +0,6% (hoà) | ❌ |
| ↳ giữ-slot, câu hỏi phân biệt | frame | −5% → −11% | ❌ |
| ↳ đối chứng: chính phép gom-khối, KHÔNG đổi thứ tự frame nào | frame | **−35% → −39%** | 🔬 artifact |
| ép các frame cách nhau ≥30/60/120 | frame | −1,3% | ❌ |
| **đặt nhiều đáp án Q&A trên các dòng khác nhau** | dòng | **±0,0%** | ❌ |
| chia lại ngân sách theo (video, keyframe, độ sâu) | dòng | −15% đến −30% | ❌ |
| **chia 100 dòng bằng PHỦ XÁC SUẤT (bỏ hẳn hàm chi phí)** (0,02; σ=30; nửa=6) | **dòng** | **+15,3% TEST** (chọn trên TUNE lẻ +15,7%, chấm 1 lần trên TEST chẵn, >2σ; seed mới +13,9→+16,2%) | ✅ ship |
| ↳ tổ hợp argmax fold xuôi (0,03; 30; 10) | dòng | TUNE chẵn +20,1% → **TEST lẻ −1,0% (HOÀ)** — overfit, protocol chặn đúng | ❌ |
| tiêm keyframe nội-video top-K vào tiên nghiệm bộ phủ | dòng | TUNE −0,4→−1,2%; **TEST +0,2% (HOÀ)** — 54/60 câu đã có coverage, 6 câu nghẽn delta=0,000 | ❌ |
| làm mượt SigLIP theo trục thời gian (định vị nội-video, đo trên 6 câu nghẽn) | keyframe | cứu 1/6 (câu 12: hạng 10→1), phá câu 5 (16→51); câu 40/41 vẫn hạng 200+ | ❌ |
| pha CLIP-B32 vào SigLIP nội-video (z-blend, 6 câu nghẽn) | keyframe | không thêm thông tin; câu 9 tệ hơn (95→141) | ❌ |
| OCR keyword-match định vị nội-video (6 câu nghẽn, 4 video có OCR) | keyframe | 0/4 nổ gần GT — kể cả logo "60 Giây" (chữ 3D cách điệu) | ❌ |

---


> **Vì sao chấm lại bằng VLM không cứu được nhóm 14 câu** — đo ngày 24/08/2026 bằng
> `scripts/experiment_sharp_rerank.py`. Giả thuyết là: kết quả −4,6% cũ được đo bằng
> cách đưa **nguyên văn đề** cho VLM, mà vòng sơ tuyển 1 cho thấy đó là cách hỏi tạo
> ra "cao nguyên điểm". Đo lại với câu hỏi phân biệt tự sinh:
>
> * độ sắc **đúng là cải thiện thật** — tỷ lệ khung được chấm ≥0,60 giảm từ ~45% xuống
>   **20,1%**, đúng như dự đoán;
> * nhưng điểm thi **giảm ở mọi biến thể**, và xếp lại *bên trong video* — tức đúng
>   cái mà giả thuyết nhắm tới — lại là biến thể **tệ nhất, −34,7%**.
>
> Bài học không phải "prompt chưa đủ tốt" mà là **sai mục tiêu**: VLM trả lời câu hỏi
> *"khung này có khớp mô tả không"*, trong khi thứ quyết định điểm là *"khung này có
> gần khoảnh khắc đúng nhất không"*. Trong một video, hàng chục khung cùng khớp mô tả;
> chỉ một khung là gần nhất. Làm cho phép phân loại sắc hơn không hề giúp định vị
> thời gian — và khi ta để nó đảo thứ tự keyframe, nó cướp mất thang phân bổ sâu từ
> keyframe mà SigLIP-2 đã chọn đúng 48% số lần.
>
> **Đo lần ba (28/08, giữ-slot — sau khi phản biện phát hiện artifact):** con số
> −34,7% hoá ra đo nhầm — phép trộn gom-khối kéo cả khối frame của video 1 lên
> đầu, tự nó gây −35% → −39% *dù không đổi thứ tự frame nào*. Bản đo sạch
> (hoán vị giữ nguyên slot video) cho: nguyên văn đề +0,6% (hoà), câu hỏi phân
> biệt −5% → −11%. Và độ đo không dính bộ phân bổ nói rõ vì sao: **trong các
> slot đã truy xuất, SigLIP thuần đã đặt keyframe gần đáp án ở hạng nội-video
> trung vị 1,0 (hạng-1: 60%)** — không còn chỗ cho bộ xếp lại nào.
>
> Hệ quả cho hướng đi (thay kết luận cũ): vấn đề của nhóm thất bại **không phải
> thứ tự slot mà là keyframe đúng không được truy xuất vào slot nào** (nhất quán
> với phép mổ xẻ "9/15 câu keyframe đúng cách ứng viên hạng-1 hơn 1.000 frame").
> ~~Việc đáng đo tiếp theo: **thêm ứng viên** — nạp toàn bộ keyframe của các video
> dẫn đầu vào tiên nghiệm của bộ phân bổ phủ xác suất.~~ **ĐÃ ĐO (28/08,
> `experiment_phu_noi_video.py`): KHÔNG ăn** — TEST +0,2% (HOÀ), TUNE âm nhẹ.
> Tiền đề sai trên toàn tập: 54/60 câu **đã có** keyframe gần đáp án nhất trong
> 400 ứng viên; ở 6 câu nghẽn (chỉ số 5, 9, 12, 15, 40, 41) delta = 0,000 vì
> SigLIP xếp keyframe đáp án hạng nội-video 95–227 ngay trong video đúng.
> Con số +27,5% là **cận trên oracle**, không phải mục tiêu với tới được bằng
> tiêm ứng viên — cần tín hiệu định vị nội-video KHÁC SigLIP thuần
> (chi tiết: `docs/SHIP_PHU_XAC_SUAT.md` §2).



> **Chia 100 dòng bằng phủ xác suất — mức tăng đơn lẻ lớn nhất đo được tới nay.**
> Đo lần đầu 24/08/2026 bằng `scripts/experiment_phu_xac_suat.py` (+10,0% với
> tổ hợp 0,02/σ=30/nửa=10 trên cả 60 câu). Quét lưới 48 tổ hợp 28–29/08 bằng
> `scripts/experiment_phu_quet_luoi.py` với chia **TUNE/TEST** chẵn/lẻ và hạt
> giống tách rời — số quyết định là số **TEST** (nửa câu chưa từng dùng để chọn):
>
> | | điểm | so nền |
> |---|---|---|
> | nền TEST chẵn (bộ phân bổ đang nộp `i + 0,5·d`) | 0,3421 ± 0,0041 | — |
> | **phủ xác suất (nhiệt 0,02, σ=30, nửa cửa sổ 6) — TEST chẵn, chọn trên TUNE lẻ** | **0,3946 ± 0,0028** | **+15,3%** (>2σ) |
> | cùng tổ hợp, TUNE lẻ (nơi nó được chọn) | 0,4097 | +15,7% |
> | tổ hợp argmax fold xuôi (0,03; 30; 10) — TUNE chẵn +20,1% | TEST lẻ 0,3491 | **−1,0% HOÀ = overfit** |
> | phủ xác suất, nhiệt 0,05 (đo 24/08, 60 câu) | 0,3299 ± 0,0018 | −5,6% |
>
> Kiểm chứng độc lập bằng 2 họ hạt giống hoàn toàn mới (70000, 123450): giữ
> **+13,9% → +16,2%** trên mọi lát cắt. Kỳ vọng thận trọng cho kế hoạch điểm:
> **≈ +7%** (trung bình out-of-fold 2 fold). Số câu có video đúng trong 100 dòng
> (fold xuôi): **27/30 → 30/30**.
>
> **Vì sao nó khác `experiment_per_video_depth.py`** (đã đo −15% đến −30%): cái đó
> vẫn đi theo chi phí tuyến tính `A·v + B·m + C·d`, chỉ đổi hệ số. Cái này **bỏ hẳn
> khái niệm chi phí**. Suy thẳng từ luật chấm thì bài toán có tên: vì `R@k` là *max*
> trên tiền tố, mỗi câu chỉ cần **một** dòng trúng — mọi dòng trúng thêm đều vô giá
> trị. Đó là cấu trúc của bài toán **phủ cực đại có trọng số**, không phải bài toán
> xếp hạng. Đặt tiên nghiệm p(v, f) cho vị trí khoảnh khắc thật rồi tham lam chọn
> dòng phủ được nhiều khối lượng **chưa phủ** nhất; trọng số hạng giảm dần
> (1,00 / 0,80 / 0,60 / 0,40 / 0,20) khiến thứ tự tham lam đúng luôn.
>
> **ĐÃ CHỐT SHIP (29/08)** với tổ hợp **nhiệt 0,02, σ=30, nửa cửa sổ 6, lưới 5**
> — kế hoạch từng bước, cổng hồi quy trước merge, và mục "điều chưa biết" nằm ở
> `docs/SHIP_PHU_XAC_SUAT.md`. Đường rút lui là cờ `--allocator hybrid`.
>
> **CỔNG BẢN SHIP ĐÃ XANH (29/08 chiều)** — `scripts/so_sanh_allocator.py` chấm
> đúng đường mã `allocate_rows` của make_submission (làm tròn 4 chữ số + lượng tử
> hoá 1e-9 + đuôi lấp): nửa chẵn **+15,3%**, nửa lẻ **+16,0%**, trùng số thí
> nghiệm tới 4 chữ số; độ trễ 167 ms/câu (max 578 ms). Chi tiết + bảng từng câu:
> `docs/SHIP_PHU_XAC_SUAT.md` §3b.


## 2b. Điểm đang mất ở ĐÂU — phép đo quan trọng nhất

Trước khi tối ưu bất cứ thứ gì, phải biết mất ở đâu. Đo trên 60 câu ground truth:

| nếu sửa hoàn hảo | điểm | tăng |
|---|---|---|
| hiện tại | 0,345 | — |
| xếp hạng video hoàn hảo | 0,487 | +41% |
| **vị trí frame hoàn hảo** | **0,740** | **+115%** |
| cả hai | 1,000 | — |

**60% phần điểm lấy lại được nằm ở VỊ TRÍ FRAME, chỉ 22% ở xếp hạng video.**

Điều đó lật ngược trực giác: OCR, VLM, lời thoại đều cải thiện *chọn video* — tức
phần 22%. Nhiều câu có video đúng ở **hạng 1** mà vẫn **0 điểm** vì không frame
nào rơi vào cửa sổ.

Mổ xẻ 22 câu trượt:

| nguyên nhân | số câu |
|---|---|
| keyframe đúng **có** trong ứng viên, **thang không vươn tới** | **14** |
| video không có trong 100 dòng | 5 |
| keyframe đúng không có trong 400 ứng viên | 3 |

Cơ chế: bộ phân bổ chi `cost(i, d) = i + 0,5·d`, nên ứng viên hạng 1 được thang
vươn tới ±120 frame, còn hạng 25 chỉ được **một dòng phẳng không thang**.

Và: keyframe *gần sự thật nhất* đứng hạng 1 trong video đúng chỉ **48%** số lần,
nhưng nằm trong top-5 tới **76%**. Rải thang quanh keyframe hạng 1 phủ **55%**
số câu; nếu chọn đúng keyframe thì phủ **98%**.

Đó là lý do "ưu tiên đỉnh cục bộ" ăn tiền: nó đẩy keyframe đúng lên vài bậc, tới
chỗ đã sẵn có thang sâu.

**Ba cách chia lại ngân sách đều đã thử và đều tệ hơn.** Cách chia hiện tại
(30 dòng phẳng rồi quét theo chi phí) là tối ưu — đo bằng ba phép quét độc lập.

---

## 3. Ba giới hạn cứng, đã định lượng

**(a) Keyframe thưa hơn cửa sổ đáp án 5–9 lần.** Cửa sổ dưới 10 frame; keyframe
cách nhau trung vị 55 frame (video múa lân: 92). Một keyframe đơn lẻ trúng cửa
sổ **13,3%** số lần. Thang ±10/±20 nâng lên trần **54,9%** mỗi sự kiện.

**(b) Với TRAKE 4 sự kiện, 100 dòng không phủ nổi lưới bù trừ.** Cần 625 tổ hợp,
có 100 → phủ 16%. Kể cả mô hình **hoàn hảo** cũng chỉ đạt 0,45 trên video múa lân.

**(c) 8/12 sự kiện TRAKE nằm dưới ngưỡng nhiễu.** Đỉnh điểm của SigLIP không nổi
hơn một đỉnh ngẫu nhiên — nó là mô hình **ảnh tĩnh**, không biểu diễn được
"bắt đầu xoay vòng", "4 chân chạm đất", "khoảnh khắc đầu tiên".

---

## 4. Chỗ còn cải thiện được, xếp theo giá trị trên công sức

### 4.1 Mở rộng tầm nhìn của VLM — **giá trị cao nhất, chưa làm**

VLM chỉ nhìn **top-24 của SigLIP**. Nếu video đúng không nằm trong đó thì nó bó
tay. Đúng chuyện đã xảy ra: `p1-19` và `p1-22` đều ở **ngoài** top-24, và chỉ
kênh lời thoại mới tìm ra.

Cách làm: gộp ứng viên từ **ba nguồn** rồi mới đưa VLM chấm — top-24 hình ảnh +
top-5 lời thoại (BM25) + top-5 OCR. Chi phí thêm ~$0,03/vòng. Đây là lỗ hổng
kiến trúc rõ nhất còn lại.

### 4.2 OCR toàn kho, chạy trước ngày thi — **cần thời gian, không cần tiền**

Hiện chỉ OCR **ứng viên của vòng** (564 khung, 25 phút). 48% khung hình có chữ,
mà chữ đó là dòng tiêu đề tin tức — thứ trả lời trực tiếp nhiều câu.

177.321 khung × 4 giây = 8 ngày trên 1 lõi CPU. Nhưng:
- chia 8 tiến trình → **1 ngày**
- hoặc Gemini Flash-Lite: 177k ảnh ≈ **$9**, vài giờ
- hoặc chỉ OCR nhóm tin tức (L21–L23, ~85 video) → vài giờ

Có OCR toàn kho thì `search_ocr.py` thành công cụ tìm kiếm thật, không phải chỉ
tra lại những gì đã chọn.

### 4.3 Chỉ mục lời thoại thành nguồn ứng viên, không chỉ là công cụ tra

Đã đo: gộp vào điểm thì **âm**. Nhưng đó là đo trên ground truth toàn mô tả cảnh
nhìn thấy. Cách đúng không phải cộng điểm, mà là **thêm ứng viên**: BM25 lấy
top-5 video, đưa keyframe của chúng vào danh sách cho VLM chấm. R@k là max trên
tiền tố nên thêm ứng viên không bao giờ hại — chỉ tốn chỗ xếp hạng.

### 4.4 TRAKE: lấy mẫu dày khi cho VLM chấm

Bài học từ `p1-4`: **8 khung rải đều cho âm tính giả cả ba video**; 16 khung nửa
sau cho kết quả đúng (100 điểm cho video đúng, 20 cho video sai). Bộ chấm TRAKE
hiện chỉ đưa **đúng 4 frame của chuỗi** cho VLM — quá thưa. Nên đưa thêm ±3
keyframe quanh mỗi sự kiện.

### 4.5 Frame chính xác — chỉ người làm được

Với các sự kiện dưới ngưỡng nhiễu, không mô hình nào trong tay ta chốt được
frame. `review.html` đã có trình phát YouTube, đi từng frame bằng `←`/`→`, lấy
thời điểm bằng phím `C`. **Đây vẫn là nguồn điểm lớn nhất trong 3 tiếng thi**, và
nó không tốn gì ngoài thời gian người.

### 4.6 Đã thử hôm nay và KHÔNG dùng (đo được, ghi lại để khỏi làm lại)

**Đặt nhiều đáp án Q&A trên các dòng khác nhau.** Luật mục 2.1.2 cho phép:
`R-Score(rᵢ) = I(vᵢ=GT_v ∧ idᵢ∈[s,e] ∧ **aᵢ**=GT_a)` — đáp án có chỉ số dòng.
Về lý thuyết có thể đặt đáp án 1 ở dòng 1–4, đáp án 2 ở dòng 5, và R@1 không đổi
nên không thể lỗ.

Đo trên 60 câu ground truth (đều có đáp án chuẩn), cho model xem **đúng khung
hình**:

* đáp án thứ nhất đúng: **81%**
* đáp án thứ hai cứu thêm: **0%**

Khi model sai, nó sai theo *cùng một kiểu*. Chuẩn "Cá cơm" → model trả
"Cá nhỏ / Tôm nhỏ / Mực nhỏ"; chuẩn "Tượng chằn Khmer" → model mô tả dài dòng mà
không gọi được tên. Đó là lỗi **độ cụ thể**, không phải **mơ hồ** — thêm phương
án không cứu được.

Con số 81% cũng nói một điều quan trọng: **Q&A coi như đã xong nếu có đúng
frame.** Bộ trả lời không phải chỗ nghẽn.

**VLM xếp lại frame bên trong video: −4,6%.** VLM chọn được đúng *video* (+3,3%)
nhưng chọn *frame* thì tệ hơn embedding — vì frame nó thích nhất không phải
frame gần khoảnh khắc đáp án nhất. Đúng cái bẫy cũ, lần thứ ba.

**Ép các frame ứng viên cách nhau tối thiểu 30/60/120 frame: −1,3%.**

---

### 4.7 Những thứ ĐỪNG làm

- **Đừng** đổi encoder (PE-Core, InternVideo…): nhúng lại 177k khung trên CPU mất
  hàng ngày, lợi ích chưa đo được trên độ đo này.
- **Đừng** mua mô hình temporal grounding chuyên dụng: CoMET-Bench cho thấy chúng
  thua trên bài toán nhiều sự kiện có điều kiện, và benchmark loại này giải được
  ~92% chỉ bằng tiên nghiệm (Otani et al., BMVC 2020).
- **Đừng** tăng trọng số VLM: đo được là **âm** từ 0,05 trở lên.
- **Đừng** đảo cả top-50 bằng reranker: INQUIRE (NeurIPS 2024) cho thấy phần lớn
  reranker làm *tệ hơn* baseline SigLIP SO400M.

---

## 5. Chi phí thực tế

| việc | thời gian | tiền |
|---|---|---|
| dựng bài nộp nền | 90 giây | 0 |
| VLM xếp hạng lại 24 câu | 10 phút | $0,08 |
| OCR ứng viên của vòng | 25 phút (nền) | 0 |
| trả lời 3 câu Q&A | 1 phút | $0,004 |
| **tổng một vòng thi** | **~40 phút máy** | **~$0,09** |

**Hạn mức miễn phí: 500 request/ngày cho mỗi model.** Một vòng dùng ~135 request,
tức khoảng 3 vòng/ngày trên một model. Quota tính riêng từng model, nên khi hết
`gemini-3.5-flash-lite` thì chuyển sang `gemini-3.1-flash-lite` là chạy tiếp
được — điều này đã xảy ra hôm nay và cần biết trước ngày thi.

Toàn bộ nghiên cứu hôm nay: **$0,22**.

Model: `gemini-3.5-flash-lite`. Các tầng pro không chính xác hơn trên bài toán
cụ thể này, chỉ chậm hơn và hay lỗi 503.
