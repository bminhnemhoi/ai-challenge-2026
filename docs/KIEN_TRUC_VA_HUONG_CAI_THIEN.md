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

> ### ⚠60 — ĐỌC TRƯỚC BẢNG NÀY
>
> Ký hiệu **⚠60** đánh dấu những dòng được đo **chỉ trên bộ 60 câu cũ**. Bộ đó
> đã được chứng minh là **thổi phồng năng lực ~2 lần** (0,4004 so với 0,1690 của
> bộ khớp phân bố) và có **0/60** câu mô tả hai cảnh nối tiếp, trong khi đề THẬT
> có 51%. **Một dòng ⚠60 có thể không còn đúng** — nó không bị xoá vì nó là bộ
> đối chứng lịch sử, nhưng đừng dùng nó để đóng cửa cho ai.
>
> Ba dòng ⚠60 nguy hiểm nhất **đã được mở lại bằng số** — xem §2a và
> `docs/KE_HOACH_DINH_VI.md` §2.2.

| tín hiệu | tầng | kết quả | có dùng? |
|---|---|---|---|
| thang frame ±10/±20 thay vì chỉ keyframe | frame | **+26% TB, +120% cửa sổ hẹp** | ✅ |
| dịch tay vào cùng một vector (không gộp 2 danh sách) | truy vấn | **+10,5%** | ✅ |
| nhận dạng đối tượng | **video** | **+3,3%** ⚠60 | ✅ |
| nhận dạng đối tượng | frame | +0,4% (nhiễu) ⚠60 | ❌ |
| khớp **số lượng** đối tượng | frame | **±0,0%** (trơ hoàn toàn) ⚠60 | ❌ |
| VLM Gemini | **video**, w=0.02 | **+3,3%** ⚠60 | ✅ |
| VLM Gemini | video, w=0.20 | **−5,7%** (vR@1 tăng 25→30!) ⚠60 | ❌ |
| lời thoại | video | −0,4% ⚠60 | ❌ |
| ↳ *(phủ sóng thật là **849/873 = 97%**, không phải 24,9% — con số kia là riêng `data/captions`)* | | | |
| ↳ *(mốc thời gian lệch trung vị **2.850 frame** ⇒ vĩnh viễn chỉ là bộ lọc **cấp video**)* | | | |
| lời thoại có cổng chặn | video | +0,5% (nhiễu) ⚠60 | ❌ |
| lời thoại theo mốc thời gian | **frame** | −1,5% ⚠60 | ❌ |
| metadata video (tiêu đề, mô tả) | video | −7,6% ⚠60 | ❌ |
| làm mượt theo thời gian | frame | −0,023 ⚠60 *(đo lại trên bộ mới → vẫn ❌, xem §2a)* | ❌ |
| chuẩn hoá theo video | frame | −0,129 ⚠60 *(đo lại trên bộ mới → vẫn ❌, xem §2a)* | ❌ |
| viết lại câu hỏi | truy vấn | 1 tốt / 3 xấu ⚠60 | ❌ |

Ba dòng in đậm ở nhóm "❌" là bài học đắt nhất của dự án: **video R@1 tăng trong
khi điểm thi giảm**. Đã gặp hai lần, với hai tín hiệu khác nhau.

| **ưu tiên đỉnh cục bộ** (keyframe nổi hơn hai bên) | **frame** | **+2,2%** ⚠60 *(trơ tuyệt đối dưới `coverage` — §2a)* | ✅ |
| VLM xếp lại frame *bên trong* video | frame | −4,6% ⚠60 | ❌ |
| ↳ đo lại bằng **câu hỏi phân biệt**, trộn kiểu gom-khối | frame | ~~−34,7%~~ **ĐO NHẦM** ⚠60 | ⚠ |
| ↳ đo lần ba, hoán vị **giữ-slot** (sạch): nguyên văn đề | frame | +0,6% (hoà) ⚠60 **← PHÉP ĐỒNG NHẤT dưới `coverage`, §2a** | ❌ |
| ↳ giữ-slot, câu hỏi phân biệt | frame | −5% → −11% ⚠60 | ❌ |
| ↳ đối chứng: chính phép gom-khối, KHÔNG đổi thứ tự frame nào | frame | **−35% → −39%** ⚠60 | 🔬 artifact |
| ép các frame cách nhau ≥30/60/120 | frame | −1,3% ⚠60 | ❌ |
| **đặt nhiều đáp án Q&A trên các dòng khác nhau** | dòng | **±0,0%** ⚠60 | ❌ |
| chia lại ngân sách theo (video, keyframe, độ sâu) | dòng | −15% đến −30% ⚠60 | ❌ |
| **chia 100 dòng bằng PHỦ XÁC SUẤT (bỏ hẳn hàm chi phí)** (0,02; σ=30; nửa=6) | **dòng** | **+15,3% TEST** (chọn trên TUNE lẻ +15,7%, chấm 1 lần trên TEST chẵn, >2σ; seed mới +13,9→+16,2%) | ✅ ship |
| ↳ tổ hợp argmax fold xuôi (0,03; 30; 10) | dòng | TUNE chẵn +20,1% → **TEST lẻ −1,0% (HOÀ)** — overfit, protocol chặn đúng ⚠60 | ❌ |
| tiêm keyframe nội-video top-K vào tiên nghiệm bộ phủ | dòng | TUNE −0,4→−1,2%; **TEST +0,2% (HOÀ)** — 54/60 câu đã có coverage, 6 câu nghẽn delta=0,000 ⚠60 **← tiền đề "đã có sẵn" SAI trên bộ mới (hai cảnh 35/65), §2a** | ❌ |
| làm mượt SigLIP theo trục thời gian (định vị nội-video, đo trên 6 câu nghẽn) | keyframe | cứu 1/6 (câu 12: hạng 10→1), phá câu 5 (16→51); câu 40/41 vẫn hạng 200+ | ❌ |
| pha CLIP-B32 vào SigLIP nội-video (z-blend, 6 câu nghẽn) | keyframe | không thêm thông tin; câu 9 tệ hơn (95→141) | ❌ |
| OCR keyword-match định vị nội-video (6 câu nghẽn, 4 video có OCR) | keyframe | 0/4 nổ gần GT — kể cả logo "60 Giây" (chữ 3D cách điệu) | ❌ |
| **Q&A: ảnh gốc + lời thoại ±30 s + cấm bỏ trống + 2 keyframe lân cận + video dự phòng** | **đáp án** | **70,0% → 93,3% TEST** (trọng tài LLM, chọn trên TUNE, >2 sd; `experiment_qa_answer.py --trong-tai`) | ✅ ship |
| ↳ chỉ đổi sang ảnh gốc, giữ prompt cũ | đáp án | 75,0% → **76,7%** cả 60 — gần như không đổi; một mình nó không ăn, model chép băng rôn | ❌ |
| ↳ cấm bỏ trống (11/60 câu đang tự bỏ) | đáp án | 90,0% → **95,0%** cả 60 — yếu tố đơn lẻ mạnh nhất | ✅ ship |
| **gpt-5.2 thay Gemini free ở bước trả lời** | đáp án | **88,3% vs 95,0%** cả 60 — Gemini free THẮNG, còn tốn $0,32/vòng | ❌ |
| **truy vấn CẶP THỜI GIAN** (W=2, gộp tích, λ=0,5) — đo trên bộ đo KHỚP PHÂN BỐ | truy vấn | TEST **+11,5%**, nhưng khoảng tin cậy theo câu **chứa 0** (14% khả năng hoà, n=12 câu qua cổng) | ⏸ chưa ship |
| ↳ đối chứng `chiA` (bỏ hẳn cảnh B) | truy vấn | âm ở MỌI ô (−6% → −10%) — phần đóng góp đúng là từ cảnh B | 🔬 |
| **câu HAI cảnh vs MỘT cảnh** (bộ đo mới, chất lượng mô tả hai nhóm y hệt) | chẩn đoán | 0,1213 vs 0,2158 = **−43,8%**, vượt 2σ — điểm mù đã lượng hoá được | 🔬 |
| **bộ đo 60 câu cũ so với bộ khớp phân bố** | chẩn đoán | 0,4004 vs 0,1690 — bộ cũ **thổi phồng năng lực ~2 lần** | ⚠ |

---

## 2a. Đo trên BỘ KHỚP PHÂN BỐ (132 mục sạch) — 01/09/2026

Bốn lane + hai lượt phản biện. Nền chung: 132 mục (66 MỘT cảnh / 66 HAI cảnh),
allocator `coverage`, chấm qua `allocate_rows()` thật, TUNE/TEST **phân tầng theo
`co_2_canh`**, bootstrap **theo câu** 4000 lần. Quyết định đầy đủ + các bước tích
hợp: **`docs/KE_HOACH_DINH_VI.md`**.

### Cái ĂN — duy nhất

| tín hiệu | tầng | kết quả | có dùng? |
|---|---|---|---|
| **hoán vị ĐIỂM nội-video theo `sim(cảnh B)`** (α=0,5; w=1,0; top-3 video × top-12 khung) | **frame** | TEST 33 câu **+80,5%**, KTC [+0,0337, +0,1604], P(≤0)=0,0%; **cả 66 câu, hạt độc lập: +57,6%**, KTC [+0,0369, +0,1261]; bỏ 3 câu đắt nhất vẫn +38,2% | ✅ **ship** |
| ↳ **kiểm độ bền dưới mô hình bốc SAU_NEO** (hạt 771000, cả 66 câu) | frame | cảnh B +33,4%/+33,3%; hoán vị **+110,1%/+117,9%** — mạnh HƠN dưới mô hình đúng hơn. Cộng dồn hai lever: **0,1049 → 0,2205, hơn gấp đôi** | 🔬 xác nhận |
| ↳ trục sigma dưới cùng phép kiểm ấy | dòng | +20,2% (bốc ĐỀU) → **+5,5%** (SAU_NEO) — **teo 4 lần**: phần lớn "lợi ích" của sigma lớn là ảo ảnh của giả định bốc sai (nó rải khối lượng sang cảnh A, vùng khoảnh khắc thật không bao giờ rơi vào) | ❌ đóng chặt hơn |
| **điểm CẮT tương đối-trong-video → xếp hạng nội-video** (nhóm MỘT cảnh) | frame | chẩn đoán CÓ tín hiệu (phân vị đáp án 0,564 vs ngẫu nhiên 0,505) nhưng TUNE +4,5% → **TEST −3,0% (ĐỀU) / −4,3% (SAU_NEO)**, P(≤0)=77%/88%; đường TUNE **không đơn điệu** = chữ ký nhiễu | ❌ |
| **hai lever đã ship có chạm được kênh ĐÁP ÁN không** | đáp án | **KHÔNG** — ba tập NEN/CANH_B/HOAN_VI cho đáp án **giống hệt** (48,5% cả ba). Đếm tất định: khung neo đổi **0/66** câu khi chọn theo THỨ TỰ DANH SÁCH, **60/66** khi chọn theo ĐIỂM | 🔬 lỗ hổng |
| ↳ vá: chọn khung đọc đáp án theo ĐIỂM | đáp án | đang đo lại | ⏸ |
| ↳ vá SAI đã thử: đọc từ `frame_rows` | đáp án | **tệ hơn** — `coverage` sinh ĐIỂM LƯỚI, chỉ 8–20/100 dòng là keyframe thật nên "dòng đầu là keyframe" là ứng viên yếu; smoke test 0% ở cả ba tập | ❌ đừng thử lại |
| **đường sinh đáp án của `make_submission` ≠ `answer_qa.py`** | đáp án | trên 8 câu đề thật đã kiểm chứng: **1/8 đúng, 4/8 RỖNG** (và 2/3 câu "sai" là model từ chối). Sau khi hợp nhất: **2/8 đúng, 0/8 rỗng** | ✅ đã vá |
| ↳ đo lại cải tiến Q&A trên bộ đo KHỚP PHÂN BỐ | đáp án | **77,3% → 84,1%** = +8,8% tương đối (TEST +4,5 điểm, chưa vượt 2 sd). Con số **+23,3%** cũ là của bộ đo cũ và **thổi phồng**; phần chắc chắn là 11 câu bỏ trống → 0 | ⚠ hiệu chỉnh |
| ↳ cổng độ bền: **4 mô hình bốc** (đều / tam giác / Gauss / sau-neo) | | +57,6% / +55,8% / +50,6% / **+62,8%** — cùng dấu, cùng độ lớn, KTC tách khỏi 0 ở cả bốn | 🔬 |
| ↳ đối chứng khoá **ngẫu nhiên** / **đảo dấu** / **cảnh A** | frame | −0,5%→−8,1% / −32%→−36,7% / −11%→−35,2% — **không đối chứng nào dương** | 🔬 |
| ↳ mục tiêu xác minh **trước** khi nhìn điểm | chẩn đoán | neo trùng khít khung cảnh-B đầu tiên **39/61 = 64%**, trung vị lệch 0 | 🔬 |

### Đóng cửa trên bộ MỚI 🟩 — đừng thử lại

| tín hiệu | tầng | kết quả | có dùng? |
|---|---|---|---|
| quét lại `CoveragePlan` (216 tổ hợp: nhiệt/σ/nửa/lưới) | dòng | TUNE đỉnh +25,4% (hai cảnh +90,6%) → **TEST +1,5%, P(≤0)=37,3% HOÀ**; hai cảnh **−3,8%**; kiểm chéo bộ cũ **−8,2%, P=99,8%** | ❌ |
| ↳ *trục σ **đảo chiều** khi đổi mô hình bốc (đều→σ60 thắng; Gauss→σ15 thắng) ⇒ chưa bao giờ là cải tiến chung* | | | 🔬 |
| chỉnh lại allocator `hybrid` (16 tổ hợp) | dòng | tất cả ≤ nền `coverage`; TEST **−16,2%**, P(≤0)=99,8% | ❌ |
| làm mượt SigLIP theo thời gian (đo lại, bộ mới) | frame | TEST −1,5%; hai cảnh +5,4% **nhưng P(≤0)=17,7%**; cơ chế CÓ THẬT (pool 35→41) mà không quy đổi thành điểm; phá nhóm một cảnh (pool 58→31) | ❌ |
| chuẩn hoá theo video (đo lại, bộ mới) | frame | TEST −6,9%; hai cảnh −11,6%, P=99,7% | ❌ |
| **ưu tiên đỉnh cục bộ dưới `coverage`** | dòng | **±0,0%, KTC [0, 0] — trơ tuyệt đối.** ⚠ chỉ trơ với 100 dòng KIS; `ranked_hits` vẫn được `answer_qa.py`/`build_review_page.py` đọc từ đầu danh sách | ❌ |
| tín hiệu biên cảnh (k=1, β=0,25) | frame | TUNE +8,4% → TEST **−9,2%** (đổi dấu); chẩn đoán: hạng **xấu đi** ở mọi thước trong khi điểm TUNE tăng | ❌ |
| **VLM xếp lại nội-video** (bộ mới, cách hỏi định vị) | frame | +32,7% TUNE / +61,3% TEST — **có ăn**, nhưng tín hiệu **0 đồng** ăn +40,9% / +80,5% trên **đúng cùng cơ chế**; trộn vào cho +35,3% ⇒ **đóng góp biên ÂM** | ❌ |
| sáu quy tắc sắp lại **thứ tự 100 dòng** | dòng | bề-rộng-trước B=2/5/20: −2,0%/−7,5%/−11,4%; mật độ −0,1%; điểm-gần +0,5%; đảo ngược (đối chứng) −44,9% | ❌ |
| ↳ `gom_video` (khối liền theo video) | dòng | TEST bộ mới +5,5% (P=6,5%) **nhưng bộ cũ −11,5%, P(≤0)=100%** ⇒ ÂM | ❌ |
| mở **ngân sách dòng** giữa các video | dòng | ORACLE-VIDEO (biết trước video + dồn 100 dòng) chỉ **0,4114** / trần tuyệt đối 0,9848 ⇒ **vẫn mất 58% quãng đường** | ❌ |
| "σ thích nghi theo bề rộng khe keyframe" | dòng | ba phân bố bề rộng khe **gần trùng nhau** (trung vị 66/74/72 khung) ⇒ hình học khe không giải thích nghịch lý σ | ❌ |
| chùm keyframe làm **bảng ranh giới shot** | tiền xử lý | 15.200 chùm/873 video = một chùm mỗi 21,5 s; 18/873 video không có chùm nào; đo trên **66 cú cắt CÓ NHÃN**: tỷ lệ neo nằm trong chùm **thấp hơn nền** | ❌ |
| **NNN khử bias-hub (EMNLP 2024) — TOÀN TRỤC ①** | dòng | V1 nội-video: cổng tất định **51% ≤ 55%** (bias gần hằng số trong video). V2 liên-video: cổng QUA sát nút (57%>55%, n=17) → đo đầy đủ sắp-lại-100-dòng (khoá r+k·z(bias), một cảnh, tập dòng bất biến): TUNE đỉnh chỉ **+1,2% phẳng không đơn điệu** → TEST k=1 **−1,7%**, P(≤0)=77,7%. Hub CÓ THẬT nhưng không quy đổi thành điểm | ❌ |
| ↳ *nhưng* cosine SigLIP liền kề trên thang **tương đối trong video** | tiền xử lý | cú cắt thật ở phân vị **0,24**, đối chứng một cảnh **0,44** — **có tín hiệu, yếu**. Kết luận "cosine không đo được cú cắt" được chứng minh bằng một **ngưỡng tuyệt đối sai** (0,5 nằm dưới trung vị cặp ngẫu nhiên khác video) | ⏸ mở một nửa |

### Ba dòng ⚠60 đã được MỞ LẠI bằng số

| câu đã đóng cửa (bộ cũ) | sự thật trên bộ mới |
|---|---|
| "SigLIP đặt keyframe gần đáp án ở hạng nội-video **trung vị 1,0** (hạng-1 60%) — không còn chỗ cho bộ xếp lại nào" | một cảnh **2,0** (hạng-1 43%); **hai cảnh 6,0 (hạng-1 11%)**. Cửa mở lại — và lever ✅ ở trên đi qua nó |
| "hoán vị giữ-slot cho +0,6% (hoà) ⇒ hết chỗ xếp lại" | hoán vị **đối tượng** là **PHÉP ĐỒNG NHẤT** dưới `coverage` (tiên nghiệm là tổng trên TẬP): đổi dòng ở **0/66** câu, so với **66/66** dưới `hybrid`. Chạy lại thí nghiệm cũ hôm nay sẽ in ra 0,0% — **trông y hệt một kết quả "hoà"** |
| "54/60 câu đã có keyframe gần đáp án trong 400 ứng viên" | câu **hai cảnh** chỉ **35/65**. Đó là cửa mà lever cảnh B (đã ship) đi qua: pool 53% → 76% |

### Chẩn đoán — tách ba loại thất bại ở bán kính ±20 frame

| | trúng | mất do **ĐẶT DÒNG** | **không có ứng viên gần** | sai video |
|---|---|---|---|---|
| MỘT cảnh (n=66) | 40 | 8 | 8 | 10 |
| HAI cảnh (n=66) | 24 | **9** | **27** | 6 |

Phần mà **chỉnh tham số phân bổ** có quyền động tới chỉ là cột thứ hai: **9/66**
và **8/66**. Đó là lý do cả bảng "đóng cửa" ở trên toàn số âm.

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
> **⚠60 — TOÀN BỘ KHỐI DƯỚI ĐÂY ĐO TRÊN BỘ 60 CÂU CŨ, và hai câu kết luận của
> nó đã bị bác bằng số ngày 01/09/2026.** Đọc `docs/KE_HOACH_DINH_VI.md` §2.2
> trước khi trích bất cứ dòng nào. Tóm tắt: (i) "hạng nội-video trung vị 1,0
> (hạng-1 60%)" là đặc điểm riêng của bộ cũ — bộ mới cho 2,0 (một cảnh) và
> **6,0 / hạng-1 11%** (hai cảnh); (ii) phép hoán vị **giữ-slot** sinh ra con số
> +0,6% ấy là **PHÉP ĐỒNG NHẤT** dưới allocator `coverage` đang sản xuất (đổi
> dòng ở 0/66 câu) — nó không đo được điều nó tưởng đã đo.
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

> **⚠60 — bảng dưới đây đo trên bộ 60 câu cũ, và con số "+115% ở vị trí frame"
> đã bị ĐỊNH GIÁ LẠI BA LẦN trên bộ khớp phân bố** (`docs/KE_HOACH_DINH_VI.md` §3):
>
> * trần **theo pool** (thứ mà một tín hiệu xếp hạng nội-video với tới được) là
>   **+100,1%**, không phải +126% của oracle — 21% khoảng oracle (hai cảnh 27%)
>   nằm ở khâu **sinh ứng viên**, ngoài tầm với;
> * **thứ tự 100 dòng là một trục THỨ HAI, độc lập** (trần +74% một cảnh / +131%
>   hai cảnh), và với nhóm MỘT cảnh nó **lớn hơn** trục đặt-frame (deficit thứ tự
>   0,2040 so với deficit đặt-frame 0,1488);
> * **ORACLE-VIDEO** (biết trước video *và* dồn cả 100 dòng vào nó) chỉ đạt
>   **0,4114** trong khi trần tuyệt đối là 0,9848 ⇒ **mở ngân sách dòng KHÔNG
>   giúp**; tường nằm ở **chọn đúng Ô trong video**.
>
> Và một phần của khoảng cách ấy là **sản phẩm của thiết bị đo**: một dòng đặt
> đúng khung neo, hạng 1, chỉ trúng **0,341** vì bộ chấm bốc **đều** trên ô
> keyframe rộng trung vị 72 frame còn cửa sổ chấm rộng nhất chỉ ±20.

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

> **⚠60 — mục 4 này viết trước khi có bộ đo khớp phân bố.** Bảng xếp hạng việc
> tiếp theo còn hiệu lực nằm ở **`docs/KE_HOACH_DINH_VI.md` §4**, và nó khác hẳn:
> ưu tiên số một bây giờ là **đo nhân tử 0/1 của kênh Q&A** (27% đề thật, 36% số
> câu vòng 2, chưa ai đo một lần nào — và `answerer` đọc `hits`, không phải
> `cands`, nên lever cảnh B đã ship **không chạm** vào nó), rồi tới **sửa chính
> thiết bị đo**, rồi **sinh bộ đo TRAKE** (8,5% đề thật, 0 câu trong mọi bộ đo).

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

**Thêm ngày 01/09/2026, đo trên bộ khớp phân bố (§2a):**

- **Đừng** quét lại tham số `CoveragePlan`. Đã quét 216 tổ hợp; đỉnh TUNE +25,4%
  là **argmax trên ~6 câu hiệu dụng** (n Kish của nửa TUNE nhóm hai cảnh = 6,4),
  hoà trên TEST, **âm có ý nghĩa** trên bộ đối chứng, và **đảo chiều** khi đổi mô
  hình bốc của bộ chấm.
- **Đừng** chỉnh lại `hybrid` (−16,2%). Nó chỉ còn là cờ rút lui.
- **Đừng** thử quy tắc sắp xếp 100 dòng thứ bảy. Sáu quy tắc đã thử; cái duy nhất
  dương trên bộ mới (`gom_video`) **âm 100% xác suất** trên bộ cũ.
- **Đừng** chia lại ngân sách dòng giữa các video (ORACLE-VIDEO chỉ 0,4114).
- **Đừng** mua VLM cho xếp hạng nội-video: nó **thua** một tín hiệu 0 đồng trên
  đúng cùng cơ chế, và đóng góp biên khi trộn là **âm**.
- **Đừng** mua TransNetV2 lúc này (26–60 GB + 20–30 h GPU T4). *Ghi chú đính
  chính, giữ lại làm bài học:* bản nháp đầu đã chép từ **tóm tắt của công cụ tìm
  kiếm** hai con số ("F1 0,92 trên tin tức/TRECVID2001", "87,0% so với 65,5% của
  PySceneDetect") mà **paper gốc không có** — bảng thật là ClipShots 77,9 / BBC
  96,2 / RAI 93,9, và không có so sánh PySceneDetect nào. Sai sót đó suýt hạ chi
  phí ước tính đi gần 10 lần.
- **Đừng** suy từ "ưu tiên đỉnh trơ tuyệt đối" ra "xoá được `PEAK_WEIGHT`": nó
  chỉ trơ với 100 dòng KIS dưới `coverage`; `ranked_hits` vẫn được đọc từ **đầu**
  danh sách bởi `answer_qa.py` và `build_review_page.py`, và +2,2% cũ vẫn đúng
  cho `hybrid`.

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
