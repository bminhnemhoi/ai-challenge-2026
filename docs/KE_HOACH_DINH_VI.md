# Kế hoạch định vị — quyết định sau bốn lane và hai lượt phản biện

Chốt 01/09/2026. Đây là tài liệu **quyết định**, không phải bản tóm tắt. Mọi con
số dưới đây đều truy được về một script và một gốc hạt cụ thể.

> Không mã video, không đáp án trong tài liệu này (`docs/` lên GitHub công khai).

Nền chung của mọi phép đo mới: bộ **sạch 132 mục** của `data/ground_truth_moi.json`
(66 MỘT cảnh / 66 HAI cảnh, đã loại shard `c` lẫn trục), chấm qua
`make_submission.allocate_rows()` thật với allocator `coverage`, cửa sổ {6,10,20},
bootstrap **theo câu** 4000 lần.

---

## 0. Ba quyết định

| # | quyết định | căn cứ |
|---|---|---|
| **A** | **SHIP** lever *hoán vị điểm nội-video theo `sim(cảnh B)`* — 0 đồng, 0 GPU, 0 lần gọi LLM | §1 |
| **B** | **ĐÓNG** toàn bộ trục "chỉnh tham số phân bổ / sắp lại thứ tự dòng bằng quy tắc" — 12 cửa, đều có số | §2 |
| **C** | Việc tiếp theo **không phải** một lever định vị nữa, mà là **đo kênh Q&A** và **sửa chính thiết bị đo** | §4 |

Và một quyết định **không** làm, phải nói rõ vì nó là thứ dễ bị làm nhất:
**KHÔNG** cho `answerer` đọc khung đã hoán vị trong cùng lần ship với (A). Nó
là thay đổi hiển nhiên đúng về cơ chế, chưa đo một lần nào, và nằm đúng trên
nhân tử 0/1 của 27% số điểm. Đo trước — §4.1.

---

## 1. Cái ĂN: hoán vị điểm nội-video theo `sim(cảnh B)`

### 1.1 Nó là gì

Với câu **qua cổng hai cảnh** (cổng đã ship, `gan_nhan_hai_canh`): trong mỗi
video, **giữ nguyên đa tập điểm** của video đó, chỉ gán lại điểm nào thuộc khung
nào — theo độ tương đồng SigLIP với riêng **cảnh B**. Tổng khối lượng softmax
của mỗi video **không đổi theo xây dựng**, nên bề rộng phủ video không thể bị
phá; chỉ *hình dạng khối lượng bên trong video* dịch đi.

Vector `sim(cảnh B)` **đã được tính sẵn** trong
`make_submission.them_ung_vien_canh_b` và hiện đang bị vứt đi sau khi lấy top-M.
Lever này không thêm một phép nhân ma trận nào.

### 1.2 Số

| lượt đo | n | nền | chốt | chênh | KTC 95% theo câu | P(≤0) | tốt/xấu/không đổi |
|---|---|---|---|---|---|---|---|
| TUNE (lane vlm) | 33 | 0,1648 | — | **+40,9%** | — | — | — |
| TEST (lane vlm, đọc 1 lần sau khi sửa phép chia) | 33 | 0,1146 | 0,2069 | **+80,5%** | [+0,0337, +0,1604] | 0,0% | 10/1/22 |
| phản biện 1, **cả 66 câu**, hạt 666000 | 66 | 0,1390 | 0,2187 | **+57,4%** | [+0,0371, +0,1258] | — | 19/3/44 |
| **cổng trước-ship (mới, §1.3)**, hạt 667000 | 66 | 0,1385 | 0,2183 | **+57,6%** | [+0,0369, +0,1261] | 0,0% | 19/3/44 |

Độ bền khi bỏ các câu đóng góp nhiều nhất (cả 66 câu): bỏ 3 câu → **+38,2%**;
bỏ 5 câu → **+28,7%**. Hiệu ứng tập trung nhưng **không** biến mất khi cắt ngọn.

**Đối chứng artifact — cả bốn đều đi đúng chiều:**

| đối chứng (qua đúng cơ chế ấy) | kết quả |
|---|---|
| khoá **ngẫu nhiên** | −0,5% → −8,1% — **không bao giờ dương** |
| tín hiệu cảnh B **đảo dấu** | −32,0% → −36,7% |
| tín hiệu **cảnh A** thay cảnh B | −11,0% → −35,2% |
| VLM hỏi bằng cảnh B (tốn tiền) | +32,7% TUNE — **thua** tín hiệu 0 đồng |

**Mục tiêu định vị được xác minh riêng, TRƯỚC khi nhìn điểm:** với 61 câu hai
cảnh tách được ranh giới A→B, khung neo **trùng khít** khung cảnh-B đầu tiên ở
**39/61 = 64%**, ≤1 keyframe ở 70%, trung vị lệch **0**; neo nằm bên cảnh B ở
60/66 câu.

**Đường tham số phẳng:** mọi giá trị `w` từ 1 tới 100 cho điểm y hệt nhau tới
chữ số thứ tư. Không có đỉnh nhọn để trượt xuống — cùng chữ ký "cơ chế thật" mà
lever cảnh B đã dùng, ngược hẳn với lever ③ (đổi cấu hình thắng khi thêm dữ liệu).

**Bất biến, bằng `assert` chứ không bằng mắt:** `w = 0` ⇒ 100 dòng giống hệt nền
ở **132/132** mục; **66 câu MỘT cảnh** ra 100 dòng **giống hệt nền** ở mọi cấu
hình. Đã được kiểm lại độc lập **hai lần** (phản biện 1, và cổng §1.3).

### 1.3 Cổng trước-ship mới — lever này có phải một đánh đổi bề-rộng/độ-sâu trá hình không?

`scripts/cong_do_ben_mo_hinh_boc.py` (mới, 0 API, ~4 phút).

Đây là phép kiểm **bắt buộc** sau phát hiện lớn nhất của phản biện 1: bộ chấm
bốc khoảnh khắc thật **đều** trên ô keyframe chứa khung neo, và **trục sigma của
`CoveragePlan` ĐỔI CHIỀU khi đổi giả định bốc** (bốc đều → sigma 60 thắng; bốc
Gauss quanh neo → sigma 15 thắng). Nếu lever này cũng đảo dấu theo mô hình bốc
thì nó không phải phép chọn ô, nó chỉ là một đánh đổi trải-rộng/gom-sâu đổi tên.

Chạy **đúng cấu hình đã chốt từ trước** (không chọn lại gì), cả 66 câu hai cảnh,
hạt giống độc lập 667000, dưới bốn mô hình bốc:

| mô hình bốc khoảnh khắc thật | nền | chốt | chênh | KTC 95% theo câu | P(≤0) | tốt/xấu/không |
|---|---|---|---|---|---|---|
| **ĐỀU** trong ô (bộ chấm hiện tại) | 0,1385 | 0,2183 | **+57,6%** | [+0,0369, +0,1261] | 0,0% | 19/3/44 |
| TAM GIÁC, đỉnh ở khung neo | 0,1588 | 0,2473 | **+55,8%** | [+0,0409, +0,1428] | 0,0% | 19/3/44 |
| GAUSS sd=12 quanh neo | 0,1851 | 0,2788 | **+50,6%** | [+0,0433, +0,1514] | 0,0% | 19/3/44 |
| SAU NEO (đều trên `[neo, hi)`) | 0,1448 | 0,2357 | **+62,8%** | [+0,0436, +0,1401] | 0,0% | 20/2/44 |

Mô hình **SAU NEO** đáng chú ý riêng: với câu hai cảnh, khung neo *được định
nghĩa* là khung đầu tiên của cảnh B — tức đúng một cú cắt. Khoảnh khắc thật nằm
**sau** cú cắt, không thể nằm trước nó; mô hình ĐỀU hiện tại đặt gần một nửa khối
lượng xác suất vào **cảnh A**. Đó là mô hình sai rõ nhất cho nhóm này, và lever
ăn **nhiều hơn** ở mô hình đúng hơn.

**Kết luận cổng: ĐẠT.** Bốn mô hình bốc, cùng dấu, cùng độ lớn (+50,6% → +62,8%),
KTC tách khỏi 0 ở cả bốn, phân rã tất định gần như y hệt. Đây là chữ ký ngược hẳn
với trục sigma. Lever này là phép **chọn Ô**, độc lập với giả định của thiết bị đo.

### 1.4 Rủi ro — nói thẳng, có số

**(a) Cổng gán nhãn sai.** Trên bộ đo, văn bản cảnh B là bản phân rã của chính bộ
sinh — sạch gần như oracle. Trên đề THẬT nó do `gan_nhan_hai_canh` trích ra, và
**độ chính xác từng câu của cổng ấy chưa ai đo**. Lever này để văn bản cảnh B
quyết định **chỗ đặt dòng**, không chỉ thêm ứng viên, nên nó nhạy với cổng sai
hơn hẳn lever cảnh B đã ship.

Định lượng thô cái đánh đổi, từ chính bảng đối chứng: câu **đúng cổng** được
+57%; câu **sai cổng** đi qua một khoá gần như vô nghĩa, tức nằm giữa "khoá ngẫu
nhiên" (−0,5% → −8,1%) và "khoá cảnh A" (−11% → −35%). Hoà vốn ở tỷ lệ cổng đúng
khoảng **0,15 – 0,4**. Bằng chứng hiệu chuẩn duy nhất đang có: cổng bật
**28/55 = 51%** trên đề thật, trùng khít tỷ lệ cấu trúc 51% của đề BTC — nhưng
đó là một tỷ lệ **tổng**, không phải độ chính xác **từng câu**.

**(b) Phép kiểm chéo đã giết hai lever khác KHÔNG áp dụng được cho lever này.**
Bộ 60 câu cũ là tập độc lập đã bác `sigma 45` (−8,2%, P=99,8%) và `gom_video`
(−11,5%, P=100%). Đọc thẳng từ cache nhãn: cổng hai cảnh bật **0/60** trên bộ cũ
⇒ lever ra dòng giống hệt nền ở cả 60 câu ⇒ **phép kiểm là phép đồng nhất**.
Lever này miễn nhiễm với phép bác bỏ ấy **theo cấu tạo**, không phải vì đã vượt
qua nó. Đó là **một lớp bảo vệ ít hơn**, và phải ghi vào biên bản.

Mặt kia của cùng sự thật: hai lever kia đổi **mọi** câu, lever này chỉ đổi câu
qua cổng — hồ sơ rủi ro khác hẳn về bản chất, không chỉ về mức độ.

**(c) KTC bootstrap ở giao thức này chỉ chứng thực DẤU.** Phản biện 1 chỉ ra:
hiệu số theo câu là tất định khi đã cố định tập bốc, nên nhân cả vector với một
hằng số dương không đổi dấu bất kỳ phân vị nào ⇒ **KTC bất biến tỷ lệ**. Ngưỡng
thực tế là "≈4–5 câu cùng chiều trên 33". Vậy `P(≤0) = 0,0%` nghĩa là **dấu chắc**,
**không** phải "độ lớn +57% sẽ giữ được".

**(d) Hiệu ứng tập trung.** 3/33 câu chiếm 52% mức tăng ở lượt TEST. Trên cả 66
câu thì bỏ 3 câu vẫn còn +38,2% — bền hơn, nhưng vẫn là hiệu ứng của ~19 câu.

**(e) `w` bão hoà** nghĩa là tín hiệu **thay hẳn** thứ tự nội-video của SigLIP
trong các khung được chấm. Đây là một phép thay thế mạnh, không phải hiệu chỉnh
nhẹ. Rủi ro tối đa bị chặn ở nhóm qua cổng.

**(f) TEST của bộ 132 mục đã bị đọc bởi ít nhất bốn lane với bốn phép chia khác
nhau**, và riêng lane vlm đã đọc hai lần trên hai phép chia. Con số +80,5% phải
đọc với hiểu biết đó. Con số nên trích là **+57,6%** (cả 66 câu, hạt độc lập,
cấu hình cố định từ trước) chứ không phải +80,5%.

### 1.5 Vì sao vẫn SHIP

Bốn điều phải đúng cùng lúc, và cả bốn đều đúng:

1. **Cơ chế được chứng minh riêng, không qua thống kê.** Mục tiêu ("khung đầu
   tiên của cảnh B") được xác nhận bằng phép đếm 64% *trước khi* nhìn điểm; ba
   đối chứng (ngẫu nhiên / đảo dấu / cảnh A) đi đúng chiều.
2. **Độc lập với thiết bị đo** — bốn mô hình bốc, §1.3. Đây là thứ mà `sigma 45`
   và `gom_video` **không** có, và là lý do chúng bị đóng còn cái này thì không.
3. **Đường tham số phẳng** (w = 1…100 cho số y hệt).
4. **Rủi ro chặn cứng:** 66 câu không qua cổng ra 100 dòng **giống hệt nền**
   (assert, kiểm lại độc lập hai lần); bề rộng phủ video **bất biến theo xây
   dựng** — đúng cái đã gây artifact −35% của phép gom-khối năm ngoái.

Đây là hiệu ứng **dương lớn nhất, bền nhất, rẻ nhất** mà dự án đo được từ khi có
bộ đo khớp phân bố. Ba lane khác đều trả về ÂM. Không ship nó thì bốn lane vừa
rồi không sinh ra một điểm nào.

### 1.6 Các bước tích hợp — chính xác

Lane này chỉ được tạo file mới, nên đây là đặc tả cho người có quyền sửa
`scripts/make_submission.py`.

**Bước 1 — trả `sims` ra khỏi `them_ung_vien_canh_b`.**
Hàm đã có `sims = engine.query_similarities(b_vi, rec.get("canh_B_en") or None)`
và đang vứt đi. Đổi chữ ký trả về thành `(cands, ghi_chu, simsB)` với
`simsB = None` ở mọi nhánh thoát sớm (`m <= 0`, không có nhãn, không qua cổng,
`b_vi` rỗng, exception). Hai chỗ gọi (`build_kis_rows`, `build_qa_rows`) nhận
thêm biến thứ ba. **Không đổi gì khác trong hàm này.**

**Bước 2 — bản đồ hàng metadata.** Dựng một lần cho cả vòng (không phải mỗi câu):
`hang_of[(video_id, int(frame_idx))] = row` từ `engine.video_id` / `engine.frame_idx`.

**Bước 3 — hàm mới `hoan_vi_theo_canh_b(cands, simsB, hang_of)`**, đặt trong
`make_submission.py`, chạy **sau** `them_ung_vien_canh_b` và **trước**
`allocate_rows`. Bốn bước, phải giống **từng chi tiết** với cấu hình đã đo — đây
không phải chỗ để cải tiến:

1. **Chọn khung được chấm** — `top-3 video × top-12 khung`. Ba video đầu theo
   **thứ tự xuất hiện lần đầu** trong danh sách ứng viên (tức theo hạng), mỗi
   video lấy 12 ứng viên **điểm cao nhất trong video đó**.
   (Tham chiếu: `do_vlm_noi_video_moi.chon_khung_de_cham(cands, 3, 12)`.)
2. **Chuẩn hoá min-max TRONG TỪNG VIDEO** trên `simsB[hang_of[(v, f)]]`. Video
   có min = max ⇒ tất cả về 0.
3. **Suy ra điểm định vị** `loc(f) = B(f) · (1 − α · B(khung được chấm liền
   trước trong cùng video))`, **α = 0,5**, các khung xếp theo `frame_idx` tăng
   dần. (Tham chiếu: `do_vlm_noi_video_moi.suy_ra_loc`.)
4. **Khoá xếp** `key[i] = điểm(i) + w · loc(v, f)` với **w = 1,0**, chỉ cho
   ứng viên đã được chấm. **Khoá phải đánh theo CHỈ SỐ ứng viên `i`, tuyệt đối
   không theo cặp `(video, khung)`** — pool sản xuất có **khung trùng** (cùng
   keyframe, hai điểm khác nhau; 22/132 câu, tới 10 cặp một câu) và đánh khoá
   theo cặp làm **vỡ** bất biến `w=0`. Nếu tổng số khoá < 2 thì bỏ qua câu.
5. **Hoán vị điểm** trong từng video: lấy các ứng viên có khoá của video đó, sắp
   điểm của chính chúng giảm dần, gán lại theo thứ tự `(-key, i)`. Video có < 2
   ứng viên có khoá thì bỏ qua. Ứng viên **không** được chấm giữ nguyên cặp
   (khung, điểm). (Tham chiếu: `do_vlm_noi_video_moi.hoan_vi_diem`.)

**Bước 4 — cờ rút lui.** `--hoan-vi-canh-b {0,1}`, mặc định **1**. `0` ⇒ bỏ qua
bước 3 hoàn toàn ⇒ đường sản xuất hôm nay, từng bit. Cờ này **độc lập** với
`--canh-b`; nhưng `--canh-b 0` cũng tự tắt lever (không có `simsB`).

**Bước 5 — cổng hồi quy trước merge, cả ba đều phải xanh:**

| cổng | tiêu chí |
|---|---|
| **G1 bất biến** | 66 câu MỘT cảnh của bộ 132 ra 100 dòng **giống hệt** nền — so từng dòng, không so điểm |
| **G2 tái lập** | `--hoan-vi-canh-b 1` trên 66 câu hai cảnh dựng lại đúng 0,2183 (hạt 667000, mô hình bốc ĐỀU) tới 4 chữ số |
| **G3 rút lui** | `--hoan-vi-canh-b 0` cho zip **giống hệt** bản hiện tại trên toàn bộ một vòng đề thật (hash từng dòng) |

**Bước 6 — kiểm bằng mắt trên ĐỀ THẬT, bắt buộc, trước ngày thi.** Chạy diff kiểu
`scripts/diff_canh_b_de_that.py` cho lever này (0 lần gọi LLM, nhãn đã cache) và
**soi tay mọi câu bị đổi video hoặc frame ở DÒNG 1**. Với lever thêm-ứng-viên,
con số đó là 7/28 câu; lever này đổi chỗ đặt dòng nên số câu bị đổi dòng 1 sẽ
**cao hơn**, và mỗi câu trong đó đáng 1,0 điểm R@1 — số hạng đắt nhất của công
thức. Danh sách ngắn, và đây là chỗ duy nhất mà mắt người thay được phép đo còn
thiếu ở §1.4(a). Lệnh rút lui nếu thấy sai: `--hoan-vi-canh-b 0`.

---

## 2. Bảng CỬA ĐÃ ĐÓNG — đừng thử lại

Ký hiệu độ tin:
**🟩 bộ MỚI** = đóng trên bộ 132 mục khớp phân bố (TUNE/TEST phân tầng, bootstrap
theo câu) — đáng tin.
**🟨 bộ CŨ** = chỉ đóng trên bộ 60 câu cũ, bộ đã được chứng minh là **thổi phồng
năng lực ~2 lần** và có **0/60** câu hai cảnh — **có thể không còn đúng**.

### 2.1 Đóng trên bộ MỚI 🟩

| cửa | số | ghi chú |
|---|---|---|
| **quét lại tham số `CoveragePlan`** (216 tổ hợp) | TUNE đỉnh +25,4% (hai cảnh +90,6%) → **TEST +1,5%, P(≤0)=37,3% = HOÀ**; nhóm hai cảnh TEST **−3,8%** | kiểm chéo bộ cũ **−8,2%, P=99,8% ÂM**; và trục sigma **đảo chiều** khi đổi mô hình bốc ⇒ nó chưa bao giờ là cải tiến chung |
| **chỉnh lại allocator `hybrid`** (16 tổ hợp) | tất cả ≤ nền coverage; TEST **−16,2%**, P(≤0)=99,8% | `hybrid` chỉ còn là đường rút lui |
| **làm mượt SigLIP theo thời gian** | TEST −1,5%; hai cảnh +5,4% nhưng P(≤0)=17,7% | **cơ chế có thật** (pool 35→41, hạng toàn video 15,0→11,0) nhưng trả giá ở nhóm một cảnh (pool 58→31) và không quy đổi thành điểm |
| **chuẩn hoá theo video** (trừ TB, λ=0,25) | TEST −6,9%; hai cảnh −11,6%, P=99,7% | |
| **ưu tiên đỉnh cục bộ** dưới `coverage` | ±0,0%, KTC [0, 0] — **trơ tuyệt đối** | ⚠ chỉ trơ với **100 dòng KIS**; `ranked_hits` vẫn được `answer_qa.py` và `build_review_page.py` đọc từ đầu danh sách, và +2,2% cũ vẫn đúng cho `hybrid` |
| **tín hiệu biên cảnh** (k=1, β=0,25) | TUNE +8,4% → TEST **−9,2%** (đổi dấu) | chẩn đoán bắt tại trận: điểm TUNE tăng mà **hạng xấu đi ở mọi thước** (pool 58→52 và 35→31) ⇒ phần điểm đó không phải định vị |
| **VLM xếp lại nội-video** | +32,7% TUNE / +61,3% TEST — **có ăn**, nhưng thua tín hiệu 0 đồng (+40,9% / +80,5%) trên **đúng cùng cơ chế**; trộn vào cho +35,3% < +40,9% ⇒ **đóng góp biên ÂM** | đóng vì **bị trội hơn**, không phải vì vô dụng. 213 lần gọi, một model cháy quota trong lúc đo |
| **sáu quy tắc sắp lại thứ tự 100 dòng** | bề-rộng-trước B=2/5/20: −2,0% / −7,5% / −11,4% (âm đơn điệu); mật độ −0,1%; điểm-gần +0,5%; đảo ngược (đối chứng) −44,9% | |
| ↳ **`gom_video`** (khối liền theo video) | TUNE +3,1% → TEST +5,5%, P(≤0)=6,5% | **kiểm chéo bộ cũ −11,5%, P(≤0)=100%** ⇒ **ÂM**. Cùng chữ ký đánh đổi như trục sigma |
| **mở ngân sách dòng giữa các video** | ORACLE-VIDEO (biết trước video + dồn cả 100 dòng) chỉ đạt **0,4114** trong khi trần tuyệt đối là 0,9848 ⇒ **vẫn mất 58% quãng đường** | tường nằm ở **chọn đúng Ô trong video**, không ở việc chia bao nhiêu dòng cho ai |
| **"sigma thích nghi theo bề rộng khe keyframe"** | ba phân bố bề rộng khe gần trùng nhau (trung vị 66 / 74 / 72 khung) ⇒ hình học khe **không** giải thích nghịch lý sigma | đóng **trước khi** ai kịp bỏ tiền vào |
| **chùm keyframe làm bảng ranh giới shot** | 15.200 chùm / 873 video = một chùm mỗi 21,5 s; **18/873 video không có chùm nào**; và phép đo có nhãn cho thấy tỷ lệ neo-tại-cú-cắt-thật nằm trong chùm **thấp hơn nền** | kết luận này giờ có bằng chứng **trực tiếp trên 66 cú cắt có nhãn**, không còn dựa vào số văn liệu |

### 2.2 Chỉ đóng trên bộ CŨ 🟨 — coi như CHƯA BIẾT

Mọi dòng trong bảng §2 của `KIEN_TRUC_VA_HUONG_CAI_THIEN.md` **không** ghi
"bộ đo mới" đều thuộc nhóm này. Nguy hiểm nhất là ba dòng sau, vì chúng đang
**đóng cửa cho người khác**:

1. **"SigLIP thuần đã đặt keyframe gần đáp án ở hạng nội-video trung vị 1,0
   (hạng-1: 60%) — không còn chỗ cho bộ xếp lại nào."** Đây là câu đã đóng cửa
   xếp-lại-nội-video suốt nhiều tuần. Nó là **đặc điểm của bộ 60 câu cũ**, nơi
   câu hỏi được viết bằng cách nhìn đúng cái keyframe đã đánh chỉ mục. Trên bộ
   mới: một cảnh trung vị **2,0** (hạng-1 43%), hai cảnh trung vị **6,0**
   (hạng-1 **11%**). Cửa mở lại, và §1 là thứ đi qua nó.
2. **"Phép hoán vị giữ-slot cho +0,6% (hoà) ⇒ không còn chỗ cho bộ xếp lại."**
   Phép hoán vị ấy đổi **đối tượng** ứng viên giữa các vị trí. Tiên nghiệm của
   `coverage` là **tổng trên TẬP** ứng viên ⇒ đổi thứ tự không đổi một bit nào.
   Đo được: `coverage` đổi dòng ở **0/66** câu, `hybrid` (bộ mà phép đo cũ dùng)
   đổi ở **66/66**. **Chạy lại thí nghiệm ấy trên đường sản xuất hôm nay sẽ in ra
   đúng 0,0% ở mọi cấu hình — và con số ấy trông y hệt một kết quả "hoà".**
3. **"Tiêm keyframe nội-video vào tiên nghiệm: TEST +0,2% (hoà), vì 54/60 câu đã
   có keyframe gần đáp án trong 400 ứng viên."** Tiền đề "đã có sẵn" đúng trên bộ
   cũ. Trên bộ mới, câu hai cảnh chỉ có **35/65**. Đó chính là cửa mà lever cảnh
   B (đã ship) đi qua.

Ba dòng này **không bị xoá** — chúng là bộ đối chứng lịch sử. Chúng bị **đánh
dấu** bằng ⚠60 trong bảng tín hiệu.

---

## 3. Trần thật sự nằm ở đâu — con số +126% đã bị định giá lại ba lần

| cách đọc trần | bộ sạch | MỘT cảnh | HAI cảnh | đọc thế nào |
|---|---|---|---|---|
| **oracle đặt dòng** (giữ video + thứ hạng, đặt thẳng quanh đáp án THẬT) | +126,4% | +59,4% | +306,4% | **thổi phồng** — dùng đáp án để chọn tâm thang |
| **trần theo POOL** (đặt quanh ứng viên GẦN ĐÁP ÁN NHẤT trong 400) | **+100,1%** | +53,8% | **+223,1%** | trần **thực tế** của mọi tín hiệu xếp hạng nội-video; chiếm 79% khoảng oracle (hai cảnh 73%) |
| **oracle thứ tự dòng** (đưa dòng đúng về hạng 1, tập dòng không đổi) | +89,5% | **+74,0%** | +130,6% | **trục KHÁC**, độc lập với đặt-frame |
| **ORACLE-VIDEO** (biết trước video, dồn cả 100 dòng) | +117,0% | +100,4% | +161,0% | 0,4114 / 0,9848 ⇒ **vẫn mất 58% quãng đường** |
| một dòng đặt **đúng khung neo**, hạng 1 | trúng **0,341** | | | phải có thang ±50 frame mới lên 0,944 |

Bốn cách đọc lại, theo thứ tự tầm quan trọng:

**(1) 21% khoảng oracle (hai cảnh: 27%) nằm NGOÀI tầm với** của mọi thứ bám vào
pool 400 ứng viên — nó thuộc khâu **sinh ứng viên**, không phải xếp hạng.

**(2) Trục thứ tự dòng là trục thứ hai, và với nhóm MỘT cảnh nó LỚN HƠN trục đặt
frame.** Phân bố hạng của dòng đúng đầu tiên (phép đếm tất định, con số đáng tin
nhất trong cả hai lượt phản biện):

| nhóm | hạng 1 | 2–5 | 6–20 | 21–50 | 51–100 | không có dòng đúng |
|---|---|---|---|---|---|---|
| MỘT cảnh (n=66) | 5,4% | 9,7% | 14,1% | 10,9% | 7,8% | 52,0% |
| HAI cảnh (n=66) | 0,7% | 3,2% | 3,4% | 8,9% | 7,8% | 76,1% |

Nhóm MỘT cảnh **đã có** dòng đúng ở 48% số câu, nhưng **32,9 điểm phần trăm**
nằm ở hạng ≥6. Deficit thuần do THỨ TỰ = **0,2040**, *lớn hơn* toàn bộ deficit
đặt-frame theo pool của chính nhóm này (0,4255 − 0,2767 = **0,1488**). Tức
"+59% của nhóm một cảnh" **phần lớn không phải bài toán định vị**.

**(3) Một phần của "61/132 câu điểm 0" là sản phẩm của THIẾT BỊ ĐO.** Một dòng
đặt đúng khung neo, hạng 1, chỉ trúng **0,341** vì bộ chấm bốc **đều** trên ô
keyframe rộng trung vị 72 frame trong khi cửa sổ chấm rộng nhất chỉ ±20.
56/66 câu hai cảnh có ô rộng > 40 frame. Bộ chấm hiện tại **thưởng RẢI RỘNG chứ
không thưởng ĐẶT ĐÚNG** — và với câu hai cảnh, nơi khung neo *là* một cú cắt,
mô hình bốc đều còn đặt nửa khối lượng xác suất vào cảnh A.

**(4) Bù lại: sai số của câu HAI cảnh là sai số MỘT Ô keyframe, không phải sai số
cả video.** Khoảng cách từ dòng gần nhất tới neo: MỘT cảnh 2 frame, HAI cảnh 56
frame ≈ 2,2 s ≈ đúng một khe keyframe (khe trung vị 2,16 s). **51/65** câu hai
cảnh đã có ứng viên cách đáp án ≤1 ô. Và ta **không thiếu dòng**: video đúng
nhận trung vị **9** dòng (không phải 21,5 — đó là số *trung bình*, phân bố lệch
mạnh), trong khi phủ trọn khe trung vị cần 4,8 dòng, khe p90 cần 9,3 dòng.
Ta đặt dòng **sai ô** (5–6 ô trên ~250 ô của video), không đặt thiếu dòng.

**Tách ba loại thất bại ở bán kính ±20 frame:**

| | trúng | mất do **ĐẶT DÒNG** | **không có ứng viên gần** | sai video |
|---|---|---|---|---|
| MỘT cảnh (n=66) | 40 | 8 | 8 | 10 |
| HAI cảnh (n=66) | 24 | **9** | **27** | 6 |

Phần mà **chỉnh tham số phân bổ** có quyền động tới chỉ là cột thứ hai:
**9/66** (hai cảnh) và **8/66** (một cảnh). Đó là lý do §2.1 toàn số âm, và nó
đã đúng như dự báo trước khi các lane chạy.

---

## 4. Xếp hạng việc tiếp theo

Tiêu chí: (tác động vào trần) × (khả thi) × (độ tin của bằng chứng).

### 4.1 — Kênh Q&A: đo cái nhân tử 0/1 chưa ai đo. **Ưu tiên số một.**

**Vì sao đứng đầu.** `calculate_vqa_r_score` đòi **cả ba**: đúng video, frame
trong khoảng, **và đáp án khớp**. Nên **điểm Q&A = điểm định vị × 1[đáp án đúng]**.
Mọi con số của bốn lane là **thừa số thứ nhất**; thừa số thứ hai chưa được đo một
lần nào, và nó gánh **27% đề thật / 36% số câu vòng 2**.

Ba sự thật cấu trúc, đọc thẳng từ mã:

- `build_qa_rows` gọi `answer = answerer(hits[:5], ...)` — **`hits`, không phải
  `cands`**. Lever cảnh B đã ship sửa `cands`, nên nó **không chạm** vào đường
  sinh đáp án. Câu hai cảnh vẫn được trả lời bằng khung **cảnh A**.
- `_make_answerer` trả `Counter(votes).most_common(1)[0][0]` — biểu quyết đa số
  **chủ động loại bỏ** phiếu của khung đúng khi 4/5 khung là cảnh A.
- Đếm tất định trên 5 khung VLM thực sự được nhìn: câu HAI cảnh **62/66 câu có
  0/5 khung đúng ô**; trung bình chỉ **1,5/5** khung nằm trên đúng video. **Ngay
  cả oracle video cũng không cứu** (56/66 vẫn 0 khung đúng ô) — vì ứng viên trong
  video đúng cũng là ứng viên cảnh A.

Câu cuối là mấu chốt: **thứ cần cho đường Q&A đúng chính là lever §1**, và nó có
một kênh thứ hai chưa ai tính vào giá trị của nó.

**Giao thức đo (bắt buộc, theo đúng kỷ luật đã dùng):**

1. **Làm trước, $0,05:** 8 mục Q&A **đề thật đã người kiểm chứng đáp án** trong
   `ground_truth_de_that.json` × 5 khung = 40 lần gọi. Không chốt được gì với
   n=8, nhưng **0/8 hoặc 1/8 đúng** là một phép đếm đủ để đổi thứ tự ưu tiên của
   cả đội.
2. **Đo chính, $1–3:** 4 tập 5 khung (`hits[:5]` hiện tại / top-1 của 5 video đầu
   / top-5 trong video #1 / **5 khung sau khi hoán vị theo cảnh B**) × 132 câu
   × 5 lần gọi = 2.640 lần gọi ảnh, cache theo (model, khung, câu hỏi).
3. Đại lượng chốt: `P(đáp án đúng)` theo `_default_answer_match`. Chia TUNE/TEST
   **phân tầng theo `co_2_canh`**, bootstrap **theo câu**, báo cáo **riêng nhóm
   HAI cảnh**. `assert` câu KIS ra 100 dòng giống hệt nền.
4. **Cấm** ship "cho answerer đọc khung đã hoán vị" trước khi bước 2 xanh, dù nó
   hiển nhiên đúng về cơ chế. Đúng cái bẫy "hiển nhiên đúng" đã sinh ra ba kết
   luận âm ở §2.1.

### 4.2 — Sửa chính thiết bị đo. Hai việc, một mục đích.

**(a) Mô hình bốc.** Bộ chấm bốc **đều** trên ô keyframe. Với câu hai cảnh, khung
neo *được định nghĩa* là khung đầu tiên của cảnh B — một cú cắt — nên khoảnh khắc
thật **không thể** nằm trước nó, mà mô hình hiện tại đặt gần nửa khối lượng xác
suất ở đó. Giả định này **quyết định dấu** của trục sigma (§2.1) và định giá phần
lớn "khoảng cách còn thiếu" (§3.3). Sửa: thêm mô hình bốc **SAU_NEO** cho nhóm
hai cảnh và báo cáo song song hai mô hình ở mọi phép đo sau này. Chi phí: vài
giờ, **0 API**. Mẫu sẵn có: `scripts/cong_do_ben_mo_hinh_boc.py`.

**(b) Sinh thêm ground truth.** n = 66 mỗi nhóm là NHỎ, và **n hiệu dụng (Kish)
của nửa TUNE nhóm hai cảnh chỉ là 6,4** (24/33 câu điểm 0; 5 câu cao nhất chiếm
82% khối lượng). Bộ đo này đủ sức bác một hiệu ứng +25%, **không** đủ sức bác
một hiệu ứng +5%. Tệ hơn: **nửa TEST của bộ 132 mục đã bị đọc bởi ít nhất bốn
lane với bốn phép chia khác nhau** — nó không còn là một nửa TEST sạch cho lever
thứ năm. Máy móc đã có (`sinh_gt_doan_video.py` + `kiem_neo_don_anh.py`
một-ảnh-một-request), giá ~$1–2 một đợt.

**Đọc thẳng:** cho tới khi (b) xong, **mọi kết luận "hoà" ở §2.1 phải đọc là
"không chứng minh được", KHÔNG phải "đã chứng minh là không có"**.

### 4.3 — TRAKE: sinh ~20 mục ground truth.

**8,5% đề thật (2/25 câu vòng 2), 0 câu trong mọi bộ đo.** Cộng với Q&A thì
**44% số câu vòng 2 nằm ngoài mọi phép đo của bốn lane.**

Ba sự thật cấu trúc: `build_trake_rows` dùng `align_sequence(..., top_k=1)` rồi
`results[0]` ⇒ **cả 100 dòng trên đúng một video**, video sai ⇒ 0 điểm, không có
lưới an toàn. Điểm TRAKE **có điểm từng phần** (`matched/N`) ⇒ luật phân bổ khác
hẳn KIS. Và `src/core/submission.py::reserve_tail_rows` **đã viết, đã có test**
(`tests/test_reserve_tail.py`), **không được gọi ở đâu** — đúng là cơ chế cần cho
một cửa lui (dòng 51–100 chỉ đáng trọng số 0,2).

**Nhưng đây là lập luận, không phải phép đo:** không ai biết `P(video 1 đúng)`,
mà toàn bộ giá trị của cửa lui nằm ở con số đó. **Không ship `reserve_tail_rows`
trước khi có bộ đo.** Sinh ~20 mục TRAKE bằng chính bộ máy đã có (một mục TRAKE =
một đoạn liên tiếp + N mốc sự kiện, xác minh từng mốc bằng một-ảnh-một-request).
Giá ~$1–2.

### 4.4 — Tín hiệu xếp hạng DÒNG (không phải quy tắc sắp xếp).

Trần: **+74% (một cảnh) / +131% (hai cảnh)**, độc lập với trục đặt-frame, và với
nhóm một cảnh nó là trục **lớn hơn** (§3.2). Rủi ro chặn cứng: hoán vị không đổi
tập dòng ⇒ R@100 bất biến.

**Sáu quy tắc sắp xếp rẻ đã thất bại** (§2.1) — đừng thử quy tắc thứ bảy. Thứ
cần là một **tín hiệu chấm điểm từng dòng**. Ứng viên duy nhất đang có sẵn:
chính điểm `loc` của lever §1, dùng lại cho **thứ tự dòng** thay vì chỉ cho chỗ
đặt dòng. Với nhóm hai cảnh đây gần như miễn phí sau khi (A) ship. Với nhóm MỘT
cảnh **chưa có tín hiệu nào** — xem 4.5.

### 4.5 — Cắt cảnh: cửa mới chỉ đóng một nửa, và nửa còn lại là 0 đồng.

Lane paper kết luận ÂM cho hai đường tắt. Phản biện 1 đo lại trên **66 cú cắt CÓ
NHÃN** (với câu hai cảnh, khung neo *là* một cú cắt; nhóm một cảnh làm đối chứng)
và kết quả **tách đôi**:

- **Kết luận (1) — chùm keyframe không đánh dấu cú cắt — ĐÚNG**, và giờ có bằng
  chứng trực tiếp thay vì suy từ số văn liệu. 🟩 **Đóng hẳn.**
- **Kết luận (2) — "cosine SigLIP liền kề không đo được cú cắt" — CHƯA ĐƯỢC CHỨNG
  MINH.** Nó được chứng minh bằng ngưỡng **tuyệt đối** 0,5, mà ngưỡng đó nằm
  **dưới trung vị của cặp ngẫu nhiên khác video** — tức đòi hai khung cùng bản
  tin phải khác nhau hơn hai khung của hai video khác nhau. Trên thang **tương
  đối trong từng video**, cú cắt thật nằm ở phân vị **0,24**, đối chứng một cảnh
  nằm ở **0,44**. **Có tín hiệu**, nhưng yếu.

**Khuyến nghị:** **KHÔNG** mua TransNetV2 lúc này (26–60 GB tải + 20–30 h GPU T4;
và lưu ý ghi chú đính chính: hai con số F1 từng chép nhầm từ tóm tắt công cụ tìm
kiếm, bảng thật là ClipShots 77,9 / BBC 96,2 / RAI 93,9, không có so sánh
PySceneDetect nào — sai sót ấy suýt hạ chi phí ước tính đi gần 10 lần). Thay vào
đó tiêu **0 đồng** thử điểm cắt tương đối-trong-video làm đặc trưng xếp hạng
nội-video cho nhóm **MỘT cảnh** — nhóm duy nhất chưa có tín hiệu nào và có
deficit thứ tự lớn nhất (0,2040).

### 4.6 — ĐỪNG làm

- **Đừng** quét lại tham số `CoveragePlan` (nhiệt / sigma / nửa cửa sổ / lưới).
  Đã quét 216 tổ hợp; đỉnh TUNE +25,4% là **argmax trên ~6 câu hiệu dụng**, hoà
  trên TEST, âm có ý nghĩa trên bộ đối chứng, và **đảo chiều** khi đổi mô hình bốc.
- **Đừng** chỉnh lại `hybrid` (−16,2%). Nó chỉ còn là cờ rút lui.
- **Đừng** thử quy tắc sắp xếp 100 dòng thứ bảy (§2.1, sáu quy tắc, `gom_video`
  âm trên bộ đối chứng).
- **Đừng** chia lại ngân sách dòng giữa các video (ORACLE-VIDEO chỉ 0,4114).
- **Đừng** mua VLM cho xếp hạng nội-video: thua tín hiệu 0 đồng trên **đúng cùng
  cơ chế**, và đóng góp biên **âm** khi trộn.
- **Đừng** suy từ "ưu tiên đỉnh trơ" ra "xoá được `PEAK_WEIGHT`": nó chỉ trơ với
  100 dòng KIS dưới `coverage`; `ranked_hits` vẫn được đọc từ đầu danh sách bởi
  `answer_qa.py` và `build_review_page.py`.

---

## 5. Điều chưa biết — trung thực

1. **Độ chính xác từng câu của cổng `gan_nhan_hai_canh` trên đề THẬT.** Đây là
   bất định lớn nhất của quyết định (A). Chỉ biết cổng bật 28/55 = 51%, trùng tỷ
   lệ cấu trúc của đề — một con số **tổng**, không phải độ chính xác **từng câu**.
   Cách duy nhất hiện có: mắt người, trên danh sách ngắn các câu bị đổi dòng 1
   (§1.6 bước 6).
2. **Đề thật giống bộ đo nào.** Hai trục **độc lập nhau** (bề rộng không gian;
   thứ tự dòng) giờ cho **cùng một chữ ký đảo dấu** giữa bộ mới và bộ cũ. Đây
   không còn là sự cố về sigma — **hai bộ đo đang bất đồng về HÌNH DẠNG bài
   toán**. Bằng chứng gián tiếp duy nhất: điểm vòng 2 = 10,0/30 = **0,333/câu**
   (đã có công soát tay), nằm **giữa** bộ cũ (0,400 tự động) và bộ mới (0,190).
   **Phép đo rẻ chưa ai làm:** chấm đường sản xuất trên 59 mục
   `ground_truth_de_that.json` và so **HÌNH DẠNG** (phân bố hạng dòng đúng đầu
   tiên; hạng nội-video của keyframe đáp án; tỷ lệ có keyframe đáp án trong pool
   400) — đây là **phép đếm**, nên n=15 người-kiểm-chứng vẫn nói được điều gì đó,
   trong khi so **điểm** ở n=15 thì không. ~15 phút, cần `KISEngine`, **0 API**.
3. **Câu của bộ đo do MÁY SINH.** Cấu trúc hai cảnh dứt khoát hơn đề người viết
   ⇒ mọi hiệu ứng gắn với cảnh B (kể cả lever §1) có thể bị **thổi phồng**. Bằng
   chứng gián tiếp duy nhất: điểm khớp neo đồng đều giữa hai nhóm (trung vị 94
   vs 92, **0/24** câu dưới 70 điểm ở cả hai nhóm).
4. **Nhóm MỘT cảnh chưa có lever nào.** Trần của cơ chế §1 cho nhóm này là
   +36,0% — chưa thử, và cảnh B không tồn tại ở đó theo định nghĩa. Deficit lớn
   nhất của nhóm này là **thứ tự dòng** (0,2040), chưa có tín hiệu.
5. **Bước từ "62/66 câu VLM không được nhìn khung đúng" tới "đáp án sai ở 62/66
   câu" CHƯA được đo.** VLM có thể trả lời đúng nhờ ngữ cảnh chung của bản tin,
   hoặc đáp án có thể là logo kênh (bộ đo đã đánh dấu vài ca như vậy). Đó là phép
   đo phải chạy (§4.1), không phải kết luận đã có.
6. **`P(video 1 đúng)` cho TRAKE.** Không có một mục TRAKE nào tồn tại ⇒ giá trị
   của `reserve_tail_rows` là **không biết**, không phải "đáng 0,2 × matched/N".
7. **Tỷ lệ câu rơi vào Ô RỘNG.** Khe keyframe rất không đều (p25 1,04 s;
   p75 4,40 s; max 8,0 s). Ô 8 giây cần ~24 dòng để phủ kín bằng thang bước 10,
   nhiều hơn trung vị 9 dòng đang có. **Chưa ai đếm bao nhiêu câu rơi vào đó** —
   con số này đặt trần cho cả lever §1 lẫn mọi bộ dò cắt cảnh.
8. **Bộ 60 câu cũ có thể sai chứ không chỉ lạc quan.** Nó vẫn là bộ đối chứng độc
   lập duy nhất, và nó đã bác hai lever. Nhưng nó có 0/60 câu hai cảnh, nên với
   lever §1 nó là **phép đồng nhất** — không xác nhận, cũng không bác bỏ.

---

## 6. Tái lập

```bash
# cổng trước-ship: bốn mô hình bốc, cả 66 câu hai cảnh, 0 API (~4 phút)
python -u scripts/cong_do_ben_mo_hinh_boc.py

# cấu hình đã chốt trên cả 66 câu, hạt độc lập 666000
python -u scripts/phan_bien_ben_vlm.py

# bảng TUNE + đối chứng của lane vlm, KHÔNG tiêu lần đọc TEST nào
python -u scripts/do_vlm_noi_video_moi.py --giai-doan vlm --cach-hoi chi-siglip \
    --videos 3 --frames 12 --khong-doc-test

# các cửa đã đóng
python -u scripts/do_phan_bo_sau.py          # tham số phân bổ
python -u scripts/do_tin_hieu_noi_video.py   # 6 họ tín hiệu nội-video
python -u scripts/phan_bien_thieu_sot.py     # thứ tự dòng, oracle-video, Q&A
python -u scripts/phan_bien_mo_hinh_boc.py   # trục sigma dưới 3 mô hình bốc
python -u scripts/phan_bien_cat_canh.py      # cú cắt trên 66 nhãn thật
```
