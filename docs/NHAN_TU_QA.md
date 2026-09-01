# Nhân tử 0/1 của Q&A — đo lần đầu, và nó đổi thứ tự ưu tiên

Chốt 01/09/2026, theo `docs/KE_HOACH_DINH_VI.md` §4.1 (ưu tiên số một).

## 0. Vì sao phép đo này quan trọng hơn mọi phép đo định vị

Luật chấm Q&A đòi **cả ba**: đúng video, frame trong khoảng, **và đáp án khớp**.
Nên **điểm Q&A = điểm định vị × 1[đáp án đúng]**. Bốn lane định vị đo thừa số
thứ nhất; thừa số thứ hai **chưa được đo một lần nào**, và nó gánh **27% đề thật
/ 36% số câu vòng 2**.

## 1. Phép đo: 8 câu Q&A đề thật đã người kiểm chứng đáp án

Chạy `make_submission.py` **đầy đủ** (có cảnh B + hoán vị nội-video), rồi so đáp
án ở dòng 1 với đáp án đã kiểm chứng trong `data/ground_truth_de_that.json`.

| | đúng | sai | **RỖNG** |
|---|---|---|---|
| đường sản xuất **trước** | 1/8 | 3/8 | **4/8** |
| đường sản xuất **sau khi sửa** | 2/8 | 6/8 | **0/8** |

Và 2 trong 3 câu "sai" của đường cũ thực chất là model **từ chối trả lời**
("Không xác định được từ hình ảnh", "Không rõ tên đường"). Tức đường cũ có
**6/8 câu không có đáp án dùng được** — mỗi câu là 0 điểm bảo đảm, bất kể khung
hình đúng hay sai.

## 2. Nguyên nhân gốc: cải tiến đã ship KHÔNG tới được công cụ ngày thi

`make_submission.build_qa_rows` có **đường sinh đáp án riêng**
(`_make_answerer` → `gemini_engine.answer_single_frame` + biểu quyết đa số),
tách hoàn toàn khỏi `answer_qa.py`. Mọi cải tiến đã đo — ảnh gốc, lời thoại,
prompt cấm bỏ trống — nằm ở `answer_qa.py` và **chưa bao giờ chạy trong
make_submission**, tức chưa bao giờ có mặt trong file nộp trừ khi người vận hành
nhớ chạy thêm một lệnh nữa.

Ba lỗi của đường cũ, mỗi lỗi đủ để hỏng một mình:

1. `max_output_tokens=25` — quá chặt; model trả **rỗng** thay vì câu ngắn.
2. Prompt **không cấm bỏ trống**, còn cho phép nói "không xác định".
3. **Biểu quyết đa số** trên 5 khung: khi 4/5 khung là cảnh A thì phiếu của khung
   đúng bị **chủ động loại bỏ** — đúng hồ sơ của câu hai cảnh.

Và một lỗi thứ tư lộ ra khi sửa: hàm mới ban đầu gọi **một model cố định**, hết
quota theo phút là hỏng cả vòng. Đã thêm xoay vòng model (`_model_order`), thứ
cần cho ngày thi chứ không chỉ cho phép đo.

## 3. Đọc số cho đúng — đừng thổi phồng

- **Chắc chắn:** 4 câu bỏ trống → 0. Đây là phép đếm, và bỏ trống là 0 điểm bảo
  đảm theo luật 2.1.2. Cải thiện này không phụ thuộc cỡ mẫu.
- **Chưa chắc:** 1/8 → 2/8 đúng. Với n = 8 đây là **nhiễu**, không phải bằng chứng.
- **Một câu bị thụt lùi:** `p1-15` từ ĐÚNG thành SAI. Nguyên nhân không phải bước
  đọc mà là **bước định vị**: video ở dòng 1 của câu đó đổi, nên model đọc đáp án
  từ **video khác**. Đây là bằng chứng trực tiếp cho mục 4.

## 4. Kết luận đổi thứ tự ưu tiên

Ngay cả sau khi sửa, chỉ **2/8** câu có đáp án đúng — và chỉ **3/8** câu có video
đúng ở dòng 1. Tức **nhân tử 0/1 đang bị chặn bởi định vị**, không phải bởi năng
lực đọc: model đọc rất tự tin nhưng đọc **nhầm video**.

Hệ quả: với câu Q&A, mọi cải tiến định vị được nhân đôi giá trị (nó vừa ăn điểm
định vị, vừa mở khoá nhân tử đáp án), còn mọi cải tiến bước đọc bị chặn trên bởi
tỷ lệ video đúng. **Đó là lý do định vị vẫn là hướng chính**, nhưng bước đọc phải
được sửa vì nó rẻ và vì bỏ trống là mất trắng.

## 5. Việc còn nợ

- n = 8 quá nhỏ. Bước 2 của §4.1 (4 tập 5 khung × 132 câu) **chưa chạy** — hết
  quota Gemini free trong ngày. Chạy lại khi quota hồi.
- Chưa đo biến thể "cho model đọc khung SAU hoán vị so với TRƯỚC hoán vị" một
  cách có đối chứng — đó mới là câu hỏi §4.1 đặt ra.

---

## 6. ĐO LẠI TRÊN BỘ ĐO KHỚP PHÂN BỐ (02/09) — cảnh B **ăn điểm hai lần**

Sau khi vá đường sinh đáp án (chọn khung theo **ĐIỂM**, xem §7), chạy lại đủ
132 câu × 3 tập qua `scripts/do_nhan_tu_qa_bo_moi.py`:

| tập | TUNE | TEST | cả bộ | **HAI cảnh (TEST)** | MỘT cảnh (TEST) | video đúng |
|---|---|---|---|---|---|---|
| NEN | 45,5% | 54,5% | 50,0% | **45,5%** | 63,6% | 45,5% |
| + ứng viên cảnh B | 47,0% | 60,6% | 53,8% | **57,6%** | 63,6% | 48,5% |
| + hoán vị nội-video | 54,5% | 59,1% | 56,8% | **54,5%** | 63,6% | 48,5% |

Bootstrap theo câu, **riêng nhóm HAI cảnh** của TEST (n=33):

| lever | chênh | KTC 95% | P(≤0) |
|---|---|---|---|
| **ứng viên cảnh B** | **+12,1 điểm** | **[+0,030, +0,242]** | **1,4%** |
| + hoán vị nội-video | +9,1 điểm | [−0,061, +0,242] | 15,8% |

**Bất biến đạt:** nhóm MỘT cảnh đứng yên ở **63,6% cho cả ba tập** — hai lever
không đụng nhóm này, và phép đo xác nhận đúng như vậy.

### Đọc kết quả

**Lever cảnh B ăn điểm HAI LẦN.** Nó vừa cải thiện định vị (+23,3% đã đo), vừa
mở khoá kênh đáp án (**+12,1 điểm**, khoảng tin cậy **tách khỏi 0**). Đây chính
là "kênh thứ hai chưa ai tính vào giá trị của nó" mà `KE_HOACH_DINH_VI.md` §1
nêu ra — nay đã đo được. Với luật chấm `điểm = định vị × 1[đáp án đúng]`, hai
hiệu ứng này **nhân với nhau**, không cộng.

**Lever hoán vị thì KHÔNG — nói thẳng.** +9,1 điểm nhưng khoảng tin cậy chứa 0
(P(hoà) = 15,8%), và trên nửa TEST nó còn **thấp hơn** cảnh B đơn thuần
(54,5% vs 57,6%). Nó vẫn giữ nguyên giá trị đã đo ở kênh **định vị** (+57,6%);
chỉ là nó không mang thêm gì cho kênh đáp án. Không có lý do rút nó ra, cũng
không được tính thêm điểm cho nó ở đây.

### Giới hạn

- n = 33 ở nhóm hai cảnh của TEST. Đủ để khoảng tin cậy của cảnh B tách khỏi 0,
  **không** đủ để phân xử chênh lệch giữa hai lever.
- TUNE/TEST ở đây dùng để **báo cáo**, không phải để chọn — không có tham số nào
  được chọn trên TUNE. Nên hai nửa là hai phép đo độc lập của cùng một thứ, và
  chúng cùng chiều.
- Bộ so khớp là `_default_answer_match` ∪ `khop_rong`, không phải trọng tài LLM
  (tiết kiệm quota). Số tuyệt đối vì thế là **cận dưới**; số **chênh lệch** giữa
  ba tập không bị ảnh hưởng vì cả ba chấm bằng cùng một bộ.

## 7. Bản vá đúng, và bản vá SAI đã thử

Đường sinh đáp án chọn khung theo **thứ tự danh sách ứng viên**, còn hai lever
thể hiện qua **điểm** — cảnh B nối ứng viên vào *cuối* danh sách, hoán vị đổi
*điểm* chứ không đổi vị trí. Phép đếm tất định: khung neo đổi ở **0/66** câu khi
chọn theo thứ tự danh sách, **60/66** khi chọn theo điểm.

**Bản vá đúng:** sắp ứng viên theo điểm rồi lấy đầu danh sách.

**Bản vá SAI đã thử — đọc từ `frame_rows` (dòng sẽ nộp):** nghe hợp lý nhưng
**tệ hơn hẳn**, smoke test 0% ở cả ba tập. Lý do: bộ phân bổ phủ xác suất sinh
ra **điểm lưới**, chỉ **8–20 trên 100 dòng** là keyframe thật, nên "dòng đầu
tiên là keyframe" thường là một ứng viên yếu hơn nhiều. Ghi lại để không ai thử
lại — và nó cũng cảnh báo rằng `answer_qa.py` chạy độc lập (vốn đọc khung từ CSV)
đang chịu đúng vấn đề này kể từ khi allocator đổi sang `coverage`.
