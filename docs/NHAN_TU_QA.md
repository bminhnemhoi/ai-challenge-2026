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
