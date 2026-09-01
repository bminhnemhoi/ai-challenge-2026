# Truy xuất thêm bằng CẢNH B — chữa đúng cơ chế hỏng của câu hai cảnh

Chốt 01/09/2026. Script đo: `scripts/do_ung_vien_canh_b.py`.
Sản xuất: `make_submission.py --canh-b M` (mặc định 100).

## 0. Chẩn đoán dẫn tới thay đổi này

Trên bộ đo khớp phân bố đề thật (132 mục, `docs/BO_DO_KHOP_PHAN_BO.md`):

| | keyframe đáp án có trong 400 ứng viên | hạng nội-video (trung vị) | video đúng trong 100 dòng |
|---|---|---|---|
| câu MỘT cảnh | 58/65 | 2 | 51/66 |
| câu HAI cảnh | **35/65** | 8 | **53/66** |

Hai cột ngoài cùng đọc cùng nhau mới thấy cơ chế: **video vẫn được tìm đúng
ngang nhau**, nhưng ở câu hai cảnh thì keyframe đáp án **không hề nằm trong danh
sách ứng viên** ở gần một nửa số câu. Truy vấn nén cả hai cảnh vào một vector nên
nó khớp cảnh **mở đầu**; hệ thống tìm ra đúng video *qua keyframe của cảnh A*,
rồi rải 100 dòng quanh cảnh A — trong khi đáp án nằm ở cảnh B.

Đây cũng là lời giải thích cho việc lever ③ (chấm lại theo cặp thời gian) chỉ mua
được +6,7%: nó **xếp lại** các ứng viên sẵn có, mà thứ cần cứu thì chưa bao giờ
có mặt. **Nghẽn ở khâu SINH ứng viên, không phải khâu xếp hạng.**

## 1. Cách chữa

Với câu qua cổng hai cảnh: lấy thêm **top-M keyframe theo độ tương đồng với
riêng cảnh B**, gộp vào cuối danh sách ứng viên rồi để allocator làm việc như cũ.

Đây **không** phải phép trộn điểm hai kênh — bằng chứng nội bộ đã đóng cửa đó
(caption, doc2query, BM25+SigLIP). Nó là **hợp hai lần truy xuất của cùng một
encoder**, cùng thang cosine, nên không có hệ số pha trộn nào phải chọn.

Câu KHÔNG qua cổng giữ nguyên 100% đường cũ — kiểm bằng `assert` trên toàn bộ
66 mục cổng tắt, với mọi giá trị M.

## 2. Bằng chứng

### (a) Cơ chế — đếm tất định, không có nhiễu thống kê

| | keyframe đáp án có mặt trong pool |
|---|---|
| chỉ 400 ứng viên gốc | 35/66 = **53%** |
| + top-100 của cảnh B | 50/66 = **76%** |

**Cứu thêm 15 câu** mà trước đó keyframe đáp án chưa bao giờ được truy xuất.
Đây là phép đếm, không phải ước lượng — không có khoảng tin cậy nào để bàn.

### (b) Điểm — TUNE/TEST, bootstrap theo câu

| M | điểm TUNE | so nền |
|---|---|---|
| 25 | 0,1568 | +9,8% |
| 50 | 0,1570 | +10,0% |
| **100 (chốt)** | **0,1575** | **+10,3%** |
| 200 | 0,1575 | +10,3% |

Đường TUNE **phẳng** từ M=25 tới M=200 — hiệu ứng không phụ thuộc việc chọn
đúng tham số, dấu hiệu của cơ chế thật chứ không phải may rủi.

TEST (đọc đúng một lần), 66 mục / 33 qua cổng:

| | nền | có cảnh B | chênh | KTC 95% theo câu | P(hoà) |
|---|---|---|---|---|---|
| toàn bộ TEST | 0,2379 | 0,2562 | **+7,7%** | [−0,0009, +0,0431] | 3,4% |
| **chỉ câu qua cổng** | 0,1576 | 0,1943 | **+23,3%** | [−0,0028, +0,0863] | **3,8%** |

## 3. Vì sao ship, dù P(hoà) ≈ 4% chưa dưới ngưỡng 2,5%

Ba lý do, và cả ba đều phải đúng cùng lúc:

1. **Cơ chế được chứng minh riêng, không qua thống kê.** Phép đếm 53% → 76% là
   tất định. Điểm số chỉ là hệ quả của việc keyframe đáp án có mặt hay không.
2. **Đường tham số phẳng.** Mọi M từ 25 đến 200 cho cùng mức tăng. So sánh với
   lever ③: cấu hình thắng ở đó ĐỔI khi thêm dữ liệu — dấu hiệu ngược lại.
3. **Rủi ro có chặn cứng.** Câu không qua cổng ra dòng giống hệt nền (assert),
   nên tác hại tối đa chỉ giới hạn ở nhóm qua cổng; và R@k là *max trên tiền tố*
   nên ứng viên nối vào cuối chỉ cạnh tranh chỗ xếp hạng, không thể xoá dòng đúng
   nào đang có.

Đường rút lui: `--canh-b 0`.

**Điều chưa biết:** độ chính xác của cổng gắn nhãn trên đề THẬT (nó bật 51% số
câu BTC nhưng chưa ai kiểm từng câu). Cổng bật nhầm ở câu một cảnh sẽ thêm ứng
viên không liên quan vào cuối danh sách — theo luật chấm thì gần như vô hại,
nhưng chưa đo.

---

## 4. Diff cấu trúc trên ĐỀ THẬT — và một rủi ro phải nói rõ

`scripts/diff_canh_b_de_that.py` (đọc nhãn từ cache, **0 lần gọi LLM**) chạy trên
55 câu đề thật của hai vòng. Cổng bật **28/55 = 51%**.

| | trung vị | min | max |
|---|---|---|---|
| ứng viên MỚI mỗi câu | 57 | 7 | 100 |
| số dòng đổi trong 100 dòng | 20 | 0 | 100 |

**Video dòng 1 đổi ở 7/28 câu (25%).** Đây là con số **rủi ro**, không phải con
số tốt: R@1 đáng 1,0 — số hạng đắt nhất trong công thức chấm — nên đổi video
dòng 1 là một canh bạc lớn hơn nhiều so với vẻ ngoài của phép đo trên harness.
Phép đo trên bộ đo mới không lộ ra điều này vì câu ở đó do máy sinh, cấu trúc
hai cảnh dứt khoát hơn đề người viết.

### Kiểm 7 câu ấy bằng đáp án đã biết

| câu | độ tin của nhãn | kết quả |
|---|---|---|
| p1-19-kis | **người kiểm chứng** | **cảnh B đổi ĐÚNG sang video đúng** |
| p1-18-kis, p1-2-kis, p1-6-kis, p2-19-qa, p2-30-qa | chỉ `suy_ra` | GT khác **cả** neo cũ lẫn neo mới ⇒ pipeline sai cả trước lẫn sau, không mất gì |
| p2-15-kis | không có nhãn | chưa biết |

Trong đúng một ca kiểm chứng được, cảnh B **sửa đúng**. Ở năm ca còn lại nó
không làm hỏng thứ gì đang đúng. Bằng chứng này yếu (n=1 ca chắc chắn) nhưng
**cùng chiều** với phép đo trên harness, và quan trọng hơn: nó cho thấy 25% đổi
video **không phải** là phá hỏng những câu vốn đã đúng.

**Việc phải làm trước ngày thi:** chạy lại diff này trên đề của vòng mới và soi
bằng mắt các câu bị đổi video dòng 1 — đó là danh sách ngắn (khoảng 7/28) và
mỗi câu trong đó đáng 1 điểm R@1. Lệnh rút lui nếu thấy sai: `--canh-b 0`.
