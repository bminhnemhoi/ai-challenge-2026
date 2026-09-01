# TRAKE lần đầu có bộ đo — và ba con số đổi cách nhìn nhánh này

Chốt 02/09/2026. Sinh: `scripts/sinh_gt_trake.py` · Chấm: `scripts/do_trake_bo_moi.py`
Dữ liệu: `data/gt_trake.json` (dưới `data/` nên `.gitignore` chặn sẵn).

> Không mã video, không đáp án trong tài liệu này.

## 0. Vì sao nhánh này im lặng suốt

TRAKE chiếm **8,5% đề thật** (2/25 câu vòng 2) nhưng có **0 câu trong mọi bộ đo**.
Không bộ đo thì không đo được gì, nên mọi ý tưởng trong `docs/CHAN_DOAN_TRAKE.md`
nằm im từ đầu — không phải vì chúng sai, mà vì không có cách biết.

## 1. Bộ đo được dựng thế nào — và tránh lỗi gì

**12 mục, 3 sự kiện mỗi mục, 12 video khác nhau**, chọn ngẫu nhiên phân tầng theo
dải (không chọn theo điểm SigLIP — nếu không bộ đo chỉ phản ánh chính cái nó dùng
để chấm).

Thiết kế hai bước, tránh đúng lỗi đã phá bộ sinh KIS (nhét nhiều ảnh vào một
request ⇒ model tả đúng nhưng **đánh sai số thứ tự ảnh** ⇒ neo lệch một cú cắt ⇒
bộ đo **sai dấu**). Ở TRAKE lỗi ấy nhân lên N lần, nên:

1. cho xem cả đoạn → hỏi N mô tả sự kiện, **thuần văn bản, không số thứ tự nào**;
2. với mỗi sự kiện → chấm **từng khung, một ảnh một request**, lấy argmax.

**Các cổng loại đã hoạt động:** 5/17 video bị loại — 3 vì không khung nào đạt
ngưỡng (điểm cao nhất 30–40), 1 vì hai sự kiện trỏ vào **cùng một khung** (không
phân biệt được), 1 vì sinh sai số sự kiện.

**Kiểm chứng độ sắc trước khi tin:** 84 lượt chấm mẫu cho trung vị **0**, chỉ 14
khung đạt ≥90, 70 khung dưới 70 — phân bố hai đỉnh, đúng dấu hiệu của một bộ định
vị chứ không phải một bộ chấm dễ dãi. Và đã **mở ảnh kiểm bằng mắt** 4 sự kiện của
2 mục: 3 khớp hoàn hảo, 1 khớp cảnh nhưng mô tả hơi quá chi tiết.

## 2. Kết quả — ba mức, ba nguồn mất điểm

Cửa sổ {6,10,20}, 4 họ hạt giống × 48 bốc, mốc thật bốc đều trong khe keyframe
của **từng sự kiện**, chấm bằng `r_score_trake`/`final_score` thật.

| mức | điểm | so nền |
|---|---|---|
| **NỀN** (đường sản xuất) | **0,2154** | — |
| ORACLE-MỐC (video của nền, mốc thật) | 0,4307 | **+100,0%** |
| ORACLE-VIDEO (video đúng + mốc thật) | 0,5350 | **+148,4%** |

**Video ở dòng 1 đúng: 10/12 mục.**

### Phân rã khoảng cách (tổng 0,3196)

| nguồn mất điểm | phần |
|---|---|
| **định vị sự kiện trong video** | **67%** |
| chọn sai video | 33% |

## 3. Ba điều số này nói ra

**(a) Chọn video KHÔNG phải nghẽn chính của TRAKE — khác hẳn trực giác.**
10/12 mục đã có video đúng ở dòng 1. Hai phần ba khoảng cách nằm ở **định vị sự
kiện**. Điều này trùng khớp với kết luận bên nhánh KIS (video tìm đúng ngang nhau,
toàn bộ khoảng cách ở định vị nội-video) — cùng một nghẽn, hai nhánh khác nhau.

**(b) Ngay cả ORACLE-VIDEO cũng chỉ đạt 0,5350, không phải 1,0.** Biết đúng video
*và* đúng cả ba mốc vẫn chỉ được nửa điểm. Lý do nằm ở cấu trúc bài toán: một dòng
phải trúng **đồng thời** cả ba cửa sổ mới được 1,0, mà mốc thật rơi đâu đó trong
khe keyframe (~±27 frame) còn thang bù trừ bước 10 phải phủ ba trục cùng lúc.
**Đây là trần của chính cách phân bổ dòng hiện tại**, không phải của việc nhận
dạng sự kiện — và nó chỉ thẳng vào ý tưởng "lưới bù trừ phi-đều" trong
`CHAN_DOAN_TRAKE.md`: chia ngân sách dòng theo **độ bất định từng sự kiện** thay
vì đều nhau.

**(c) Luật chấm TRAKE rộng lượng hơn KIS nhiều.** Điểm từng phần, không ràng buộc
thứ tự: một câu 3 sự kiện trúng 1 đã được 1/3. Cộng với (b), chiến lược tối ưu cho
TRAKE **khác hẳn** KIS — và giờ mới kiểm chứng được bằng số thay vì suy luận.

## 4. Giới hạn — nói thẳng

- **n = 12** là rất nhỏ. Đủ để thấy phân rã 67/33 và trần 0,535; **không** đủ để
  phân xử chênh lệch vài phần trăm giữa các cấu hình phân bổ.
- Cả 12 mục đều có **đúng 3 sự kiện**. Đề thật có câu 4 sự kiện; chưa đo được ảnh
  hưởng của N lớn hơn (mà theo (b), N càng lớn thì trần càng thấp).
- Mốc do máy định vị, đã kiểm mắt 4/36 sự kiện. Sai sót còn lại chưa đo được.
- Bộ sinh dừng ở 12/20 vì **hết quota cả 5 model**. Sinh tiếp khi quota hồi.
