# Điểm cắt tương đối-trong-video cho nhóm MỘT cảnh — ÂM, đóng nốt §4.5

Chốt 01/09/2026. Script: `scripts/do_diem_cat_mot_canh.py` (0 API, ~8 phút CPU).

## 0. Kết luận một dòng

**ÂM. Không ship.** Tín hiệu **có tồn tại** đúng như phản biện dự đoán, nhưng
quá yếu: TUNE +4,5% (đỉnh của một đường không đơn điệu), TEST **−3,0%** dưới mô
hình bốc ĐỀU và **−4,3%** dưới SAU_NEO, P(hoà hoặc âm) = 77% / 88%.

## 1. Vì sao thử

`docs/KE_HOACH_DINH_VI.md` §4.5 để lại đúng một việc 0 đồng. Lane paper kết luận
"cosine SigLIP liền kề không đo được cú cắt", nhưng kết luận ấy dựa trên một
ngưỡng **tuyệt đối** (0,5) nằm **dưới trung vị của cặp ngẫu nhiên khác video** —
tức nó đòi hai khung cùng bản tin phải khác nhau hơn hai khung của hai video
hoàn toàn khác nhau. Trên thang **tương đối trong từng video** thì có tín hiệu.

Nhóm MỘT cảnh là nhóm duy nhất **chưa có tín hiệu nội-video nào**: cả lever cảnh B
lẫn lever hoán vị đều chỉ chạm câu hai cảnh.

## 2. Chẩn đoán — đo TRƯỚC khi nhìn điểm

Phân vị điểm cắt (1 − cos với keyframe liền trước, xếp hạng **trong chính video
đó**) của keyframe đáp án, so với một keyframe ngẫu nhiên cùng video:

| | trung vị phân vị |
|---|---|
| keyframe **đáp án** | **0,564** |
| keyframe ngẫu nhiên cùng video | 0,505 |

**Có tín hiệu** — khoảnh khắc người ra đề mô tả quả thật hơi nghiêng về chỗ mở
đầu một cú cắt. Nhưng chênh lệch chỉ **0,06 phân vị**: rất yếu.

## 3. Kết quả — và vì sao tín hiệu yếu không dùng được

TUNE (nhóm một cảnh, n=33), nền 0,2290:

| w | điểm | so nền |
|---|---|---|
| 0,01 | 0,2300 | +0,5% |
| 0,03 | 0,2283 | −0,3% |
| **0,10** | **0,2392** | **+4,5%** |
| 0,30 | 0,2095 | −8,5% |

TEST (đọc đúng một lần, n=33, bootstrap theo câu, song song hai mô hình bốc):

| mô hình bốc | nền | chốt | chênh | KTC 95% | P(≤0) |
|---|---|---|---|---|---|
| ĐỀU | 0,3189 | 0,3093 | **−3,0%** | [−0,0352, +0,0158] | 77,4% |
| SAU_NEO | 0,3309 | 0,3166 | **−4,3%** | [−0,0384, +0,0099] | 87,8% |

**Đường TUNE không đơn điệu** (+0,5% / −0,3% / +4,5% / −8,5%) — không có cao
nguyên. Đó là chữ ký của **nhiễu**, ngược hẳn với hai lever đã ship: lever cảnh B
phẳng từ M=25 đến M=200, lever hoán vị phẳng từ w=1 đến w=100. Bài học lặp lại:
**đường tham số phẳng là điều kiện cần**, và ở đây nó không có.

## 4. Điều này đóng cửa gì

- 🟩 **Đóng hẳn:** điểm cắt cosine liền kề (dù ở thang tương đối) làm đặc trưng
  **xếp hạng nội-video**. Có tín hiệu, không đủ mạnh, âm trên TEST ở cả hai mô
  hình bốc.
- Cùng với kết luận (1) đã đóng ở §4.5, **toàn bộ hướng "cắt cảnh giá rẻ" nay
  đóng cả hai nửa**. Khuyến nghị **không mua TransNetV2** giữ nguyên: nó phải
  thắng được một tín hiệu mà ta vừa đo là gần như vô dụng ở thang này, với giá
  26–60 GB tải + 20–30 giờ GPU.
- **Nhóm MỘT cảnh vẫn chưa có tín hiệu nội-video nào.** Đó vẫn là lỗ hổng lớn
  nhất còn lại: 47% của trần thuộc tầng xếp hạng, và nhóm này không có gì để xếp.

## 5. Giới hạn

n = 33 mỗi nửa là nhỏ. Đọc đúng là **"không chứng minh được là có"**, không phải
"đã chứng minh là không có" — nhưng chẩn đoán 0,564 vs 0,505 nói tín hiệu thật
sự yếu, nên khả năng một bộ đo lớn hơn cứu được nó là thấp.
