# Lever ③ — truy vấn cặp thời gian, đo trên bộ đo khớp phân bố

Chốt 31/08/2026. Script: `scripts/do_cap_thoi_gian_moi.py`.
Bộ đo: `data/ground_truth_moi.json` (xem `docs/BO_DO_KHOP_PHAN_BO.md`).

## 0. Kết luận một dòng

**CHƯA KẾT LUẬN ĐƯỢC — không ship.** Hiệu ứng đo được là **+11,5% trên TEST**
và lặp lại qua hai lần chia mẫu khác nhau, nhưng khoảng tin cậy 95% **theo câu**
là `[−0,0138, +0,0751]` — **chứa 0**, còn 14% khả năng là hoà. Với 12 câu qua
cổng ở mỗi nửa, cỡ mẫu chưa loại được may rủi.

Cần khoảng **81 mục TEST** (hiện 24) để khoảng tin cậy tách khỏi 0 nếu hiệu ứng
giữ nguyên độ lớn — tức sinh thêm ~114 mục cho cả hai nửa.

## 1. Vì sao lần này đo được, lần trước thì không

Lần trước (`docs/CAP_THOI_GIAN.md`) kết luận **CHƯA ĐO ĐƯỢC** vì cổng "câu này
mô tả hai cảnh nối tiếp" bật **0/60** trên bộ ground truth cũ — mọi cấu hình là
phép đồng nhất. Bộ mới có 50% câu hai cảnh, cổng bật **24/48** trên bộ sạch.

## 2. Kết quả

TUNE 24 mục (12 qua cổng) — chọn tham số; TEST 24 mục (12 qua cổng) — đọc một lần.

| | điểm | so nền |
|---|---|---|
| nền (allocator coverage, không đổi điểm) | 0,2107 ± 0,0019 | — |
| **chốt: W=2, gộp = tích, λ=0,5** | **0,2350 ± 0,0021** | **+11,5%** |

Cấu hình thắng **nhất quán**: `tích` với λ=0,5 dương ở cả bốn giá trị W
(+5,4% → +8,3% trên TUNE), `chiA` (bỏ hẳn cảnh B) âm ở mọi ô (−6% → −10%) —
tức phần đóng góp đến từ **cảnh B**, đúng cơ chế mà lever này giả định, không
phải từ việc rút gọn truy vấn.

Bất biến đã kiểm bằng assert: **32 mục cổng tắt ra 100 dòng giống hệt nền**.

## 3. Chỉnh phương pháp — áp dụng cho MỌI phép đo sau

Luật 2σ đang dùng khắp dự án lấy σ **giữa các họ hạt giống**. Ở bộ phân bổ phủ
xác suất nó đúng: tập câu cố định 30 mục và hiệu ứng lớn. Ở đây nó **sai**, và
sai theo hướng nguy hiểm:

> σ hạt giống đo nhiễu **bốc thăm đáp án**. Tăng số lần bốc là nó nhỏ đi, dù ta
> chẳng biết thêm gì về câu hỏi. Khi số câu nhỏ, nguồn bất định chính không phải
> bốc thăm mà là **đổi tập câu** — và σ hạt giống mù hoàn toàn với nguồn đó.

Cụ thể ở đây: 2σ hạt giống tuyên bố **GIỮ ĐƯỢC**, còn bootstrap theo câu nói
**14% khả năng hoà**. Hai thước đo cho hai kết luận trái ngược trên cùng một dữ
liệu. Script giờ in cả hai và **phán quyết theo bootstrap**.

**Quy tắc từ nay:** phép đo nào có dưới ~30 câu ở nhóm bị tác động thì phải kèm
bootstrap theo câu; con số 2σ hạt giống chỉ được ghi kèm để đối chiếu, không
được dùng làm cổng.

## 4. Việc phải làm để chốt được lever này

1. Sinh thêm ~114 mục cho bộ đo mới (ưu tiên câu hai cảnh), theo đúng quy trình
   đã có: sinh từ đoạn video → **xác minh neo bằng một-ảnh-một-request** →
   kiểm độc lập. Bước xác minh neo là bắt buộc: 8/64 mục lần đầu có neo lệch.
2. Chạy lại script này. Nếu khoảng tin cậy tách khỏi 0 và điểm giữ ≥ +5%,
   ship với cờ `--cap-thoi-gian` mặc định bật cho câu qua cổng.
3. Trước khi bật trong trận: chạy diff cấu trúc trên đề thật
   (`experiment_cap_thoi_gian.py --de`) để biết nó đổi những gì khi không có
   ground truth.

## 5. Điều chưa biết

1. **Độ chính xác của cổng trên đề THẬT.** Trên bộ mới, câu được sinh ra với
   cấu trúc hai cảnh rõ ràng nên bộ gắn nhãn dễ đúng. Trên đề BTC nó bật 51%
   nhưng chưa ai kiểm từng câu xem có đúng không. Cổng sai ⇒ đổi điểm những câu
   lẽ ra không được đụng.
2. Câu do máy sinh có thể có cấu trúc hai cảnh "sạch" hơn đề người viết
   (chuyển cảnh dứt khoát), khiến lever này ăn hơn thực tế.
3. Chưa đo trên allocator `hybrid` (đối chứng) — script có sẵn cờ.

---

## 6. LẦN ĐO THỨ HAI (01/09/2026) — mẫu gấp 2,75 lần, hiệu ứng CO LẠI

Đã sinh thêm 84 mục (shard e/f/g), xác minh neo bằng một-ảnh-một-request
(**72 xác nhận / 12 dời neo / 0 nghi ngờ**, $0,64 — tỷ lệ neo lệch 14%, khớp
12,5% của lượt trước). Bộ sạch: **132 mục, 66 câu hai cảnh**.

| | n TEST | qua cổng | TEST | khoảng tin cậy theo câu | P(hoà) |
|---|---|---|---|---|---|
| lần 1 | 24 | 12 | **+11,5%** | [−0,0138, +0,0751] | 14% |
| **lần 2** | **66** | **33** | **+6,7%** | [−0,0135, +0,0496] | **17%** |

**Hiệu ứng giảm gần một nửa khi có thêm dữ liệu**, và cấu hình thắng trên TUNE
cũng đổi (W=2, λ=0,5 → W=3, λ=1,0). Cả hai đều là dấu hiệu của **ước lượng bị
thổi phồng do mẫu nhỏ**, không phải của một hiệu ứng vững.

Cỡ mẫu cần để chốt tăng từ 81 lên **259 mục TEST** (phải sinh thêm ~386 mục) —
tức càng đo càng xa đích. **Kết luận vận hành: dừng đầu tư vào lever này.**
Nó có vẻ là một khoản dương nhỏ (~+5–7%), nhưng chi phí xác nhận đã vượt xa giá
trị kỳ vọng, trong khi cùng công sức đó đổ vào hướng khác đáng giá hơn.

### Chẩn đoán sắc hơn, và nó chỉ sang hướng khác

Trên bộ 132 mục, câu hai cảnh đạt 0,1036 so với 0,2767 của câu một cảnh
(**−62,5%**, n=66 mỗi nhóm). Nhưng:

| | video đúng trong 100 dòng |
|---|---|
| câu MỘT cảnh | 51/66 = 77% |
| câu HAI cảnh | **53/66 = 80%** |

**Hệ thống tìm đúng video NGANG NHAU ở cả hai nhóm.** Toàn bộ khoảng cách nằm ở
**định vị khoảnh khắc TRONG video** — không phải ở việc chọn video. Lever ③ tấn
công đúng chỗ đó nhưng chỉ mua được ~+7%, nên vấn đề không nằm ở phía truy vấn.

Hướng có bằng chứng tốt hơn cho cùng nghẽn: **encoder mạnh hơn cho định vị
nội-video** (lever ④ — PE-Core + hợp nhất RRF hai tầng). Bảng tín hiệu cũ đã
đóng mọi tín hiệu RẺ cho định vị nội-video (làm mượt thời gian, CLIP-blend,
OCR keyword), nên đây là hướng còn lại chưa thử.
