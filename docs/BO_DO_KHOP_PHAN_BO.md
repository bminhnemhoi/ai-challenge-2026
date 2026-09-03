# Bộ đo khớp phân bố đề thật — có gì, tin được đến đâu, và nó cho thấy gì

Chốt 30/08/2026. Dữ liệu: `data/ground_truth_moi.json` (64 mục, dưới `data/` nên
`.gitignore` chặn sẵn — **đừng `git add -f`**).

> Không mã video, không đáp án trong tài liệu này (docs/ lên GitHub công khai).

---

## 0. Vì sao phải dựng bộ đo mới

Bộ 60 câu cũ có **0/60** câu mô tả hai cảnh nối tiếp, còn đề THẬT của BTC có
**28/55 = 51%** (`docs/NGHIEN_CUU_SOTA.md` §3b). Nguyên nhân gốc: 60 câu cũ được
viết bằng cách **nhìn một keyframe**, còn BTC ra đề bằng cách **xem một đoạn
video**. Hệ quả: mọi kỹ thuật khai thác cấu trúc thời gian — đúng họ kỹ thuật mà
các đội vô địch VBS dùng — không thể đo được trên bộ cũ.

## 1. Bộ mới được dựng và kiểm thế nào

- **Sinh** (`scripts/sinh_gt_doan_video.py`): chọn video ngẫu nhiên phân tầng
  theo dải L21–L30, mỗi video lấy **một đoạn liên tiếp 8–12 keyframe**, cho VLM
  xem cả đoạn theo thứ tự thời gian và viết câu theo văn phong BTC (few-shot lấy
  từ chính đề thật). Nhắm ~50% câu hai cảnh.
- **Xác minh khung neo** (`scripts/kiem_neo_don_anh.py`) — bước quan trọng nhất.
- **Gộp** (`scripts/gop_bo_do_moi.py`): 64 mục, 64 video khác nhau, **32/64 =
  50% câu hai cảnh** (đề thật: 51%).

### Lỗi phải sửa, và vì sao nó nguy hiểm hơn vẻ ngoài

Bước sinh đưa **cả đoạn 9–12 ảnh vào một request**. Model tả nội dung đúng nhưng
**đánh sai số thứ tự ảnh**, nên khung neo trỏ lệch một cú cắt. Khung neo là
*đáp án* của bộ đo: neo sai không làm bộ đo yếu đi mà làm nó **sai dấu** — hệ
thống trả về đúng chỗ lại bị chấm là trượt, và mọi cải tiến đo trên đó đọc ngược.
Tệ hơn: bộ tự kiểm đầu tiên cũng lệch ±1 vì **mắc đúng nguyên nhân đó**.

Cách sửa tận gốc: **bỏ hẳn việc đánh số ảnh** — mỗi lần hỏi chỉ gửi ĐÚNG MỘT ảnh
kèm câu mô tả, hỏi "ảnh này khớp mô tả không, 0–100", quét ±2 keyframe quanh neo.
Không còn danh sách thì không còn chỗ để nhầm chỉ số.

Kết quả trên 64 mục: **56 xác nhận, 8 đề nghị dời neo, 0 nghi ngờ**, tổng $0,52
(gpt-5.2, ảnh 512px). Hai ca biên độ lớn nhất đã mở ảnh kiểm bằng mắt và **cả hai
đều đúng** — neo cũ trỏ vào mặt đường trống / bát bột cận cảnh, neo mới trỏ đúng
cảnh được tả. Script **không tự ghi đè** file gốc: nó xếp loại và ghi bằng chứng,
vì một bộ kiểm tự động ghi đè hàng loạt chính là cách lỗi lệch-một-khung lan ra
cả bộ đo.

### Nhiễu đã biết, ghi thành cờ chứ không giấu

- **shard c lẫn trục**: bước sinh dùng `dai[i % len(dai)]` cùng `chi_tieu =
  (i % 2 == 0)`; với 2 dải thì hai trục trùng khít, nên "câu hai cảnh" đồng nhất
  với "câu lấy từ dải thứ nhất". Mọi phép so *hai cảnh vs một cảnh* **phải loại
  shard c** (cờ `lan_truc`) — còn lại 48 mục, vẫn 50% hai cảnh.
- **shard b** thiếu trường cảnh_A/cảnh_B, nhãn hai cảnh là tự khai (cờ
  `nhan_hai_canh_tu_khai`).
- Vài đáp án Q&A là logo kênh — đúng kỹ thuật là danh từ cụ thể nhưng hằng số
  trên toàn kênh nên gần như không định vị được. Giữ lại, đánh dấu.

## 2. KẾT QUẢ ĐO — hệ thống sản xuất trên hai bộ

`scripts/do_bo_do_moi.py`, allocator `coverage` (mặc định sản xuất), cửa sổ
{6,10,20}, 4 họ hạt giống × 48 bốc, gốc hạt 77000 (tách khỏi mọi gốc đã dùng):

| nhóm | n | điểm | ± | video đúng trong 100 dòng |
|---|---|---|---|---|
| bộ CŨ (người viết, 0% hai cảnh) | 60 | **0,4004** | 0,0019 | 57/60 = 95% |
| bộ MỚI, cả 4 shard | 64 | 0,1974 | 0,0009 | 51/64 = 80% |
| bộ MỚI sạch (bỏ shard c) | 48 | 0,1690 | 0,0019 | 38/48 = 79% |
| ├─ câu **MỘT** cảnh | 24 | **0,2158** | 0,0015 | 20/24 |
| └─ câu **HAI** cảnh | 24 | **0,1213** | 0,0018 | 18/24 |

### Đọc kết quả

**(1) Câu hai cảnh khó hơn câu một cảnh −43,8%**, vượt 2σ (0,0036). Đây là lần
đầu tiên điểm mù về cấu trúc thời gian được **đo bằng số** thay vì suy luận.

**(2) Loại trừ nguyên nhân "câu máy sinh tả sai".** Nếu câu hai cảnh bị tả sai
nhiều hơn thì hệ thống không tìm được vì lỗi phép đo, không phải vì hệ thống yếu.
Kiểm bằng chính điểm khớp neo (mỗi mục một ảnh một request):

| nhóm | trung vị điểm khớp | ≥85 điểm | <70 điểm |
|---|---|---|---|
| một cảnh | 94 | 18/24 | **0/24** |
| hai cảnh | 92 | 18/24 | **0/24** |

Hai nhóm có chất lượng mô tả **y hệt nhau**. Khoảng cách −43,8% không phải do
câu tả sai.

**(3) Bộ mới khó hơn bộ cũ −46,1% ngay ở nhóm MỘT cảnh.** Hai bộ khác nhau ở
chỗ: câu cũ được viết *từ chính keyframe* mà bộ truy xuất đánh chỉ mục, nên nó
mô tả đúng thứ SigLIP nhìn thấy — một lợi thế nhân tạo. Câu mới viết từ đoạn
video, đúng cách BTC làm.

Bằng chứng bộ mới sát thực tế hơn: điểm thi thật vòng 2 là **10,0/30 = 0,333 mỗi
câu**, *đã tính công soát tay của người*. Bộ cũ cho 0,400 khi hoàn toàn tự động —
tức nó **lạc quan hơn cả kết quả có người sửa**. Bộ mới cho 0,197, thấp hơn.
Sự thật nằm giữa; nhưng **bộ cũ đang thổi phồng năng lực hệ thống khoảng 2 lần**.

## 3. Dùng bộ này thế nào

- **Chấm hai bộ song song**, đừng bỏ bộ cũ: nó là bộ đối chứng lịch sử cho mọi
  con số đã ghi trong `KIEN_TRUC_VA_HUONG_CAI_THIEN.md`.
- **Mọi phép so hai-cảnh/một-cảnh phải dùng bộ SẠCH 48 mục** (loại `lan_truc`).
- n = 24 mỗi nhóm là **nhỏ**: đủ để thấy một khoảng cách 44% nhưng không đủ để
  phân xử các cải tiến vài phần trăm. Cần sinh thêm trước khi dùng làm cổng chốt.
- Lever ③ (truy vấn cặp thời gian) **giờ đã đo được** — trước đây cổng bật 0/60
  nên mọi cấu hình là phép đồng nhất; trên bộ này cổng bật 24/48.

## 4. Điều chưa biết

1. Câu do máy sinh có thiên lệch riêng chưa đo được (dùng từ vựng gần với
   SigLIP hơn, hoặc ngược lại). Bằng chứng gián tiếp duy nhất: điểm khớp neo
   đồng đều giữa hai nhóm.
2. 4/8 câu hai cảnh của shard a là chuyển cảnh "trường quay ↔ hiện trường" —
   đặc thù bản tin. Nếu gộp nhiều shard mà tỷ lệ này vẫn cao thì bộ đo đang
   nghiêng về một kiểu chuyển cảnh dễ.
3. Chưa có câu TRAKE nào. Nhánh TRAKE vẫn bị chặn y như trước.
4. Neo đã dời được kiểm bằng gpt-5.2 + mắt cho 2/8 ca; 6 ca còn lại tin vào
   biên độ điểm (≥15) chứ chưa mở ảnh từng cái.

---

## ĐỢT MỞ RỘNG 03/09 — shard h (L28,L29) + i (L30,L25): 132 → 158 mục sạch

Lý do (`KE_HOACH_DINH_VI.md` §4.2b): n hiệu dụng Kish của nửa TUNE hai cảnh
chỉ 6,4; nửa TEST bộ 132 đã bị đọc ≥5 lần — cần GT mới trước mọi lần đọc TEST.

- Sinh: 27 mục dùng được (h: 11, i: 16; chỉ tiêu 50/50 giao trước khi nhìn
  video), $0,021 + kiểm $0,021.
- Kiểm neo một-ảnh-một-request: 19 xác nhận / 7 dời neo (áp dụng, ghi vết) /
  1 nghi ngờ → cách ly `gt_moi_can_soat.json`, KHÔNG chấm.
- Nhãn hai cảnh: bộ gắn nhãn độc lập (`gan_nhan_hai_canh.py`), tách A/B.
- Bộ sau ghép: **174 mục dùng được, 87 hai cảnh = 50%** (đề thật 51%);
  bộ SẠCH loại shard c: **158 mục, 79 hai cảnh**; 153 video khác nhau;
  dải: L21 22, L22 22, L23 16, L24 15, L25 21, L26 22, L27 22, L28 10,
  L29 6, L30 18.
- Cache ứng viên đang dựng lại cho mục mới (`do_bo_do_moi.py`).

**Kỷ luật đọc:** các mục mới CHƯA từng bị lane nào đọc — dành nửa TEST mới
cho các lever tiếp theo; mọi phép đo cũ muốn so số phải chạy lại trên bộ 158.

### Số nền trên bộ 158 (cache ứng viên dựng lại 03/09 chiều)

- pool đáp án: 160/174 mục có keyframe đáp án trong pool sản xuất.
- nền coverage (4 họ × 48 bốc): bộ sạch 158 = **0,2048 ± 0,0011**;
  MỘT cảnh 0,2873 / HAI cảnh 0,1233 (−57,1%, > 2σ) — bức tranh cũ tái lập
  với n gấp rưỡi. Video đúng trong 100 dòng: 127/158 (63/79 vs 64/79 —
  tìm-video ngang nhau giữa hai nhóm, đúng như bộ 132).
- bão hoà nội-video (bộ mới): MỘT cảnh hạng-1 trong ±20 = 34% (n=85),
  đáp án trung vị hạng 2, ≤3 ở 62%; HAI cảnh 9% (n=70), lệch trung vị 480
  frame. **Hình dạng trận-tay-đôi tái lập.**
