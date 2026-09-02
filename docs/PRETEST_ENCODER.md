# Pre-test encoder thứ hai (PE-Core) — cổng GO/NO-GO trước khi đốt GPU

Lane `pe-core`, 02/09/2026. Script: `scripts/pretest_pe_core.py`.
File này được viết **trước khi chạy**: mục §2 (ngưỡng) chốt xong mới nhìn số.
Kết quả điền vào §5 sau khi chạy, không sửa §1–§3.

> Không mã video, không đáp án trong tài liệu này (docs/ công khai).

---

## 1. Câu hỏi cổng này trả lời

Nhóm MỘT cảnh chưa có tín hiệu xếp hạng nội-video nào sống sót (làm mượt thời
gian ÂM, chuẩn hoá video ÂM, ưu tiên đỉnh trơ, điểm cắt ÂM, VLM thua tín hiệu
0 đồng), trong khi tầng XẾP HẠNG nội-video chiếm **40% trần**
(`docs/QUY_TRINH_TU_DONG.md` §6). Encoder thứ hai là hướng duy nhất chưa thử.

Câu hỏi — trả lời bằng phép đếm, trước khi ai tốn 2–4h GPU Colab dựng chỉ mục
đầy đủ: **PE-Core có xếp keyframe đáp án nội-video TỐT HƠN SigLIP-2 SO400M-384
trên bộ đo khớp phân bố không?**

## 2. Ngưỡng GO — chốt TRƯỚC khi nhìn số

Trạng thái SigLIP đã đo (bộ sạch 132 mục, `docs/KE_HOACH_DINH_VI.md` §2.2):
hạng nội-video của keyframe đáp án — MỘT cảnh trung vị **2,0**, hạng-1 **~43%**;
HAI cảnh trung vị **6,0**, hạng-1 **11%**.

**GO** nếu, trên mẫu pre-test, PE-Core đạt **ít nhất một** trong hai:

1. trung vị hạng nội-video nhóm **HAI cảnh ≤ 3** (SigLIP: 6), HOẶC
2. tỷ lệ hạng-1 nhóm **MỘT cảnh ≥ 55%** (SigLIP: ~43%).

Không đạt cả hai → **NO_GO**, ghi số vào bảng cửa đóng. Nếu không tải nổi model
nào (cả L-14-336 lẫn B-16-224) → **NO_GO-vì-hạ-tầng**, viết notebook Colab cho
người có mạng khác chạy, và dừng.

Ngưỡng áp cho **cấu hình văn bản tốt nhất trong hai cấu hình đã khai báo trước**
ở §3.4 (cả hai đều triển khai được — lớp dịch đã có sẵn trong hệ thống). Các
biến thể chẩn đoán (vi-thuần, en-thuần) chỉ để đọc nguyên nhân, không dự thi.

## 3. Giao thức — chốt trước khi chạy

### 3.1 Mẫu

- **40 mục** từ bộ SẠCH 132 (`data/ground_truth_moi.json`, lọc `lan_truc`):
  **20 MỘT cảnh / 20 HAI cảnh**, bốc phân tầng, `numpy RandomState(92026)`
  (gốc hạt mới, chưa dùng ở đâu).
- Mỗi mục: video ĐÚNG của nó, toàn bộ keyframe hợp lệ của video đó
  (mặt nạ `valid` sản xuất: bỏ 2 khung đầu + khung trắng).

### 3.2 Phía SigLIP — đúng đầu vào sản xuất (kỷ luật đo #6)

Đọc thẳng `data/cache_tin_hieu_noi_video/sims_sach.npy` — sims toàn kho của
132 câu sạch, dựng bằng đúng ngữ nghĩa `query_similarities` sản xuất (ensemble
4 prompt + cắt khúc, đã đối chiếu trùng từng điểm với ứng viên sản xuất).
Không encode lại gì.

Kiểm nền: số trên toàn bộ 132 mục phải khớp số đã công bố (trung vị 2/6) —
nếu lệch thì quy ước hạng khác nhau, phải ghi rõ trước khi đọc tiếp.

### 3.3 Phía PE-Core

- Model theo thứ tự thử: `timm/PE-Core-L-14-336` (3 lần retry, hf_transfer);
  hỏng → `timm/PE-Core-B-16-224`; hỏng nốt → NO_GO-vì-hạ-tầng.
- **Ảnh:** encode toàn bộ keyframe hợp lệ của video đúng, từ bản
  `data/frames/` **512×288** — *hạn chế ghi nhận*: chỉ mục SigLIP dựng từ ảnh
  gốc; PE chỉ được nhìn bản 512px. Tiền xử lý: **resize ép về vuông**
  (squash), giữ trọn nội dung khung 16:9 — cùng triết lý với processor SigLIP
  sản xuất (squash 384×384), vì bài toán truy xuất cần cả khung, center-crop
  chuẩn của CLIP vứt ~44% bề ngang. Kiểm độ nhạy: 5 video đầu encode thêm bản
  center-crop, báo cáo lệch.
- **Văn bản:** tokenizer của chính model (context ngắn, câu dài bị cắt —
  ghi nhận, SigLIP sản xuất cũng cắt ở 64 token nhưng có cắt khúc).

### 3.4 Hai cấu hình văn bản dự thi (khai báo trước, quyết định lấy max)

| tên | prompt | trọng số |
|---|---|---|
| **A ensemble-vi** | [en, vi, "a high quality video keyframe of {en}", "a photo of {en}"] | 0,45/0,35/0,10/0,10 — y hệ số sản xuất |
| **B ensemble-en** | [en, "a high quality video keyframe of {en}", "a photo of {en}"] | 0,6923/0,1538/0,1538 — sản xuất bỏ vi, chuẩn hoá lại |

`en` = `kis_query_en` có sẵn trong bộ đo (không dịch mới). Chẩn đoán thêm
(không dự thi): en-thuần, vi-thuần — để trả lời "text tower PE có yếu tiếng
Việt không"; nếu en tốt mà vi kém thì triển khai cần lớp dịch (đã có sẵn).

### 3.5 Thước đo

- **Hạng nội-video** của keyframe đáp án trong tập keyframe hợp lệ của video
  đúng: `1 + #{khung có sim > sim(khung đáp án)}` — lạc quan với hoà, dùng
  **cùng quy ước cho cả hai encoder**.
- Báo cáo TÁCH RIÊNG hai nhóm: trung vị hạng, tỷ lệ hạng-1, tỷ lệ top-5.
- **Ghép cặp theo mục** (cùng 40 mục, cùng video, cùng mặt nạ valid):
  thắng/thua/hoà theo hạng, bootstrap **theo câu** 4000 lần cho hiệu
  hạng-1 và hiệu trung vị.
- n=20 mỗi nhóm là NHỎ — cổng này chỉ đủ phân xử hiệu ứng cỡ ngưỡng §2
  (kéo trung vị 6→3, nâng hạng-1 43→55). Chênh vài phần trăm: không kết luận.

### 3.6 Cái pre-test này KHÔNG đo

- Điểm nộp cuối (không chạy allocate_rows — cổng này đo đúng một tầng:
  xếp hạng nội-video).
- Khả năng **tìm video** của PE (chỉ đo trong video đúng). Nếu GO, bước sau
  vẫn phải đo tìm-video trên chỉ mục đầy đủ trước khi thay/trộn encoder.
- Ảnh gốc độ phân giải cao (chỉ có bản 512px tại chỗ).

## 4. Vì sao các số này quyết định được GO/NO-GO

Tầng xếp hạng nội-video chiếm 40% trần. Nếu PE không kéo nổi trung vị hạng
nhóm hai cảnh xuống ≤3 và không nâng nổi hạng-1 nhóm một cảnh lên ≥55% ngay
trên video ĐÚNG (điều kiện dễ nhất — không phải tìm video giữa 873), thì chỉ
mục đầy đủ 177k keyframe không thể cho nó nhiều hơn: **đừng đốt GPU**.

---

## 5. KẾT QUẢ (điền sau khi chạy — không sửa gì phía trên)

*(chưa chạy)*
