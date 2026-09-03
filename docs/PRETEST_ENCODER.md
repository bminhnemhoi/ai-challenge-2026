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

## 2b. Phụ lục quy ước hạng — ghi 03/09, TRƯỚC khi chấm PE

*(Viết khi giai đoạn encode đang chạy — chưa tồn tại một con số PE nào.
Đây chính là tình huống §3.2 đã dự liệu: "nếu lệch thì quy ước hạng khác
nhau, phải ghi rõ trước khi đọc tiếp".)*

Kiểm nền §3.2 đã chạy và **lệch** với số công bố ở nhóm HAI cảnh:

| quy ước | MỘT cảnh (trung vị / hạng-1) | HAI cảnh (trung vị / hạng-1) |
|---|---|---|
| công bố (`KE_HOACH_DINH_VI.md` §2.2) — hạng **TRONG POOL 400**, chỉ đếm câu có đáp án lọt pool | 2,0 / 43% | 6,0 / 11% |
| pre-test — hạng trên **TOÀN video** (mọi khung valid), cả 132 mục | 2,0 / 37,9% | **15,0 / 1,5%** |

Nguyên nhân, đọc thẳng từ `do_tin_hieu_noi_video._chan_doan`: số công bố là
thước "hạng trong pool" — (i) **có điều kiện** đáp án lọt pool 400 (nhóm hai
cảnh chỉ 35/65 lọt, trước cảnh B), (ii) pool do **chính SigLIP dựng**, nên
không so được hai encoder một cách đối xứng. Pre-test dùng thước "hạng trên
TOÀN video" — docstring của chính script đó gọi đây là "thước sạch của định
vị nội-video". Quy ước pre-test là quy ước ĐÚNG cho câu hỏi của cổng này;
chỉ có *mốc số* của ngưỡng phải dịch lại.

**Ngưỡng §2 dịch sang quy ước đo, giữ nguyên Ý ĐỊNH, chốt bây giờ:**

1. "trung vị HAI cảnh 6→3" nghĩa là **giảm một nửa** trung vị → dịch:
   trung vị PE (hai cảnh, 20 mục) ≤ **½ trung vị SigLIP trên CÙNG 20 mục**;
2. "hạng-1 MỘT cảnh 43%→55%" nghĩa là **+12 điểm phần trăm** → dịch:
   hạng-1 PE (một cảnh, 20 mục) ≥ **hạng-1 SigLIP cùng mẫu + 12pp**.

**GO** nếu đạt ít nhất một trong hai (điểm ước lượng, như §2 gốc; mỗi tiêu
chí lấy cấu hình dự thi tốt hơn trong A/B, đúng như mã đã viết trước).
Ngưỡng tuyệt đối gốc (≤3; ≥55%) vẫn báo cáo song song — dưới quy ước
toàn-video chúng **khắt khe hơn** ngưỡng dịch; đạt được thì ghi rõ là
vượt cả mốc gốc.

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

Chạy 03/09/2026, `round2/pe_cham.log`, `data/cache_pretest_pe/ket_qua.json`.
Mẫu 20 MỘT / 20 HAI cảnh, seed 92026; PE nạp đường fp16 ít bộ nhớ.

| cấu hình | MỘT: trung vị / hạng-1 / top-5 | HAI: trung vị / hạng-1 / top-5 |
|---|---|---|
| SigLIP (sản xuất) | 2,0 / 35,0% / 70,0% | 13,5 / 0,0% / 10,0% |
| **PE A_ensemble_vi** | 2,0 / **45,0%** / 80,0% | 11,5 / 0,0% / 20,0% |
| PE B_ensemble_en | 2,0 / 35,0% / 70,0% | 12,0 / 5,0% / 30,0% |
| PE en_thuần | 2,0 / 35,0% / 75,0% | 11,5 / 10,0% / 30,0% |
| PE vi_thuần | 64,0 / 5,0% / 10,0% | 84,5 / 5,0% / 5,0% |

**Ngưỡng dịch §2b (chốt trước):** (1) trung vị HAI ≤ 6,8? PE 11,5 → KHÔNG.
(2) hạng-1 MỘT ≥ 47,0%? PE 45,0% → KHÔNG. Bootstrap Δhạng-1 MỘT của A:
KTC95 [−15,0%, +35,0%] — chứa 0.

## ⇒ **NO_GO — GPU vẫn KHOÁ, cửa đóng KÈM SỐ.**

Ba điều đọc được từ số, ghi lại cho trung thực:

1. **Tín hiệu dương có thật nhưng dưới ngưỡng và không chắc:** A_ensemble_vi
   +10pp hạng-1 một cảnh (9 thắng/5 thua), nhưng ngưỡng +12pp đặt trước để bù
   rủi ro chỉ-mục-đầy-đủ, và KTC chứa 0 ở n=20. Ngưỡng không được hạ sau khi
   nhìn số — đó là toàn bộ giá trị của cổng tiền-đăng-ký.
2. **Nhóm HAI cảnh — lý do chính muốn encoder mới — PE gần như không nhúc
   nhích** (11,5 so 13,5; cần ≤6,8): khớp chẩn đoán `dem_bao_hoa_noi_video.py`
   (lệch hệ thống 752 frame vì text tả cảnh A) — đây là lỗi CẤU TRÚC đề, không
   phải lỗi encoder; không encoder nào tự sửa.
3. **Giới hạn công cụ phải mang theo:** text tower PE context 32 token BPE
   tiếng Anh — 40/40 câu vi và 36/40 câu en BỊ CẮT; vi_thuần sập hoàn toàn
   (trung vị 64). Cổng này đo "PE-Core-L trong điều kiện triển khai của TA"
   (câu dài, tiếng Việt) — không phủ định PE trên benchmark câu ngắn của họ.
   Encoder ứng viên tương lai phải có text tower context ≥64 token và đo lại
   bằng đúng cổng này.
