# Quyết định encoder & TRAKE — tổng hợp ba lane, 03/09/2026

Tài liệu **quyết định**, viết bởi vai tổng hợp sau ba lane `trake-them`,
`pe-core`, `paper`. Mọi con số truy được về script + gốc hạt ghi trong
`docs/GT_TRAKE.md`, `docs/PRETEST_ENCODER.md`, `docs/PAPER_XEP_HANG_NOI_VIDEO.md`.

> Không mã video, không đáp án trong tài liệu này (`docs/` công khai).

**Khai báo bắt buộc trước khi đọc:** lượt **phản biện độc lập cho phiên này
KHÔNG chạy** (biến `critic` của bộ điều phối trả về rỗng). Mọi quyết định dưới
đây vì thế được chọn theo nguyên tắc *mặc định bảo thủ*: không ship thay đổi
sản xuất nào, không đốt khoản chi lớn nào; khoản chi duy nhất được duyệt là
một phép-đếm-trước ~$0,3. Ai phản biện sau phải đọc tài liệu này như bản
**chưa qua phản biện**.

---

## 0. Bốn quyết định

| # | câu hỏi | quyết định |
|---|---|---|
| **A** | Đốt GPU Colab dựng chỉ mục PE-Core? | **CHƯA GO — cấm đốt GPU cho tới khi cổng pre-test có số.** Cổng đang dở (encode 3/~40 video), KHÔNG phải cửa đóng. §1 |
| **B** | Ship `soft_order` cho TRAKE? | **KHÔNG SHIP — cửa ĐÓNG, có TEST đứng sau** (chênh +0,0000, 0/12 mục đổi điểm). Giữ `align_mode=ordered`, `min_gap=2` — đang là mặc định sản xuất, **không có việc tích hợp nào cần làm**. §2 |
| **C** | Hai hướng đáng thử nhất cho nhóm MỘT cảnh? | **② GQE paraphrase-ensemble** (ưu tiên 1) và **③ PRF Rocchio một bước** (ưu tiên 2, 0 đồng, chạy song song). Cả hai bắt đầu bằng **phép-đếm-trước**, không phải phép đo điểm. §3 |
| **D** | Việc kế tiếp của toàn dự án? | 1) chạy nốt pre-test PE (~2,5 h CPU, resumable); 2) hai phép-đếm-trước của §3; 3) đợt sinh thêm GT (`KE_HOACH_DINH_VI.md` §4.2b) **trước** mọi lần đọc TEST mới. |

---

## 1. Quyết định A — PE-Core: CHƯA GO, và vì sao không được ghi "NO-GO" vào bảng cửa đóng

### 1.1 Trạng thái đo được

- **Nền SigLIP đã chốt xong** theo quy ước toàn-video (132 mục, phụ lục §2b của
  `PRETEST_ENCODER.md`, viết TRƯỚC khi có bất kỳ số PE nào): MỘT cảnh trung vị
  **2,0** / hạng-1 **37,9%**; HAI cảnh trung vị **15,0** / hạng-1 **1,5%**.
- **PE: 0 con số.** Phiên lane bị cắt giữa giai đoạn encode; tiến trình nền
  (PID 16820) **đã chết**, cache theo video còn nguyên
  (`data/cache_pretest_pe/img/`: 3 video xong trên ~40, ~230 s/video).

### 1.2 Vì sao "CHƯA GO" chứ không phải "NO-GO"

Ghi NO-GO vào bảng cửa đóng đòi **kèm số** — mà số chưa tồn tại. Đóng cửa
không số chính là vi phạm luật đo #7 (kết luận âm phải *có số liệu* mới có giá
trị). Ngược lại, GO khi cổng chưa chạy là vứt bỏ chính cái cổng tiền-đăng-ký
đã dựng để chặn việc đốt 2–4 h GPU vô căn cứ. Trạng thái đúng duy nhất:
**cổng mở, đang đo dở, GPU bị khoá cho tới khi §5 của `PRETEST_ENCODER.md`
được điền.**

### 1.3 Bước tiếp theo — chính xác, theo thứ tự

```bash
# 1) chạy nốt encode (resumable theo video; ~37 video × ~230 s ≈ 2,4 h CPU)
python -u scripts/pretest_pe_core.py --giai-doan encode
# 2) chấm + bootstrap + đọc GO/NO-GO theo ngưỡng ĐÃ chốt ở §2b
python -u scripts/pretest_pe_core.py --giai-doan cham
```

Ràng buộc thi hành:

1. **Không đổi ngưỡng sau khi nhìn số.** Ngưỡng dịch đã chốt trước ở
   `PRETEST_ENCODER.md` §2b: (i) trung vị HAI cảnh của PE ≤ ½ trung vị SigLIP
   trên cùng 20 mục, HOẶC (ii) hạng-1 MỘT cảnh của PE ≥ SigLIP cùng mẫu +12 pp.
   Đạt một trong hai → GO; trượt cả hai → NO-GO, **lúc đó** mới ghi bảng cửa
   đóng kèm số.
2. **RAM:** không nạp `KISEngine` cùng tiến trình với model PE (os error 1455
   đã tái hiện ổn định); phép đo nặng RAM chạy song song phải qua
   `scripts/chay_gon_ram.py`.
3. Nếu GO: bước kế **không** phải Colab ngay — viết `notebooks/index-pe-core.ipynb`
   theo mẫu `build_siglip2_index_colab.py`, và nhớ pre-test **chưa đo tìm-video**
   (chỉ đo trong video đúng): chỉ mục đầy đủ phải qua một phép đo tìm-video
   trước khi thay/trộn encoder.
4. **Đọc kỳ vọng cho đúng chỗ ăn:** `dem_bao_hoa_noi_video.py` (03/09) cho thấy
   nhóm HAI cảnh trượt *hệ thống* vì văn bản khớp cảnh A (lệch trung vị 752
   frame) — encoder mới **không tự sửa** nhóm này; lever hoán-vị-cảnh-B đã ship
   mới là thứ sửa nó. Giá trị thật của PE nằm ở tiêu chí (ii) — nhóm MỘT cảnh,
   nơi mọi tín hiệu xếp-lại 0 đồng đã chết và encoder thứ hai là hướng duy nhất
   chưa thử của tầng XẾP HẠNG 40%.

---

## 2. Quyết định B — TRAKE: soft_order KHÔNG ship; nhánh này giờ có bộ đo n=24

### 2.1 Số phận soft_order — đóng bằng TEST, đúng kỷ luật tiền-đăng-ký

Giả thuyết chọn trên TUNE (12 mục cũ, +3,7% ở ±6, không bao giờ âm trong mẫu);
TEST là 12 mục mới `data/gt_trake_test_moi.json`, đọc **đúng một lần**
(`scripts/do_soft_order_test.py`):

| | ±6 (quyết định) | ±10 | ±20 | video đúng |
|---|---|---|---|---|
| ordered/gap=2 (sản xuất) | 0,1863 | 0,2179 | 0,2923 | 11/12 |
| soft_order/gap=2 | 0,1863 | 0,2179 | 0,2923 | 10/12 |

Chênh **+0,0000 ở cả ba cửa sổ, 0/12 mục đổi điểm** — và soft_order **lật một
video từ ĐÚNG thành SAI**, mất 0 điểm chỉ vì mục đó vốn 0 điểm: một rủi ro thật
được may mắn che. Chuỗi hiệu ứng +0,0063 (TUNE) → +0,0032 (gộp n=24) → 0,0000
(TEST) là đúng chữ ký thổi phồng (luật đo #7). **Giữ nguyên sản xuất
`align_mode=ordered`, `min_gap=2` — không sửa dòng mã nào, không cần cổng hồi
quy nào vì không có thay đổi.**

Điều kiện mở lại (ghi để không ai thử lại rẻ tiền): một **tín hiệu mới về
chất** — không phải biến thể cấu hình căn chỉnh thứ n — kèm **TEST mới chưa bị
đọc**. `gt_trake_test_moi.json` đã cháy vai trò TEST cho cả họ giả thuyết
căn chỉnh; dùng lại nó làm TEST là tự chấm bài mình.

### 2.2 Cửa TRAKE đóng lần hai trên n=24 (xác nhận, không phải phát hiện mới)

| cửa | số n=24 (±6) |
|---|---|
| unordered | 0,1372 vs nền 0,1823 = **−24,7%** |
| min_gap 0–1 | 0,1646 = **−9,7%**, video đúng 21→20 |
| hedge video | video đúng hạng-1 = trong-top-3 = **21/24** ⇒ mọi dòng chia cho hạng 2–3 mất trắng theo cấu trúc; mọi mức chia đơn điệu giảm ở cả ba cửa sổ |

### 2.3 Hướng TRAKE còn mở — theo thứ tự trần

Bộ đo n=24 (24 video khác nhau, mỗi mục 3 sự kiện; nền 0,2275, ORACLE-MỐC
+107,8%, ORACLE-VIDEO +148,1%; phân rã: định vị sự kiện 72,8% / sai video 27,2%):

1. **Lưới bù trừ phi-đều theo độ bất định từng sự kiện** — chưa ai đo; trần
   ORACLE-VIDEO chỉ 0,5645 chỉ thẳng vào cách tiêu dòng, và đây là trục
   thuần-phân-bổ (0 model mới). Ứng viên đo đầu tiên khi quay lại TRAKE.
2. **Định vị sự kiện trong video** (72,8% khoảng cách) — cùng nghẽn với tầng 2
   của KIS; ăn theo kết cục pre-test PE (§1), không cần việc riêng.
3. **Tín hiệu xếp hạng video khác** cho 3/24 mục sai video — chia lại dòng
   trong top-3 hiện tại đã chứng minh vô nghĩa hai lần, cần tín hiệu mới thật.
4. **Sinh mục 4 sự kiện** — cả 24 mục hiện có đúng 3 sự kiện, đề thật có câu 4;
   máy sinh đã hỗ trợ `--provider openai`, chi phí ~$0,25/12 mục.

---

## 3. Quyết định C — nhóm MỘT cảnh: hai hướng, cả hai bắt đầu bằng PHÉP ĐẾM

Nguồn: lượt soát độc lập của lane paper (03/09) — bản đồ 4 trục **hội tụ** qua
hai lượt soát; trục ① (khử-hub NNN/QB-Norm) đã đi trọn ba tầng đo và **đóng
toàn trục** (V1 cổng đếm 51% < ngưỡng; V2 qua cổng sát nút 57% rồi đo đầy đủ
TEST −1,7%, P(≤0)=77,7%). Ba đề xuất còn sống: ② > ③ > ④.

**Hình dạng chỗ ăn — thước đo chung cho mọi phép-đếm-trước từ nay**
(`scripts/dem_bao_hoa_noi_video.py`): ở nhóm một cảnh, keyframe đáp án đã đứng
trung vị **hạng 2** nội-video (≤3 ở 68% câu). Bài toán không phải "tìm cho ra"
mà là **thắng trận tay đôi hạng 1↔2**. Thước quyết định: trên các câu đáp án
đứng hạng 2–3, tín hiệu mới phân xử đúng **≥62%** số trận hạng-1↔đáp-án thì
mới đi tiếp; thước "trung bình toàn pool" đã chứng minh vô dụng sáu lần.

### 3.1 Hướng 1 — ② GQE paraphrase-ensemble (majority-vote k=2)

Vì sao đứng đầu: +5,2 điểm R@1 zero-shot CLIP (GQE, arXiv:2408.07249); VIREO
textual-KIS median rank 12,08→6,33 kèm bộ lọc-nhất-quán 7 khía cạnh chống
drift; **ba đội đối thủ của chính giải này dùng nó năm 2025**. Đây là nguồn
*từ vựng mới* — đúng chỗ SigLIP nhạy bất thường — khác hẳn cắt-khúc (cùng từ)
và doc2query (phía tài liệu, đã đóng).

Kế hoạch đo:

1. **Phép-đếm-trước, ~$0,3, 0 lần đọc TEST:** gpt-5.2 sinh paraphrase cho 132
   câu (cache theo câu; mẫu gọi `kiem_neo_don_anh.hoi_openai`; Gemini đang cạn
   — nếu muốn dùng phải kiểm bằng một request nhỏ + xoay `_model_order`).
   **Soi tay ~20 paraphrase tiếng Việt trước khi đếm**; nếu lởm chởm, thêm bộ
   lọc-nhất-quán kiểu VIREO (~1 lượt LLM/câu). Đếm theo thước trận-tay-đôi:
   majority-vote(k=2) thắng ≥62% trận hạng-1↔đáp-án ở các câu đáp án hạng 2–3
   → đi tiếp; dưới → dừng, ghi bảng cửa đóng kèm số đếm.
2. **Chỉ khi (1) đạt:** đo đầy đủ theo 5 cổng, biến thể **(a) hẹp** trước
   (chỉ xếp lại nội-video trên pool nền — giữ được nhóm bất biến và so được
   với lịch sử); biến thể (b) động vào tầng sinh ứng viên phải báo cáo song
   song bộ 60 câu cũ làm đối chứng.
3. **Kỷ luật TEST:** nửa TEST của bộ 132 đã bị đọc **≥5 lần** — dồn lần đọc
   TEST của ② (và mọi đề xuất mới) **sau** đợt sinh thêm GT của
   `KE_HOACH_DINH_VI.md` §4.2b.

### 3.2 Hướng 2 — ③ PRF Rocchio một bước trên embedding ảnh (0 đồng, song song)

`q' = normalize(q + λ·mean(e_ảnh top-m khung thuộc video KHÁC))`, xếp lại
khung trong video theo `cos(q', e_i)`; **leave-one-video-out bắt buộc** (thiếu
nó phép đo thành vòng tự xác nhận). Cơ chế duy nhất còn sống vượt khe
text–image: so ảnh-với-ảnh nơi SigLIP sắc hơn chữ-với-ảnh; cùng họ có số
(SuperGlobal ICCV'23; WeiMoCIR — nhưng phần "pseudo" cho CLIP **chưa có số
văn liệu**, nên xếp sau ②).

Kế hoạch đo: phép-đếm-trước **0 API, 0 GPU** — quét λ ∈ {0,25; 0,5; 1},
m ∈ {3, 5, 10} trên embedding mmap, **chỉ đọc thước trận-tay-đôi ≥62%**, chưa
đọc điểm. Chạy song song với 3.1 được vì khác đường ống. Nếu đạt: 5 cổng, biến
thể hoán-vị-giữ-điểm (R@100 bất biến theo xây dựng). Kỳ vọng ghi trước: 0–5%;
giá trị chính là rẻ tuyệt đối.

### 3.3 Không chọn — ghi rõ để khỏi ai làm trước

- **④ crop max-pool:** bằng chứng chuyển miền yếu nhất (toàn bộ số MTA là phân
  loại ảnh), chi phí mỗi câu cao nhất, phép đếm của nó không 0 đồng. Chỉ đáng
  chạy nếu ②③ xong mà nhóm một cảnh còn ≥20% headroom xếp hạng.
- **Mọi biến thể của trục ① (khử-hub/DBNorm/DBSN):** toàn trục đã đóng bằng
  TEST 03/09. Các "ứng viên mới" của lượt soát đều quy về trục cũ theo cơ chế.
- Nếu cả hai phép đếm ②③ cùng trượt: kết luận đã viết sẵn trong
  `PAPER_XEP_HANG_NOI_VIDEO.md` §10d — không gian tín-hiệu training-free cho
  trục nội-video một cảnh **cạn theo văn liệu hiện có**; đầu tư còn lại thuộc
  encoder thứ hai (§1) và trục thứ tự dòng bằng tín hiệu ngoài-SigLIP.

---

## 4. Đã cập nhật `docs/QUY_TRINH_TU_DONG.md` §6

Cùng phiên này: thêm lever hoán-vị-cảnh-B vào bảng đã-ship; thêm trạng thái
TRAKE n=24 và pe-core đang dở; chuyển các cửa mới đóng (soft_order, unordered,
min_gap thấp, hedge video, toàn trục khử-hub, cặp thời gian) vào bảng cửa đóng
bộ MỚI; ghi khoá GPU Colab.

---

## 5. Điều chưa biết — trung thực

1. **PE-Core: chưa có một con số nào.** Mọi câu "encoder thứ hai là hướng duy
   nhất còn lại" hiện là *lập luận loại trừ*, không phải phép đo. Cổng §1.3 là
   thứ biến nó thành phép đo.
2. **Phiên này không có phản biện độc lập.** Ba lane tự báo số của mình; vai
   tổng hợp chỉ kiểm chéo được với tài liệu và cache trên đĩa (đã kiểm: cache
   pre-test 3 video khớp mô tả lane; cờ `--hoan-vi-canh-b` và mặc định
   `ordered` khớp mã sản xuất). Sai sót nội bộ của từng lane chưa có ai soi.
3. **Bộ đo TRAKE n=24 vẫn nhỏ và đồng nhất 3-sự-kiện** — chênh vài phần trăm
   giữa cấu hình không kết luận được; câu 4 sự kiện của đề thật chưa đo được;
   điểm chấm GT đến từ hai model (argmax nội-sự-kiện nên không đổi mốc, nhưng
   là nguồn nhiễu phải nhớ).
4. **Hai TEST đã mòn:** nửa TEST bộ 132 bị đọc ≥5 lần; 12 mục TEST TRAKE đã
   cháy cho họ giả thuyết căn chỉnh. Đợt sinh GT mới (§4.2b kế hoạch định vị)
   là điều kiện cho mọi lần đọc TEST sạch tiếp theo.
5. **Chất lượng paraphrase tiếng Việt của gpt-5.2 chưa ai soi** — và bộ đo là
   câu máy-sinh: paraphrase của câu máy-sinh có thể "dễ" hơn đề người viết.
   Phải ghi vào biên bản khi đọc số đếm của ②.
6. **Độ chính xác từng câu của cổng `gan_nhan_hai_canh` trên đề thật** — bất
   định lớn nhất của lever đã ship, vẫn chỉ có tổng 28/55=51%; bước soi tay
   dòng-1 (`KE_HOACH_DINH_VI.md` §1.6 bước 6) vẫn phải làm trước ngày thi.
7. **`P(đáp án đúng)` của kênh Q&A** (27% đề thật) vẫn chưa đo — ưu tiên số
   một của `KE_HOACH_DINH_VI.md` §4.1 chưa ai chạy; các quyết định ở đây không
   thay thế nó, chỉ xếp cạnh nó.
