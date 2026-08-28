# Đánh giá "tiền xử lý toàn kho" — tổng hợp 5 lane khảo sát + phản biện độc lập

> Viết ngày 28/08/2026, chuẩn bị VÒNG 2. Kho hiện tại: 873 video / 177.321 keyframe /
> 130,67 giờ, chỉ mục SigLIP-2 SO400M-384. Nền đo được: **0,3496 ± 0,0023** trên 60 câu
> ground truth (một lần tái lập gần đây ra 0,3480 ± 0,0020 và 0,3502 ± 0,0012 — cùng
> khoảng sai số, dùng chung một harness).
>
> Mọi con số đều kèm cách đo; chỗ nào là ước lượng có chữ **[PHỎNG ĐOÁN]**.
> Mọi phép đo điểm đều theo quy tắc harness bắt buộc: `ranked_hits` của
> `scripts/make_submission.py` (đường sản xuất), khoảnh khắc KHÔNG bám keyframe (bốc
> ngẫu nhiên trong khe giữa hai keyframe), nhiều họ hạt giống độc lập, chênh < 2 lần
> sai số = HÒA, công thức chấm từ `src/core/submission.py`.
> Một phản biện độc lập đã mở script, chạy lại từ cache và kiểm từng trích dẫn — kết
> quả kiểm ghi ở mục 3. Mã video thật trong tài liệu này được che thành `L··_V···`.

---

## 1. Đánh giá thẳng ý tưởng của đội

Ý tưởng gốc gồm 5 phần: (a) OCR hết kho, (b) API xử lý hết video, (c) sinh câu hỏi
tương tự để đánh chỉ mục, (d) lưu hồ sơ đặc trưng từng video, (e) ráp đề giả vào tập
duyệt/kiểm. Phán quyết từng phần:

### 1.1. Phần ĐÚNG

**(d) Hồ sơ đặc trưng đa kênh — ĐÚNG, trùng nguyên văn cách nhà vô địch làm.**
VISIONE (vô địch VBS 2024, thắng 4/7 task) tiền xử lý offline cho *từng keyframe*:
CLIP-LAION ViT-L/14 + CLIP2Video + ALADIN + GEM + 3 detector vật thể gộp 1.460 lớp +
màu chủ đạo, đánh chỉ mục Lucene (Surrogate Text Representation) + FAISS (đọc trực
tiếp preprint ICMR'23, phản biện xác nhận từng kênh). Lưu ý chiều ngược: **VISIONE
KHÔNG dùng caption/OCR/ASR làm kênh lõi.** Với bài của ta, hồ sơ đa kênh chỉ đáng
tiền nếu ưu tiên kênh cho tín hiệu **mức FRAME** — vì nghẽn đã đo là chọn frame trong
video đúng (trần nội-video +27,5%), không phải tìm video (recall 60/60 trong top-400).

**(e) Ráp đề vào kiểm — ĐÚNG NHẤT, chi phí gần 0.** Phân tích 12th VBS chỉ ra: khi
mọi đội đều có CLIP, yếu tố phân thắng bại là **kỹ năng phối hợp nhiều chế độ tìm +
giao diện duyệt tay** — không phải chất lượng model. Khớp với cấu trúc vòng thi của
ta: chỉ ~15/180 phút là máy chạy phụ thuộc câu hỏi, ~150 phút là việc người, trong đó
block soát tay 55 phút là nguồn điểm lớn nhất (mỗi câu kéo từ hạng 6–20 lên hạng 1 ăn
+0,4). Sinh đề giả + diễn tập bấm giờ là phần giống nhà vô địch nhất trong cả ý tưởng.

**(a) OCR hết — ĐÚNG CÓ ĐIỀU KIỆN.** Đây là kênh **DUY NHẤT trả về FRAME** thay vì
video, tức kênh duy nhất chạm vào 60% điểm mất ở vị trí frame bằng thông tin ngoài
encoder. Căn cứ: 46/60 khung đáp án có chữ đọc được. Cũng là chuẩn của chính giải
HCMC (LameFrames 2024, MERVIN và U-CESE 2025 đều có OCR tiếng Việt + Elasticsearch).
Điều kiện: chỉ 7/60 khung trùng từ *nội dung* câu hỏi, chữ phần lớn là đồ đạc của
kênh ("tuitreTV", logo, dòng chạy chân); qi=41 OCR đọc ra 39 ký tự về **một bản tin
khác** — OCR sinh khớp-sai-tự-tin. Nên: bắt buộc qua cổng hai chặng B1 (chặng 1 OCR
~16.000 khung của 60 video GT, đòi keyframe đúng vào top-3 ở ≥4/60 câu — không đạt
thì DỪNG), và kết quả chỉ chèn CUỐI danh sách hoặc làm tiên nghiệm bộ phân bổ,
**tuyệt đối không chèn hạng 0–2**.

### 1.2. Phần BẪY (kèm số)

**(b) "API xử lý hết video" — BẪY ở cả hai biến thể.**
- *Caption toàn kho:* $13,7–105 tùy model/nguồn giá, hoặc 9 ngày chiếm TRỌN quota
  Gemini miễn phí (2.500 gọi/ngày cả vòng xoay 5 model) — để tấn công một nhóm thất
  bại **không tồn tại** (video đúng đã trong top-400 ở 60/60 câu). Thực nghiệm trực
  tiếp (mục 3.2) đóng hẳn: caption-BM25 định vị nội-video **thua SigLIP 8/12 câu**,
  trung vị hạng 60,5 so với 18,5. Chi phí cơ hội vô hình: 9 ngày đó kênh VLM rerank
  sản xuất (+3,3% đã đo) không còn một gọi nào.
- *"Phân tích từng frame" nghĩa đen:* 11,76–12,18 triệu khung → ~$1.800 list/~$900
  Batch, hoặc 353 ngày SigLIP-2 CPU, hoặc 588 ngày quota miễn phí, kèm 0,4–2 TB đĩa —
  cho một hướng đã có bằng chứng "trích keyframe dày hơn cứu được **0/18 câu**".
  KHÔNG LÀM trong mọi kịch bản.
- Ngoài ra "API xử lý hết" không phải lợi thế độc quyền: U-CESE (HCMC 2025) đã làm
  đúng thế bằng Gemini caption từng keyframe — nó là chuẩn mới của giải, không phải
  vũ khí riêng.

**(c) "Sinh câu hỏi tương tự" (doc2query) — BẪY, đã đo tận nơi, kỳ vọng ≈ 0.**
Mức tăng +44% MRR@10 của docTTTTTquery là ở **text IR**; trong thi video CHƯA AI đánh
chỉ mục câu-hỏi-sinh-sẵn, và biến thể gần nhất (PolySmart TRECVID 2024, sinh biến thể
phía truy vấn) chỉ được ~+6%. Thực nghiệm của ta (mục 3.3): kênh hồ sơ có tín hiệu
thật (AUC 0,7294) nhưng thua SigLIP trực tiếp ở chính bài nó nhắm (6/20 vs 17/20
R@1), và mọi nhánh trộn không rò rỉ đều hòa/âm. Con số +8,3%/+44,5% ở nhánh trộn thô
là **rò rỉ đánh giá thuần túy** (chỉ 20/873 video có hồ sơ, video GT luôn nằm trong
đó). Lý do gốc: bài nó giải (recall video) đã giải xong từ trước.

### 1.3. Bảng đối chiếu với đội thắng VBS/TRECVID

| ý tưởng của đội | đội thắng có làm không | phán quyết cho ta |
|---|---|---|
| hồ sơ đặc trưng từng video | **CÓ** — cốt lõi VISIONE (đa model + detector + màu, Lucene+FAISS) | LÀM, ưu tiên kênh mức frame |
| ráp đề vào kiểm / duyệt tay | **CÓ** — yếu tố phân thắng bại theo phân tích 12th VBS | LÀM, chi phí ~0 |
| OCR hết | Chuẩn của các đội HCMC (PARSeq+ES); VISIONE thì KHÔNG dùng làm kênh lõi | LÀM CÓ ĐIỀU KIỆN (cổng ≥4/60) |
| API caption hết kho | U-CESE đã làm (không độc quyền); VISIONE không cần | KHÔNG LÀM (mục 3.2 đóng bằng số) |
| sinh câu hỏi để đánh chỉ mục | **CHƯA AI LÀM** trong thi video; query-side chỉ +6% | KHÔNG MỞ RỘNG (mục 3.3 đóng bằng số) |

---

## 2. Bảng chi phí 4 phương án tiền xử lý

Mọi giá Gemini neo vào 2.5-flash-lite ($0,10/$0,40 mỗi 1M token — hai nguồn độc lập
cùng xác nhận); giá 3.5-flash-lite (model mặc định repo) hai nguồn công khai **vênh
nhau gấp đôi** nên không dùng làm mỏ neo. Cảnh báo thời hạn: Google khai tử
2.5-flash-lite ngày **16/10/2026** — sau đó mọi chi phí Gemini dưới đây tăng 2,5–4 lần.

| phương án | tiền API | giờ máy | ngày trên bậc miễn phí | tải/đĩa | nhắm vào nhóm nào | phán quyết |
|---|---|---|---|---|---|---|
| **P1. easyocr toàn kho** | $0 | 709.284 giây-lõi = 8,2 ngày 1 lõi; **~24,6 giờ** chia 8 tiến trình | 0 (không dùng API) | tải ~29,2 GB ảnh gốc (~172 nghìn request); chỉ mục ra ~27 MB | nhóm 3 (keyframe ngoài 400 ứng viên) + nhóm 2; kênh duy nhất trả FRAME | **LÀM CÓ ĐIỀU KIỆN** — sau cổng B1 chặng 1 |
| **P2. Caption toàn kho (Gemini)** | ~$27 list / ~$13,7 Batch (2.5-fl); $52–105 nếu 3.5-fl | 28–37 giờ gọi tuần tự | **9 ngày**, chiếm trọn quota VLM | tải 29 GB; caption ra 40–90 MB | **không nhóm nào** (video đã 60/60) | **KHÔNG LÀM** |
| **P3. "Từng frame" nghĩa đen** | ~$1.808 list / ~$904 Batch | 353 ngày SigLIP-2 CPU; 68 ngày easyocr 8 tiến trình | **588 ngày** ≈ 1,6 năm | 0,4–2 TB; riêng embedding 54,2 GB | **không nhóm nào** (mật độ khung chưa bao giờ là nghẽn: 0/18) | **KHÔNG LÀM, mọi kịch bản** |
| **P4. doc2query 873 video** | $1–2 list / <$1 Batch | ~2,4 giờ gọi | ~2 ngày (≤70% quota/ngày) | tải 1,2–2,4 GB; ra 2–5 MB | cấp video — đã bão hòa; chỉ là bảo hiểm nhóm 2 trên đề thật | **KHÔNG MỞ RỘNG** (đã đo pilot ≈ 0, mục 3.3) |

Cách đo tóm tắt: 177.321 khung × 4 s/khung/lõi (`src/core/ocr.py:14`); 1.100
token/ảnh là số đo của repo (`src/core/vlm.py:20`); ảnh gốc trung bình ~169–175
KB/ảnh (HEAD ngẫu nhiên trên CDN, hai lần đo độc lập); số khung thật 12.178.717
(tổng theo fps từng video — phép nhân 470.428 s × 25 fps thiếu 3,6% vì 92/873 video
chạy 29,97–30 fps). Phản biện đã tái kiểm toàn bộ số nền ở bảng này và xác nhận đúng.

**Hai số lỗi thời cần biết khi dùng bảng:** (i) dòng "177k ảnh ≈ $9" trong
`docs/KIEN_TRUC_VA_HUONG_CAI_THIEN.md:211` thấp hơn thực tế ~3 lần — phải sửa thành
~$22–27 list / ~$11–14 Batch; (ii) con số "đĩa chỉ có 2,75% keyframe" đã lỗi thời
ngay trong ngày phản biện vì mirror đang chạy (229/873 video lúc kiểm) — ngân sách
"phải tải X GB" từ nay phải trừ phần mirror đã xong.

---

## 3. Kết quả 3 thực nghiệm + phán quyết phản biện

Phản biện độc lập đã (i) chạy lại mọi script từ cache — **0 gọi API mới**, (ii) kiểm
5 trích dẫn web nặng ký nhất, (iii) tự viết một phép đo đối chứng. Cả 4 lane có kết
quả đều **tái lập được 100% từng con số** — kỷ luật cache + harness chung biến phản
biện thành việc chạy lại 10 phút.

### 3.1. Thực nghiệm OCR — **CHƯA CÓ PHÁN QUYẾT**

Lane trả về rỗng tại thời điểm tổng hợp, nhưng **không chết**: phản biện xác nhận
tiến trình đang chạy thật (14 file cache OCR mới trong ngày, một tiến trình python ăn
865 giây CPU). **Không kết luận thay nó.** Khi có kết quả, phản biện nó riêng bằng
đúng bộ kiểm này, và không cho qua cổng B1 ≥4/60 chỉ vì các lane khác sạch.

### 3.2. Thực nghiệm caption cấp frame — **KHÔNG LÀM** (phản biện: TIN ĐƯỢC mức cao nhất)

Câu hỏi quyết định "caption có định vị nội-video tốt hơn SigLIP không?" — trả lời
**KHÔNG**, trên chính nhóm thất bại:

- Hạng nội-video của keyframe gần đáp án, 12 câu điểm nền = 0: SigLIP trung vị
  **18,5** vs caption-BM25 **60,5**; đối đầu caption thua **8/12** (thắng 2, hòa 2).
  Trên cả 14 câu có caption: 16,5 vs 71,5, thua 10/14.
- Nguyên nhân nhìn thấy trong dữ liệu: 60–85% khung của một video khớp ít nhất một từ
  của câu hỏi — caption mô tả **bối cảnh lặp lại** chứ không tách được khoảnh khắc,
  đúng cao nguyên đã giết VLM-chấm-điểm, chỉ chuyển từ ảnh sang chữ. Prompt ép tả
  hành-động-thoáng-qua không thoát được (trùng nguyên văn chỉ 4% nhưng trùng danh từ
  bối cảnh vẫn 60–85%).
- Điểm harness 60 câu: mọi biến thể trộn HÒA hoặc THUA (tốt nhất −0,2% HÒA, tệ nhất
  replace −3,3%). Trên 12 câu thất bại chỉ cứu 1 câu (qi=46, +0,105 câu đó ≈ +0,002
  thang 60 câu), đồng thời **phá một câu đang khỏe** (qi=12: 0,400 → 0,133).
- Ngân sách đã dùng: 199 gọi Gemini (trần 350), 1.578 khung mới, ~$0,20 giá list —
  chạy bậc miễn phí.

Phản biện: chạy lại toàn bộ, **khớp 100% từng số**, log API khớp từng token. Kết
luận: không có căn cứ bỏ $13,7–27 caption toàn kho. Điều kiện mở lại duy nhất: một
cách đối chiếu khác hẳn BM25-từ-vựng (vd nhúng caption bằng encoder văn bản) phải
thắng SigLIP về trung vị hạng nội-video **trên chính 14 video đã caption sẵn** trong
`data/captions_frame/` (0 gọi API thêm).

**Phát hiện phụ quan trọng nhất đợt này (phản biện xác nhận bằng phép đo độc lập):**
phép trộn "gom khối theo video" — đúng cách `xep_lai_trong_video` của
`experiment_sharp_rerank.py` — **tự nó** gây −37,8% dù không đổi thứ tự frame nào, vì
nó phá cấu trúc xen kẽ video của `ranked_hits`. Phản biện đo độc lập hoán vị gom-khối
thuần túy (w=0, không VLM, không đổi một điểm số nào): 0,3480 → 0,2146 = **−38,3%**,
và log lịch sử cho thấy mọi biến thể "trong video" rơi −34,7%…−37,3% gần như bất kể
trọng số. **Hệ quả: con số −34,7% "VLM chấm frame phá điểm" trong tài liệu nhiều khả
năng đo nhầm cơ chế** (lẫn artifact gom-khối). Kết luận "VLM không giúp" có thể vẫn
đúng, nhưng phải đo lại bằng hoán vị giữ-nguyên-slot trước khi đóng hồ sơ vĩnh viễn —
cache VLM còn nguyên, chi phí ≈ 0 gọi mới.

### 3.3. Thực nghiệm doc2query — **KHÔNG MỞ RỘNG** (phản biện: TIN ĐƯỢC, 2 dè dặt nhỏ)

Pilot 20 video (10 nhóm thất bại + 10 đang đúng), 20 gói Gemini, 100 câu hỏi tổng hợp:

- Tầng 1 — kênh có tín hiệu thật: cosine với hồ sơ đúng 0,5668 vs hồ sơ sai 0,4945,
  **AUC 0,7294**. Ý tưởng đúng về tín hiệu.
- Tầng 2 — nhận diện 20-chọn-1 KHÔNG rò rỉ: SigLIP trực tiếp R@1 **17/20** vs hồ sơ
  **6/20**; trộn w=0,05/0,1/0,2 không cải thiện (17/16/17).
- Tầng 3 — harness 60 câu: nền 0,3502 ± 0,0012; trộn canh giữa −0,4%/−0,8%/+0,8%
  (hòa theo ngưỡng 2×sd). Nhánh trộn thô +8,3% (60 câu) và +44,5% (20 câu có hồ sơ)
  là **rò rỉ thuần túy** — chỉ video GT được lập hồ sơ.

Phản biện: chạy lại measure từ cache, mọi số khớp. Hai dè dặt không đổi kết luận:
(i) theo đúng quy tắc hòa của chính script, nhánh canh giữa w=0,2 (+0,0028 > ngưỡng
0,0024) lẽ ra đọc là "thắng-nhẹ-cận-trên" chứ không phải hòa; (ii) đối chứng công
bằng cho phép trộn là nhánh w=0 (−0,8%) chứ không phải thứ tự cache. Vì mọi nhánh vẫn
là CẬN TRÊN có rò rỉ, "KHÔNG MỞ RỘNG" đứng vững. Điều kiện mở lại duy nhất: vòng sau
kho lớn hơn nhiều và đo lại thấy recall video < 100% trong top-400. Giữ 20 gói câu
hỏi đã cache làm bộ đề KIS tổng hợp cho diễn tập.

### 3.4. Một sub-claim bị phản biện bắt SAI (lane cửa sổ 3 tiếng)

Câu "mirror chính là input của OCR toàn kho / OCR chạy trên mirror đã tải" **SAI so
với mã hiện tại**: `src/core/ocr.py` tải ảnh **gốc 1280×720** từ CDN (không đọc
`data/frames`), số 4 s/khung đo trên ảnh gốc, và chất lượng easyocr trên bản 512×288
**chưa ai đo** — docstring mirror chỉ bảo hành "đủ cho soát mắt và VLM". Đây là biến
số quyết định ngân sách chênh nhau ~5 lần (tải 29 GB gốc vs dùng ~6 GB mirror) — cách
đo ở mục 5. Hai con số 30 GB vs 5,5 GB của hai lane **đều đúng** nhưng đo hai thứ
khác nhau (dây tải ảnh gốc vs đĩa lưu bản 512px).

---

## 4. KẾ HOẠCH VÒNG 2

### 4.1. LÀM TRƯỚC NGÀY THI — xếp theo điểm kỳ vọng

| # | việc | điểm kỳ vọng | chi phí | điều kiện / cổng | nguồn số |
|---|---|---|---|---|---|
| 1 | **A1 — bộ phân bổ PHỦ XÁC SUẤT** thay `allocate_hybrid_rows` | **+10,0% đã đo** (0,3496→0,3845; hy vọng trên +28% chưa xác nhận) | 4–6 giờ người + **port JavaScript `build_review_page.py`** (chặng khó nhất, không được bỏ) | cổng ba-mô-hình-đáp-án: không âm ở cả ba, dương ≥8% ở hai mô hình đầu | `experiment_phu_xac_suat.py`, HUONG_DI_TIEP §A1 |
| 2 | **B3 — nối chỉ mục batch 2, GIỐNG HỆT cách cũ** | 0 tăng — **tránh SỤP** (video không vector = phủ 0 tuyệt đối) | [PHỎNG ĐOÁN] 2–4 giờ GPU T4/177k khung | chờ BTC công bố link — ưu tiên tuyệt đối khi có; hồi quy: batch-1-only phải trùng 0,345 ± 0,0014; CẤM đổi encoder cùng lúc | HUONG_DI_TIEP §B3 |
| 3 | **Mirror 177.321 keyframe 512px về đĩa (~5,5–6 GB)** | 0 điểm trực tiếp — bảo hiểm block soát 55' (nguồn điểm lớn nhất, trần nội-video +27,5%) + VLM +3,3% | vài giờ tải; **đang chạy** (229/873 lúc kiểm) | vô điều kiện; đồng thời là phòng thủ cho rủi ro token HF ghi chưa thu hồi | lane cửa sổ 3 tiếng + phản biện |
| 4 | **B1 — OCR: chặng 1 rồi mới toàn kho** | [PHỎNG ĐOÁN] +2% đến +5%, sai số rất rộng (bằng chứng 7/60) | chặng 1: nửa ngày; toàn kho: ~24,6 giờ máy 8 tiến trình + tải (xem đối chứng 512px, mục 5) | **cổng ≥4/60** keyframe đúng vào top-3 BM25 — không đạt thì DỪNG; kết quả chỉ chèn CUỐI hoặc tiên nghiệm B2, cấm hạng 0–2 | HUONG_DI_TIEP §B1; lane OCR đang chạy |
| 5 | **A2 — vá lỗi ứng viên lời thoại bị chấm rồi vứt** (`vlm_rerank_run.py:283/295`), chèn CUỐI | 0,0000 trên 60 câu (đã đo) — bảo hiểm 1–2 câu/vòng trên đề thật | 30–45 phút | không giảm quá 1 sd (0,0007); ĐỪNG chèn giữa | HUONG_DI_TIEP §A2 |
| 6 | **Đo lại VLM-rerank bằng hoán vị GIỮ-SLOT** | 0 điểm trực tiếp — làm sạch hồ sơ −34,7% có thể đo nhầm cơ chế (artifact gom-khối tự gây −38,3%) | ~1 giờ, 0 gọi API (cache VLM còn nguyên) | chỉ sửa tài liệu nếu kết quả đổi dấu; không đưa gì vào đường thi | mục 3.2 + phản biện |
| 7 | **A3/A5/A4/A6 — vệ sinh**: thu hồi token HF (5'), sao lưu lab khỏi thư mục tạm (10'), xóa số 24,9% sai, ghi 3 phán quyết đóng hướng **+ sửa dòng "$9" thành ~$22–27** | 0 điểm — chống mất mát và chống quyết định dựa trên số sai | ~2 giờ tổng | không có lý do để không làm | HUONG_DI_TIEP §A3–A6 + mục 2 |
| 8 | **B2 — tín hiệu phụ thành TIÊN NGHIỆM bộ phân bổ** | [PHỎNG ĐOÁN] +2% đến +5%, hoàn toàn chưa đo | 3–4 giờ người + 2 giờ máy | **chỉ SAU khi A1 vào sản xuất**; quét từng hệ số một; với lời thoại/OCR tiêu chí là "không âm" | HUONG_DI_TIEP §B2 |
| 9 | **Diễn tập một vòng đầy đủ có bấm giờ** + sáng thi bắn 1 lệnh VLM dò model sống + đăng nhập/nộp thử sẵn | 0 điểm trực tiếp — con số "build review ~2 phút" là ước lượng duy nhất chưa đo trong cả bảng ngân sách | 1 buổi | pytest xanh trước khi diễn tập | lane cửa sổ 3 tiếng, QUY_TRINH_NOP.md |

**KHÔNG LÀM (đóng bằng số, ghi vào tài liệu):** caption toàn kho (hai lane độc lập —
chi phí và thực nghiệm — cùng kết luận KHÔNG bằng hai con đường không chung giả định;
sự hội tụ này mạnh hơn từng lane); "từng frame" nghĩa đen; mở rộng doc2query ra 873
video; và toàn bộ danh mục §5 của HUONG_DI_TIEP.md (nộp dày hơn, thêm keyframe, hợp
nhất CLIP, họ DETR, đổi encoder bây giờ, cộng điểm theo frame…).

### 4.2. LÀM TRONG 3 TIẾNG THI (25 câu)

Đường găng đã đo: ~15 phút máy phụ thuộc câu hỏi + ~150 phút việc người + ~5 phút đệm.
Precompute OCR toàn kho chỉ rút đường găng ~5–10 phút danh nghĩa (OCR ứng viên vốn
chạy NỀN phút 6–35), tối đa ~25 phút ở kịch bản mạng xấu — giá trị thật của nó là
`search_ocr` phủ 177k khung thành kênh TÌM trả FRAME, và +10 phút dồn vào soát tay
(≈ 13 lượt soát câu thêm ≈ có thể thêm vài lần +0,4).

| phút | việc | ai |
|---|---|---|
| 0–3 | nhận đề, dán 25 câu | người |
| 3–5 | `make_submission` (90 giây) | máy |
| 5–9 | verify + **NỘP LẦN 1** (bảo hiểm) | người |
| 6–35 | OCR ứng viên chạy NỀN (bỏ được nếu OCR toàn kho đã precompute) | máy nền |
| 10–25 | viết `.en.txt` (dịch tay 15') | người |
| 25–40 | VLM rerank ~10' (~135 req/vòng; quota 500 req/ngày/model ≈ 3 vòng/ngày) | máy |
| 40–42 | answer Q&A (1') + build trang soát (~2', **chưa bấm giờ**) | máy |
| 45–100 | **SOÁT TAY 55'** — 4 người × ~6 câu, nhịp ~3'/người/câu | người |
| 100–110 | áp picks + **NỘP LẦN 2** | người |
| 110–160 | đào sâu 50' các câu còn yếu | người |
| 160–175 | **NỘP CUỐI** | người |

Mọi thứ query-independent còn lại đã precompute sẵn từ trước: chỉ mục SigLIP-2
(offline hoàn toàn), transcript 849/873 (tra mili-giây), model cache. Có mirror thì
soát + OCR + VLM chạy offline hoàn toàn — chỉ còn cần mạng cho Gemini/YouTube/trang
nộp (đã có phương án 4G).

---

## 5. Những con số còn thiếu và cách đo

1. **Chất lượng easyocr trên ảnh 512×288 vs gốc 1280×720** — quyết định ngân sách
   chênh ~5 lần (dùng mirror ~6 GB vs tải 29 GB gốc). Đo: ~50 khung có chữ, chạy
   easyocr cả hai bản, so văn bản đọc được, ~1 giờ máy. **Phải đo TRƯỚC khi chạy OCR
   toàn kho** — hiện `ocr.py` tải ảnh gốc từ CDN, đừng hành động theo câu "OCR chạy
   trên mirror" cho tới khi có số này.
2. **Kết quả lane OCR đang chạy + cổng B1 chặng 1 (≥4/60)** — đang chạy, chưa có
   phán quyết. Không OCR cả kho trước khi có.
3. **VLM-rerank đo lại bằng hoán vị giữ-slot** — con số −34,7% hiện hành nhiều khả
   năng lẫn artifact gom-khối (−38,3% tự thân, đo độc lập 2 lần). Cache còn nguyên,
   ~1 giờ, 0 gọi API.
4. **Phân bố đáp án thật của BTC trong ô keyframe** — KHÔNG đo được (56/60 frame GT
   nằm đúng trên keyframe → zero bằng chứng trong ô). Toàn bộ biên độ +9% ↔ +28,5%
   của A1 treo trên giả định này; giảm rủi ro duy nhất là cổng ba-mô-hình-đáp-án.
5. **Hệ số truyền 60 câu GT → bảng xếp hạng** — chưa ai lập được ánh xạ (5,8 → 8,6
   trên bảng không khớp tuyến tính với điểm nội bộ). Mọi con số "quy sang bảng" là
   [PHỎNG ĐOÁN].
6. **Thời gian build trang soát (~2')** — ước lượng duy nhất trong bảng ngân sách
   vòng thi; bấm giờ trong buổi diễn tập.
7. **Thời gian GPU đánh chỉ mục batch 2** — [PHỎNG ĐOÁN] 2–4 giờ T4/177k khung,
   notebook không lưu execution output; bấm giờ lần chạy thật. Kích thước batch 2:
   BTC chưa công bố.
8. **Giá gemini-3.5-flash-lite** — hai nguồn công khai vênh nhau gấp đôi
   ($0,15/$1,25 vs $0,30/$2,50); model mặc định của repo hiện không có giá đáng tin.
   Nếu còn định tiêu tiền Gemini, làm trước **16/10/2026** khi 2.5-flash-lite còn
   $0,10/$0,40, và đo token ra thực tế của một mẻ nhỏ (số 100 token/caption hiện là
   ước, chưa đo).
9. **Độ bền cache OCR theo thời gian** — đơn giá đĩa đã trôi 153 → 213 byte/khung
   giữa hai lần đo; kiểm lại khi cache lớn để ước đĩa đầu ra toàn kho.

---

*Nguồn: 5 lane khảo sát 28/08/2026 + phản biện độc lập cùng ngày (tái lập 100% từ
cache); docs/HUONG_DI_TIEP.md 24/08; docs/KIEN_TRUC_VA_HUONG_CAI_THIEN.md;
docs/QUY_TRINH_NOP.md. Mã video thật đã che; không tài liệu nào trong repo được phép
chứa API key.*
