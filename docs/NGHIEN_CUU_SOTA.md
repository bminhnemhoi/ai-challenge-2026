# Khảo sát SOTA 2025–2026 và kế hoạch vượt nhóm đầu

Chốt ngày 30/08/2026. Nguồn: 5 mũi khảo sát web độc lập (giải pháp các đội top
của **chính giải này**, hệ thống thắng VBS/TRECVID, embedder 2025–2026, định vị
thời gian, OCR/ASR tiếng Việt) → **33 đề xuất, mỗi cái có URL nguồn thật đã đọc**
→ 2 phản biện độc lập chấm (khả thi × tác động, đối chiếu 9 cửa đã đóng) + 1
phản biện dò thiếu sót (6 hướng bổ sung).

**Bối cảnh xếp hạng:** vòng 2 ta được 10,0 — top-1 được 15. Khoảng cách 5 điểm
trên 30 câu. Tài liệu này xếp ưu tiên theo **chỗ điểm đang chảy máu thật**, không
theo độ mới của paper.

---

## 0. Chẩn đoán: 5 điểm thiếu đang nằm ở đâu

| nguồn mất điểm | bằng chứng | trần thu hồi |
|---|---|---|
| **Q&A trả lời sai** | vòng 2: **4/9 câu Q&A sai đáp án** ở chế độ tự động; sửa tay bằng gpt-5.2 ảnh gốc mới đúng | **~4 điểm** — mỗi câu Q&A sai = 0 tuyệt đối dù khung hình đúng |
| Truy xuất trượt video | 6 câu nghẽn: keyframe đáp án hạng nội-video 95–276 dưới mọi tín hiệu rẻ | ~2–3 điểm, đắt |
| TRAKE dưới trần | `docs/CHAN_DOAN_TRAKE.md` | ~1 điểm, cần GT TRAKE |

Kết luận thẳng: **hướng đắt nhất (đổi encoder) KHÔNG phải hướng đáng làm trước.**
Một câu Q&A có khung hình đúng mà đáp án sai vẫn là số 0 tròn trĩnh — đó là chỗ
duy nhất trong hệ thống mà một ngày công có thể đổi lấy vài điểm.

---

## 1. TOP 5 việc đáng làm, xếp theo (tác động × khả thi)

### ① Q&A đa kênh + ảnh gốc thành mặc định — *đang triển khai*

**Nguồn:** NII-UIT tại VBS 2026 (á quân 2026, vô địch 2025),
"Towards Effective Visual Question Answering for Interactive and Multimodal Video
Retrieval" — https://link.springer.com/chapter/10.1007/978-981-95-6963-2_26 ;
kèm phản biện dò-thiếu-sót (hardening đường trả lời).
**Điểm phản biện:** khả thi 5,0 / tác động 3,5 — cao nhất bảng 33 đề xuất.

**Làm gì:** `answer_qa.py` hiện gửi 12 thumbnail 512px cho Gemini một lượt.
Đổi thành: (a) **ảnh gốc ~1900px** (đúng thao tác tay đã cứu 4 câu vòng 2);
(b) nhét **lời thoại ±30 giây** quanh khung hình (kho `data/captions/` có sẵn
**873/873 video**); (c) prompt ép **danh từ cụ thể nhất** + bắt trích nguồn
(nhìn thấy / nghe thấy / đọc thấy) để người soát kiểm nhanh; (d) **tự nhất quán**
trên khung lân cận, bất đồng thì gắn cờ ⚠ để ưu tiên soát tay.

**Đo:** độ chính xác đáp án trên **cả 60 câu GT** (mọi câu đều có
`vqa_question` + `vqa_answer`), chia TUNE/TEST 30/30 chẵn-lẻ như luật đã ship,
dùng bộ so khớp `_default_answer_match` (rộng lượng — chỉ có thể đếm thừa,
không đếm thiếu). Đây là độ đo **đáp án**, không phải R@k — hợp lệ vì luật 2.1.2
chấm cả trường đáp án.

**Chi phí:** 0,5–1 ngày người; Gemini free-tier + vài đô gpt-5.2. Rủi ro thấp:
không đụng đường truy xuất hay allocator.

### ② Mở rộng harness GT — cái chặn mọi thứ còn lại

**Nguồn:** phản biện dò-thiếu-sót (không phải paper — là kỷ luật đo lường).
**Vì sao đứng thứ hai dù không ăn điểm trực tiếp:** ngưỡng 2σ trên 30 câu TEST
hiện tại đòi chênh ~+5% mới kết luận được. Phần lớn trong 33 đề xuất còn lại
ăn 1–3 câu — **không bao giờ qua nổi cổng** với bộ 60 câu. Bộ 120 câu hạ ngưỡng
xuống ~1,4 lần.

**Làm gì:** (1) thu hoạch đề thật vòng 1 + vòng 2 đã chốt bằng mắt (24 + 30 câu —
đây là các câu DUY NHẤT chắc chắn khớp phân bố ra đề của BTC); (2) viết thêm
30–40 câu phân tầng theo loại đang thiếu (tên riêng/địa danh → kênh lời thoại;
chữ trên màn hình → kênh OCR); (3) dựng **5–8 câu TRAKE có 4 mốc chốt tay** —
việc này mở khoá toàn bộ nhánh TRAKE đang bị chặn vì không có bộ đo;
(4) **đóng băng** chia TUNE/TEST trước khi chạy bất kỳ thí nghiệm nào.

**Chi phí:** 2–3 ngày người, phần lớn là chốt mắt trên `review.html` đã có.

### ③ Truy vấn cặp thời gian (temporal pair scoring)

**Nguồn:** vibro (vô địch VBS 2022+2023) —
https://files.visual-computing.com/research/Vibro_Video_Browsing_with_Semantic_and_Visual_Image_Embeddings.pdf ;
vitrivr — http://lucaro.ch/papers/ICME20_vitrivr.pdf. Kỹ thuật chung của **mọi**
hệ thống thắng VBS. **Điểm:** khả thi 4,5 / tác động 3,5.

**Làm gì:** lớp dịch đã gọi LLM sẵn — bắt trả thêm `{có_2_cảnh, cảnh_A, cảnh_B}`.
Câu có cổng bật: `s_temp(i) = HM(s_A(i), max_{j cùng video, i<j≤i+W} s_B(j))`,
W ∈ {2,3,5,8} keyframe. Câu không có cấu trúc hai cảnh **giữ nguyên 100%** đường cũ.
Tác động kép: nâng video có đủ cả hai cảnh, và kéo keyframe A lên hạng nội-video.

**Đo:** gắn nhãn 60 câu xem bao nhiêu câu có cấu trúc hai cảnh (60 request free);
quét (W, HM/tích, λ) trên TUNE, chấm một lần TEST, luật 2σ. Nếu số câu qua cổng
< 8 thì không đủ lực thống kê — báo cáo per-query thay vì ép kết luận.

**Chi phí:** 0,5–1 ngày; ~100 request Gemini free.

### ④ Ensemble RRF hai tầng + chỉ mục thứ hai (PE-Core)

**Nguồn:** Vortex 2026 (fuse CLIP + SigLIP2 bằng RRF) — https://arxiv.org/html/2606.19682 ;
Perception Encoder, Meta NeurIPS 2025 — https://arxiv.org/abs/2504.13181 ,
trọng số https://huggingface.co/timm ; bài học 11 kỳ VBS —
https://link.springer.com/article/10.1007/s00530-023-01143-5.
**Điểm:** khả thi 4,0 / tác động 3,0 (fusion) và 4,0/3,0 (PE-Core).

**Làm gì:** dựng chỉ mục thứ hai bằng PE-Core-L14-336 (Colab free ~2–4 h T4)
song song SigLIP-2 hiện có, rồi hợp nhất **RRF hai tầng tách bạch** — tầng video
và tầng keyframe nội-video — vì bảng đo nội bộ chứng minh hai tầng phản ứng
**ngược nhau** với cùng tín hiệu. Đầu ra giữ nguyên schema `ranked_hits` để bộ
phủ xác suất nuốt nguyên, không đụng allocator.

**Cổng rẻ trước khi tốn GPU:** embed vài trăm keyframe của 6 video nghẽn bằng
PE trên CPU, đo hạng nội-video của keyframe GT so bảng SigLIP (95–276). Kéo được
≥3/6 câu vào top-20 mới đáng index toàn kho.

**Khác cửa đã đóng #8** (CLIP-B32 z-blend vô dụng): ở đó model thứ hai *yếu*
(B32, 2021) và trộn ở tầng điểm; ở đây model thứ hai *ngang cơ hoặc mạnh hơn* và
hợp nhất ở tầng **hạng** (RRF bất biến thang đo). Phải nói rõ điều này khi báo cáo
kết quả, và nếu RRF vẫn hoà thì ghi vào bảng cửa đóng — không diễn giải lại.

### ⑤ Kênh hợp-ứng-viên OCR/ASR/tiêu-đề (không cộng điểm, chèn đuôi)

**Nguồn:** U-CESE (chung kết chính giải này 2025) — https://arxiv.org/html/2605.23274v1 ,
"jointly retrieves timestamps from VisualDB and TextualDB, and merges them into a
single list" — **hợp danh sách, KHÔNG cộng điểm**; MADTempo/AIO —
https://arxiv.org/abs/2512.12929. **Điểm:** khả thi 5,0 / tác động 2,5.

**Làm gì:** (a) vá A2 (`vlm_rerank_run.py:283/295`) để ứng viên do lời thoại tìm ra
không bị vứt; (b) pool VLM = top-24 SigLIP ∪ top-5 BM25 lời thoại ∪ top-5 BM25 OCR
∪ top-3 BM25 tiêu đề; (c) ứng viên chỉ-do-text-tìm-ra nhận dòng ở **đuôi** danh
sách 100. **Tuyệt đối không viết công thức trộn điểm SigLIP + BM25** — bằng chứng
nội bộ nói đừng; nếu buộc phải trộn hạng thì RRF k=60.

**Vì sao an toàn:** R@k là *max* trên tiền tố ⇒ thêm ứng viên vào đuôi không bao
giờ hại. Tiêu chí nghiệm thu vì thế là **KHÔNG ÂM** trên TEST (không tụt quá 1 sd),
cộng chỉ số chẩn đoán riêng: đếm số câu mà pool mở rộng chứa video đúng còn
top-24 thuần thì không.

---

## 2. Bảng đầy đủ 33 đề xuất (điểm = khả thi × tác động, trung bình 2 phản biện)

| điểm | lane | đề xuất | kết luận |
|---|---|---|---|
| 17,5 | VBS | Q&A đa kênh khi sinh đáp án (NII-UIT) | **TOP ①** |
| 17,5 | đội VN | review.html: filmstrip, gộp theo video, phím Tab ghi mốc TRAKE | làm cùng ②, không đo được bằng R@k |
| 15,8 | VBS | truy vấn cặp thời gian | **TOP ③** |
| 14,0 | đội VN | neo truy vấn bằng ảnh web (image-to-image) | ứng viên số 6 — nhắm đúng 6 câu nghẽn |
| 12,5 | OCR/ASR | kênh hợp-ứng-viên cross-video | **TOP ⑤** |
| 12,5 | VBS | trình xác nhận chuỗi cảnh liên tiếp cho operator | gộp vào nâng cấp review.html |
| 12,5 | đội VN | OCR lại bằng VLM thay OCR cổ điển | sau khi có kho OCR |
| 12,2 | định vị | Gemini video-native grounding qua YouTube URL | ứng viên mạnh cho 6 câu nghẽn — cổng triage nửa ngày |
| 12,0 | embedder | ensemble RRF/z-fusion 400 ứng viên | **TOP ④** |
| 12,0 | embedder | PE-Core thay/ghép trunk | **TOP ④** |
| 10,5 | embedder | FG-CLIP 2 so400m | dự bị cho ④ |
| 10,0 | OCR/ASR | lọc "đồ đạc của kênh" theo tần suất tài liệu | van bắt buộc nếu làm OCR toàn kho |
| 10,0 | VBS | RRF nhiều embedder ở tầng sinh ứng viên | trùng ④ |
| 10,0 | đội VN | chấm theo độ phủ sub-query (U-CESE) | liên quan ③ |
| 9,0 | định vị | TRAKE neo bằng mốc thời gian liên tục | chờ GT TRAKE (②) |
| 9,0 | VBS | viết lại truy vấn bằng LLM + hợp nhất hạng 4 danh sách | rẻ, làm cùng ③ |
| 8,8 | OCR/ASR | OCR là tín hiệu cấp đoạn-tin, bơm Gauss vào tiên nghiệm | thiết kế đúng cho ⑤ |
| 8,8 | OCR/ASR | PP-OCRv5 mobile hạ giá OCR toàn kho 25 h → 3–6 h | tiền đề của ⑤ mở rộng |
| 8,8 | embedder | mexma-siglip2 (tiếng Việt trực tiếp) | thành viên ensemble |
| 8,8 | đội VN | TRAKE kiểu MADTempo (phân rã LLM, neo 2 biên) | chờ GT TRAKE |
| ≤8,0 | — | 13 đề xuất còn lại | xem `scratchpad/tatca.json`, không đủ ưu tiên vòng này |

### Kết luận ÂM có giá trị (đóng cửa, khỏi tốn công)

- **Bỏ hẳn nhánh DETR chuyên dụng** (Moment-DETR, QD-DETR, CG-DETR, UniVTG,
  SG-DETR): huấn luyện trên domain khác, không có nhãn tiếng Việt, chi phí GPU lớn.
- **SigLIP2 g-opt và EVA-CLIP-18B KHÔNG đáng re-index** — có số liệu, chặn hai
  đường tốn kém nhất.
- **ĐỪNG re-transcribe 130,67 giờ để "sửa mốc thời gian"**: lệch 2.850 frame là
  lệch *ngữ nghĩa*, không phải lệch đồng bộ.
- **OCR toàn kho bằng API: KHÔNG** — giữ quota API cho đọc-ảnh-lúc-trả-lời
  (chính là ①, nơi API đáng tiền nhất).

---

## 3. Sáu hướng phản biện dò ra mà cả 5 mũi khảo sát bỏ lỡ

1. **Mở rộng harness GT** → đã lên TOP ②.
2. **Hardening đường trả lời Q&A** → đã lên TOP ①.
3. **Kênh tiêu đề/metadata video của BTC** làm nguồn ứng viên (873 dòng, BM25
   vài phút CPU) — cửa bị bỏ quên hoàn toàn; gộp vào ⑤.
4. **TRAKE khai thác điểm TỪNG PHẦN**: lưới bù trừ *phi đều* theo độ bất định
   từng sự kiện + hedge video theo chênh lệch điểm — chính là mở rộng tư duy
   "phủ xác suất" đã ship cho KIS sang TRAKE. Chặn bởi ② (cần GT TRAKE).
5. **Bản đồ video song sinh** (near-duplicate) từ chính ma trận SigLIP có sẵn:
   chẩn đoán rẻ nửa ngày (đếm câu có video GT thuộc cụm song sinh) trước khi
   đụng allocator.
6. **Kênh vận hành nhiều người**: chia trang soát theo người, `merge_picks.py`
   có kiểm xung đột, thiết kế lượt nộp để **delta điểm quy được về nhóm nào**
   (lượt 2 = nhóm độ-tin-cao, lượt 3 = phần còn lại) — biến ba lượt nộp thành
   ba phép đo thay vì ba lần đoán.

---

## 4. Trình tự triển khai đề nghị

```
TUẦN 1  ① Q&A đa kênh + ảnh gốc      (đo được ngay, ăn điểm thật lớn nhất)
        ⑤ vá A2 + mở pool ứng viên   (rẻ, rủi ro ~0, chạy song song)
TUẦN 2  ② mở rộng harness GT         (chặn mọi thứ sau; chia được cho nhiều người)
        ③ truy vấn cặp thời gian     (đo trên harness mới, mạnh hơn)
TUẦN 3  ④ PE-Core + RRF hai tầng     (cổng rẻ 6 câu nghẽn trước khi tốn GPU)
        Gemini grounding (triage)    (tầng A nửa ngày, dừng nếu lệch > 3 s)
KHI CÓ BATCH 2: dựng lại chỉ mục GIỐNG HỆT, rồi chạy lại cả 2 fold
        (luật sau merge trong SHIP_PHU_XAC_SUAT.md) trước khi giữ tham số
```

**Quy tắc bất di bất dịch:** mỗi hạng mục chỉ được vào sản xuất sau khi qua cổng
TUNE/TEST với luật hoà 2σ, và mọi con số vào tài liệu phải có file cache đứng sau.
Cửa đóng thì ghi vào bảng tín hiệu `KIEN_TRUC_VA_HUONG_CAI_THIEN.md` — không
diễn giải lại một kết quả hoà thành "có tiềm năng".
