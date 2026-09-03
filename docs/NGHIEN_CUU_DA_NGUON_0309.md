# NGHIÊN CỨU ĐA NGUỒN 03/09 — TỔNG HỢP 4 LANE + PHẢN BIỆN

> **Nguồn gốc tài liệu**: tổng hợp từ nghiên cứu đa nguồn ngày 03/09 (4 lane song song:
> `vbs-doi-thu`, `phan-bo-dong`, `trake-dinh-vi`, `qa-nhan-tu`), tổng cộng 18 đề xuất trên
> ~30 nguồn đã đọc thật (WebFetch toàn văn/abstract/README, có 4 PDF đọc trực tiếp).
> **Các verdict GIỮ/NGHI_NGỜ là của lượt phản biện trong workflow** — phản biện đã tự đọc lại
> toàn bộ 18 URL, tìm ra 3 sai lệch số liệu (đã sửa trong tài liệu này, xem §5.3).
> Không chứa mã video thật, không chứa đáp án.

**Bối cảnh chốt**: thi sơ tuyển vòng 3 tối 04/09 19h30. Điểm vòng 2: 10,0/30 (top-1 = 15).
Bốn khoảng trống đang mở: (A) trải 100 dòng dưới prefix-max; (B) TRAKE định vị sự kiện
(72,8% khoảng cách); (C) Q&A nhân từ 0/1 (nền 93,3%); (D) sinh ứng viên một cảnh + gộp
3 nguồn (hình + BM25 lời thoại + OCR toàn kho — **OCR đang quét đêm nay**).

**Hai luật đo không du di** (áp cho mọi phép đếm dưới đây):
1. Nhóm hai-cảnh trên TUNE có hiệu lực ngoại suy yếu (bộ đo 0/60 câu hai cảnh vs 51% đề
   BTC) — bắt buộc phân tầng một-cảnh/hai-cảnh, bootstrap theo câu khi <30 câu, TEST đọc
   đúng MỘT lần.
2. Mọi số nguồn đo bằng nDCG@10/H@5 (nhấn đầu danh sách) phải chấm lại bằng
   Final = 1/5·ΣR@k prefix-max của ta trước khi tin.

**Ba cặp trùng đã gộp** (4 lane hội tụ về cùng cơ chế — tín hiệu tốt, nhưng chạy rời là
đốt đôi quỹ giờ và mở cửa chọn-ngưỡng-hậu-nghiệm):
- WRRF: `vbs-doi-thu#1` ≡ `phan-bo-dong#5` → MỘT thí nghiệm, MỘT bộ ngưỡng (dòng R6 §0).
- DP TRAKE: DANTE-DP ≡ Drop-DTW k-best → MỘT harness DP, HAI chế độ phạt (dòng R5 §0).
- Kênh OCR/ASR cho QA: `vbs-doi-thu#4` ⊃ `qa-nhan-tu#2` → MỘT đường A/B đa kênh (dòng R3 §0).

---

## §0. BẢNG XẾP HẠNG ĐỀ XUẤT GIỮ (theo giá trị dự kiến / chi phí phép đếm)

Thứ tự = thứ tự CHẠY đề nghị của phản biện (an-toàn-cấu-trúc × tốc-độ-phép-đếm), vì 16
phép đếm không thể cùng xong trước 04/09.

| # | Đề xuất (đã gộp trùng) | Gap | Chi phí đếm | Vì sao xếp đây |
|---|---|---|---|---|
| R1 | Chi lại phần ĐUÔI 51..100 (suffix re-spend) | A | 0 đ, tất định, vài phút CPU | An toàn TUYỆT ĐỐI theo luật prefix-max: 4/5 số hạng Final bất biến |
| R2 | Chẩn đoán lưới keyframe thưa (1 giờ) | D | 0 đ, 1 giờ | Quyết SỐNG/CHẾT cả nhánh D; ngưỡng giết ý tưởng hai phía chuẩn nhất trong 18 đề xuất |
| R3 | Đếm đáp-án-trong-OCR + OCR-augmented prompting QA (gộp kênh OCR/ASR 3 kênh) | C | 0 đ | Phép đếm rẻ nhất của cả 18 đề xuất; tài sản OCR sẵn đêm nay |
| R4 | ASC-vote THUẦN K lần (margin chỉ làm cổng escalate chéo model) | C | ~0 đ (Gemini free) | Bằng chứng mạnh nhất + mới nhất trên chính Gemini; phép đếm instability chạy được đêm nay |
| R5 | Harness DP TRAKE gộp: λ-penalty (DANTE) vs drop-cost (Drop-DTW) + k-best 100 chuỗi | B | 0 đ, CPU <1s/câu | Nhắm thẳng 72,8% khoảng cách; 100 dòng = 100 chuỗi là cách khai thác prefix-max mới thật |
| R6 | WRRF gộp: chỉ mục GHÉP OCR+ASR + WRRF k nhỏ, α theo mật độ OCR | D | 0 đ (chờ OCR xong) | Số cứng trên kho video TIN TỨC đa ngữ, đúng cấu trúc 3 nguồn của ta |
| R7 | Bảo lãnh suất nguồn top-5 trong 100 dòng (VISIONE) | A | 0 đ | Khoảng trống (A) đang VÔ CHỦ — không đối thủ AIC nào công bố cơ chế trải 100 dòng |
| R8 | TFVTG static+dynamic scorer cho từng sự kiện TRAKE | B | 0 đ, CPU vài phút | Scorer chuyển thẳng sang sims SigLIP-2; ablation nguồn rõ từng bậc |
| R9 | Box-Cox (TAG) sửa nén dải sim — bật/tắt riêng từng thành phần | B | 0 đ | Sửa đúng bệnh nén dải SigLIP-2; rẻ nhất trong họ TAG |
| R10 | Lưới phi-đều theo entropy từng sự kiện (chồng lên R5) | B | 0 đ | Ý nhà tự có, văn liệu xác nhận CHƯA AI LÀM; tự hoãn nếu R5 trượt |
| R11 | VRisk/CVaR đặt dòng nội-video (chạy SAU khi R1 chốt) | A | 0 đ | Tổng quát hoá thật allocator đã ship (β=1 thu về nền) — nhưng đụng dòng ≤20, lãnh thổ đã sập TEST 2 lần |
| R12 | QA 3 kênh phương án đáp án kiểm chứng được (khung gộp cho R3/R4) | C | ~0,2–0,5 USD | Nhắm đúng 6,7% còn lại; luật ≥2 kênh ủng hộ |
| R13 | ViCrop crop-zoom đọc đáp án (cổng vào quyết trong vài phút) | C | 0 đ | Nhiều khả năng không đủ cổng vào — chết sớm không tiếc |
| R14 | GUARD-RAIL verify-mirage: cấm blanket self-verification | C | 0 đ | Không phải máy tăng điểm — là luật CHẶN một chiều thua, tiên quyết cho mọi bước verify/bầu chọn |

### Chi tiết từng dòng (URL + số liệu gốc + phép-đếm-trước + ngưỡng tiền-đăng-ký)

**R1 — Suffix re-spend (A)** — lane `phan-bo-dong`, verdict GIỮ ("đề xuất mạnh nhất trong 18").
- URL: https://www.microsoft.com/en-us/research/wp-content/uploads/2009/02/diversifying-wsdm09.pdf (Agrawal WSDM'09, toàn văn; phản biện đã tải lại PDF và đọc trang 1–2).
- Số liệu gốc: mục tiêu P(≥1 kết quả hữu ích) là submodular; greedy IA-Select đạt (1−1/e)≈63% tối ưu. Cấu trúc trùng luật R@k = max trên prefix (hit thứ hai không cộng gì).
- Phép-đếm-trước (0 đ, tất định): mô phỏng qua `scripts/do_phan_bo_sau.py` (đường sinh dòng đã assert trùng từng dòng với `make_submission.allocate_rows`). Giữ dòng 1..50, chi lại 51..100 theo sàn phủ đều (mỗi cụm ứng viên trong video đúng-tiềm-năng nhận ≥1 dòng, kể cả ứng viên softmax ~0). Đếm: (i) bao nhiêu mục nhóm "mất do đặt dòng" (9 hai cảnh + 8 một cảnh, `docs/PHAN_BO_TREN_BO_MOI.md` §1b) chuyển trượt→trúng ở ≤100; (ii) bao nhiêu mục đang trúng hạng 51..100 bị mất.
- Ngưỡng tiền-đăng-ký: net ≥ +4 mục trên nửa TUNE phân tầng VÀ assert 0 thay đổi ở mọi hạng ≤50. Chỉ khi qua mới bootstrap điểm + đọc TEST một lần. Bậc mở rộng ranh 21..100 CHỈ nếu bậc 1 giữ được.
- Lưu ý phản biện: trần lợi ích khiêm tốn (chỉ ăn số hạng R@100, mỗi câu ±0,2) — nhưng chi phí 0, rủi ro 0, né đúng dạng hỏng TUNE→TEST đã giết lane phân bổ 2 lần; có tiền lệ `reserve_tail_rows` trong mã (đã từng cứu 2 truy vấn vòng 1).

**R2 — Chẩn đoán lưới keyframe thưa (D)** — lane `vbs-doi-thu`, verdict GIỮ.
- URL: https://openaccess.thecvf.com/content/CVPR2025W/IViSE/papers/Quan_Toward_Automation_in_Text-based_Video_Retrieval_with_LLM_Assistance_CVPRW_2025_paper.pdf (toàn văn PDF 9 trang; phản biện đọc lại trực tiếp).
- Số liệu gốc (ĐÃ SỬA theo phản biện): lưới FFmpeg dày hơn Cineast 43,40% (1.082.659 → 1.552.550 khung trên V3C1) làm SigLIP tăng +61,11% và EVA-CLIP +88,28% toàn chỉ số truy xuất. Bảng 1: SigLIP H@10 lưới dày là **0,6304** (số 0,6957 mà lane trích là của EVA-CLIP — lane ghép nhầm cột, không đổi kết luận "mật độ lưới là biến số lớn hơn cả chọn encoder"). Tỉ lệ khung/giờ kho ta (~203 khung/video) so V3C1: không rõ — phải đo.
- Phép-đếm-trước (0 đ, 1 giờ): trên các câu TUNE một-cảnh đang TRƯỢT (video đúng, khung sai/không lọt top), đo khoảng cách thời gian từ mốc GT đến keyframe gần nhất trong chỉ mục.
- Ngưỡng tiền-đăng-ký HAI PHÍA: ≥30% câu trượt có GT cách keyframe >2s → lưới thưa là nút nghẽn thật, đáng làm mịn cục bộ (decode ±2s quanh ứng viên top allocator, encode SigLIP-2 tại chỗ, CHỈ cho ~100 dòng ứng viên — không toàn kho); <10% → bỏ ngay cả nhánh.
- Lưu ý: nếu dương, phần decode+encode là GPU cục bộ sát giờ thi — giữ phạm vi hẹp đúng đề xuất.

**R3 — Đáp-án-trong-OCR + OCR-augmented prompting (C, gộp `qa-nhan-tu#2` vào kênh OCR/ASR của `vbs-doi-thu#4`)** — verdict GIỮ.
- URL: https://arxiv.org/html/2510.02543 (toàn văn).
- Số liệu gốc: KOCRBench (250 câu VQA song ngữ): Gemini 2.5 Flash 182→212 câu đúng (+12 điểm %); InternVL 2.5 7B 87→162; gain lớn nhất ở nhóm KIE (đọc số/tên/biển hiệu — đúng loại câu AIC); không ghi nhận OCR làm giảm điểm; training-free thuần prompting.
- Phép-đếm-trước (0 đ, rẻ nhất trong 18 đề xuất): với các câu Q&A của bộ đo, đếm bao nhiêu câu có đáp án GT xuất hiện NGUYÊN VĂN (hoặc fuzzy ≥0,8) trong text OCR của keyframe định vị (và 1–2 khung lân cận) — đặc biệt nhóm câu đang sai.
- Ngưỡng tiền-đăng-ký: ≥1 câu đang sai có đáp án nằm sẵn trong OCR → chạy A/B chèn "Chữ xuất hiện trên hình: ..." vào prompt; ship khi net ≥ +1 câu và KHÔNG câu đúng nào bị lật sai.

**R4 — ASC-vote thuần (C)** — verdict GIỮ với SỬA PHẠM VI quan trọng.
- URL: https://arxiv.org/html/2606.04323 (ASC-MQRA, CVPR 2026 VidLLMs — toàn văn; phản biện đọc lại).
- Số liệu gốc (kèm sửa của phản biện): Gemini 3.1 Pro Preview: single-pass 72,71% → ASC K=10 đạt 81,16% (+8,45 điểm). NHƯNG bảng test của chính nguồn: MQRA (M=0) 80,85 < 81,16 và bài nộp leaderboard cuối của họ là **ASC THUẦN** ("re-arbitration slightly degrades performance") — nửa "vòng trọng tài" bị chính nguồn phủ nhận ở chế độ dùng-mù.
- Phạm vi ship SAU SỬA: voting thuần K=5, temperature 1.0, trên cùng bộ khung đã định vị; margin phiếu CHỈ giữ làm cổng escalate sang model KHÁC (giả thuyết riêng của ta, nguồn chưa đo — phải qua ma trận lật R14 trước).
- Phép-đếm-trước (~0 đ, Gemini free, chạy đêm nay): lấy RIÊNG các câu Q&A đang SAI, chạy lại mỗi câu 3 lần temp 1.0. Nếu ≥1/3 số câu sai ra đáp án đúng ở ≥1 lần chạy thì voting có cửa.
- Ngưỡng tiền-đăng-ký: net trên toàn bộ Q&A GT ≥ +2 câu VÀ 0 câu đúng bị lật sai. Với nền 93,3% headroom chỉ ~4 câu: ngưỡng "không lật câu đúng" quan trọng hơn ngưỡng tăng.

**R5 — Harness DP TRAKE gộp (B)** — DANTE-DP (`vbs-doi-thu#3`) ≡ Drop-DTW k-best (`trake-dinh-vi#2`), cả hai verdict GIỮ, phản biện bắt buộc MỘT harness HAI chế độ phạt.
- URL: https://arxiv.org/html/2512.13169 (DANTE, đội Outstanding TRAKE tại AIC 2025 — toàn văn) + https://github.com/SamsungLabs/Drop-DTW (README, official NeurIPS'21) + https://arxiv.org/abs/2108.11996 (abstract).
- Số liệu gốc: DANTE: DP[i,t] = S[i,t] + max_τ(DP[i−1,τ] − λ(t−τ)); λ=0.001 tốt khi gap sự kiện 3–15 chỉ số, λ=0.01 khi 1–3; O(N·T); KHÔNG có bảng ablation (mức tin cậy = cơ chế + giải thưởng). Drop-DTW: drop-cost theo percentile phân bố match cost (keep_percentile≈0.3); KHÔNG có số zero-shot trên benchmark grounding — phép đếm là điều kiện sống còn, không phải thủ tục.
- Cơ chế gộp: ma trận S (N sự kiện × T keyframe của video đã đúng, sims SigLIP-2 cache sẵn); DP đơn điệu t_1 ≤ … ≤ t_N; hai chế độ phạt = {λ tuyến tính (quét λ∈{0.001, 0.003, 0.01} + biến thể λ_i theo khe — phần MỚI của ta), drop-cost percentile}; backtracking k-best → top-100 chuỗi PHÂN BIỆT = 100 dòng nộp (luật TRAKE không phạt thứ tự nộp → coverage@100 là cận trên ăn được). Đơn điệu là prior cấu trúc, KHÔNG phạm cửa đã đóng soft_order/unordered vì đây là ORDERED trên lưới — đúng hướng mở gap B.
- Phép-đếm-trước (0 đ, CPU <1s/câu, ~100 dòng numpy): trên câu TRAKE TUNE có video đúng, đếm 3 số: (1) hit-rate từng-sự-kiện DP-path vs argmax độc lập (DP không được giảm); (2) tỉ lệ câu full-correct; (3) coverage@100 của k-best vs cách trải hiện tại.
- Ngưỡng tiền-đăng-ký: điểm TRAKE trung bình ≥ +5 điểm % so cơ chế hiện hành trên TUNE VÀ coverage@100 ≥ +5 điểm; bootstrap theo câu (bắt buộc, <30 câu).

**R6 — WRRF gộp OCR+ASR (D)** — `vbs-doi-thu#1` ≡ `phan-bo-dong#5`, cả hai verdict GIỮ, bắt buộc MỘT thí nghiệm MỘT bộ ngưỡng.
- URL: https://arxiv.org/html/2503.20698v4 (MMMORRF, SIGIR 2025 — toàn văn; phản biện kiểm từng số, khớp chính xác).
- Số liệu gốc: MultiVENT 2.0 (218k video TIN TỨC đa ngữ, 3906 câu): SigLIP vision-only nDCG@10 = 0,375; ASR-only 0,427; OCR-only 0,347; chỉ mục GHÉP OCR+ASR **0,551**; +vision RRF thường 0,562; WRRF **0,586** (+4,2% so RRF thường, có ý nghĩa thống kê); **k=0** thay vì 60 (dồn giá trị vào hạng 1–5, hợp R@1/R@5 của ta); TVR R@10 0,540 vs LanguageBind 0,258. Điểm yếu nguồn (lane tự khai đúng): họ KHÔNG quét α trên validation.
- Cơ chế: (1) ghép OCR + transcript cùng đoạn thành MỘT chỉ mục BM25 (phát hiện then chốt: chỉ mục ghép thắng từng nguồn rời, +12,4% so ASR đơn); (2) WRRF(q,d) = α_d/(r_text+k) + (1−α_d)/(r_vision+k), k∈0..10; (3) α_d theo video, lấy 0 đ từ mật độ ký tự OCR (thay câu thăm dò); tie-break đầu danh sách bằng đếm-phủ-nguồn kiểu U-CESE.
- Phép-đếm-trước (0 đ, chờ OCR toàn kho xong): 6 cấu hình {3 nguồn đơn; RRF k=60; RRF k=0; WRRF α theo mật độ OCR} đo bằng R@100-video + hạng video đúng đầu tiên; chọn trên nửa TUNE, TEST một lần, bootstrap theo câu. Chấm thêm nhóm MỘT CẢNH bằng Final prefix-max (nguồn đo nDCG@10 — nhấn đầu — phải chấm lại).
- Ngưỡng tiền-đăng-ký (một bộ duy nhất): chỉ mục ghép phải thắng BM25-ASR đơn ≥3 điểm % R@10; WRRF phải thắng CẢ RRF thường LẪN nguồn đơn tốt nhất ≥2 điểm % R@10 (tương đương ≥2σ trên R@100-video); nếu WRRF chỉ hoà RRF thường → dùng RRF thường (ít tham số hơn).
- Rủi ro giữ trong phép đếm: truy vấn MultiVENT thiên văn-bản/named-entity hơn truy vấn tả cảnh AIC — phần thắng của chỉ mục ghép có thể co lại.

**R7 — Bảo lãnh suất nguồn top-5 trong 100 dòng (A)** — lane `vbs-doi-thu`, verdict GIỮ.
- URL: https://zenodo.org/records/13903347 (VISIONE 5.0, vô địch VBS 2024 — toàn văn PDF 8 trang; phản biện tải và đọc trực tiếp, trích dẫn có nguyên văn trang 4–5, Fig 2 đúng như mô tả).
- Số liệu gốc: "late fusion... RRF, with an additional enhancement on the top-5 results from each ranked list"; log Fig 2: có câu video đúng rank ≤5 ở model phụ trong khi >100 ở model chính. Bằng chứng phụ (Quan CVPRW 2025, Bảng 3) xác nhận CẢ HAI CHIỀU: gộp top-10 hai encoder tăng H@5 (0,7478 > 0,7348/0,6609) nhưng GIẢM H@1 (0,5957 < 0,6174 của EVA đơn) → ngưỡng "không giảm R@1" là sống còn.
- Cảnh giác phản biện: nguồn mô tả cơ chế bằng MỘT câu, không số chi tiết — phần dải-prefix cụ thể là thiết kế riêng của ta, bằng chứng thực chất là PHÉP ĐẾM chứ không phải paper.
- Cơ chế: mỗi nguồn xếp hạng (SigLIP-2, BM25 lời thoại, BM25 OCR, biến thể truy vấn) được BẢO LÃNH chỗ cho top-m của nó trong dải prefix cố định (vd top-5 mỗi nguồn nằm trong 20 dòng đầu), phần còn lại xen kẽ theo allocator. Allocator quyết ĐỘ SÂU mỗi video, cơ chế này quyết THỨ TỰ CHÈN giữa các nguồn.
- Phép-đếm-trước (0 đ): trên pools.json TUNE, dựng 100 dòng hai kiểu (fusion thuần điểm vs fusion + bảo lãnh top-5/nguồn trong 20 dòng đầu), chấm Final prefix-max, bootstrap theo câu, tách nhóm một-cảnh/hai-cảnh.
- Ngưỡng tiền-đăng-ký: KHÔNG giảm R@1 quá 0,5 điểm; tổng Final tăng ≥ +1,5 điểm %.

**R8 — TFVTG static+dynamic scorer (B)** — lane `trake-dinh-vi`, verdict GIỮ.
- URL: https://arxiv.org/html/2408.16219 (ECCV 2024 — toàn văn; phản biện kiểm, mọi số khớp) + README https://github.com/minghangz/TFVTG.
- Số liệu gốc: zero-shot Charades-STA R@0.5=49,97, R@0.7=24,32, mIoU=44,51; ablation từng bậc: chỉ static 45,48 → +dynamic 48,01 (+2,5) → +lọc thứ tự sub-event 49,97 (+1,96). Encoder đóng băng — cơ chế chỉ cần đường sim khung–văn bản → chuyển được sang SigLIP-2.
- Cơ chế: dynamic = tổng vi phân dương liên tiếp (bắt "cú tăng sim" lúc sự kiện bắt đầu); static = trung bình sim trong đoạn trừ ngoài đoạn; mốc nộp = điểm kết thúc transition; top-k mốc/sự kiện + lọc tổ hợp theo thứ tự.
- Hai caveat chuyển giao (phản biện): (i) lưới ta ~203 khung/video thưa + phi-đều — vi phân dynamic phải CHUẨN HOÁ theo khoảng cách thời gian thật giữa keyframe kẻo thành nhiễu; (ii) δ≈5e-4 hiệu chỉnh cho dải sim BLIP-2 — fit lại δ trên TUNE, không mang số của họ sang. Ranh giới: lọc Gaussian ở đây là tiền xử lý CỤC BỘ cho định vị sự kiện, không được trượt thành "làm mượt thời gian" cho truy xuất (cửa đã đóng).
- Phép-đếm-trước (0 đ): trên câu TRAKE TUNE video đúng, so hit-rate từng-sự-kiện của (a) argmax sim thô vs (b) static+dynamic + lọc thứ tự.
- Ngưỡng tiền-đăng-ký: (b) tăng ≥ +3 điểm, bootstrap theo câu; không đạt thì bỏ, không tune tiếp.

**R9 — Box-Cox / TAG (B)** — lane `trake-dinh-vi`, verdict GIỮ (ưu tiên 3 trong gap B).
- URL: https://arxiv.org/html/2508.07925 (toàn văn) + README https://github.com/Nuetee/TAG.
- Số liệu gốc: vượt TFVTG cả 2 bộ (Charades mIoU 45,69 vs 44,51; ActivityNet 36,55 vs 34,10); KHÔNG cần LLM lúc suy diễn (LLM chỉ +0,73 mIoU). (Phản biện: số mIoU chi tiết chưa đối chiếu được từ abstract nhưng nhất quán với chuỗi TFVTG→TAG.)
- Cách chạy phản biện đề nghị: bật/tắt RIÊNG Box-Cox trước (rẻ nhất, sửa đúng bệnh nén dải sim SigLIP-2, ghép được làm scorer cho DP R5); change-point clustering để lại nếu còn giờ.
- Ngưỡng tiền-đăng-ký: ≥ +2 điểm hit-rate mỗi thành phần mới giữ; bootstrap theo câu.

**R10 — Lưới phi-đều theo entropy từng sự kiện (B)** — lane `trake-dinh-vi`, verdict GIỮ (ý nhà tự có; điều kiện: chồng lên R5, R5 trượt thì tự hoãn).
- URL: https://arxiv.org/html/2607.00672 (DART, ECCV 2026 — toàn văn; CHỈ mượn thước spectral entropy + xác nhận khoảng trống, KHÔNG nhập đường suy diễn cần LLaVA).
- Số liệu gốc: DART dùng spectral entropy chuẩn hoá H∈[0,1] làm thước bất định mức-truy-vấn. Lưới phi-đều MỨC TỪNG SỰ KIỆN cho nộp đa dòng: chưa tồn tại trong văn liệu (điểm mới của ta).
- Cơ chế: phân bổ số mốc thay thế mỗi sự kiện tỉ lệ entropy của softmax đường sim (sự kiện "nhọn" 1–2 ứng viên, "phẳng" nhiều ứng viên quanh đỉnh phụ); beam ≈100 chuỗi.
- Phép-đếm-trước (0 đ): cùng ngân sách B (quét B=5,10,20 mốc/sự kiện), so coverage-cửa-sổ chia đều B/N vs chia theo entropy.
- Ngưỡng tiền-đăng-ký: entropy thắng chia đều ≥ +3 điểm coverage ở CÙNG B; thua thì ý tưởng chết tại chỗ.

**R11 — VRisk/CVaR đặt dòng nội-video (A)** — lane `phan-bo-dong`, verdict GIỮ có điều kiện (chạy SAU R1; lớp đếm là KHẾ ƯỚC).
- URL: https://arxiv.org/html/2510.22681 (WSDM 2026, Takehi–Diaz–Sakai — toàn văn).
- Số liệu gốc: VRisk = CVaR_β trên phân bố ý định; β=1 ≡ expected loss (tức TỔNG QUÁT HOÁ allocator phủ-xác-suất đã ship); greedy VRisker bảo đảm (1−1/e); giảm 20–33% worst-case loss đổi 2–5% điểm trung bình (INTENT-2, TREC Web 2012); các thuật toán đa dạng hoá cũ "không robust hơn naive" trên metric tuyến tính.
- Vì sao xếp sau: cơ chế đụng dòng ≤20 — đúng lãnh thổ lane phân bổ đã sập TEST 2 lần (TUNE +90,6% → TEST −3,8% nhóm hai cảnh); nhóm hai-cảnh TUNE hiệu lực ngoại suy yếu.
- Phép-đếm-trước (0 đ, LỚP ĐẾM tất định chạy trước): β∈{0,1; 0,25; 0,5}, thay vòng argmax trong `allocate_coverage_rows` bằng VRisker-β; đếm (i) trong 17 mục addressable, bao nhiêu được đặt ≥1 dòng trong ±20 của keyframe-gần-đáp-án; (ii) bao nhiêu mục đang trúng mất dòng khỏi ±20.
- Ngưỡng tiền-đăng-ký (giữ như khế ước): cứu ≥5/17 VÀ hỏng 0 mục đang trúng hạng ≤20 — trượt là BỎ LANE, không quét thêm β. Lớp điểm chỉ khi qua lớp đếm; TEST một lần, báo riêng nhóm hai cảnh.

**R12 — QA 3 kênh phương án đáp án kiểm chứng được (C)** — lane `vbs-doi-thu`, verdict GIỮ (khung gộp cho R3/R4).
- URL: https://openaccess.thecvf.com/content/CVPR2025W/IViSE/papers/Quan_Toward_Automation_in_Text-based_Video_Retrieval_with_LLM_Assistance_CVPRW_2025_paper.pdf (toàn văn 9 trang) + https://link.springer.com/chapter/10.1007/978-981-95-6963-2_26 (NII-UIT VBS2026 — abstract, Springer chặn fetch, kiểm qua tìm kiếm).
- Số liệu gốc: Quan: Question Extraction giữ ngữ cảnh (§3.3.1), temporal-assisted +103,65% H@1 (SigLIP), rerank InternVL2.5-8B 4-bit +10,07% H@1; NII-UIT VBS2026: Candidate Answer Suggestion gộp manh mối đa phương thức thành phương án kiểm chứng được + Answer Span Prediction. YẾU: cả hai nguồn KHÔNG có số đo QA riêng (lane khai trung thực).
- Cơ chế: mỗi câu QA sinh phương án ĐỘC LẬP từ 3 kênh — VLM trên cụm khung quanh mốc; lời thoại ±20s; OCR ±20s — rồi đối chiếu/bầu chọn. LUẬT: đáp án phải được ≥2 kênh ủng hộ HOẶC kênh duy nhất là loại độc quyền (biển số/tên riêng → OCR; lời nói → ASR). Nhắm 6,7% còn lại (đáp án nằm trong chữ/lời thoại, không trong pixel).
- Điều kiện phản biện: bước "LLM đối chiếu/bầu chọn" phải qua ma trận lật sai↔đúng TRƯỚC (R14 — nguy cơ verification mirage).
- Ngưỡng tiền-đăng-ký: lật ròng ≥ +2 câu (tức ≥95%) VÀ 0 câu đúng bị lật; bootstrap theo câu. Chi phí ≈0,2–0,5 USD API cho 60 câu × 3 kênh (0 đ nếu VLM cục bộ).

**R13 — ViCrop crop-zoom đọc đáp án (C)** — lane `qa-nhan-tu`, verdict GIỮ (xếp cuối lane QA).
- URL: https://arxiv.org/html/2310.16033v3 (toàn văn).
- Số liệu gốc: TextVQA (BLIP-2 FlanT5XL): 25,91% → att-CROP 34,26% (+8,35), human-crop 37,68%; accuracy MLLM giảm tới 46% theo kích thước chủ thể; mạnh nhất ở text <0,5% diện tích ảnh. Caveat lớn (lane tự khai): số đo trên BLIP-2 2023–24, Gemini đời 2026 nhiều khả năng tự xử lý được chữ nhỏ → gain có thể ~0. KHÔNG đụng cửa đã đóng (đây là bước ĐỌC đáp án sau định vị, không phải crop cho retrieval).
- Phép-đếm-trước (0 đ, vài phút): phân loại tay các câu QA đang sai — bao nhiêu thuộc loại "đọc chữ/số nhỏ trên hình".
- Cổng vào tiền-đăng-ký: ≥2 câu sai thuộc loại này mới chạy A/B (full-frame vs full-frame+crop từ OCR-box, nới biên 2x, upscale); ship khi net ≥ +1, 0 lật đúng→sai. Nhiều khả năng không đủ cổng — chết sớm không tiếc.

**R14 — GUARD-RAIL verify-mirage (C)** — lane `qa-nhan-tu`, verdict GIỮ (luật chặn, không phải máy tăng điểm).
- URL: https://arxiv.org/html/2605.10850v1 (toàn văn, số trích nguyên văn).
- Số liệu gốc: 6 VLM open-weight, 5 bộ VQA, 7 loại task, actor-verifier loop 4 vòng: "Only 2.2–3.8% of initially wrong answers are corrected over four turns, while 69.5%–87.1% are locked in by false verification"; verifier "lazy", under-attend bằng chứng ảnh; cross-verification giảm mirage nhưng không xoá hẳn. Khớp bằng chứng độc lập từ nguồn ASC (MQRA giảm điểm trên test).
- LUẬT ÁP DỤNG XUYÊN SUỐT 4 LANE: (1) KHÔNG blanket self-verification; (2) nếu verify thì CHÉO model (gpt-5.2 kiểm Gemini) và CHỈ ở câu margin thấp; (3) verify dạng đọc-lại-bằng-chứng (crop/OCR làm evidence), không hỏi "đáp án này đúng không"; (4) MỌI bước verify/trọng tài/bầu chọn phải qua ma trận lật (sai→đúng) vs (đúng→sai) trên GT trước khi ship — chỉ ship nếu net ≥ +2 câu.
- Caveat trung thực: miền nguồn là medical VQA + model nhỏ; độ LỚN mirage trên Gemini/tin tức chưa biết, nhưng chiều lập luận (verifier cùng năng lực → đồng thuận sai) chuyển được.

---

## §1. GAP A — TRẢI 100 DÒNG DƯỚI PREFIX-MAX (oracle: 43% trần)

**Phát hiện chiến lược số một của đợt nghiên cứu** (lane `vbs-doi-thu`): KHÔNG đối thủ AIC
2025 nào công bố cơ chế trải 100 dòng dưới prefix-max — họ đều nộp danh sách fusion thuần
điểm. Khoảng trống (A) đang VÔ CHỦ. Đồng thời lane `phan-bo-dong` xác nhận: không tồn tại
paper "submission construction dưới recall@k prefix-max"; khung hình thức gần nhất là
mục tiêu P(≥1 hit) submodular (IA-Select, WSDM'09) và coverage/pass@k (Large Language
Monkeys — chỉ đọc abstract, dùng làm khung khái niệm).

**Bài học cấu trúc** (soi mã + lịch sử): greedy của `allocate_coverage_rows` đã gần tối ưu
CHO MÔ HÌNH PHỦ của nó; lane phân bổ chết 2 lần đều do TINH CHỈNH THAM SỐ cùng cơ chế
(TUNE +25,4%/+20,1% → TEST +1,5%/−1,0%). Vì vậy mọi đề xuất giữ lại đều đổi CẤU TRÚC,
không đổi tham số.

Thứ tự trong gap: **R1 (suffix re-spend) → R7 (bảo lãnh nguồn) → R11 (VRisk)**.
- R1 an toàn tuyệt đối: khoá 1..50 → R@1/5/20/50 bất biến; chỉ được, khó mất.
- R7 cần bảng hạng BM25/OCR cạnh SigLIP-2; ngưỡng "R@1 không giảm quá 0,5" là sống còn
  (bằng chứng Quan Bảng 3: gộp tăng H@5 nhưng GIẢM H@1).
- R11 duy nhất đụng dòng ≤20 → chạy cuối, lớp đếm là khế ước, trượt là bỏ không quét β.
- Phần "danh mục đuôi đơn-nguồn" của đề xuất semivariance (NGHI_NGỜ, §5) được GỘP vào
  phép đếm của R1+R7: bảng chéo nguồn×trúng-trượt + đếm trần tay đôi (mục trượt ≤100
  nhưng video đúng CÓ trong top BM25/OCR). Không dựng máy Markowitz riêng.
- TRECVID AVS: đã rà, không có gì nhập được (xem §5.2).

---

## §2. GAP B — TRAKE ĐỊNH VỊ SỰ KIỆN (72,8% khoảng cách)

**Toàn cảnh văn liệu**: SOTA training-free 2024–2025 cho grounding zero-shot đều chấm trên
đường sim khung–văn bản y hệt hạ tầng ta có (TFVTG ECCV'24 → TAG 8/2025, TAG vượt và
không cần LLM). Chưa paper grounding nào dùng Viterbi k-best cho chuỗi sự kiện;
Drop-DTW/StepFormer chứng minh DP đơn điệu + drop-cost hoạt động cho step localization.
DANTE (arXiv 2512.13169) là cơ chế của đội Outstanding TRAKE tại chính AIC 2025.
VBS không có bài toán chuỗi mốc (KIS chỉ 1 khoảnh khắc) — không vay thêm được từ hướng đó.

**Kiến trúc gộp** (phản biện bắt buộc — 4 đề xuất cùng gap tranh MỘT quỹ giờ):

```
sims SigLIP-2 cache (N sự kiện × T keyframe, video đã đúng)
  → [R9] Box-Cox sửa nén dải (bật/tắt đếm riêng, ≥+2 điểm mới giữ)
  → [R8] scorer static+dynamic (vi phân chuẩn hoá theo Δt thật; δ fit lại trên TUNE)
  → [R5] DP đơn điệu k-best, HAI chế độ phạt: λ-penalty (λ toàn cục → λ_i theo khe)
         vs drop-cost percentile  →  100 dòng = 100 chuỗi phân biệt
  → [R10] phân bổ ứng viên/sự kiện theo entropy (chồng lên R5; R5 trượt thì hoãn)
```

Thứ tự đếm đề nghị (lane `trake-dinh-vi`, phản biện đồng thuận): R8+R9 (scorer) → R5
(k-best DP) → R10 (entropy) — cả bốn cùng MỘT pipeline đếm trên TUNE, xong trong một buổi.
Ba chỉ số tiền-đăng-ký chung: hit-rate từng-sự-kiện, tỉ lệ câu full-correct, coverage@100.
Luật chấm nhắc lại: sai video = 0; đúng video = (1/N)·số sự kiện lọt cửa sổ, KHÔNG ràng
thứ tự — nhưng đơn điệu của DP là prior cấu trúc thu hẹp không gian tìm, chỉ lợi.

**Ranh giới cửa-đã-đóng phải canh**: lọc Gaussian trong TFVTG chỉ là tiền xử lý cục bộ cho
định vị sự kiện — không được trượt lại thành "làm mượt thời gian" cho truy xuất; DP ordered
trên lưới là hướng mở đã ghi, KHÔNG phải soft_order/unordered/hedge-video (đã đóng).

---

## §3. GAP C — Q&A NHÂN TỪ 0/1 (nền 93,3% ≈ 56/60; headroom ~4 câu)

**Ba cơ chế dương tính + một cảnh báo âm tính**, thứ tự chạy: R3 (OCR, 0 đ, tài sản sẵn
đêm nay) → R4 (ASC-vote thuần) → R12 (3 kênh) → R13 (ViCrop, cổng vào quyết trong vài
phút); R14 là guard-rail xuyên suốt.

Nguyên tắc chung với nền cao: **ngưỡng "0 câu đúng bị lật" quan trọng hơn ngưỡng tăng** —
mọi kênh mới đều phải đo ma trận lật hai chiều trên GT trước khi ship.

Điểm đáng nhớ từ phản biện về R4: chính nguồn ASC cho thấy vòng trọng tài (re-arbitration)
GIẢM điểm ở chế độ dùng-mù (MQRA 80,85 < ASC thuần 81,16) và bài nộp cuối của họ là vote
thuần → ta ship vote thuần; margin chỉ làm cổng escalate CHÉO model (gpt-5.2 kiểm Gemini),
và bước escalate đó là giả thuyết riêng chưa ai đo — bắt buộc qua ma trận lật.

Phân loại tay câu sai (đêm nay, 0 đ) phục vụ đồng thời 3 cổng: loại đọc-chữ-nhỏ (cổng
R13), loại đáp-án-trong-chữ-trên-hình (cổng R3), loại đáp-án-trong-lời-thoại (kênh ASR
của R12).

---

## §4. GAP D — SINH ỨNG VIÊN MỘT CẢNH (17% trần) + GỘP 3 NGUỒN

**Chạy R2 (chẩn đoán lưới thưa) ĐẦU TIÊN của gap** — 1 giờ, quyết cả nhánh: nếu ≥30% câu
một-cảnh trượt có GT cách keyframe gần nhất >2s → làm mịn cục bộ (decode ±2s quanh ứng
viên top allocator, encode SigLIP-2 tại chỗ, CHỈ ~100 dòng ứng viên/câu); nếu <10% → bỏ,
dồn giờ cho R6. Đây là dạng "truy xuất thô → làm mịn cục bộ", không đụng cửa đã đóng nào.

**R6 (WRRF gộp)** chờ OCR toàn kho xong. Số của MMMORRF trên MultiVENT 2.0 (218k video
tin tức đa ngữ — miền sát kho ta nhất trong mọi nguồn đã đọc): chỉ mục GHÉP OCR+ASR 0,551
đè vision-only 0,375; WRRF k=0 đạt 0,586. Ba điều chép được: (1) ghép OCR+ASR thành MỘT
chỉ mục BM25, không hai bảng riêng; (2) k nhỏ (0..10) thay k=60 — dồn giá trị vào hạng
1–5, hợp R@1/R@5 prefix-max; (3) α_d theo từng video — bản ta dùng mật độ ký tự OCR,
0 đồng. Hai lane đề xuất trùng nhau → MỘT thí nghiệm, MỘT bộ ngưỡng (đã hợp nhất ở R6
§0); kết quả WRRF (nếu qua) nối vào allocator và làm đầu vào nguồn cho R7 (gap A).

Sai lệch đã sửa khi trích Quan CVPRW: SigLIP H@10 lưới dày = 0,6304 (0,6957 là EVA-CLIP).

---

## §5. DANH SÁCH LOẠI / NGHI NGỜ (để khỏi ai làm lại)

### 5.1. NGHI_NGỜ (verdict phản biện) — để sau giải, KHÔNG tốn quỹ giờ trước 04/09

1. **Conformal Risk Control hai tầng thay quét lưới** (lane `phan-bo-dong`;
   https://arxiv.org/abs/2404.17769v2 — lane chỉ đọc abstract).
   Lý do loại khỏi quỹ giờ: (a) lane trích SAI dữ kiện — paper thực nghiệm trên MSLR-Web
   + Yahoo LTRC, KHÔNG có MS MARCO; (b) đòn nặng hơn về chuyển giao: bảo đảm conformal
   đòi exchangeability giữa dữ liệu hiệu chuẩn và lúc dùng — mà bộ đo của ta lệch phân bố
   cấu trúc so đề BTC (0/60 vs 51% câu hai cảnh) nên "E[rủi ro]≤α" hiệu chuẩn trên TUNE
   không bảo đảm gì cho đề thật; (c) n=66 → độ phân giải α ~1,5%, bảo đảm thô. Giá trị
   còn lại chỉ là phanh chống overfit, không phải máy tăng điểm.

2. **Danh mục đuôi mean-semivariance (Markowitz/PMPT cho 100 dòng)** (lane `phan-bo-dong`;
   https://pdfs.semanticscholar.org/05f0/8b783bc492bd412275b4dcc731925d2a22df.pdf — đã đọc
   đủ 15 trang, lane trích trung thực).
   Lý do: bằng chứng là văn bản tĩnh 2009, gain 1–7% trên metric đầu-danh-sách
   (MAP5/P@5/NDCG@5), miền cách xa; mắt xích quyết định — proxy "hiệp phương sai = trùng
   nguồn" — là sáng chế riêng chưa có số chống lưng. Phần ăn được thật (bảng chéo
   nguồn×trúng-trượt; dòng đuôi cho ứng viên đơn-nguồn) ĐÃ GỘP vào R1+R7. Chỉ xét lại nếu
   phép đếm trần tay đôi ra ≥6 mục thật sau khi OCR xong.

### 5.2. LOẠI ngay từ trong lane (đã có lý do, đừng đào lại)

- **CoMET-Agent multi-event** (arXiv 2606.15320): cần GPT-5, F1@0.5 chỉ 16,2% — quá đắt,
  quá yếu.
- **FOCUS cropping** (arXiv 2506.21710): cần KV-cache nội bộ model → không dùng được với
  model API (Gemini/gpt-5.2); chỉ còn giá trị bằng-chứng-phụ cho hướng crop.
- **Đường suy diễn của DART** (arXiv 2607.00672): cần LLaVA-1.6-7B — chỉ mượn thước
  spectral entropy làm thước bất định (R10).
- **Vòng trọng tài MQRA (re-arbitration)**: chính bảng test của nguồn ASC phủ nhận
  (80,85 < 81,16; bài nộp leaderboard cuối của họ là ASC thuần).
- **Blanket verify-loop / self-verification**: Verification Mirage (2605.10850) đo được
  chỉ 2,2–3,8% câu sai được sửa trong khi 69,5–87,1% bị KHOÁ bởi xác minh sai — dự đoán
  ÂM cho mọi phiên bản dùng-mù; chỉ còn phiên bản gated + chéo model (R14).
- **TRECVID AVS run-construction**: KHÔNG tồn tại paper về diversity-vs-depth trong xây
  run; trang luật NIST (đã đọc toàn trang) chỉ quy định 1000 shot/run + xinfAP; "novelty
  run" thưởng shot-độc-nhất — khác hẳn prefix-max, không có gì nhập được.
- **PIKA Search** (github BaryuH/Ho-Chi-Minh-AI-Challenge-2025): README nghèo chi tiết,
  không có gì trích được.
- Nhắc lại **CỬA ĐÃ ĐÓNG BẰNG SỐ** (phản biện đã soát: không đề xuất nào trong 18 là
  cửa-đóng đổi tên): NNN/QB-Norm/DBNorm, PRF Rocchio ảnh-với-ảnh, GQE/MUGI paraphrase,
  cut-score/TransNetV2, VLM chấm lại nội-video từng khung, PE-Core, làm mượt thời gian,
  chuẩn hoá theo video, soft_order/unordered/hedge-video cho TRAKE, doc2query.
  Hai chỗ SÁT RANH cần canh khi triển khai: lọc Gaussian trong TFVTG (R8) và mọi bước
  trọng tài/bầu chọn LLM của các đường QA (R4/R12).

### 5.3. Ba sai lệch số liệu phản biện tìm thấy (đã sửa ở các mục tương ứng)

| # | Chỗ sai (bản lane) | Sửa (bản này) |
|---|---|---|
| a | Đề xuất CRC trích "MS MARCO" | Paper dùng MSLR-Web + Yahoo LTRC; lane chỉ đọc abstract |
| b | Lưới dày: "SigLIP H@10 0,3913→0,6957" | 0,6957 là EVA-CLIP; SigLIP lưới dày = 0,6304 (ghép nhầm cột Bảng 1) — kết luận không đổi |
| c | ASC-vote kèm "vòng trọng tài khi margin thấp" | Bảng test nguồn: MQRA 80,85 < ASC thuần 81,16; ship vote THUẦN, margin chỉ làm cổng escalate chéo model (giả thuyết chưa ai đo, phải qua ma trận lật) |

---

## §6. KẾ HOẠCH 24 GIỜ TỚI (đêm 03/09 → giờ thi tối 04/09 19h30)

### Đêm nay 03/09 (OCR toàn kho đang quét — các việc dưới KHÔNG chờ nó)

| Giờ ước | Việc | Gap | Cổng quyết định |
|---|---|---|---|
| ~1h | **R1** mô phỏng suffix re-spend qua `scripts/do_phan_bo_sau.py`: khoá dòng 1..50, chi lại 51..100 theo sàn phủ đều; đếm net; assert 0 thay đổi ≤50 | A | net ≥ +4 mục → mai bootstrap + TEST một lần; giữ được thì mở bậc ranh 21..100 |
| ~1h | **R2** chẩn đoán lưới thưa: đo khoảng cách GT→keyframe gần nhất trên câu một-cảnh trượt | D | ≥30% >2s → mai làm mịn cục bộ; <10% → bỏ hẳn nhánh |
| ~1h | **R4-đếm** rerun 3 lần (temp 1.0) các câu QA đang sai (Gemini free) — đo instability | C | ≥1/3 câu sai ra đúng ≥1 lần → mai chạy ASC-vote K=5 toàn bộ |
| ~30' | Phân loại tay câu QA sai: đọc-chữ-nhỏ / đáp-án-trong-chữ / đáp-án-trong-lời-thoại | C | cổng vào cho R3, R12, R13 (R13 cần ≥2 câu loại chữ nhỏ) |
| nền | Dựng sẵn harness DP TRAKE (R5, ~100 dòng numpy) + scorer R8/R9 để sáng mai chỉ việc chạy đếm | B | — |

### Sáng mai 04/09 (mốc: OCR toàn kho xong)

1. **R3-đếm** (chạy NGAY khi OCR xong, ~30 phút): dò đáp án GT nguyên văn/fuzzy ≥0,8
   trong OCR của khung định vị, ưu tiên nhóm câu sai. Qua cổng (≥1 câu sai có đáp án
   trong OCR) → A/B OCR-augmented prompting.
2. **Bảng chéo nguồn×trúng-trượt** 3 nguồn (SigLIP / BM25 lời thoại / OCR) + trần tay
   đôi: đếm mục trượt ≤100 nhưng video đúng CÓ trong top BM25/OCR — phục vụ đồng thời
   R6, R7 và quyết số phận semivariance (≥6 mục mới xét lại).
3. **R6** WRRF gộp: 6 cấu hình {3 nguồn đơn; RRF k=60; RRF k=0; WRRF α mật độ OCR},
   chọn trên nửa TUNE, chấm cả R@100-video lẫn Final prefix-max nhóm một cảnh;
   WRRF chỉ hoà RRF thường → dùng RRF thường.
4. **R5** chạy harness DP TRAKE: baseline argmax → R8/R9 scorer (đếm riêng từng bậc,
   +3/+2 điểm mới giữ) → DP hai chế độ phạt (λ vs drop-cost) → k-best coverage@100;
   R5 qua ngưỡng thì đếm nhanh R10 (entropy vs chia đều, cùng ngân sách B).
5. **R7** bảo lãnh top-5/nguồn (sau khi có kết quả bước 2–3): dựng 100 dòng hai kiểu,
   ngưỡng R@1 không giảm >0,5 điểm VÀ Final +1,5 điểm %.
6. **R11** VRisk lớp đếm — CHỈ nếu R1 đã chốt và còn giờ; trượt khế ước (cứu <5/17 hoặc
   hỏng ≥1 mục đang trúng ≤20) là bỏ hẳn, không quét thêm β.

### Chiều mai 04/09 (trước giờ thi 19h30)

- **15h00**: chốt danh sách cơ chế QUA NGƯỠNG tiền-đăng-ký; mỗi cơ chế qua → đọc TEST
  đúng MỘT lần; cơ chế nào TEST không giữ → trả về cấu hình sản xuất, không cứu vãn.
- **QA**: nếu instability dương → ASC-vote K=5 toàn bộ + ma trận lật hai chiều; escalate
  chéo model chỉ ở câu margin thấp VÀ chỉ khi ma trận lật dương (R14). Tuyệt đối không
  blanket verify.
- **17h30–18h00**: ĐÓNG BĂNG cấu hình nộp; chạy end-to-end kiểm định dạng (≤100 dòng/câu),
  smoke-test allocator + đường TRAKE + đường QA trên vài câu TUNE.
- **18h00–19h30**: đệm sự cố + nghỉ. Không đổi cấu hình sau 18h00 trừ lỗi định dạng nộp.

**Nguyên tắc cắt khi thiếu giờ**: R1–R6 bắt buộc thử theo đúng thứ tự; R7–R13 tuỳ giờ;
R14 là luật, không tốn giờ. CRC + semivariance: để sau giải.

---

## KẾT QUẢ CHẠY R1 — ĐẾM QUA, ĐIỂM ÂM, DỪNG TRƯỚC TEST (đêm 03/09)

`dem_chi_lai_duoi.py` (bậc đếm): TUNE net **+7 mục** (cứu 8/mất 1; ngưỡng ≥+4)
→ QUA. `do_diem_chi_lai_duoi.py` (bậc điểm, chấm y sản xuất {6,10,20} × 4 họ
× 48 bốc): TUNE **−1,2%** (0,1962→0,1939; KTC [−0,0094,+0,0051]; P(≤0)=74,1%)
→ **DỪNG, không đọc TEST, không ship.**

Vì sao đếm và điểm ngược dấu — ghi để khỏi ai lặp lại:
1. Hàng cứu được nằm hạng 51..100 ⇒ chỉ ăn số hạng R@100 (1/5 trọng số) với
   BUCKET hạng sâu — mỗi mục cứu chỉ đáng vài phần nghìn điểm.
2. Đuôi mới đặt dòng ĐƠN tại keyframe, giãn >20 frame; bộ chấm thật bốc cửa sổ
   ±6/±10 quanh khoảnh khắc thật ⇒ dòng đơn trượt nhiều lần bốc mà thang bù
   trừ dày (step 10) của đuôi cũ đỡ được. **Phép đếm tất định ±20 là thước
   LẠC QUAN hơn bộ chấm thật** — bài học công cụ đo thứ tư của tuần.
3. Biến thể "mini-ladder trong đuôi" = quay lại chỉnh tham số phủ/độ sâu —
   lãnh thổ đã giết lane phân bổ HAI lần trên TEST. Không đi.

R1 ĐÓNG cho biến thể dòng-đơn. R11 (VRisk) tự động hoãn theo điều kiện ghi
sẵn ("chạy SAU khi R1 chốt" — chốt ÂM thì độ ưu tiên tụt sau R5/R6).

---

## KẾT QUẢ CHẠY R5 BẬC ĐẾM (đêm 03/09) — DP-định-vị SỐNG, k-best-trải CHẾT

`scripts/do_dp_trake.py` (sims KhoSims cache, TUNE = 12 mục cũ, chấm y
`do_trake_bo_moi`, thang so sánh A = argmax độc lập + `allocate_trake_rows`
trên CÙNG tín hiệu S):

| cấu hình | ±6 (quyết định) | ±10 | ±20 |
|---|---|---|---|
| A argmax + thang | 0,1947 | 0,2301 | 0,3050 |
| **B DP(λ=0,003) + thang** | **0,2308 (+18,5%)** | 0,2755 | 0,3661 |
| C DP(λ=0,003) k-best-100 | 0,1389 (−28,6%) | 0,2137 | 0,3832 |

- **B (DP chỉ làm ĐỊNH VỊ):** dương ở CẢ BA cửa sổ, đường λ có đỉnh
  (0,001:+0,025 / 0,003:+0,036 / 0,01:−0,027 — khớp khuyến nghị dải λ của
  DANTE, không phải đường phẳng nhiễu); bootstrap câu P(≤0)=8,3%. **NHƯNG
  +3,6 điểm tuyệt đối < ngưỡng chặt +5 điểm đã đăng ký** (đọc "điểm %" theo
  nghĩa chặt = điểm tuyệt đối ×100). Trạng thái: **TÍN HIỆU MẠNH, CHƯA XÁC
  NHẬN** — không đọc 12 mục TEST đêm nay.
- **C (k-best chuỗi làm cách TRẢI 100 dòng): ÂM dứt khoát** ở cột quyết định
  (P(≤0)=100%): chuỗi không đệm trượt cửa sổ ±6, thang bù trừ dày thắng;
  +0,078 ở ±20 xác nhận cơ chế phủ-rộng nhưng BTC nói cửa sổ <10 frame ⇒
  đóng biến thể này, đừng thử lại với cửa sổ hẹp.
- Chế độ drop-cost CHƯA cài (ghi rõ trong script); λ_i theo khe (R10) chưa đo.

KẾ HOẠCH SÁNG MAI cho R5 (đăng ký trước khi nhìn thêm số): (1) so B với đường
sản xuất THẬT (align_sequence) trên cùng 12 mục; (2) cài λ_i theo khe + drop-cost,
chọn trên TUNE; (3) đọc 12 mục mới MỘT lần; ship chỉ khi TEST ≥ +5 điểm tuyệt
đối ở ±6 VÀ so-sản-xuất-thật dương. Không đạt thì ghi "tín hiệu treo, cần n
lớn hơn sau giải".

---

## KẾT QUẢ CHẠY R2 (đêm 03/09) — nhánh làm-mịn-lưới BỎ, kèm một lỗi công cụ đáng ghi

`scripts/dem_luoi_thua.py` trên bộ đo: 0/8 câu một-cảnh trượt có GT cách
keyframe >2s — nhưng số này **VÔ GIÁ TRỊ vì vòng lặp công cụ**: GT của bộ đo
được SINH TỪ chính keyframe (neo là keyframe) nên khoảnh khắc GT nằm trên lưới
THEO CẤU TRÚC. Bộ đo này mù bẩm sinh với câu hỏi mật độ lưới — suýt giết nhánh
bằng một con số giả tạo.

Đo lại KHÔNG-vòng-lặp trên **42 mốc người-xác-minh của đề thật vòng 2**
(`picks_final*.txt` — mốc do người xem video chọn): trung vị 0,00s; p90 0,89s;
**>2s: 3/42 = 7% < ngưỡng bỏ 10%** ⇒ **nhánh decode-làm-mịn-cục-bộ BỎ cho
vòng 3** (không đáng GPU giờ chót). Caveat còn lại ghi trung thực: 34/42 mốc
đúng 0 frame — một phần vì người soát chọn từ trang review (hiển thị keyframe),
nên 7% là CẬN DƯỚI; nhưng khi người soát LỆCH khỏi lưới thì lệch nhỏ (≤3s,
đa phần <1s) — kết luận BỎ vẫn đứng với dữ liệu tốt nhất hiện có. R6 (WRRF gộp
nguồn) KHÔNG bị ảnh hưởng — vẫn chờ OCR toàn kho.

### R5 bậc đếm — vòng 2 (rạng sáng 04/09): biến thể λ_i entropy (R10) dẫn đầu

Quét đủ 3 kiểu phạt × λ∈{0,001..0,05} trên TUNE (C đã đóng, chỉ B):

| kiểu | λ tốt nhất | ±6 | so A |
|---|---|---|---|
| goc (DANTE thuần) | 0,003 | 0,2308 | +18,5% |
| drop (Drop-DTW thích nghi) | — | **trơ** (không sự kiện nào dưới phân vị 30 trên TUNE — không phân định được, không phải âm) | — |
| **lam_i (λ_i entropy khe — ý R10 của TA)** | **0,01** | **0,2389** | **+22,7%**, P(≤0)=6,5% |

λ_i là đỉnh NỘI VÙNG thật (0,02→0,2200; 0,05→0,1751 — hai bên đều tụt), và nó
CỨU vùng λ cao (goc 0,01 sập còn 0,1673; lam_i 0,01 đứng đầu): sự kiện chắc
chắn chịu phạt gap đầy đủ, sự kiện mù mờ được thả lỏng — đúng trực giác R10,
văn liệu chưa ai làm ở mức từng-sự-kiện. Vẫn **+4,4 điểm < ngưỡng chặt +5**.

Sáng 04/09 (đăng ký từ đêm qua, không đổi): so với đường sản xuất THẬT
(align_sequence) trên 12 TUNE; đọc 12 mục TEST **một lần** với cấu hình đã
chốt `lam_i/λ=0,01`; ship qua cờ `--dp-trake` chỉ khi TEST ≥ +5 điểm tuyệt
đối ở ±6 và so-sản-xuất dương.

### R3 số đầy đủ (01h15 04/09) — ngưỡng QUA, kèm caveat một-chữ-số

`dem_dap_an_trong_ocr.py --quet`, 158/158 mục có OCR quanh keyframe-đáp-án ±1:
nhóm ĐANG SAI **8/81** có đáp án nằm sẵn trong OCR; nhóm đang đúng 3/77.
Ngưỡng ≥1 QUA. **Caveat giữ nguyên:** cả 8 ví dụ đều là đáp án MỘT chữ số
('1','2','3') — token đơn xuất hiện khắp nơi trong khung tin tức, khớp có thể
ăn may. Trọng tài duy nhất: A/B chèn OCR-của-khung-model-đọc vào prompt
(net ≥ +1 câu, 0 lật đúng→sai). Đang OCR đích danh các khung mà đường trả lời
THẬT SỰ đọc (`quet_ocr_khung_doc.py` — top-4 điểm ±1 lân cận) để A/B chạy
được ngay sáng.

### PROBE R4 — QUA (01h45 04/09): 9/24 ≥ ngưỡng 8 → VOTING CÓ CỬA

24 mục sai-đáp-án/đúng-video × 3 lần temp=1.0 (vá temp cục bộ tiến trình đo,
sản xuất không đụng): **9/24 mục có ≥1 lần ra đáp án đúng**; 2 mục ra đúng cả
3/3 — tức một phần cái "sai" trong cache là nhiễu lấy mẫu/xoay model, đúng
chẩn đoán ASC. Bước tiếp: A/B chèn-OCR đang chạy (158 call, nhánh đối chứng
đã có sẵn trong cache); nếu muốn vote K=3 phải đo net toàn bộ + ma trận lật
(luật R14 — không blanket self-verification).
