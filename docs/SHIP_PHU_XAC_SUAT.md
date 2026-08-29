# SHIP: bộ phân bổ PHỦ XÁC SUẤT vào sản xuất

Chốt ngày 29/08/2026, sau vòng sơ tuyển 2 (10,0 điểm / hạng 153, top-1 = 15).
Tổng hợp từ 3 lane thí nghiệm + 1 lượt phản biện độc lập (tái lập từng chữ số,
kiểm bằng 2 họ hạt giống hoàn toàn mới). Mọi con số dưới đây đều có cách đo kèm theo;
số dùng để quyết định là số **TEST** (nửa câu hỏi chưa từng dùng để chọn tham số),
không phải số TUNE.

---

## 1. Tham số chốt và bằng chứng

### Tham số đưa vào sản xuất

```
nhiệt      = 0,02
sigma      = 30
nửa_cửa_sổ = 6
lưới       = 5
ngân sách  = 100 dòng (MAX_ROWS), đuôi lấp bằng allocate_hybrid_rows khi thiếu
```

### Con số quyết định (TEST — 30 câu chẵn, hạt giống tách rời chưa dùng khi chọn)

| | điểm | so nền |
|---|---|---|
| nền (bộ phân bổ hybrid đang nộp) | 0,3421 ± 0,0041 | — |
| **phủ xác suất (0,02; 30; 6; 5)** | **0,3946 ± 0,0028** | **+15,3%** |

Chênh 0,0525 > 2σ = 0,0082 → **GIỮ ĐƯỢC** theo luật hoà của harness.

Cách đo: tổ hợp được **chọn sạch trên 30 câu lẻ** (fold đảo, TUNE),
rồi chấm **đúng một lần** trên 30 câu chẵn với 4 họ hạt giống tách rời
(gốc 90000) × 48 bốc; ứng viên từ đường sản xuất `ranked_hits` (400/câu);
đáp án bốc đều trong khe giữa hai keyframe (không snap); cửa sổ chấm {6, 10, 20};
chấm bằng `final_score`/`r_score_kis` của `src/core/submission.py`
(bản vector hoá đối chiếu lệch < 1e-12, assert trong script).
Script: `scripts/experiment_phu_quet_luoi.py` (fold đảo: `--tune-phia le`);
cache: `data/cache_phu_quet_luoi/`.

### Kiểm chứng độc lập (phản biện, hạt giống hoàn toàn mới, chưa lane nào dùng)

| lát cắt | gốc 70000 | gốc 123450 |
|---|---|---|
| cả 60 câu | +14,9% (0,3996 vs 0,3479) | +13,9% |
| nửa chẵn | +15,0% | +15,1% |
| nửa lẻ | +15,8% | +16,2% |

Mức +14–16% **không phụ thuộc họ hạt giống**. Ngược lại, tổ hợp thắng fold xuôi
(0,03; 30; 10) — đạt +20,1% trên TUNE chẵn — sụp về **−1,0%** trên TEST lẻ
(0,3491 vs 0,3527, < 2σ = HOÀ) và −1,1% khi đo lại bằng seed mới 70000:
đó là tổ hợp thắng nhờ nhiễu, protocol TUNE/TEST đã chặn đúng.

### Kỳ vọng khi lập kế hoạch điểm

- Ước lượng cho **tổ hợp cố định** (0,02; 30; 6): **+14% đến +16%** trên mọi lát cắt đã đo.
- Ước lượng **thận trọng cho quy trình** (trung bình out-of-fold 2 fold:
  (−1,0% + 15,3%)/2): **≈ +7%** — dùng số này khi lập kế hoạch điểm vòng sau.
- Đỉnh là **VÙNG chứ không phải ĐIỂM**: á quân (0,015; 20; 6) chênh đúng 0,0001
  trên TUNE lẻ. Vùng an toàn nếu chỉnh tay trong trận: nhiệt 0,015–0,02,
  sigma 20–30, nửa_cửa_sổ 6–10. **TRÁNH** nhiệt ≥ 0,03 (bất ổn giữa hai nửa),
  nửa_cửa_sổ 15 (âm/yếu cả hai bảng), nhiệt 0,05 (đo −5,6%).

**CẢNH BÁO tham số:** bản kế hoạch ship đầu tiên ghi nhầm mặc định
`sigma=20, nửa_cửa_sổ=10` — tổ hợp đó đo được chỉ **+4,6% trên nửa lẻ**.
Mặc định trong mã phải là **(0,02; 30; 6; 5)**, không chép chữ ký hàm từ bản kế hoạch cũ.

---

## 2. Nạp ứng viên nội-video vào tiên nghiệm: KHÔNG ghép

**Kết luận: NO-GO.** Tiêm toàn bộ keyframe của top-K video dẫn đầu vào tiên nghiệm
của bộ phủ **không ăn**:

- TEST (30 câu lẻ), biến thể chọn trên TUNE (K=3, m=1.0):
  **0,3729 ± 0,0033 vs nền 0,3719 ± 0,0040 → +0,2%, < 2σ = HOÀ.**
- TUNE (30 câu chẵn): cả 6 biến thể K∈{3,5,10} × m∈{1.0, 0.5} đều ≤ nền
  (−0,4% đến −1,2%).

Nguyên nhân **đo được**, không phải đoán:

1. Tiền đề coverage sai trên toàn tập: **54/60 câu ĐÃ có** keyframe gần đáp án nhất
   trong 400 ứng viên toàn cục; tiêm chỉ thêm 2–3 câu.
2. Ở chính 6 câu nghẽn (chỉ số **5, 9, 12, 15, 40, 41** trong `ground_truth.json`),
   delta sau tiêm = **0,000 đúng bằng máy**: SigLIP xếp keyframe đáp án ở hạng
   nội-video 95/147, 227/307, 152/262 — softmax nhiệt 0,02 cấp cho chỗ đúng trọng
   số 1,8e-05 đến 2,3e-04, khối lượng tiên nghiệm không bao giờ tới nơi.
3. Trong khi đó 4,9–12,9% khối lượng bị kéo sang ứng viên mới (đa số video sai),
   làm loãng 54 câu khoẻ.

Hệ quả: con số **+27,5% nội-video của chẩn đoán 24/08 là cận trên kiểu oracle**,
không phải thứ tiêm-ứng-viên với tới được — nó đòi một tín hiệu ĐỊNH VỊ trong video
mà SigLIP thuần không có. Mọi tài liệu còn trích "+27,5% tiềm năng" như mục tiêu
cần sửa lại.

Chú ý mồi overfit: K=10 m=0.5 hiện +1,1% trên TEST nhưng **không** được chọn trên
TUNE (ở đó nó −1,2%) và vẫn < 2σ — không được nhặt số này.

Script tái lập (~2,5 phút CPU, không API): `scripts/experiment_phu_noi_video.py`,
dùng cache `data/_prodhits60.json` + `data/_sims_siglip60.npz` (sim SigLIP của
MỌI keyframe × 60 câu — thí nghiệm định vị nội-video sau này đều cần).
~~Việc còn nợ khi rảnh: chạy lại một lần với nền (0,02; 30; 6).~~ **ĐÃ TRẢ
(29/08 chiều):** chạy lại với đúng nền chốt — TUNE chẵn: cả 6 biến thể ≤ nền
(−0,2% đến −1,9%); biến thể chọn trên TUNE (K=3, m=0.5) hoà trên TEST
(0,4048 vs 0,4058, −0,2% < 2σ). K=10 m=1.0 lại nhô +1,3% trên TEST mà không
được chọn trên TUNE — đúng mồi overfit đã cảnh báo, không nhặt. NO-GO đứng vững.

---

## 3. Kế hoạch ship từng bước

Tổng: **16–23 giờ người** (~2–3 ngày một người; chia được: người A bước 1–3,
người B bước 4, bước 5–6 làm chung cuối).

### Bước 1 — Port allocator vào `src/core/submission.py` (3–4 giờ)

- Chuyển `phu_xac_suat` (74 dòng, `scripts/experiment_phu_xac_suat.py:66-139`)
  thành `allocate_coverage_rows(candidates, plan)` + dataclass
  `CoveragePlan(nhiet=0.02, sigma=30.0, nua_cua_so=6, luoi=5, budget=MAX_ROWS)`
  — đặt cạnh `allocate_hybrid_rows` để mọi nơi import từ một chỗ.
- **Đuôi lấp bắt buộc:** vòng tham lam break khi hết khối lượng chưa phủ và có thể
  trả < 100 dòng (pool 1 ứng viên: 7 dòng; pin video ngắn: 13–21; pin video dài:
  63–100 — đo trực tiếp), trong khi `verify_submission_zip`
  (`src/core/submission.py:552-564`) **chặn upload** file < 100 dòng như lỗi
  truncation. Phần thiếu lấp bằng `allocate_hybrid_rows` (bỏ trùng) cho đủ 100.
  Tin tốt đã đo: trên 60 câu GT coverage trả đủ **100/100 dòng cả 60 câu** —
  đuôi lấp không chạm vào kết quả +15,3% đã đo, nhưng vẫn bắt buộc cho pool nhỏ/pin.
- Cắm vào `build_kis_rows`/`build_qa_rows` (`make_submission.py:371-410`) qua hàm
  điều phối `allocate_rows(cands, allocator, ...)`. TRAKE giữ nguyên.
- Cờ rút lui `--allocator {hybrid,coverage}` (mặc định **vẫn hybrid** cho tới khi
  qua cổng bước 3); in allocator ra stdout + ghi `allocator.txt` cạnh csv để trang
  review và `apply_picks` đối chiếu.

### Bước 2 — `apply_picks`: đường pin giữ hybrid (2–3 giờ)

- Pick **có frame/chuỗi frame** → `pin_rows` đi đường hybrid y nguyên
  (`apply_picks.py:111,115-116` gán score 1e9; :258-269 tách pin_rows/rest_rows);
  `rest_rows` (ngân sách 100−|pin_rows|, video khác) đi theo `--allocator`.
- Pick **chỉ có video** → ứng viên giữ score thật, coverage chạy nguyên bản.
- Lý do: score 1e9 + softmax nhiệt 0,02 làm tiên nghiệm sụp về 1 frame; và khi
  người đã chốt frame thì bất định ~0, thang dày quanh frame chốt (hybrid) chính
  là "phủ" tối ưu của một điểm đã biết. (Lưu ý phản biện: với pin trong video dài,
  coverage vẫn trả đủ 100 dòng nhờ đuôi Gauss — nhưng dòng 1 là điểm LƯỚI chứ
  không phải frame chốt, nên pin vẫn phải đi hybrid.)
- Test: với pick có frame, |pin_rows| dòng đầu **giống hệt** giữa hai `--allocator`.

### Bước 3 — Cổng TUNE/TEST trên BẢN SHIP, rồi mới đổi mặc định (~2 giờ: 1 giờ máy + 1 giờ đọc)

- Chạy lại theo luật harness trên **bản ship** (sau đuôi lấp + lượng tử hoá +
  làm tròn score — không phải bản thí nghiệm): chốt trên 30 câu TUNE (chẵn),
  đọc 30 câu TEST (lẻ) **đúng một lần**.
- Đổi mặc định sang coverage **chỉ khi** TEST thắng hybrid ≥ 2σ; không thì
  coverage ở lại dạng cờ opt-in.

### Bước 4 — Port JS cho trang review (4–6 giờ port + 2–3 giờ test)

- DATA: thêm score làm phần tử thứ 4 của mỗi cands row `[vid,f,last,s]`
  (~11–12 KB, +1,5% kích thước trang — đo trên `round_p1/review.html` 787.464 byte).
  PLAN thêm `{allocator, nhiet, sigma, nuaCuaSo, luoi}`.
- `allocateCoverageRows` trong `review_export.js` ≈ 90–120 dòng; nối tại `rowsFor()`
  trong PAGE của `build_review_page.py` (~dòng 867-882), điều phối theo
  `PLAN.allocator`; `candOf`/`orderedCands` mang score theo.
- Câu operator **đã đụng** (`st.touched`) đi đường hybrid CŨ với thứ tự đã kéo —
  giữ lời hứa "thẻ #1 = dòng 1" mà `test_dragging_a_candidate_to_the_top` khoá;
  coverage chỉ áp cho câu máy tự quyết.
- **Bịt bẫy làm tròn (phản biện tìm ra, các test fuzz thường KHÔNG bắt được):**
  hoặc nhúng score đầy đủ độ chính xác round-trip vào DATA (+~2–3% trang), hoặc
  làm tròn 4 chữ số **Ở CẢ PHÍA PYTHON** tại biên `ranked_hits` rồi chạy lại cổng
  bước 3 trên bản đã làm tròn. Ở nhiệt 0,02, lệch 5e-5 điểm đổi trọng số ~0,25%,
  đủ lật argmax ô lưới sát nút → trang ≠ pipeline trong khi fuzz test (hai bên
  cùng input đã làm tròn) vẫn pass giả.
- Chống trôi float `np.exp` vs `Math.exp`: lượng tử hoá khối lượng về bậc 1e-9
  ở CẢ HAI phía ngay sau khi tính Gauss; duyệt video theo thứ tự chèn
  (Python dict / JS Map); argmax lấy max đầu tiên.
- Test đối chiếu: (a) `test_js_allocator.py::test_coverage_matches_row_for_row`
  — fuzz ≥ 5 seed, ứng viên CỤM theo video để ép cửa sổ gần hoà, so 100 dòng
  từng-dòng; (b) test đuôi lấp: pool nhỏ/video ngắn vẫn ra đúng 100 dòng giống
  nhau; (c) `test_page_export_matches_pipeline.py`: driver đọc `PLAN.allocator`
  từ trang thật; (d) mạnh nhất: **Python đọc pool TỪ TRANG**
  (`round_p1/review.html`, 24 câu thật) rồi diff từng dòng với node — không đọc
  từ `ranked_hits` tươi, nếu không bẫy làm tròn lọt lưới.

### Bước 5 — Cổng hồi quy trước khi merge (3–5 giờ)

Tất cả phải xanh trước khi merge nhánh:

1. `scripts/so_sanh_allocator.py` (viết mới): truy xuất MỘT lần (`ranked_hits`),
   phân bổ HAI lần, xuất hai bộ csv + bảng từng câu.
   - Trên 60 câu GT: `final_score` từng câu, 3–4 họ hạt giống × 48 bốc,
     luật hoà 2σ, cột "video dòng 1 có đổi không".
   - Trên đề vòng 2 thật (không GT): diff cấu trúc — video dòng 1 giống/khác,
     số video được phủ, số dòng thuộc video top.
2. `pytest tests/test_js_allocator.py tests/test_page_export_matches_pipeline.py`
   + test kéo-thả hiện có (khoá lời hứa thẻ #1 = dòng 1).
3. Cổng bước 3 đã chạy trên bản ship và kết quả TEST được ghi vào tài liệu này.
4. Đo độ trễ thật (ước < 2 s/câu, ~100 bước × ~10^7 phép numpy — chưa đo).
5. Tổng duyệt trên `round_p1` (24 câu thật) dưới cả node lẫn Python.

**Luật sau merge:** sau BẤT KỲ thay đổi nào tới chỉ mục hoặc `ranked_hits`
(kể cả lane nội-video tương lai làm đổi pool), chạy lại cả hai fold
(`python scripts/experiment_phu_quet_luoi.py` và `--tune-phia le`, thêm
`--refresh`) trước khi giữ tham số — sigma/nhiệt được tính trên pool 400 hiện tại.

### Bước 6 — Runbook ngày thi (1–2 giờ, gộp vào tổng duyệt)

- Chạy cả hai bộ song song từ MỘT lần truy xuất; nộp bộ theo cờ đã chốt.
- Lệnh rút lui `--allocator hybrid` in sẵn trong runbook — hybrid cách một cờ.
- Trước mỗi upload: `pytest tests/test_page_export_matches_pipeline.py` (vài giây).

---

## 3b. KẾT QUẢ CỔNG BƯỚC 3 — chạy 29/08/2026 trên BẢN SHIP

`python scripts/so_sanh_allocator.py` — phân bổ bằng đúng `allocate_rows` của
make_submission (làm tròn score 4 chữ số + lượng tử hoá 1e-9 + đuôi lấp),
chấm theo luật harness (họ hạt giống TEST gốc 90000, cửa sổ 6/10/20, 4 họ × 48 bốc):

| nửa | nền (hybrid) | coverage bản ship | so nền | phán quyết |
|---|---|---|---|---|
| chẵn | 0,3421 ± 0,0041 | 0,3946 ± 0,0027 | **+15,3%** | GIỮ ĐƯỢC (> 2σ) |
| lẻ | 0,3527 ± 0,0047 | 0,4090 ± 0,0014 | **+16,0%** | GIỮ ĐƯỢC (> 2σ) |

Số bản ship TRÙNG số thí nghiệm tới 4 chữ số — làm tròn, lượng tử hoá và đuôi
lấp không đổi kết quả (điều chưa biết #2: ĐÃ ĐÓNG). Độ trễ thật: **167 ms/câu
trung bình, p95 278 ms, max 578 ms** (điều chưa biết #4: ĐÃ ĐÓNG — xa dưới
ước lượng 2 s). Từng câu: 26 câu tăng / 12 câu giảm / còn lại hoà; video dòng 1
đổi ở 17/60 câu. Câu giảm mạnh nhất: chỉ số 37 (−0,40), 12 và 24 (−0,20) —
đúng dạng đã biết: coverage rải rộng nên khi hybrid vốn đặt trúng video ở
dòng 1, phần thang sâu bị mỏng đi; bù lại là 26 câu tăng, tổng vẫn +15/16%.

Điều kiện đổi mặc định make_submission ĐÃ THOẢ; thứ tự thao tác vẫn là:
port JS + test parity xanh TRƯỚC (bước 4), rồi mới lật mặc định trong MỘT
commit — nếu lật trước, trang review (còn xuất hybrid) sẽ lệch với zip của
pipeline, chính là chỗ kế hoạch này cấm.

## 4. Điều chưa biết

1. **Phân bố điểm SigLIP của đề vòng sau** có thể khác 60 câu GT — tham số nhạy
   (nhiệt 0,05 đã −5,6%). Không đo trước được; giảm rủi ro bằng diff cấu trúc của
   `so_sanh_allocator.py` trên đề thật + lệnh rút lui một cờ.
2. **+15,3% thuộc về bản thí nghiệm**, chưa phải bản ship (đuôi lấp + lượng tử hoá
   1e-9 + làm tròn score). Đã biết đuôi lấp không kích hoạt trên 60 GT
   (100/100 dòng), nhưng làm tròn score thì chưa đo — cổng bước 3 trả lời.
3. **Danh tính đỉnh không xác định**: điều được kiểm chứng là VÙNG tham số
   (nửa=6, nhiệt 0,01–0,02, sigma 20–30), không phải một điểm — á quân
   (0,015; 20; 6) chênh 0,0001 trên TUNE lẻ và cao hơn tổ hợp chốt trên TUNE chẵn.
4. **Độ trễ thật trên máy thi** chưa đo (ước < 2 s/câu) — đo ở bước 5.4.
5. **Coverage không chữa nghẽn truy xuất**: nó chỉ chia lại 100 dòng trong pool
   sẵn có. Nhóm 6 câu nghẽn (5, 9, 12, 15, 40, 41) cần tín hiệu định vị nội-video
   KHÁC SigLIP thuần. **Các tín hiệu RẺ đã đo trên đúng 6 câu này (29/08 chiều)
   — KHÔNG CÓ CỬA:**

   | câu | #kf | SigLIP thô | mượt w=3 | mượt w=7 | +CLIP z-blend |
   |---|---|---|---|---|---|
   | 5 | 150 | 16 | 51 | 49 | 18 |
   | 9 | 145 | 95 | 38 | 40 | 141 |
   | 12 | 379 | 10 | **4** | **1** | 10 |
   | 15 | 470 | 17 | 26 | 20 | 14 |
   | 40 | 305 | 225 | 265 | 276 | 207 |
   | 41 | 260 | 150 | 238 | 248 | 148 |

   (hạng nội-video của keyframe đáp án; nhỏ hơn = tốt hơn). Làm mượt thời gian
   cứu đúng 1/6 (câu 12) và làm câu 5 tệ gấp ba; CLIP không thêm thông tin;
   OCR keyword-match không nổ ở câu nào trong 4 câu có OCR (kể cả câu 40 —
   logo "60 Giây" 3D cách điệu, OCR không đọc được). Kết luận: nhóm này chỉ còn
   đường **quét VLM tại trận trên top 3–5 video** (đã có quy trình trong playbook,
   từng crack p2-17/p2-26) — không có tín hiệu tiền tính rẻ nào đáng ship.
6. **Hai số không còn cache đứng sau**: `prodhits_uniq7841.json` (đối chiếu cache
   24/08) không có trong repo; "+16,0% lẻ/hạt giống TEST" không nằm trong cache
   nào. Cả hai khớp phép đo độc lập bằng seed mới (+15,8%/+16,2%) nên đáng tin,
   nhưng quy tắc từ nay: **mọi số vào báo cáo phải có file cache đứng sau**.
7. **Chưa kiểm trên đề thật có GT** — 60 câu GT là proxy duy nhất; số liệu vòng 2
   thật chỉ có sau ngày thi.

---

## Phụ lục: kiểm chứng harness (vì sao tin được các số trên)

- Tái lập **từng chữ số** thí nghiệm gốc 24/08: nền 0,3496, phủ (0,02; 30; 10)
  = 0,3845 (+10,0%), phủ (0,02; 30; 6) = 0,3999 (+14,4%) — cả lane 1 lẫn phản
  biện độc lập (script tự dựng lại 100 dòng từ đầu, 9 giây).
- Bộ chấm vector hoá lệch tuyệt đối < 1e-12 so `final_score`/`r_score_kis`
  chính thức (assert trong script, fail nếu lệch).
- Cache TEST (`diem_test_le.json`, `diem_test_chan.json`) mỗi file **chỉ chứa
  đúng 2 mục** (nền + tổ hợp chốt) — bằng chứng vật lý TEST không bị đọc trước
  khi chốt.
- Đường sản xuất ổn định: chạy lại `ranked_hits` sau 5 ngày, tập 400 ứng viên
  giao 100,0% với cache cũ, top-1 khớp 59/60.
- Con số +10,0% của thí nghiệm gốc thuộc tổ hợp sigma=**30** (không phải 20 như
  dễ nhầm từ mô tả cũ) — phát hiện nhờ tái lập trùng từng chữ số.
