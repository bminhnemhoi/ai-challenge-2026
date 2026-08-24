# Hướng đi tiếp — bản chốt phương án cho trưởng nhóm

> Viết ngày 24/08/2026. Mọi con số trong tài liệu này đều ghi rõ **đo ở đâu**.
> Chỗ nào là phỏng đoán thì có chữ **[PHỎNG ĐOÁN]** ngay tại chỗ.
> Nền so sánh: 0,345 trên 60 câu ground truth (`scripts/loss_breakdown.py`),
> 8,6 trên bảng luyện tập.

---

## 1. Điều gì đang chặn điểm, và việc gì đáng làm nhất

Điểm không bị chặn ở khâu **tìm video** — nó bị chặn ở khâu **tiêu 100 dòng**.
Phân rã tổn thất đã nói rõ từ lâu: xếp hạng video hoàn hảo chỉ đáng +41%, còn vị
trí frame hoàn hảo đáng +115%, tức 60% phần lấy lại được nằm ở frame và chỉ 22%
ở video. Nhưng cách chúng ta *đặt tên* nhóm thất bại lớn nhất đã dẫn cả đội đi
sai hướng gần ba tuần: nhóm "14 câu — keyframe đúng có trong ứng viên nhưng thang
không vươn tới" **không phải lỗi thang ngắn, mà là lỗi thang đặt nhầm chỗ**. Mổ
xẻ 15 câu đó cho thấy khoảng cách từ keyframe đúng tới ứng viên hạng 1 của chính
video ấy có trung vị 1.312 frame (tối đa 24.932), tức cần trung vị 264 nấc thang
step=10 trong khi cả câu chỉ có 100 dòng; chỉ **2/15** câu thực sự được cứu bằng
thang dài hơn, còn 9/15 là vô vọng với mọi cấu hình thang. Hệ quả trực tiếp: mọi
đề xuất kiểu "thang dài hơn / dày hơn / chia lại ngân sách theo chi phí" đều nhắm
sai đích, và quả thật đã đo ra âm (nộp dày step 5 = −6,1%, step 3 = −11,7%; chia
lại ngân sách theo `A·v + B·m + C·d` = −15% đến −30%).

Việc đáng làm nhất là thứ **đã nằm sẵn trong repo và đã có số**: thay bộ phân bổ
100 dòng từ "đi theo chi phí tuyến tính `i + 0,5·d`" sang **phủ xác suất cực đại**
(`scripts/experiment_phu_xac_suat.py`). Đo trên **đường sản xuất đầy đủ**, 60 câu,
4 họ hạt giống độc lập × 48 lần bốc: **0,3496 ± 0,0023 → 0,3845 ± 0,0016, tức
+10,0%**, và số câu có video đúng trong 100 dòng đi từ 55/60 lên 58/60 — nghĩa là
nó chạm luôn cả nhóm thất bại số 2 mà không ai bảo nó làm thế. Lý do nó khác về
chất với mọi thứ đã đo âm: các thí nghiệm cũ đều **đổi thứ tự** (cộng điểm rồi sắp
lại) nên luôn vướng bẫy "đẩy cái này lên thì đá cái kia xuống"; bộ phân bổ mới
không đụng vào thứ hạng video một chút nào, nó chỉ đổi cách tiêu 100 dòng đã có.
Đây là bài toán **phủ cực đại có trọng số** suy thẳng từ luật chấm (vì R@k là
*max* trên tiền tố, mỗi câu chỉ cần **một** dòng trúng), chứ không phải bài toán
xếp hạng — và suốt từ đầu ta đã giải nó bằng công cụ xếp hạng.

---

## 2. Xây lại từ đầu có tốt hơn không?

### **KHÔNG.** Nhưng có đúng MỘT tầng phải xây lại, và ta không có quyền chọn.

**Bắt buộc xây lại: tầng chỉ mục — vì batch 2, không phải vì nó tồi.**
Trang 3 của "Thong tin vong So tuyen AIC2026.pdf" ghi nguyên văn: *"Đây cũng là dữ
liệu batch 1 của AIC 2025. Dữ liệu đầy đủ của vòng sơ tuyển AIC 2026 sẽ bao gồm
thêm dữ liệu batch 2, dự kiến được thông báo cho các đội thi trong thời gian tới."*
Một video batch 2 không có vector thì độ phủ của nó bằng **0 tuyệt đối**, không
tầng nào phía sau cứu được. Đây là **nghĩa vụ bảo trì**, không phải cơ hội cải
tiến — và phải làm **giống hệt cách cũ**: giữ nguyên encoder, giữ nguyên step,
giữ nguyên n_flat. Thực chất đây là phép **nối thêm** chứ không phải dựng lại:
177.321 vector batch 1 vẫn đúng nguyên, chỉ mã hoá khung mới rồi append **cả .npy
lẫn metadata.json cùng lúc** (`src/core/kis_engine.py:90` ném ValueError nếu lệch
— đó là lá chắn duy nhất). `notebooks/index-siglip2.ipynb` tái dùng nguyên vẹn từ
cell 2; chỉ phải thêm URL zip mới. **[PHỎNG ĐOÁN]** 2–4 giờ GPU T4 cho mỗi 177k
khung — notebook không lưu execution output nên chưa có số thật, phải bấm giờ.

**Phán quyết từng tầng:**

| tầng | quy mô | phán quyết | lý do |
|---|---|---|---|
| chấm điểm + phân bổ dòng (`src/core/submission.py`) | 628 dòng | **GIỮ, ĐÓNG BĂNG lõi chấm** | 34 test lấy thẳng từ ví dụ đã giải trong luật BTC. Chỉ thay *hàm phân bổ*, sau cờ |
| chỉ mục / embedding | .npy 817 MB | **XÂY LẠI — BẮT BUỘC** | batch 2 |
| truy xuất (`kis_engine.py`) | 412 dòng | GIỮ KHUNG | đổi encoder chỉ động 4 chỗ; nhưng **đừng đổi bây giờ** |
| kênh lời thoại | 8,7 KB | GIỮ, sửa 1 lỗi | xem §3 dòng 2 |
| kênh OCR | 10 KB | SỬA (có cổng) | hiện chỉ OCR ứng viên của vòng, không tìm được gì mới |
| kênh objects | 8,3 KB | GIỮ | +3,3% đã đo, ở tầng video |
| VLM | 393 dòng | GIỮ, sửa 1 lỗi | `--from-speech` đang lỗ tiền API |
| vòng duyệt bằng mắt | 1.521 + 213 dòng | **GIỮ — TÀI SẢN LỚN NHẤT** | 55/180 phút của vòng thi |
| bộ test (400 test / 45 giây) | — | **GIỮ — TÀI SẢN LỚN THỨ HAI** | chạy hôm nay: xanh hết |
| tài liệu | ~280 KB | **SỬA — đã lệch khỏi mã** | xem §6 |
| `app.py` + `frontend/` + `task1_kis` + `task2_vqa` | 5.207 dòng | XOÁ (hoặc sửa `app.py`) | nợ thuần + nguồn sai lệch |

**Vì sao KHÔNG xây lại toàn bộ.** Chi phí thật không phải 9.478 dòng mã đường thi
(viết lại được trong một tuần) mà là **400 test + 3.148 dòng `experiment_*.py` +
danh mục ~32 kết quả âm**. Xây lại từ đầu nghĩa là đi lại từng ngõ cụt. Ba cái
bẫy trong repo — "video R@1 tăng mà điểm thi giảm" (đã sập **bốn** lần: objects
theo frame 26→30, VLM w=0,20 25→29, chuẩn hoá theo video, và hợp nhất CLIP làm
vR@5 39→45 mà điểm không nhúc nhích), "chấm bằng ground truth thô đã snap" (đảo
kết luận n_flat=100 từ *tốt nhất 0,562* thành *kém nhất 0,257*), và "gộp hai danh
sách ứng viên tưởng miễn phí mà hoá ra âm" (0,321 → 0,305) — đều là loại lỗi mà
một đội bắt đầu lại **chắc chắn** mắc lại. **[PHỎNG ĐOÁN]** 2 tuần để quay về 8,6
điểm, với xác suất không nhỏ là không quay về được. Đây là phương án duy nhất có
thể làm **mất** điểm đã có.

---

## 3. Bảng xếp hạng đề xuất theo (điểm kỳ vọng / công sức)

> Cột "điểm kỳ vọng" ghi mức tăng **tương đối trên nền 0,345**. Quy sang bảng thi
> là **[PHỎNG ĐOÁN]** ở mọi dòng — ta chưa bao giờ đo được hệ số truyền từ 60 câu
> ground truth sang bảng xếp hạng.

### Hạng A — làm ngay, tỷ lệ giá trị/công sức cao nhất

**A1. Bộ phân bổ 100 dòng theo PHỦ XÁC SUẤT (thay `allocate_hybrid_rows`)**
- **Nhắm vào:** nhóm 1 (14 câu thang đặt nhầm chỗ) là đích chính; chạm luôn nhóm 2 (video đúng vào 100 dòng: 55/60 → 58/60).
- **Điểm kỳ vọng:** **+10,0%** (số của repo, đường sản xuất đầy đủ, 4 họ hạt giống × 48 lần bốc). Công sức: 4–6 giờ người, <2 phút máy mỗi lần đo.
- **Cách đo trước khi tin:** (a) chạy `scripts/experiment_phu_xac_suat.py` trên **cả 60 câu**, không `--limit`; (b) **ba mô hình đáp án**, không chỉ một — rút thăm đều trong ô Voronoi, Gauss σ=12 frame quanh keyframe, và đúng bằng frame ground truth. **Điều kiện bật: không âm ở cả ba, dương ≥8% ở hai mô hình đầu.** Cổng này là bắt buộc: bản *ngây thơ* của chính ý tưởng (bỏ khối lượng điểm trên keyframe) cho +29,6% ở mô hình rút thăm đều nhưng **−34,4%** khi đáp án đúng bằng keyframe — đúng cái bẫy đã sinh ra "n_flat=100 đạt 0,562"; (c) 6 họ hạt giống, đòi ≥5 lần độ lệch chuẩn; (d) đúng 100 dòng, 0 dòng ngoài `[0, last_frame]`, `verify_submission_zip` xanh; (e) **xem tay 6–7 câu bị xấu đi**.
- **LÝ DO MẠNH NHẤT ĐỂ KHÔNG LÀM — và nó là việc phải xử lý trước, không phải lý do bỏ:** `scripts/build_review_page.py` **cài lại bộ phân bổ bằng JavaScript**, và `tests/test_js_allocator.py::test_hybrid_allocation_matches_row_for_row` ghim hai bản phải khớp **từng dòng**. Tôi mở dòng 1418: trang nhúng ứng viên dưới dạng `[video_id, frame_idx, last_frame]` — **KHÔNG CÓ ĐIỂM SỐ**, mà bộ tham lam cần `p(k)=softmax(điểm/τ)` và biên ô Voronoi. Tệ hơn: người soát **kéo thả** để xếp lại ứng viên — sau khi kéo ứng viên #7 lên đầu thì `p(k)` là cái gì? **Không xác định.** Bộ cũ tiêu thụ một *hoán vị* nên hợp tay người; bộ mới tiêu thụ một *phân bố* nên không. Nếu không port kịp, đường xuất zip trong 3 tiếng thi sẽ **lặng lẽ khác** đường đã đo — đúng lỗi "hai bộ xếp hạng" mà chính ta tố cáo ở chỗ khác.
- **Cảnh báo trung thực về con số:** hai lane nghiên cứu độc lập báo **+28,5%** và **+32,2%** cho cùng ý tưởng, nhưng trên nền thấp hơn (0,2617 và 0,2731 so với 0,3496 của repo) và với tham số tiên nghiệm khác. **Tôi lấy +10,0% làm con số chính thức** vì nó đo trên đúng đường sản xuất với nền khớp tài liệu. Coi +28% là hy vọng trên, không phải dự báo. Ngoài ra mức tăng **suy giảm đơn điệu** khi đáp án tập trung về keyframe (+28,5% → +18,9% → +9,0% qua ba mô hình đáp án), mà ta **không biết** đáp án thật của BTC nằm ở đâu trên trục đó — 56/60 frame ground truth nằm đúng trên một keyframe, nên ta có **zero bằng chứng thực nghiệm** về phân bố bên trong ô.

**A2. Sửa lỗi: ứng viên từ lời thoại bị VLM chấm rồi vứt đi**
- **Nhắm vào:** nhóm 2 (video không có trong 100 dòng).
- **Điểm kỳ vọng: 0,0000 trên 60 câu** (đã đo — vì video đúng đã có trong pool ở 60/60 câu). Giá trị là bảo hiểm cho đề thật: **[PHỎNG ĐOÁN]** 1–2 câu/vòng, căn cứ là p1-19 ("Rạch Giá") và p1-22 chỉ được kênh này tìm ra. Công sức: **30–45 phút**.
- **TÔI ĐÃ XÁC MINH LỖI TRONG MÃ NGUỒN, đây là sự kiện chứ không phải ý kiến.** `scripts/vlm_rerank_run.py:283` gọi `judge.score(text, cands + extras)` — tức **trả tiền Gemini** cho ứng viên lời thoại — rồi dòng 295 xếp lại bằng `sorted((h.score + bonus(h), i, h) for i, h in enumerate(hits))`, duyệt `hits` là đúng 400 ứng viên **hình ảnh** ban đầu. Video do lời thoại tìm ra mà nằm ngoài top-400 được chấm điểm rồi **bị vứt hoàn toàn**. Chính khối chú thích trên đầu hàm lại tuyên bố ngược ("adds candidates rather than adding score").
- **Cách đo:** sau khi vá, chạy `evaluate_official.py` 60 câu, đòi **không giảm quá 1 độ lệch chuẩn** (0,0007); rồi chạy một vòng thật và kiểm bằng mắt rằng video lời thoại tìm ra **có** trong CSV.
- **Lý do mạnh nhất để không làm (theo cách hữu ích):** bản vá "chèn vào vùng hạng ~24" chính là thao tác đã làm 0,321 tụt xuống 0,305, vì nó đẩy lùi các ứng viên **có thang**. Chỉ có hai lựa chọn trung thực: **chèn CUỐI** (an toàn, đo được 0, có ích trên đề thật), hoặc **tắt `--from-speech`** để thôi trả tiền API. **Đừng chèn vào giữa.**

**A3. Thu hồi token HuggingFace — 5 phút**
- **Nhắm vào:** không nhóm nào. Bảo hiểm thuần.
- Tôi mở `.secrets-revoked.txt` hôm nay: nó ghi **"ĐANG CHỜ XỬ LÝ"** và ba vân tay vẫn bị chú thích lại. **Mã nguồn đã dọn** (`stream_upload_hf.py` giờ đọc `HF_TOKEN` từ môi trường) — nhưng **token chưa hề được thu hồi** ở phía HuggingFace, mà lịch sử git trên hai remote công khai vẫn đọc được nó. Đây là token **GHI** vào chính kho ảnh mà trang duyệt + VLM + OCR đều phụ thuộc.
- **Cách đo:** thử ghi bằng token cũ sau khi thu hồi, phải bị từ chối; rồi bỏ dấu `#` ba dòng vân tay để CI xanh lại.
- **Lý do để không làm:** không có. 5 phút, không đụng dòng mã nào trên đường thi.

**A4. Sửa phép đo độ phủ lời thoại — con số 24,9% đang lưu hành là SAI**
- **Điểm kỳ vọng: 0.** Đây là sửa *phép đo*, không phải sửa *hệ thống*. Công sức: **15 phút**.
- **Tôi đếm hôm nay:** `data/captions` đúng là chỉ có **217/873** file có nội dung. Nhưng `vlm_rerank_run.py:100` đặt `--transcripts` mặc định = `ROOT.parent/transcripts_full`, và `D:/AI2026TOP1SV/transcripts_full` có **849/873 file có segment thật**. **Độ phủ sản xuất là 97%, không phải 25%.** Mọi phán quyết âm về kênh lời thoại đang bị đổ lỗi cho "chỉ mục mù 75% kho" là **đổ oan**. Đo lại với chỉ mục đúng: hạng video đúng top-1 3→8/60, top-5 8→17/60, top-10 13→22/60, có mặt 29→57/60 — **mạnh gấp đôi** con số đang được dùng để kết tội nó.
- **Không cần tải gì, không cần chạy Whisper.** Nhánh "tải ~8 GB audio + 15–20 giờ CPU Whisper" đã hết lý do tồn tại.
- **Nhưng phán quyết về MỐC THỜI GIAN vẫn đứng** kể cả với chỉ mục đầy đủ: sai trung vị 2.850 frame, chỉ 1/55 câu trong ±150 frame. **Kênh này vĩnh viễn chỉ là bộ lọc CẤP VIDEO — tuyệt đối không cho nó chạm vào việc chọn frame.**

**A5. Sao lưu phòng thí nghiệm đo lường ra khỏi thư mục TẠM**
- **Điểm kỳ vọng: 0.** Công sức: **5–10 phút**, một lệnh copy.
- Toàn bộ bằng chứng của bốn lane nằm trong thư mục `Temp/claude/.../scratchpad` — **thư mục tạm theo phiên, trên ổ C, ngoài git**. Trong đó có `sig_scores60.npy` (ma trận điểm 177.321×60), `prodhits_uniq7841.json` (cache `ranked_hits` sản xuất — chính thứ cho phép chạy cổng sản xuất trong 19 giây thay vì nạp chỉ mục 817 MB), cùng ~20 script đo. Khoảng 215 MB. Copy sang `data/_lab/` và thêm vào `.gitignore`. **Đây là loại mất mát chỉ phát hiện ra đúng lúc đang cần.**

**A6. Ghi ba phán quyết ĐÓNG HƯỚNG vào tài liệu — 1,5 giờ**
- **Điểm kỳ vọng: 0. Tiết kiệm nhiều ngày người.** Chi tiết ở §6.

### Hạng B — làm trong tuần, có cổng

**B1. OCR toàn kho làm nguồn ứng viên có định vị FRAME chính xác**
- **Nhắm vào:** nhóm 3 (keyframe ngoài 400 ứng viên) và nhóm 2. **Đây là kênh DUY NHẤT trả về một FRAME chứ không phải một video** — tức kênh duy nhất tấn công phần 60% điểm mất ở vị trí frame bằng thông tin nằm **ngoài** encoder.
- **Điểm kỳ vọng: [PHỎNG ĐOÁN] +2% đến +5%** (1–2 câu), sai số rất rộng vì bằng chứng chỉ có 7/60 câu. Công sức: 4 giờ người + **25 giờ máy** (8 tiến trình) hoặc ~9 USD Gemini Flash-Lite, **cộng thời gian tải 177k ảnh** (trên đĩa chỉ có ~2,8% ảnh keyframe).
- **Căn cứ:** 46/60 khung đáp án **có chữ đọc được**; 7/60 chứa ít nhất một từ nội dung của câu hỏi. Trong đó có **qi=48** (Lxx_Vnnn, thầy giáo dạy Lịch sử) — một trong 5 câu **mất video**, hiện 0,000 điểm — với 39 dòng OCR và 6 từ trùng.
- **Cách đo — HAI CHẶNG, đừng làm cả kho ngay:** Chặng 1 (nửa ngày): OCR riêng ~16.000 keyframe của 60 video ground truth, dựng chỉ mục ngược BM25, hỏi *"chỉ mục có đưa được keyframe đúng vào top-3 ứng viên không, ở bao nhiêu câu?"*. **Không đạt ≥4/60 thì DỪNG, đừng OCR cả kho.** Chặng 2: chỉ khi qua chặng 1, đòi vượt nền quá 2 độ lệch chuẩn.
- **LÝ DO MẠNH NHẤT ĐỂ KHÔNG LÀM:** đề xuất muốn chèn vào **hạng 0–2**, ô đắt nhất trên bàn cờ, dựa trên bằng chứng 7/60 câu. Mà chính mẫu cho thấy chữ trên khung đáp án phần lớn là **đồ đạc của kênh** chứ không phải nội dung — qi=5 ra "Online NGÓn", qi=17 "tuitreTV", qi=22 "H HỂ THAO" — và **không câu nào** bắt được danh từ riêng mà câu hỏi nêu. Tệ hơn, qi=41 đọc ra 39 ký tự về **một bản tin hoàn toàn khác** (dòng chạy chân màn hình): OCR sẽ sinh **khớp sai tự tin**. Chèn khớp-sai-tự-tin vào hạng 1 là hành động nguy hiểm nhất có thể làm dưới luật chấm này, vì R@1 có hệ số 1,00 và chỉ có một ô.

**B2. Đổi kênh tín hiệu phụ: từ cộng điểm xếp hạng sang TIÊN NGHIỆM của bộ phân bổ**
- **Nhắm vào:** nhóm 2 chính, nhóm 1 phụ. **Chỉ khởi động được SAU khi A1 vào sản xuất.**
- **Điểm kỳ vọng: [PHỎNG ĐOÁN] +2% đến +5%, hoàn toàn chưa đo.** Công sức: 3–4 giờ người + ~2 giờ máy quét hệ số.
- **Lập luận kiến trúc:** hiện mọi tín hiệu phụ (objects w=0,01, VLM w=0,02, lời thoại, OCR) đều đi qua **một cửa duy nhất** — cộng điểm rồi sắp lại — mà cửa đó có **tổng bằng không**: đẩy cái này lên là đá cái kia xuống, và 30 chỗ đầu là hữu hạn. Đó **chính là** cơ chế sinh ra bốn lần "video R@1 tăng, điểm thi giảm". Bộ phân bổ theo phủ tiêu thụ một **phân bố**, không tiêu thụ **thứ tự**, nên nó mở một cửa thứ hai không có tính chất tổng-bằng-không. Ứng viên chỉ do lời thoại/OCR tìm ra được cấp khối lượng nhỏ nhưng khác không, và bộ tham lam tự cấp cho nó một hai dòng ở đuôi mà **không cướp dòng nào** của ứng viên mạnh.
- **Cách đo:** quét **TỪNG hệ số một**, các hệ số khác bằng 0, ba mô hình đáp án, 6 họ hạt giống. Giữ riêng chỉ báo "số câu có video đúng trong 100 dòng" — tụt khỏi 58/60 là dấu hiệu đang cướp chỗ. Với riêng **lời thoại và OCR**, tiêu chí phải là **"không âm"** chứ không phải "dương".
- **Lý do mạnh nhất để không làm bây giờ:** bốn hệ số quét trên 60 câu là **công thức khớp nhiễu**, và dự án này đã có ba lần khớp nhiễu thắng trên máy rồi thua trên bảng. Ngoài ra 60 câu ground truth **toàn mô tả cảnh nhìn thấy**, nên bộ đo về cấu trúc **không thể** cho lời thoại/OCR điểm dương — ta sẽ quét bốn hệ số trên một bộ đo mù với hai trong bốn.

**B3. Dựng lại chỉ mục cho batch 2 — giống hệt cách cũ**
- **Nhắm vào:** nhóm 2 và nhóm 3 ở dạng nghiêm trọng nhất. **Điểm kỳ vọng: 0 tăng; tránh SỤP.**
- **Cách đo:** sau khi dựng xong, chạy lại 60 câu ground truth **CHỈ TRÊN PHẦN BATCH 1** của chỉ mục mới. Đòi trùng 0,345 ± 0,0014. Lệch ra ngoài ngưỡng đó nghĩa là **chỉ mục dựng sai**, không phải mô hình đổi tính. Đây là phép thử hồi quy duy nhất bắt được lỗi lệch thứ tự hàng.
- **Lý do để chưa làm hôm nay:** BTC **chưa công bố link**, nên không có URL để thêm. Đây là việc **chờ sẵn**, không phải việc làm ngay. **[PHỎNG ĐOÁN]** kho lớn gấp đôi nghĩa là gấp đôi vật gây nhiễu, nên điểm **có thể** tụt kể cả khi chỉ mục dựng đúng hoàn hảo.
- **RỦI RO TÂM LÝ LỚN NHẤT:** *"đằng nào cũng dựng lại thì đổi luôn encoder"*. **Cấm ghép hai thay đổi.** Đổi encoder cùng lúc với đổi kho làm mất điểm gốc so sánh.

**B4. Xoá 5.207 dòng ngoài đường thi (hoặc sửa `app.py`)**
- **Nhắm vào:** nhóm 2, qua một cơ chế cụ thể chứ không phải "cho sạch": `app.py` và `query_kis.py` dùng `src/task1_kis/retriever.py`, một **bộ xếp hạng KHÁC** với `ranked_hits` (giới hạn 2 frame/video, ép cách nhau 10 giây — đã đo là tệ hơn ở mọi bề rộng cửa sổ). Ai mở web app để kiểm tra một câu đang xem **danh sách video khác** danh sách sắp nộp, rồi kết luận "video này có trong top rồi" trong khi bài nộp thật không có nó. **Sai kiểu này hoàn toàn vô hình.**
- **Cách đo (biến lập luận thành số):** TRƯỚC khi xoá, lấy 10 câu, so top-20 video mà `app.py` trả với top-20 mà `ranked_hits` trả, đếm số câu khác nhau. Lớn hơn 0 thì rủi ro là thật.
- **Lý do mạnh nhất để không XOÁ:** nếu có người trong đội đang dùng web app, xoá nó là lấy đi một công cụ giữa kỳ thi. **Phương án SỬA** (cho `app.py` gọi `ranked_hits`) tốn thêm 1 giờ, diệt được sai lệch **và** giữ công cụ. Và `src/task2_vqa` (1.882 dòng) gần như không có test bảo vệ, không liên quan gì tới ba nhóm thất bại — nó chỉ là dọn dẹp, mà dọn dẹp giữa kỳ thi là rủi ro thuần. **Nếu làm A1 thì HOÃN cái này** — A1 phải sửa `build_review_page.py` và bộ test JS; xoá 5.200 dòng cùng lúc làm mọi lỗi test không truy được nguyên nhân.

### Hạng C — chỉ nếu còn nhiều thời gian, kỳ vọng thấp

**C1. Tách truy vấn thành "bối cảnh" và "khoảnh khắc"** — **[PHỎNG ĐOÁN] 0% đến +3%**, kỳ vọng ở đầu thấp. **Chỉ chạy CỔNG RẺ trước (30 phút):** xếp lại nội-video bằng riêng prompt tiếng Anh (đã đo +1,6%) và xem có qua nổi cổng ba-mô-hình-đáp-án không. Không qua thì tầng frame trơ với việc đổi vector, bỏ cả hướng, tiết kiệm 4 giờ. **Lý do mạnh nhất để không làm:** trần bị chặn ở tầng **encoder** chứ không ở tầng truy vấn — khả năng phân biệt trong cảnh của SigLIP-2 chỉ **60,3%** (ngẫu nhiên 50%) và yếu đều ở mọi thang thời gian (64,5% ngay cả khi so "gần <200 frame" với "xa >800 frame"). Hỏi khéo hơn một encoder gần như mù trong cảnh chỉ được thêm chút ít. Thêm nữa nó cắm một **phụ thuộc mạng** vào đường chạy chính trong 3 tiếng thi.

**C2. TRAKE: mở rộng bộ phân bổ theo phủ sang không gian tích 4 sự kiện** — **[PHỎNG ĐOÁN] +0% đến +2%** trên tổng điểm (TRAKE chỉ **3/24** câu vòng 1: p1-4, p1-16, p1-18 — tôi đếm trong `round_p1/queries`, trần tuyệt đối 12,5%). Quan sát lõi đúng và chưa ai khai thác: R-Score TRAKE là điểm **từng phần** theo sự kiện, nên lưới bù trừ đều 5×5×5×5 (phủ 16%) đang đối xử mọi tổ hợp như nhau trong khi luật cho phép trúng 3/4 vẫn được 0,75. Ràng buộc đơn điệu `f1<f2<f3<f4` giải quyết bằng **cấu trúc** cái mâu thuẫn p1-16 (E1 giây 9,9, E2 giây 79,9 — bất khả về vật lý) thay vì bằng một tham số phải chỉnh. **Lý do mạnh nhất để không làm: KHÔNG CÓ BỘ ĐO.** Repo không có ground truth TRAKE thật, và benchmark tổng hợp duy nhất **đã nói dối một lần rồi** (mu càng lớn càng tốt trên tổng hợp nhưng phá mạch lạc thời gian trên dữ liệu thật). Tệ hơn: **8/12 sự kiện nằm dưới ngưỡng nhiễu** — khi `p_j` thực chất là nhiễu thì "phân bổ theo độ bất định" chỉ là rải đều bằng một cách phức tạp hơn. Phép tham lam không tạo ra thông tin không có sẵn.

**C3. TRAKE: dùng VLM có ĐẦU VÀO VIDEO chốt mốc thời gian** — **chỉ chạy phép mô phỏng rẻ (2 giờ), đừng gọi API.** Giả lập "biết đúng giây nhưng không biết đúng khung": làm tròn frame đáp án về bội số fps rồi rải thang, chấm bằng công thức chính thức. Nếu ngay cả oracle-cấp-giây cũng không nâng điểm thì hạn chế MM:SS đã giết ý tưởng, biết sau 2 giờ chứ không sau 12. **Lý do mạnh nhất để không làm bản đầy đủ:** mốc thời gian của Gemini chỉ chính xác **cấp giây** = 25 frame, **rộng gấp 2,5 lần** cửa sổ đáp án 10 frame. VLM không thay được thang, nó chỉ **đặt thang đúng chỗ** — mà đó chính là thứ A1 làm với chi phí 2 phút máy, không cần mạng.

**C4. Nâng `RETRIEVE_TOP_N` 400 → 800** — **đúng 0,0% tự thân** (đã đo, có lý do cơ học: bộ phân bổ chỉ với tới trung vị ~32 ứng viên). Độ phủ pool đi từ 54/60 lên 57/60 rồi **chững** suốt tới độ sâu 60.000, nên 800 đúng là điểm dừng chứ không phải tham số phải quét. **ĐỪNG BẬT MỘT MÌNH:** nó làm bộ cộng điểm objects chạy trên 800 hit thay vì 400, có thể đổi thứ tự 30 hạng đầu — đúng chỗ dự án đã thua bốn lần. Một thay đổi 0 điểm mà có rủi ro khác 0 là **lỗ ròng**. Chỉ bật trong **cùng một commit** với B1.

---

## 4. Lộ trình

### Trong MỘT BUỔI (4–6 giờ) — làm đúng theo thứ tự này

1. **A3** — thu hồi token HF (5 phút). Làm trước tiên, không chờ ai.
2. **A5** — copy phòng thí nghiệm ra `data/_lab/`, thêm `.gitignore` (10 phút).
3. **A4** — sửa `--transcripts` trong các script đo trỏ về `transcripts_full`, chạy lại (15 phút). Xoá con số "24,9%" khỏi mọi tài liệu.
4. **A2** — vá lỗi `extras` bị vứt trong `vlm_rerank_run.py`, **chèn vào CUỐI**, chạy 400 test (45 phút).
5. **A1 chặng 1** — chạy `experiment_phu_xac_suat.py` trên cả 60 câu với **ba mô hình đáp án** và 6 họ hạt giống (2–3 giờ, phần lớn là máy chạy). **Không đụng vào `submission.py` trong buổi này.**

Kết thúc buổi: có câu trả lời dứt khoát cho A1, đã bịt hai lỗ hổng thật, và không rủi ro nào được đưa vào đường thi.

### Trong MỘT TUẦN

1. **A1 chặng 2** — nếu qua cổng: đưa bộ phân bổ mới vào `submission.py` **sau một cờ** (`ALLOC=coverage`), lùi lại bằng một biến môi trường trong 10 giây. Giữ đường cũ làm mặc định cho tới khi port xong JS.
2. **A1 chặng 3 — PORT JAVASCRIPT. Đây là chặng khó nhất và không được bỏ qua.** Phải thêm `score` vào payload dòng 1418 của `build_review_page.py`, thêm biên ô Voronoi, và **quyết định xem kéo-thả nghĩa là gì**. Đề nghị: khi người soát kéo thả, gán lại `p(k)` theo thứ hạng mới bằng một thang hình học cố định (`p_r ∝ γ^r`), tài liệu hoá rõ rằng đó là quy ước chứ không phải điểm mô hình. Chạy `test_hybrid_allocation_matches_row_for_row` cho bản mới.
3. **A6** — viết ba phán quyết đóng hướng vào `docs/KIEN_TRUC_VA_HUONG_CAI_THIEN.md` (1,5 giờ). Sửa luôn bốn con số sai trong tài liệu (§6).
4. **B1 chặng 1** — OCR 16.000 khung của 60 video ground truth, dựng chỉ mục, chạy cổng ≥4/60.
5. **B4** — sửa `app.py` gọi `ranked_hits` (không xoá), **chỉ sau khi A1 đã ổn định**.
6. Diễn tập một vòng thi đầy đủ với đường mới, có bấm giờ.

### NẾU CÒN NHIỀU THỜI GIAN

1. **B2** — đổi kênh tín hiệu phụ sang tiên nghiệm. Quét từng hệ số một.
2. **B1 chặng 2** — OCR toàn kho (25 giờ máy) nếu chặng 1 qua cổng.
3. **C1 cổng rẻ** (30 phút) — nếu không qua thì đóng hướng vĩnh viễn.
4. **C3 phép mô phỏng** (2 giờ) — chỉ mô phỏng, không gọi API.
5. **B3** — dựng chỉ mục batch 2 ngay khi BTC công bố link. **Đây là việc chờ sẵn, ưu tiên tuyệt đối khi link xuất hiện.**
6. **C2** — TRAKE, và **chỉ sau khi đã dựng 5–8 câu TRAKE có đáp án chốt bằng mắt**. Không có bộ đo thì không làm.

---

## 5. KHÔNG NÊN LÀM — nghe hay nhưng dữ liệu nói không

| ý tưởng | dữ liệu nói gì | nguồn |
|---|---|---|
| **Nộp dày hơn (step 5, step 3)** | −6,1% và −11,7%, lệch 32–53 lần sai số. Cửa sổ rộng ít nhất 13 frame nên thang step=10 **đã kín không hở**; giảm step chỉ **cắt đôi tầm với** mà không thêm độ phủ nào | 6 họ hạt giống × 48 lần bốc |
| **Trích thêm keyframe (dày gấp 2/4/8)** | Với bộ truy hồi **mù** (đúng thực tế): −0,1% / −2,5% / −2,4%. Toàn bộ "lợi ích mật độ" trong mô phỏng ngây thơ đến từ **lén dùng đáp án** để chọn khung. Với p=0,60 đo thật thì dày gấp đôi chỉ +5,6%. **0/18 câu ~0 điểm được cứu bởi riêng mật độ, kể cả ở mật độ vô hạn** | `scratchpad/free.py`, `sim2.py` |
| ↳ và chi phí của nó | phải tải lại **130,7 giờ video** từ `watch_url` (không có .mp4 nào trên máy), rồi **469 giờ = 19,5 ngày** nhúng SigLIP-2 trên CPU máy này. Nút cổ chai là **GPU**, không phải đĩa hay mạng | đo thật 0,42 ảnh/giây |
| **Hợp nhất CLIP B/32 vào truy xuất** | Ba lane đo độc lập, cả ba âm hoặc nhiễu: z-score −0,3% đến −13,0%; đường cong **không đơn điệu** (−0,9/+1,4/+1,5/−0,2/−1,5/−4,9%); RRF làm video R@1 tụt 26→21. Ở đỉnh w=0,20 thì **12 câu tốt lên, đúng 12 câu xấu đi** — xáo trộn, không phải cải thiện | 3 lane độc lập |
| ↳ và nó bẫy được cả độ đo mới | hạng-1 nội-video tăng 43,3→46,7%, top-5 68,3→73,3%, vR@5 39→45 mà **điểm thi không theo**. Cái bẫy "chỉ báo lên, điểm xuống" **lần thứ tư** | — |
| **Họ Moment-DETR / QD-DETR / UniVTG / CG-DETR** | Nhánh training-free của chính họ này đo ra −10,9% đến −45,5% (Gauss σ=0,75/1,5/3: −12,3/−29,7/−45,5%; REZE+Kadane −33,2%; RRF −7,6 đến −8,6%). Số câu keyframe đúng hạng 1 nội-video **rơi 26/54 → 7–13/54**. Lý do gốc: cả họ tối ưu mIoU của **đoạn** dài vài giây, cửa sổ chấm của ta **dưới 0,4 giây** — lệch 1–2 bậc độ lớn. Làm mượt là thành phần **sống còn** của họ đó và là thứ **phá huỷ** chính xác thời điểm ta cần | `scratchpad/tg.py`, `tg2.py` |
| **Đổi encoder sang PE-Core / MetaCLIP2 BÂY GIỜ** | Đổi hai tổn thất **đã biết** lấy một mức tăng **chưa đo**: cửa sổ văn bản 32 token (so với 64 hiện tại, mà **13/24 câu vòng 1 đã tràn 64**) và mất kênh tiếng Việt (đo được đáng 26→23 video R@1). Bằng chứng ủng hộ là MSR-VTT — benchmark **lấy trung bình 8 khung/video**, nên nó **không đo một chút nào** năng lực xếp hạng khung TRONG video, tức đúng chỗ 60% điểm đang mất | — |
| ↳ MetaCLIP2 riêng | trần **+1/60 câu** (hợp top-400 hai mô hình: 55/60 so với 54/60), +0 ở tầng video; và **chậm hơn** mô hình hiện tại trên CPU (0,252 vs 0,386 ảnh/giây) | — |
| **Đổi trọng số 4-prompt / bỏ tiếng Việt** | SigLIP-2 vốn **đã đa ngữ** và `kis_engine.py` **đã** đưa nguyên văn tiếng Việt vào vector ở trọng số 0,35. Quét: 0,00→23 \| 0,15→24 \| 0,25→25 \| **0,35→26** \| 0,50→25 \| 0,65→23 \| 0,80→23. **Đang ở đúng đỉnh, không có món hời nào** | `scratchpad/vi_vs_en.py` |
| **Cộng điểm theo FRAME (bất kỳ tín hiệu nào)** | Đã hỏng **bốn lần** với bốn tín hiệu khác nhau. Hệ quả cấu trúc, không phải trùng hợp: đẩy một frame lên vì "đúng nội dung" **luôn** đá văng một frame **cùng video** vốn gần khoảnh khắc đáp án hơn. Bất kỳ đề xuất mới nào cộng điểm theo frame sẽ vấp đúng chỗ này, và video R@1 sẽ **báo tin vui giả** | bảng §2 tài liệu kiến trúc |
| **`n_flat=100` (chỉ nộp keyframe)** | Chấm bằng GT thô đã snap: 0,562 ("tốt nhất"). Chấm sau khi rút thăm lại khoảnh khắc: **0,257** so với 0,333 của n_flat=30. **Kết luận đảo ngược hoàn toàn.** Đây đúng kiểu sai số đã tạo ra "Top-1 41,67% trên máy mà 5,8 trên bảng" | `experiment_allocation.py --snapped` |
| **Quét lại `n_flat × depth_cost × step`** | **Đã cạn.** Ba phép quét độc lập đều kết luận (30 / 0,5 / 10) là đỉnh trong họ đó. Cấu hình tốt nhất tìm được (28/1,0/14) chênh 1,5% — trong nhiễu. *Ghi chú: A1 không mâu thuẫn với điều này vì nó bỏ hẳn khái niệm chi phí, tức nhảy sang một họ khác* | `experiment_allocation.py` |
| **Đặt nhiều đáp án Q&A trên các dòng khác nhau** | ±0,0%. Đáp án thứ nhất đúng 81%; đáp án thứ **hai** cứu thêm **đúng 0%**. Lỗi là **độ cụ thể** chứ không phải mơ hồ ("Cá cơm" → "Cá nhỏ/Tôm nhỏ"), nên thêm phương án không bao giờ cứu được. **Hệ quả tích cực: bộ trả lời Q&A KHÔNG phải chỗ nghẽn — dồn hết công vào tìm đúng frame** | `experiment_answer_hedge.py` |
| **VLM xếp lại frame bên trong video** | −4,6%, và −34,7% khi đo lại bằng câu hỏi phân biệt (biến thể **tệ nhất**). Sai mục tiêu: VLM trả lời *"khung này có khớp mô tả không"*, còn thứ quyết định điểm là *"khung này có gần khoảnh khắc đúng nhất không"* | `experiment_sharp_rerank.py` |
| **Metadata video của BTC** | KIS R@1 43,3% → 40,0% (−7,6%). Tín hiệu cấp video, mà đúng video là điều kiện **cần** chứ xa mới là đủ | `experiment_metadata.py` |
| **Giới hạn 2 frame/video, cách ≥10 giây** | Tệ hơn ở **mọi** bề rộng cửa sổ. Tối ưu cho một độ đo **khác** (đa dạng video), không phải độ đo của cuộc thi | `kis_engine.py:15-20` |
| **Dựng "bộ thử oracle đẩy hạng" / "độ đo nội-video" làm cổng** | Công thức chấm **thật** chạy ~11–19 giây cho 60 câu — **không có bài toán tốc độ nào để giải**, nên không có lý do tồn tại cho một chỉ số đại diện. Tệ hơn, chỉ số đó **đã được chứng minh là lừa người dùng** (hợp nhất CLIP làm nó tăng mà điểm không tăng). Đừng chôn quả mìn thứ năm | — |
| **Chạy Whisper trên 130,7 giờ audio** | Hết lý do tồn tại: **849/873 video đã có bản chép lời** trong `transcripts_full`. Con số "656 file rỗng" là đo nhầm thư mục | tôi đếm hôm nay |
| **Cộng điểm bằng lời thoại (mọi biến thể)** | −0,1% đến −23% ở **mọi** cách. Và sai mốc thời gian trung vị **2.850 frame** kể cả với chỉ mục đầy đủ, chỉ 1/55 câu trong ±150 frame. **Kênh này vĩnh viễn là bộ lọc cấp video** — dùng nó để **nới ứng viên**, không bao giờ để cộng điểm hay chọn frame | `experiment_transcripts.py`, đo lại 24/08 |

---

## 6. Tài liệu đang nói sai — phải sửa cùng lúc

Bốn con số đang lưu hành trong `docs/` mà tôi đối chiếu lại thấy sai. Chúng nguy
hiểm vì đang được dùng để **biện minh cho quyết định**:

1. **"Ứng viên hạng 1 được thang vươn tới ±120 frame"** (`KIEN_TRUC_VA_HUONG_CAI_THIEN.md`). Đo thật: **±61 frame** với 11,3 frame id, vì `n_flat=30` ăn mất 30 dòng trước khi quét theo chi phí. Con số ±120 chỉ đúng nếu `max_depth=24` được dùng hết — điều **không bao giờ xảy ra**.
2. **"Một keyframe đơn lẻ trúng cửa sổ 10 frame 13,3% số lần"** và **"thang ±10/±20 nâng trần lên 54,9%"**. Mô phỏng 200.000 lần trên phân bố khoảng cách keyframe thật: **30,8%** và **77,8%**. Con số 13,3% khớp với `W/mean(gap) = 10/69,3 = 14,4%`, tức **lấy trung bình sai chiều** (bất đẳng thức Jensen: `E[W/g] ≠ W/E[g]`). Quan trọng vì chính con số 13,3% đang được dùng để lập luận "trần cứng, phải nhờ người chốt frame" — **trần thật cao gấp đôi**.
3. **"Keyframe đúng đứng hạng 1 trong video 48%, top-5 76%"**. Đo lại trên **mọi** keyframe hợp lệ của video (trung vị 266 keyframe/video): **43,3%** và **68,3%**. Con số 48%/76% có lẽ đo trong phạm vi đã lọt top-400 — phạm vi đó dễ đẹp hơn thực tế.
4. **"Lời thoại phủ 217/873 = 24,9%"**. Sản xuất nạp `transcripts_full` với **849/873 = 97%**. Con số 24,9% là đếm nhầm `data/captions`.

Và **§4.1 của `KIEN_TRUC_VA_HUONG_CAI_THIEN.md` ghi "mở rộng tầm nhìn VLM — giá
trị cao nhất, CHƯA LÀM"** trong khi `--from-speech` đã **đang chạy mặc định = 4**
từ lâu. **Tài liệu đã lệch khỏi mã nguồn.** Nhân tiện: nó đang chạy mà **chưa hề
được đo trên 60 câu** — trong một repo có nguyên tắc xuyên suốt là "không đo thì
không vào đường chấm điểm", đó là mâu thuẫn nội bộ cần đóng.

**Ba phán quyết cần ghi vào tài liệu (mục A6):**
- Đóng họ **DETR/moment-retrieval** — ghi là *"đóng vì chi phí cơ hội"*, **không** ghi *"đã chứng minh vô dụng"*. Bằng chứng là các phép biến đổi đường cong **thủ công**, không phải một checkpoint đã huấn luyện; viết quá mạnh thì lần sau ai đó tìm ra phản ví dụ sẽ bác bỏ luôn cả bảng kết quả âm.
- Đóng **ensemble CLIP B/32** — ghi là *"đóng cửa ensemble RẺ TIỀN với một encoder yếu"* (CLIP B/32 chỉ đạt video R@1 13/60 so với 26/60), **không** ghi *"đóng cửa ensemble"*.
- Đóng **tầng truy hồi** — ghi là *"đóng với câu mô tả cảnh"*, **không** ghi *"đóng"*. Bảy phép đo âm đều chạy trên 60 câu mà **toàn bộ là mô tả cảnh nhìn thấy**; đề thi thật thì không (p1-19 "Rạch Giá", p1-18 "củ năng" — cả kho 873 video chỉ **2 video** nhắc "củ năng"). Tuyên bố "truy hồi đã đóng" dựa trên một bộ đo về cấu trúc **không thể chứa** dạng thất bại đó chính là loại sai lầm ta liên tục mắc, chỉ theo chiều ngược lại.

---

## 7. Ba điều tôi không biết, và không định giả vờ là biết

1. **Đáp án thật của BTC nằm ở đâu trong ô keyframe.** 56/60 frame ground truth nằm **đúng trên** một keyframe, nên ta có **zero bằng chứng thực nghiệm** về phân bố bên trong ô. Toàn bộ giá trị của A1 là hàm của giả định này: +28,5% nếu rút thăm đều, +9,0% nếu đáp án đúng bằng keyframe. Đó là lý do tôi lấy con số thấp và đòi ba mô hình đáp án.
2. **Hệ số truyền từ 60 câu ground truth sang bảng xếp hạng.** Ta đi 5,8 → 8,6 trên bảng trong khi điểm nội bộ đi từ đâu tới 0,345 — không ai lập được ánh xạ. Mọi con số "quy sang bảng" trong tài liệu này đều là phỏng đoán.
3. **Batch 2 lớn cỡ nào.** BTC chưa công bố. Câu "điểm rơi về xấp xỉ một nửa nếu không dựng lại chỉ mục" là **giả định tuyến tính**, không phải đo.

---

## Tóm tắt 10 dòng cho trưởng nhóm

1. **Không xây lại từ đầu.** Chỉ tầng **chỉ mục** phải dựng lại, và chỉ vì **batch 2** (PDF luật trang 3 nói rõ) — đó là nghĩa vụ, không phải cơ hội; phải làm **giống hệt cách cũ**, cấm nhân dịp đổi encoder.
2. **Nhóm thất bại lớn nhất bị đặt sai tên.** Không phải "thang không vươn tới" mà là **"thang đặt nhầm chỗ"** — chỉ 2/15 câu được cứu bằng thang dài hơn, 9/15 cần hơn 100 nấc thang. Mọi đề xuất "thang dài/dày hơn" đều nhắm sai đích và đã đo ra âm.
3. **Việc số 1: bộ phân bổ 100 dòng theo PHỦ XÁC SUẤT.** Đã có sẵn `scripts/experiment_phu_xac_suat.py`, đo trên đường sản xuất đầy đủ: **+10,0%** (0,3496 → 0,3845), video đúng vào 100 dòng 55/60 → 58/60. Hai lane khác báo +28% nhưng trên nền thấp hơn — **tôi lấy +10% làm con số chính thức**.
4. **Chặn đường của việc số 1 là bản JavaScript trong `build_review_page.py`** — trang duyệt nhúng ứng viên **không có điểm số** (dòng 1418) và cho người **kéo thả** xếp lại, mà bộ mới cần một *phân bố* chứ không phải một *hoán vị*. Không port kịp thì zip xuất trong 3 tiếng thi sẽ **lặng lẽ khác** đường đã đo.
5. **Đã tìm thấy một lỗi thật trong sản xuất:** `vlm_rerank_run.py:283` trả tiền Gemini chấm ứng viên lời thoại rồi dòng 295 **vứt hết đi**. Vá mất 30–45 phút; trên 60 câu nó đáng 0,0000 nhưng trên đề thật đó là con đường **duy nhất** để một video ngoài top-400 thành dòng nộp (p1-19, p1-22).
6. **Con số "lời thoại chỉ phủ 24,9% kho" là SAI** — đó là `data/captions`; sản xuất nạp `transcripts_full` với **849/873 = 97%**. Không cần tải gì, không cần Whisper. Nhưng mốc thời gian vẫn sai trung vị 2.850 frame, nên kênh này **vĩnh viễn chỉ là bộ lọc cấp video**.
7. **Token HuggingFace có quyền GHI vẫn CHƯA được thu hồi** (`.secrets-revoked.txt` ghi "ĐANG CHỜ XỬ LÝ"); mã nguồn đã dọn nhưng lịch sử git công khai vẫn đọc được. Nó ghi vào chính kho ảnh mà trang duyệt + VLM + OCR phụ thuộc. **5 phút, làm ngay.**
8. **Buổi đầu tiên:** thu hồi token → sao lưu phòng thí nghiệm khỏi thư mục tạm → sửa phép đo lời thoại → vá lỗi extras → chạy cổng ba-mô-hình-đáp-án cho bộ phân bổ. **Không đụng `submission.py` trong buổi này.**
9. **Đừng làm:** nộp dày hơn (−6 đến −12%), trích thêm keyframe (0/18 câu được cứu, cần 19,5 ngày GPU), hợp nhất CLIP (ba lane đều âm), họ DETR (−12 đến −45%), đổi encoder bây giờ (đổi 2 tổn thất đã biết lấy 1 mức tăng chưa đo), cộng điểm theo frame (đã hỏng **4 lần**), thêm đáp án Q&A thứ hai (cứu đúng 0%).
10. **Bốn con số trong `docs/` đang sai và đang được dùng để ra quyết định:** thang hạng 1 vươn ±61 chứ không phải ±120; trần keyframe là 30,8%/77,8% chứ không phải 13,3%/54,9% (lỗi bất đẳng thức Jensen); hạng 1 nội-video 43,3% chứ không phải 48%; lời thoại phủ 97% chứ không phải 24,9%. **Sửa cùng lúc với việc ghi ba phán quyết đóng hướng.**
