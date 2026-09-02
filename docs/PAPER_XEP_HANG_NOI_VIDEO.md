# Tín hiệu xếp hạng nội-video TRAINING-FREE cho câu MỘT cảnh — khảo sát có mục tiêu hẹp

Chốt 02/09/2026, lane `paper`. **Đọc và đánh giá, không chạy thí nghiệm.** Mọi con
số trong tài liệu này thuộc một trong hai loại: (i) số đã có trong tài liệu nội bộ
(luôn ghi nguồn nội bộ); (ii) số của paper, luôn kèm URL **đã đọc thật** (§8).

> Không mã video, không đáp án trong tài liệu này (docs/ lên GitHub công khai).

---

## 0. Bảng xếp hạng — đọc cái này trước

Bối cảnh bắt buộc: nhóm MỘT cảnh (66/132 mục của bộ đo khớp phân bố) có hạng
nội-video trung vị của keyframe đáp án là **2** (hạng-1: 43%) nhưng còn **+59%**
headroom, và **chưa có tín hiệu xếp hạng nào sống sót**: làm mượt thời gian ÂM,
chuẩn hoá theo video ÂM, ưu tiên đỉnh trơ, điểm cắt cosine liền kề ÂM (TEST
−3,0%/−4,3%), biên cảnh đổi dấu TUNE→TEST, VLM chấm khung thua tín hiệu 0 đồng
(`docs/KE_HOACH_DINH_VI.md` §2.1, `docs/DIEM_CAT_MOT_CANH.md`). Deficit lớn nhất
của nhóm này là **thứ tự dòng** (0,2040), lớn hơn deficit đặt-frame (0,1488)
(`docs/KE_HOACH_DINH_VI.md` §3.2).

Xếp theo (tác động vào tầng XẾP HẠNG 40%) × (khả thi trong ràng buộc: không huấn
luyện, không video gốc, quota Gemini cạn) × (độ tin bằng chứng):

| # | đề xuất | trục thông tin MỚI | chi phí | khả thi | tác động ước tính |
|---|---|---|---|---|---|
| ① | **Khử bias-hub theo bank câu hỏi** (NNN/QB-Norm) | thống kê theo trục **CÂU KHÁC** trên từng khung | 0 API, 0 GPU, ~phút CPU | **5,0** | **4,0** |
| ② | **Ensembling cách diễn đạt** (GQE k=2, majority vote) | trục **CÁCH TẢ KHÁC** của cùng câu | 1 lượt LLM/câu (~$0,3/132 câu gpt-5.2) | 4,0 | 3,0 |
| ③ | **PRF Rocchio 1 bước trên embedding ẢNH** | trục **KHÔNG GIAN ẢNH** (đồng thuận kho) | 0 API, 0 GPU | 4,5 | 2,5 |
| ④ | **Chấm dưới-khung: max-pool trên crop** | trục **DƯỚI-KHUNG** của ảnh ứng viên | ~200 lượt mã hoá ảnh/câu (GPU nhẹ) | 2,5 | 2,5 |
| — | *(cửa đóng mới, §6: shot-pooling/cửa sổ trượt, phủ định bằng ngôn ngữ, TPT, relevance feedback tương tác, các bộ rerank cần huấn luyện)* | | | | |

**Một câu tóm tắt:** đề xuất đầu bảng chạy trên chính ma trận tương đồng đã có,
không cần một byte video hay một lần gọi API nào — và nó là kỹ thuật 2024–2025 có
số **trên đúng SigLIP** (NNN: +3,1 điểm R@1 tuyệt đối cho SigLIP trên COCO).

---

## 1. Vì sao sáu cửa đã đóng ĐỀU đóng — và trục thông tin mới nằm ở đâu

Nhìn lại theo **cơ chế** chứ không theo tên, mọi tín hiệu đã chết ở nhóm một cảnh
đều là **hàm của đúng một đường cong**: điểm `s_q(i)` của MỘT câu hỏi trải theo
khung `i` trong video (hoặc của cosine ảnh–ảnh liền kề, cũng nội-video):

| cửa đã đóng | là hàm gì của `s_q(i)` / ảnh liền kề |
|---|---|
| làm mượt Gauss thời gian | tích chập theo `i` |
| chuẩn hoá theo video (z-score/min-max) | chuẩn hoá trên trục `i` trong từng video |
| ưu tiên đỉnh cục bộ | cực trị theo `i` |
| biên cảnh (đạo hàm bậc một) | sai phân theo `i` |
| điểm cắt cosine liền kề (thang tương đối) | phân vị nội-video của `1 − cos(e_i, e_{i−1})` |
| VLM chấm khung | hàm khác của cùng cặp (câu, khung) — vẫn "khung này khớp mô tả tới đâu" |

Chúng cùng chết vì cùng một lý do cấu trúc: **thông tin trong một đường cong điểm
của một câu đã bị SigLIP vắt gần hết** (trung vị hạng đã là 2), phần còn thiếu
không nằm trong biến đổi của chính đường cong ấy. Vậy tín hiệu mới **bắt buộc**
phải mang thông tin từ một trục khác:

1. **Trục CÂU KHÁC** — khung này điểm cao vì khớp câu này, hay vì nó điểm cao với
   *mọi* câu? (bias-hub; §2)
2. **Trục CÁCH TẢ KHÁC** — cùng khoảnh khắc, tả cách khác thì khung nào vẫn đứng
   đầu? (ensembling diễn đạt; §3)
3. **Trục KHÔNG GIAN ẢNH** — kho nói "câu này trông như thế này"; so ảnh-với-ảnh
   thay vì chữ-với-ảnh. (PRF; §4)
4. **Trục DƯỚI-KHUNG** — vật được tả chiếm góc nhỏ của khung thì embedding toàn
   ảnh pha loãng nó. (crop; §5)

Encoder thứ hai (trục **MODEL KHÁC**) là hướng của lane pe-core, tài liệu này
không lấn — nhưng bốn trục trên đều **trực giao** với nó: nếu pe-core ăn, các đề
xuất ở đây vẫn áp được lên encoder mới y nguyên.

---

## 2. Đề xuất ① — Khử bias-hub theo bank câu hỏi (NNN / QB-Norm / DBSN)

### 2.1 Nguồn đã đọc

- **NNN — Nearest Neighbor Normalization** (EMNLP 2024, MIT/Stanford):
  [arXiv:2410.24114](https://arxiv.org/html/2410.24114v1). Training-free. Với mỗi
  ứng viên `r` trong kho, tính bias `b(r) = α · (1/k) Σ_{j∈top-k} s(q_j, r)` trên
  **k câu hỏi tham chiếu giống `r` nhất** trong một bank câu; điểm mới
  `s_D(q,r) = s(q,r) − b(r)`. Khuyến nghị của chính paper cho bank **lệch phân
  bố**: α = 0,25–0,5, k = 8–16; bank nhỏ chỉ giảm hiệu quả nhẹ. Cải thiện đo được
  trên **đúng SigLIP**: COCO image-retrieval R@1 47,15 → 50,24 (**+3,1 điểm tuyệt
  đối**), Flickr30k +1,9. Giới hạn tự khai: không giúp model cross-attention.
- **QB-Norm** (CVPR 2022): [arXiv:2112.12777](https://arxiv.org/abs/2112.12777) +
  mã [github.com/ioanacroi/qb-norm](https://github.com/ioanacroi/qb-norm) —
  tiền thân, dynamic inverted softmax với β=20, tăng R@1 text-video trên
  MSR-VTT/MSVD/DiDeMo không cần huấn luyện lại.
- **DBSN — Dual Bank Sinkhorn Normalization** (2025):
  [arXiv:2508.02538](https://arxiv.org/html/2508.02538) — bản mạnh nhất của họ
  này (Sinkhorn 10 vòng lặp, thêm target-bank), nhưng chính nó ghi rõ: **suy giảm
  đáng kể khi bank nhỏ**. Với bank ~141 câu của ta (xem dưới), NNN top-k là lựa
  chọn đúng, DBSN là việc thử sau nếu ① sống.

### 2.2 Cơ chế — và nó khác gì thứ đã ÂM

Hiện tượng gốc là **hubness**: trong không gian embedding chiều cao, một nhóm nhỏ
ảnh trở thành "hàng xóm gần nhất của mọi câu hỏi". Trong kho bản tin, ứng viên hub
hiển nhiên: **khung trường quay/người dẫn, đồ hoạ hiệu, bảng chữ, cảnh toàn thành
phố** — những khung "khớp vừa vừa với mọi thứ". Câu một cảnh của BTC phần lớn tả
**hiện trường**; khung hub chen vào giữa hạng 1 và keyframe đáp án là đúng hình
thất bại "trung vị hạng 2, hạng-1 chỉ 43%".

Khác biệt cơ chế với hai cửa đã đóng gần nhất — phải nói rõ vì trông giống:

- **Chuẩn hoá theo video** (ÂM): chuẩn hoá `s_q(i)` **trên trục khung `i`** trong
  từng video, cho MỘT câu. Nó không biết khung nào là hub — mọi thống kê đều lấy
  từ chính đường cong của câu đang hỏi.
- **NNN**: trừ vào từng khung một lượng đo **trên trục câu hỏi** — "khung này hấp
  dẫn mọi câu tới đâu" — tính từ một bank câu **độc lập với câu đang hỏi**. Hai
  phép chuẩn hoá nằm trên hai trục **trực giao** của ma trận (câu × khung). Kết
  luận âm của cái trước không nói gì về cái sau.
- **Ưu tiên đỉnh / biên cảnh** (ÂM/trơ): hàm của hình dạng cục bộ theo thời gian;
  NNN không đụng trục thời gian.

Một cách đọc trực quan: NNN là **IDF cho khung hình** — trừ đi "tần suất tài
liệu" của khung trên tập câu hỏi, đúng như IDF trừ độ phổ biến của từ.

### 2.3 Bank câu hỏi — có sẵn, không rò rỉ

Đã đếm trong repo: **81 câu đề THẬT của BTC** (25 round1 + 26 round_p1 + 30
round2) cộng **60 câu bộ đo cũ** = **141 câu**, không giao với 132 mục của bộ đo
mới. Đó là bank đúng phân bố (chính văn phong BTC) và không rò rỉ đáp án — chỉ
dùng **văn bản câu hỏi**, không dùng nhãn. NNN đã chứng minh hoạt động với bank
lệch phân bố ở α thấp; 141 câu nằm trong vùng k = 8–16 mà paper khuyến nghị.

Biến thể 0-hạ-tầng để đối chứng: **bank thủ công ~10 câu "nền bản tin"** ("người
dẫn chương trình trong trường quay", "bảng đồ hoạ tiêu đề", …) — chính là "negative
prompting" làm bằng số học điểm. §6.2 giải thích vì sao phủ định phải làm bằng số
học, không bằng ngôn ngữ.

### 2.4 Hai biến thể áp dụng — theo đúng hai trục deficit

- **V1 — chọn Ô trong video (đặt-frame):** dùng lại **nguyên xi phép hoán vị
  giữ-đa-tập-điểm** đã chứng minh an toàn (`docs/KE_HOACH_DINH_VI.md` §1.6): trong
  từng video, giữ nguyên tập điểm, gán lại điểm theo thứ tự `s_D`. Bề rộng phủ
  video bất biến theo xây dựng, R@100 bất biến. Trần của cơ chế này cho nhóm một
  cảnh đã tính: **+36,0%** (`KE_HOACH_DINH_VI.md` §5.4).
- **V2 — tín hiệu chấm TỪNG DÒNG (thứ tự dòng):** đúng thứ mà §4.4 của kế hoạch
  đòi ("cần một tín hiệu chấm điểm từng dòng, không phải quy tắc sắp xếp thứ
  bảy"): sắp lại 100 dòng theo `s_D` của ứng viên sinh ra dòng. Tập dòng không
  đổi ⇒ R@100 bất biến. Trần trục này cho nhóm một cảnh: **+74%**
  (`KE_HOACH_DINH_VI.md` §3.2) — trục lớn hơn.

Phạm vi tác động: chỉ câu **không qua cổng hai cảnh** (câu qua cổng đã thuộc lever
hoán vị cảnh B) ⇒ nhóm hai cảnh ra 100 dòng **giống hệt nền** — có nhóm bất biến
để `assert`, đúng cấu trúc rủi ro của lever đã ship.

### 2.5 Đo thế nào (5 cổng) — và phép đếm PHẢI làm trước khi nhìn điểm

1. **Phép đếm tất định trước, 0 lần đọc TEST:** trên cả 132 mục, với mỗi câu một
   cảnh có keyframe đáp án trong pool: đếm số câu mà **bias bank của keyframe đáp
   án < bias của các khung đang đứng trên nó** trong cùng video. Đây là bản sao
   của phép đếm 53%→76% (cảnh B) và 64% (khung neo). Nếu con số này không nghiêng
   rõ (ví dụ ≤ 55%), **dừng ở đây, ghi ÂM, 0 đồng đã tiêu**.
2. Chia TUNE/TEST **phân tầng theo `co_2_canh`**, chọn (α, k, V1/V2) trên TUNE,
   đọc TEST **đúng một lần**, hạt giống MỚI chưa dùng (ví dụ họ 99x000).
3. **Bootstrap theo câu** 4000 lần; báo cáo KTC 95% và P(≤0).
4. `assert`: 66 câu hai cảnh ra 100 dòng giống hệt nền ở mọi cấu hình; α=0 ⇒
   132/132 giống hệt nền.
5. Chấm qua `allocate_rows` thật, cửa sổ {6,10,20}, **song song hai mô hình bốc**
   ĐỀU và SAU_NEO theo luật `docs/MO_HINH_BOC.md` (mẫu:
   `scripts/cong_do_ben_mo_hinh_boc.py`).
6. Điều kiện cần đã học từ ba lever trước: **đường tham số phẳng** — quét α trên
   một dải rộng; nếu điểm chỉ dương ở một đỉnh nhọn (như điểm cắt: +4,5% ở đúng
   w=0,1) thì đó là nhiễu, không ship.

### 2.6 Ước tác động vào tầng XẾP HẠNG 40%

NNN cho SigLIP +3,1 điểm R@1 tuyệt đối (≈ +6,6% tương đối) trên COCO — bài toán
*giữa các ảnh của cả kho*, tương tự trục V2 hơn V1. Chuyển sang bài của ta với
mọi dè dặt (câu máy sinh, kho một miền, bank 141): kỳ vọng thực tế **+3–8% trên
nhóm một cảnh** nếu phép đếm bước 1 nghiêng rõ; quy ra toàn bộ 132 mục khoảng
+1,5–4%. So khung: toàn bộ trần V1+V2 của nhóm là +36%/+74% — đề xuất này không
ăn hết trần, nó là **tín hiệu đầu tiên có cơ sở văn liệu để bước vào** hai trần
đó. Rủi ro chặn cứng ở cả hai biến thể (R@100 bất biến, nhóm hai cảnh bất biến).

---

## 3. Đề xuất ② — Ensembling cách diễn đạt của CHÍNH câu hỏi (GQE)

### 3.1 Nguồn đã đọc

- **GQE — Generalized Query Expansion** (2024):
  [arXiv:2408.07249](https://arxiv.org/html/2408.07249). Phần test-time: LLM sinh
  10 câu tả lại cảnh, **Farthest Query Sampling** giữ câu gốc làm neo rồi chọn
  thêm câu xa nhất; **k=2 là tối ưu** (thắng cả k=10); hợp nhất bằng **majority
  voting** trên danh sách xếp hạng (thắng trộn embedding lẫn trộn điểm). Zero-shot
  CLIP mean-pooling: R@1 **31,2 → 36,4 (+5,2 điểm)**, R@5 +10,2 điểm.
- **Query paraphrasing bằng LLM cho video search** (nhóm VIREO, TRECVID):
  [arXiv:2407.12341](https://arxiv.org/pdf/2407.12341) — cùng họ, training-free,
  tìm-với-từng-bản-tả rồi hợp nhất danh sách.
- **Hai đội đối thủ của CHÍNH giải này đã dùng nó năm 2025** — bằng chứng khả thi
  tại trận, không phải phòng lab:
  [MADTempo, arXiv:2512.12929](https://arxiv.org/html/2512.12929) (GPT-5 phân rã
  câu, 75,4 điểm sơ tuyển, vào chung kết) và
  [EEIoT_newbie, arXiv:2512.06334](https://arxiv.org/html/2512.06334v1) (Gemini
  sinh tập câu tương đương ngữ nghĩa, lấy max theo keyframe).

### 3.2 Cơ chế — khác gì doc2query và cắt-khúc-câu-dài

- **doc2query** (đã đóng): sinh văn bản cho **phía tài liệu** (ảnh) lúc đánh chỉ
  mục. Đây là phía **câu hỏi**, lúc truy vấn.
- **Cắt khúc câu dài** (đang chạy trong `query_similarities`): tách *cùng những
  từ ấy* thành đoạn. Paraphrase sinh **từ vựng mới** — đúng thứ đem lại thông tin
  mới, vì SigLIP nhạy bất thường với cách chọn từ.
- Với **xếp hạng nội-video**: hai khung cùng video, cùng điểm với câu gốc, hiếm
  khi cùng điểm với cả 2–3 bản tả; khung đáp án là khung **bền qua các cách tả**
  (đây là giả thuyết phải đếm — xem 3.3). Majority-vote trên hạng nội-video là
  phép hợp nhất không cần hệ số trộn.

### 3.3 Đo, chi phí, rủi ro

- Chi phí: 1 lượt LLM/câu (sinh một lần, cache theo câu). gpt-5.2 cho 132 câu ≈
  $0,3, dưới trần $3/lane; đề thật ~30 câu/vòng là không đáng kể. Gemini quota
  cạn — nếu dùng phải kiểm bằng một request nhỏ trước (luật trong bối cảnh lane).
- Phép đếm trước: hạng nội-video của keyframe đáp án theo majority-vote(k=2) so
  với theo câu gốc, trên 132 mục — nếu trung vị không nhích khỏi 2, dừng.
- 5 cổng y hệt §2.5. **Khác một điều:** paraphrase tác động cả tầng SINH ứng viên
  (pool 400 đổi) — nên đo hai biến thể tách bạch: (a) chỉ xếp lại nội-video trên
  pool nền (phạm vi hẹp, so được với ①); (b) cả sinh ứng viên (phạm vi rộng,
  nhưng mất nhóm bất biến ⇒ phải báo cáo song song bộ 60 câu cũ làm đối chứng).
- Rủi ro thật: **drift** — bản tả sai kéo cả danh sách lệch (GQE cũng thấy k=10
  thua k=2 vì lý do này); tiếng Việt → chất lượng paraphrase phải soi tay ~20 câu
  trước khi đo; và điểm của bộ đo là câu máy-sinh tả lại đoạn video — paraphrase
  của câu máy-sinh có thể "dễ" hơn đề người viết (ghi vào biên bản khi đọc số).

---

## 4. Đề xuất ③ — PRF Rocchio MỘT bước trên embedding ảnh

### 4.1 Nguồn đã đọc

- PRF cho dense retrieval là kỹ thuật sống khoẻ 2024–2025 ở IR văn bản:
  [LLM-VPRF, arXiv:2504.01448](https://arxiv.org/pdf/2504.01448);
  [PRF đóng gần hết khoảng cách model nhỏ–lớn, arXiv:2503.14887](https://arxiv.org/html/2503.14887v2).
  Bài học nhất quán của cả họ: **được trung bình, chết vì query drift** khi tài
  liệu phản hồi sai.
- Phía CLIP: [Revisiting Relevance Feedback for CLIP-based Interactive Image
  Retrieval, arXiv:2404.16398](https://arxiv.org/html/2404.16398v2) — feedback
  nhị phân của **người** + 1-NN, training-free, +9,5% R@1; **không** thử
  pseudo-feedback tự động. Tức phần "pseudo" cho CLIP là vùng chưa có số — bằng
  chứng yếu hơn ① và ②, xếp sau là vì vậy.
- [MADTempo](https://arxiv.org/html/2512.12929) làm bản thủ công của đúng ý này
  tại trận: lấy ảnh Google Search, mã hoá CLIP, trộn vào truy vấn khi văn bản
  không tả nổi khái niệm hiếm.

### 4.2 Cơ chế — khác gì mọi thứ đã thử

`q' = normalize(q + λ · mean(e_ảnh của top-m khung thuộc CÁC VIDEO KHÁC))`, rồi
xếp lại khung trong mỗi video theo `cos(q', e_i)`. Hai điểm cơ chế:

1. **Vượt khe modality**: `q` nằm phía văn bản của khe text–image; một bước
   Rocchio kéo nó sang phía ảnh — phép so trong video trở thành **ảnh-với-ảnh**
   ("khung nào giống cái mà cả kho đồng thuận là đúng"), nơi SigLIP sắc hơn hẳn
   so với chữ-với-ảnh. Không cửa nào đã đóng làm điều này: tất cả đều giữ nguyên
   `q` và biến đổi `s_q(i)`.
2. **Leave-one-video-out bắt buộc**: khi xếp khung trong video V, phần mở rộng chỉ
   lấy từ các video ≠ V. Thiếu điều này, khung hạng-1 hiện tại của V tự bơm chính
   nó — phép đo thành vòng lặp tự xác nhận.

Tương tác quan trọng: PRF **khuếch đại hub** (khung hub lọt top-m sẽ kéo `q'` về
phía hub). Vậy thứ tự thử đúng là **① trước, ③ sau, và ③ chạy trên điểm đã khử
bias**. Đo ③ độc lập trước ① là tự làm khó số của chính nó.

### 4.3 Đo, chi phí

0 API, 0 GPU — vài phép cộng vector trên embedding mmap đã có. Phép đếm trước:
hạng nội-video của keyframe đáp án theo `q'` so với `q`, quét (λ ∈ {0,25; 0,5;
1}, m ∈ {3, 5, 10}) — chỉ đọc hạng, chưa đọc điểm. 5 cổng y hệt §2.5, biến thể
hoán-vị-giữ-điểm (R@100 bất biến). Kỳ vọng dè dặt: 0–5% nhóm một cảnh; giá trị
lớn nhất của nó là **rẻ tuyệt đối** và cùng đường ống với ①.

---

## 5. Đề xuất ④ — Chấm dưới-khung: max-pool tương đồng trên crop

### 5.1 Nguồn đã đọc

[MTA — MeanShift Test-time Augmentation, CVPR 2024,
arXiv:2405.02266](https://arxiv.org/html/2405.02266v1): 64 view (ảnh gốc + 63
crop ngẫu nhiên), tìm mode mật độ với trọng số inlierness, training-free, +2,05%
trên ImageNet so CLIP zero-shot, nhanh hơn TPT ~3 lần. **Giới hạn phải nói
thẳng: toàn bộ số của MTA là phân loại ảnh; paper không có một bảng retrieval
nào.** Đây là đề xuất có bước chuyển-miền xa nhất trong tài liệu này.

### 5.2 Cơ chế và giá

Câu một cảnh của BTC thường tả **một vật/một người cụ thể** chiếm phần nhỏ khung
hình; embedding toàn ảnh pha loãng vùng đó, còn khung trường quay "khớp toàn
cục" thì không bị pha loãng — một nguồn lỗi xếp hạng mà không biến đổi nào của
`s_q(i)` chữa được. Cách thử rẻ nhất: với top-3 video × top-12 khung (đúng khuôn
khung-được-chấm của lever hoán vị), mã hoá 5 crop + ảnh gốc mỗi khung, lấy
`max(cos)` làm điểm xếp lại, qua phép hoán vị giữ-điểm.

Giá: ~216 lượt mã hoá SigLIP/câu — offline cho 132 mục là vài phút GPU (hoặc
chậm hơn trên CPU); tại trận thêm ~vài giây/câu. Cần ảnh keyframe trên đĩa (có —
`mirror_keyframes.py`). Không tải video, không model mới.

**Vì sao xếp cuối trong các đề xuất:** bằng chứng chuyển miền yếu (phân loại →
xếp hạng nội-video), chi phí trên-mỗi-câu cao nhất, và trước nó chưa có phép đếm
0 đồng nào (phép đếm trước của nó — "keyframe đáp án có thắng ở thang crop-max
không" — đã tốn tiền mã hoá). Chỉ đáng chạy nếu ① và ③ để lại headroom rõ.

---

## 6. Cửa đóng MỚI — kết luận ÂM có căn cứ, đóng trước khi ai bỏ công

### 6.1 Shot-pooling / cửa sổ trượt trên embedding (kiểu VBS) — ĐÓNG THEO TƯƠNG ĐƯƠNG CƠ CHẾ

Câu hỏi đặt ra cho lane: "các đội VBS làm temporal context không cần model mới —
pooling nhiều keyframe liền kề thành một đơn vị truy xuất?" Trả lời: có, đó là
kỹ thuật phổ biến (VISIONE và các hệ VBS chấm *cặp truy vấn* trên cửa sổ thời
gian: [VISIONE, ACM ICMR 2023](https://dl.acm.org/doi/10.1145/3591106.3592226);
[EEIoT_newbie](https://arxiv.org/html/2512.06334v1) dùng đúng cửa sổ trượt
`0<f₂−f₁<wd` cho đa-cảnh). **Nhưng nó không áp được cho bài của ta, vì hai lẽ:**

1. **Toán:** với embedding chuẩn hoá, `cos(q, mean(e_{i−w..i+w}))` =
   `Σcos(q,e_k) / ‖Σe_k‖` — tử số **chính là** làm-mượt-trung-bình điểm, thứ đã
   ÂM trên TEST (`KE_HOACH_DINH_VI.md` §2.1: làm mượt SigLIP −1,5%, và cơ chế
   "pool rộng ra" được ghi nhận *có thật nhưng trả giá ở nhóm một cảnh*, pool
   58→31). Mẫu số chỉ thêm một trọng số "độ kết dính cụm" bậc hai. Đây là **cùng
   họ cơ chế đã đo**, khác tên.
2. **Mục tiêu:** VBS dùng pooling để tăng **recall của shot** cho người duyệt
   tay; bài của ta là **precision của khung** khi không có người duyệt. Hai hàm
   mục tiêu ngược nhau ở đúng chỗ nhóm một cảnh đang kẹt (chọn 1 khung trong
   một video đã đúng). Còn dạng "cặp truy vấn thời gian" thì bộ đo mới đã đóng
   riêng (+6,7%, KTC chứa 0, hiệu ứng co khi thêm dữ liệu).

### 6.2 Phủ định bằng NGÔN NGỮ trong câu truy vấn — ĐÓNG

[NegBench, CVPR 2025, arXiv:2501.09425](https://arxiv.org/abs/2501.09425): các
VLM đối chiếu (CLIP-họ) xử lý phủ định **ở mức ngẫu nhiên**; sửa được chỉ bằng
finetune trên hàng triệu caption phủ định — tức **không** training-free. Hệ quả
thực hành cho ta: đừng bao giờ viết "không phải trong trường quay" vào truy vấn
SigLIP. Mọi ý "trừ nền studio" phải làm bằng **số học điểm** — chính là biến thể
bank thủ công của ① (trừ `max cos` với các câu-nền), nơi phủ định là phép trừ
tường minh chứ không phải một từ mà encoder không đọc được.

### 6.3 Test-time prompt tuning (TPT và họ hàng) — ĐÓNG

Cần lan truyền gradient qua text encoder **cho từng câu** lúc truy vấn; MTA đã
thắng nó rẻ hơn ở phân loại ([arXiv:2405.02266](https://arxiv.org/html/2405.02266v1)).
Với 30 câu/vòng thi có giới hạn thời gian, chi phí/lợi ích không đứng được ngay
cả khi có số retrieval — mà số retrieval cũng chưa có.

### 6.4 Relevance feedback TƯƠNG TÁC — ĐÓNG cho đường nộp tự động

[arXiv:2404.16398](https://arxiv.org/html/2404.16398v2) cho +9,5% R@1 nhưng cần
**người bấm đúng/sai** giữa hai lượt truy xuất. Đường sản xuất của ta nộp 100
dòng một phát, không có vòng lặp người. Phần dùng được duy nhất là ý tưởng
pseudo — đã thành đề xuất ③.

### 6.5 Các bộ sửa-hub CẦN HUẤN LUYỆN — ngoài phạm vi lane, chuyển pe-core nếu muốn

NeighborRetr (CVPR 2025 — cân bằng hub centrality **trong lúc huấn luyện**,
[openaccess](https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_NeighborRetr_Balancing_Hub_Centrality_in_Cross-Modal_Retrieval_CVPR_2025_paper.pdf))
và mọi reranker cross-attention (BLIP-ITM…): cần train hoặc cần model mới; NNN
còn ghi rõ bias-correction không giúp model cross-attention. Không thuộc
ràng buộc training-free của lane này.

### 6.6 Dò khung người-dẫn kiểu cổ điển (anchor-shot detection) — ĐÓNG lối cũ, giữ lối mới

Văn liệu anchor-shot detection là dòng 2005–2013 (mặt người lặp, phân cụm slice
không-thời-gian — ví dụ
[News videos anchor person detection by shot clustering, Neurocomputing 2014](https://www.sciencedirect.com/science/article/abs/pii/S0925231213006115)):
cần dò mặt, cần khung liên tiếp của video gốc, không có bản training-free trên
embedding có sẵn. Lối đúng cho cùng mục tiêu chính là ①: khung studio **tự lộ**
qua bias bank cao (nó khớp vừa vừa với mọi câu) — không cần bộ dò riêng. Nếu ①
sống, một phép kiểm bằng mắt rẻ: in 50 khung bias cao nhất, xác nhận chúng là
studio/đồ hoạ — bằng chứng cơ chế kiểu "mở ảnh xem" đã dùng cho lever cảnh B.

---

## 7. Trình tự thi hành đề nghị (cho người điều phối — lane này không chạy gì)

1. **①-bước-1** (phép đếm bias, 0 đồng, ~15 phút): quyết định có tiếp không.
2. ① đầy đủ (TUNE/TEST, V1 rồi V2) — script mới, ví dụ
   `scripts/do_khu_hub_mot_canh.py`, theo đúng §2.5.
3. ③ trên nền ① (cùng script, thêm một cờ) — chỉ nếu ① sống.
4. ② phạm vi hẹp (xếp lại nội-video, cache LLM) — song song được với 2–3 vì
   không đụng file.
5. ④ chỉ khi 1–3 xong mà nhóm một cảnh vẫn còn ≥20% headroom xếp hạng.

Nhắc hai điều từ kỷ luật đo: nửa TEST của bộ 132 đã bị đọc ≥4 lần — mọi số TEST
mới của các đề xuất trên phải đọc với hiểu biết đó, và việc sinh thêm GT
(`KE_HOACH_DINH_VI.md` §4.2b) càng làm càng đáng trước khi tiêu lần đọc TEST cho
①. Và mọi kết luận hoà ở n=33/nhóm đọc là "chưa chứng minh được", không phải
"không có".

---

## 8. Nguồn đã đọc (WebFetch thật, 02/09/2026)

| nguồn | URL |
|---|---|
| NNN (EMNLP 2024) | https://arxiv.org/html/2410.24114v1 |
| QB-Norm (CVPR 2022) | https://arxiv.org/abs/2112.12777 · https://github.com/ioanacroi/qb-norm |
| DBSN (2025) | https://arxiv.org/html/2508.02538 |
| GQE (2024) | https://arxiv.org/html/2408.07249 |
| VIREO query paraphrasing | https://arxiv.org/pdf/2407.12341 |
| MADTempo (đối thủ AIC 2025) | https://arxiv.org/html/2512.12929 |
| EEIoT_newbie (đối thủ AIC 2025) | https://arxiv.org/html/2512.06334v1 |
| MTA (CVPR 2024) | https://arxiv.org/html/2405.02266v1 |
| NegBench (CVPR 2025) | https://arxiv.org/abs/2501.09425 |
| RF cho CLIP (2024) | https://arxiv.org/html/2404.16398v2 |
| VISIONE (ICMR 2023) | https://dl.acm.org/doi/10.1145/3591106.3592226 |
| Anchor-shot cổ điển | https://www.sciencedirect.com/science/article/abs/pii/S0925231213006115 |
| PRF dense (2025) | https://arxiv.org/pdf/2504.01448 · https://arxiv.org/html/2503.14887v2 |
| NeighborRetr (CVPR 2025) | https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_NeighborRetr_Balancing_Hub_Centrality_in_Cross-Modal_Retrieval_CVPR_2025_paper.pdf |

Ghi chú trung thực về mức đọc: các dòng NNN, QB-Norm, DBSN, GQE, MTA, MADTempo,
EEIoT_newbie, NegBench, RF-CLIP là **đọc nội dung** (bản html/pdf/README, có số
và công thức trích ở trên); VISIONE, anchor-shot, NeighborRetr, hai bài PRF là
**đọc mức tóm tắt/abstract** đủ cho vai trò phụ trợ của chúng trong tài liệu.

---

## KẾT QUẢ CỔNG TẤT ĐỊNH cho đề xuất ① (NNN) — **ÂM, ĐÓNG** (02/09)

`scripts/dem_bias_hub.py`, 0 API, bank 141 câu (không giao bộ đo 132 mục).

Phép đếm đã công bố ngưỡng trước khi chạy (>55% mới đi tiếp), trên 68 mục mà
keyframe đáp án có mặt trong pool nhưng không đứng hạng-1 nội-video:

| nhóm | khung-trên có bias-bank CAO hơn đáp án | chênh bias trung vị |
|---|---|---|
| MỘT cảnh | 17/34 = **50%** | −0,0001 |
| HAI cảnh | 18/34 = **53%** | +0,0010 |
| **TỔNG** | **35/68 = 51%** | ≈ 0 |

**Kết luận: khung đứng trên đáp án KHÔNG phải hub** — tỷ lệ 51% là tung đồng xu,
chênh bias trung vị bằng 0 tới ba chữ số. NNN không có mục tiêu để sửa trên kho
này. Cửa đóng ở trạm gác rẻ nhất, không tiêu lần đọc TEST nào.

Vì sao kết quả COCO (+3,1 điểm) không chuyển miền được: kho COCO có gallery đa
dạng chủ đề nên hub (ảnh "chung chung") nổi rõ; kho của ta khi xét NỘI-VIDEO thì
mọi khung cùng video đã cùng chủ đề — độ "khớp với mọi câu hỏi" gần như đồng đều
trong video, nên trừ nó đi không đổi thứ hạng nội-video.

→ Đề xuất kế tiếp trong bảng xếp hạng của tài liệu này: ② GQE (LLM query
expansion, ~$0,3/132 câu — hai đội đối thủ của chính giải này đã dùng tại trận).
