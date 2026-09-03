# Tín hiệu xếp hạng nội-video TRAINING-FREE cho câu MỘT cảnh — khảo sát có mục tiêu hẹp

Chốt 02/09/2026, lane `paper`. **Sửa lần 2 cùng ngày**, sau khi cổng tất định của
đề xuất ① chạy xong và ra ÂM (phụ lục cuối tài liệu + §9 đọc lại phạm vi cổng).
**Sửa lần 3, 03/09** — lượt soát độc lập: tìm lại cả bốn hướng được giao bằng
truy vấn mới, đối chiếu lại số của nguồn chịu-tải (NNN — khớp từng chữ số),
bổ sung nguồn mới (§8) và tổng kết lượt soát (§10). Cùng ngày, phiên đo song
song đóng nốt TOÀN TRỤC ① (V2 qua cổng đếm 57% sát nút rồi ÂM ở phép đo đầy đủ,
TEST −1,7% — phụ lục cuối) và trả lời câu hỏi mở của §9.1 bằng phép đếm bão hoà
(nhóm một cảnh = trận tay đôi hạng 1↔2). Bảng xếp hạng §0 và trình tự §7 đã
viết lại theo cả hai kết quả ấy. Bản đồ bốn trục đề xuất KHÔNG đổi — lượt soát
hội tụ về đúng bản đồ cũ.
**Đọc và đánh giá, không chạy thí nghiệm.** Mọi con
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
luyện, không video gốc, quota Gemini cạn) × (độ tin bằng chứng). **Bảng đã cập
nhật lần 3, sau khi TOÀN TRỤC ① đóng ngày 03/09** (V1 ÂM 51% ngày 02/09; cổng
V2 qua sát nút 57% sáng 03/09; đo đầy đủ V2 ra ÂM cùng ngày — TEST −1,7%,
P(≤0)=77,7%, xem phụ lục):

| # | đề xuất | trục thông tin MỚI | chi phí | trạng thái sau 03/09 |
|---|---|---|---|---|
| ② | **Ensembling cách diễn đạt** (GQE k=2, majority vote) — §3 | trục **CÁCH TẢ KHÁC** của cùng câu | 1 lượt LLM/câu (~$0,3/132 câu gpt-5.2) | **SỐNG — đứng đầu, trần cao nhất** (chạm được cả R@1 lẫn tầng sinh ứng viên). Ba đội đối thủ của chính giải đã dùng tại trận; chưa có phép đếm trên kho |
| ③ | **PRF Rocchio 1 bước trên embedding ẢNH** — §4 | trục **KHÔNG GIAN ẢNH** (đồng thuận kho) | 0 API, 0 GPU | **SỐNG — chạy độc lập** (§9.2); bằng chứng cùng họ: SuperGlobal + WeiMoCIR (trộn ảnh-chữ zero-shot); pseudo-cho-CLIP vẫn chưa có số văn liệu |
| ④ | **Chấm dưới-khung: max-pool trên crop** — §5 | trục **DƯỚI-KHUNG** của ảnh ứng viên | ~200 lượt mã hoá ảnh/câu (GPU nhẹ) | SỐNG — cuối hàng, bằng chứng chuyển miền yếu nhất |
| — | *(cửa đóng: **TOÀN TRỤC ① khử-bias** — V1 nội-video 51%, V2 liên-video đo đầy đủ −1,7% (phụ lục); shot-pooling/cửa sổ trượt — nay có thêm ví dụ cùng-bài-toán 2025 dùng đúng cơ chế đã ÂM, §6.1; mở rộng bằng keyframe HÀNG XÓM thời gian, phủ định bằng ngôn ngữ, TPT, relevance feedback tương tác, bộ rerank cần model mới — §6)* | | | |

**Một câu tóm tắt:** họ khử-bias chết đủ ba tầng đo trong hai ngày (V1 51% =
tung đồng xu; V2 qua cổng đếm 57% sát nút nhưng đo đầy đủ −1,7% — hub liên-video
có thật về thống kê mà không quy đổi thành điểm); đề xuất đứng đầu duy nhất còn
lại có bằng chứng mạnh là GQE (**+5,2 điểm R@1** zero-shot CLIP MSR-VTT, ba đội
đối thủ của chính giải dùng tại trận 2025), và phép đếm-trước của nó từ nay phải
đo đúng HÌNH DẠNG chỗ ăn mà phép đếm bão hoà 03/09 chỉ ra: nhóm một cảnh là
**trận tay đôi hạng 1↔2 nội-video** (đáp án trung vị hạng 2, ≤3 ở 68% số câu) —
tín hiệu nào phân xử đúng ≥62% số trận là ăn, tín hiệu xáo cả pool là thua
(xem "TRẢ LỜI §9.1" cuối tài liệu).

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

> **TRẠNG THÁI 03/09: TOÀN TRỤC ① ĐÃ ĐÓNG, đủ ba tầng đo.** V1 (chọn Ô
> nội-video): cổng đếm 51% ≈ tung đồng xu, ĐÓNG 02/09. V2 (thứ tự dòng
> liên-video): cổng đếm QUA sát nút 57% sáng 03/09, nhưng phép đo đầy đủ ra ÂM
> cùng ngày (TEST −1,7%, P(≤0)=77,7% — phụ lục). Phần dưới đây giữ nguyên làm
> hồ sơ cơ chế và làm hàng rào chống đề-xuất-lại-dưới-tên-khác.

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
  MSR-VTT/MSVD/DiDeMo không cần huấn luyện lại. Chi tiết cơ chế **DIS** (đọc
  thêm 03/09): dựng trước tập "ứng-viên-hub" = các mục lọt top-k của BẤT KỲ câu
  bank nào; chỉ áp chuẩn hoá khi mục đứng đầu là ứng-viên-hub, ngược lại giữ
  điểm gốc — khuôn "chỉ can thiệp khi có mục tiêu", đáng nhớ cho mọi tín hiệu
  yếu về sau. *(Ghi chú 03/09: khi phép đo NNN-liên-video ra ÂM −1,7% cùng
  ngày, toàn trục ① đã đóng — khuôn DIS không còn chỗ dùng trong trục này;
  giữ lại làm mẫu thiết kế chung.)*
- **DBNorm — dual bank** (2023): [arXiv:2310.11612](https://arxiv.org/html/2310.11612)
  — thêm bank GALLERY bên cạnh bank câu (DualIS/DualDIS); trên MSR-VTT
  CLIP4Clip R@1 44,20 → 45,00, và ablation của chính nó: **bank câu ảnh hưởng
  mạnh hơn bank gallery**. *(Ghi chú 03/09: trục ① đã đóng toàn phần bằng phép
  đo −1,7% — DBNorm/DBSN ghi ở đây làm hàng rào chống đề-xuất-lại-dưới-tên-khác,
  cùng vai trò với §9.4: chúng là biến thể của đúng cơ chế đã đóng, không phải
  cửa mới.)*
- **DBSN — Dual Bank Sinkhorn Normalization** (2025):
  [arXiv:2508.02538](https://arxiv.org/html/2508.02538) — bản mạnh nhất của họ
  này (Sinkhorn 10 vòng lặp, thêm target-bank), nhưng chính nó ghi rõ: **suy giảm
  đáng kể khi bank nhỏ**. Với bank ~141 câu của ta (xem dưới), NNN top-k là lựa
  chọn đúng; DBSN chưa bao giờ đến lượt — trục ① đóng ở tầng NNN.

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
  tìm-với-từng-bản-tả rồi hợp nhất danh sách. Đọc lại nội dung 03/09, ba chi
  tiết đáng vay: (i) trên **textual KIS** — bài giống ta nhất, không phải AVS —
  mean median rank cải thiện **12,08 → 6,33**; (ii) trước khi dùng, mỗi bản tả
  bị **lọc bằng kiểm tra nhất quán**: sinh cặp hỏi-đáp từ câu gốc trên 7 khía
  cạnh (người/hành động/vật/nơi/thời gian/màu/số lượng) rồi chỉ giữ bản tả trả
  lời khớp — đúng công cụ cho rủi ro drift mà §3.3 nêu, rẻ (thêm ~1 lượt
  LLM/câu); (iii) nhánh sinh-ẢNH từ câu (T2I qua Stable Diffusion) bị chính họ
  hạ trọng số còn 0,5 vì yếu với chuyển động — thêm một lý do để ta chỉ lấy
  nhánh văn bản.
- **Hai đội đối thủ của CHÍNH giải này đã dùng nó năm 2025** — bằng chứng khả thi
  tại trận, không phải phòng lab:
  [MADTempo, arXiv:2512.12929](https://arxiv.org/html/2512.12929) (GPT-5 phân rã
  câu, 75,4 điểm sơ tuyển, vào chung kết) và
  [EEIoT_newbie, arXiv:2512.06334](https://arxiv.org/html/2512.06334v1) (Gemini
  sinh tập câu tương đương ngữ nghĩa, lấy max theo keyframe).
- Đội đối thủ **thứ ba** (đọc thêm 02/09, lượt sửa 2):
  [arXiv:2512.12935](https://arxiv.org/html/2512.12935v1) — 76,4/88 điểm, cũng
  phân rã câu bằng GPT-4o, và hợp nhất hai danh sách xếp hạng bằng **SRRF**
  (score-reflected reciprocal rank fusion). SRRF là phương án hợp nhất dự phòng
  nếu majority-vote của GQE tỏ ra giòn ở k=2 trên câu tiếng Việt — cùng
  training-free, chỉ đổi công thức trộn hạng.

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
- **Bằng chứng cùng họ, thêm 02/09 (lượt sửa 2):**
  [SuperGlobal, ICCV 2023, arXiv:2308.06954](https://arxiv.org/abs/2308.06954) —
  rerank chỉ bằng đặc trưng toàn cục: tinh chỉnh descriptor của truy vấn và của
  top-k bằng pooling trên tập hàng xóm nhỏ, **không huấn luyện**; +3,7% two-stage
  trên ROxford+1M-Hard với speedup 64.865× so rerank đặc trưng cục bộ (mức đọc:
  abstract). Đáng kể vì một đội moment-retrieval ở **CVPRW 2025 IViSE**
  (Nguyen-Nhu et al., "A Lightweight Moment Retrieval System with Global
  Re-Ranking", openaccess.thecvf.com — PDF chặn 403, chỉ đọc được phần dẫn qua
  kết quả tìm kiếm) dùng đúng SuperGlobal cho bài keyframe-retrieval cùng dạng
  với ta. Tức "tinh chỉnh truy vấn/ứng viên bằng hàng xóm trong KHÔNG GIAN ẢNH"
  đã có cả số benchmark lẫn tiền lệ dùng tại trận cùng miền — mạnh hơn hồ sơ
  "chưa có số pseudo cho CLIP" mà mục này tự khai ở trên.
- **Bằng chứng cùng họ, thêm 03/09 (lượt soát 3):**
  [WeiMoCIR, TAAI 2024, arXiv:2409.04918](https://arxiv.org/abs/2409.04918) —
  composed image retrieval **training-free**: truy vấn = trung bình có trọng số
  của embedding ẢNH tham chiếu và embedding CHỮ, không huấn luyện gì. Đây là
  phép cộng `q' = q + λ·e_ảnh` của ③ dưới tên khác, được đo có số trên
  FashionIQ/CIRR (mức đọc: abstract + tóm tắt máy tìm kiếm). Khác biệt duy
  nhất: CIR có ảnh tham chiếu do NGƯỜI đưa, ③ lấy pseudo từ top-m — phần
  "pseudo" vẫn là chỗ chưa có số, như tự khai ở trên.

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

Tương tác quan trọng — **đã đổi sau cổng 02/09**: lo ngại ban đầu là PRF
**khuếch đại hub** (khung hub lọt top-m kéo `q'` về phía hub), nên bản gốc của
tài liệu này xếp ③ chạy *sau* ① trên điểm đã khử bias. Cổng đếm cho thấy khung
chặn đường **nội-video** không phải hub (51%, chênh bias ≈ 0) ⇒ với biến thể
hoán-vị-nội-video của ③, lo ngại ấy giảm hẳn và **③ chạy độc lập được ngay**.
Lo ngại còn lại là **drift liên-video** (top-m toàn khung của một chủ đề sai kéo
`q'` đi) — chính là rủi ro kinh điển của cả họ PRF, xử bằng λ nhỏ và phép đếm
trước khi nhìn điểm.

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

Cùng lý do đóng luôn một biến thể nghe khác tên: **mở rộng truy vấn thị giác bằng
keyframe HÀNG XÓM THỜI GIAN** (chấm khung `i` bằng tổ hợp `sim` của chính nó và
các khung liền kề, hoặc trộn embedding hàng xóm vào truy vấn). Khai triển ra thì
điểm mới của khung `i` là một tổ hợp tuyến tính của `s_q(i−w..i+w)` — **chính là
làm-mượt-thời-gian với một nhân khác**, cửa đã ÂM trên TEST. Muốn "hàng xóm" mang
thông tin mới thì hàng xóm phải đến từ **ngoài video** (đồng thuận kho — tức ③),
không phải từ trục thời gian đã vắt kiệt.

Thêm 03/09, một ví dụ cùng-bài-toán để không ai mở lại cửa này vì "đội khác cũng
làm": một hệ moment-retrieval 2025 (BEiT3 + OpenCLIP + TransNetV2,
[arXiv:2504.08384](https://arxiv.org/html/2504.08384v1)) dùng "**neighbor score
aggregation**" — cộng dồn điểm của các khung lân cận để thưởng ứng viên có hàng
xóm ổn định — làm bước rerank chính. Khai triển ra, đó vẫn là tổ hợp tuyến tính
của `s_q(i−w..i+w)`: đúng cơ chế đã ÂM trên bộ đo của ta (pool một cảnh 58→31),
và paper ấy cũng **không có số định lượng** cho riêng bước này, chỉ có ví dụ
định tính. Kỹ thuật phổ biến ≠ kỹ thuật đo được là dương trên phân bố của ta.

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

Ghi nhận thêm (02/09, lượt sửa 2): đội đối thủ thứ ba của chính giải
([arXiv:2512.12935](https://arxiv.org/html/2512.12935v1), 76,4/88 điểm) rerank
top-100 bằng **đầu ITM của BLIP-2** — tức câu trả lời chuẩn của giới cho tầng
XẾP HẠNG là *một model thứ hai*. Điều đó không mở lại cửa này cho lane paper
(vẫn cần model mới ~4GB), nhưng là một dữ kiện định hướng cho lane **pe-core**:
nếu encoder thứ hai được chọn, phương án dùng nó làm reranker cross-attention
trên top-100 có tiền lệ tại trận với điểm cao.

### 6.6 Dò khung người-dẫn kiểu cổ điển (anchor-shot detection) — ĐÓNG lối cũ, giữ lối mới

Văn liệu anchor-shot detection là dòng 2005–2013 (mặt người lặp, phân cụm slice
không-thời-gian — ví dụ
[News videos anchor person detection by shot clustering, Neurocomputing 2014](https://www.sciencedirect.com/science/article/abs/pii/S0925231213006115)):
cần dò mặt, cần khung liên tiếp của video gốc, không có bản training-free trên
embedding có sẵn. Lối đúng cho cùng mục tiêu chính là ①: khung studio **tự lộ**
qua bias bank cao (nó khớp vừa vừa với mọi câu) — không cần bộ dò riêng. Nếu ①
sống, một phép kiểm bằng mắt rẻ: in 50 khung bias cao nhất, xác nhận chúng là
studio/đồ hoạ — bằng chứng cơ chế kiểu "mở ảnh xem" đã dùng cho lever cảnh B.

**Cập nhật sau cổng 02/09: lối mới cũng chết cho trục NỘI-video.** Cổng đếm cho
thấy khung đứng trên đáp án trong cùng video **không** "hấp dẫn mọi câu" hơn đáp
án (51%, chênh bias trung vị ≈ 0) ⇒ "loại khung studio khỏi ứng viên khi câu tả
hiện trường" không có mục tiêu để bắn **bên trong video đúng** — nếu khung chặn
là studio thì bias bank của nó đã phải cao hơn. Toàn bộ hướng khai thác cấu trúc
bản-tin (studio vs hiện trường) cho xếp hạng nội-video coi như đóng; phần còn
sống duy nhất của ý tưởng khi ấy là trục liên-video §9.3 (khung studio của video
SAI chen vào giữa 100 dòng). **Cập nhật 03/09: phần đó cũng đã chết** — phép đo
NNN-liên-video đầy đủ ra ÂM (TEST −1,7%, phụ lục) ⇒ ý tưởng studio-vs-hiện-trường
đóng ở CẢ HAI trục, bằng số.

---

## 7. Trình tự thi hành đề nghị (cho người điều phối — lane này không chạy gì)

*(Cập nhật 03/09 lượt soát 3, lần hai trong ngày — trục ① đã đi TRỌN đường đo:
V1 ÂM 51% ⇒ đóng; V2 QUA cổng đếm 57% sát nút ⇒ đo đầy đủ ⇒ **ÂM −1,7%** ⇒
toàn trục đóng, không viết thêm script khử-bias nào. Trình tự dưới đây thay
bản cũ, và mọi phép-đếm-trước từ nay đo theo HÌNH DẠNG chỗ ăn mà phép đếm bão
hoà đã chỉ ra: trận tay đôi hạng 1↔2 nội-video, ngưỡng thắng ≥62% số trận —
xem "TRẢ LỜI §9.1" cuối tài liệu.)*

1. **②-phép-đếm-trước** (~$0,3 sinh paraphrase + đếm hạng, 0 lần đọc TEST),
   hai thước, thước (a) là thước quyết định:
   (a) **thước trận-tay-đôi**: trên các câu một cảnh mà đáp án đứng hạng 2–3
   nội-video, majority-vote(k=2) phân xử đúng bao nhiêu % trận hạng-1↔đáp-án?
   ≥62% mới đi tiếp; (b) hạng nội-video VÀ hạng dòng trung vị trên cả 132 mục
   (thước cũ, giữ để so). Soi tay ~20 paraphrase tiếng Việt trước khi đếm;
   cân nhắc lọc-nhất-quán kiểu VIREO (§3.1) nếu chất lượng lởm chởm.
2. ② đầy đủ (TUNE/TEST phân tầng, biến thể (a) hẹp trước) theo §3.3 + 5 cổng §2.5.
3. **③-phép-đếm-trước** (0 đồng, chạy độc lập, §9.2): cùng thước trận-tay-đôi
   với `q'` thay `q`, quét (λ, m) chỉ đọc hạng. Song song được với 1–2.
4. ④ chỉ khi 1–3 xong mà nhóm một cảnh vẫn còn ≥20% headroom xếp hạng.

Nhắc hai điều từ kỷ luật đo: nửa TEST của bộ 132 đã bị đọc **≥5 lần** (thêm lần
đọc của phép đo NNN-liên-video 03/09) — mọi số TEST mới của các đề xuất trên
phải đọc với hiểu biết đó, và nếu đợt sinh thêm GT (`KE_HOACH_DINH_VI.md`
§4.2b) sắp chạy thì dồn các lần đọc TEST mới lại sau đợt ấy. Và mọi kết luận
hoà ở n=33/nhóm đọc là "chưa chứng minh được", không phải "không có".

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

Thêm ở lượt sửa 2 (WebFetch thật, 02/09/2026):

| nguồn | URL |
|---|---|
| DN — Distribution Normalization (NeurIPS 2023) | https://arxiv.org/abs/2302.11084 · https://fengyuli-dev.github.io/dn-website/ |
| SuperGlobal (ICCV 2023) | https://arxiv.org/abs/2308.06954 |
| Đối thủ AIC 2025 thứ ba (SRRF + BLIP-2 ITM rerank) | https://arxiv.org/html/2512.12935v1 |
| Moment retrieval dùng SuperGlobal (CVPRW 2025 IViSE, Nguyen-Nhu et al.) | https://openaccess.thecvf.com/content/CVPR2025W/IViSE/papers/Nguyen-Nhu_A_Lightweight_Moment_Retrieval_System_with_Global_Re-Ranking_and_Robust_CVPRW_2025_paper.pdf |

Ghi chú trung thực về mức đọc: các dòng NNN, QB-Norm, DBSN, GQE, MTA, MADTempo,
EEIoT_newbie, NegBench, RF-CLIP là **đọc nội dung** (bản html/pdf/README, có số
và công thức trích ở trên); VISIONE, anchor-shot, NeighborRetr, hai bài PRF là
**đọc mức tóm tắt/abstract** đủ cho vai trò phụ trợ của chúng trong tài liệu.
Lượt sửa 2: GQE được **fetch lại và đối chiếu số** (31,2→36,4; k=2 > k=10;
majority vote 49,6 > average-similarity 46,1 — khớp bản gốc); DN đọc **nội dung**
(công thức và số từ trang chính thức của paper); SuperGlobal đọc **abstract**;
2512.12935 đọc **nội dung**; bài CVPRW Nguyen-Nhu bị chặn 403, chỉ xác nhận được
việc dùng SuperGlobal qua trích dẫn của máy tìm kiếm — ghi rõ để ai cần thì tải
tay.

Thêm ở lượt soát 3 (WebSearch/WebFetch thật, 03/09/2026):

| nguồn | URL | mức đọc |
|---|---|---|
| DBNorm — dual bank (2023) | https://arxiv.org/html/2310.11612 | nội dung (công thức DualIS/DualDIS, số MSR-VTT, ablation bank) |
| QB-Norm — chi tiết DIS | https://www.emergentmind.com/topics/querybank-normalisation-qb-norm | tóm lược thứ cấp có công thức (β≈20, bank 5–20k, cơ chế activation-set); trang abstract arXiv fetch lại cùng ngày |
| Hệ moment-retrieval "neighbor score aggregation" (2025) | https://arxiv.org/html/2504.08384v1 | nội dung (thuật toán rerank, không có số định lượng) |
| GenSearch/VIREO — fetch lại bản v2 | https://arxiv.org/html/2407.12341v2 | nội dung (KIS 12,08→6,33; lọc nhất quán 7 khía cạnh; trọng số T2I 0,5) |
| NNN — đối chiếu lại số chịu-tải | https://arxiv.org/html/2410.24114v1 | nội dung; **47,15→50,24 SigLIP COCO, α/k cho bank lệch phân bố — khớp từng chữ số với bản 02/09** |
| WeiMoCIR (TAAI 2024) | https://arxiv.org/abs/2409.04918 | abstract + tóm tắt máy tìm kiếm |
| DN (NeurIPS 2023) — fetch lại trang abstract | https://arxiv.org/abs/2302.11084 | abstract (bản PDF không parse được ở lượt này) |
| MTA — xác nhận lại phạm vi (chỉ phân loại) | https://github.com/MaxZanella/MTA · https://arxiv.org/abs/2405.02266 | tóm tắt máy tìm kiếm |
| Kết quả VBS 2024/2025 (bối cảnh §6.1) | https://videobrowsershowdown.org/hall-of-fame/ | tóm tắt máy tìm kiếm |

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

---

## 9. Đọc lại cổng ① cho đúng phạm vi — V1 đóng chắc, V2 chưa hề được đo

*(Viết ở lượt sửa 2, sau khi cổng chạy. Đây không phải phản cung kết quả cổng —
kết quả 51% đứng vững — mà là xác định chính xác nó đóng cửa NÀO.)*

### 9.1 Cổng đã đo gì

`scripts/dem_bias_hub.py` đếm, cho 68 mục có keyframe đáp án trong pool nhưng
không đứng hạng-1 **nội-video**: bias-bank của các khung đứng trên đáp án **trong
cùng video** so với bias-bank của đáp án. Kết quả 51% ≈ tung đồng xu, chênh trung
vị ≈ 0. Ba hệ quả đóng **chắc**, đều giới hạn trong trục nội-video:

1. **①-V1** (hoán vị chọn Ô nội-video theo `s_D`) — không có mục tiêu. ĐÓNG.
2. **Bank thủ công "nền studio"** (phủ định bằng số học điểm, §2.3/§6.2) cho
   xếp hạng nội-video — đóng theo hệ luận: nếu khung chặn nội-video là khung
   studio "khớp vừa vừa mọi thứ" thì bias bank-141-câu của nó đã phải cao hơn
   đáp án; nó không cao hơn. (Caveat một dòng: bank thủ công là ước lượng bias
   *hẹp* hơn bank câu thật, nên hệ luận này là suy rộng có căn cứ chứ không phải
   phép đo trực tiếp — nhưng không đáng một lần đo riêng khi tín hiệu gốc là 0
   tròn trĩnh.)
3. **Toàn bộ hướng cấu-trúc-bản-tin cho nội-video** (§6.6) — đóng cùng lý do.

Và một câu hỏi mở mà cổng đặt ra nhưng không trả lời: **vậy khung đứng trên đáp
án là AI?** Không phải hub, vậy hoặc là hàng xóm thời gian cùng cảnh (vô hại nếu
nằm trong cửa sổ chấm — và khi đó headroom V1 thật sự nhỏ hơn +59% trên giấy),
hoặc là khung khác-Ô thật sự sai. Phép đếm 0 đồng phân định: khoảng cách frame
giữa khung hạng-1 nội-video và đáp án, cho 57% số câu một cảnh mà đáp án không
đứng hạng-1. Nếu phần lớn ≤ cửa sổ chấm thì trục V1 gần bão hoà **cho mọi tín
hiệu**, không riêng NNN — và toàn bộ cược của nhóm một cảnh dồn về trục thứ tự
dòng. Đáng chạy trước khi viết bất kỳ script V1 nào khác.

### 9.2 Hệ quả cho ③

Bản gốc xếp ③ sau ① vì sợ PRF khuếch đại hub nội-video. Hub nội-video hoá ra
không tồn tại đáng kể ⇒ ③ chạy độc lập ngay (§4.2 đã sửa). Rủi ro còn lại của ③
là drift liên-video, xử bằng λ nhỏ + phép đếm trước.

### 9.3 Trục V2 (thứ tự dòng liên-video) — cửa cổng chưa chạm tới

Ba lý do nó KHÔNG bị kết quả 51% đóng:

1. **Phép so của cổng toàn bộ là nội-video.** V2 so ứng viên của dòng đứng trên
   dòng đúng với ứng viên của dòng đúng — phần lớn các dòng ấy thuộc **video
   khác** (48% câu một cảnh có dòng đúng nhưng 32,9 điểm phần trăm kẹt ở hạng ≥6,
   `KE_HOACH_DINH_VI.md` §3.2). Bias "khớp vừa vừa mọi câu" là tính chất **liên
   chủ đề**: trong một video cùng chủ đề nó gần như hằng số (chính lời giải thích
   chuyển-miền của phụ lục nói vậy!), nhưng **giữa các video** nó biến thiên thật.
2. **Bằng chứng văn liệu của NNN nằm đúng ở trục này.** +3,1 R@1 của SigLIP trên
   COCO là xếp hạng *toàn gallery* — bản sao của bài thứ-tự-dòng, không phải của
   bài chọn-Ô-nội-video. Lời giải thích "COCO không chuyển miền được vì nội-video
   cùng chủ đề" trong phụ lục là đúng, và chính nó **dự đoán trục liên-video thì
   chuyển được**.
3. Deficit thứ tự dòng của nhóm một cảnh (0,2040) **lớn hơn** deficit đặt-frame
   (0,1488) — trục V2 là trục lớn hơn ngay từ đầu.

**Cổng đếm V2 — công bố ngưỡng trước, như V1:** trên các câu một cảnh có dòng
đúng đầu tiên ở hạng r ≥ 2 (lấy cả r ∈ [2,5] cho đủ n), xét r−1 dòng đứng trên:
(a) báo cáo tách **cùng-video / khác-video**; (b) trong phần khác-video, đếm tỷ
lệ dòng có ứng viên bias-bank **cao hơn** ứng viên của dòng đúng. **Ngưỡng đi
tiếp: >55%**, y hệt V1. ≤55% ⇒ đóng nốt toàn trục ①, khỏi viết script đo nào
nữa. Chi phí: sửa vài chục dòng của chính `dem_bias_hub.py`, 0 API, ~15 phút.
Nếu qua: đo V2 đầy đủ theo 5 cổng §2.5, biến thể sắp-lại-100-dòng (tập dòng
không đổi ⇒ R@100 bất biến; nhóm hai cảnh giữ nguyên ⇒ có nhóm assert).

### 9.4 DN (NeurIPS 2023) — phân loại vào đúng họ, để không ai đề xuất lại

[Distribution Normalization](https://arxiv.org/abs/2302.11084) trông như một đề
xuất mới ("chuẩn hoá phân bố lúc test, +3,6 R@1 T→I trên MSCOCO trong stack
TTA+DN"): `s(x,y) = (φ(x) − ½μ_x)ᵀ(ψ(y) − ½μ_y)`, μ ước lượng từ ~100 mẫu. Nhưng
khai triển cho truy vấn cố định `q`: phần phụ thuộc ứng viên là `−½μ_text·ψ(y)` —
tức **phạt khung thẳng hàng với CÂU HỎI TRUNG BÌNH**. Đó chính là trục CÂU KHÁC
của ①, bản bậc-nhất với bank = toàn bộ, k = ∞, α cố định. Hệ quả: (i) cho
nội-video, nó chết theo cổng V1 — không cần đo; (ii) cho liên-video, nó là
trường hợp riêng của cùng phép sửa mà NNN làm có chọn lọc hơn (NNN ước lượng
bias bằng top-k câu *giống ứng viên nhất* thay vì câu trung bình toàn cục — đó
là chữ "nearest neighbor" trong tên). Nếu cổng V2 (§9.3) qua, thứ để đo là NNN
top-k; DN không thêm được gì. Ghi ở đây làm hàng rào chống
đề-xuất-lại-dưới-tên-khác — đúng vai trò của §6.

---

## KẾT QUẢ CỔNG V2 (§9.3) — **QUA, sát nút** (03/09)

`scripts/dem_bias_hub.py --truc v2`, 0 API, ngưỡng >55% công bố trước.

Nhóm MỘT cảnh, mục có dòng-đúng ở hạng ≥2 và có dòng khác-video đứng trên (n=17):

| thước | kết quả |
|---|---|
| theo DÒNG (gộp 441 dòng khác-video đứng trên) | **57%** có bias-bank cao hơn ứng viên của dòng đúng |
| theo MỤC (đa số dòng-trên bias cao hơn) | 65% |
| tỷ lệ dòng-trên CÙNG video | 53% |

**QUA cổng (57% > 55%) — hub liên-video là có thật**, ngược với trục nội-video
(51% = tung đồng xu). Đúng như §9.3 dự đoán từ chính lời giải thích chuyển-miền.

Ba giới hạn phải mang theo khi đo đầy đủ:
1. **Sát nút** — 57% với ngưỡng 55%; và n=17 mục là nhỏ.
2. **Lát r∈[2,5] rỗng**: mọi mục đủ điều kiện đều có dòng-đúng nằm SÂU (hạng >5).
   Tức NNN-liên-video, nếu ăn, sẽ ăn ở phần R@20/50/100 chứ khó cứu R@1/R@5.
3. Phép đo đầy đủ phải là **sắp-lại-100-dòng** (tập dòng không đổi ⇒ R@100 bất
   biến theo xây dựng; nhóm HAI cảnh giữ nguyên ⇒ nhóm assert), quét k trên TUNE,
   đọc TEST một lần, bootstrap theo câu — đủ 5 cổng.

---

## KẾT QUẢ ĐO ĐẦY ĐỦ SAU CỔNG V2 — **ÂM, TOÀN TRỤC ① ĐÓNG** (03/09)

`scripts/do_nnn_lien_video.py` — sắp lại 100 dòng bằng khoá `r + k·z(bias)`,
chỉ câu một cảnh, tập dòng bất biến (R@100 giữ nguyên theo xây dựng), nhóm hai
cảnh bất biến (assert), TUNE/TEST 33/33 xáo seed cố định, TEST đọc một lần.

| k | TUNE một-cảnh | chênh |
|---|---|---|
| 0 | 0,2871 | — |
| 1 | 0,2906 | **+0,0036** (đỉnh) |
| 2–8 | 0,2885–0,2895 | +0,0015…+0,0024 |
| 16–32 | 0,2835/0,2697 | âm dần |

TEST (k=1): 0,2649 → 0,2605 = **−1,7%**; bootstrap theo câu KTC [−0,0167, +0,0047],
P(≤0) = 77,7%.

**Đọc số cho đúng:** đường TUNE phẳng và không đơn điệu với biên độ +1,2% là
chữ ký của nhiễu, và TEST xác nhận. Chuỗi ba tầng giờ khép kín:
V1 51% (hub nội-video không tồn tại — bias gần hằng số trong video) →
V2 57% sát nút (hub liên-video TỒN TẠI về mặt thống kê) →
đo đầy đủ −1,7% (nhưng không quy đổi thành điểm, vì các dòng hub đứng trên
dòng-đúng-nằm-sâu chủ yếu ở vùng hạng mà BUCKET đã trả gần như cùng số điểm).

**Toàn trục ① đóng bằng số ở cả ba tầng đo. Không viết thêm script nào cho họ
khử-bias.** Trục sống còn lại của tài liệu này: ② GQE, ③ PRF, ④ crop max-pool.

---

## TRẢ LỜI §9.1 — trục nội-video CHƯA bão hoà, và chỗ ăn có hình dạng cụ thể (03/09)

`scripts/dem_bao_hoa_noi_video.py`, 0 API, 132 mục sạch:

| nhóm | hạng-1 trong ±20 | lệch hạng-1→đáp án (trung vị) | hạng của keyframe-đáp-án |
|---|---|---|---|
| MỘT cảnh (pool 8 kf/video) | **38%** | 75 frame | **trung vị 2**; ≤3: 68% |
| HAI cảnh (pool 16 kf/video) | **5%** | 752 frame | trung vị 6; ≤3: 31% |

1. **Không bão hoà** — 38%/5% còn rất xa 100%, khớp với oracle tier-2 (+40%).
   Pretest PE-Core vẫn đáng chạy đúng như ngưỡng tiền-đăng-ký.
2. **Hình dạng chỗ ăn ở MỘT cảnh:** đáp án đã đứng trung vị HẠNG 2 nội-video —
   bài toán không phải "tìm cho ra" mà là "thắng trận đấu tay đôi hạng 1↔2".
   Tín hiệu nào phân xử được cặp đôi ấy (thắng 62% số trận trở lên) là ăn;
   mọi tín hiệu đo kiểu "trung bình toàn pool" đã thử đều thua vì chúng
   xáo cả 8 keyframe thay vì phân xử đúng một cặp.
3. **HAI cảnh trượt hệ thống** (trung vị 752 frame — text khớp cảnh A, đáp án ở
   cảnh B): đúng cái đòn hoán-vi-cảnh-B đã ship khai thác; đừng mong encoder
   mới tự sửa nhóm này.

---

## 10. Tổng kết lượt soát độc lập 03/09 — bản đồ hội tụ, và điều đó nghĩa là gì

Lượt soát này chạy lại từ đầu cả bốn hướng được giao (kỹ thuật query-time cho
CLIP-family; temporal context kiểu VBS/TRECVID; khai thác cấu trúc bản tin;
đo-trên-132-mục cho từng đề xuất) bằng truy vấn tìm kiếm MỚI, không nhìn bảng
xếp hạng cũ trước khi tìm: 8 lượt WebSearch + 7 lượt WebFetch (danh sách §8,
bảng "lượt soát 3"). Bốn kết quả:

**(a) Bản đồ hội tụ.** Không tìm thấy họ cơ chế thứ năm nào vừa training-free,
vừa áp được vào ràng buộc của ta (không huấn luyện, không video gốc, không vòng
lặp người), vừa chưa nằm trong bốn trục của §1 hoặc bảng cửa đóng §6. Các "ứng
viên mới" mà truy vấn mới trồi lên đều quy về chỗ cũ khi nhìn theo cơ chế:
DBNorm/DBSN → biến thể của trục ① (đã đóng toàn phần); MTA/TTA-ảnh → §5/§6.3,
và chính văn liệu TTA tự cảnh báo crop/flip phá ngữ nghĩa toàn ảnh ở bài
retrieval; "neighbor score aggregation" của một hệ moment-retrieval 2025 →
đúng cửa làm-mượt đã ÂM (§6.1); composed-retrieval training-free (WeiMoCIR,
CIReVL) → phép cộng ảnh-chữ của ③; MUGI/PRF-LLM cho văn bản → họ của ② và ③.
Hai lượt soát độc lập ra cùng một bản đồ là bằng chứng tốt nhất hiện có rằng
bản đồ ấy **đủ** — cái còn thiếu không nằm trong văn liệu query-time
training-free 2024–2026, nó nằm ở encoder (lane pe-core).

**(b) Nguồn chịu-tải đã được đối chiếu lại.** Số NNN (SigLIP COCO 47,15→50,24;
α/k cho bank lệch phân bố) khớp từng chữ số với bản 02/09 — quan trọng vì toàn
bộ chuỗi cổng V1/V2 đứng trên trích dẫn ấy, và vì dự án từng có tiền lệ chép
nhầm số từ tóm tắt máy tìm kiếm (ca TransNetV2, `KE_HOACH_DINH_VI.md` §4.5).

**(c) Ba món mới thật sự, đều nhỏ:** bộ lọc-nhất-quán 7-khía-cạnh của VIREO
cho rủi ro drift của ② (§3.1 — công cụ rẻ, đúng chỗ yếu nhất của đề xuất đứng
đầu); WeiMoCIR làm bằng chứng cùng-họ cho phép cộng ảnh-chữ của ③ (§4.1); và
khuôn DIS "chỉ can thiệp khi có mục tiêu" làm mẫu thiết kế cho mọi tín hiệu
yếu (§2.1 — dù chỗ dùng dự kiến đã đóng cùng ngày).

**(d) Điều lượt soát KHÔNG làm được — nói thẳng.** Nó không tìm ra tín hiệu
nội-video mới nào cho nhóm một cảnh ngoài ba đề xuất còn sống (②③④), và cả ba
đều chưa có phép đếm nào trên kho. Sau khi trục ① đi trọn ba tầng đo trong hai
ngày và chết ở tầng cuối, quy trình cho ②③④ đã rõ: phép-đếm-trước theo thước
trận-tay-đôi (§7, ngưỡng ≥62%), rồi mới tới 5 cổng. Nếu cả ba phép đếm cùng
trượt, kết luận của lane này là: **không gian tín-hiệu-training-free cho trục
nội-video nhóm một cảnh đã cạn theo văn liệu hiện có** — mọi đầu tư tiếp theo
của nhóm này thuộc về encoder thứ hai (pe-core) và trục thứ tự dòng bằng tín
hiệu ngoài-SigLIP, không thuộc về thêm một phép biến đổi điểm nào nữa.
