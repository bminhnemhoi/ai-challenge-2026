# Paper 2025–2026 cho nghẽn ĐỊNH VỊ NỘI-VIDEO — cái nào chạy được, cái nào không

Chốt 01/09/2026. Lane `paper-2026`: **đọc và đánh giá**, không chạy thí nghiệm
điểm số. Mọi con số trong tài liệu này thuộc một trong hai loại:

* **đếm tất định** trên dữ liệu đã có — sinh lại bằng
  `scripts/khao_sat_hinh_hoc_dong.py` và `scripts/tham_do_cau_truc_shot.py`
  (hai file MỚI của lane này, không sửa gì của đường sản xuất);
* **số của paper**, luôn kèm URL đã đọc thật.

> Không mã video, không đáp án trong tài liệu này (docs/ lên GitHub công khai).

---

## 0. Bảng xếp hạng — đọc cái này trước

Xếp theo (tác động vào trần +126%) × (khả thi trong ràng buộc: không huấn luyện,
T4 Colab free, tiếng Việt, không có video gốc trên máy).

| # | đề xuất | nhắm vào | chi phí | khả thi | tác động ước tính |
|---|---|---|---|---|---|
| ① | **Chấm theo PHỦ hai truy vấn con trên một cửa sổ** (U-CESE *Unified Clipping*) | đúng nhóm HAI cảnh: 51/65 câu đã có ứng viên cách đáp án ≤1 ô | 0 GPU, chạy trên sim đã có | **5,0** | **4,5** |
| ② | **Điểm CHUYỂN CẢNH** — chấm *hiệu* giữa hai keyframe liền kề (TFVTG *dynamic score*) | khoảnh khắc "cảnh B bắt đầu" | 0 GPU, 1,4 s cho cả kho | **5,0** | 3,5 |
| ③ | **TAG**: gộp thời gian + phân cụm liên kết + hiệu chỉnh điểm | chọn đúng ô trong video đúng | 0 GPU, chạy trên sim đã có | 4,5 | 3,0 |
| ④ | **Bảng ranh giới shot dựng một lần bằng TransNetV2** | biến "cảnh B bắt đầu" từ vùng mờ thành điểm xác định | tải ~26–60 GB video + **20–30 h GPU T4** (2–3 phiên Colab) | 2,5 | 4,0 |
| ⑤ | **VideoPrism-LvT** làm chỉ mục **CỬA SỔ** thứ hai | cửa sổ nào chứa A-rồi-B | 2–5 h T4, JAX | 2,5 | 3,0 |
| — | *(cửa đóng: §5 — MLLM grounding trực tiếp, REZE, ClipTBP, OmniShotCut, GranAlign, ReCap toàn kho, trích lại keyframe)* | | | | |

**Một câu tóm tắt:** ba việc đầu bảng **không cần một giờ GPU nào và không cần
tải một byte video nào** — chúng là thuật toán chạy trên chính vector tương đồng
SigLIP mà hệ thống đã tính. Việc tốn kém nhất (④) chỉ đáng làm sau khi ①–③ đã nói
được nó còn thiếu gì.

---

## 1. Nghẽn thật ra CỠ NÀO — đo lại, không suy luận

`scripts/khao_sat_hinh_hoc_dong.py`, bộ đo sạch 132 mục (66 MỘT cảnh / 66 HAI
cảnh), đường sản xuất `coverage`, đếm tất định:

| | MỘT cảnh (n=66) | HAI cảnh (n=66) |
|---|---|---|
| có ≥1 dòng trên video đúng | 51 | 53 |
| hạng dòng ĐẦU TIÊN của video đúng (trung vị) | 1 | 4 |
| số dòng trên video đúng (trung vị) | 16 | 15 |
| dòng gần nhất → khung neo, tính **FRAME** (trung vị) | **2** | **56** |
| dòng gần nhất → khung neo, tính **Ô KEYFRAME** (trung vị) | **0** | **1** |
| dòng rơi ĐÚNG ô keyframe neo | 41/51 | **24/53** |
| … cách ≤1 ô | 45/51 | 32/53 |
| **[400 ứng viên thô]** đúng ô | 58/65 | 35/65 |
| **[400 ứng viên thô]** cách ≤1 ô | 62/65 | **51/65** |
| số ô keyframe được phủ / tổng số ô của video | 5 / 254 (2,1%) | 6 / 235 (3,1%) |

### Ba điều bảng này nói, và không cái nào đoán được trước

**(1) Sai số của câu HAI cảnh là sai số MỘT Ô, không phải sai số cả video.**
Trung vị lệch 56 frame ≈ 2,2 giây ≈ đúng **một khe keyframe** (khe trung vị của
kho là 2,16 s). Và trên ứng viên thô, **51/65 câu HAI cảnh đã có ứng viên cách
đáp án ≤ 1 ô**, 55/65 cách ≤ 2 ô. Bài toán còn lại là *chọn đúng ô trong vài ô
liền kề*, không phải *dò cả 235 ô của video*. Đây là hai họ công nghệ khác hẳn
nhau, và nó quyết định toàn bộ phần còn lại của tài liệu.

**(2) Ta KHÔNG thiếu dòng, ta đặt dòng sai ô.** Video đúng đã nhận trung vị 15–16
dòng. Thang frame bước 10 phủ liên tục `10 × số_dòng` khung, nên **6 dòng là đủ
phủ trọn một ô 55 khung**. Ta có gấp 2,5 lần số dòng cần thiết — nhưng chúng rải
trên 5–6 ô của một video ~250 ô (2–3%), và ô đúng không nằm trong đó ở 29/53 câu
hai cảnh.

**(3) Khâu chọn VIDEO đã tốt, và bảng này chốt lại bằng hạng.** Dòng đầu tiên
thuộc video đúng đứng ở hạng trung vị 1 (một cảnh) và 4 (hai cảnh). Trần oracle
0,4308 giữ nguyên video và hạng, chỉ sửa frame — nên khoảng từ 0,1903 lên 0,4308
là **thuần chọn ô**, còn từ 0,4308 lên 1,0 mới là chọn video. Toàn bộ +126% nằm ở
việc chọn ô.

---

## 2. Vì sao 90% văn liệu temporal grounding KHÔNG áp thẳng được

Đây là kết luận âm quan trọng nhất của lane, và nó tiết kiệm nhiều nhất.

| | đơn vị thời gian nhỏ nhất | so với luật chấm của ta |
|---|---|---|
| Charades-STA | video TB 30,6 s, **đoạn đáp án TB 8,1 s** | đoạn đáp án dài gấp ~4× ô keyframe của ta |
| QVHighlights | **chú thích trên ô 2 giây**; đoạn TB 24,6 s | ô 2 s ≈ đúng ô keyframe của ta (2,16 s) |
| Gemini File API | lưu video ở **1 fps**, mốc thời gian mỗi giây | thô hơn cửa sổ ±6 khung (0,20 s) 5× |
| **luật chấm AIC** | **một frame nằm trong ±6/10/20 khung = ±0,20/0,33/0,67 s** | — |

(số liệu benchmark: [SMART, arXiv:2511.14143](https://arxiv.org/html/2511.14143v2);
[Gemini video understanding](https://ai.google.dev/gemini-api/docs/video-understanding))

**Đơn vị chú thích mịn nhất mà cả ngành dùng — ô 2 giây của QVHighlights — đúng
bằng một ô keyframe của ta.** Nghĩa là: đầu ra của họ là *đầu vào* của ta. Không
một phương pháp temporal grounding nào trả lời được câu hỏi "khung nào" của luật
chấm AIC; chúng chỉ trả lời được "ô nào".

Và đó lại chính là câu hỏi ta cần (§1, điều 1). Nên cách đọc đúng **không phải**
"bỏ hết văn liệu này", mà là: **dùng nó để chọn ô, rồi để thang frame sẵn có phủ
trong ô.** Mọi đề xuất dưới đây viết theo đúng khuôn đó. Bất cứ đề xuất nào hứa
hẹn "định vị chính xác tới giây" đều phải bị nghi ngờ ngay.

**Bằng chứng thứ hai, mạnh hơn:**
[*Natural-Language Temporal Grounding in Hour-Long Videos is a Search Problem*
(arXiv:2606.12300)](https://arxiv.org/html/2606.12300v1) mổ xẻ lỗi trên video dài
và thấy **85% lỗi là lỗi TÌM (chọn sai vùng), chỉ 11% là lỗi ĐỊNH VỊ (đặt biên
trong vùng đúng)**. Cùng bài: **CLIP ViT-L/14-336 chấm theo khung (mIoU 0,269)
đánh bại MỌI video-LLM mã nguồn mở** trên thang giờ, còn Qwen3.5-9B tụt 5,3 lần
khi chuyển từ Charades sang video dài (0,579 → 0,110). Cấu hình lai họ đề nghị —
CLIP lấy top-3 vùng ±1 phút rồi mới đưa cho LLM — ăn 6,7 lần so với model nguyên
khối (0,354 so 0,053).

Đọc cho ta: kiến trúc hiện tại (SigLIP theo keyframe → phân bổ dòng) **đúng hướng
theo văn liệu mới nhất**, và cách sửa là *thêm tín hiệu chọn ô cho SigLIP*, không
phải thay SigLIP bằng một video-LLM.

---

## 3. Năm đề xuất — chi tiết, mỗi cái trả lời bằng số

### ① Chấm theo PHỦ hai truy vấn con trên một CỬA SỔ *(U-CESE Unified Clipping)*

**Nguồn đã đọc:** [U-CESE, arXiv:2605.23274](https://arxiv.org/html/2605.23274v1)
— hệ thống chung kết chính giải này năm 2025. Thuật toán *Unified Clipping*: gộp
mọi nguồn truy xuất vào **một danh sách**, quét hai con trỏ tuyến tính tạo các cụm
có `end − start ≤ T`, rồi **xếp hạng theo SỐ TRUY VẤN CON RIÊNG BIỆT mà cụm phủ
được**, hoà mới so tới điểm tương đồng cao nhất.

**Nhắm vào phần nào của +126%:** đúng nhóm HAI cảnh, đúng cơ chế hỏng. Ta đã ship
kênh ứng viên cảnh B (`docs/UNG_VIEN_CANH_B.md`, +23,3% trên nhóm qua cổng) —
nhưng ta mới làm **nửa đầu** của U-CESE (hợp danh sách) và chưa làm **nửa sau**
(xếp theo độ phủ). Hiện hai danh sách A và B chỉ được nối đuôi; một ô mà *cả A lẫn
B đều nổi ở lân cận* không hề được ưu tiên hơn ô chỉ có A. Theo §1, **51/65 câu
HAI cảnh đã có ứng viên ≤1 ô** mà chỉ **24/53** được đặt dòng đúng ô — độ phủ
chính là tín hiệu phân biệt ô đúng khỏi ô cảnh A.

**Chạy được trong ràng buộc không:** có, hoàn toàn. Không model mới, không GPU,
không tải video, không nhãn tiếng Việt. Điểm `s_A` và `s_B` cho mọi keyframe của
video ứng viên đã có sẵn (kênh cảnh B đang tính chúng). Chỉ thêm: với mỗi ô `i`,
`phu(i) = 1[s_A nổi trong (i−W..i)] + 1[s_B nổi trong (i..i+W)]`, W ∈ {1,2,3,5}
ô; ô có `phu = 2` được đẩy lên trước trong tiên nghiệm của bộ phủ xác suất.
**Không có hệ số trộn điểm nào phải chọn** — cùng lý lẽ đã dùng cho kênh cảnh B,
và đúng điều U-CESE nhấn mạnh ("merge into a single list", không cộng điểm).

**Đo thế nào trên bộ đo mới:**
1. **Phép đếm tất định trước** (như 53% → 76% của kênh cảnh B): tỷ lệ câu HAI cảnh
   mà **ô đúng nằm trong các ô được phủ** — hiện **24/53**. Đây là số phải nhúc
   nhích; nếu nó đứng yên thì khỏi cần chấm điểm.
2. TUNE/TEST **phân tầng theo `co_2_canh`** (không chia chẵn/lẻ — bước sinh đặt
   câu hai cảnh vào chỉ số chẵn), chọn W trên TUNE, đọc TEST **đúng một lần**.
3. **Bootstrap theo CÂU** (`scripts/do_ung_vien_canh_b.py`, hàm `tung_cau`), báo
   cáo KTC 95% và P(hoà). Không dùng 2σ hạt giống.
4. `assert` 66 câu cổng tắt ra 100 dòng giống hệt nền, với mọi W.

**Rủi ro phải nói trước:** cổng gắn nhãn bật nhầm ở câu một cảnh sẽ đẩy sai ô lên
trước. Khác kênh cảnh B (chèn đuôi; R@k là max trên tiền tố nên gần như vô hại),
đề xuất này **đổi thứ tự đầu danh sách** nên **có thể hại thật**. Bắt buộc báo cáo
riêng nhóm cổng-tắt và nhóm một-cảnh-bị-bật-nhầm.

---

### ② Điểm CHUYỂN CẢNH — chấm *hiệu* giữa hai keyframe, không chấm keyframe

**Nguồn đã đọc:** [TFVTG, arXiv:2408.16219](https://arxiv.org/abs/2408.16219)
(ECCV 2024), mã: [github.com/minghangz/TFVTG](https://github.com/minghangz/TFVTG).
Ý cốt lõi, nguyên văn abstract: VLM huấn luyện trên ảnh / clip-đã-cắt nên **không
nhạy với "dynamic transition of events" — chuyển từ sự kiện này sang sự kiện
kia**. TFVTG tách mỗi sự kiện con thành **phần chuyển động (dynamic transition)**
và **phần trạng thái tĩnh (static status)**, chấm bằng hai hàm khác nhau, rồi dùng
thứ tự thời gian do LLM cấp để lọc và hợp nhất các đề xuất. Zero-shot:
Charades-STA IoU@0.5 **49,97** (mIoU 44,51), ActivityNet IoU@0.5 **27,02**
(mIoU 34,10) — theo README của chính repo.

**Nhắm vào phần nào của +126%:** đúng điểm mù cơ học của ta. SigLIP chấm *một ảnh
khớp mô tả tới đâu*, mà "khoảnh khắc cảnh B bắt đầu" **không phải thuộc tính của
một ảnh** — nó là thuộc tính của **cặp** (ảnh trước, ảnh sau). Bảng tín hiệu nội
bộ đã ghi đúng hình lỗi này ở chỗ khác: *"VLM trả lời 'khung này có khớp mô tả
không', trong khi thứ quyết định điểm là 'khung này có gần khoảnh khắc đúng nhất
không'"*. Điểm chuyển cảnh là hàm mục tiêu đúng cho nhóm hai cảnh.

**Chạy được trong ràng buộc không:** có, và **rẻ đến mức bất thường**. Cosine giữa
mọi cặp keyframe liền kề của cả kho tính hết **1,4 giây** (đã chạy thật:
`tham_do_cau_truc_shot.py` phần B — đọc memmap, không nạp 817 MB vào RAM). Điểm
chuyển cảnh cho ô `i` với truy vấn hai cảnh:
`s_chuyen(i) = s_A(i−1) + s_B(i) − s_A(i) − s_B(i−1)` — thuần đại số trên vector
đã có, không gọi model, không GPU. Ô nào cực đại hoá nó là ô "A vừa hết, B vừa bắt
đầu".

**Đo thế nào:** hệt ① (đếm ô trước, rồi TUNE/TEST phân tầng + bootstrap theo câu).
Chỉ số chẩn đoán riêng, **không dính allocator**: hạng của ô đúng theo `s_chuyen`
trong nội bộ video đúng, so với hạng theo `s_A` thuần — cùng kiểu phép so mà
`SHIP_PHU_XAC_SUAT.md` §2 đã dùng để bóc tách nguyên nhân.

**Cảnh báo phải nói thẳng:** bảng tín hiệu có dòng *"làm mượt theo thời gian |
frame | −0,023 ❌"*. Đề xuất này **không phải** làm mượt — làm mượt kéo điểm về
trung bình lân cận, đây lấy *hiệu*, tức đạo hàm; hai phép ngược dấu nhau. Nhưng
dòng âm kia đo trên bộ 60 câu cũ, nơi *"SigLIP đã đặt keyframe gần đáp án ở hạng
nội-video trung vị 1,0 (hạng-1: 60%)"* — không còn headroom, nên mọi tín hiệu định
vị nội-video đều buộc phải ra ~0. Đó đúng là cửa bị đóng bằng một thước không nhìn
thấy vấn đề. Mở lại là hợp lệ; **nhưng nếu lần này cũng hoà thì ghi vào bảng cửa
đóng và đừng diễn giải lại.**

---

### ③ TAG — gộp thời gian, phân cụm liên kết, hiệu chỉnh điểm

**Nguồn đã đọc:** [TAG, arXiv:2508.07925](https://arxiv.org/pdf/2508.07925), mã:
[github.com/Nuetee/TAG](https://github.com/Nuetee/TAG). Ba khối: *temporal
pooling*, *temporal coherence clustering*, *similarity adjustment*, dựng để chữa
đúng bệnh **phân mảnh đoạn** — điểm cao rải rác thay vì tụ thành một đoạn.
Charades-STA IoU@0.3 đạt 67,82 theo README.

**Nhắm vào phần nào:** chọn ô trong video đúng, cho **cả hai nhóm** — khác ① và ②
chỉ ăn ở nhóm hai cảnh. Đường cong `s(i)` theo ô của một video bản tin có nhiều
đỉnh giả (cùng studio, cùng băng rôn, cùng logo lặp lại suốt bản tin); phạt đỉnh
đơn độc bằng liên kết thời gian là cách chuẩn để chữa.

**Chạy được trong ràng buộc không:** **thuật toán** thì có — nó chỉ đụng vector
tương đồng theo khung. **Bản cài đặt của tác giả thì KHÔNG**: repo dùng đặc trưng
**BLIP-2 ITM trích ở 3 fps**. Với 129,8 giờ kho, 3 fps = **1,4 triệu khung** qua
BLIP-2 ITM, lại còn phải giải mã video gốc mà ta **không có trên máy**. Cửa đó
đóng. Việc đúng là **cài lại ba khối trên lưới keyframe sẵn có** (0,46 khung/giây
hiệu dụng) — vài chục dòng numpy, 0 GPU.

Lưu ý một mâu thuẫn ngay trong repo: abstract nói không cần LLM, còn lệnh chạy lại
nhận `--llm_output`. Nên đọc TAG như **một họ hàm hiệu chỉnh trên đường cong
điểm**, đừng chép nguyên pipeline và đừng trích dẫn nó như "không cần LLM".

**Đo thế nào:** đây là tín hiệu tác động lên **mọi** câu, nên **không có nhóm
cổng-tắt để `assert`** — mất một lớp bảo vệ. Bù bằng: chấm riêng hai nhóm
`co_2_canh`, và **bắt buộc báo cáo song song cả bộ 60 câu cũ** làm đối chứng lịch
sử. Nếu nó ăn ở bộ mới mà phá bộ cũ thì phải nói rõ cả hai, không được chọn bộ nào
có lợi.

---

### ④ Bảng ranh giới SHOT dựng một lần — trả lời trực tiếp câu hỏi của nhiệm vụ

#### 4a. Kho có sẵn ranh giới shot không? — **KHÔNG.** Đã kiểm, kết luận âm

Đáng kiểm trước tiên, vì nếu có thì mọi thứ khác thành miễn phí.
`scripts/tham_do_cau_truc_shot.py` hỏi hai đường rẻ nhất; **cả hai đều âm**:

**(A) Dấu vết bộ trích keyframe.** Bộ trích của BTC không lấy đều: nền ~5–6 s, xen
vào là những **chùm** keyframe cách nhau 0,04 s. Nếu chùm = cú cắt thì ta có bảng
ranh giới miễn phí. Đếm được: **15.200 chùm / 873 video**, 22,6% keyframe nằm
trong chùm, **855/873 video có ít nhất một chùm** — nhưng **nhịp chỉ 2,8
chùm/phút, tức một chùm mỗi 21,5 giây**. Bản tin truyền hình cắt cảnh mỗi ~4–6 s.
Chùm bắt được **chưa tới 1/4** số cú cắt, và 18 video không có chùm nào. **Chùm là
dấu vết của bộ dò chuyển cảnh MỀM (dissolve/wipe), không phải bảng ranh giới.**

**(B) Cosine SigLIP giữa hai keyframe liền kề.** Khe trung vị 2,16 s so nhịp cắt
4–6 s ⇒ **phần lớn cặp liền kề phải vắt qua một cú cắt**. Vậy mà cosine trung vị
là **0,903**, p5 = 0,635, và **chỉ 0,8% số cặp xuống dưới 0,5**. Dải động của
SigLIP quá nén: nó là bộ mã hoá *ngữ nghĩa*, hai cảnh khác nhau trong cùng một bản
tin vẫn rất giống nhau với nó. **Ngưỡng tuyệt đối trên cosine SigLIP không dò được
cắt cảnh.** (Chuẩn hoá z theo video có thể cứu phần nào, nhưng không có nhãn shot
để kiểm — mà đó chính là lý do phải chạy ④.)

Hai kết luận âm này đóng đúng hai đường tắt hấp dẫn nhất, và chúng rẻ: tổng cộng
vài giây máy.

#### 4b. Nếu dựng thật thì bao nhiêu tiền

| công cụ | độ chính xác (F1) | tốc độ | nguồn |
|---|---|---|---|
| **TransNetV2** | **ClipShots 77,9 · BBC 96,2 · RAI 93,9** (số của chính paper, per-frame) | **~250 fps ở 48×27 trên RTX 2080Ti** | [ar5iv:2008.04838](https://ar5iv.labs.arxiv.org/html/2008.04838), [github](https://github.com/soCzech/TransNetV2), [emergentmind](https://www.emergentmind.com/topics/transnetv2) |
| AutoShot | hơn TransNetV2 **4,2%** trên SHOT; 1,1/0,9/1,2% trên ClipShots/BBC/RAI | nặng hơn (NAS 3D-ConvNet) | [arXiv:2304.06116](https://arxiv.org/abs/2304.06116) |
| PySceneDetect | *(thấy trích "F1 < 0,6" trong tóm tắt tìm kiếm, **chưa đối chiếu được với nguồn gốc** — đừng trích lại con số này)* | CPU, nhanh nhất | — |
| thuật toán 4× thời gian thực (2025) | "cực kỳ bền" — nhưng **không báo số so TransNetV2, không thấy mã** | 4× thời gian thực | [arXiv:2502.09202](https://arxiv.org/abs/2502.09202) |
| **DAKE** (U-CESE) | ρ=0,02 bắt **>80% số phát hiện của AutoShot** | **không suy luận model** — chỉ đọc *độ dốc kích thước file JPEG* | [arXiv:2605.23274](https://arxiv.org/html/2605.23274v1) |

> **Đính chính, ghi lại để khỏi lặp:** bản nháp đầu của mục này chép hai con số
> "F1 0,92 trên tin tức/TRECVID2001" và "87,0% so 65,5% của PySceneDetect" từ
> **tóm tắt của công cụ tìm kiếm**. Mở paper gốc ra thì bảng F1 thật là
> 77,9/96,2/93,9 và **không có so sánh với PySceneDetect nào**. Con số "~200 fps
> trên một lõi CPU" cũng chỉ có trong tóm tắt, không có trong paper. Bài học đúng
> bằng bài học của bộ đo: **tóm tắt tìm kiếm không phải nguồn**, và ở đây nó suýt
> hạ chi phí ước tính của ④ đi gần 10 lần.

**Chi phí thật cho kho của ta, tính lại bằng con số đã đối chiếu:** 129,8 giờ ×
~30 fps ≈ **14,0 triệu khung**. Ở **250 fps** (2080Ti) ⇒ **~15,6 giờ GPU**; T4
free chậm hơn đáng kể ⇒ ước **20–30 giờ**, mà một phiên Colab free chỉ ~12 giờ ⇒
**2–3 phiên**, chưa kể lưu trữ tạm. Cộng phần **tải 873 video** (`watch_url` có
đủ trong `media-info-aic25-b1.zip`), ước **~26–60 GB** ở 360–480p, 2–4 giờ băng
thông, cộng rủi ro bị YouTube chặn tốc độ.

Nghĩa là ④ là **việc vài ngày**, không phải việc một buổi chiều — đúng lý do nó
xếp sau ①–③ chứ không phải làm song song. Điểm bù: nó là **offline một lần**, đầu
ra là 873 file JSON vài MB; lúc thi chỉ tra bảng, không tốn giây nào trong 5 phút
mỗi câu. Nếu muốn rẻ hơn: chỉ chạy trên **tập con video hay xuất hiện trong ứng
viên** trước, đo trên bộ đo mới, rồi mới mở rộng.

**Nhắm vào phần nào của +126%:** biến "cảnh B bắt đầu" từ *vùng mờ* thành *một
khung xác định*. Có bảng shot thì ① và ② không phải đoán W: "ô kế tiếp" thành
"**shot kế tiếp**", và neo của cảnh B thành **khung đầu của shot đó** — đúng loại
độ chính xác mà luật ±6 khung đòi. Theo §1, 51/65 câu HAI cảnh đã có ứng viên ≤1
ô, nên phần lớn công việc đúng là "nhảy đúng một shot".

**Vì sao vẫn xếp sau ①–③:** nó chỉ *làm sắc* cơ chế mà ①–② thử được miễn phí. Nếu
① và ② hoà thì tiền đề sai và ④ không cứu được; nếu ① và ② ăn thì ④ là bước nâng
độ chính xác có địa chỉ rõ ràng. **Đừng làm ④ trước ①–②** — đó đúng là lỗi "tiêu
GPU trước khi mua kết luận rẻ" mà `tran_dinh_vi_noi_video.py` được viết ra để
chặn.

**Cửa đóng kèm theo:** [OmniShotCut (arXiv:2604.24762, 27/04/2026)](https://arxiv.org/html/2604.24762v1)
đạt F1 0,883 so 0,814 của **cả** TransNetV2 lẫn AutoShot — nhưng **chỉ trên
benchmark do chính họ dựng**, không báo số trên ClipShots/BBC/RAI/SHOT, và **chưa
phát hành mã/trọng số** (mới hứa mở benchmark). Không dùng được vòng này.

---

### ⑤ VideoPrism-LvT làm chỉ mục CỬA SỔ thứ hai

**Nguồn đã đọc:** [github.com/google-deepmind/videoprism](https://github.com/google-deepmind/videoprism).
Bốn biến thể đã phát hành, **Apache 2.0**: B (114M), L (354M), **LvT-B (248M)**,
**LvT-L (580M)**. Hai bản LvT là **video–text**, cho embedding toàn cục để so
cosine — dùng truy xuất trực tiếp được. Đầu vào
`[batch, num_frames, 288, 288, 3]`, huấn luyện ở 8/16 khung, nội suy được số khung
khác.

**Nhắm vào phần nào:** đây là ứng viên **duy nhất** trong khảo sát này trả lời
được *"cửa sổ này có chứa A-rồi-B không"* **bằng một vector**, thay vì bằng luật
ghép do ta viết tay. Nếu ①/② hoà **vì luật ghép quá thô**, ⑤ là đường lùi có cơ
sở. Nếu ①/② hoà vì tiền đề sai thì ⑤ cũng vô nghĩa — nên thứ tự vẫn là ①② trước.

**Chạy được trong ràng buộc không — có, nhưng đắt nhất nhóm khả thi.** Không cần
video gốc: cửa sổ 8 keyframe liên tiếp đọc thẳng từ `data/frames/`. Stride 4 ⇒
**~44.000 cửa sổ** phủ 177.321 keyframe. LvT-L 580M trên T4 free: ước **2–5 giờ**,
vừa một phiên Colab, cộng thời gian đọc ~352 nghìn lượt ảnh. **Trở ngại thật là
JAX/Flax**: repo ghi rõ "Add PyTorch model support" **vẫn là việc chưa làm**. Ta
chưa có đường JAX nào; dựng nó tốn nửa ngày và dễ vỡ trên Colab free.

**Cổng rẻ bắt buộc trước khi tốn GPU** (đúng khuôn đã dùng cho PE-Core): nhúng chỉ
các cửa sổ của những video đúng ở **29 câu HAI cảnh mà ô đúng hiện không được
phủ**, đo hạng của cửa sổ chứa đáp án. Kéo được ≥1/3 số đó vào top-3 cửa sổ mới
đáng index toàn kho.

**Cảnh báo bộ nhớ:** `KhoSims` đã giữ ~780 MB chỉ mục SigLIP; nạp thêm một model
580M trong cùng tiến trình là đường thẳng tới SEGFAULT không traceback. ⑤ phải
chạy ở **tiến trình riêng**, xuất `.npy`, không bao giờ nạp chung.

---

## 4. Trả lời thẳng câu hỏi "có kỹ thuật nào khai thác cấu trúc SHOT không?"

Có. Văn liệu 2025–2026 nói ba điều rời nhau:

1. **SMART** ([arXiv:2511.14143](https://arxiv.org/abs/2511.14143), 18/11/2025) —
   *Shot-Aware Multimodal Video Moment Retrieval*: dùng **cấu trúc thời gian cấp
   shot** làm khung nén token ("Shot-aware Token Compression": giữ token nhiều
   thông tin **trong mỗi shot**). Ăn **+1,61% R1@0.5** và **+2,59% R1@0.7** trên
   Charades-STA so SOTA. Đây là xác nhận độc lập rằng **shot là đơn vị đúng** cho
   moment retrieval. Nhưng: nó là framework **MLLM**, trang abstract không nói
   dùng bộ dò shot nào và không nói có mã; và mức tăng 1,6–2,6% là mức bộ đo
   66-câu của ta **không phân xử nổi**. Dùng nó làm **lý lẽ ủng hộ ④**, đừng dùng
   làm công thức.
2. **U-CESE / DAKE** — trích keyframe theo **độ dốc kích thước file JPEG**, ρ=0,02
   bắt >80% phát hiện của AutoShot, **không cần suy luận model**. Bộ dò ranh giới
   rẻ nhất tồn tại — *nếu* ta có video gốc.
3. **ClipTBP** ([arXiv:2604.27591](https://arxiv.org/html/2604.27591v1),
   30/04/2026) — học quan hệ ngữ nghĩa giữa các đoạn đáp án bằng ba hàm mất mát
   (clip-level similarity, main boundary, auxiliary boundary), gắn lên FlashVTG
   cho QVHighlights R1@0.7 53,61. **Cần HUẤN LUYỆN, đặc trưng SlowFast** ⇒ đóng
   cửa theo đúng nguyên tắc đã đóng nhánh DETR.

**Kết luận cho mục 3 của nhiệm vụ:** ranh giới shot đúng là thứ biến "khoảnh khắc
cảnh B bắt đầu" thành một điểm xác định — nhưng **kho của ta chưa có nó** (§4a,
hai phép đếm âm), và lấy nó về đòi tải 873 video. Trong khi đó ① và ② khai thác
**xấp xỉ ở mức ô-keyframe** của cùng ý tưởng với chi phí bằng không, và §1 nói sai
số cần chữa **đúng bằng một ô**. Thứ tự đúng: ①② → nếu ăn thì ④ để làm sắc → ⑤ nếu
luật ghép tay hoá ra là chỗ nghẽn.

---

## 5. Kết luận ÂM — cửa đóng, ghi để khỏi ai thử lại

| hướng | vì sao ĐÓNG | nguồn |
|---|---|---|
| **MLLM/video-LLM chấm mốc thời gian trực tiếp** | MarkIt bản **training-free**: Charades-STA **R@0.5 = 11,6 / R@0.7 = 3,0 / mIoU 21,8**; phải fine-tune mới lên 43,2/21,8. Ở dạng ta dùng được (không huấn luyện) nó gần như vô dụng — và nó **không xuất mốc thời gian**, nó xuất câu văn "From x to y". | [arXiv:2604.25886](https://arxiv.org/html/2604.25886v1) |
| **Agent MLLM cho truy vấn nhiều sự kiện** | CoMET-Agent training-free thật, nhưng **F1@0.5 chỉ 10,1→16,2% (GPT-5)**, 14,6→19,0% (Gemini 3 Flash) trên video TB 33,8 phút. Mức tuyệt đối quá thấp; lại phải đẩy video vào MLLM — ta không có video gốc và không có quota. | [arXiv:2606.15320](https://arxiv.org/html/2606.15320) |
| **REZE** (zero-shot, model-agnostic, rẻ token) | Cơ chế hay (tách nhận dạng khỏi gộp thời gian; 1,1–1,4k token/truy vấn, <0,35 GB, 7 VLM). Nhưng **0,27 truy vấn/giây** và phải đẩy clip video vào VLM; và mục hạn chế **tự nói** nó *"không mô hình hoá quan hệ giữa các clip"*, *"thứ tự thời gian… nằm ngoài thiết kế hiện tại"* — tức nó **không giải nhóm HAI cảnh**, đúng nhóm ta cần. | [arXiv:2608.04480](https://arxiv.org/html/2608.04480v1) |
| **ClipTBP / FlashVTG / SG-DETR và họ có huấn luyện** | Cần nhãn huấn luyện + đặc trưng SlowFast. Trùng đúng lý do đã đóng nhánh DETR. | [arXiv:2604.27591](https://arxiv.org/html/2604.27591v1) |
| **OmniShotCut** | Chưa phát hành mã/trọng số; số đẹp chỉ trên benchmark tự dựng. | [arXiv:2604.24762](https://arxiv.org/html/2604.24762v1) |
| **GranAlign** (SOTA training-free QVHighlights, +3,23% mAP@avg) | Cần **viết lại truy vấn nhiều mức granularity** *và* **sinh caption theo truy vấn** cho nội dung video. Khoản sau là caption toàn kho theo từng truy vấn — cùng bậc chi phí với "OCR toàn kho bằng API" đã đóng, và kênh caption nội bộ đã đo âm. | [arXiv:2601.00584](https://arxiv.org/abs/2601.00584) |
| **ReCap toàn kho** (U-CESE) | Gemini có nhớ ngữ cảnh, **mỗi shot một lần gọi**. 129,8 giờ ở ~4–6 s/cảnh ⇒ **~90–120 nghìn lượt gọi**. Free-tier không tới gần. *(Bản có địa chỉ — chỉ caption ~20 cửa sổ ứng viên lúc thi — là chuyện khác, và nó thuộc nhóm ①.)* | [arXiv:2605.23274](https://arxiv.org/html/2605.23274v1) |
| **TAG nguyên bản** | Đặc trưng **BLIP-2 ITM ở 3 fps** = 1,4 triệu khung, và phải có video gốc. Chỉ **thuật toán** dùng lại được (③). | [github.com/Nuetee/TAG](https://github.com/Nuetee/TAG) |
| **Trích lại keyframe dày hơn (DAKE) để có lưới mịn hơn** | Lưới mới ⇒ **phải nhúng lại toàn kho bằng SigLIP-2 SO400M**, tức vứt toàn bộ chỉ mục 177.321 vector đang có. Chi phí bằng một lần dựng kho. Và §1 nói ta **không thiếu độ phân giải lưới** — thang frame đã phủ trong ô; ta thiếu *chọn đúng ô*. | — |
| **Bộ dò cắt cảnh miễn phí từ chính chỉ mục SigLIP** | Đo rồi: cosine liền kề trung vị 0,903; chỉ 0,8% xuống dưới 0,5, trong khi phần lớn cặp phải vắt qua một cú cắt. Dải động quá nén. | `scripts/tham_do_cau_truc_shot.py` |
| **Chùm keyframe trong metadata = bảng ranh giới shot** | Đo rồi: 2,8 chùm/phút = một chùm mỗi 21,5 s, so nhịp cắt thật 4–6 s ⇒ bắt chưa tới 1/4; 18/873 video không có chùm nào. | `scripts/tham_do_cau_truc_shot.py` |

---

## 6. Khuôn đo bắt buộc — dùng chung cho ①–⑤

Không đề xuất nào được vào sản xuất nếu thiếu bốn thứ sau. Đây không phải thủ tục:
mỗi dòng tương ứng với một lần bộ đo đã nói dối trong chính dự án này.

1. **Phép đếm tất định TRƯỚC phép chấm điểm.** Chỉ số phải nhúc nhích là *"ô
   keyframe đúng có nằm trong các ô được phủ không"* — hiện **24/53** ở nhóm HAI
   cảnh, **41/51** ở nhóm MỘT cảnh. Phép đếm, không có khoảng tin cậy để bàn, và
   nó nói cơ chế có chạy hay không **trước khi** nhiễu thống kê xen vào. (Đúng
   khuôn 53% → 76% đã dùng cho kênh cảnh B.)
2. **Chia TUNE/TEST PHÂN TẦNG theo `co_2_canh`.** Tuyệt đối không chẵn/lẻ: bước
   sinh đặt câu hai cảnh vào chỉ số chẵn, chia thô sẽ cho TEST một **phép đồng
   nhất** mà script vẫn in ra con số trông bình thường. Chọn trên TUNE, **đọc TEST
   đúng một lần**.
3. **Bootstrap theo CÂU**, không phải 2σ hạt giống. Với n = 66 mỗi nhóm, nguồn bất
   định chính là *đổi tập câu*, không phải *đổi hạt giống*; tăng số lần bốc chỉ
   làm σ nhỏ đi mà chẳng biết thêm gì về câu hỏi. Đã có ca cụ thể: 2σ hạt giống
   tuyên bố "giữ được" cho một hiệu ứng mà bootstrap theo câu nói còn 14% khả năng
   hoà. Mẫu: `scripts/do_ung_vien_canh_b.py`, hàm `tung_cau`.
4. **`assert` bất biến.** Câu không thuộc diện tác động phải ra 100 dòng **giống
   hệt** nền, với mọi giá trị tham số. Với ③ (tác động lên mọi câu) không có nhóm
   để assert — bù bằng cách bắt buộc báo cáo song song **cả bộ 60 câu cũ**.

Luật đọc kết quả: **kết luận ÂM có số liệu là kết quả có giá trị.** Nếu ① hay ②
hoà, ghi vào bảng cửa đóng của `KIEN_TRUC_VA_HUONG_CAI_THIEN.md` với đúng con số,
**không** diễn giải thành "có tiềm năng". Nếu một hiệu ứng đổi dấu hoặc đổi cấu
hình thắng khi thêm dữ liệu, đó là dấu hiệu **ước lượng thổi phồng** và phải nói
thẳng — lever ③ đã dính đúng chuyện này.

---

## 7. Điều tài liệu này CHƯA biết — đọc trước khi tin

1. **n = 66 mỗi nhóm là nhỏ.** Đủ để thấy khoảng cách 24/53 so 41/51; không đủ để
   phân xử chênh lệch vài phần trăm. Mọi ô "tác động ước tính" ở §0 là **phán đoán
   từ cơ chế**, không phải phép đo — đừng trích chúng như số đo.
2. **Câu do máy sinh có thiên lệch chưa đo được.** Bằng chứng gián tiếp duy nhất
   vẫn là điểm khớp neo đồng đều giữa hai nhóm (`BO_DO_KHOP_PHAN_BO.md` §2). Nếu
   câu máy sinh mô tả *chuyển cảnh* rõ hơn đề thật thì ① và ② sẽ **được thổi
   phồng** trên bộ đo này.
3. **Nhịp cắt cảnh 4–6 giây của bản tin là con số văn liệu, chưa đo trên kho
   này.** Toàn bộ lập luận §4a dựa vào nó. Đo được rẻ (chạy dò cắt cảnh trên 5–10
   video là đủ) và **nên đo trước khi trích dẫn lại §4a**.
4. **Ô keyframe rất không đều**: khe p25 = 1,04 s, p75 = 4,40 s, max 8,0 s. Ở đuôi
   trên, một ô 8 giây (~240 khung) cần ~24 dòng mới phủ kín bằng thang bước 10 —
   nhiều hơn trung vị 15–16 dòng đang có. **Câu rơi vào ô rộng vẫn trượt dù chọn
   đúng ô.** Chưa ai đếm bao nhiêu câu rơi vào đó; đây là phép đếm rẻ và nó đặt
   trần cho cả ① lẫn ④.
5. **Độ chính xác của cổng gắn nhãn hai cảnh trên đề THẬT vẫn chưa ai kiểm từng
   câu** — cùng điều chưa biết đã ghi ở `UNG_VIEN_CANH_B.md`. Với ① nó nguy hiểm
   hơn hẳn, vì ① đổi thứ tự **đầu** danh sách chứ không chèn đuôi.
6. **Chưa có câu TRAKE nào trong bộ đo.** Mọi kết luận ở đây là cho KIS/Q&A; nhánh
   TRAKE vẫn bị chặn y như trước.
