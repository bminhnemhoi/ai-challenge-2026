# Phản biện: bốn thứ cả bốn lane đều bỏ lỡ

Chốt 01/09/2026. Script đo: `scripts/phan_bien_thieu_sot.py` (chỉ tạo file mới,
không sửa file sản xuất, không nạp `KISEngine`).

> Không mã video, không đáp án trong tài liệu này (`docs/` lên GitHub công khai).

Bốn lane vừa rồi (phân-bố-sâu, tín-hiệu-nội-video, paper-2026, vlm-nội-video)
đo **cùng một đại lượng**: điểm KIS của 100 dòng, trên cùng 132 mục, với cùng
giả định "tập dòng là biến, thứ tự dòng là hằng". Tài liệu này đo bốn thứ nằm
ngoài vòng tròn đó, bằng cùng đường sản xuất và cùng kỷ luật đo.

Nền dùng chung: bộ sạch 132 mục, allocator `coverage`, gốc hạt **91000**
(tách khỏi 77000/81000/310000/320000/330000), 4 họ × 48 bốc, cửa sổ {6,10,20}.

| | TẤT CẢ | MỘT cảnh | HAI cảnh |
|---|---|---|---|
| nền | 0,1896 | 0,2757 | 0,1036 |

---

## §0. Bảng xếp hạng sáu hướng

| # | hướng | trạng thái bằng chứng | chi phí |
|---|---|---|---|
| 1 | **Kênh Q&A — 27% đề thật, chưa đo một lần nào, và có một nhân tử 0/1 đứng trước mọi điểm định vị** | đếm tất định: 62/66 câu hai cảnh có **0/5** khung đúng khe trong số khung VLM được nhìn | ~$1–3 API, 0 GPU |
| 2 | **Thứ tự 100 dòng là trục thứ hai, độc lập với trục đặt-frame; trần +74% (một cảnh) / +131% (hai cảnh)** | trần đo được; 6 hoán vị rẻ đã thử: 1 dương yếu trên bộ mới nhưng **âm có ý nghĩa** trên bộ cũ | 0 GPU |
| 3 | **Trần theo ngân sách dòng — trả lời gợi ý (a): mở ngân sách KHÔNG giúp** | ORACLE-VIDEO = 0,4114; bề-rộng-trước âm đơn điệu −2,0% → −11,4% | 0 (đã đo) |
| 4 | **TRAKE — 8,5% đề thật, 0 câu trong mọi bộ đo, và `reserve_tail_rows` đã viết + đã test nhưng không được dùng** | lập luận cấu trúc, **chưa đo được**; phải sinh ~20 mục TRAKE trước | ~$1–2 sinh bộ đo |
| 5 | **Bác bỏ: "khe keyframe không đều nên sigma toàn cục sai"** | ÂM — ba phân bố khe gần trùng nhau (trung vị 66/74/72 khung) | 0 (đã đo) |
| 6 | **Hai bộ đo đang bất đồng về HÌNH DẠNG bài toán; bất đồng đó giờ là bất định ràng buộc** | hai trục độc lập (sigma, thứ tự dòng) cùng một chữ ký đảo dấu | ~15 phút chạy, 0 API |

---

## §1. Kênh Q&A — chỗ 27% điểm đang bị nhân với một số 0/1 chưa ai đo

### 1a. Cấu trúc chấm, đọc thẳng từ mã

`src/core/evaluator.py::calculate_vqa_r_score` yêu cầu **cả ba**: đúng video,
frame trong khoảng, **và đáp án khớp**. `scripts/make_submission.py::build_qa_rows`:

```
hits   = merged_hits(engine, context or query_text, query_en)
cands, _ = them_ung_vien_canh_b(engine, cands, ...)     # <- cands, KHÔNG phải hits
frame_rows = allocate_rows(cands, ...)
answer = answerer(hits[:5], context, question)          # <- hits[:5]
return [(v, f, answer) for v, f in frame_rows]          # MỘT đáp án trên CẢ 100 dòng
```

Ba hệ quả không ai ghi ở đâu:

1. **Điểm Q&A = điểm định vị × 1[đáp án đúng].** Sai đáp án thì cả 100 dòng bằng
   0, bất kể định vị đúng đến đâu. Mọi con số của bốn lane là điểm định vị —
   tức là *thừa số thứ nhất*. Thừa số thứ hai chưa từng được đo.
2. **Lever cảnh B đã ship không chạm vào đường sinh đáp án.** Nó sửa `cands`;
   `answerer` đọc `hits`. Câu hai cảnh vẫn được trả lời bằng khung cảnh A.
3. **Biểu quyết đa số khuếch đại lỗi.** `_make_answerer` trả về
   `Counter(votes).most_common(1)[0][0]`. Nếu 4/5 khung là cảnh A và 1/5 là cảnh
   B, phiếu của khung đúng **bị bỏ**. Đây không phải "hơi kém" mà là một cơ chế
   chủ động loại bỏ khung đúng.

### 1b. Đếm tất định — trong 5 khung VLM được nhìn, bao nhiêu khung ở gần neo

Khoảng cách tính theo **ô keyframe** (0 ô = đúng khung neo).

| tập 5 khung | nhóm | 0 khung đúng ô | ≥1 đúng ô | ≥1 cách ≤1 ô | ≥3 cách ≤2 ô | TB khung đúng video |
|---|---|---|---|---|---|---|
| `hits[:5]` (**hiện tại**) | MỘT cảnh | 46/66 | 20 | 30 | 0 | 1,61 |
| `hits[:5]` (**hiện tại**) | HAI cảnh | **62/66** | **4** | 14 | 2 | 1,50 |
| top-1 của 5 video đầu | MỘT cảnh | 50/66 | 16 | 27 | 0 | 0,64 |
| top-1 của 5 video đầu | HAI cảnh | 66/66 | 0 | 8 | 0 | 0,56 |
| top-5 trong video #1 | MỘT cảnh | 47/66 | 19 | 25 | 3 | 1,94 |
| top-5 trong video #1 | HAI cảnh | 61/66 | 5 | 13 | 6 | 1,88 |
| **[oracle] top-5 của video ĐÚNG** | MỘT cảnh | 21/66 | 45 | 54 | 9 | 4,18 |
| **[oracle] top-5 của video ĐÚNG** | HAI cảnh | 56/66 | 10 | 27 | 8 | 4,67 |

Đọc bảng:

- Ở câu hai cảnh, **62/66 câu VLM không được nhìn một khung nào của khoảnh khắc
  phải trả lời**. Đáp án chỉ đúng nếu đoán mò hoặc nếu câu hỏi trả lời được từ
  ngữ cảnh chung của bản tin.
- Trung bình chỉ **1,5/5 khung nằm trên đúng video** — phần lớn phiếu đến từ
  video khác hẳn.
- **Ngay cả chọn đúng video cũng không cứu** câu hai cảnh (56/66 vẫn 0 khung
  đúng ô): ứng viên trong video đúng cũng là ứng viên cảnh A. Nghĩa là thứ cần
  cho đường Q&A đúng là **xếp lại hạng nội-video theo cảnh B** — chính lever mà
  lane vlm-nội-video đo được +80,5% và **chưa ship**. Nó có một kênh thứ hai
  chưa ai tính vào.

### 1c. Đo thế nào, giá bao nhiêu

- `ground_truth_moi.json` đã có `vqa_question` / `vqa_answer` cho cả 132 mục;
  `ground_truth_de_that.json` có **8 mục Q&A ĐỀ THẬT đã người kiểm chứng đáp án**.
- Phép đo: 4 tập 5 khung ở bảng trên × 132 câu × 5 lần gọi = 2.640 lần gọi ảnh,
  cache theo (model, khung, câu hỏi) như `data/vlm/` đang làm ⇒ **~$1–3**.
- Đại lượng chốt: `P(đáp án đúng)` theo `_default_answer_match`, báo cáo **riêng
  nhóm HAI cảnh**, chia TUNE/TEST phân tầng theo `co_2_canh`, bootstrap theo câu.
- Nhóm bị tác động = câu Q&A; câu KIS ra 100 dòng giống hệt nền (assert).
- **Rẻ hơn nữa và làm trước**: 8 mục đề thật đã kiểm chứng — 40 lần gọi, ~$0,05.
  Không chốt được gì với n=8, nhưng nếu 0/8 hoặc 1/8 đúng thì đó là một phép đếm
  đủ để đổi thứ tự ưu tiên của cả đội.

---

## §2. Thứ tự 100 dòng — trục thứ hai, độc lập, chưa ai đụng

### 2a. Trần

Điểm = `bucket(hạng của dòng ĐÚNG đầu tiên)`. Một **hoán vị** 100 dòng không đổi
tập dòng ⇒ **R@100 bất biến** ⇒ rủi ro bị chặn cứng ở các bucket 1/5/20/50.

| | nền | hoán vị oracle (đưa dòng đúng về hạng 1) | |
|---|---|---|---|
| TẤT CẢ | 0,1896 | 0,3593 | **+89,5%** |
| MỘT cảnh | 0,2757 | 0,4797 | **+74,0%** |
| HAI cảnh | 0,1036 | 0,2388 | **+130,6%** |

Đây là **trục khác** với trần đặt-frame mà lane phân-bố-sâu đo (+126% oracle /
+100% theo pool). Trần đặt-frame trả lời "dòng nằm sai chỗ"; trần này trả lời
"dòng đúng đã có sẵn nhưng bị chôn".

### 2b. Trả lời gợi ý (c): vì sao nhóm MỘT cảnh cũng có headroom

Phân bố hạng của dòng đúng đầu tiên (nền, 100 dòng):

| nhóm | hạng 1 | 2–5 | 6–20 | 21–50 | 51–100 | không có dòng đúng |
|---|---|---|---|---|---|---|
| MỘT cảnh (n=66) | 5,4% | 9,7% | 14,1% | 10,9% | 7,8% | 52,0% |
| HAI cảnh (n=66) | 0,7% | 3,2% | 3,4% | 8,9% | 7,8% | 76,1% |

**Nhóm MỘT cảnh đã có dòng đúng ở 48% số câu, nhưng 32,9 điểm phần trăm trong
số đó nằm ở hạng ≥6.** Deficit thuần do THỨ TỰ = **0,2040** — *lớn hơn* toàn bộ
headroom đặt-frame theo pool mà lane phân-bố-sâu đo cho nhóm này (0,4255 −
0,2767 = 0,1488). Đó là câu trả lời cho "vì sao nhóm một cảnh cũng có +59%":
**phần lớn không phải bài toán định vị, mà là bài toán xếp thứ tự dòng.**

Nhóm HAI cảnh: deficit do thứ tự = 0,1353, đứng sau deficit do pool/đặt-frame.

### 2c. Sáu hoán vị rẻ đã thử — và kết quả ÂM

Tất cả đều assert `sorted(rows_mới) == sorted(rows_nền)`.

| hoán vị | TẤT CẢ | MỘT cảnh | HAI cảnh |
|---|---|---|---|
| bề-rộng-trước B=2 | −2,0% | −1,8% | −2,5% |
| bề-rộng-trước B=5 | −7,5% | −7,2% | −8,3% |
| bề-rộng-trước B=20 | −11,4% | −11,4% | −11,5% |
| **gom theo VIDEO (khối liền)** | **+4,3%** | +1,0% | **+13,2%** |
| sắp theo MẬT ĐỘ tiên nghiệm | −0,1% | −0,1% | +0,0% |
| sắp theo ĐIỂM ứng viên gần nhất | +0,5% | −3,5% | +11,2% |
| đảo ngược 100 dòng (đối chứng) | −44,9% | −53,0% | −23,5% |

Giao thức chốt cho họ này (TUNE/TEST phân tầng theo `co_2_canh`, ngẫu nhiên
trong tầng — **không** chẵn/lẻ; chọn trên TUNE, đọc TEST một lần, bootstrap
4000 lần theo câu):

- TUNE (n=66, nền 0,2112): `gom_video` +3,1% (và K=5, K=10 cho **số y hệt** —
  đường tham số phẳng, dấu hiệu cơ chế thật).
- TEST (n=66, nền 0,1711): **+5,5%**, KTC95 [−0,0024, +0,0218], P(≤0) = 6,5%.
  Nhóm HAI cảnh +10,1% (P = 8,8%); MỘT cảnh +2,4% (P = 22,6%).
  Phân rã tất định: 14 câu tốt lên / 6 câu xấu đi / 46 câu không đổi.
- **KIỂM CHÉO bộ 60 câu CŨ (tập độc lập, đọc một lần): −11,5%, KTC95
  [−0,0712, −0,0230], P(≤0) = 100%. 9 tốt lên / 17 xấu đi.**

**Kết luận §2: ÂM.** `gom_video` không phải cải tiến chung — nó là một đánh đổi
giữa hai phân bố, y hệt chữ ký của trục sigma trong `PHAN_BO_TREN_BO_MOI.md`.
Thứ **còn lại** là trần: +74% / +131% headroom thứ tự vẫn đứng nguyên, và sáu
hoán vị rẻ không chạm tới nó. Muốn khai thác phải có **tín hiệu xếp hạng dòng**
thật (ví dụ điểm cảnh-B nội-video của lane vlm), không phải một quy tắc sắp xếp.

---

## §3. Trần theo ngân sách dòng — trả lời gợi ý (a)

Trần oracle +126% giữ nguyên **số dòng mỗi video**. Mở ràng buộc đó ra:

**(a) ORACLE-VIDEO** — chỉ giữ ứng viên của video đúng, chạy allocator THẬT,
toàn bộ 100 dòng dồn cho một video:

| | nền | oracle-video | |
|---|---|---|---|
| TẤT CẢ | 0,1896 | 0,4114 | +117,0% |
| MỘT cảnh | 0,2757 | 0,5526 | +100,4% |
| HAI cảnh | 0,1036 | 0,2703 | +161,0% |

Video đúng có mặt trong 400 ứng viên ở **130/132** câu ⇒ trần tuyệt đối nếu chọn
đúng video **và** đặt đúng dòng là **0,9848**. Biết trước video và dồn cả ngân
sách vào nó chỉ đưa được 0,1896 → 0,4114, tức **vẫn mất 58% quãng đường**.

**(b) Bề-rộng-trước là ÂM ĐƠN ĐIỆU** (bảng §2c): −2,0% ở B=2 tới −11,4% ở B=20.

**(c)** Số dòng rơi vào video đúng, **trung vị = 9** (không phải 21,5 — đó là số
*trung bình* mà lane phân-bố-sâu báo cáo; phân bố lệch mạnh). Phủ trọn một khe
keyframe trung vị cần 4,8 dòng, khe p90 cần 9,3 dòng.

**Trả lời gợi ý (a), bằng số: KHÔNG. Đổi ngân sách dòng giữa các video không
mở thêm gì.** Ngay cả oracle ngân sách + oracle video vẫn kẹt ở 0,41. Tường
nằm ở **chọn đúng ô trong video**, không ở việc chia bao nhiêu dòng cho ai.
Đây là một kết luận ÂM có số liệu, và nó **hạ giá** cách đọc "+126% là headroom
của khâu phân bổ".

---

## §4. TRAKE — 8,5% đề thật, 0 câu trong mọi bộ đo

Thành phần đề thật (`ground_truth_de_that.json`, 59 mục ba vòng): **KIS 38 /
Q&A 16 / TRAKE 5**. Riêng vòng 2: KIS 14 / Q&A 9 / TRAKE 2 ⇒ **44% số câu vòng 2
nằm ngoài mọi phép đo của bốn lane.**

Ba sự thật cấu trúc, đọc từ mã:

1. `build_trake_rows` gọi `align_sequence(..., top_k=1)` rồi `results[0]` ⇒
   **cả 100 dòng nằm trên đúng một video**. Video sai ⇒ 0 điểm, không có lưới an toàn.
2. Điểm TRAKE **có điểm từng phần**: `matched/N`, không nhị phân như KIS. Nên
   R@k = max trên tiền tố của một đại lượng *liên tục* — luật phân bổ khác hẳn KIS.
3. `src/core/submission.py::reserve_tail_rows` **đã viết, đã có test**
   (`tests/test_reserve_tail.py`) và **không được `make_submission.py` gọi ở đâu**.
   Đó đúng là cơ chế cần cho một cửa lui: dòng 51–100 chỉ đáng trọng số 0,2, nên
   nhường 20 dòng cuối cho video xếp hạng 2 tốn phần jitter xa nhất (ít khả năng
   nhất) và thu về 0,2 × (matched/N) mỗi khi video 1 sai.

**Nói thẳng: đây là lập luận quyết định, KHÔNG phải phép đo.** Không có một câu
TRAKE nào trong bất kỳ ground truth nào, nên không ai biết `P(video 1 đúng)` là
0,2 hay 0,8 — mà toàn bộ giá trị của cửa lui nằm ở con số đó. **Việc phải làm
trước là sinh ~20 mục TRAKE**, dùng đúng bộ máy đã có: `sinh_gt_doan_video.py`
đã lấy đoạn liên tiếp 8–12 keyframe (một mục TRAKE = một đoạn + N mốc sự kiện),
và `kiem_neo_don_anh.py` (một-ảnh-một-request) xác minh từng mốc — đúng quy trình
đã chặn được lỗi lệch-một-khung lần trước. Giá ~$1–2.

---

## §5. Bác bỏ: "khe keyframe không đều nên sigma toàn cục là sai"

Giả thuyết hấp dẫn: `sigma=30` là hằng số toàn cục trong khi khe keyframe trải
từ ~1 s tới 8 s, nên bộ so sigma trên hai bộ đo phải cho kết quả ngược chiều.
**Đo rồi: SAI.**

Bề rộng khe keyframe chứa đáp án (khung; harness bốc ĐỀU trong khe này):

| bộ | n | p25 | trung vị | p75 | p90 | max |
|---|---|---|---|---|---|---|
| bộ CŨ 60 câu | 60 | 34 | 66 | 95 | 138 | 164 |
| bộ MỚI – MỘT cảnh | 66 | 47 | 74 | 112 | 150 | 172 |
| bộ MỚI – HAI cảnh | 66 | 59 | 72 | 104 | 139 | 178 |

Ba phân bố **gần trùng nhau**. Hình học khe **không** giải thích được nghịch lý
sigma. Đóng cửa này trước khi ai kịp bỏ tiền vào "sigma thích nghi theo khe".

Con số dùng được rút ra từ đây: mỗi dòng phủ `nua = max(1, 6//5) = 1` ô lưới mỗi
bên = **15 khung**. Phủ trọn khe trung vị cần **4,8 dòng**, khe p90 cần **9,3
dòng** — trong khi trung vị số dòng rơi vào video đúng chỉ là **9** (§3c). Tức
ở một nửa số câu, ngân sách trên video đúng vừa đủ phủ **một** khe. Không có chỗ
cho sai lầm chọn ô.

---

## §6. Thứ ở tầng trên mà cả bốn lane đều bỏ lỡ

Hai trục **độc lập nhau** giờ đã cho **cùng một chữ ký**:

| trục | bộ MỚI | bộ CŨ (đối chứng độc lập) |
|---|---|---|
| bề rộng không gian (sigma 45, lane phân-bố-sâu) | +25,4% TUNE / hoà TEST | **−8,2%, P(≤0)=99,8%** |
| thứ tự dòng (`gom_video`, lane này) | +3,1% TUNE / +5,5% TEST | **−11,5%, P(≤0)=100%** |

Đây không còn là một sự cố về sigma. Đây là **hai bộ đo bất đồng về hình dạng
bài toán**, và mọi lever mang tính "trải rộng hơn / gom sâu hơn" sẽ đâm vào đúng
bức tường này. Bất định ràng buộc của cả dự án hiện giờ **không phải là thiếu
lever, mà là không biết đề thật giống bộ nào**.

Bằng chứng gián tiếp duy nhất đang có: điểm vòng 2 = 10,0/30 = **0,333/câu**
(đã có công soát tay), nằm **giữa** bộ cũ (0,400 tự động) và bộ mới (0,190).

**Đề xuất, và nó rẻ:** chấm đường sản xuất trên 59 mục đề thật
(`ground_truth_de_that.json`: 15 người kiểm chứng + 44 suy ra) và so **HÌNH
DẠNG**, không so điểm:

- phân bố hạng của dòng đúng đầu tiên (bảng §2b) — bộ cũ tập trung ở hạng 1–5,
  bộ mới trải ra hạng 6–100;
- hạng nội-video của keyframe đáp án (bộ cũ trung vị 2,0 / hạng-1 45%; bộ mới
  HAI cảnh trung vị 6,0 / hạng-1 11%);
- tỉ lệ câu hai cảnh có keyframe đáp án trong pool 400 (bộ mới: 35/66).

Ba đại lượng này là **phép đếm**, không phải điểm trung bình, nên n=15 (và n=59
với nhãn yếu hơn) vẫn nói được điều gì đó — trong khi so điểm ở n=15 thì không.
Chi phí: một lượt `ranked_hits` trên 59 truy vấn (~15 phút, cần `KISEngine`) +
đếm. **0 lần gọi API.** Chưa ai làm: `grep` cho thấy chỉ
`thu_hoach_de_that.py` và `experiment_cap_thoi_gian.py` từng chạm tới file này,
và không script nào chấm đường sản xuất trên nó.

---

## §7. Giới hạn của chính tài liệu này — đọc trước khi trích

1. **§2c đã đọc TEST của bộ mới thêm một lần nữa.** Bộ 132 mục nay đã bị đọc
   TEST bởi ít nhất bốn lane với bốn phép chia khác nhau. Con số +5,5% phải đọc
   với hiểu biết đó, và nó **âm có ý nghĩa trên bộ đối chứng** nên kết luận là ÂM.
2. **n = 66 mỗi nửa vẫn là NHỎ**, và hai nửa lệch nền (TUNE 0,2112 vs TEST
   0,1711). Phép đo này đủ sức bác một hiệu ứng +25%, không đủ sức bác +5%.
3. **§1 chưa có một con số điểm nào.** Toàn bộ là đếm tất định về *khung nào
   được đưa cho VLM*. Bước từ "62/66 câu không được nhìn khung đúng" tới "đáp án
   sai ở 62/66 câu" **chưa được đo** — VLM có thể trả lời đúng nhờ ngữ cảnh
   chung của bản tin, hoặc đáp án có thể là logo kênh (bộ đo đã đánh dấu vài ca
   như vậy). Đó chính là phép đo phải chạy, không phải kết luận đã có.
4. **§4 không có bằng chứng đo lường nào.** Không có mục TRAKE nào tồn tại.
   Đừng trích "hedge TRAKE đáng 0,2×(matched/N)" như một con số đã đo.
5. **§3 ORACLE-VIDEO vẫn là oracle** — nó dùng đáp án để chọn video. Nó là trần
   *có điều kiện*, không phải mục tiêu với tới được.
6. **Câu của bộ đo do máy sinh.** Cấu trúc hai cảnh dứt khoát hơn đề người viết,
   nên §1 (khoảng cách giữa nhóm một cảnh và hai cảnh) có thể bị thổi phồng —
   nhưng chiều của nó thì khó đảo, vì nó là hệ quả trực tiếp của việc truy vấn
   nén hai cảnh vào một vector, đã được `UNG_VIEN_CANH_B.md` chứng minh riêng.
7. **§2b (bảng phân bố hạng) là phép đếm tất định** và là con số đáng tin nhất
   trong tài liệu này. Nếu chỉ trích một thứ, hãy trích bảng đó.
