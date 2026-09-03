# TRAKE lần đầu có bộ đo — và ba con số đổi cách nhìn nhánh này

Chốt 02/09/2026, **mở rộng n=12 → 24 tối cùng ngày** (§1.1);
**đo lại toàn bộ trên n=24 ngày 03/09, chốt số phận soft_order** (§8).
Sinh: `scripts/sinh_gt_trake.py` · Chấm: `scripts/do_trake_bo_moi.py`
Dữ liệu: `data/gt_trake.json` (dưới `data/` nên `.gitignore` chặn sẵn);
bản 12 mục cũ giữ ở `data/gt_trake_12.json`, 12 mục mới tách riêng ở
`data/gt_trake_test_moi.json` (vai trò TEST cho các giả thuyết chọn trên 12 cũ).

> Không mã video, không đáp án trong tài liệu này.

## 0. Vì sao nhánh này im lặng suốt

TRAKE chiếm **8,5% đề thật** (2/25 câu vòng 2) nhưng có **0 câu trong mọi bộ đo**.
Không bộ đo thì không đo được gì, nên mọi ý tưởng trong `docs/CHAN_DOAN_TRAKE.md`
nằm im từ đầu — không phải vì chúng sai, mà vì không có cách biết.

## 1. Bộ đo được dựng thế nào — và tránh lỗi gì

**12 mục, 3 sự kiện mỗi mục, 12 video khác nhau**, chọn ngẫu nhiên phân tầng theo
dải (không chọn theo điểm SigLIP — nếu không bộ đo chỉ phản ánh chính cái nó dùng
để chấm).

Thiết kế hai bước, tránh đúng lỗi đã phá bộ sinh KIS (nhét nhiều ảnh vào một
request ⇒ model tả đúng nhưng **đánh sai số thứ tự ảnh** ⇒ neo lệch một cú cắt ⇒
bộ đo **sai dấu**). Ở TRAKE lỗi ấy nhân lên N lần, nên:

1. cho xem cả đoạn → hỏi N mô tả sự kiện, **thuần văn bản, không số thứ tự nào**;
2. với mỗi sự kiện → chấm **từng khung, một ảnh một request**, lấy argmax.

**Các cổng loại đã hoạt động:** 5/17 video bị loại — 3 vì không khung nào đạt
ngưỡng (điểm cao nhất 30–40), 1 vì hai sự kiện trỏ vào **cùng một khung** (không
phân biệt được), 1 vì sinh sai số sự kiện.

**Kiểm chứng độ sắc trước khi tin:** 84 lượt chấm mẫu cho trung vị **0**, chỉ 14
khung đạt ≥90, 70 khung dưới 70 — phân bố hai đỉnh, đúng dấu hiệu của một bộ định
vị chứ không phải một bộ chấm dễ dãi. Và đã **mở ảnh kiểm bằng mắt** 4 sự kiện của
2 mục: 3 khớp hoàn hảo, 1 khớp cảnh nhưng mô tả hơi quá chi tiết.

### 1.1 Mở rộng 12 → 24 mục (tối 02/09, provider OpenAI)

Quota Gemini cạn nên cả hai bước chuyển sang **gpt-5.2** (`--provider openai`) —
bước chấm dùng đúng cơ chế một-ảnh-một-request đã kiểm chứng 56/64 trong
`kiem_neo_don_anh.py`. Giữ nguyên seed, `--so 20` và định dạng cache nên **12 mục
cũ replay từ cache, giống hệt từng trường** (assert tự động: 12/12 khớp
`su_kien/frames/n/boi_canh/diem_khop/sinh_tu`, đúng thứ tự).

    python -u scripts/sinh_gt_trake.py --so 20 --muc-tieu 24 --provider openai

Chi phí phiên này: **235 lần gọi gpt-5.2, 96.266 token vào + 12.542 ra ≈ $0,25**
(ngân sách $3). Điểm chấm có ba nguồn, ghi rõ để khỏi đọc nhầm về sau:

| mục | nguồn điểm chấm |
|---|---|
| 12 mục cũ | Gemini (cache phiên sáng 02/09) |
| 7 mục kế (13–19) | Gemini, phiên chiều 02/09 — phiên chết trước khi ghi file, cache còn nguyên |
| 4 mục cuối (21–24) | gpt-5.2, phiên này |
| 1 mục (20) | trộn 22 điểm cache cũ + 20 điểm gpt-5.2 — đã kiểm mắt cả 3 sự kiện |

Thang hai nguồn lệch nhẹ (trung vị `diem_khop`: 12 cũ = 100, 12 mới = 95 —
gpt-5.2 chấm chặt hơn), nhưng argmax chỉ so **trong từng sự kiện** nên lệch thang
giữa các mục không đổi mốc nào; mục duy nhất trộn nguồn trong một sự kiện đã kiểm
bằng mắt.

**Cổng loại vẫn hoạt động:** 6 video bị loại — 4 vì không khung nào đạt ngưỡng 70
(điểm cao nhất 30/35/40/68), 2 vì hai sự kiện trỏ vào cùng một khung.

**Kiểm bằng mắt — toàn bộ 12 mục mới, 36 sự kiện:** 34 khớp rõ (vật thể, chữ trên
bao bì/phông nền, hành động đều đúng — ví dụ đúng túi "Bột mắm cá linh sấy khô",
đúng phông "MOU SIGNING CEREMONY"), 2 khớp cảnh nhưng mô tả chi tiết hơn mức một
khung đơn thể hiện được (pha nước rút đua xe đạp; cận cảnh kéo con lươn). Tỷ lệ
tương đương đợt kiểm 4 sự kiện của bộ 12 cũ.

**Phủ:** 24 video khác nhau, đủ cả 10 dải L21–L30 (1–3 mục/dải), mọi mục đúng 3
sự kiện, khe keyframe quanh mốc trung vị ~67 frame (kho vốn thưa).

## 2. Kết quả — ba mức, ba nguồn mất điểm

Cửa sổ {6,10,20}, 4 họ hạt giống × 48 bốc, mốc thật bốc đều trong khe keyframe
của **từng sự kiện**, chấm bằng `r_score_trake`/`final_score` thật.

| mức | điểm | so nền |
|---|---|---|
| **NỀN** (đường sản xuất) | **0,2154** | — |
| ORACLE-MỐC (video của nền, mốc thật) | 0,4307 | **+100,0%** |
| ORACLE-VIDEO (video đúng + mốc thật) | 0,5350 | **+148,4%** |

**Video ở dòng 1 đúng: 10/12 mục.**

### Phân rã khoảng cách (tổng 0,3196)

| nguồn mất điểm | phần |
|---|---|
| **định vị sự kiện trong video** | **67%** |
| chọn sai video | 33% |

## 3. Ba điều số này nói ra

**(a) Chọn video KHÔNG phải nghẽn chính của TRAKE — khác hẳn trực giác.**
10/12 mục đã có video đúng ở dòng 1. Hai phần ba khoảng cách nằm ở **định vị sự
kiện**. Điều này trùng khớp với kết luận bên nhánh KIS (video tìm đúng ngang nhau,
toàn bộ khoảng cách ở định vị nội-video) — cùng một nghẽn, hai nhánh khác nhau.

**(b) Ngay cả ORACLE-VIDEO cũng chỉ đạt 0,5350, không phải 1,0.** Biết đúng video
*và* đúng cả ba mốc vẫn chỉ được nửa điểm. Lý do nằm ở cấu trúc bài toán: một dòng
phải trúng **đồng thời** cả ba cửa sổ mới được 1,0, mà mốc thật rơi đâu đó trong
khe keyframe (~±27 frame) còn thang bù trừ bước 10 phải phủ ba trục cùng lúc.
**Đây là trần của chính cách phân bổ dòng hiện tại**, không phải của việc nhận
dạng sự kiện — và nó chỉ thẳng vào ý tưởng "lưới bù trừ phi-đều" trong
`CHAN_DOAN_TRAKE.md`: chia ngân sách dòng theo **độ bất định từng sự kiện** thay
vì đều nhau.

**(c) Luật chấm TRAKE rộng lượng hơn KIS nhiều.** Điểm từng phần, không ràng buộc
thứ tự: một câu 3 sự kiện trúng 1 đã được 1/3. Cộng với (b), chiến lược tối ưu cho
TRAKE **khác hẳn** KIS — và giờ mới kiểm chứng được bằng số thay vì suy luận.

## 4. Giới hạn — nói thẳng

*(Các con số §2–§3 đo trên bộ 12 mục ban đầu; phép đo lại trên n=24 ở §8.)*

- **n = 24** (từ 02/09 tối). Đủ để thấy phân rã lớn và trần; chênh lệch vài phần
  trăm giữa các cấu hình vẫn cần đọc thận trọng.
- Cả 24 mục đều có **đúng 3 sự kiện**. Đề thật có câu 4 sự kiện; chưa đo được ảnh
  hưởng của N lớn hơn (mà theo (b), N càng lớn thì trần càng thấp).
- Mốc do máy định vị; đã kiểm mắt **40/72 sự kiện** (4 của bộ cũ + toàn bộ 36 của
  12 mục mới). Sai sót còn lại chưa đo được.
- Điểm chấm đến từ hai model (Gemini / gpt-5.2, §1.1) — lệch thang giữa mục không
  đổi mốc vì argmax nội-sự-kiện, nhưng là một nguồn nhiễu cần nhớ.

---

## 5. Lưới bù trừ: hai giả thuyết, cả hai ÂM — và một suýt-sai của phép đo

`scripts/do_luoi_trake.py`, 0 API.

### 5.1 Giả thuyết cấu trúc: tích Descartes đầy đủ — **SAI**

Điểm một dòng là `(1/N)·Σ_j I(t_j ∈ f_j ± w)` và `R@k` là **max trên tiền tố**.
Nếu tập dòng là **tích đầy đủ** `S_1 × … × S_N` thì trong đó **có sẵn** dòng chọn
offset tốt nhất cho *mọi* trục cùng lúc, nên max **phân rã theo từng trục**. Với
tập không đầy đủ, đẳng thức thành `≤`.

Đếm được: allocator hiện tại sắp theo tổng độ dời — một **quả cầu L1** — phủ
offset {−2..2} cả ba trục nhưng chỉ chứa **100/125** điểm tích, thiếu đúng 25
điểm góc.

**Nhưng tích đầy đủ THUA ở mọi bước** (−1,6% → −2,0%). Lý do: 100 dòng với N=3
chỉ cho tích 5×5×4, tức một trục **hẹp đi** (4 offset thay vì 5). Cái mất do hẹp
lớn hơn cái được do đầy đủ. Giả thuyết đúng về mặt cấu trúc, **sai về mặt số học**.

### 5.2 Bước thang rộng hơn: thắng đậm — rồi hoá ra là **ảo ảnh của phép đo**

Trung bình trên cửa sổ {6,10,20}, `step=20` thắng `step=10` rõ rệt:
**+11,3%** trên bài toán tâm-oracle và **+16,2%** trên đường sản xuất
(0,2154 → 0,2503), bootstrap theo câu KTC [+0,0374, +0,0822], P(≤0) = 0,0%.

Đủ sức thuyết phục để ship. **Nhưng tách theo từng cửa sổ thì đảo dấu:**

| cửa sổ chấm | step=10 | step=20 |
|---|---|---|
| **±6** | **0,4250** | 0,4197 (**−1,2%**) |
| ±10 | 0,5073 | 0,6126 (+20,8%) |
| ±20 | 0,6727 | 0,7546 (+12,2%) |

Quy định của BTC ghi cửa sổ TRAKE *"thường rất ngắn, thông thường là **dưới 10
frame**"* — tức bề rộng tổng dưới 10, xấp xỉ **±5**. Cột gần nhất với thực tế là
**±6**, và ở đó `step=20` **thua**. Toàn bộ mức thắng đến từ hai cột cửa sổ
**rộng hơn quy định**, và chúng chiếm 2/3 trong phép lấy trung bình.

**Kết luận: KHÔNG đổi `--step`. Giữ 10.**

### 5.3 Bài học phương pháp — sửa ngay, không chỉ ghi lại

Bộ cửa sổ {6,10,20} được chọn cho **KIS** (nơi ví dụ của BTC dùng cửa sổ 11
frame). Dùng nguyên bộ đó cho **TRAKE** là sai: quy định nói cửa sổ TRAKE hẹp
hơn hẳn. Lấy trung bình trên một bộ cửa sổ quá rộng đã **suýt** đẩy một thay đổi
âm vào sản xuất, với khoảng tin cậy đẹp và P(≤0) = 0,0%.

Đây là ca thứ ba trong hai ngày mà **giả định của thiết bị đo quyết định dấu của
kết luận** — sau trục sigma (đổi chiều theo mô hình bốc) và bộ so khớp đáp án
(đếm thiếu theo hai kiểu). Quy tắc bổ sung:

> **Mọi phép đo TRAKE phải báo cáo theo TỪNG cửa sổ, và cột quyết định là ±6.**
> Con số trung bình chỉ được ghi kèm để đối chiếu.

---

## 6. Hedge VIDEO cho TRAKE — **ĐÓNG, và đóng bằng một sự thật cấu trúc**

`scripts/do_hedge_video_trake.py`, 0 API.

33% khoảng cách của TRAKE nằm ở chọn sai video, và luật chấm cho **0 tuyệt đối**
khi sai video. Nên câu hỏi "chia bớt dòng cho video hạng 2" nghe như một canh bạc
đáng cân nhắc: mất một ít thang bù trừ ở 10/12 mục đang đúng, để cứu 2/12 mục
đang 0 điểm.

**Nhưng nó không phải canh bạc — nó bất khả thi:**

| | số mục |
|---|---|
| video đúng ở **hạng 1** | **10/12** |
| video đúng trong **top-3** | **10/12** |

Hai con số **bằng nhau**. Bộ căn chỉnh hoặc đặt video đúng ở hạng 1, hoặc không
có nó trong top-3 chút nào. Không có mục nào mà video đúng nằm ở hạng 2–3, nên
mọi dòng chia cho hạng 2–3 là **mất trắng theo định nghĩa**.

Đúng như vậy trong số đo, ở cột quyết định ±6:

| chia dòng | ±6 (quyết định) | ±10 | ±20 |
|---|---|---|---|
| **không hedge (100/0/0)** | **0,1717** | 0,2029 | 0,2715 |
| 85/15/0 | 0,1709 | 0,2025 | 0,2714 |
| 75/25/0 | 0,1707 | 0,2024 | 0,2712 |
| 60/40/0 | 0,1701 | 0,2020 | 0,2708 |
| 70/20/10 | 0,1707 | 0,2023 | 0,2712 |
| 50/30/20 | 0,1692 | 0,2012 | 0,2705 |

Đơn điệu giảm theo mức chia, ở **cả ba cửa sổ**. Không có ô nào dương.

**Hệ quả cho hướng đi:** 33% khoảng cách thuộc khâu chọn video **không lấy lại
được bằng cách chia lại dòng**. Nó đòi một **tín hiệu xếp hạng video khác** —
thứ đưa được video đúng vào top-3 ở 2 mục hiện đang trượt hẳn. Chia lại ngân sách
trong top-3 của chính bộ căn chỉnh hiện tại là vô nghĩa.

**Ghi chú về một lỗi của phép đo, đã sửa:** lần chạy đầu tôi dựng văn bản sự kiện
bằng cách ghép `bối_cảnh + sự_kiện` cho từng sự kiện, và chỉ được **5/12** video
đúng — trong khi đường sản xuất dùng `split_events` trên đề đầy đủ và được
**10/12**. Số của lần ấy (0,0693) không so được với sản xuất. Kết luận về hedge
không đổi, nhưng bài học thì đổi: **phép đo phải dựng đầu vào y hệt sản xuất**,
kể cả ở những chỗ trông như chi tiết vụn.

---

## 7. Chế độ căn chỉnh — giả thuyết "giải bài toán khó hơn cần thiết" là **SAI**

`scripts/do_che_do_can_chinh.py`, 0 API.

### Giả thuyết

Luật chấm TRAKE chấm **từng sự kiện độc lập**, **không** đòi thứ tự, **không**
đòi khoảng cách tối thiểu. Nhưng `align_sequence` chạy `align_mode="ordered"` với
`min_gap=2` — áp hai ràng buộc mà bộ chấm không hề đòi. Nghe như đang tự trói tay:
nếu mô tả bị viết lệch thứ tự, ràng buộc ấy **ép** bỏ mốc tốt nhất của một sự kiện
để giữ tính đơn điệu.

### Kết quả — ràng buộc thứ tự GIÚP, không hại

| chế độ | min_gap | video đúng | **±6 (quyết định)** | ±10 | ±20 |
|---|---|---|---|---|---|
| **ordered** (sản xuất) | **2** | **10/12** | **0,1717** | 0,2029 | 0,2715 |
| ordered | 0 / 1 | 9/12 | 0,1363 | 0,1609 | 0,2084 |
| **unordered** | bất kỳ | 9/12 | **0,1385** | 0,1613 | 0,2082 |
| soft_order | 0 / 1 | 9/12 | 0,1427 | 0,1689 | 0,2195 |
| soft_order | 2 | 10/12 | **0,1781** | 0,2110 | 0,2826 |

**`unordered` THUA rõ** (0,1385 vs 0,1717 = −19,3%). Giả thuyết sai: thứ tự
không phải ràng buộc nhân tạo mà là **thông tin tiên nghiệm thật** — sự kiện được
kể theo trình tự thì thường cũng xảy ra theo trình tự, và bộ căn chỉnh dùng đúng
quy luật đó để loại các phương án vô lý. Bỏ nó đi là vứt thông tin.

**`min_gap` cũng quan trọng hơn vẻ ngoài:** gap 0 hoặc 1 làm tụt video đúng từ
10/12 xuống 9/12 và mất ~20% điểm. Ràng buộc "hai mốc phải cách nhau ≥2 keyframe"
chặn nghiệm suy biến kiểu ba sự kiện dồn vào một chỗ.

### Manh mối nhỏ: `soft_order`

`soft_order/gap=2` cho **+3,7%** so sản xuất ở cột ±6, và dương ở **cả ba** cửa sổ.
Bootstrap theo câu: chênh +0,0063, KTC [+0,0000, +0,0190], P(≤0) = 35,2%.

Đọc con số này cho đúng: KTC **chạm 0 ở cận dưới** và P(≤0) lớn — nhưng đó là vì
với n=12, phần lớn mẫu bootstrap cho chênh **đúng bằng 0** (chỉ 1–2 mục thật sự
đổi). Hiệu ứng **không bao giờ âm** trong mẫu này; nó chỉ bằng 0 hoặc dương.

**Chưa ship.** Đây là manh mối đúng hướng với hồ sơ rủi ro tốt (không hại ai),
nhưng n=12 quá nhỏ để chốt. Việc cần làm trước: sinh thêm mục TRAKE cho đủ 20+,
rồi đo lại. Nếu giữ được dấu, đây là một cờ đổi mặc định rẻ (`align_mode`).

**Cập nhật 03/09 — đã đo trên TEST: hiệu ứng đúng bằng 0, KHÔNG ship (§8.2).**

---

## 8. Đo lại trên n=24 (03/09) — soft_order: KHÔNG SHIP; các cửa khác giữ nguyên kết luận

Ba script chạy lại nguyên trạng trên bộ 24 mục, qua wrapper tiết kiệm RAM
`scripts/chay_gon_ram.py` (máy chỉ còn ~3GB trống vì lane pe-core đang encode;
wrapper chỉ đổi *cách nạp* trọng số — fp32 giống hệt từng bit — và bỏ tháp thị
giác không dùng đến, nên không đổi con số nào):

    python -u scripts/chay_gon_ram.py scripts/do_trake_bo_moi.py --gt data/gt_trake.json
    python -u scripts/chay_gon_ram.py scripts/do_soft_order_test.py --gt data/gt_trake_test_moi.json
    python -u scripts/chay_gon_ram.py scripts/do_che_do_can_chinh.py --gt data/gt_trake.json
    python -u scripts/chay_gon_ram.py scripts/do_hedge_video_trake.py --gt data/gt_trake.json

### 8.1 Ba mức đo — bức tranh n=12 giữ nguyên, tin cậy hơn

| mức | n=24 | so nền | (n=12 cũ) |
|---|---|---|---|
| **NỀN** (đường sản xuất) | **0,2275** | — | 0,2154 |
| ORACLE-MỐC (video của nền, mốc thật) | 0,4727 | **+107,8%** | +100,0% |
| ORACLE-VIDEO (video đúng + mốc thật) | 0,5645 | **+148,1%** | +148,4% |

**Video ở dòng 1 đúng: 21/24** (12 cũ: 10/12 · 12 mới: 11/12). Phân rã khoảng
cách: **định vị sự kiện 72,8%** / chọn sai video 27,2% (n=12: 67/33). Kết luận
§3 không đổi chữ nào: nghẽn chính là định vị sự kiện trong video, không phải
chọn video.

### 8.2 soft_order trên TEST — hiệu ứng đúng bằng 0, cửa ĐÓNG

Kỷ luật tiền-đăng-ký: giả thuyết `soft_order/gap=2` chốt trên TUNE (12 mục cũ,
§7), TEST là 12 mục mới (`data/gt_trake_test_moi.json`), đọc **đúng một lần**
qua `scripts/do_soft_order_test.py` — chỉ so đúng cặp đã đăng ký, không quét.

| chế độ | video đúng | **±6 (quyết định)** | ±10 | ±20 |
|---|---|---|---|---|
| ordered/gap=2 (sản xuất) | **11/12** | 0,1863 | 0,2179 | 0,2923 |
| soft_order/gap=2 | 10/12 | 0,1863 | 0,2179 | 0,2923 |

Chênh **+0,0000 ở cả ba cửa sổ; 0/12 mục đổi điểm**. Tệ hơn: soft_order lật
video của một mục từ ĐÚNG thành SAI — mất 0 điểm chỉ vì mục đó vốn 0 điểm ở
mọi cửa sổ, tức một rủi ro thật đang được che bởi may mắn.

Trên n=24 gộp (quét đầy đủ, chỉ để đối chiếu): soft_order/gap=2 còn +1,7% ở ±6
(chênh +0,0032, KTC [+0,0000, +0,0095], P(≤0)=36,5%) — **toàn bộ phần dương là
dư ảnh của chính các mục TUNE đã dùng để chọn giả thuyết**. Hiệu ứng co từ
+0,0063 (TUNE) về +0,0032 (gộp) về **0,0000 (TEST)** — đúng chữ ký của thổi
phồng (luật đo lường #7).

> **KẾT LUẬN: KHÔNG ship soft_order. Giữ `align_mode=ordered`, `min_gap=2`.
> Cửa này ĐÓNG trên bộ đo mới, có TEST đứng sau.**

### 8.3 Các cửa khác — xác nhận lại trên n=24, không cửa nào mở lại

Từ bảng quét `do_che_do_can_chinh.py` (±6, n=24, nền ordered/gap=2 = 0,1823):

- **unordered vẫn THUA rõ**: 0,1372 (−24,7%), mọi gap. Thứ tự kể chuyện vẫn là
  thông tin tiên nghiệm thật (§7 giữ nguyên).
- **min_gap=2 vẫn đúng**: ordered/gap 0–1 = 0,1646 (−9,7%), video đúng tụt
  21→20. Ràng buộc chống nghiệm suy biến vẫn cần.
- **Hedge video vẫn bất khả thi về cấu trúc** (`do_hedge_video_trake.py`):
  video đúng ở hạng 1 = trong top-3 = **21/24** — sự thật "hoặc hạng 1 hoặc
  ngoài top-3" tái lập nguyên vẹn trên gấp đôi dữ liệu. Mọi mức chia đơn điệu
  giảm ở cả ba cửa sổ (không hedge 0,1823 > 85/15 0,1817 > … > 50/30/20
  0,1800). Cửa ĐÓNG, lần hai.

### 8.4 Hướng còn mở sau §8

1. **Định vị sự kiện trong video** — 72,8% khoảng cách, chưa có tín hiệu nào
   ăn được (soft_order vừa rụng). Cùng nghẽn với KIS tầng 2; encoder thứ hai
   (lane pe-core) là hướng chưa thử duy nhất còn lại.
2. **Phân bổ dòng quanh mốc** — trần ORACLE-VIDEO vẫn chỉ 0,5645: lưới bù trừ
   phi-đều theo độ bất định từng sự kiện (§3b) vẫn chưa ai đo.
3. **3/24 mục sai video** — cần tín hiệu xếp hạng video khác; chia lại dòng
   trong top-3 hiện tại đã chứng minh vô nghĩa hai lần.
