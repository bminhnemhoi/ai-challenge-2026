# TRAKE lần đầu có bộ đo — và ba con số đổi cách nhìn nhánh này

Chốt 02/09/2026. Sinh: `scripts/sinh_gt_trake.py` · Chấm: `scripts/do_trake_bo_moi.py`
Dữ liệu: `data/gt_trake.json` (dưới `data/` nên `.gitignore` chặn sẵn).

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

- **n = 12** là rất nhỏ. Đủ để thấy phân rã 67/33 và trần 0,535; **không** đủ để
  phân xử chênh lệch vài phần trăm giữa các cấu hình phân bổ.
- Cả 12 mục đều có **đúng 3 sự kiện**. Đề thật có câu 4 sự kiện; chưa đo được ảnh
  hưởng của N lớn hơn (mà theo (b), N càng lớn thì trần càng thấp).
- Mốc do máy định vị, đã kiểm mắt 4/36 sự kiện. Sai sót còn lại chưa đo được.
- Bộ sinh dừng ở 12/20 vì **hết quota cả 5 model**. Sinh tiếp khi quota hồi.

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
