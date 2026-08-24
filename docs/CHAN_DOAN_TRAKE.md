# Vì sao TRAKE yếu, và trần lý thuyết của nó

Tài liệu này trả lời một câu hỏi cụ thể: người soát xem video múa lân
(`query-p1-16-trake`) và thấy mô tả của ban tổ chức **không khớp** với các frame
hệ thống chọn. Dưới đây là nguyên nhân, đo được, không phỏng đoán.

---

## 1. Trần cứng: keyframe thưa hơn cửa sổ đáp án 9 lần

| | |
|---|---|
| Cửa sổ đáp án mỗi sự kiện | **dưới 10 frame** (luật) = 0,4 giây ở 25 fps |
| Khoảng cách keyframe, trung vị toàn kho | **55 frame** = 2,2 giây |
| Riêng video múa lân `L24_V009` | **92 frame** = 3,7 giây |

Một keyframe đơn lẻ rơi trúng cửa sổ 10 frame chỉ **13,3%** số lần. Thang
±10/±20 nâng lên tối đa **54,9%** *cho mỗi sự kiện*, và chỉ khi keyframe neo đã
đúng.

Với 4 sự kiện, lưới bù trừ cần `(2·2+1)^4 = 625` tổ hợp nhưng chỉ có **100
dòng** — phủ được **16%**. Mô phỏng, giả định mô hình **hoàn hảo** chọn đúng
keyframe gần nhất:

| khoảng cách keyframe | điểm TRAKE kỳ vọng | trúng cả 4 sự kiện |
|---|---|---|
| 55 frame (trung vị kho) | 0,656 | 12,2% |
| **92 frame (video múa lân)** | **0,452** | **1,6%** |
| 150 frame | 0,300 | 0,2% |

Nghĩa là **ngay cả khi thuật toán không sai gì cả**, câu múa lân vẫn chỉ đạt
khoảng 0,45. Đây không phải lỗi mô hình.

Tái lập: mục 1–2 của phép đo trong `scripts/` (xem git log của tài liệu này).

---

## 2. Trần mềm: 8/12 sự kiện KHÔNG có tín hiệu nào

Với mỗi sự kiện, đo **độ nhọn** của điểm số trong video đã chọn:
`(đỉnh − trung bình) / độ lệch chuẩn`. So với **ngưỡng nhiễu** — độ nhọn mà một
ma trận điểm hoàn toàn ngẫu nhiên cùng kích thước cũng đạt được.

| câu | video | #keyframe | ngưỡng nhiễu | E1 | E2 | E3 | E4 |
|---|---|---|---|---|---|---|---|
| p1-16 múa lân | L24_V009 | 156 | 2,62 | **1,68** | 2,64 | **1,42** | **2,05** |
| p1-4 măng tây | L26_V208 | 189 | 2,67 | **2,66** | **2,60** | 2,71 | **2,62** |
| p1-18 nấm | L26_V198 | 164 | 2,65 | 3,00 | 3,58 | 3,32 | **2,23** |

**In đậm = dưới ngưỡng nhiễu.** Với những sự kiện đó, đỉnh điểm của mô hình
không nổi bật hơn một đỉnh ngẫu nhiên — mô hình đang chọn keyframe gần như
ngẫu nhiên, và bất kỳ tiêu chí phụ nào (ưu tiên sớm, phạt khoảng cách) cũng sẽ
quyết định kết quả thay cho nó.

**Vì sao:** SigLIP-2 là mô hình **ảnh–văn bản tĩnh**. Nó không có biểu diễn cho:

- **chuyển động**: "lân *bắt đầu xoay vòng*", "miếng măng tây *rời khỏi* chảo"
- **chuyển trạng thái**: "4 chân *hoàn toàn chạm đất*"
- **thứ tự**: "khoảnh khắc **đầu tiên**" — mọi frame đều là "một khoảnh khắc"
- **đếm**: "*4* chân", "cột số *4*" (xem Paiss et al., *Teaching CLIP to Count
  to Ten*, ICCV 2023 — CLIP không đếm được)

Mọi keyframe của một video múa lân đều trông như "một con lân đang múa".

Ngược lại, `p1-18` có 3/4 sự kiện **trên** ngưỡng — vì các sự kiện của nó là
**đối tượng** ("cắt nấm", "cắt củ năng", "cắt đậu hủ"), và video còn có **chữ
cháy sẵn** trên hình ghi tên nguyên liệu. Đó chính là loại tín hiệu SigLIP thấy được.

---

## 3. Chọn sai video: 6 ứng viên gần như đồng hạng

`p1-16`, 6 video hàng đầu, chênh nhau tối đa 0,58 điểm:

```
1. L24_V009  16.63  Kim Lân Thượng Sơn Hái Linh Chi – Đoàn Lân Huỳnh Kim Lân
2. L24_V038  16.57  Khải Sự Phi Đằng triển Hùng Uy – Đoàn Lân Khải Uy
3. L24_V013  16.13  Hồng Sư Vượt Trường Sơn Hái Linh Chi – Đoàn Lân Minh Hào
...
```

Cả 6 đều là màn múa lân cùng một giải (Cúp Chợ Lớn – HTV 2024). Câu hỏi nói rõ
"**một con lân màu vàng đen trắng**" — đó là dấu hiệu phân biệt, nhưng chênh
lệch hạng 1 với hạng 2 chỉ **0,07 điểm (0,4%)**. Hệ thống đang tung đồng xu.

Với TRAKE, **sai video là 0 điểm ngay** — không có điểm thành phần.

---

## 4. Tên video mang thông tin mà hình ảnh không có

Đối chiếu tay trên vòng 1 tìm ra điều hình ảnh không bao giờ thấy:

- Trong **toàn bộ 873 video**, đúng **2 video** nhắc tới "củ năng" — một trong đó
  là `L26_V012` **"CỦ NĂNG OM NẤM CHAY"**, khớp chính xác `p1-18`
  ("món ăn về nấm", cắt nấm, cắt **củ năng**, cắt đậu hủ). Nó **không nằm trong
  top-6** của hệ thống.
- Đúng **1 video** khớp "măng tây" + "chiên": `L26_V194` **"MĂNG TÂY CHIÊN BIA
  XỐT CÁ NGỪ"**. `p1-4` mô tả *tẩm bột rồi chiên ngập dầu*; ta đang nộp
  `L26_V208` "TIM HEO **XÀO** MĂNG TÂY" — món xào, không tẩm bột.

Lưu ý quan trọng: metadata cấp video **đã đo và làm điểm KIS tệ đi**
(R@1 43,3% → 40,0%), nên nó **không** được đưa vào bộ chấm điểm. Nó được **hiển
thị cho người soát** trên `review.html`, cạnh mỗi ứng viên, với các từ trùng
câu hỏi được đánh dấu. Người quyết định, không phải máy.

---

## 5. Ưu tiên "khoảnh khắc đầu tiên" đang làm hỏng tính mạch lạc thời gian

Với `p1-16`, bật/tắt `first_occurrence`:

```
mu = 2.0 :  E1= 9.9s   E2=79.9s   E3=109.1s   E4=281.0s
mu = 0   :  E1=69.4s   E2=79.9s   E3=109.1s   E4=281.0s
```

Câu hỏi nói lân **xoay vòng rồi tiếp đất** — hai việc cách nhau vài giây.
Với `mu=2.0`, E1 và E2 cách nhau **70 giây**: bất khả về mặt vật lý.

Cơ chế hiện tại là một **dốc tuyến tính trải khắp video** trừ vào điểm sự kiện 1,
biên độ bằng **0,4–0,5 lần toàn bộ biên độ điểm** trong video — không phải tiêu
chí phân định, mà là lực áp đảo. Nó thắng cả hình phạt khoảng cách vốn đang giữ
chuỗi mạch lạc.

**Nhưng chưa sửa, vì hai phép đo mâu thuẫn.** Trên benchmark tổng hợp
(`src/task3_trake/bench`, có cài mồi nhử đúng cho tình huống này), `mu` càng lớn
càng tốt: 76% ở `mu=0` lên 92% ở `mu=3`. Trên dữ liệu thật nó phá chuỗi.

Lý do của mâu thuẫn nằm ở mục 2: benchmark cài **đỉnh điểm mạnh**, nên tiêu chí
phụ chỉ phân định giữa hai đỉnh thật. Dữ liệu thật **phẳng, dưới ngưỡng nhiễu**,
nên tiêu chí phụ **tự nó quyết định đáp án**. Một tiêu chí phụ chạy trên nhiễu
thì thu nhỏ nó cũng không cứu được gì — chỉ đổi nhiễu này lấy nhiễu khác.

Việc cần làm không phải chỉnh `mu`, mà là **cấp cho hệ thống tín hiệu thật**
(mục 6) hoặc **để người chốt frame** (`review.html`, nút ▶).

---

## 6. Việc nên làm, theo thứ tự

1. **Người chốt frame trên video gốc.** Đã dựng: `review.html` nhúng trình phát
   YouTube, `Space` dừng, `←`/`→` đi từng frame, `C` lấy đúng thời điểm, chọn vị
   trí rồi `Enter`. Đây đúng là cách hai đội AIC 2025 đạt hạng Outstanding về
   TRAKE làm (MERVIN arXiv:2605.16120 §3.3; U-CESE arXiv:2605.23274 §4.3).
   Chuyển tỷ lệ trúng mỗi sự kiện từ 33–55% lên gần 100% cho mọi sự kiện mắt
   người thấy được. Dòng do người chốt đặt ở hàng 1, các dòng tự động giữ nguyên
   phía sau — R@k lấy max nên không bao giờ lỗ.
2. **Kênh văn bản** (`scripts/fetch_captions.py`). Phụ đề tự động tiếng Việt,
   miễn phí, không cần API. Xem `docs/WHAT_CHANGED.md`.
3. **Không** mua mô hình temporal grounding chuyên dụng: CoMET-Bench
   (arXiv:2606.15320) cho thấy chúng thua trên bài toán nhiều sự kiện có điều
   kiện, và benchmark temporal grounding **giải được ~92% chỉ bằng tiên nghiệm**
   (Otani et al., BMVC 2020) nên con số công bố không chuyển giao.
