# Những câu đang nộp SAI, và bằng chứng

Tìm ra bằng ba kênh mới — lời thoại, OCR, và mô hình thị giác — trên bài nộp
7,8 điểm. Mỗi mục dưới đây có bằng chứng kiểm được, không phải phỏng đoán.

---

## `query-p1-19-qa` — SAI VIDEO. Đã tìm ra video và cả hai câu thơ.

**Đề:** *"Trong đoạn video có 2 câu thơ của một nhà thơ ca ngợi anh hùng Nguyễn
Trung Trực trong đình thần Nguyễn Trung Trực tại Kiên Giang. Hai câu thơ đó là gì?"*

**Đang nộp:** `L28_V012` "Tản Mạn Mê Kông TẬP 12", frame 20790.

**Bằng chứng nó sai:** OCR đọc khung hình đó ra
*"Trích Văn bia **THOẠI NGỌC HẦU**"* và một đoạn về **núi**. Không liên quan
Nguyễn Trung Trực. Không một điểm tương đồng hình ảnh nào nói ra được điều này.

**Video đúng:** `L27_V010` *"Việt Nam đi là ghiền Mùa 3 — Tập 7 **Rạch Giá**"*.
Lời thoại ở 1:56 nói thẳng: *"…đây là **Đình thần Nguy**[ễn Trung Trực]"*.
BM25 74,2 — bỏ xa mọi video khác.

**Frame đúng:** `3275` (giây 131). Mô hình thị giác đọc được
*"Anh Hùng Dân Tộc NGUYỄN TRUNG TRỰC (1838 - 1868)"* và hai câu thơ, tin cậy 95%:

> **Hỏa hồng Nhựt Tảo oanh thiên địa / Kiếm bạc Kiên Giang khấp quỷ thần**

(Trên bia đền viết "Nhựt Tảo" và "Kiếm bạc"; bản phổ biến là "Nhật Tảo",
"Kiếm bạt". Nên dùng đúng chữ trên bia.)

```
query-p1-19-qa=L27_V010:3275:Hỏa hồng Nhựt Tảo oanh thiên địa / Kiếm bạc Kiên Giang khấp quỷ thần
```

---

## `query-p1-22-qa` — SAI VIDEO. Đáp án bạn gõ đúng, chỗ nộp thì sai.

**Đề:** *"…một người phụ nữ dạy nấu ăn cho những người khác… một người đang cầm
công thức món ăn với nguyên liệu chính là 200g thịt nạc xay. Hỏi tiêu đề của
công thức nấu ăn (tên món ăn) này là gì?"*

**Đang nộp:** `L26_V205` "BÒ NƯỚNG LÁ CÁCH — MÓN NGON MỖI NGÀY" — một chương
trình truyền hình, không phải lớp học.

**Bằng chứng nó sai:** mô hình thị giác nhìn 12 khung hình đang nộp và trả lời
thẳng: *"không có tên món ăn với nguyên liệu chính là 200g thịt nạc xay trên các
khung hình"*.

**Video đúng:** `L30_V078` *"Lan tỏa năng lượng tích cực 2024 — **Lớp học** 0
đồng cho người yêu bếp"*. Lời thoại ở 0:15: *"…lớp học làm bánh miễn phí của
**cô Nguyễn Thị Hồng Hạnh**…"* — đúng "một người phụ nữ dạy nấu ăn cho người khác".

**Frame đúng:** `1623`. Mô hình đọc tờ công thức: **"Nhân bánh cuốn"**, tin cậy 100%.

```
query-p1-22-qa=L30_V078:1623:Nhân bánh cuốn
```

---

## `query-p1-15-qa` — ĐÚNG. Đã xác nhận.

`L30_V072` frame 1745. Mô hình đọc được nguyên văn trên băng rôn:
*"**Xã Giang Ly**, huyện Khánh Vĩnh, tỉnh Khánh Hòa"*, tin cậy 100%.
Đáp án đang nộp khớp.

---

## `query-p1-16-trake` (múa lân) — ĐÚNG VIDEO. Đã xác nhận bằng màu.

Đề nêu *"một con lân màu **vàng** đen trắng"*. Đưa 6 khung hình đầu của 6 chuỗi
ứng viên cho mô hình thị giác:

| video | mô hình đọc màu con lân | điểm khớp |
|---|---|---|
| **`L24_V009`** (đang nộp) | **vàng, đen, trắng** | **100** |
| `L24_V038` | trắng, đen | 20 |
| `L24_V013` | đỏ | 0 |
| `L24_V014` | trắng, đen | 20 |
| `L24_V042` | trắng, đỏ, đen | 30 |
| `L24_V011` | trắng, xanh | 20 |

Đây chính là câu hỏi *"sao mô tả lân vàng mà top lại có lân đỏ"* — và câu trả lời
là chọn video thì đúng, nhưng danh sách gợi ý trộn lẫn vì hình ảnh không phân
biệt được màu. Mô hình thị giác phân biệt được, mất 4,5 giây và $0,0008.

Vấn đề còn lại của câu này là **frame**, không phải video: 3/4 sự kiện nằm dưới
ngưỡng nhiễu (xem `docs/CHAN_DOAN_TRAKE.md`), và video này không có lời thoại
(nhóm L24 thiếu 34/43 transcript). Cách duy nhất là xem video và chốt tay.

---

## `query-p1-18-trake` — nhiều khả năng SAI VIDEO.

**Đang nộp:** `L26_V198` "MÍT NON KHO NẤM".

**Ứng viên mạnh hơn:** `L26_V072` *"Soup Gấc Nấm Mỡ"* — được **cả hai kênh** xác nhận:

* lời thoại ở 1:48: *"…thêm phần **củ năng** á Mình cũng **cắt** sợi…"*
* mô hình thị giác: *"người đầu bếp đang thực hiện **cắt nấm, cắt củ năng, cắt
  đậu hủ** trên thớt"* — 100 điểm

Trong **toàn bộ 873 video chỉ có 2 video** nhắc tới "củ năng" trong tiêu đề/mô tả,
nên đây là dấu hiệu phân biệt mạnh. Cần xem video để chốt bốn frame sự kiện.

---

## `query-p1-4-trake` — chưa kết luận được.

Tiêu đề và lời thoại chỉ về `L26_V194` "MĂNG TÂY CHIÊN BIA" (đề mô tả *tẩm bột
chiên ngập dầu*, còn video đang nộp `L26_V208` là "TIM HEO **XÀO**"). Nhưng khi
lấy 8 khung hình rải đều cả video đưa cho mô hình thị giác thì nó chấm 0 cho cả
hai — lấy mẫu quá thưa, trượt mất đoạn chiên. Cần xem video mới kết luận được.
