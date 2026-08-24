# Đọc được nội dung trong ảnh: chữ, màu, và lời nói

Ba kênh thông tin mà một mô hình nhúng ảnh–văn bản **không** biểu diễn được, và
cách dự án dùng chúng.

Nguyên tắc chung, rút ra từ ba lần đo trước đó: **mọi tín hiệu đưa vào bảng xếp
hạng dựa trên "nghe có lý" thay vì đo đạc đều làm mất điểm.** Metadata cấp video
(R@1 43,3% → 40,0%), điểm cộng đối tượng theo frame (video R@1 tăng nhưng điểm
thi giảm), lời thoại theo video (−0,4%), lời thoại theo mốc thời gian (−1,5%).
Nên ba kênh dưới đây đi tới **mắt người soát**, không tới bộ chấm điểm — trừ
đúng một trường hợp đã đo dương (đối tượng ở tầng video, +3,3%).

---

## 1. OCR — chữ cháy trên khung hình

`src/core/ocr.py`, `scripts/run_ocr.py`, `scripts/search_ocr.py`

EasyOCR (`vi`+`en`), CPU, ~4 giây mỗi khung hình 1280×720, có cache theo
`(video, frame)` nên chạy lần hai là tức thì. Một vòng thi (top-24 mỗi câu ≈ 560
khung) mất khoảng **25 phút chạy nền** trước giờ soát.

**48% khung hình có chữ đáng kể** — chủ yếu là dòng tiêu đề tin tức cháy sẵn:

```
L21_V008 f21240: NASA HOÃN CHUYẾN BAY GIẢI CỨU PHI HÀNH GIA KẸT TRÊN ISS
L21_V009 f19989: TP HCM: Học sinh tựu trường sớm nhất vào ngày 19/8/2024
L21_V018 f19230: Thư ký Liên quốc kêu đình chiến tại Gaza để tiêm vaccine
```

**Nó bắt được lỗi không công cụ nào khác bắt được.** OCR trên khung hình đang
nộp cho `query-p1-19` đọc ra *"Trích Văn bia THOẠI NGỌC HẦU"* và một đoạn về
**núi** — trong khi câu hỏi về **Nguyễn Trung Trực**. Bài nộp đang ở sai video,
và không một điểm tương đồng nào nói ra được điều đó.

Ba câu Q&A của vòng 1 về bản chất là bài toán đọc chữ: tên xã trên băng rôn, hai
câu thơ trên trang sách, tên món trên tờ công thức.

---

## 2. Màu sắc — đo trên ĐỐI TƯỢNG, không phải cả khung hình

`src/core/colours.py`

Câu múa lân nói rõ *"một con lân màu **vàng** đen trắng"*, nhưng danh sách gợi ý
trả về lân **đỏ**. Nửa số câu vòng 1 có nêu màu ("áo sơ mi hồng", "xe màu đỏ
mận", "áo xanh dương").

Điểm mấu chốt: **đo màu trên hộp bao của đối tượng đã nhận dạng**, không phải cả
ảnh. Một khung hình múa lân phần lớn là sân khấu đỏ và băng rôn vàng dù con lân
màu gì — biểu đồ màu toàn ảnh nói "đỏ" cho mọi ứng viên và không phân định được
gì. Dữ liệu BTC có sẵn `detection_boxes`, nên việc cắt vùng là miễn phí.

Có test chứng minh đúng điều đó: một khung hình sân khấu đỏ với con lân vàng nhỏ
cho ra "đỏ" nếu đo cả ảnh, và "vàng" nếu đo trên hộp bao.

Bỏ qua các lớp nền (`Tree`, `Clothing`, `Building`…) và các hộp phủ gần hết
khung hình — đó là nền đeo nhãn, không phải chủ thể.

Lưu ý về tiếng Việt: "xanh" một mình phủ cả xanh lá lẫn xanh dương nên không
phân định được; chỉ "xanh dương" / "xanh lá" mới được tính.

---

## 3. Lời nói — đo hai lần, âm cả hai lần, nên chỉ để cho người

`src/core/transcripts.py`, `scripts/search_transcripts.py`

811/873 video có lời thoại (809 do nhóm cung cấp + 217 phụ đề tự lấy, có trùng).
Thiếu 62, tập trung ở **L24 (34 — nhóm múa lân)** và **L28 (24)**.

| cách đưa vào điểm | kết quả |
|---|---|
| cộng theo video, mọi trọng số | −0,1% đến −23% |
| có cổng chặn theo độ quyết đoán | +0,5% (nhiễu) |
| **cộng theo mốc thời gian từng frame** | **−1,5% đến −20%** |

Cách thứ ba đáng chú ý vì nó trả lời một câu hỏi khác: lời thoại có **mốc thời
gian**, nên về nguyên tắc nó định vị được *frame*, không chỉ *video* — thứ mà
điểm cộng cấp video không làm được. Vẫn âm.

Lý do lộ ra khi nhìn dữ liệu: **60 câu ground truth đều là mô tả cảnh nhìn
thấy** ("xe ô tô con màu đỏ mận có cánh gió đuôi xe"). Không ai *nói* ra những
câu đó, nên bằng chứng lời thoại tản mát (chênh lệch hạng 1–hạng 2 chỉ 0–18%).
Câu thật của vòng 1 thì khác: `p1-19` chênh 21%, `p1-18` chênh 33%.

Phép đo **trung thực về loại câu nó bao phủ và im lặng về loại câu nó không bao
phủ**. Không đủ cơ sở để đưa vào bộ chấm điểm.

**Nhưng nó tìm đúng những video hình ảnh bỏ sót** — đã kiểm tay trên vòng 1:

| câu | hệ thống hình ảnh | lời thoại tìm ra |
|---|---|---|
| `p1-4` măng tây tẩm bột chiên | `L26_V208` "TIM HEO **XÀO**" | **`L26_V194` "MĂNG TÂY CHIÊN BIA"** hạng 1, đoạn nói *"măng tây xanh… lại bột mì lên"* |
| `p1-18` cắt nấm/củ năng/đậu hủ | `L26_V198` "MÍT NON KHO NẤM" | `L26_V012` "**CỦ NĂNG** OM NẤM CHAY" — chỉ 2/873 video nhắc "củ năng" |
| `p1-22` phụ nữ dạy nấu ăn | `L26_V205` "BÒ NƯỚNG LÁ CÁCH" (chương trình TV) | `L30_V078` "**Lớp học** 0 đồng cho người yêu bếp" |

---

## Quy trình dùng trong ngày thi

```bash
python scripts/make_submission.py --queries round1/q --out round1/a   # 90 giây
python scripts/run_ocr.py --queries round1/q --top 24                 # ~25 phút, chạy nền
python scripts/build_review_page.py --queries round1/q --run-out round1/a
```

Trong `review.html` mỗi khung hình giờ có ba dòng:

* 🔤 **chữ đọc được**, từ khoá câu hỏi được tô đậm
* 🎨 **màu chủ thể**, màu câu hỏi yêu cầu tô xanh, và cảnh báo `thiếu …` khi
  khung hình không có màu nào câu hỏi nêu
* đối tượng nhận dạng được (`Woman×2, Boy×2`)

Nghi hệ thống bỏ sót video thì tìm bằng hai kênh còn lại:

```bash
python scripts/search_transcripts.py "măng tây chiên bột"   # theo lời nói
python scripts/search_ocr.py "Nguyễn Trung Trực"            # theo chữ trên hình
```

Cả hai trả về video + mốc thời gian + số frame + link YouTube mở đúng lúc đó.
