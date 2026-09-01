# Tín hiệu định vị nội-video — mở lại bốn cánh cửa, và cái bẫy đứng sau chúng

Chốt 01/09/2026. Script: `scripts/do_tin_hieu_noi_video.py`.
Bộ đo: `data/ground_truth_moi.json`, bộ **sạch 132 mục** (66 câu hai cảnh /
66 câu một cảnh) — xem `docs/BO_DO_KHOP_PHAN_BO.md`.

> Không mã video, không đáp án trong tài liệu này.

---

## 0. Kết luận một dòng

**ÂM — cả bốn tín hiệu đều không mua được điểm, và phát hiện đáng giá nhất
không nằm ở tín hiệu nào cả.**

Hai họ **đối chứng** — chỉ vặn tham số bộ phân bổ, **không đụng một chút nào vào
truy xuất**, không đổi một thứ tự ứng viên nào — **thắng đậm hơn mọi tín hiệu
thật** trên TUNE (+157,6% ở nhóm hai cảnh, so với +68,2% của tín hiệu tốt nhất).
Rồi cả hai cũng bay hơi trên TEST.

Nghĩa là: trên bộ đo này, ở cỡ mẫu này, **điểm số bị chi phối bởi việc vặn hai
núm của bộ phân bổ chứ không phải bởi thông tin định vị nội-video**. Bất kỳ "tín
hiệu" nào đo trên đây mà **không kèm hai đối chứng này** sẽ được ghi công nhầm —
và lần này một tín hiệu suýt được ghi công thật, với +68,2%.

(Bản thân hai núm ấy cũng đã được một lane song song quét đủ lưới và kết luận ÂM
— `docs/PHAN_BO_TREN_BO_MOI.md`. Không có việc nào để làm ở đó; xem §9.4.)

Một kết quả phụ nhưng chắc chắn: **ưu tiên đỉnh cục bộ (`PEAK_WEIGHT`) hiện TRƠ
TUYỆT ĐỐI** với 100 dòng nộp — xem §6.

---

## 1. Vì sao phải mở lại

`docs/KIEN_TRUC_VA_HUONG_CAI_THIEN.md` đóng cửa cả họ này: làm mượt theo thời
gian −0,023, chuẩn hoá theo video −0,129, ép khoảng cách frame −1,3%, pha
CLIP-B32 "không thêm thông tin". **Tất cả đo trên bộ 60 câu cũ** — bộ mà SigLIP
đã đặt keyframe gần đáp án ở **hạng nội-video trung vị 1,0 (hạng-1 60%)**. Không
còn headroom thì mọi tín hiệu định vị nội-video **buộc phải** đo ra ~0. Cánh cửa
bị đóng bằng một cái thước không nhìn thấy vấn đề.

Trên bộ khớp phân bố thì khác hẳn — oracle định vị nội-video
(`scripts/tran_dinh_vi_noi_video.py`) cho **+126%**, riêng câu hai cảnh **+306%**.
Headroom là có thật. Đây là lý do đo lại.

Nền đo lại được ở đây khớp với các con số đã ghi, nên hai phép đo nói cùng một
thứ: cả 132 mục **0,1907** (đã ghi 0,1903); một cảnh **0,2760** (0,2767); hai
cảnh **0,1054** (0,1036).

---

## 2. Đo cái gì, và đo thế nào

Bốn họ tín hiệu, **mỗi họ một cửa riêng, không gộp** (điểm mới thay điểm SigLIP
gốc, rồi đi tiếp đúng đường sản xuất):

| | tín hiệu | lưới quét |
|---|---|---|
| (a) | làm mượt Gauss theo trục thời gian **trong cùng video** | σ ∈ {1, 2, 3, 5} keyframe × trọng số trộn w ∈ {0,25; 0,5; 1,0} |
| (b) | chuẩn hoá điểm theo video: trừ trung bình / z-score / min-max | λ ∈ {0,25; 0,5; 1,0} |
| (c) | **quét lại** ưu tiên đỉnh cục bộ đang chạy trong sản xuất | w ∈ {0; 0,002; 0,005; **0,01**; 0,02; 0,05; 0,1} |
| (d) | **biên cảnh** — keyframe cao hơn hẳn k keyframe TRƯỚC nó (đạo hàm bậc một). Chưa ai thử. | k ∈ {1,2,3} × β ∈ {0,05; 0,1; 0,25; 0,5; 1,0} |

Cả ba tín hiệu (a), (b), (d) đều được định nghĩa để **giữ nguyên thang cosine**,
vì lý do ở §3.

### Đường đo là đường sản xuất, kiểm bằng assert

sims đầy đủ (`query_similarities`, có cắt khúc câu dài y hệt `search`) → mặt nạ
`valid` → top-400 → ưu tiên đỉnh → nhận dạng đối tượng → `allocate_rows(coverage)`
→ 100 dòng → `final_score`/`r_score_kis`.

Ba bất biến, **kiểm bằng `assert` chứ không bằng mắt**:

1. **Nền dựng lại từ sims trùng khít ứng viên sản xuất đã cache** — cả video,
   cả frame, cả **điểm** (sai khác < 1e-9), trên **132/132 mục**. Nếu bước này
   trượt thì mọi số sau nó vô nghĩa.
2. **Cấu hình đơn vị của mọi họ** (w=0 / λ=0 / β=0 / α=1 / σ_phủ=30) ra 100 dòng
   **giống hệt nền** trên cả 132 mục.
3. **Bản vector hoá của tín hiệu khớp bản lặp từng video** (lệch tối đa < 1e-5,
   6 tín hiệu × 3 mục). Cả họ (a) và (d) đứng trên đúng một mẹo dịch mảng + mặt
   nạ biên video; bất biến (2) chỉ chứng minh mã ngắn mạch ở tham số 0, không
   chứng minh mẹo đó đúng.

### Kỷ luật thống kê

- Chia TUNE/TEST **phân tầng theo nhóm hai cảnh** (33/33 mỗi nửa), không chia
  chẵn/lẻ thô.
- **Mỗi họ chốt hai lần trên TUNE**: theo điểm tổng, và theo điểm nhóm HAI CẢNH.
  Lý do: hai nhóm đi ngược chiều nhau ở họ (a), nên chốt theo tổng một mình sẽ
  luôn chọn cấu hình ít hại nhất cho nhóm dễ và **không bao giờ đọc được** cấu
  hình tốt nhất cho nhóm khó — đúng nhóm mà lane này đi tìm.
- **TEST đọc đúng một lượt**, bootstrap **theo câu** (4 000 lần bốc), báo cáo
  riêng nhóm hai cảnh và nhóm một cảnh.
- Gốc hạt mới: TUNE 81000, TEST 82000 (tách khỏi mọi gốc đã dùng).
- **51 cấu hình được so trên cùng một TUNE.** Cấu hình thắng vì thế bị thổi
  phồng; đó chính là lý do TEST tồn tại, và §5 cho thấy nó bắt đúng.

---

## 3. Hai họ ĐỐI CHỨNG — và vì sao chúng mới là kết quả chính

Bộ phủ xác suất biến điểm thành khối lượng bằng `exp((s − max)/0,02)` rồi rải
mỗi ứng viên bằng một Gauss `sigma = 30 frame`. Hai núm đó, chứ không phải tín
hiệu, quyết định 100 dòng trông thế nào. Nên phải có hai đối chứng **không chạm
vào truy xuất**:

| | đối chứng | nó làm gì |
|---|---|---|
| (e) | kéo giãn thang điểm quanh trung bình toàn kho, α ∈ {0,5; 0,75; 1,5; 2,0} | đúng một phép **đổi nhiệt độ softmax** trá hình. Thứ tự: **không đổi**. |
| (f) | nới `sigma` của tiên nghiệm phủ, σ_phủ ∈ {60; 120; 240; 480} | nới bề rộng vùng được phủ. Điểm: **không đổi**. Thứ tự: **không đổi**. |

(f) sinh ra vì chính kết quả của (a): làm mượt theo thời gian **cũng** nới rộng
vùng được phủ, mà bộ phủ đã có sẵn một tham số làm đúng việc đó — và nó đang đặt
ở **30 frame, hẹp hơn một khe keyframe** (khoảng cách keyframe trung vị 55
frame). Nếu chỉ vặn núm ấy đã mua được phần điểm của (a) thì (a) không thêm
thông tin nào.

Chẩn đoán xác nhận (f) sạch tuyệt đối: **hạng nội-video và số câu có keyframe
đáp án trong pool y hệt nền, từng con số một** (§7). Toàn bộ tác động của nó nằm
ở khâu phân bổ dòng.

---

## 4. Kết quả TUNE (66 mục — nền 0,1429 | hai cảnh 0,0519 | một cảnh 0,2339)

### (a) làm mượt Gauss theo thời gian

| cấu hình | TUNE | so nền | HAI cảnh | so nền | MỘT cảnh | so nền |
|---|---|---|---|---|---|---|
| σ=1 w=0,25 | 0,1277 | −10,6% | 0,0586 | +12,9% | 0,1969 | −15,8% |
| σ=1 w=0,5 | 0,1154 | −19,3% | 0,0709 | +36,6% | 0,1598 | −31,7% |
| σ=1 w=1,0 | 0,0888 | −37,9% | 0,0731 | +40,9% | 0,1044 | −55,4% |
| σ=2 w=0,25 | 0,1235 | −13,6% | 0,0618 | +19,0% | 0,1852 | −20,8% |
| σ=2 w=0,5 | 0,1027 | −28,1% | 0,0772 | +48,7% | 0,1282 | −45,2% |
| σ=2 w=1,0 | 0,0683 | −52,2% | 0,0841 | +62,0% | 0,0525 | −77,6% |
| σ=3 w=0,25 | 0,1237 | −13,5% | 0,0645 | +24,3% | 0,1828 | −21,8% |
| σ=3 w=0,5 | 0,0961 | −32,8% | 0,0743 | +43,1% | 0,1178 | −49,6% |
| **σ=3 w=1,0** | 0,0647 | −54,8% | **0,0873** | **+68,2%** | 0,0420 | −82,1% |
| σ=5 w=0,25 | 0,1176 | −17,7% | 0,0630 | +21,4% | 0,1722 | −26,4% |
| σ=5 w=0,5 | 0,0842 | −41,1% | 0,0658 | +26,8% | 0,1025 | −56,2% |
| σ=5 w=1,0 | 0,0517 | −63,8% | 0,0680 | +31,0% | 0,0354 | −84,9% |

**Hai nhóm đi ngược chiều nhau, đơn điệu theo cả hai tham số.** Càng mượt thì
câu hai cảnh càng lên, câu một cảnh càng sập. Đó là dấu hiệu của một cơ chế thật
chứ không phải nhiễu — và cơ chế ấy đọc được: khi truy vấn tả đúng một khoảnh
khắc, làm mượt **phá** chính cái đỉnh là đáp án; khi truy vấn tả hai cảnh, keyframe
điểm cao nhất **không phải** đáp án, nên trung bình theo vùng kéo được hàng xóm
lên. Tổng luôn âm vì phần mất ở nhóm dễ lớn hơn phần được ở nhóm khó.

### (b) chuẩn hoá theo video

| cấu hình | TUNE | so nền | HAI cảnh | so nền | MỘT cảnh | so nền |
|---|---|---|---|---|---|---|
| trừ TB λ=0,25 | 0,1389 | −2,8% | 0,0396 | −23,6% | 0,2382 | +1,8% |
| trừ TB λ=0,5 | 0,1134 | −20,7% | 0,0242 | −53,5% | 0,2025 | −13,4% |
| trừ TB λ=1,0 | 0,0854 | −40,3% | 0,0112 | −78,5% | 0,1596 | −31,8% |
| z-score λ=0,25 | 0,1293 | −9,5% | 0,0250 | −51,9% | 0,2337 | −0,1% |
| z-score λ=0,5 | 0,0836 | −41,5% | 0,0076 | −85,3% | 0,1595 | −31,8% |
| z-score λ=1,0 | 0,0180 | −87,4% | 0,0004 | −99,2% | 0,0356 | −84,8% |
| min-max λ=0,25 | 0,1247 | −12,8% | 0,0356 | −31,5% | 0,2137 | −8,6% |
| min-max λ=0,5 | 0,1067 | −25,3% | 0,0151 | −71,0% | 0,1983 | −15,2% |
| min-max λ=1,0 | 0,0183 | −87,2% | 0,0035 | −93,2% | 0,0331 | −85,8% |

**Âm ở mọi ô, và có lý do cấu trúc để không bao giờ dương ở việc định vị:** cả
ba kiểu đều là phép affine **hệ số dương trong từng video**, nên chúng **không
đổi được thứ tự nội-video một chút nào**. Chúng chỉ đổi việc so **giữa** các
video (kéo video lạ vào pool) và độ tán của điểm trong video. Họ này mang tên
"chuẩn hoá nội video" nhưng về mặt toán học **không thể** là một bộ định vị.
Con số −0,129 của bảng cũ vẫn đúng dấu, và giờ có lời giải thích.

### (c) ưu tiên đỉnh cục bộ — **trơ tuyệt đối**

| cấu hình | TUNE | so nền | HAI cảnh | MỘT cảnh |
|---|---|---|---|---|
| w = 0 / 0,002 / 0,005 / **0,01 (sản xuất)** / 0,02 / 0,05 / 0,1 | **0,1429** | **+0,0%** | **0,0519** | **0,2339** |

Bảy giá trị, **cùng một con số đến từng chữ số**. Xem §6.

### (d) biên cảnh (đạo hàm bậc một)

| cấu hình | TUNE | so nền | HAI cảnh | so nền | MỘT cảnh | so nền |
|---|---|---|---|---|---|---|
| k=1 β=0,05 | 0,1477 | +3,3% | 0,0480 | −7,5% | 0,2473 | +5,7% |
| k=1 β=0,1 | 0,1529 | +7,0% | 0,0451 | −13,1% | 0,2608 | +11,5% |
| **k=1 β=0,25** | **0,1549** | **+8,4%** | 0,0380 | −26,8% | 0,2718 | +16,2% |
| k=1 β=0,5 | 0,1180 | −17,5% | 0,0231 | −55,5% | 0,2128 | −9,0% |
| k=1 β=1,0 | 0,0636 | −55,5% | 0,0011 | −97,9% | 0,1262 | −46,1% |
| k=2 β=0,05 | 0,1450 | +1,5% | 0,0496 | −4,4% | 0,2404 | +2,8% |
| k=2 β=0,1 | 0,1500 | +5,0% | 0,0452 | −12,9% | 0,2548 | +8,9% |
| k=2 β=0,25 | 0,1539 | +7,7% | 0,0369 | −28,9% | 0,2709 | +15,8% |
| k=2 β=0,5 | 0,1225 | −14,3% | 0,0228 | −56,0% | 0,2222 | −5,0% |
| k=2 β=1,0 | 0,0697 | −51,2% | 0,0048 | −90,8% | 0,1346 | −42,4% |
| k=3 β=0,05 | 0,1431 | +0,1% | 0,0458 | −11,8% | 0,2404 | +2,8% |
| k=3 β=0,1 | 0,1459 | +2,1% | 0,0451 | −13,2% | 0,2467 | +5,5% |
| k=3 β=0,25 | 0,1495 | +4,6% | 0,0366 | −29,6% | 0,2624 | +12,2% |
| k=3 β=0,5 | 0,1278 | −10,6% | 0,0239 | −54,0% | 0,2316 | −1,0% |
| k=3 β=1,0 | 0,0747 | −47,7% | 0,0042 | −92,0% | 0,1453 | −37,9% |

Tín hiệu chưa ai thử này là họ **duy nhất dương trên tổng TUNE** (+8,4%) — và nó
dương **ngược hướng giả thuyết**: nó giúp câu MỘT cảnh (+16,2%) và **hại** câu
HAI cảnh (−26,8%). Trực giác "khoảnh khắc phải nộp là lúc một cảnh bắt đầu"
không được số liệu ủng hộ. Đọc lại thì hợp lý: đạo hàm dương đánh dấu chỗ **bắt
đầu đoạn khớp truy vấn**, mà với câu hai cảnh thì đoạn khớp truy vấn là cảnh A
— tức nó đẩy mạnh thêm đúng chỗ **sai**.

### (e) + (f) ĐỐI CHỨNG — không đụng truy xuất, thắng mọi tín hiệu thật

| đối chứng | TUNE | so nền | HAI cảnh | so nền | MỘT cảnh | so nền |
|---|---|---|---|---|---|---|
| (e) α=0,5 | 0,1104 | −22,7% | 0,0302 | −41,8% | 0,1907 | −18,5% |
| (e) α=0,75 | 0,1289 | −9,8% | 0,0432 | −16,9% | 0,2146 | −8,3% |
| (e) α=1,5 | 0,1553 | +8,7% | 0,0618 | +19,1% | 0,2488 | +6,4% |
| **(e) α=2,0** | **0,1659** | **+16,1%** | 0,0721 | +38,8% | 0,2597 | +11,0% |
| **(f) σ_phủ=60** | **0,1713** | **+19,8%** | 0,1020 | +96,6% | 0,2405 | +2,8% |
| (f) σ_phủ=120 | 0,1645 | +15,1% | 0,1276 | +145,8% | 0,2013 | −13,9% |
| (f) σ_phủ=240 | 0,1479 | +3,5% | 0,1227 | +136,3% | 0,1732 | −25,9% |
| **(f) σ_phủ=480** | 0,1447 | +1,3% | **0,1337** | **+157,6%** | 0,1557 | −33,4% |

**Đọc bảng này trước khi đọc bất cứ bảng nào ở trên.** Hai họ không mang một bit
thông tin mới nào — không đổi điểm ai, không đổi thứ tự ai — mà:

- trên **tổng**: +19,8% (f) và +16,1% (e), so với **+8,4%** của tín hiệu thật
  tốt nhất;
- trên **hai cảnh**: +157,6% (f), so với **+68,2%** của tín hiệu thật tốt nhất.

Toàn bộ "phần thắng ở nhóm hai cảnh" mà làm mượt thời gian có vẻ mua được, chỉ
là **một phiên bản tệ hơn của việc vặn một tham số bộ phân bổ**.

---

## 5. Kết quả TEST — đọc đúng một lượt (66 mục, nền 0,2365 | hai cảnh 0,1547 | một cảnh 0,3183)

| họ [chốt theo] | cấu hình | TEST | so nền | KTC 95% theo câu | P(≤0) |
|---|---|---|---|---|---|
| (a) [tổng] | σ=1 w=0,25 | 0,2330 | −1,5% | [−0,0189, +0,0120] | 66,0% |
| ├─ HAI cảnh (n=33) | | 0,1630 | +5,4% | [−0,0097, +0,0269] | 17,7% |
| └─ MỘT cảnh (n=33) | | 0,3030 | −4,8% | [−0,0407, +0,0111] | 86,5% |
| **(a) [hai cảnh]** | **σ=3 w=1,0** | 0,1190 | −49,7% | [−0,1762, −0,0645] | 100% |
| ├─ **HAI cảnh** | | **0,1450** | **−6,2%** | [−0,0622, +0,0415] | 62,5% |
| └─ MỘT cảnh | | 0,0930 | −70,8% | [−0,3115, −0,1466] | 100% |
| (b) [tổng] | trừ TB λ=0,25 | 0,2202 | −6,9% | [−0,0289, −0,0051] | 99,8% |
| ├─ HAI cảnh | | 0,1367 | −11,6% | [−0,0336, −0,0049] | 99,7% |
| └─ MỘT cảnh | | 0,3038 | −4,5% | [−0,0343, +0,0034] | 94,3% |
| (c) [tổng] | w=0 (tắt hẳn) | 0,2365 | **±0,0%** | [0, 0] | *lệch đúng bằng 0* |
| **(d) [tổng]** | **k=1 β=0,25** | 0,2147 | **−9,2%** | [−0,0485, +0,0058] | 93,8% |
| ├─ HAI cảnh | | 0,1124 | −27,4% | [−0,0717, −0,0175] | 100% |
| └─ MỘT cảnh | | 0,3171 | −0,4% | [−0,0479, +0,0481] | 51,3% |
| (d) [hai cảnh] | k=2 β=0,05 | 0,2364 | −0,0% | [−0,0079, +0,0094] | 51,9% |
| ├─ HAI cảnh | | 0,1465 | −5,3% | [−0,0147, −0,0023] | 99,7% |
| └─ MỘT cảnh | | 0,3264 | +2,5% | [−0,0060, +0,0241] | 15,0% |
| *(e) đối chứng* | α=2,0 | 0,2450 | +3,6% | [−0,0055, +0,0218] | 11,6% |
| ├─ HAI cảnh | | 0,1559 | +0,8% | [−0,0171, +0,0178] | 44,2% |
| └─ MỘT cảnh | | 0,3342 | +5,0% | [−0,0050, +0,0378] | 7,0% |
| *(f) đối chứng* | σ_phủ=60 | 0,2440 | +3,2% | [−0,0125, +0,0293] | 23,1% |
| ├─ HAI cảnh | | 0,1554 | +0,5% | [−0,0269, +0,0304] | 49,0% |
| └─ MỘT cảnh | | 0,3325 | +4,5% | [−0,0149, +0,0459] | 17,7% |
| *(f) đối chứng* | σ_phủ=480 | 0,1922 | −18,7% | [−0,0876, +0,0002] | 97,5% |
| ├─ HAI cảnh | | 0,1713 | +10,8% | [−0,0412, +0,0825] | 30,6% |
| └─ MỘT cảnh | | 0,2130 | −33,1% | [−0,1601, −0,0551] | 100% |

### Sập TUNE → TEST là hệ thống, không phải một ca lẻ

| | TUNE | TEST |
|---|---|---|
| (a) σ=3 w=1,0, nhóm hai cảnh | **+68,2%** | **−6,2%** |
| (d) k=1 β=0,25, tổng | **+8,4%** | **−9,2%** (đổi dấu) |
| (e) α=2,0, tổng | +16,1% | +3,6% |
| (f) σ_phủ=480, nhóm hai cảnh | **+157,6%** | +10,8% (P(hoà) 30,6%) |

Bốn phép đo: hai lần **đổi dấu**, hai lần **co lại 4,5 và 15 lần**. **Không một cấu hình nào —
kể cả hai đối chứng — có khoảng tin cậy tách khỏi 0.** Giao thức đã làm đúng
việc của nó.

---

## 6. `PEAK_WEIGHT` hiện TRƠ với 100 dòng nộp

Bảy giá trị w từ 0 đến 0,1, trên cả TUNE và TEST, ở cả hai nhóm: **điểm giống
hệt nền đến từng chữ số, khoảng tin cậy theo câu là [0, 0]** — tức mọi câu lệch
đúng bằng 0, tức **100 dòng không đổi một dòng nào** trên cả 132 mục.

Lý do là cấu trúc, không phải may rủi: `_peak_preference` **chỉ xếp lại thứ tự**
hits và mang điểm cũ đi theo, còn bộ phủ `coverage` dựng khối lượng bằng tổng
Gauss có trọng số theo **điểm**, hoàn toàn không đọc thứ hạng. Con số **+2,2%**
trong bảng tín hiệu được đo dưới bộ phân bổ **`hybrid`** — bộ xếp dòng theo
hạng — nên nó **không còn hiệu lực** kể từ khi `coverage` thành mặc định.

**Đây là phát hiện dành cho lane sản xuất, không phải để lane này tự sửa:**

1. Bảng tín hiệu nên hạ dòng "ưu tiên đỉnh cục bộ ✅ +2,2%" xuống *chỉ áp dụng
   cho allocator `hybrid`*.
2. **Đừng vội xoá code.** `ranked_hits` còn được `answer_qa.py` và
   `build_review_page.py` đọc từ **đầu danh sách** — ở đó thứ tự vẫn có tác dụng.
   Phép đo này chỉ nói nó trơ với **100 dòng KIS**, không nói gì về Q&A.
3. Nó vẫn là đường lui hợp lệ nếu có ngày quay lại `hybrid`.

---

## 7. Chẩn đoán bắt buộc — hạng nội-video trước và sau

Hai thước, vì một mình thước thứ nhất đọc sai được: **hạng trong pool** lẫn cỡ
pool (tín hiệu kéo thêm keyframe của video đúng vào pool sẽ làm hạng *xấu đi* dù
đang giúp), còn **hạng trên toàn video** không phụ thuộc pool lẫn bộ phân bổ.
Đo trên cả 132 mục.

| nhóm | | đáp án trong pool | hạng trong pool | hạng-1 | hạng TOÀN video | hạng-1 |
|---|---|---|---|---|---|---|
| MỘT cảnh | nền | 58/66 | 2,0 | 41% | 2,0 | 38% |
| HAI cảnh | nền | **35/66** | **8,0** | 3% | **15,0** | 2% |
| MỘT cảnh | (d) k=1 β=0,25 | 52/66 | 2,0 | 46% | 2,0 | 38% |
| HAI cảnh | (d) k=1 β=0,25 | **31/66** | **9,0** | 6% | **16,5** | 3% |
| MỘT cảnh | (a) σ=3 w=1,0 | 31/66 | 6,0 | 13% | 7,5 | 9% |
| HAI cảnh | (a) σ=3 w=1,0 | **41/66** | 8,0 | 10% | **11,0** | 6% |
| MỘT / HAI | (f) σ_phủ bất kỳ | *y hệt nền* | *y hệt nền* | | *y hệt nền* | |

Ba điều đọc ra, và điều đầu tiên là lý do bước chẩn đoán này bắt buộc:

**(1) (d) tăng điểm TUNE trong khi KHÔNG cải thiện định vị ở đâu cả.** Nó **đẩy
keyframe đáp án RA KHỎI pool**: một cảnh 58→52, hai cảnh 35→31. Ở nhóm hai cảnh
mọi thước đều xấu đi (hạng trong pool 8,0→9,0; hạng toàn video 15,0→**16,5**);
ở nhóm một cảnh hạng đứng yên (2,0→2,0, hạng-1 toàn video 38%→38%) — con số
41%→46% ở cột hạng-1 *trong pool* chỉ là ảo ảnh của việc pool teo lại, đúng cái
bẫy mà thước thứ hai sinh ra để chặn.

Một tín hiệu **định vị** mà không cải thiện định vị ở bất kỳ thước nào thì phần
điểm +8,4% nó "mua" trên TUNE không thể là định vị — và TEST xác nhận: **−9,2%**.
Hai thước này độc lập hoàn toàn với bộ phân bổ và với việc chấm điểm, nên chúng
là thứ duy nhất trong phép đo có thể bác một con số TUNE đẹp mà không cần chờ
TEST.

**(2) (a) có cơ chế THẬT ở nhóm hai cảnh.** Pool 35→**41** (cứu thêm 6 câu mà
keyframe đáp án chưa từng được truy xuất — đây là phép **đếm tất định**, không
có khoảng tin cậy nào để bàn), hạng toàn video 15,0→**11,0**, hạng-1 2%→6%. Và
nó trả giá đúng như dự đoán ở nhóm một cảnh: pool 58→31, hạng 2,0→7,5. Cơ chế
đúng hướng, nhưng **không quy đổi được thành điểm** (TEST −6,2%).

**(3) (f) sạch tuyệt đối** — mọi con số y hệt nền, xác nhận nó không chạm vào
truy xuất, nên +157,6% của nó trên TUNE hoàn toàn là hiệu ứng phân bổ dòng.

---

## 8. Bộ đo này chưa phân xử được cỡ hiệu ứng đang xét — số liệu

Chấm nền trên cả 132 mục bằng **cùng một họ hạt giống** (nên chênh lệch dưới đây
**chỉ do tập câu**, không do bốc thăm):

| | TEST | TUNE | chênh | KTC 95% theo câu | P(≤0) |
|---|---|---|---|---|---|
| toàn bộ | 0,2376 | 0,1438 | **+0,0938** | [+0,0069, +0,1822] | 1,6% |
| chỉ HAI cảnh | 0,1566 | 0,0542 | **+0,1024** | [+0,0202, +0,1934] | 0,7% |
| chỉ MỘT cảnh | 0,3186 | 0,2333 | +0,0853 | [−0,0580, +0,2245] | 12,2% |

**Hai nửa của cùng một bộ đo không hoán đổi được cho nhau.** Nhóm hai cảnh ở nửa
TEST dễ gấp **2,9 lần** nửa TUNE, và khoảng cách đó **lớn hơn mọi hiệu ứng mà
lane này đi đo**.

Nguyên nhân nằm ở phân bố điểm, không ở cách chia:

| nhóm | số câu **0 điểm** | trung vị | p90 | max |
|---|---|---|---|---|
| HAI cảnh | **38/66** | 0,0000 | 0,3443 | 0,8472 |
| MỘT cảnh | 23/66 | 0,1566 | 0,7196 | 0,9212 |

Hơn một nửa câu hai cảnh **trượt sạch**, nên trung bình của nhóm do ~28 câu gánh.
Chia đôi thì mỗi nửa còn ~14 câu mang điểm — chia thế nào cũng lệch.

**Hệ quả vận hành:** với n=33 mỗi nhóm mỗi nửa, nửa rộng KTC 95% cho một chênh
lệch ở nhóm hai cảnh chạy từ **0,006** (cấu hình đụng ít dòng, như (d) k=2 β=0,05)
tới **0,06** (cấu hình đụng nhiều dòng, như (a) σ=3 w=1,0 hay (f) σ_phủ=480). Một
hiệu ứng +20% trên nền 0,155 là +0,031 — **nằm gọn trong nhiễu** của đúng nhóm
cấu hình mạnh tay, tức nhóm duy nhất có cửa mua được ngần ấy. Bộ đo này đủ sức
thấy khoảng cách 3× giữa hai loại câu (§1), **không** đủ sức phân xử vài chục
phần trăm.

---

## 9. Nên làm gì tiếp

**Không ship gì từ lane này.** Bốn tín hiệu đều âm hoặc hoà trên TEST; hai đối
chứng dương nhưng khoảng tin cậy chứa 0.

1. **Đừng mở lại (b).** Nó không thể là bộ định vị vì lý do đại số (§4b), không
   phải vì mẫu nhỏ. Cửa này đóng vĩnh viễn.
2. **Đừng mở lại (d) theo hướng cũ.** Đạo hàm dương đánh dấu chỗ bắt đầu đoạn
   **khớp truy vấn**, mà ở câu hai cảnh thì đó là cảnh A — đúng chỗ đã thừa ứng
   viên. Nếu còn thử biên cảnh thì phải là **biên SAU đoạn khớp** (cảnh kế tiếp),
   tức đúng cơ chế mà `--canh-b` đã chữa bằng cách rẻ hơn và đã ship.
3. **(a) là thứ duy nhất còn đáng theo, nhưng chỉ khi có cổng và có thêm dữ
   liệu.** Nó cứu 6 câu vào pool và kéo hạng toàn video 15,0→11,0 ở nhóm hai
   cảnh — cơ chế thật, đo bằng phép đếm. Điều kiện để đo lại cho ra kết luận:
   (i) **chỉ bật cho câu qua cổng hai cảnh** (bật toàn bộ thì phần mất ở nhóm
   một cảnh nuốt hết); (ii) đủ mẫu theo §8. Trước khi có (ii) thì đo lại cũng
   chỉ ra một con số nữa không tách khỏi 0.
4. **Đừng đi chỉnh lại `CoveragePlan` vì hai đối chứng ở đây — việc đó đã làm
   xong và ra ÂM.** Hai con số của (e) và (f) trên TUNE rất dễ đọc thành "tham
   số bộ phủ đang đặt sai, đi chỉnh đi": chúng được chốt trên bộ 60 câu cũ
   (`docs/SHIP_PHU_XAC_SUAT.md`), chính bộ đã bị chứng minh thổi phồng ~2 lần.
   Một lane song song đã quét đủ lưới tham số phân bổ trên đúng bộ 132 mục này
   (`docs/PHAN_BO_TREN_BO_MOI.md`): tổ hợp thắng **+25,4% TUNE → +1,5% TEST
   (hoà, P(hoà) 37%)**, và **−8,2% trên bộ 60 câu cũ (P(≤0) 99,8%)**. Tức σ lớn
   hơn không phải "tham số đặt sai" mà là **đánh đổi giữa hai phân bố câu hỏi**.
   Kết quả của họ khớp hoàn toàn với (e)/(f) ở đây (+16–20% TUNE, +3% TEST) và
   khép lại hướng đó. Hai đối chứng này vì thế **chỉ còn giá trị làm đối chứng**
   — đúng vai trò chúng sinh ra để làm.
5. **Luật chung rút ra, và đây mới là thứ đáng mang đi:** mọi phép đo tín hiệu
   chấm qua bộ phủ **bắt buộc** kèm hai đối chứng (e) và (f). Chúng rẻ (mỗi cái
   4 cấu hình, không cần model, không cần truy xuất lại) và chúng đặt đúng câu
   hỏi: *"phần điểm này có cần thông tin mới không, hay chỉ cần vặn núm?"* Không
   có chúng, một tín hiệu chỉ làm phẳng hoặc làm nhọn tiên nghiệm sẽ được ghi
   công là tri thức mới.

---

## 10. Điều chưa biết

1. **Tín hiệu này chưa từng được đo dưới allocator `hybrid`.** Kết luận "(c)
   trơ" và phần lớn bảng đều gắn chặt với `coverage`. Nếu có ngày quay lại
   `hybrid` thì phải đo lại từ đầu.
2. **Chưa thử làm mượt CÓ CỔNG.** Mọi cấu hình ở đây bật cho toàn bộ 132 câu.
   Bản có cổng (chỉ bật cho câu hai cảnh) chưa được đo, và theo §7 nó là biến
   thể duy nhất có cơ sở cơ học.
3. **Chưa thử tổ hợp.** Cố ý: đo riêng từng họ để biết cái nào ăn. Vì không cái
   nào ăn nên chưa có gì để gộp.
4. **Nhãn `co_2_canh` trên bộ đo là tự khai ở shard b**, nên nhóm "hai cảnh" ở
   đây rộng hơn cổng của `--canh-b` (vốn đòi tách được cảnh A/B). Con số theo
   nhóm vì thế không so trực tiếp được với `docs/UNG_VIEN_CANH_B.md`.
5. **Tất cả đo trên câu do máy sinh.** Nếu câu máy sinh có cấu trúc thời gian
   "sạch" hơn đề người viết thì cả phần được lẫn phần mất của làm mượt đều bị
   phóng đại.

---

## Chạy lại

```
python -u scripts/do_tin_hieu_noi_video.py --sims   # lần đầu: dựng cache sims (~2,5 phút)
python -u scripts/do_tin_hieu_noi_video.py          # toàn bộ (a)-(f), ~25 phút
python -u scripts/do_tin_hieu_noi_video.py --ho f   # chỉ một họ
python -u scripts/do_tin_hieu_noi_video.py --nhanh  # chạy thử, mỗi họ 2 cấu hình
```

Số trong tài liệu này đến từ **hai lượt chạy**: (a)–(e) trước, rồi `--ho f` sau
khi chính kết quả của (a) đòi thêm đối chứng thứ hai. Chia TUNE/TEST và gốc hạt
là tất định nên hai lượt dùng đúng một nền — cả hai đều in `nen TEST 0,2365`.
Chạy một lượt duy nhất bây giờ cho ra cùng bảng.

Cache `data/cache_tin_hieu_noi_video/sims_sach.npy` (93 MB, 132 × 177 321
float32) là sims đầy đủ của từng câu; dựng một lượt bằng **một** lượt quét chỉ
mục (gọi thẳng `query_similarities` 132 lần thì mỗi lần quét lại 817 MB và chậm
gấp mười trên máy đang chạy nhiều lane).
