# Rộng vs Sâu trên bộ đo khớp phân bố — cánh bậc mở lại, và đóng lại có số

Chốt 01/09/2026. Bộ đo: `data/ground_truth_moi.json`, **132 mục sạch**
(bỏ shard c lẫn trục), 66 câu một cảnh / 66 câu hai cảnh.
Script: `scripts/do_phan_bo_sau.py`, `scripts/chan_doan_dat_dong.py`,
`scripts/tran_dat_dong_theo_pool.py`, `scripts/quet_sigma_bo_cu.py`,
`scripts/doc_luoi_phan_bo_sau.py`.

> Không mã video, không đáp án trong tài liệu này (docs/ lên GitHub công khai).

**Kết luận một dòng: ÂM.** Không tổ hợp tham số phân bổ nào giữ được lợi thế.
Tổ hợp thắng trên TUNE **+25,4%** rơi xuống **+1,5% (hoà, P(hoà) = 37%)** trên
TEST và **−8,2% (P(≤0) = 99,8%)** trên bộ 60 câu cũ. Họ `hybrid` âm ở mọi ô.
Dòng "chia lại ngân sách theo (video, keyframe, độ sâu) — KHÔNG DÙNG" trong bảng
tín hiệu **vẫn đứng**, và giờ đứng trên bộ đo khớp phân bố chứ không chỉ bộ cũ.

Nhưng phép đo mở ra một thứ khác, lớn hơn nhiều, ở §1 và §5.

---

## 0. Vì sao mở lại cánh cửa này

Bảng tín hiệu đóng nó bằng một phép đo trên bộ 60 câu CŨ. Lập luận mở lại:
oracle định vị nội-video trên bộ mới cho **+126%**, và oracle đó *không đổi việc
chọn video, không đổi thứ hạng* — nó chỉ đặt lại **frame id**. Nghĩa là headroom
nằm đúng ở thứ tham số phân bổ điều khiển.

**Lập luận đó sai ở một chỗ, và chỗ ấy quyết định toàn bộ kết quả.** Oracle được
đặt thang quanh **khoảnh khắc thật**. Bộ phân bổ chỉ được đặt dòng quanh
**keyframe mà khâu truy xuất đã trả về**. Hai thứ đó không cùng một trần — xem §5.

Ghi chú lịch sử đáng để ý: bộ 60 câu cũ **cũng** có oracle vị trí frame
**+115%** (`KIEN_TRUC_VA_HUONG_CAI_THIEN.md` §2b) mà mọi phép chia lại ngân sách
trên nó vẫn âm 15–30%. Tức "oracle nói to" chưa bao giờ là bằng chứng cho lane
phân bổ, kể cả trên bộ cũ. Bộ mới không đổi điều đó.

---

## 1. Chẩn đoán — 100 dòng hiện tại rơi vào đâu (phép ĐẾM, không có khoảng tin cậy)

Hệ sản xuất, allocator `coverage`, tham số chốt (0,02 / σ=30 / nửa=6 / lưới=5):

| nhóm | n | có ≥1 dòng ở video đúng | dòng/video đúng | câu có dòng ≤50 | dòng ≤50 | câu ≤100 | câu ≤200 | hạng dòng đúng-video đầu | số video trong 100 dòng | trải rộng dòng trong video đúng |
|---|---|---|---|---|---|---|---|---|---|---|
| bộ SẠCH | 132 | 104 | 21,5 | 67 | 2,6 | 76 | 84 | 2,0 | 14,5 | 3.055 |
| ├─ MỘT cảnh | 66 | 51 | 20,7 | 41 | 3,5 | 43 | 45 | 1,0 | 15,2 | 2.015 |
| └─ HAI cảnh | 66 | 53 | 22,4 | **26** | **1,7** | 33 | 39 | 4,0 | 13,8 | **4.135** |

**Có, allocator đang rải quá mỏng — nhưng chỉ ở câu hai cảnh.** Hai nhóm chọn
đúng video ngang nhau (51 vs 53) và tiêu ngần ấy dòng vào video đúng (20,7 vs
22,4), nhưng câu hai cảnh chỉ có **1,7 dòng** nằm trong ±50 frame quanh đáp án
so với 3,5 — và 22 dòng ấy trải trên **4.135 frame**, tức trung bình một dòng
mỗi ~190 frame trong khi cửa sổ chấm rộng nhất chỉ ±20.

Dòng gần nhất cách đáp án bao xa (`chan_doan_dat_dong.py` bảng 1):

| nhóm | có dòng ở video đúng | trung vị khoảng cách | ≤20 | ≤50 | ≤100 |
|---|---|---|---|---|---|
| MỘT cảnh | 51 | **2 frame** | 40 | 41 | 43 |
| HAI cảnh | 53 | **56 frame** | 24 | 26 | 33 |

Ở câu một cảnh bộ phân bổ đặt gần như trúng tim. Ở câu hai cảnh nó trượt.

### 1b. Nhưng phần trượt đó phần lớn KHÔNG phải lỗi phân bổ

Tách ba loại thất bại ở bán kính ±20 frame (`chan_doan_dat_dong.py` bảng 3):

| nhóm | n | đã trúng | **mất do ĐẶT DÒNG** | không có ứng viên nào gần | sai video |
|---|---|---|---|---|---|
| MỘT cảnh | 66 | 40 | 8 | 8 | 10 |
| HAI cảnh | 66 | 24 | **9** | **27** | 6 |

Cột "mất do đặt dòng" — video đúng CÓ dòng, trong 400 ứng viên CÓ keyframe cách
đáp án ≤20 frame, mà không dòng nào rơi vào đó — là **phần duy nhất** tham số
phân bổ có quyền động tới: **9/66 câu hai cảnh, 8/66 câu một cảnh**.

Với 27/66 câu hai cảnh (41%) thì trong cả 400 ứng viên **không có keyframe nào**
của video đúng nằm gần đáp án. Không tham số nào cứu được chúng. Đó là nghẽn ở
khâu **sinh ứng viên** — đúng thứ `docs/UNG_VIEN_CANH_B.md` đang chữa.

Hạng nội-video của keyframe gần đáp án nhất, đo lại trên cả ba tập:

| tập | trung vị | hạng-1 | ≤3 |
|---|---|---|---|
| bộ CŨ 60 câu | 2,0 | 45% | 58% |
| bộ mới — MỘT cảnh | 2,0 | 43% | 68% |
| bộ mới — HAI cảnh | **6,0** | **11%** | 31% |

Bộ cũ và nhóm MỘT cảnh của bộ mới **trùng khít**. Cánh cửa đóng trên bộ cũ đóng
đúng cho nhóm một cảnh, không có gì phải mở lại. Đất mới duy nhất là nhóm hai
cảnh — và §2 đo xem nó có mua được gì không.

---

## 2. Quét lưới — 216 tổ hợp, chọn trên TUNE

Chia TUNE/TEST **phân tầng theo trục bị tác động** (một cảnh / hai cảnh), mỗi
nửa 66 mục / 33 câu hai cảnh. TUNE: 3 họ hạt giống × 32 bốc (gốc 310000).
TEST: 4 họ × 48 bốc (gốc 320000). Cửa sổ chấm {6, 10, 20}.

**Bất biến đã kiểm bằng assert:** trên toàn bộ 132 mục, đường sinh dòng của
script **trùng từng dòng** với `make_submission.allocate_rows()` khi truyền tham
số mặc định. Đây là bất biến đúng cho lane này — tham số phân bổ tác động lên
MỌI câu, nên không có nhóm "cổng tắt" nào để so.

Nền trên TUNE: **0,1415** (nhóm hai cảnh 0,0528).

### (a) CoveragePlan — 200 tổ hợp

12 tổ hợp cao nhất trên TUNE:

| nhiệt | σ | nửa | lưới | TUNE | so nền | nhóm HAI cảnh | so nền |
|---|---|---|---|---|---|---|---|
| 0,01 | 45 | 6 | 5 | **0,1774** | **+25,4%** | 0,1007 | +90,6% |
| 0,015 | 60 | 6 | 5 | 0,1757 | +24,2% | 0,1077 | +104,1% |
| 0,01 | 60 | 6 | 5 | 0,1738 | +22,9% | 0,1068 | +102,2% |
| 0,01 | 60 | 10 | 5 | 0,1737 | +22,8% | 0,1102 | +108,6% |
| 0,015 | 45 | 6 | 5 | 0,1733 | +22,5% | 0,0947 | +79,4% |
| 0,02 | 60 | 6 | 5 | 0,1726 | +22,0% | 0,1059 | +100,5% |

60/200 tổ hợp ≥ nền. Phân phối toàn lưới: min 0,0637 / trung vị 0,1259 / max 0,1774.

Biên từng tham số (trung bình mọi tổ hợp mang giá trị đó — `doc_luoi_phan_bo_sau.py`):

| σ | TUNE trung bình | nhóm HAI cảnh |  | nhiệt | TUNE trung bình |
|---|---|---|---|---|---|
| 15 | 0,0973 | 0,0286 |  | 0,01 | 0,1432 |
| 20 | 0,1082 | 0,0365 |  | 0,015 | 0,1383 |
| **30 (nền)** | 0,1289 | 0,0574 |  | **0,02 (nền)** | 0,1319 |
| 45 | 0,1465 | 0,0822 |  | 0,03 | 0,1182 |
| 60 | **0,1498** | **0,0964** |  | 0,05 | 0,0989 |

`nửa_cửa_sổ` phẳng ở 6/10/15 rồi tụt ở 25; `lưới` 5 và 10 gần như bằng nhau.

### (b) hybrid — 16 tổ hợp, bảng đầy đủ

| n_flat | dc=0,25 | dc=0,5 | dc=0,75 | dc=1,0 |
|---|---|---|---|---|
| 10 | **0,1422** | 0,1370 | 0,1294 | 0,1247 |
| 20 | 0,1301 | 0,1259 | 0,1184 | 0,1165 |
| 30 | 0,1217 | 0,1187 (sản xuất cũ) | 0,1132 | 0,1099 |
| 50 | 0,1054 | 0,1015 | 0,0995 | 0,0971 |

Cả 16 ô đều ≤ nền coverage 0,1415. Đơn điệu theo cả hai trục. Không có gì để chọn.

---

## 3. TEST — đọc ĐÚNG MỘT LẦN

Nền TEST: **0,2392** (hai cảnh 0,1573 / một cảnh 0,3212).

| tổ hợp chốt | nhóm | nền | mới | chênh | KTC 95% theo câu | P(≤0) |
|---|---|---|---|---|---|---|
| coverage (0,01; 45; 6; 5) | toàn bộ 66 | 0,2392 | 0,2428 | **+1,5%** | [−0,0160, +0,0247] | **37,3%** |
| | HAI cảnh 33 | 0,1573 | 0,1514 | **−3,8%** | [−0,0332, +0,0226] | 66,2% |
| | MỘT cảnh 33 | 0,3212 | 0,3342 | +4,1% | [−0,0158, +0,0433] | 19,1% |
| hybrid (10; 0,25) | toàn bộ 66 | 0,2392 | 0,2005 | **−16,2%** | [−0,0679, −0,0129] | 99,8% |
| | HAI cảnh 33 | 0,1573 | 0,1105 | **−29,8%** | [−0,0873, −0,0131] | 99,9% |
| | MỘT cảnh 33 | 0,3212 | 0,2906 | −9,5% | [−0,0734, +0,0074] | 93,6% |

**+25,4% trên TUNE → +1,5% hoà trên TEST.** Và nhóm hai cảnh — chính nhóm mà
TUNE hứa **+90,6%** — trên TEST là **−3,8%**. Đây là dạng hỏng mà giao thức
TUNE/TEST tồn tại để bắt, và nó đã bắt đúng lần thứ hai: lần đầu là tổ hợp
(0,03; 30; 10) hồi 29/08, TUNE +20,1% → TEST −1,0%.

Nhân tiện, một con số nói rõ vì sao n = 66 không phân xử nổi mức vài phần trăm:
hai nửa cùng phân tầng nhưng **nền lệch nhau 69%** (0,1415 vs 0,2392). Với biên
độ giữa-câu lớn như thế, một hiệu ứng +25% trên một nửa hoàn toàn nằm trong tầm
mà việc bốc câu tự sinh ra.

---

## 4. Kiểm chéo bộ 60 câu CŨ — và đây là chỗ câu chuyện kết thúc

Đọc một lần, gốc hạt 330000 (tách khỏi 50000/90000 đã dùng):

| | điểm | so nền | KTC 95% theo câu | P(≤0) |
|---|---|---|---|---|
| nền sản xuất | 0,4024 | — | — | — |
| coverage (0,01; 45; 6; 5) | 0,3694 | **−8,2%** | [−0,0578, −0,0096] | **99,8%** |
| hybrid (10; 0,25) | 0,3026 | −24,8% | [−0,1429, −0,0607] | 100,0% |

Tổ hợp thắng trên TUNE của bộ mới **thua có ý nghĩa** trên bộ cũ. Nói cách khác:
nó không hoà ở cả hai nơi — nó hoà ở một nơi và âm ở nơi kia.

### Trục σ trên bộ cũ đi NGƯỢC chiều — đây là đánh đổi, không phải cải tiến

Cám dỗ lớn nhất của §2 là đọc biên σ đơn điệu (0,0973 → 0,1498) thành "σ lớn thì
tốt hơn, chỉ là TEST chưa đủ mạnh để thấy". Phải chặn cách đọc đó bằng hai điều.

Thứ nhất, **biên đó không phải bằng chứng khái quát**: cả 200 tổ hợp dùng CHUNG
66 câu TUNE, nên 40 tổ hợp mỗi mức σ không phải 40 phép đo độc lập — biên nói về
hình dạng mặt tham số, không nói về khả năng khái quát.

Thứ hai, có sẵn một tập dữ liệu độc lập để hỏi. Quét đúng trục σ trên bộ 60 câu
cũ (`quet_sigma_bo_cu.py`):

| σ | nhiệt 0,02 | so nền | | nhiệt 0,01 |
|---|---|---|---|---|
| 15 | 0,3974 | −1,2% | | 0,4037 |
| 20 | 0,4043 | +0,5% | | 0,3994 |
| **30 (nền)** | **0,4024** | — | | 0,3841 |
| 45 | 0,3754 | **−6,7%** | | 0,3694 |
| 60 | 0,3625 | **−9,9%** | | 0,3412 |

**Đơn điệu ngược hẳn.** Trên bộ mới σ lớn kéo TUNE lên; trên bộ cũ σ lớn kéo
điểm xuống đều đặn tới −9,9%. Cơ chế khớp với §1b: σ trải khối lượng xác suất
rộng ra quanh mỗi keyframe ứng viên. Khi đáp án nằm **ngay trên** ứng viên tốt
nhất (bộ cũ: hạng nội-video trung vị 2, hạng-1 45%) thì trải rộng là lỗ thuần.
Khi đáp án nằm **xa** ứng viên tốt nhất (bộ mới, câu hai cảnh: trung vị 6,
hạng-1 11%) thì trải rộng *có thể* có lý — nhưng TEST nói mức đó là hoà.

Kết luận đúng: **σ = 30 hiện tại không phải sai; nó là điểm cân giữa hai phân
bố.** Đẩy σ lên đổi điểm của câu một cảnh lấy câu hai cảnh, và tỷ giá đó không
có lãi ở bất kỳ mức nào đo được.

---

## 5. Phát hiện đáng giá nhất của lane này: trần +126% KHÔNG thuộc về khâu phân bổ

`tran_dinh_vi_noi_video.py` cho +126% và tiền đề của lane này dựa vào đó. Nhưng
oracle ấy đặt thang quanh **khoảnh khắc thật**. Bộ phân bổ không biết khoảnh
khắc thật; nó chỉ đặt được dòng quanh các **keyframe đã được truy xuất**.

`tran_dat_dong_theo_pool.py` đo cái trần đúng: giữ nguyên VIDEO và THỨ HẠNG từng
dòng y hệt oracle kia, nhưng thang đặt quanh **ứng viên gần đáp án nhất trong
400 ứng viên** thay vì quanh đáp án.

| nhóm | n | nền | TRẦN-POOL | | TRẦN-ORACLE | | phần pool/oracle |
|---|---|---|---|---|---|---|---|
| bộ SẠCH | 132 | 0,1903 | **0,3808** | +100,1% | 0,4308 | +126,4% | 79% |
| ├─ MỘT cảnh | 66 | 0,2767 | 0,4255 | +53,8% | 0,4409 | +59,4% | 91% |
| └─ HAI cảnh | 66 | 0,1036 | **0,3348** | **+223,1%** | 0,4211 | +306,4% | 73% |

Ba điều đọc ra, và điều thứ ba là điều quan trọng:

1. **Khoảng nền → TRẦN-POOL (+100%) là có thật và rất lớn.** Ứng viên đúng
   thường đã nằm trong pool; cái thiếu là **biết ứng viên nào**.
2. **21% khoảng oracle nằm ngoài tầm với của mọi thứ bám vào pool 400 ứng viên**
   (27% ở nhóm hai cảnh) — phần đó thuộc khâu sinh ứng viên.
3. **TRẦN-POOL vẫn là oracle.** Nó dùng đáp án để chọn tâm. Cái mua được nó
   không phải tham số phân bổ — §2–§4 vừa cho thấy chỗ đó cạn — mà là một
   **tín hiệu xếp hạng nội-video** đủ tốt để chỉ ra ứng viên nào của video đúng
   là đúng chỗ. Đó là lever ④ (PE-Core / FG-CLIP / làm mượt thời gian), và đây
   là lần đầu nó có một con số trần **thực tế** thay vì con số oracle thổi phồng:
   **+100% toàn bộ, +223% ở câu hai cảnh**, chứ không phải +126%/+306%.

---

## 6. Đọc kết quả — nói thẳng

- **Cánh cửa "chia lại ngân sách" đóng lại, lần này có số trên đúng bộ đo.**
  Họ `hybrid` âm ở cả 16 ô trên TUNE và −16,2% trên TEST (P(≤0) = 99,8%). Dòng
  trong bảng tín hiệu không cần sửa; chỉ cần thêm ghi chú là nó đã được xác nhận
  trên bộ khớp phân bố.
- **Tham số CoveragePlan hiện tại (0,02 / 30 / 6 / 5) giữ nguyên.** Tổ hợp thay
  thế tốt nhất hoà trên TEST bộ mới và âm có ý nghĩa trên bộ cũ.
- **Không được diễn giải +25,4% TUNE thành "có tiềm năng".** Nó là ước lượng
  thổi phồng của một phép chọn argmax trên 200 tổ hợp với n = 66. Bằng chứng
  trực tiếp cho việc thổi phồng: trục σ — thứ tạo ra gần hết phần tăng ấy —
  đi **ngược chiều** trên tập dữ liệu độc lập duy nhất có sẵn.
- **Nếu có thêm dữ liệu và hiệu ứng σ quay lại**, phải nhớ rằng lần này nó đã
  không giữ, và mọi con số mới phải trừ đi phần đã biết là thổi phồng.

## 7. Điều chưa biết / giới hạn của chính phép đo này

1. **n = 66 mỗi nửa là nhỏ**, và hai nửa lệch nền 69%. Phép đo này đủ sức bác
   một hiệu ứng +25% nhưng **không** đủ sức bác một hiệu ứng +5% thật. Nói
   "không có gì ở đây" là quá mạnh; nói đúng là "không có gì đủ lớn để đo được,
   và hướng duy nhất trông có lý thì âm trên bộ đối chứng".
2. **Chỉ đọc TEST một lần**, đúng giao thức — nên câu hỏi "một tổ hợp ở *giữa*
   cao nguyên σ (thay vì đỉnh) có giữ được không" **chưa được trả lời và không
   được trả lời bằng cách đọc lại TEST này**. Muốn trả lời phải sinh thêm mục
   ground truth.
3. Lưới không quét `step` của thang đuôi lấp và không quét `max_depth`. Cả hai
   chỉ tác động tới phần dòng mà coverage không tự lấp, mà trên bộ này coverage
   lấp đủ 100 dòng — nên khả năng cao là vô hại, nhưng chưa đo.
4. `nửa_cửa_sổ = 6` với `lưới = 5` cho `nua = max(1, 6//5) = 1` ô — tức trục
   "đào sâu hơn nữa" đã **chạm sàn** ngay tại tham số sản xuất. Muốn đi sâu hơn
   phải giảm `lưới` xuống dưới 5, chưa nằm trong lưới quét.
5. TRẦN-POOL (§5) dùng thang `frame_ladder` bước 10 quanh ứng viên tốt nhất, tức
   nó cũng giả định số dòng dành cho video đúng giữ nguyên như nền. Một bộ chọn
   video tốt hơn sẽ nâng trần đó lên nữa; con số +100% là trần *có điều kiện*.

---

## Tái lập

```
python -u scripts/do_phan_bo_sau.py --workers 4        # chẩn đoán + lưới + TUNE/TEST + kiểm chéo
python -u scripts/do_phan_bo_sau.py --workers 4 --chi-quet 60   # chạy lưới theo lát (máy yếu)
python -u scripts/doc_luoi_phan_bo_sau.py              # biên từng tham số, không đọc thêm dữ liệu
python -u scripts/chan_doan_dat_dong.py                # ba bảng đếm ở §1
python -u scripts/tran_dat_dong_theo_pool.py           # trần §5
python -u scripts/quet_sigma_bo_cu.py                  # trục sigma trên bộ 60 câu cũ
```

Cache lưới: `data/cache_phan_bo_sau/tune.json` (216 tổ hợp, ghi tăng dần sau mỗi
10 kết quả — tiến trình từng bị giết ngang giữa lưới, ghi ở cuối là mất trắng).
