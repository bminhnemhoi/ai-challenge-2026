# Bộ đo từ ĐỀ THẬT: có gì, tin được đến đâu, dùng thế nào

Chốt ngày 30/08/2026. Sinh bằng `scripts/thu_hoach_de_that.py`, đầu ra
`data/ground_truth_de_that.json` (đầy đủ, có bằng chứng từng mục) và
`data/ground_truth_hop_nhat.json` (60 câu GT cũ + phần dùng được).

> **Không mã video, không đáp án trong tài liệu này.** `docs/` lên GitHub công
> khai, mà đây là đề của BTC. Mã video ghi dạng `Lxx_Vnnn`, đáp án ghi
> `<đáp án>` — đúng quy ước của `docs/VI_DU_LUAN_CHUNG.md`. Bảng thật nằm ở
> `data/ground_truth_de_that.json`, và `.gitignore` dòng `data/*` đã chặn sẵn
> (chỉ ba file được whitelist). **Đừng `git add -f`.**

---

## 1. Kết quả một câu

**15 mục đạt `nguoi_kiem_chung`** trên tổng 79 câu đề thật của ba vòng:
**7 KIS, 8 Q&A, 0 TRAKE**. Trong đó 8 mục có thêm đáp án đủ tin để chấm Q&A.

**Nói thẳng: 15 mục là ÍT, và lợi ích thống kê gần như bằng không.**
Chia TUNE/TEST theo chỉ số chẵn/lẻ thì 15 mục rơi 8/7, TEST đi từ 30 lên 37 câu.
Sai số chuẩn nhị thức chỉ giảm 10%:

| n TEST | 1 sd (p=0,5) | cổng 2 sd | hệ số ngưỡng so với n=30 |
|---|---|---|---|
| **30** (hiện tại) | 9,13% | 18,26% | 1,000 |
| **37** (sau lượt này) | 8,22% | 16,44% | 0,900 |
| 45 | 7,45% | 14,91% | 0,816 |
| 60 | 6,45% | 12,91% | **0,707** |
| 120 | 4,56% | 9,13% | 0,500 |

Giả định của công thức, nói rõ để ai cũng bác được: (a) mỗi câu là một Bernoulli
độc lập cùng xác suất thành công; (b) độ đo là một **tỉ lệ trên số câu** (tỉ lệ
trả lời đúng Q&A, tỉ lệ câu có video đúng trong 100 dòng) chứ không phải điểm
R@k trung bình — với R@k thì `±` trong các báo cáo hiện tại là độ lệch giữa các
**họ hạt giống**, một nguồn nhiễu khác, và thêm câu không làm nó nhỏ đi theo
công thức này; (c) p=0,5 là trường hợp xấu nhất, p=0,85 cho sd nhỏ hơn ~30%;
(d) so sánh hai biến thể trên **cùng bộ câu** là phép đo *ghép cặp*, nên cổng
2 sd tính kiểu hai mẫu độc lập là **quá chặt** — nó chỉ đang được dùng như một
ngưỡng bảo thủ, không phải một kiểm định đúng chuẩn.

Muốn hạ ngưỡng đúng 1,41 lần như `NGHIEN_CUU_SOTA.md` §② hứa, TEST phải lên 60
câu, tức phải thu **thêm ~60 mục nữa** — gấp bốn lần lượt này. Vậy nên hãy đọc
lượt này là **hạ tầng + một cuộc kiểm toán**, không phải là đã mở khoá lever ②.

---

## 2. Phần đáng giá hơn con số: mở khung hình ra xem thì 6 "bằng chứng" sụp

Lane này không chỉ chép các file `picks_verified.txt` vào JSON. Với mọi ứng
viên có triển vọng, nó **tải khung hình gốc 1280px từ CDN và mở ra nhìn**.
Sáu mục có bằng chứng nghe rất chắc đã bị chính bước đó bác bỏ:

| câu | bằng chứng đã ghi trong file picks | mở khung ra thì thấy |
|---|---|---|
| `query-p1-19-qa` (luyện tập) | "mô hình thị giác đọc được tên anh hùng **và hai câu thơ**, tin cậy 95%" | nội thất đình, **không có tấm bia, không có câu thơ nào** |
| `query-p1-22-qa` (luyện tập) | "mô hình đọc tờ công thức: `<đáp án>`, tin cậy **100%**" | đang bóp bột vào khuôn giấy làm bánh ngọt, **không có tờ công thức** |
| `query-p1-4-kis` (sơ tuyển 1) | "[V] 1.00, hơn kẻ sau +0.80" | cảnh **phỏng vấn** nhân viên, không phải đàn sư tử trên bục gỗ cũng không phải cảnh cân |
| `query-p2-9-qa` (sơ tuyển 2) | khung neo của câu hỏi | cảnh **sơ chế** cá, chưa tới cảnh nhồi gia vị vào bụng 4 con |
| `query-p2-10-kis` (sơ tuyển 2) | frame suy từ mốc lời thoại 2:40 | mới chỉ có phi hành trong chảo, **chưa có** hai nguyên liệu của đề |
| `query-p2-24-kis` (sơ tuyển 2) | pick lượt cuối | **không có** khoảnh khắc buông hai tay ăn mừng |

Ba điều rút ra được, không cái nào đoán trước được:

1. **"VLM đọc được X, tin cậy 100%" là câu văn tự tin nhất và không đáng tin
   nhất trong kho.** Hai mục hỏng nặng nhất đều mang chữ "tin cậy 95–100%".
   Đây chính là dạng lỗi mà `docs/VI_DU_LUAN_CHUNG.md` Ca 5 đã dặn ("bắt model
   chép nguyên văn") — nhưng chép nguyên văn *rồi vẫn không mở ảnh ra đối
   chiếu* thì lời dặn không có tác dụng.
2. **Chốt đúng VIDEO không kéo theo chốt đúng FRAME.** Bốn trong sáu mục trên
   có video gần như chắc chắn đúng (tiêu đề hoặc cụm từ hiếm trong lời thoại
   chốt được) mà khung hình vẫn trượt. Luật chấm là *video đúng **và** frame
   trong cửa sổ*, nên nửa đúng vẫn là 0.
3. **Frame suy từ mốc lời thoại thường quá thô.** Khoảng cách trung vị giữa hai
   keyframe là **55 khung** (~2 giây); các cửa sổ chấm nội bộ là ±6/10/20 khung.
   Một mốc lời thoại lệch 15–20 giây (đã gặp ở `query-p1-6-kis`) là trượt chắc.

Chiều ngược lại cũng có: mở khung hình **đóng được** một mâu thuẫn. Câu
`query-p2-27-qa` bị đổi qua lại giữa hai video múa lân; mở 16 giây đầu của ứng
viên thua thì thấy đó là cảnh khiêng bao tải cạnh lò gạch, không có trụ đánh số
nào — mâu thuẫn đóng, mục được nhận. Cùng lúc, khung của ứng viên thắng cho
thấy các số 1, 3, 4, 6, 7, 8, 9 **đều nhìn thấy được**, tức đáp án đã nộp
(`1 2 8`) gần như chắc chắn sai. Nhãn truy xuất dùng được; nhãn đáp án thì không.

---

## 3. Tin được đến đâu — và chỗ KHÔNG được tin

### Cảnh báo lớn nhất, đọc trước khi dùng bất cứ con số nào

**Không mục nào ở đây được BTC xác nhận.** Ta chỉ có điểm tổng: **8,6/24** ở
vòng luyện tập và **10,0/30** ở sơ tuyển 2. Quy về trung bình mỗi câu là 0,36 và
0,33 — nghĩa là **xét theo tổng thể, phần lớn pick trong các file nguồn SAI**.
(Tiện thể: 0,33 khớp gần như đúng con số nền 0,342 của harness 60 câu — đó là
bằng chứng gián tiếp tốt nhất hiện có rằng harness nội bộ *có* hiệu chuẩn với
thực tế.)

`nguoi_kiem_chung` vì thế **không** có nghĩa "BTC chấm đúng". Nó có nghĩa hẹp
hơn nhiều, và đó là toàn bộ giá trị của nó: *bằng chứng vẫn đứng vững khi mở
khung hình ra xem.*

### Luật xếp độ tin (mã hoá trong `xep_do_tin`, không phải cảm tính)

Một mục được `nguoi_kiem_chung` khi **cả bốn** điều kiện đúng:

1. **VIDEO do một kênh độc lập với SigLIP chốt** — tiêu đề/mô tả video của BTC,
   một cụm từ hiếm trong lời thoại, chữ đọc được trên chính khung hình, hoặc
   loại trừ thủ công cả một nhóm video. *"Engine xếp hạng 1, VLM gật đầu" không
   tính* — chấm đường truy xuất trên chính đầu ra của nó là tự chấm bài mình:
   nền bị thổi lên và mọi cải tiến bị nén vào trần.
2. **Pick có ghi frame.** Nhãn chỉ-có-video không chấm được (luật chấm đòi cả
   frame trong cửa sổ). Bốn câu vòng luyện tập rơi ở đây.
3. **Lane này đã mở khung hình 1280px và thấy khớp** (`kiem_lai_lane == "khop"`).
4. **Không còn mâu thuẫn giữa các lượt chọn**, trừ khi mâu thuẫn đã được đóng
   bằng cách mở khung của ứng viên kia.

Độ tin của **đáp án** là một trường riêng (`do_tin_dap_an`), vì hai thứ hỏng độc
lập với nhau: `query-p1-15-qa` (sơ tuyển 1) có khung hình chắc chắn đúng nhưng
đáp án thì chính người soát đã ghi "độ tin thấp"; `query-p2-19-qa` thì ngược
lại — khung hình chứng minh *đáp án* chứ không phải *cảnh* đề tả.

### Ba nguồn đối chiếu bằng máy mà script tự chạy

* **`data/media-info-aic25-b1.zip`** — tiêu đề + mô tả **873/873** video. Đây là
  kênh chốt video rẻ nhất và mạnh nhất, và nó **đối chiếu lại được** (khác hẳn
  một câu văn chép tay trong file picks). Nó xác nhận 9 mục và giúp loại 1.
* **`data/captions/`** — lời thoại có mốc thời gian. **Chú ý: repo này chỉ có
  217/873 video có nội dung thật**; các file còn lại tồn tại nhưng RỖNG. Con số
  "873/873" lưu truyền trong tài liệu là số *file*, không phải số transcript.
  Hệ quả: "không tìm thấy cụm từ" ở đây **không** phải bằng chứng phủ định.
  Trong phạm vi 217 video đó, script xác nhận được ba cụm từ hiếm mà mỗi cụm chỉ
  xuất hiện ở **đúng một** video (tên một con đèo, "kim cương thô", "remember").
* **`data/metadata.json`** — kiểm khung hình có thật, tra `n` / `frame_filename`
  / `pts_time` / `cdn_url`, và ghi `keyframe_gan_nhat` + `lech_keyframe` khi
  `frame_idx` không rơi đúng keyframe nào.

---

## 4. Dùng thế nào

```bash
# sinh lại (thuần CPU, vài giây, không gọi API)
PYTHONIOENCODING=utf-8 python scripts/thu_hoach_de_that.py
PYTHONIOENCODING=utf-8 python scripts/thu_hoach_de_that.py --in-bang   # bảng từng mục
```

**Quy tắc hợp nhất — đã chọn sao cho KHÔNG phá chia TUNE/TEST đang đóng băng:**
các mục mới được **nối vào SAU** 60 câu cũ, nên chỉ số của 60 câu cũ không đổi
một cái nào; luật chẵn/lẻ vẫn nguyên. 15 mục mới rơi 8 TUNE / 7 TEST.
`data/ground_truth_hop_nhat.json` chỉ chứa mục `nguoi_kiem_chung`, **có frame**,
**không TRAKE** — tức phần chấm R@k được ngay.

**Bốn điều tuyệt đối đừng làm:**

* **Đừng chấm trên mục `suy_ra`.** Chúng nằm trong file để làm *danh sách việc
  cho người soát*, không phải để làm nhãn.
* **Đừng lọc bộ GT theo hạng của pipeline.** "Bỏ mấy câu engine không tìm ra cho
  đỡ nhiễu" là cách nhanh nhất biến bộ đo thành cái gương.
* **Đừng so con số trên bộ 75 câu với con số cũ trên bộ 60 câu.** Câu đề thật
  KHÓ hơn 60 câu mẫu (chúng là đúng những câu đã làm ta mất điểm). Phải **chạy
  lại nền** trên bộ mới rồi mới so — đúng "luật sau merge" của
  `SHIP_PHU_XAC_SUAT.md` §5.
* **Đừng dùng file hợp nhất cho thí nghiệm Q&A mà không lọc.** Mục KIS đề thật
  có `vqa_answer` rỗng; `experiment_qa_answer.py` phải lọc `vqa_answer != ""`
  (60 câu cũ + 8 câu đề thật = 68 câu có đáp án).

---

## 5. Còn thiếu gì

**TRAKE: 0 mục.** Đây là chỗ hụt đau nhất, vì `CHAN_DOAN_TRAKE.md` và hai đề
xuất trong `NGHIEN_CUU_SOTA.md` đều đang bị chặn vì thiếu đúng thứ này. Bốn
chuỗi TRAKE có trong ba vòng đều trượt cổng: hai chuỗi bị đổi video giữa các
lượt, hai chuỗi có mốc do VLM chấm mà chưa ai xem lại. **Không có đường tắt:**
chốt một mốc TRAKE đòi tua video, và bốn mốc sai thứ tự là 0 điểm cả câu. Ước
lượng thật: ~20 phút/câu × 8 câu ≈ **3 giờ người**, và đó là việc rẻ nhất trong
toàn bộ danh sách còn lại.

**Bốn câu mâu thuẫn chưa đóng** (`p2-2`, `p2-8`, `p2-18`, `p2-24`): mỗi câu có
hai ứng viên video, ít nhất một phải sai. `p2-2` là câu đáng làm trước — cả hai
ứng viên đều có lời thoại đỡ lưng (hai bản tin cùng đưa một loạt tranh tường),
nên chỉ cần mở khung là chốt được, ~10 phút.

**20/79 câu không có pick nào** — bài nộp giữ nguyên đầu ra engine. Chúng là
nguồn thu hoạch lớn nhất còn lại, nhưng cũng đắt nhất: phải soát từ đầu.

**Sáu mục "video chắc, frame sai"** ở mục 2 là nguồn rẻ nhất: video đã chốt
xong, chỉ còn tìm đúng khung trong **một** video. Với mỗi câu, dựng dải keyframe
của video đó rồi mở bằng mắt là xong — không cần API:

```bash
# đọc chữ/số ở độ phân giải gốc khi cần (đừng tin bản 512px)
python scripts/read_answer.py --video <VID> --frames <F> --neighbours 2 \
    --max-side 1900 --question "..."
```

Cộng lại: ~3 giờ cho TRAKE + ~1 giờ cho 4 câu mâu thuẫn + ~2 giờ cho 6 câu
frame-sai ≈ **6 giờ người** để đưa bộ này từ 15 lên khoảng 30 mục. Kể cả thế,
TEST mới lên ~45 câu (hệ số ngưỡng 0,82). Muốn tới 0,71 thì phải soát cả 20 câu
trắng nữa.

---

## 6. Sổ ghi các con số trong tài liệu này

| con số | lấy từ đâu |
|---|---|
| 15 / 7 KIS / 8 Q&A / 0 TRAKE | `data/ground_truth_de_that.json` → `thong_ke` |
| 79 câu đề thật, 59 có pick, 20 không | cùng file, `cau_khong_co_pick` |
| 6 mục bị bác khi mở khung | cùng file, lọc `kiem_lai_lane == "khong_khop"` |
| 4 mâu thuẫn video chưa đóng | cùng file, `mau_thuan_video` khác rỗng |
| 217/873 transcript có nội dung | script in ra ở dòng đầu |
| trung vị 55 khung giữa hai keyframe | `data/metadata.json`, 176.448 khoảng |
| 8,6/24 và 10,0/30 | `docs/KIEN_TRUC_VA_HUONG_CAI_THIEN.md` dòng 3; `docs/SHIP_PHU_XAC_SUAT.md` dòng 3 |
| bảng sai số | script in ra, công thức `sqrt(p(1-p)/n)` với p=0,5 |

Một cặp đề **trùng nhau từng byte** trong vòng sơ tuyển 1 (md5 khớp) — script
gộp lại để không đếm hai lần một câu. Nếu BTC lặp lại thói quen đó, mọi phép
đếm "số câu" ở các vòng sau đều phải khử trùng trước.
