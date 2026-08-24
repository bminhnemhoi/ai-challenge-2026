# Đợt cải tiến này đã làm gì

Tóm tắt cho cả nhóm review trước khi merge. Chi tiết kỹ thuật nằm trong docstring của từng file.

---

## Phát hiện gốc rễ

Điểm 5.8 không phải do mô hình yếu. Truy xuất của nhóm đã tốt — ba ý tưởng cải thiện encoder mà tôi thử đều **không** ăn thua (xem bảng trong README). Vấn đề nằm ở chỗ khác:

**Hệ thống đang tối ưu một độ đo không phải độ đo của cuộc thi.**

`scripts/evaluate_official_pipeline.py` chỉ so `video_id`, chưa bao giờ so `frame_idx`:

```python
rank = ranked_vids.index(tgt) + 1     # chỉ có video, không có frame
```

Nhưng luật (mục 2.1.1) yêu cầu **đúng video VÀ frame nằm trong đoạn `[s,e]`**. Nên con số "Top-1 41.67%" là độ chính xác cấp video, không phải điểm thi.

Đo cụ thể: keyframe cách nhau trung vị **55 frame**; ví dụ đoạn đáp án trong luật rộng **11 frame**. Nộp riêng index keyframe có trần **17.6%** — dù truy xuất hoàn hảo vẫn trượt 82% số câu.

---

## Ba thay đổi ăn điểm

| | Trước | Sau |
|---|---|---|
| **Thang frame** | Chỉ nộp index keyframe | Thêm số nguyên ±10, ±20… quanh keyframe tốt nhất. `frame_id` là số nguyên bất kỳ, luật không bắt phải là keyframe. |
| **Số dòng/video** | Tối đa 2, cách nhau ≥10 giây | Không giới hạn. R@k là **max**, dòng thừa không bao giờ hại. |
| **Đáp án Q&A** | Từ dòng 6 trở đi ghi "Không xác định" | Cùng một đáp án trên cả 100 dòng. Bỏ trống là vứt 3/5 thành phần điểm. |

Kết quả đo trên 60 mẫu, theo **đúng công thức chính thức**:

| | W=10 | W=20 | W=50 | W=100 | W=200 | TB |
|---|---|---|---|---|---|---|
| Mới | **0.205** | **0.259** | **0.384** | **0.465** | **0.485** | **0.360** |
| Cũ | 0.093 | 0.155 | 0.317 | 0.417 | 0.450 | 0.286 |

**+26% trung bình, +120% ở cửa sổ hẹp.** Tốt hơn ở *mọi* độ rộng đã thử — không phải đánh đổi.

---

## Sửa lỗi chặn

**Repo không import nổi.** `from dotenv import load_dotenv` không có `try/except` và không có `requirements.txt`, nên `import src` chết ngay trên máy sạch — kéo theo cả Task 1 và Task 3 dù cả hai không dùng Gemini. Đã bọc import, thêm `requirements.txt`, và cho các engine nặng import mềm.

**Lỗi import bị che.** Mẫu `try: from .x / except ImportError: from x` bắt cả lỗi thiếu thư viện bên thứ ba rồi báo lại thành `No module named 'config'` — nguyên nhân thật (`cv2`) biến mất. Giờ báo đúng nguyên nhân qua `src.task2_vqa.UNAVAILABLE`.

**Định dạng nộp bài.** Backend cũ xuất `submission_{query_id}.csv`, không có thư mục `submission/`, không zip. Luật bắt buộc cả ba. Nộp sai định dạng vẫn mất 1 trong 3 lượt.

---

## Task 3 (TRAKE)

Bản cũ lấy top-1 độc lập từng sự kiện rồi sắp xếp — trên benchmark có ground truth cấy sẵn đạt **20%**. CHRONOS (quy hoạch động) đạt **88%**, kèm 213 test riêng và bốn vòng kiểm định đối kháng.

Luật TRAKE: sai video là **0 ngay**, đúng video thì tính tỉ lệ sự kiện khớp, cửa sổ mỗi sự kiện *"thường dưới 10 frame"*. Nên mọi dòng dùng **một video duy nhất**, 100 dòng dành để thử tổ hợp frame nhiễu dần quanh nghiệm căn chỉnh.

---

## Công cụ mới

| File | Việc |
|---|---|
| `scripts/make_submission.py` | Thư mục query → `submission.zip` hợp lệ, tự kiểm tra định dạng. Dùng trong ngày thi. |
| `scripts/evaluate_official.py` | Đo **điểm chính thức**, thay cho script chỉ đo cấp video. |
| `scripts/experiment_*.py` | Các phép đo mà mọi mặc định được chọn từ đó, kể cả các kết quả âm. |
| `src/core/submission.py` | Chấm điểm, thang frame, phân bổ dòng, đóng gói + kiểm tra. |
| `src/core/kis_engine.py` | Truy xuất gọn nhẹ, dịch VI→EN 3 lớp dự phòng + cache. |
| `scripts/build_review_page.py` | Sinh `review.html` — soát cả vòng bằng mắt, chọn frame, gõ đáp án Q&A, phóng to đọc chữ trên hình. |
| `scripts/apply_picks.py` | Áp dụng **mọi** lựa chọn của người trong một lần nạp index. |
| `scripts/inspect_run.py` | Xếp hạng độ chắc chắn từng câu + cảnh báo Q&A còn trống đáp án. |
| `scripts/rerank_vlm.py` | Xếp hạng lại bằng VLM, có chế độ `--evaluate` để đo trên ground truth trước khi tin. |
| `scripts/experiment_allocation.py` | Quét cách chia 100 dòng, có sửa thiên vị ground truth. |
| `scripts/experiment_merge.py` | Đo cách kết hợp bản dịch tay. |
| `scripts/experiment_objects_rerank.py` | Đo nhãn đối tượng: theo frame vô dụng, theo video +3.3%. |
| `scripts/review_export.js` | Bộ phân bổ dòng + ghi zip chạy trong trình duyệt. |
| `scripts/verify_zip.py` | Kiểm tra bất kỳ file zip nào trước khi nộp, chạy trong một giây. |
| `src/core/objects.py` | Điểm cộng nhận dạng đối tượng ở tầng video. |
| `src/core/transcripts.py` | BM25 trên lời thoại có mốc thời gian. |
| `scripts/fetch_captions.py` | Lấy phụ đề tự động YouTube (không cần API key). |
| `scripts/search_transcripts.py` | Tìm video theo **lời nói** — công cụ cho người soát. |
| `scripts/experiment_transcripts.py` | Đo lời thoại ở tầng video (kết luận: không dùng). |
| `scripts/experiment_frame_from_speech.py` | Đo lời thoại ở tầng frame (kết luận: không dùng). |
| `src/core/ocr.py` | Đọc chữ cháy trên khung hình + đo màu chủ thể, có cache. |
| `scripts/run_ocr.py` | Chạy OCR + đo màu cho ứng viên của một vòng. |
| `scripts/search_ocr.py` | Tìm khung hình theo **chữ hiện trên hình**. |
| `src/core/colours.py` | Đặt tên màu tiếng Việt, đo trên hộp bao đối tượng. |
| `docs/DOC_NOI_DUNG_ANH.md` | Ba kênh đọc nội dung ảnh: chữ, màu, lời nói. |
| `docs/CHAN_DOAN_TRAKE.md` | Vì sao TRAKE yếu và trần lý thuyết của nó. |
| `docs/CONTEST_RUNBOOK.md` | Quy trình 3 tiếng thi. |
| `tests/` | 388 test, trong đó các khẳng định về chấm điểm lấy thẳng từ ví dụ trong luật. |

---

## Đợt rà soát đối kháng thứ hai (51 agent, 5 góc nhìn)

46 phát hiện được nêu, **16 sống sót** qua vòng phản biện (mỗi phát hiện phải chịu một agent khác được giao nhiệm vụ bác bỏ nó). Đã sửa toàn bộ 16, mỗi cái kèm test hồi quy.

Bốn cái đáng kể nhất:

**1. Toàn bộ `<script>` của `review.html` là lỗi cú pháp JavaScript.** `PAGE` là chuỗi ba nháy Python không phải raw-string, nên một `\n` viết cho JS bị Python nở thành xuống dòng thật *bên trong* một chuỗi JS nháy đơn. Trình duyệt vứt bỏ **toàn bộ** khối script khi gặp lỗi phân tích — nghĩa là bấm khung hình không ăn, phím tắt không chạy, ba nút trên thanh công cụ đều báo lỗi, mà trang vẫn hiển thị bình thường. Runbook gọi trang này là *"việc đáng giá nhất trong cả 3 tiếng"*; nó đang cho 0. Giờ có test chạy `node --check` trên chính JS sinh ra.

**2. Cái bạn nhìn không phải cái được nộp.** `make_submission` xếp hạng bằng một hàm, `review.html` bằng hàm thứ hai, `apply_picks` bằng hàm thứ ba. Ngay khi có file `.en.txt` — đúng thứ runbook bảo cả nhóm viết — ba thứ hạng này khác nhau, nên người soát duyệt một khung hình *không phải* khung hình ở dòng 1. Đây là kiểu hỏng tệ nhất: im lặng, và nó phá đúng cái phán đoán của con người mà cả vòng lặp sinh ra để thu thập. Giờ tất cả dùng chung `ranked_hits`, có test chặn mọi lời gọi `engine.search()` trực tiếp, và có kiểm tra đối chiếu 24/24 câu giữa trang duyệt và CSV.

**3. Đáp án Q&A có dấu phẩy làm hỏng dòng CSV.** `csv.writer` bọc nháy kép đúng chuẩn RFC 4180, nhưng bộ chấm tách theo dấu phẩy sẽ đọc thành 4 trường với một dấu nháy dính vào — và lần chạy sau, `split(",")[2]` cắt cụt nó thêm lần nữa. `make_submission` có lọc dấu phẩy; `apply_picks` và `pin_video` thì không. Câu `p1-19` hỏi *"Hai câu thơ đó là gì?"* — không ai gõ hai câu thơ mà không có dấu phẩy. Giờ việc lọc nằm trong `write_query_csv`, chỗ duy nhất mọi đường đi đều qua, và bộ kiểm tra **từ chối** file có dấu nháy kép.

**4. Chốt khung hình bằng mắt xong thì dòng 2–30 bị lãng phí.** Sau khi người dùng xác nhận đúng khung hình, các dòng duy nhất có thể cứu một cú trượt sát — F±10, F±20 — bị đẩy xuống hạng 31, ra khỏi cả R@5 lẫn R@20; hạng 2–30 dành cho các keyframe khác cách đó 55+ frame, không thể rơi vào cửa sổ ~10 frame. Đo trên ground truth đúng kịch bản này: **0.810 so với 0.654, tốt hơn 24%**.

Các lỗi còn lại đã sửa: `read_query_text` để codec `utf-16` nuốt file cp1258 và biến tiếng Việt thành chữ Hán (giờ chấm điểm từng cách giải mã và chọn cách ra tiếng Việt); `.en.txt` đọc bằng UTF-8 nghiêm ngặt nên một lần lưu "Unicode" từ Notepad làm cả câu rơi xuống dòng giữ chỗ = 0 điểm; `.en.txt` viết dạng văn xuôi làm câu TRAKE 4 sự kiện co thành 2 cột (giờ đối chiếu với bản gốc và cảnh báo); bộ kiểm tra mù với CRLF, với CSV thừa, và với số cột TRAKE không đồng nhất; tải dữ liệu đứt giữa chừng để lại file cụt rồi báo "HOÀN THÀNH 100%"; `download_data.py` xoá đúng file zip mà trang duyệt cần đọc.

---

## Đo lại: cách kết hợp bản dịch tay

`ranked_hits` trước đây **gộp hai danh sách ứng viên** — một từ dịch máy, một từ bản dịch tay — lấy điểm cao hơn cho mỗi frame, với lập luận "R@k là max trên tiền tố nên thêm ứng viên chỉ có lợi". Lập luận đó **sai**, vì gộp cũng *đảo thứ tự*: một frame mà bản dịch tay thích được đẩy lên trước frame mà bản dịch máy tìm ra, và 30 chỗ đầu là hữu hạn.

Đo trên cả 60 câu ground truth (đều có bản dịch tiếng Anh của người), công thức chấm chính thức, đáp án rút thăm không snap, 24 lần:

| Cách | W=6 | W=10 | W=20 | trung bình | video R@1 |
|---|---|---|---|---|---|
| chỉ dịch máy | 0.235 | 0.276 | 0.363 | 0.292 | 23/60 |
| chỉ bản dịch tay | 0.262 | 0.311 | 0.391 | 0.321 | 23/60 |
| gộp hai danh sách (cũ) | 0.248 | 0.289 | 0.378 | 0.305 | 25/60 |
| **cả hai trong một vector (nay)** | **0.274** | **0.325** | **0.413** | **0.337** | **26/60** |

Hai cách đọc thuộc về **một** vector truy vấn, nơi bộ 4-prompt đã cân trọng số sẵn, chứ không phải hai danh sách tranh nhau chỗ xếp hạng. Chạy lại: `python scripts/experiment_merge.py`.

---

## Cảnh báo: đừng tin `data/ground_truth.json` khi đo vị trí frame

**93% (56/60) frame đáp án trong file ground truth trùng khít một keyframe của index** — vì nó được tạo bằng cách chọn keyframe từ chính index này. Trung vị khoảng cách tới keyframe gần nhất là **0**.

Hậu quả: mọi phép đo vị trí frame chạy trên bộ này đều **thiên vị nặng về chỉ-nộp-keyframe**. Đáp án thật của BTC là một khoảnh khắc người ta đánh dấu trong video gốc, rơi bất kỳ đâu trong khoảng ~60 frame giữa hai keyframe.

Đo trên bản gốc (đã snap):

| Cấu hình | Điểm |
|---|---|
| `n_flat=100` (chỉ keyframe) | **0.562** ← "tốt nhất" |
| `n_flat=30` (đang dùng) | 0.526 |

Đo lại sau khi rút thăm lại khoảnh khắc thật trong đúng khe keyframe của nó (24 lần, lấy trung bình):

| Cấu hình | Điểm |
|---|---|
| `n_flat=100` (chỉ keyframe) | 0.257 |
| `n_flat=30` (đang dùng) | **0.333** ← thật sự tốt nhất |

Kết luận đảo ngược hoàn toàn: chỉ-keyframe **kém hơn 21%**, không phải tốt hơn 7%. Đây chính là kiểu sai số đã tạo ra "Top-1 41.67%" trên máy mà chỉ 5.8 trên bảng xếp hạng.

`scripts/experiment_allocation.py` mặc định chạy bản đã sửa thiên vị; cờ `--snapped` chỉ để tái hiện lại cái bẫy.

**Kết quả phụ:** đã quét `n_flat` × `depth_cost` × `step` quanh đỉnh với 24 lần rút thăm — cấu hình đang chạy (30 / 0.5 / 10) đạt 0.333 so với 0.338 của cấu hình tốt nhất tìm được (28 / 1.0 / 14). Chênh 1.5%, nằm trong nhiễu, **nên không đổi**. Hướng tinh chỉnh phân bổ đã cạn; điểm còn lại không nằm ở đây.

---

## Đã kiểm định

Bản thân code mới cũng qua một vòng kiểm định đối kháng (33 agent, mỗi phát hiện phải sống sót qua một agent khác được giao nhiệm vụ bác bỏ). Vòng đó tìm ra **7 lỗi mức critical trong chính code này**, đáng chú ý nhất:

- **Q&A luôn nộp đáp án rỗng** — gọi `answer_single_frame()` sai chữ ký, exception bị nuốt. Task 2 sẽ ăn 0 điểm mà không có dấu hiệu gì.
- **Một file query không phải UTF-8 làm hỏng cả lượt chạy** — không sinh ra zip nào.
- **Đề TRAKE đánh số `(1)…(2)…`** (đúng phrasing trong luật) bị gộp thành 1 sự kiện → sai số cột → 0 điểm.
- **Ma trận điểm TRAKE bị phá** do mask trước khi z-score.
- **File `.en.txt` bị nhận nhầm thành query**, sinh CSV thừa trong zip.

Tất cả đã sửa và có test hồi quy. Bộ kiểm tra trước khi nộp giờ **từ chối** gói bài có đáp án Q&A rỗng, thiếu file, có BOM, hoặc có dòng tiêu đề ở bất kỳ vị trí nào.

---

## Việc nhóm cần làm

1. **`GEMINI_API_KEY` — việc số một.** Ở lượt thử vừa rồi, cả 3 câu Q&A nộp với **ô đáp án trống**, tức là 3/24 câu ăn 0 điểm chắc chắn dù frame có đúng. Không có key thì cũng phải gõ tay đáp án trong `review.html`. Key miễn phí ở https://aistudio.google.com/apikey.
2. **Hôm nay:** mỗi người clone, `pip install -r requirements.txt`, chạy `pytest`, rồi chạy thử `make_submission.py` một lần. Ai chưa chạy được lệnh đó thì chưa sẵn sàng.
3. **Nạp sẵn model SigLIP-2 vào cache** — lần đầu tải ~3.5 GB, đừng để rơi vào giờ thi.
4. **Phân công soát `review.html`** trong giờ thi — chia mỗi người vài câu. Đây là nguồn điểm lớn nhất còn lại: mỗi câu kéo từ hạng 6–20 lên hạng 1 là +0.4.
5. **Phân công dịch tay sang tiếng Anh** — đáng ~8 điểm phần trăm video R@1, và dịch tự động hay bị chặn.
6. Đọc [CONTEST_RUNBOOK.md](CONTEST_RUNBOOK.md).

---

## Kênh lời thoại: dùng cho MẮT NGƯỜI, không cho bộ chấm điểm

Nhóm cung cấp **809 transcript** (5,6 triệu ký tự, có mốc thời gian từng câu); tôi lấy thêm được 217 phụ đề YouTube trước khi bị chặn IP. Gộp lại: **811/873 video**. Thiếu 62, tập trung ở **L24 (34 video — chính là nhóm múa lân)** và **L28 (24 video)**.

Đây là kênh dữ liệu MERVIN có mà ta không (đội AIC 2025 đạt 79/88 chạy ba kênh văn bản; ta chạy không kênh nào).

**Nó tìm được thứ hình ảnh mù hoàn toàn.** Tìm tay trên vòng 1:

- `p1-4` "măng tây tẩm bột chiên ngập dầu" → **`L26_V194` "MĂNG TÂY CHIÊN BIA" hạng 1**, kèm đoạn nói *"măng tây xanh... lại bột mì lên"*. Hệ thống hình ảnh xếp nó hạng 3, hạng 1 là món **xào**.
- `p1-18` "cắt nấm / cắt củ năng / cắt đậu hủ" → `L26_V012` "CỦ NĂNG OM NẤM CHAY". Trong **toàn bộ 873 video chỉ có 2 video** nhắc "củ năng". Hình ảnh không đưa nó vào top-6.

**Nhưng đo trên ground truth thì nó làm điểm TỆ ĐI** — mọi cách gộp, mọi trọng số:

| cách | trọng số tốt nhất | thay đổi |
|---|---|---|
| cộng theo video | 0,005 | −0,4% |
| lọc theo video | 0,005 | −0,1% |
| cộng theo mốc thời gian | 0,005 | ±0,0% |
| cả hai | 0,005 | −0,4% |
| **có cổng chặn theo độ quyết đoán** | 0,02 (cổng 0,2) | **+0,5% (nhiễu)** |

Lý do rõ ràng khi nhìn vào dữ liệu: **60 câu ground truth đều là mô tả cảnh NHÌN THẤY** ("xe ô tô con màu đỏ mận có cánh gió đuôi xe") — không ai *nói* ra những câu đó, nên bằng chứng lời thoại tản mát (chênh lệch 0–18%, tức nhiễu). Câu thật của vòng 1 thì khác: `p1-19` chênh 21%, `p1-18` chênh 33%.

Nghĩa là **phép đo trung thực về loại câu nó bao phủ, và im lặng về loại câu nó không bao phủ**. Không có cơ sở để đưa vào bộ chấm điểm.

Cách bố trí theo đúng bằng chứng:

1. **`scripts/search_transcripts.py`** — bạn gõ từ khoá, nó trả video + câu trích + mốc thời gian + link YouTube + số frame. Chạy trong 2 giây.
2. **Trong `review.html`**, mỗi câu hỏi có bảng 🎙 hiện chỗ lời nói khớp nhất **trong các ứng viên đã có**, từ khoá được tô đậm, bấm vào là mở video đúng lúc đó.

Một phiên bản trước còn tự đoán cả những video *chưa* có trong danh sách. Đã bỏ: nó sai cả hai chiều — bỏ sót `p1-4` (đơn vị phân biệt là **cụm** "măng tây", không phải từ đơn hiếm) và lại đưa vào những đoạn nói về robot không liên quan cho `p1-21`. Đoán sai còn tệ hơn không đoán, vì người soát phải kiểm từng dòng.

**Một lỗi test bắt được:** `best_segment` trả về đầu cửa sổ 5 câu thay vì đúng câu chứa từ khoá — lệch tới 15 giây, đủ để mở video sang một tin khác trong bản tin thời sự.

---

## Nhãn đối tượng: dùng để chọn VIDEO, tuyệt đối không để chọn FRAME

Mỗi keyframe có sẵn kết quả nhận dạng đối tượng của BTC (`Woman×2, Boy×2, Girl`). Câu hỏi là nó có nên vào **điểm số** không, chứ không chỉ hiện lên màn hình. 43/60 câu ground truth có gọi tên ít nhất một lớp mà bộ nhận dạng biết, nên tín hiệu là có sẵn.

Đo bằng `scripts/experiment_objects_rerank.py` (60 câu, công thức chính thức, đáp án không snap, **64 lần rút thăm**, 4 độ rộng cửa sổ):

| Cách cộng điểm | Điểm | video R@1 |
|---|---|---|
| không dùng đối tượng | 0.374 | 26/60 |
| cộng theo **frame**, trọng số tốt nhất | 0.375 (+0.4%) | 27/60 |
| khớp **số lượng** ("ba người"), mọi trọng số | 0.374 (**+0.0%**) | 26/60 |
| cộng theo frame, trọng số 0.05 | 0.346 (−7.4%) | 24/60 |
| **cộng theo VIDEO, trọng số 0.01** | **0.386 (+3.3%)** | 26/60 |

Chỗ tách biệt này chính là **đúng cái bẫy đã tạo ra 5.8 điểm ban đầu**: cộng điểm theo frame làm video R@1 tăng 26 → 30 nhưng **điểm thi giảm**. Vì frame chứa đúng đối tượng không phải frame gần khoảnh khắc đáp án nhất — đẩy nó lên là đá văng một frame cùng video vốn gần đáp án hơn.

Nên điểm cộng được tính **một lần cho mỗi video** (lấy khớp tốt nhất trong các frame ứng viên của video đó) rồi cộng đều cho mọi frame của nó. Video được xếp lại; thứ tự frame *bên trong* mỗi video giữ nguyên như embedding đã xếp — vì đó mới là thứ tự hiểu về thời điểm.

Đáng chú ý: cách khớp **số lượng** — nghe hợp lý nhất — hoàn toàn vô tác dụng, đúng 0.0% ở mọi trọng số.

Đã bật mặc định trong `ranked_hits`; tắt bằng `--no-objects`. Trên 24 câu vòng 1, nó đổi video hạng 1 của 3 câu. Cài đặt trong `src/core/objects.py`.

Về **chữ cháy trên hình** (tên xã trên banner, `NẤM RƠM CẮT ĐÔI`): đưa cho người, không tự động hoá. Trang duyệt có nút 🔍 phóng to để đọc ở cỡ thật. Metadata cấp video (`media-info-aic25-b1.zip`) đã đo và **làm điểm tệ đi** (R@1 43.3% → 40.0%), đã loại. OCR cấp frame chưa đo được với 25 câu mỗi vòng; nếu làm cho vòng sau thì dựng chỉ mục offline rồi đo bằng `experiment_objects_rerank.py` trước khi cho chạm vào thứ hạng.

---

## Xem video gốc ngay trong trang duyệt

`media-info` của BTC có `watch_url` cho **cả 873 video**, và `metadata.json` có `pts_time` + `fps` cho từng keyframe. Ba bất biến đã kiểm chứng trên toàn bộ 177.321 frame — `frame_filename == f"{n:03d}.jpg"`, mỗi video một fps, `pts_time * fps == frame_idx` — nên chỉ cần nhúng mảng `frame_idx` là đủ dựng lại toàn bộ dòng thời gian, giữ trang ở 676 KB thay vì vài MB.

Kết quả: bấm `▶` trên bất kỳ khung hình nào là mở YouTube tại đúng giây đó, kèm dải keyframe xung quanh để soi từng nhịp.

Thứ đáng giá nhất là **chốt frame bằng tay**: tạm dừng video đúng khoảnh khắc, đọc số giây, và trang đổi thành số frame rồi đặt lên dòng 1 với thang ±10 quanh nó. Frame đó **không cần** nằm trong danh sách hệ thống gợi ý — mà thường thì không, vì cửa sổ đáp án rộng ~10 frame còn keyframe cách nhau ~55. Với TRAKE, chốt được riêng từng sự kiện; nếu chốt nhầm sang video khác thì trang từ chối, vì các sự kiện phải cùng một video.

Trang tự đánh dấu những câu đáng bỏ công xem: toàn bộ TRAKE, toàn bộ Q&A, và câu mà hai video đầu bảng gần hoà — **11/24 câu** ở vòng thử. Nút `Chỉ hiện câu cần soi video` lọc còn đúng chừng đó.

**Hai lỗi lộ ra khi làm phần này:**

*Điểm cộng đối tượng đổi thứ tự nhưng không đổi điểm.* Mọi chỉ số tin cậy phía sau vẫn đọc điểm cũ, nên khoảng cách giữa hạng 1 và video tốt thứ nhì ra **số âm** ở 3 câu, và cờ "cần xem kỹ" im lặng ngừng bật đúng ở những câu mà điểm cộng vừa đổi ý. Giờ điểm cộng được ghi thẳng vào `score`.

*Quy tắc đánh dấu dùng phép AND.* Một câu bị coi là chắc chắn nếu 24 khung hình đầu tập trung vào một video — kể cả khi video xếp nhì chỉ kém 0.30 độ lệch chuẩn. Tập trung cao chỉ có nghĩa video đó có nhiều frame giống nhau, không có nghĩa nó đúng. Giờ chỉ xét margin.

---

## Soát và nộp thẳng từ trình duyệt

`review.html` giờ tự dựng file nộp:

- **Kéo thả** khung hình để đổi thứ hạng — vị trí #1 là dòng 1 của CSV.
- **Ô nhập đáp án** cho từng câu Q&A, tự loại `,` `"` `;` ngay khi gõ.
- **`⬇ Tải submission.zip`** dựng cả 24 CSV + zip ngay trong trình duyệt, không cần Python.

`scripts/review_export.js` là bản cài đặt **thứ hai** của bộ phân bổ dòng trong `src/core/submission.py`. Hai bản cài đặt cùng một luật thường là mùi code xấu; ở đây là cố ý — người soát đang cầm chuột, bắt họ chuyển sang terminal để thấy hiệu quả của một thao tác kéo chính là thứ khiến không ai chịu kéo.

Nó không được tin suông. Hai bộ test bắt buộc hai bản phải khớp:

- `tests/test_js_allocator.py` — chạy JS qua `node` với dữ liệu ứng viên ngẫu nhiên, đòi **từng dòng** trùng bản Python, và đòi zip do JS tạo qua được `verify_submission_zip`.
- `tests/test_page_export_matches_pipeline.py` — bóc `DATA`/`PLAN`/allocator ra khỏi `review.html` **thật**, chạy qua `node`, và đối chiếu với các CSV `make_submission` đã ghi ra đĩa.

Trước khi nộp: `python scripts/verify_zip.py <file> --queries <thư mục đề>` — một giây, không nạp index, dùng đúng bộ kiểm tra của pipeline.

Một hệ quả phải xử lý: điểm cộng đối tượng là **max trên các ứng viên của mỗi video**, nên cỡ pool khác nhau cho ra thứ hạng khác nhau. `make_submission` dùng 200 còn trang duyệt dùng 400 → trang hiển thị một video, CSV chứa video khác. Giờ có một hằng số duy nhất `RETRIEVE_TOP_N = 400` cho mọi công cụ.
