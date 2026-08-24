# Hướng dẫn phát triển — dành cho người sắp sửa mã

Tài liệu này không dạy cách *chạy* một vòng thi (cái đó ở
[`docs/CONTEST_RUNBOOK.md`](CONTEST_RUNBOOK.md) và
[`docs/HUONG_DAN_KIEM_THU.md`](HUONG_DAN_KIEM_THU.md)). Nó dành cho lúc bạn định
**mở một file `.py` ra và đổi một dòng**.

Điểm của đội đi từ 5.8 lên 8.6. Phần lớn quãng đường đó không phải nhờ thêm ý
tưởng mới, mà nhờ **gỡ bỏ những ý tưởng nghe rất có lý nhưng đo ra là âm**. Cả
tài liệu này xoay quanh đúng một câu: *nghe có lý không phải là bằng chứng*.

---

## 1. Chạy test

Chạy từ **thư mục gốc repo**. `conftest.py` chỉ có ba dòng và việc duy nhất nó
làm là chèn gốc repo vào `sys.path`, nên `pytest` gọi từ chỗ khác sẽ không
import được `src.` hay `scripts.`.

```bash
python -m pytest tests -q          # 186 test — bộ chính
python -m pytest -q                # 415 test — thêm src/task3_trake + src/task2_vqa
```

Đo lúc viết tài liệu này (2026-08-24, máy Windows 11 của dự án):

| Lệnh | Kết quả | Thời gian |
|---|---|---|
| `python -m pytest tests -q` | `186 passed, 12 warnings` | 34.9 s |
| `python -m pytest -q` | `415 passed, 12 warnings` | 35.2 s |

Hai con số gần bằng nhau vì 229 test của `src/task3_trake/tests/` chỉ dùng
NumPy trên ma trận nhỏ, chạy trong vài mili-giây.

**Bộ test KHÔNG cần chỉ mục 780 MB.** 35 giây là toàn bộ chi phí. Không có lý do
gì để bỏ qua nó trước khi commit.

12 warning là `DeprecationWarning` của Pillow ở `src/core/colours.py:116`
(`Image.Image.getdata`), không phải lỗi.

### Chạy đúng phần mình vừa sửa

```bash
python -m pytest tests/test_submission.py -q            # chấm điểm, phân bổ 100 dòng, đóng gói
python -m pytest tests/test_audit_regressions.py -q     # các lỗi từng làm mất điểm thật
python -m pytest tests/test_js_allocator.py -q          # bản Python và bản JS phải khớp
python -m pytest tests/test_transcripts.py tests/test_ocr_and_colours.py -q
python -m pytest tests/test_vlm_quota.py -q             # không cần mạng, không cần API key
```

### Bản đồ: mỗi file test canh giữ cái gì

| File | Canh giữ điều gì |
|---|---|
| `tests/test_submission.py` (34) | Công thức R-Score / R@k / Final Score đúng bằng ví dụ trong luật BTC. Test đỏ ở đây nghĩa là **ta đã lệch khỏi luật**, không phải code hỏng. |
| `tests/test_audit_regressions.py` (23) | Mỗi test là một lỗi có thật đã từng hoặc sắp làm mất điểm. Docstring từng test kể lại lỗi đó. |
| `tests/test_js_allocator.py` (16) | `scripts/review_export.js` là **bản cài đặt thứ hai** của bộ phân bổ. Test chạy nó qua `node` và đòi từng dòng trùng bản Python. |
| `tests/test_review_workflow.py` (17) | Trong đó có chốt chặn `ranked_hits` — xem mục 7. |
| `tests/test_pipeline_e2e.py` (17) | Thư mục đề vào, zip hợp lệ ra. |
| `tests/test_page_export_matches_pipeline.py` (7) | Zip do trình duyệt dựng phải trùng CSV do Python ghi. |
| `tests/test_transcripts.py` (9) | Kênh lời nói + kỷ luật "chỉ hiển thị, không chấm điểm". |
| `tests/test_ocr_and_colours.py` (17) | OCR + đo màu trên chủ thể. |
| `tests/test_vlm_quota.py` (13) | Hai sự cố quota có thật của vòng 1. |
| `tests/test_video_inspector.py` (11), `test_manual_frame_placement.py` (9), `test_pin_chain.py` (6), `test_chunking.py` (7) | Chốt frame bằng tay, cắt câu hỏi dài. |

### Test tự bỏ qua khi máy thiếu thứ gì

Đây là cố ý, đừng "sửa":

- `tests/test_js_allocator.py` — `skipif` khi không có `node`
  (`tests/test_js_allocator.py:33`).
- `tests/test_transcripts.py:96,117` — `skipif` khi máy chưa có thư mục
  transcripts.
- `tests/test_ocr_and_colours.py:156` — `skipif` khi chưa có cache OCR thật.
- `tests/test_video_inspector.py:30-31` — cần `node` **và** một `review.html`
  đã dựng.

Nếu bạn thấy `186 passed` nhưng đồng nghiệp thấy `170 passed, 16 skipped`, đó là
bình thường. Nếu bạn thấy **failed**, dừng lại.

---

## 2. Quy ước mã nguồn của repo này

Repo này có một quy ước duy nhất mà mọi quy ước khác chỉ là hệ quả:

> **Docstring giải thích LÝ DO. Comment ghi lại BẪY. Không cái nào mô tả lại mã.**

Mã đã tự nói nó làm gì rồi. Thứ mã không nói được là *tại sao nó không làm cách
kia* — và trong dự án này, "cách kia" thường là cách nghe hợp lý hơn.

### 2.1 Docstring đầu module = bản án đã tuyên

Mở `src/core/objects.py` dòng 1-27. Nó không mô tả class nào cả, nó dán một bảng
số:

```
    baseline                                  0.374   video R@1 26/60
    per-frame bonus, best weight              0.375   (+0.4%, noise)
    per-frame count match, any weight         0.374   (+0.0%, completely inert)
    per-frame bonus, weight 0.05              0.346   (-7.4%)
    PER-VIDEO bonus, weight 0.01              0.386   (+3.3%)
```

rồi giải thích tại sao chia theo video lại thắng chia theo frame. Đó là mẫu.
Cùng kiểu ở `src/core/submission.py:1-31`, `src/core/kis_engine.py:1-24`,
`src/core/transcripts.py:1-24`, `src/core/ocr.py:1-18`.

Khi bạn viết module mới, docstring phải trả lời được ba câu:

1. **Vì sao thứ này tồn tại** — vấn đề nào không có nó thì không giải được.
2. **Đã thử gì khác và ra sao** — kèm số.
3. **Đừng làm lại điều gì** — viết thẳng, để người sau không mất công.

### 2.2 Comment ghi bẫy, đặt ngay tại chỗ bẫy

Không gom vào cuối file, không viết vào Slack. Ví dụ chuẩn ở
`scripts/apply_picks.py:229-231`:

```python
# ranked_hits, not eng.search(..., query_en=en) — the two rank
# differently, and only ranked_hits is what make_submission wrote
# and what the operator saw on the review page
hits = ranked_hits(eng, probe or text, en)
```

Ba dòng comment đó đắt hơn cả hàm. Người đọc lần sau sẽ định "đơn giản hoá" nó
thành `eng.search`, và comment là thứ duy nhất chặn được.

### 2.3 Hằng số đi kèm lý do bằng `#:`

```python
#: below this the recogniser is usually hallucinating on texture, not reading
MIN_CONF = 0.35                                    # src/core/ocr.py:35-36

#: measured optimum; the curve is flat from 0.005 to 0.02 and turns sharply
#: negative by 0.05, so this sits in the middle of the plateau
DEFAULT_WEIGHT = 0.01                              # src/core/objects.py:41-43

#: how much a keyframe gains for standing out from the ones beside it in time.
#: Measured on the ground truth: +2.2% at 0.01, +1.0% at 0.002
PEAK_WEIGHT = 0.01                                 # scripts/make_submission.py:261-264
```

Một hằng số không có lý do là một hằng số người sau sẽ đổi bừa. Nếu con số đến
từ một lần quét, **ghi tên script đã quét ra nó**.

### 2.4 Tên test là một câu khẳng định

Không `test_allocate_1`, không `test_edge_case`. Đọc tên test phải biết nó bảo
vệ điều gì:

```
test_extra_wrong_rows_can_never_reduce_the_score
test_moving_a_hit_within_a_bucket_is_worth_nothing
test_a_scene_setting_first_line_is_not_counted_as_an_event
test_colour_is_measured_on_the_subject_not_the_stage
test_per_minute_429_is_not_treated_as_the_daily_quota
test_cost_note_shouts_when_nothing_was_judged
```

Và docstring của test kể lại **sự cố có thật** đã sinh ra nó. Xem
`tests/test_audit_regressions.py:56-63`: nó chép nguyên hình dạng đề TRAKE thật
của vòng 1 và giải thích tại sao thiếu nhánh đó thì ghi 5 cột cho câu 4 sự kiện.

### 2.5 Khuôn mẫu một script trong `scripts/`

Mọi script đều mở đầu giống nhau, và thứ tự này quan trọng:

```python
"""Câu hỏi mà script này trả lời, và vì sao đoán mò không đủ.

    python scripts/ten_script.py --co-gi-do
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.make_submission import ranked_hits  # noqa: E402
```

Ba điểm bắt buộc:

- **`safe_console()` gọi TRƯỚC mọi lệnh in.** Console Windows mặc định là
  cp1252; in một chữ `ạ` là `UnicodeEncodeError`. Lý do nó nghiêm trọng chứ
  không chỉ khó chịu, chép từ `scripts/_console.py:5-7`: *cú crash đó rơi vào
  **giữa** lúc ghi một số CSV và lúc đóng gói zip, để lại một bài nộp nửa vời.*
- **`__doc__` làm `description` của argparse** —
  `argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)`.
  Nhờ vậy `--help` chính là docstring, không có bản mô tả thứ hai để lệch nhau.
- **`# noqa: E402`** cho các import sau `sys.path.insert` — cố ý, đừng dọn.

### 2.6 Nuốt exception thì phải nói rõ vì sao

Repo này nuốt lỗi ở khá nhiều chỗ, nhưng mỗi chỗ đều kèm một lý do một dòng:

```python
except Exception as exc:  # noqa: BLE001 - a tie-break must never break a run
except Exception as exc:  # noqa: BLE001 - a nicety must not break the page
except Exception:  # noqa: BLE001 - a pipe or a captured buffer; nothing to fix
except OSError: pass       # a full disk must not fail the round
```

Quy tắc: **một bước làm đẹp không bao giờ được phép làm hỏng cả lượt chạy.** Hệ
quả rõ nhất ở `scripts/make_submission.py:554-562` — một câu hỏi bị crash vẫn
được ghi một dòng giữ chỗ, vì CSV rỗng sẽ khiến bộ kiểm tra chặn **cả gói nộp**,
còn một dòng sai thì không bao giờ làm tụt R@k nào.

Nhưng có một ngoại lệ tuyệt đối, xem mục 7: **im lặng khi KHÔNG LÀM ĐƯỢC GÌ là
cấm.** So sánh `cost_note()` ở `src/core/vlm.py:365-382` — nó hét lên
`!! KHONG CHAM DUOC KHUNG HINH NAO`.

### 2.7 Một việc — một chỗ làm

`sanitise_field` (`src/core/submission.py:416-431`) là điểm **duy nhất** mọi CSV
đi qua. Docstring của nó (dòng 426-429) nói thẳng đây là sửa một lỗi có thật:
`make_submission` có strip dấu phẩy, nhưng hai đường thao tác tay (`apply_picks`,
`pin_video`) thì không — nên cùng một đáp án an toàn khi pipeline ghi ra mà hỏng
khi người sửa lại.

Tương tự `RETRIEVE_TOP_N = 400` (`scripts/make_submission.py:55-62`): một hằng số
cho mọi công cụ, vì khi `make_submission` dùng 200 còn trang duyệt dùng 400 thì
trang hiển thị một video còn CSV chứa video khác.

---

## 3. Thêm một kênh thông tin mới

"Kênh" ở đây nghĩa là một nguồn tín hiệu mà mô hình nhúng ảnh–văn bản không biểu
diễn được. Hiện có bốn: **nhãn đối tượng** (`src/core/objects.py`), **lời nói**
(`transcripts.py`), **chữ trên hình** (`ocr.py`), **màu chủ thể** (`colours.py`).
Làm theo đúng khuôn của chúng.

### Bước 1 — Viết lõi ở `src/core/<kenh>.py`

Docstring mở đầu phải nêu **một câu hỏi cụ thể mà kênh khác không trả lời
được**. Mẫu ở `src/core/transcripts.py:4-8`: câu `query-p1-21` hỏi về nghiên cứu
ở một đại học tại Lausanne, mô hình thị giác xếp nhầm một clip đời sống lên
trước vì *"một người dẫn chương trình ngồi bàn thì trông y hệt một người dẫn
chương trình ngồi bàn"* — còn một phát thanh viên **nói** "Lausanne" thì không
có vấn đề đó.

### Bước 2 — Cache ra đĩa, mỗi video một file

Quy ước: `data/<kenh>/<video_id>.json`. Lý do chép từ `src/core/ocr.py:44-49`:
một lượt chạy dở dang không mất gì, và vòng sau dùng lại được toàn bộ. Flush
định kỳ (OCR flush mỗi 100 khung).

Nếu kênh cần tải ảnh: **tải song song, xử lý tuần tự**. Mạng là nút cổ chai;
reader EasyOCR không thread-safe và reader thứ hai chỉ tốn gấp đôi bộ nhớ mà
không nhanh hơn trên CPU (`src/core/ocr.py:108-112`).

### Bước 3 — Script sinh dữ liệu, dùng lại đúng ứng viên của bài nộp

Xem `scripts/run_ocr.py:27-34`. Nó import `ranked_hits`, `detect_task`,
`split_events`, `split_qa` từ `make_submission` **thay vì tự chọn khung hình**.
Nhờ vậy tập khung hình được xử lý trùng khít danh sách ứng viên mà bài nộp đang
đề cử. Tự chọn thì trang duyệt sẽ hiện dữ liệu cho những khung không có trên
trang, và thiếu dữ liệu cho những khung có.

### Bước 4 — Công cụ tra cứu cho người soát

`scripts/search_transcripts.py`, `scripts/search_ocr.py`. Đây là chỗ **con người
cung cấp từ khoá**. Docstring `search_transcripts.py:16-17` nói rõ vì sao nó là
công cụ chứ không phải một tầng trong pipeline: *"The operator knows what to
search for; the machine does not."*

Kèm luôn: khi không tìm thấy gì thì gợi ý bước tiếp theo, đừng chỉ in danh sách
rỗng (`scripts/search_ocr.py:117-119`).

### Bước 5 — Cắm vào `build_review_page.py`, sau một cờ `--no-<kenh>`

Và **bọc try/except**:

```python
except Exception as exc:  # noqa: BLE001 - a nicety must not break the page
    print(f"  ! bo qua loi thoai ({type(exc).__name__}: {exc})")
    tx = None
```

Thiếu dữ liệu thì in một dòng bảo người dùng chạy script nào để có
(`build_review_page.py:1116`: `"OCR: chua doc khung hinh nao (chay scripts/run_ocr.py)"`).

### Bước 6 — Viết `scripts/experiment_<kenh>.py` (mục 4) và ĐO

**Trước khi** kênh được phép chạm vào bất kỳ thứ hạng nào.

### Bước 7 — Đặt kênh đúng chỗ mà bằng chứng cho phép

Đây là bước hay bị bỏ qua nhất, và là bước quyết định.

| Kết quả đo | Kênh được đặt ở đâu |
|---|---|
| Dương rõ, ổn định qua nhiều độ rộng cửa sổ | Được vào `ranked_hits` — **nhưng cộng theo VIDEO**, xem mục 6 |
| Trong nhiễu, hoặc âm | Chỉ lên `review.html` + công cụ tra cứu tay |
| Chưa đo được | Chỉ lên `review.html` + công cụ tra cứu tay |

Ba trong bốn kênh hiện tại nằm ở hàng dưới. Điều đó **không** có nghĩa chúng vô
dụng: OCR phát hiện bài nộp `query-p1-19` đang ở **sai video**, và tìm lời thoại
bằng tay tìm ra `MĂNG TÂY CHIÊN BIA` cùng `CỦ NĂNG OM NẤM CHAY` mà xếp hạng hình
ảnh bỏ sót hoàn toàn. Nghĩa là: phép đo trên 60 câu ground truth **trung thực về
loại câu nó bao phủ và im lặng về loại câu nó không bao phủ** — 60 câu đó đều là
mô tả cảnh nhìn thấy ("xe ô tô con màu đỏ mận có cánh gió đuôi xe"), thứ không ai
nói ra thành lời (`docs/WHAT_CHANGED.md:205-209`).

Đừng gộp vì "nghe có lý", và cũng đừng vứt vì phép đo âm. Đưa nó tới mắt người.

### Bước 8 — Test

Xem `tests/test_ocr_and_colours.py:62-82` để biết một test tốt trông thế nào: nó
**dựng một ảnh giả để chứng minh luận điểm thiết kế**, chứ không chỉ kiểm hàm
chạy được. Ảnh có sân khấu đỏ và một con lân vàng nhỏ; đo cả ảnh ra "đỏ", đo trên
hộp bao ra "vàng". Đó chính là lý do `colours.py` tồn tại, viết thành assert.

Thêm một test trên **dữ liệu thật** để bắt lúc thư viện bên dưới đổi hành vi:
`test_the_real_cache_actually_read_vietnamese_text` đòi tỉ lệ khung có chữ đọc
được phải trên 20%; sập nghĩa là bộ nhận dạng hoặc gói ngôn ngữ đã đổi.

---

## 4. Thêm một script `experiment_*`

Khuôn mẫu đã ổn định qua 14 script. Đọc `scripts/experiment_allocation.py` —
nó ngắn nhất và có đủ mọi thành phần.

### Docstring: đặt câu hỏi, rồi nói vì sao không được đoán

So sánh ba mở đầu, chúng cố tình nói cùng một điều:

- `experiment_objects_rerank.py:1-11` — "Nó *trông* rõ ràng là hữu ích… nhưng một
  lớp detector là công cụ rất cùn. `Person` bắn trên 57% keyframe."
- `experiment_transcripts.py:1-12` — "Transcript hiển nhiên *chứa* thứ mà pixel
  không có… Nhưng đó không phải là *giúp*, và dự án này đã đo được một tín hiệu
  (nhãn đối tượng, theo frame) **làm tăng video Recall@1 trong khi LÀM TỤT
  điểm**."
- `experiment_vlm.py:1-8` — "Mọi thứ khác đã đo… đều ra âm hoặc trong nhiễu, và
  cái nào trước đó cũng có vẻ hiển nhiên hữu ích. Nên cái này bị đối xử y hệt."

### Bốn quy tắc đo, không được bỏ quy tắc nào

**(1) Dùng công thức chấm chính thức, không dùng chỉ số thay thế.**

```python
from src.core.submission import MAX_ROWS, AllocationPlan, Candidate, \
    allocate_hybrid_rows, final_score, r_score_kis
```

Không tự viết lại công thức trong script thí nghiệm. `final_score` đã có test
tái hiện đúng ví dụ 0.74 của BTC (`tests/test_submission.py:73-79`).

**(2) Rút thăm lại khoảnh khắc thật — KHÔNG dùng ground truth thô.**

Chép đoạn này (`scripts/experiment_allocation.py:91-104`):

```python
def truth_frames(seed: int):
    """One plausible answer key: the marked instant, un-snapped."""
    rng = np.random.default_rng(seed)
    out = []
    for g, _ in cached:
        a = kf_by_video[g["video_id"]]
        f = int(g["frame_idx"])
        i = int(np.argmin(np.abs(a - f)))
        lo = (a[i] + a[i - 1]) // 2 if i > 0 else a[i] - 30
        hi = (a[i] + a[i + 1]) // 2 if i + 1 < len(a) else a[i] + 30
        out.append(int(rng.integers(lo, max(lo + 1, hi))))
    return out
```

Lý do ở mục 6. Số lần rút thăm: 8 là mức tối thiểu, 24 là mức đáng tin, 64 khi
hiệu ứng nhỏ.

**(3) Báo cáo trên cả DẢI độ rộng cửa sổ, không phải một con số.**

BTC không công bố cửa sổ `[s,e]` rộng bao nhiêu, và chiến lược tối ưu **đảo
chiều** theo bề rộng đó. `evaluate_official.py:46` dùng
`WINDOWS = (10, 20, 50, 100, 200)`; `experiment_allocation.py` mặc định
`--windows 6,10,20`. Một cải tiến chỉ thắng ở một độ rộng thì không phải cải
tiến.

**(4) Gọi `ranked_hits`, không gọi `engine.search`.** Xem mục 7.

### In ra cả số âm

`experiment_vlm.py:7-8`: *"the honest negative reported if that is what comes
out."* Nếu script của bạn chỉ in ra khi kết quả đẹp thì nó không phải thí nghiệm.

Và in kèm **cấu hình đang chạy** để so sánh, như
`experiment_allocation.py:144-152` làm — nó import `DEFAULT_N_FLAT` /
`DEFAULT_DEPTH_COST` từ `make_submission` rồi in cả `best` lẫn `shipping default`
cùng phần trăm chênh lệch. Nhờ vậy mới thấy được kết luận "chênh 1.5%, nằm trong
nhiễu, nên không đổi".

### Đăng ký vào chốt chặn nếu script có truy hồi

Nếu script của bạn gọi bộ truy hồi, thêm tên nó vào danh sách `must_agree` trong
`tests/test_review_workflow.py:100-108`.

---

## 5. Vì sao mọi thay đổi phải đo bằng ground truth trước khi tin

Đây là bài học đã trả giá bằng điểm số.

### Chuyện đã xảy ra

`data/ground_truth.json` có 60 câu. **93% (56/60) frame đáp án trong đó trùng
khít một keyframe của chỉ mục** — vì file được tạo bằng cách chọn keyframe từ
chính chỉ mục này. Trung vị khoảng cách tới keyframe gần nhất là **0**
(`docs/WHAT_CHANGED.md:132-136`).

Đáp án thật của BTC không như thế. Nó là một khoảnh khắc người ta đánh dấu trên
dòng thời gian video gốc, rơi bất kỳ đâu trong khe giữa hai keyframe.

Hậu quả đo được:

| Cấu hình | Chấm bằng GT thô (đã snap) | Chấm sau khi rút thăm lại, 24 lần |
|---|---|---|
| `n_flat=100` — chỉ nộp keyframe | **0.562** ← "tốt nhất" | 0.257 |
| `n_flat=30` — đang dùng | 0.526 | **0.333** ← thật sự tốt nhất |

Kết luận **đảo ngược hoàn toàn**: chỉ-keyframe kém hơn 21%, không phải tốt hơn
7%. `docs/WHAT_CHANGED.md:152` ghi thẳng đây chính là kiểu sai số đã tạo ra
*"Top-1 41.67%"* trên máy mà chỉ **5.8** trên bảng xếp hạng.

Cờ `--snapped` của `experiment_allocation.py` tồn tại **chỉ để tái hiện cái
bẫy**, không phải để dùng.

### Bảng những thứ "nghe có lý" đã đo ra âm

Ghi lại để không ai làm lại:

| Ý tưởng | Kết quả đo |
|---|---|
| Nhãn đối tượng cộng theo **frame**, trọng số 0.05 | **−7.4%** (0.374 → 0.346) |
| Nhãn đối tượng khớp **số lượng** ("ba người") | **+0.0%**, trơ hoàn toàn ở mọi trọng số |
| Metadata cấp video (tiêu đề/mô tả) | KIS R@1 **43.3% → 40.0%** |
| Chuẩn hoá điểm theo từng video | 0.405 → **0.276** |
| Làm mượt điểm theo thời gian (±1, ±2 keyframe) | 0.405 → **0.382** |
| Lời thoại gộp cấp video | **−0.1% … −23%** |
| Lời thoại gộp theo mốc thời gian từng frame | **−1.5% … −20%** |
| Gộp hai danh sách ứng viên (máy dịch + người dịch) | 0.337 → **0.305** |
| VLM trọng số 0.10 / 0.20 | **−2.1%** / **−5.7%** |
| Giới hạn 2 frame mỗi video | Tệ hơn ở mọi độ rộng cửa sổ đã thử |

Nguồn: `src/core/objects.py:8-16`, `src/core/kis_engine.py:15-23`,
`scripts/make_submission.py:238-246`, `scripts/vlm_rerank_run.py:3-11`,
`docs/DOC_NOI_DUNG_ANH.md:72-80`, `README.md`.

### Thước đo SAI mà cả đội từng dùng

**`video R@1` không phải là điểm thi.** Ba lần riêng biệt, cùng một hiện tượng:

- Nhãn đối tượng theo frame: video R@1 **26 → 30**, điểm thi **giảm**.
- VLM ở trọng số 0.10 / 0.20: video R@1 **25 → 29**, điểm thi **giảm**.
- Metadata video: điểm thi trong nhiễu, R@1 **giảm** 43.3% → 40.0%.

Lý do luôn giống nhau: **frame chứa đúng đối tượng / được VLM thích nhất không
phải frame gần khoảnh khắc đáp án nhất.** Đẩy nó lên là đá văng một frame cùng
video vốn gần sự thật hơn.

`scripts/evaluate_official.py:141` in `video R@1` kèm đúng chữ
`(for reference only)`. Hãy đọc đúng như thế.

### Trước khi tin một cải tiến, hỏi bốn câu

1. Đo bằng **công thức chính thức** hay bằng chỉ số cấp video?
2. Đáp án đã **rút thăm lại** chưa, hay còn snap vào keyframe?
3. Có thắng ở **mọi độ rộng cửa sổ** không, hay chỉ trung bình?
4. Cỡ mẫu bao nhiêu? Con số +7.3% của VLM đo trên 20 câu là nhiễu; con số đáng
   tin là **+3.3%** trên đủ 60 câu (`scripts/vlm_rerank_run.py:19-20`).

---

## 6. Cảnh báo — những cái bẫy đã làm mất điểm thật

Đọc hết mục này trước khi sửa bất cứ thứ gì.

### 6.1 Không bao giờ gọi `engine.search()` trong công cụ vận hành

`engine.search` chỉ trả thứ hạng thô của embedding. `ranked_hits`
(`scripts/make_submission.py:227-256`) mới là thứ hạng thật, vì nó còn chạy
`_peak_preference` rồi `_object_boost`.

Trước đây `make_submission` xếp bằng một hàm, `review.html` bằng hàm thứ hai,
`apply_picks` bằng hàm thứ ba. Ngay khi có một file `.en.txt` — **đúng thứ mà
runbook bảo cả nhóm viết** — ba thứ hạng đó khác nhau, nên người soát duyệt một
khung hình **không phải** khung hình ở dòng 1 của bài nộp. Hỏng kiểu này im lặng
tuyệt đối và nó phá đúng cái phán đoán của con người mà cả vòng lặp sinh ra để
thu thập.

Đã có chốt chặn: `tests/test_review_workflow.py:95-128` đọc mã nguồn 7 script,
đòi mỗi file phải chứa chuỗi `ranked_hits`, và fail nếu tìm thấy `eng.search(`
hoặc `engine.search(` ngoài comment.

Cũng **đừng đổi `top_n`**. Điểm cộng đối tượng là max trên ứng viên của mỗi
video, nên pool khác cho ra thứ tự video khác — chính là bug 200-vs-400. Dùng
`RETRIEVE_TOP_N` cho tất cả.

### 6.2 Đừng cộng điểm theo FRAME. Cộng theo VIDEO.

Xem mục 5. Đây là cái bẫy đã tạo ra 5.8 điểm ban đầu, và nó đã lặp lại với VLM.

Thứ tự frame **bên trong** một video phải giữ nguyên như embedding đã xếp — đó
mới là thứ tự biết về thời điểm (`src/core/objects.py:23-26`).

### 6.3 Sửa `src/core/submission.py` thì phải sửa cả `scripts/review_export.js`

Có **hai** bản cài đặt của bộ phân bổ dòng. Bản JS chạy trong trình duyệt để
trang duyệt tự dựng được file nộp. Đây là cố ý
(`docs/WHAT_CHANGED.md:273`: bắt người soát đang cầm chuột phải chuyển sang
terminal chính là thứ khiến không ai chịu kéo thả).

Sau mỗi lần sửa bộ phân bổ: `python -m pytest tests/test_js_allocator.py -q`.

### 6.4 Giá trị mặc định trong dataclass KHÔNG phải giá trị đang chạy

| Chỗ | Mặc định | Đang nộp thật |
|---|---|---|
| `AllocationPlan.depth_cost` (`submission.py:232`) | `0.75` | **0.5** (`DEFAULT_DEPTH_COST`, `make_submission.py:54`) |
| `allocate_hybrid_rows(n_flat=…)` (`submission.py:279`) | `20` | **30** (`DEFAULT_N_FLAT`, `make_submission.py:53`) |

Viết `AllocationPlan()` trần trong script thí nghiệm là bạn đang đo một cấu hình
**khác** với cấu hình đã nộp. Luôn viết rõ:

```python
plan = AllocationPlan(breadth_cost=1.0, depth_cost=0.5, step=10)
```

Đó là cách `evaluate_official.py:97`, `loss_breakdown.py:67`, `app.py:297` đang
làm.

### 6.5 Hai chỗ trong repo ghi số sai / trỏ sai — đã kiểm chứng

- **`AllocationPlan` docstring trỏ tới `scripts/tune_allocation.py`. File đó
  KHÔNG tồn tại** (đã `ls`). Bộ quét thật là
  `scripts/experiment_allocation.py`.
- **Khoảng cách keyframe: `submission.py:26` ghi 55, `experiment_allocation.py:11`
  ghi 62.** Tôi đã đo lại trên `data/metadata.json` (177.321 keyframe / 873
  video) ngày 2026-08-24:

  | Cách tính | Kết quả |
  |---|---|
  | Trung vị mọi khoảng cách gộp chung, toàn kho | **55.0** |
  | Trung bình toàn kho | 69.0 |
  | Trung vị của các trung vị theo từng video | 48.0 |
  | Trung vị trên riêng 60 video trong ground truth | 71.0 |

  Con số **55 là đúng** (trung vị gộp toàn kho). Con số **62 không tái hiện được**
  ở bất kỳ cách tính nào trong bốn cách trên — đừng chép nó vào báo cáo.

Nếu sửa hai chỗ này, sửa bằng Edit/Write, **không** bằng PowerShell (mục 8).

### 6.6 Số cột của file TRAKE chính là số sự kiện

Sai số cột thì **cả câu ăn 0 điểm**, và sai kiểu này **vô hình** vì mọi kiểm tra
khác đều pass trên một dòng 2 cột. Nếu `make_submission.py` in cảnh báo về số sự
kiện, xử lý ngay.

Đặc biệt: file `.en.txt` viết dạng văn xuôi rơi qua mọi bộ tách và biến câu 4 sự
kiện thành file 2 cột. Code có đối chiếu chéo bản gốc và giữ bản cho ra nhiều sự
kiện hơn (`make_submission.py:427-440`), nhưng nó chỉ cứu được khi bản gốc tách
đúng. Viết `.en.txt` cho câu TRAKE bằng đúng các dấu `E1:` / `(1)` / `;`.

### 6.7 Một lượt "không làm được gì" không được đọc giống "làm rồi mà không thấy gì"

Sự cố có thật vòng 1: hết quota Gemini → mọi lời gọi trả 429 → `_ask_batch` nuốt
lỗi trả `[]` → `score()` trả `{}` → pipeline vui vẻ đóng gói một bài nộp hoàn
chỉnh mà **VLM chưa hề nhìn khung hình nào**
(`tests/test_vlm_quota.py:5-9`).

Nay `cost_note()` (`src/core/vlm.py:365-382`) in
`!! KHONG CHAM DUOC KHUNG HINH NAO — dung coi ket qua nay la da xet.`
**Đọc dòng cuối của `cost_note()` trước khi tin bất kỳ kết quả VLM nào.**

Bản vá đầu tiên cho lỗi đó lại sinh ra lỗi thứ hai: nó coi **mọi** 429 là hết
quota ngày, nên một model chỉ chạy hơi nhanh cũng bị gạch tên đến hết vòng. Nay
`_is_daily_quota()` (`vlm.py:70-81`) đọc `quotaId` để phân biệt. Cả hai đều có
test ghim bằng chuỗi lỗi thật của Google.

Cùng nguyên tắc: khung hình tải không được thì **bỏ qua và đếm**, không chấm 0 —
chấm 0 sẽ khiến một mạng chết đọc thành "model đã nhìn và nói không".

Và `test_cost_note_stays_quiet_on_a_healthy_run` đòi lượt chạy tốt **không được
có `!!` nào** — để dấu `!!` giữ được sức nặng. Đừng thêm `!!` cho cảnh báo vặt.

### 6.8 Những thứ đã thử và ĐỪNG thêm lại

- Làm mượt theo thời gian, chuẩn hoá điểm theo video, cap "tối đa 2 frame mỗi
  video" (`src/core/kis_engine.py:15-23`).
- Đổi `combine` sang `'max'` trong `query_similarities` — nó thưởng frame khớp
  **một mệnh đề bất kỳ**, và đó là cách "một cái giá sách" thắng chính cái cảnh
  mà mệnh đề đó thuộc về (`kis_engine.py:344-346`).
- Gộp hai danh sách ứng viên thay vì gộp vào một vector truy vấn
  (`make_submission.py:227-246`). Lưu ý `merged_hits` chỉ còn là **alias** của
  `ranked_hits`; đừng suy hành vi từ cái tên.
- Tinh chỉnh `n_flat` / `depth_cost` / `step`. Hướng này **đã cạn**: quét quanh
  đỉnh với 24 lần rút thăm cho 0.333 (đang chạy) so với 0.338 (tốt nhất tìm
  được) — chênh 1.5%, trong nhiễu (`docs/WHAT_CHANGED.md:156`).

### 6.9 Bẫy định dạng bài nộp

Mỗi vòng chỉ **3 lượt nộp**, chỉ lượt cuối được tính, và một lần bị từ chối vì
sai định dạng vẫn tiêu mất một lượt.

- **Đừng mở CSV bằng Notepad hay Excel rồi Save** — chúng ghi lại kết dòng CRLF
  và ký tự carriage return trở thành một phần của trường cuối. Từng có lần bộ
  kiểm tra bỏ lọt vì mọi kiểm tra bên dưới chạy trên `splitlines()`, hàm này nuốt
  `\r`. Nay đã có kiểm tra `b"\r\n" in raw` (`submission.py:537-546`).
- **Zip các file CSV rời sẽ bị từ chối** — archive bắt buộc chứa thư mục tên đúng
  là `submission/` bên trong.
- **File CSV thừa từ vòng trước** nằm lại trong `csv/` sẽ bị đóng gói theo. Dùng
  `--queries` khi chạy `verify_zip.py` (để truyền `expect_names`), hoặc dùng thư
  mục `--out` mới tinh.
- **Đáp án Q&A rỗng = 0 điểm** theo luật 2.1.2, dù frame đúng đến đâu. Cờ
  `--allow-blank-answers` chỉ dành cho lần nộp thử định dạng có chủ ý.
- Chạy `python scripts/verify_zip.py <file> --queries <thư mục đề>` trước **mỗi**
  lần upload, kể cả với zip do `review.html` tự dựng trong trình duyệt — đó là
  một đường đi tới BTC không hề đi qua `make_submission.py`. Không cần nạp chỉ
  mục, trả lời dưới 1 giây.

---

## 7. Hai điều tuyệt đối không được làm

### 7.1 KHÔNG BAO GIỜ sửa file bằng PowerShell `Get-Content` / `Set-Content`

Trên máy Windows 11 / PowerShell 5.1 của dự án, lệnh này **làm hỏng file**:

```powershell
# TUYỆT ĐỐI KHÔNG
(Get-Content f -Raw).Replace(...) | Set-Content f -Encoding utf8
```

**Vì sao:** console là UTF-8 nhưng ANSI code page là Windows-1252, và các cmdlet
`*-Content` đi theo ANSI code page chứ không theo console. `Get-Content` giải mã
byte UTF-8 thành Windows-1252 nên `—` biến thành `â€”` và dấu tiếng Việt thành
mojibake; rồi `-Encoding utf8` ghi thêm một BOM vào đầu file.

**Việc này đã xảy ra hai lần trong chính dự án này** — một lần trên docs, một lần
trên `scripts/experiment_objects_rerank.py`.

**Làm thế nào cho đúng:** dùng công cụ Edit/Write, hoặc một heredoc Python qua
Bash với `encoding="utf-8", newline="\n"` ghi tường minh.

**Đọc file để kiểm tra cũng dính đúng cái bẫy đó:** `Get-Content` sẽ **hiển thị**
UTF-8 đúng thành mojibake. Trước khi kết luận một file bị hỏng, hãy kiểm byte
bằng Python.

**Sửa file đã hỏng:**

```python
text.lstrip("﻿").encode("cp1252", "replace").decode("utf-8", "replace")
```

Liên quan: `write_query_csv` cố tình ghi UTF-8 **không BOM**, kết dòng LF, không
header (`submission.py:434-447`), và `tests/test_submission.py:258-264` khẳng
định byte ghi ra đúng bằng `b"L01_V001,505\nL01_V001,515\n"`. Một lần chạm
PowerShell vào file đó là hỏng cả ba điều kiện cùng lúc.

### 7.2 `.env` KHÔNG BAO GIỜ được commit

`.env` chứa `GEMINI_API_KEY`. Repo này công khai.

Đã kiểm chứng lúc viết tài liệu (2026-08-24):

```
$ git check-ignore -v .env
.gitignore:28:.env      .env

$ git ls-files | grep -iE "\.env|round1/"
(không có kết quả)
```

`.gitignore:28-30` chặn `.env`, `.env.*`, `.env.local`. Đừng thêm ngoại lệ, đừng
`git add -f`.

Khoá được nạp bởi `load_env()` (`src/core/vlm.py:84-93`), đọc cả
`<data>/../.env` lẫn `./.env`. Nó dùng `os.environ.setdefault`, nên **biến môi
trường đã có sẽ THẮNG file `.env`** — nếu thấy dùng nhầm khoá, kiểm tra biến môi
trường của shell trước khi nghi ngờ `.env`.

**Cùng mức nhạy cảm, cũng không commit và cũng không dán ra ngoài:**
`round1/`, `round2/`, `round3/`, `vongthi*/` (bị `.gitignore:72-75` chặn). Chúng
chứa đề của BTC, 100 dòng đáp án mỗi câu, và `picks_verified.txt` — tức là đáp án
kèm lý luận. `round1/sharp_questions.json` mô tả rất chi tiết cảnh đúng của từng
câu vòng 1; **coi nó ngang với đáp án**, đừng dán vào slide hay chat nhóm mở.

Phần lý luận thì đáng chia sẻ, phần đáp án thì không — nên bản dùng để dạy nằm ở
[`docs/VI_DU_LUAN_CHUNG.md`](VI_DU_LUAN_CHUNG.md) với đáp án đã gỡ bỏ.

---

## 8. Danh sách kiểm trước khi commit

```bash
python -m pytest tests -q                    # phải thấy 186 passed
python -m pytest tests/test_js_allocator.py -q   # nếu đã sửa bộ phân bổ
git status                                   # .env và round*/ KHÔNG được xuất hiện
```

Và tự hỏi:

- [ ] Docstring có nói **lý do**, hay chỉ mô tả lại mã?
- [ ] Hằng số mới có `#:` kèm lý do và tên script đã đo ra nó chưa?
- [ ] Thay đổi này đã được đo bằng **công thức chính thức** trên **đáp án đã rút
      thăm lại**, qua **nhiều độ rộng cửa sổ** chưa?
- [ ] Nếu nó cộng điểm: cộng theo **video** hay theo **frame**?
- [ ] Nếu nó sửa thứ hạng: có gọi `ranked_hits` không?
- [ ] Nếu nó có thể thất bại im lặng: có chỗ nào **hét lên** khi thất bại chưa?
- [ ] Lỗi vừa sửa đã có một test mang tên là một câu khẳng định chưa?
