# Phương pháp — vì sao cách làm này ăn điểm

Tài liệu này giải thích **tư tưởng**, không phải cách chạy lệnh. Nếu bạn cần quy
trình bấm nút trong ba tiếng thi thì đọc [CONTEST_RUNBOOK.md](CONTEST_RUNBOOK.md);
nếu cần cách phân xử khi ba kênh nói ba điều khác nhau thì đọc
[VI_DU_LUAN_CHUNG.md](VI_DU_LUAN_CHUNG.md).

Mọi con số dưới đây đều **đã có trong mã nguồn hoặc trong docs của repo**, kèm
đường dẫn file và số dòng để bạn tự mở ra kiểm chứng. Chỗ nào tôi không chắc,
tôi nói rõ là không chắc.

---

## 0. Tóm tắt trong một đoạn

Điểm đi từ 5.8 lên 8.6 **không phải vì mô hình mạnh hơn**. Encoder không đổi.
Ba thứ đổi:

1. **Đọc lại luật chấm và cài đúng nó bằng mã** — hoá ra hệ thống cũ đang tối ưu
   một độ đo hoàn toàn khác với độ đo BTC dùng.
2. **Vứt bỏ "độ chính xác cấp video" làm kim chỉ nam** — đó là chỉ số đánh lừa,
   và repo này có **ba** phép đo độc lập chứng minh nó tăng trong khi điểm thi giảm.
3. **Ba kênh đọc nội dung độc lập nhau** (hình ảnh / lời thoại / mắt VLM), cộng
   với kỹ thuật "câu hỏi phân biệt" để tách *bối cảnh* khỏi *khoảnh khắc*.

Phần còn lại của tài liệu là ba ý đó, chi tiết.

---

## 1. Luật chấm thật sự chấm cái gì

Ba công thức dưới đây được chép nguyên văn từ tài liệu luật của BTC vào đầu
`src/core/submission.py` (dòng 8–13), và cài đặt nằm ngay bên dưới:

```
Textual KIS   R-Score(r) = I(v = GT_v  và  id ∈ [s, e])
Q&A           R-Score(r) = I(v = GT_v  và  id ∈ [s, e]  và  answer = GT_a)
TRAKE         R-Score(r) = 0 nếu v ≠ GT_v, ngược lại (1/N)·Σ_j I(id_j ∈ [s_j, e_j])

              R@k         = MAX trên k dòng ĐẦU TIÊN
              Final Score = (1/5)·Σ R@k, k ∈ {1, 5, 20, 50, 100}
```

Đọc kỹ từng vế, vì mỗi vế đẻ ra một quyết định thiết kế:

### 1.1 "Đúng video **VÀ** frame nằm trong cửa sổ"

Đây là chữ **VÀ**, không phải **HOẶC**. `r_score_kis`
(`src/core/submission.py:60-65`) trả `0.0` ngay khi `video_id != gt_video`, và
chỉ trả `1.0` khi `s <= frame_id <= e` — **biên đóng**, cả hai đầu đều tính.

Chọn đúng video là **điều kiện cần**, không phải điều kiện đủ. Đây là câu quan
trọng nhất trong cả tài liệu này, và mục 2 sẽ cho thấy nó đã bị làm ngược suốt
một thời gian dài.

### 1.2 Q&A: đáp án gắn với **từng dòng**, không phải với cả câu

`r_score_qa` (`src/core/submission.py:68-86`) đòi đủ điều kiện KIS **cộng thêm**
đáp án khớp ngữ nghĩa. Điều dễ bỏ sót: chỉ số dòng `i` đi kèm cả frame lẫn đáp
án — mỗi dòng được chấm bằng **đáp án của chính nó**.

Hệ quả rất cụ thể: nếu bạn ghi đáp án ở dòng 1–5 rồi bỏ trống (hoặc ghi "Không
xác định") từ dòng 6 trở đi, bạn **vứt luôn R@20, R@50, R@100** — ba trong năm
thành phần của Final Score — mà chẳng đổi lại được gì. Vì vậy `build_qa_rows`
ghi **cùng một chuỗi đáp án lên cả 100 dòng** (`scripts/make_submission.py:381-410`),
và bộ kiểm tra trước khi nộp **từ chối** gói bài có ô đáp án rỗng trừ khi bạn cố
ý truyền cờ `--allow-blank-answers`.

BTC khớp đáp án **theo nghĩa**, chấp nhận cả tiếng Việt lẫn tiếng Anh — nên
`answer_matcher` được thiết kế tiêm được từ ngoài vào. Bản mặc định
(`_default_answer_match`, dòng 142–153) cố ý **nới tay** (khớp chuỗi con hai
chiều): nó chỉ có thể *đếm thừa*, không bao giờ *đếm thiếu*, và khi phải sai thì
đếm thừa là chiều an toàn hơn để quyết định một thay đổi có giúp ích hay không.

### 1.3 TRAKE: bất đối xứng cố ý

`r_score_trake` (`src/core/submission.py:89-110`): sai video là **0 tuyệt đối**,
nhưng đúng video thì các sự kiện được **chấm điểm từng phần**. Sự bất đối xứng
này quyết định toàn bộ chiến lược TRAKE: một video ứng viên đáng được nộp **nhiều
lần** với nhiều tổ hợp frame khác nhau, và **không đáng nộp cho bất kỳ video nào
khác** (`allocate_trake_rows`, dòng 330–399).

Thêm một cái bẫy chết người và **vô hình**: cột thứ `j` của file CSV giữ frame
của sự kiện `j`, nên **số cột chính là số sự kiện**. Một bộ tách đề bị rơi về 1
sự kiện sẽ sinh file 2 cột cho câu 4 sự kiện và ăn 0 điểm — trong khi *mọi* kiểm
tra khác vẫn pass trên dòng 2 cột đó. Đã có kiểm tra riêng cho việc này
(`verify_submission_zip`, dòng 567–584).

### 1.4 R@k là **MAX trên tiền tố** — không phải tổng, không phải trung bình

`r_at_k` (`src/core/submission.py:113-116`) chỉ đơn giản là `max(row_scores[:k])`.
Từ đó suy ra hai hệ quả mà không ai đoán ra nếu chỉ liếc qua luật.

**Hệ quả A — dòng sai không bao giờ làm mất điểm.**
Về mặt toán học, một dòng sai không thể kéo bất kỳ `R@k` nào xuống. Cái giá duy
nhất của một dòng thừa là **chỗ hạng nó chiếm** — mà hạng 51–100 vốn chẳng có gì
để mất. Vì vậy: **luôn nộp đủ 100 dòng, không bao giờ cắt bớt vì "dòng sau chắc sai"**.
Khẳng định này được ghim bằng test
`test_extra_wrong_rows_can_never_reduce_the_score` (`tests/test_submission.py:82`).

**Hệ quả B — điểm là hàm bậc thang, chỉ có 5 mốc đáng tiền.**
`score_bucket` (`src/core/submission.py:124-133`) tồn tại chỉ để suy luận việc này:
đẩy một cú trúng từ **hạng 15 lên hạng 10 được 0 điểm**; đẩy từ **hạng 6 lên hạng 5
được +0.2**. Test `test_moving_a_hit_within_a_bucket_is_worth_nothing`
(`tests/test_submission.py:99-102`) viết đúng hai câu đó thành assert.

Biết điều này thì bạn biết công sức soát tay nên đổ vào đâu: kéo một câu từ hạng
6–20 lên hạng 1 đáng **+0.4**; loay hoay trong khoảng hạng 21–49 đáng **0**.

### 1.5 `frame_id` là **số nguyên bất kỳ**, không bắt buộc là keyframe

Đây là hệ quả ăn nhiều điểm nhất, và nó hoàn toàn nằm trong luật chứ không phải
mẹo: `frame_id` là một số nguyên bất kỳ trong video gốc — **không có chỗ nào bắt
nó phải là một keyframe đã trích**.

Đối chiếu hai con số (`src/core/submission.py:22-30`):

- Keyframe trong kho này cách nhau **trung vị ~55 frame**.
- Ví dụ đã giải trong luật dùng cửa sổ `[500, 510]`, tức **11 frame**; luật cũng
  nói cửa sổ TRAKE *"thường rất ngắn, thông thường dưới 10 frame"*.

Nghĩa là: **nộp thuần chỉ số keyframe là tự chặn trần điểm ở khoảng 17.6%** — dù
truy xuất có hoàn hảo đến đâu (`README.md:15`). Rải một **thang số nguyên**
±10, ±20… quanh keyframe đã chọn đưa trần đó lên **~90%** và không tốn gì khác
ngoài vài chỗ hạng rẻ tiền.

> ⚠️ **Hai con số vênh nhau trong repo.** `src/core/submission.py:26` ghi trung vị
> **55 frame**; `scripts/experiment_allocation.py` ghi trung vị **62 frame**. Tôi
> không xác định được số nào đo sau, hay đo trên tập nào. Nếu ai cần con số chính
> xác cho báo cáo thì **phải đo lại**, đừng chép bừa từ đây.

Thang frame được sinh bởi `frame_ladder` (`src/core/submission.py:161-192`) và
sắp theo **khoảng cách tăng dần**, không phải theo thứ tự tăng của frame id:

```
frame_ladder(1000, 5, step=10) == [1000, 990, 1010, 980, 1020]
```

Thứ tự này quan trọng vì ngân sách hạng **luôn** bị cắt cụt ở đâu đó. Sắp theo
khoảng cách bảo đảm cắt ở bất kỳ chỗ nào cũng giữ lại đúng những id khả dĩ nhất,
thay vì giữ toàn nửa bên trái.

`step` mặc định bằng **10**, đúng bằng `ASSUMED_WINDOW_FRAMES` (dòng 52), và
**không được vượt quá** bề rộng cửa sổ giả định — nếu step lớn hơn, thang chừa ra
những khe mà cửa sổ đáp án có thể lọt trọn vào giữa hai nấc. Test
`test_ladder_step_covers_any_window_at_least_that_wide`
(`tests/test_submission.py:117-124`) quét mọi cửa sổ rộng `step+1` trong tầm với
của thang và đòi cửa sổ nào cũng phải trúng.

---

## 2. "Độ chính xác cấp video" là chỉ số đánh lừa

Đây là bài học đắt nhất của cả dự án. Nếu bạn chỉ nhớ một mục trong tài liệu này,
nhớ mục này.

### 2.1 Cái sai gốc

`scripts/evaluate_official_pipeline.py` — script đánh giá cũ — **chưa bao giờ so
`frame_idx`**. Nó chỉ làm đúng một việc:

```python
rank = ranked_vids.index(tgt) + 1     # chỉ có video, không có frame
```

Nên con số **"Top-1 41.67%"** mà cả nhóm nhìn vào là **độ chính xác cấp video**,
không phải điểm thi (`docs/WHAT_CHANGED.md:13-19`). Trên bảng xếp hạng nó ra 5.8.

Khoảng cách giữa hai con số đó không phải nhiễu — nó là **hai đại lượng khác nhau**.

### 2.2 Điểm đang mất ở đâu (phép đo quan trọng nhất)

Đo trên 60 câu ground truth (`docs/KIEN_TRUC_VA_HUONG_CAI_THIEN.md:80-95`), bằng
cách hỏi "nếu sửa hoàn hảo một thứ thì được bao nhiêu":

| nếu sửa hoàn hảo | điểm | tăng |
|---|---|---|
| hiện tại | 0,345 | — |
| xếp hạng **video** hoàn hảo | 0,487 | +41% |
| **vị trí frame** hoàn hảo | **0,740** | **+115%** |
| cả hai | 1,000 | — |

**60% phần điểm lấy lại được nằm ở VỊ TRÍ FRAME, chỉ 22% ở xếp hạng video.**

Điều này lật ngược trực giác. Rất nhiều câu có video đúng ở **hạng 1** mà vẫn ăn
**0 điểm**, vì không frame nào rơi vào cửa sổ.

Mổ xẻ 22 câu trượt (`docs/KIEN_TRUC_VA_HUONG_CAI_THIEN.md:97-103`):

| nguyên nhân | số câu |
|---|---|
| keyframe đúng **có** trong ứng viên, **thang không vươn tới** | **14** |
| video không có trong 100 dòng | 5 |
| keyframe đúng không có trong 400 ứng viên | 3 |

Ba nhóm này cần ba cách sửa **hoàn toàn khác nhau** — và đó chính là lý do
`scripts/loss_breakdown.py` tồn tại. Docstring của nó nói thẳng: *"Optimising
before measuring is how this project got to 5.8 with a good encoder."*

### 2.3 Ba lần đo độc lập: video R@1 **tăng** trong khi điểm thi **giảm**

Đây không phải một tai nạn lẻ. Nó lặp lại ba lần, ở ba hệ thống con khác nhau:

**Lần 1 — điểm cộng nhãn đối tượng theo từng frame.**
`src/core/objects.py:17-22` và `docs/WHAT_CHANGED.md:227-235`:

| Cách cộng điểm | Điểm | video R@1 |
|---|---|---|
| không dùng đối tượng | 0.374 | 26/60 |
| cộng theo **frame**, trọng số tốt nhất | 0.375 (+0.4%) | 27/60 |
| khớp **số lượng** ("ba người"), mọi trọng số | 0.374 (**+0.0%**) | 26/60 |
| cộng theo frame, trọng số 0.05 | 0.346 (−7.4%) | 24/60 |
| **cộng theo VIDEO, trọng số 0.01** | **0.386 (+3.3%)** | 26/60 |

Bản per-frame kéo video R@1 từ **26 lên 30** trong khi **làm tụt điểm thi**.

**Lần 2 và 3 — trọng số VLM.**
`scripts/vlm_rerank_run.py:3-17`:

| cấu hình | Điểm | video R@1 |
|---|---|---|
| baseline | 0.387 | 25/60 |
| **per-video, w=0.02** | **0.400 (+3.3%)** | — |
| per-frame, w=0.01 | 0.398 (+2.8%) | — |
| per-video, w=0.10 | 0.379 (−2.1%) | **29/60** |
| per-video, w=0.20 | 0.365 (−5.7%) | **29/60** |

Hai dòng cuối là **đúng cái bẫy đó thu nhỏ**: video R@1 leo từ 25 lên 29 trong
khi điểm thi rơi.

**Cơ chế chung, và nó luôn giống nhau:** frame chứa đối tượng khớp — hay frame mà
VLM thích nhất — **không phải** frame gần khoảnh khắc đáp án nhất. Đẩy nó lên là
**đá văng** một frame cùng video vốn gần sự thật hơn.

Vì vậy điểm cộng nhãn đối tượng được tính **một lần cho mỗi video** (lấy khớp tốt
nhất trong các frame ứng viên của video đó) rồi **cộng đều cho mọi frame** của nó
(`src/core/objects.py:161-205`). Video được xếp lại; thứ tự frame *bên trong* mỗi
video giữ **y nguyên** như embedding đã xếp — nhờ sắp xếp ổn định theo vị trí gốc
(dòng 202–204) — **vì đó mới là thứ tự hiểu về thời điểm**.

### 2.4 Cái bẫy thứ hai: ground truth đã bị "snap" về keyframe

Ngay cả khi bạn đã chấm bằng công thức chính thức, vẫn còn một chỗ để tự lừa mình.

**93% (56/60) frame đáp án trong `data/ground_truth.json` trùng khít một keyframe
của index** — vì file đó được tạo bằng cách chọn keyframe từ chính index này.
Trung vị khoảng cách tới keyframe gần nhất là **0** (`docs/WHAT_CHANGED.md:132-136`).

Đáp án thật của BTC là một khoảnh khắc người ta đánh dấu **trên trục thời gian
video gốc**, rơi bất kỳ đâu trong khe ~60 frame giữa hai keyframe. Chấm với bản đã
snap thì một dòng đặt lệch keyframe **chỉ có thể sai**, nên chiều sâu (thang frame)
đo ra vô dụng, và bộ quét sẽ khuyên bạn đi thuần breadth.

Kết luận **đảo ngược hoàn toàn** sau khi rút thăm lại khoảnh khắc thật trong đúng
khe của nó (24 lần, lấy trung bình — `docs/WHAT_CHANGED.md:138-152`):

| Cấu hình | GT thô (đã snap) | GT rút thăm lại |
|---|---|---|
| `n_flat=100` (chỉ keyframe) | **0.562** ← "tốt nhất" | 0.257 |
| `n_flat=30` (đang dùng) | 0.526 | **0.333** ← thật sự tốt nhất |

Chỉ-keyframe **kém hơn 21%**, không phải tốt hơn 7%. Tài liệu ghi thẳng: đây đúng
là kiểu sai số đã tạo ra *"Top-1 41.67%"* trên máy mà chỉ **5.8** trên bảng.

### 2.5 Kỷ luật đo lường — quy tắc bốn điểm

Từ ba mục trên, mọi thí nghiệm trong repo giờ **bắt buộc** tuân theo bốn điều, và
bạn sẽ thấy đúng bốn điều này lặp lại trong docstring của từng `experiment_*.py`:

1. **Công thức chính thức** (`final_score` trong `src/core/submission.py`), không
   phải video R@1, không phải bất kỳ proxy nào.
2. **Đáp án không snap** — rút thăm lại khoảnh khắc thật trong khe keyframe.
3. **Nhiều lần rút thăm** (24 hoặc 64 lần), lấy trung bình, và báo trên **cả dải**
   độ rộng cửa sổ vì BTC không công bố nó.
4. **Báo số âm nếu nó âm.** Docstring `scripts/experiment_vlm.py` viết nguyên văn:
   *"the honest negative reported if that is what comes out."*

Riêng điểm 3, `scripts/experiment_strategies.py` (PART 3, dòng 317–321) còn đi xa
hơn: nó xếp hạng các chiến lược theo **trường hợp xấu nhất** trên dải bề rộng khả
dĩ, không theo trung bình — *"A strategy that wins at W=200 and scores 0.10 at
W=10 is a gamble."*

> **Cảnh báo vận hành:** cờ `--snapped` của `scripts/experiment_allocation.py` chỉ
> tồn tại để **tái hiện lại cái bẫy**, không phải để dùng. Đừng bao giờ chấm điểm
> bộ phân bổ bằng ground truth thô.

---

## 3. Ba kênh đọc độc lập

Hệ thống cũ quyết định từ **một** modality: một keyframe *trông* như thế nào. Nó
hỏng theo một kiểu rất cụ thể và đo được.

| kênh | trả lời được câu hỏi | mù ở đâu |
|---|---|---|
| **Hình ảnh** — SigLIP-2 trên 177.321 keyframe | *"khung nào trông giống câu mô tả nhất"* | màu sắc, số lượng, hành động, thứ tự — tất cả bị nén vào **một vector 1152 chiều** |
| **Lời thoại** — BM25 trên transcript có mốc thời gian | *"phóng sự này nói về cái gì"* | video không có lời thoại; từ đồng âm |
| **Mắt VLM** — Gemini xem từng khung hình | gần như mọi thứ, **nếu hỏi đúng cách** | đắt, có hạn ngạch, và đếm rất tệ |

Chúng độc lập vì chúng **hỏng vì những lý do khác nhau** — đó mới là định nghĩa
hữu ích của "độc lập" ở đây, chứ không phải vì chúng dùng ba mô hình khác nhau.

### 3.1 Kênh hình ảnh — SigLIP-2

Lõi là `KISEngine` (`src/core/kis_engine.py`), chạy trên
`google/siglip2-so400m-patch14-384`. Bốn quyết định đáng nói:

**Bộ 4-prompt có trọng số** (`PROMPT_WEIGHTS`, dòng 39): EN 0.45 / VI 0.35 /
`"a high quality video keyframe of ..."` 0.10 / `"a photo of ..."` 0.10, gộp
thành **một** vector truy vấn (`query_vector`, dòng 304–322). Đo trên 60 mẫu:
**hơn bản chỉ-tiếng-Anh 6.6 điểm video R@1** (docstring dòng 11–14).

**Dịch Việt → Anh trước khi mã hoá**, có cache đĩa và không bao giờ để lỗi dịch
làm hỏng lượt chạy. Corpus caption mà SigLIP-2 học là tiếng Anh. Đo được: video
R@1 đi từ **35.0% lên 43.3%** (dòng 163–166). Khi dịch tự động bị chặn (endpoint
miễn phí bị rate-limit đúng lúc tải cao), engine trả lại nguyên văn tiếng Việt và
in cảnh báo — **cách cứu là viết tay file `.en.txt`**.

**Cắt mô tả dài thành nhiều đoạn ≤ 58 token** (`chunk_text`, dòng 253–302). Text
tower của SigLIP-2 chỉ nhìn **64 token** và **âm thầm** bỏ phần dư. Đo trên vòng 1:
**13 trên 24 câu vượt quá giới hạn**. Và thứ rơi khỏi đuôi lại đúng là chi tiết
phân biệt, vì người tả cảnh luôn nói bối cảnh trước, đặc điểm nhận dạng sau
(*"...người chạy thứ hai đội mũ đỏ"*).

Các đoạn được kết hợp bằng **trung bình**, không phải max (`query_similarities`,
dòng 334–362). Lý do ghi ngay tại chỗ (dòng 344–346): `mean` thưởng frame thoả
**toàn bộ** mô tả; `max` thưởng frame chỉ khớp **một mệnh đề bất kỳ** — và đó
chính là cách "một cái giá sách" thắng chính cái cảnh mà mệnh đề đó thuộc về.

**Xếp hạng phẳng theo frame, không giới hạn số frame mỗi video.** Bản retriever cũ
giữ tối đa 2 frame/video và ép khoảng cách 10 giây — tối ưu cho một độ đo *cấp
video*. Nhưng điểm chính thức cần một frame rơi **trong cửa sổ**, mà R@k là max,
nên thêm frame của một video đã có trong danh sách là **bảo hiểm miễn phí**
(docstring dòng 15–20).

Bọc ngoài lõi là **`ranked_hits`** (`scripts/make_submission.py:227-256`) — lớp
xếp hạng chuẩn mà **mọi công cụ bắt buộc phải gọi**. Nó chạy thêm hai bước:

- `_peak_preference` (dòng 267–324): ưu tiên keyframe là **cực đại cục bộ** trên
  trục thời gian của chính video đó. Lý do: *một khoảnh khắc là một đỉnh, một cảnh
  là một cao nguyên.* Đo được **+2.2%** ở trọng số 0.01.
- `_object_boost` (dòng 333–368): điểm cộng nhãn đối tượng **theo video** (+3.3%),
  đã giải thích ở mục 2.3.

> ⚠️ **Đừng bao giờ gọi `engine.search()` trực tiếp trong công cụ vận hành.** Đã
> từng có lúc `make_submission` xếp hạng bằng một hàm, `review.html` bằng hàm thứ
> hai, `apply_picks` bằng hàm thứ ba — và ngay khi có một file `.en.txt` (đúng thứ
> runbook bảo cả nhóm viết) ba thứ hạng đó khác nhau. Người soát duyệt một khung
> hình **không phải** khung hình ở dòng 1 của bài nộp. Hỏng kiểu này **im lặng
> tuyệt đối** và nó phá đúng cái phán đoán của con người mà cả vòng lặp sinh ra để
> thu thập (`docs/WHAT_CHANGED.md:105`). Nay đã có test đọc mã nguồn 7 script và
> fail nếu thấy `eng.search(` ngoài comment
> (`tests/test_review_workflow.py:95-128`).

### 3.2 Kênh lời thoại — BM25 trên transcript

`src/core/transcripts.py`. Hai tính chất làm nó đáng giá hơn metadata thường
(docstring dòng 13–18):

- Nó **có mốc thời gian**, nên một cú khớp định vị được *một khoảnh khắc*, không
  chỉ một video — đúng thứ TRAKE cần và đúng thứ tiêu đề không bao giờ cho được.
- Nó **mang danh từ riêng**. "Lausanne", "củ năng", "Nguyễn Trung Trực" là vô hình
  với một bộ mã hoá ảnh và hoàn toàn không nhập nhằng trong văn bản.

Dùng **BM25 trên unigram + bigram**, không phải một mô hình nhúng thứ hai. Tiếng
Việt viết theo âm tiết, nên chỉ unigram thì "măng tây" khớp cả "phương Tây",
"hành tây" — **bigram mới là đơn vị phân biệt**. ASR tiếng Việt nhiễu nhưng danh
từ thường đúng, và trùng khớp từ vựng chính xác đúng là tín hiệu mà bộ mã hoá dày
vứt đi. Đổi lại: không tải model, không GPU, **mili-giây mỗi truy vấn**.

Chi tiết đáng học ở `best_segment` (dòng 179–210): việc chấm điểm chạy trên **cửa
sổ trượt 5 cue**, nhưng mốc thời gian trả về là của **đúng cue chứa từ khoá**, chứ
không phải đầu cửa sổ. Cửa sổ 5 cue ≈ **15 giây** — trong một bản tin thời sự, 15
giây trước là **một tin khác hẳn**. Comment tại dòng 199–202 giải thích, và test
`tests/test_transcripts.py:63-74` ghim nó bằng `assert at == 60.0, "the passage,
not the start of the video"`. Đây là **lỗi do test bắt được**
(`docs/WHAT_CHANGED.md:217`).

**Nhưng kênh này KHÔNG nằm trong đường chấm điểm.** Đo mọi cách gộp, mọi trọng số
(`docs/WHAT_CHANGED.md:196-208`):

| cách gộp | trọng số tốt nhất | thay đổi |
|---|---|---|
| cộng theo video | 0,005 | −0,4% |
| lọc theo video | 0,005 | −0,1% |
| cộng theo mốc thời gian | 0,005 | ±0,0% |
| cả hai | 0,005 | −0,4% |
| có cổng chặn theo độ quyết đoán | 0,02 | +0,5% (nhiễu) |

Lý do lộ ra khi nhìn dữ liệu: **60 câu ground truth đều là mô tả cảnh NHÌN THẤY**
(*"xe ô tô con màu đỏ mận có cánh gió đuôi xe"*) — không ai *nói* ra những câu đó.
Nên phép đo **trung thực về loại câu nó bao phủ, và im lặng về loại câu nó không
bao phủ**. Không đủ cơ sở để đưa vào bộ chấm.

Đổi lại, nó bắt được thứ hình ảnh mù hoàn toàn. Tìm tay trên vòng 1
(`docs/WHAT_CHANGED.md:191-194`): câu về măng tây chiên → transcript đưa đúng video
lên **hạng 1** trong khi hệ thống hình ảnh xếp nó hạng 3 (hạng 1 là món *xào*);
câu về "củ năng" → trong **toàn bộ 873 video chỉ có 2 video** nhắc cụm này, và
hình ảnh không đưa video đúng vào top-6.

Nên cách bố trí đúng theo bằng chứng là: kênh này đi tới **mắt người**, qua
`scripts/search_transcripts.py` (bạn gõ từ khoá, nó trả video + đoạn trích + mốc
thời gian + link mở đúng lúc) và qua bảng 🎙 trong `review.html`. Bảng đó **chỉ
xét các video đã có trong danh sách ứng viên** — một phiên bản trước có tự đoán cả
video chưa có, và nó sai theo cả hai chiều, nên đã bỏ: *đoán sai còn tệ hơn không
đoán, vì người soát phải kiểm từng dòng.*

### 3.3 Kênh mắt VLM — Gemini

`src/core/vlm.py`. Docstring đầu file (dòng 1–17) nói thẳng lý do tồn tại:

> SigLIP-2 chấm một câu hỏi với một keyframe bằng **đúng một phép nhân vô hướng**.
> Đó là lý do 177.321 khung hình tìm được tức thì, và cũng là lý do nó **không phân
> biệt nổi con lân vàng với con lân đỏ**: màu, số lượng, hành động, thứ tự đều sập
> vào một vector 1152 chiều.

Đo trên chính dữ liệu của dự án: với câu hỏi múa lân, **3/4 sự kiện có điểm DƯỚI
ngưỡng nhiễu** — keyframe tốt nhất của embedding không nổi bật hơn một khung hình
ngẫu nhiên. Cùng câu hỏi đó, `gemini-3.5-flash-lite` gọi **đúng màu con lân ở cả 6
video ứng viên** và chấm video đúng **100** so với 0–30 cho phần còn lại, trong
**4,5 giây** cho 6 ảnh.

Phân vai rất rõ: **VLM là bộ CHẤM LẠI, không phải bộ TÌM KIẾM.** SigLIP-2 lọc ra
danh sách ngắn, VLM đọc danh sách ngắn đó. VLM không bao giờ quét cả 177.321 khung.

Prompt (`src/core/vlm.py:204-217`) liệt kê **tường minh** đúng bốn thứ mà một
vector embedding làm sập vào nhau:

```
- MÀU SẮC nêu trong câu hỏi phải khớp (áo vàng ≠ áo đỏ)
- SỐ LƯỢNG phải khớp (bốn em nhỏ ≠ hai em nhỏ)
- HÀNH ĐỘNG/TƯ THẾ phải đúng khoảnh khắc được mô tả
- CHỮ hiện trên hình, nếu câu hỏi có nhắc, phải khớp
Cảnh cùng chủ đề nhưng sai chi tiết thì cho 20-40, không cho điểm cao.
```

Câu cuối cùng là câu quan trọng nhất: nếu không ép, model sẽ chấm **theo chủ đề**
và cả loạt video cùng chủ đề đều cao điểm.

Trọng số VLM cộng vào điểm truy xuất chỉ **0.02**, và **0.02 là TRẦN chứ không
phải điểm khởi đầu** (`scripts/vlm_rerank_run.py:73-75`). Xem lại bảng ở mục 2.3:
w=0.10 ra −2,1%, w=0.20 ra −5,7%. Để VLM nói to lên thì nó **đè lên cảm nhận về
thời điểm** của embedding và điểm tụt.

> ⚠️ **Đọc dòng cuối của `cost_note()` trước khi tin kết quả.** Nếu thấy
> `!! KHONG CHAM DUOC KHUNG HINH NAO` thì VLM **chưa hề nhìn cái gì** — bản nộp
> sinh ra từ lượt đó là bản *chưa được xét*, không phải bản *đã xét và không thấy
> gì*. Đây là sự cố có thật của vòng 1: quota cạn → mọi lời gọi 429 → `_ask_batch`
> nuốt lỗi trả `[]` → `score()` trả `{}` → đường ống vui vẻ đóng gói thành một bản
> nộp hoàn chỉnh, **không một lời phàn nàn**. Nay đã có 3 test ghim
> (`tests/test_vlm_quota.py:68-75, 122-133, 145-158`), và test ngược
> `test_cost_note_stays_quiet_on_a_healthy_run` đòi lượt chạy tốt **không được có
> `!!` nào** — để dấu `!!` giữ được sức nặng.

---

## 4. "Câu hỏi phân biệt" — tách bối cảnh khỏi khoảnh khắc

Đây là kỹ thuật riêng của dự án, và nó xử lý đúng cái vấn đề mà mục 2.2 chỉ ra:
**vị trí frame, không phải chọn video.**

### 4.1 Vấn đề: cao nguyên phẳng

Đề bài của BTC mô tả **cả BỐI CẢNH lẫn KHOẢNH KHẮC** trong cùng một đoạn văn. Nếu
bạn đưa nguyên văn câu truy vấn cho VLM ở bước **chốt frame**, VLM sẽ chấm theo
bối cảnh và cho **gần như mọi khung hình của đúng video ấy** điểm cao.

Kết quả: profile điểm trở thành một **cao nguyên phẳng, không có đỉnh** — hoàn
toàn vô dụng cho việc chỉ ra khung hình nào. Con số đo được ghi trong khoá `_doc`
của `round1/sharp_questions.json`: **một video ứng viên có 72/193 keyframe đạt
≥ 0.60**, vì khung nào cũng đúng chủ đề.

Cùng ý đó được nhắc lại trong help của cờ `--questions-json`
(`scripts/verify_hypotheses.py:92-96`): *"the frame-pinning question, which has to
differ from the retrieval query or every frame of a topical video scores high"*.

### 4.2 Cách chữa

Nói ngắn gọn:

> **Câu truy vấn** hỏi *"video này có phải chuyện đó không"*.
> **Câu hỏi phân biệt** hỏi *"khung hình này có phải giây phút đó không"*.

Hai nguyên tắc, chép từ `_doc`:

1. Chỉ hỏi về một **chi tiết THOÁNG QUA** — thứ chỉ đúng ở **đúng khoảnh khắc ấy**,
   không đúng ở phần còn lại của video.
2. Hỏi dưới dạng **CÓ/KHÔNG** để model **buộc phải quyết**, thay vì chấm độ giống
   nhau.

### 4.3 Khuôn mẫu ba phần

Mọi entry thật trong `round1/sharp_questions.json` (24 câu) đều theo đúng khuôn
này — tôi đã đọc và đối chiếu:

```
(1)  Trong khung hình này, có <CHI TIẾT THOÁNG QUA> không?
(2)  Chấm 100 CHỈ KHI thấy rõ <CHI TIẾT ĐÓ>.
(3)  Nếu chỉ thấy <BỐI CẢNH CHUNG> mà không thấy <CHI TIẾT> thì chấm 0.
```

Vế (3) là vế làm nên chuyện: nó **gán sẵn một mức điểm thấp** cho cảnh cùng chủ đề
nhưng thiếu chi tiết. Không có vế đó thì cao nguyên quay lại ngay.

Ví dụ minh hoạ theo đúng khuôn (đã thay chi tiết thật bằng chỗ trống, xem cảnh báo
bên dưới):

```
Trong khung hình này, có bàn tay đang ĐẶT <nguyên liệu> VÀO <vật chứa> đang
bốc hơi không? Chấm 100 chỉ khi thấy rõ động tác xếp vào. Thấy <món ăn> bày
sẵn mà không có động tác xếp thì chấm 20.
```

So sánh với câu truy vấn gốc — vốn mô tả cả căn bếp, cả món ăn, cả người nấu — bạn
sẽ thấy ngay vì sao câu gốc cho điểm cao ở **cả trăm** khung hình của video đó,
còn câu này chỉ cho điểm cao ở **vài** khung.

> 🔒 **`round1/sharp_questions.json` mô tả rất chi tiết cảnh đúng của từng câu hỏi
> vòng 1. Hãy coi nó ngang với đáp án.** Đừng dán nội dung file đó vào tài liệu
> công khai, slide, hay chat nhóm mở. Đó là lý do ví dụ ở trên đã bị che.

### 4.4 Đọc profile, không đọc mỗi frame tốt nhất

`scripts/verify_hypotheses.py` in **cả profile** (số frame ≥ 0.60, top N frame kèm
lý do), không chỉ frame tốt nhất. Docstring dòng 8–12 giải thích vì sao: profile
đầy đủ mới là thứ phân biệt **một cú trúng thật** (đỉnh nhọn ngay chỗ có khoảnh
khắc) với **một cú trượt cùng chủ đề** (cao nguyên phẳng, điểm tầm tầm).

Hai chi tiết vận hành đáng nhớ:

- **Đừng lấy N khung hình ĐẦU** của một video để đại diện cho cả video. Keyframe
  xếp theo thời gian, nên cắt phần đầu là chỉ hỏi về hai phút đầu rồi kết luận cả
  video không có. Dùng `evenly()` (mặc định) hoặc `--range`.
- Khung hình **không có phán quyết** thì bị **LOẠI khỏi bảng**, không bị gộp vào
  như 0.0 — vì gộp 0.0 sẽ khiến một mạng chết đọc thành *"model đã nhìn và nói
  không"*. Nếu bảng in ra `THIEU n`, đó là n khung bị loại.

---

## 5. Cách 100 dòng được chia

Đây là chỗ tất cả những điều trên biến thành điểm.

**Chiến lược lai** (`allocate_hybrid_rows`, `src/core/submission.py:277-327`):
`n_flat` dòng đầu tiêu cho `n_flat` keyframe **khác nhau**, phần đuôi mới rải
thang frame. Cấu hình chạy thật: **n_flat = 30**.

Lý do là một phép **hedge**: BTC không bao giờ công bố cửa sổ `[s,e]` rộng bao
nhiêu, và chiến lược tối ưu **đảo chiều** theo bề rộng đó — cửa sổ rộng thì tiêu
mọi dòng cho keyframe khác nhau là tốt nhất; cửa sổ hẹp thì keyframe gần như vô
dụng và phải rải thang. Vì R@k là max trên **tiền tố**, hai chiến lược ghép lại
**gần như miễn phí**: giao các hạng đắt (1, 2–5, 6–20 — đáng 1.0, 0.8, 0.6) cho
keyframe riêng biệt, còn phần đuôi rẻ (21–100, chỉ đáng 0.4 và 0.2) cho thang.

Phần đuôi được chia bằng một hàm chi phí **tuyến tính**
(`AllocationPlan`, dòng 211–236):

```
cost(i, d) = breadth_cost·i + depth_cost·d      # chạy thật: i + 0.5·d
```

- **Vì sao cần một công thức:** mỗi dòng nộp thực chất là một cặp (ứng viên thứ
  `i`, nấc thang thứ `d`). Một hàm chi phí tuyến tính biến hai chiều đó thành
  **một thứ tự duy nhất** — sắp một lần rồi lấy dần, không cần luật `if/else` nào.
- **Vì sao tuyến tính:** nó đơn điệu theo cả `i` lẫn `d`, nên ô `(0,0)` luôn rẻ
  nhất → keyframe tốt nhất **luôn** chiếm hạng 1, chỗ duy nhất đáng trọn 1.0.
- **Vì sao depth rẻ hơn breadth (0.5 < 1.0):** hai loại sai lệch **không đối
  xứng**. Sai video là chết hẳn, không thang nào cứu được. Còn đúng video mà frame
  lệch thì **cứu được** — và trường hợp này phổ biến hơn nhiều so với cảm giác,
  đúng vì lý do ở mục 1.5. Đặt `depth_cost = 0.5` nghĩa là *"hai id frame nữa cho
  ứng viên hiện tại đắt bằng một video mới"*.
- **Vì sao đúng 0.5:** không phải trực giác, mà **quét thực nghiệm**
  (`scripts/experiment_allocation.py`).

Hệ quả cụ thể, và nó nối thẳng về mục 2.2: ứng viên **hạng 1** được thang vươn tới
**±120 frame** (`max_depth=24 × step=10`), còn ứng viên **hạng 25** chỉ được **một
dòng phẳng, không thang** — nó chỉ ăn điểm nếu đáp án rơi trong ±5 của đúng một
keyframe đó, khoảng **18%** số lần.

**Đó chính là lý do `_peak_preference` ăn tiền.** Đo được: keyframe *gần sự thật
nhất* đứng hạng 1 trong video đúng chỉ **48%** số lần, nhưng nằm trong **top-5 tới
76%**. Rải thang quanh keyframe hạng 1 phủ **55%** số câu; nếu chọn đúng keyframe
thì phủ **98%** (`docs/KIEN_TRUC_VA_HUONG_CAI_THIEN.md:108-113`). Đẩy đúng keyframe
lên vài bậc là đưa nó **tới chỗ đã sẵn có thang sâu**.

> ⚠️ **Giá trị mặc định trong dataclass KHÔNG phải giá trị đang chạy.**
> `AllocationPlan.depth_cost` mặc định là **0.75** (dòng 232) nhưng cấu hình nộp
> thật là **0.5** (`DEFAULT_DEPTH_COST`, `scripts/make_submission.py:54`). Tương tự
> `allocate_hybrid_rows` có `n_flat=20` mặc định (dòng 279) trong khi bản chạy thật
> dùng **30**. Viết `AllocationPlan()` trần trong một script thí nghiệm là bạn đang
> đo một cấu hình **khác** với cấu hình đã nộp.

> 📌 **Docstring của `AllocationPlan` (dòng 228) trỏ tới `scripts/tune_allocation.py`
> — file đó KHÔNG tồn tại.** Tôi đã kiểm bằng `ls`. Bộ quét thật là
> `scripts/experiment_allocation.py`. Đừng đi tìm.

**Hướng tinh chỉnh này đã cạn.** Quét `n_flat × depth_cost × step` quanh đỉnh với
24 lần rút thăm: cấu hình đang chạy (30 / 0.5 / 10) đạt **0.333** so với **0.338**
của cấu hình tốt nhất tìm được (28 / 1.0 / 14) — chênh 1.5%, **nằm trong nhiễu**,
nên không đổi (`docs/WHAT_CHANGED.md:156`). Điểm còn lại **không** nằm ở việc chỉnh
ba tham số này; nó nằm ở việc đẩy đúng keyframe lên hạng cao.

---

## 6. Những gì đã thử và **KHÔNG** ăn

**Đây là mục quan trọng nhất để đồng đội không làm lại.** Mỗi dòng dưới đây là một
ý tưởng *nghe rất có lý*, đã được cài đặt, đã được đo tử tế, và **thua**.

### 6.1 Cải thiện bộ truy xuất

| Ý tưởng | Kết quả | Nguồn |
|---|---|---|
| Đổi trọng số ensemble 4 prompt (0.60/0.40) | **hoà** với 0.45/0.35/0.10/0.10 — không cải thiện | `README.md:42` |
| Làm mượt điểm theo thời gian (±1, ±2 keyframe) | **giảm** 0.405 → 0.382 | `README.md:43` |
| Chuẩn hoá điểm theo từng video | **giảm mạnh** 0.405 → 0.276 | `README.md:44` |
| Giới hạn 2 frame/video, cách nhau ≥10 giây | tối ưu cho độ đo *khác*; bỏ cap **tăng điểm ở mọi bề rộng** | `src/core/kis_engine.py:15-20` |
| `combine='max'` khi ghép các đoạn của câu dài | thưởng frame chỉ khớp một mệnh đề — "giá sách" thắng chính cảnh chứa nó | `src/core/kis_engine.py:344-346` |

Kết luận trong README: *"khâu truy xuất của nhóm đã được tinh chỉnh tốt."*

### 6.2 Gộp thêm tín hiệu vào điểm số

| Ý tưởng | Kết quả | Nguồn |
|---|---|---|
| Metadata video (tiêu đề, mô tả, keyword) | KIS R@1 **43,3% → 40,0%** | `scripts/build_review_page.py:1156-1157` |
| Nhãn đối tượng cộng theo **frame** | +0.4% (nhiễu); ở w=0.05 thì **−7,4%** — và video R@1 tăng 26→30 | `src/core/objects.py:8-16` |
| Khớp **số lượng** đối tượng ("ba người") | **+0.0%** — hoàn toàn trơ, ở mọi trọng số | `src/core/objects.py:13` |
| Lời thoại gộp ở **cấp video** | −0,4% ở trọng số tốt nhất; cả dải là −0,1% đến −23% | `docs/WHAT_CHANGED.md:200`, `docs/DOC_NOI_DUNG_ANH.md:74` |
| Lời thoại gộp theo **mốc thời gian từng frame** | ±0,0% ở trọng số tốt nhất; cả dải là **−1,5% đến −20%** | `docs/WHAT_CHANGED.md:202`, `docs/DOC_NOI_DUNG_ANH.md:76` |
| Lời thoại có cổng chặn theo độ quyết đoán | +0,5% — **nhiễu** | `docs/WHAT_CHANGED.md:204` |
| Trọng số VLM cao (w=0.10 / 0.20) | −2,1% / −5,7%, trong khi video R@1 tăng 25→29 | `scripts/vlm_rerank_run.py:9-10` |

**Mẫu số chung:** mọi tín hiệu **cấp video** đều chỉ có thể xếp lại **video**, mà
chọn đúng video chỉ chiếm 22% phần điểm lấy lại được. Còn mọi tín hiệu cộng **theo
frame** đều có nguy cơ đá văng một frame gần sự thật hơn.

### 6.3 Gộp danh sách ứng viên

| Ý tưởng | Kết quả |
|---|---|
| Gộp hai danh sách (dịch máy + dịch tay), lấy điểm cao hơn mỗi frame | **0.305** |
| chỉ dịch máy | 0.292 |
| chỉ bản dịch tay | 0.321 |
| **cả hai trong MỘT vector truy vấn** (đang dùng) | **0.337** |

Lập luận cũ *"R@k là max trên tiền tố nên thêm ứng viên chỉ có lợi"* là **sai**, vì
gộp danh sách cũng **ĐẢO thứ tự**: một frame mà bản dịch tay thích bị đẩy lên trước
frame mà bản dịch máy tìm ra, và **30 chỗ đầu là hữu hạn**
(`scripts/make_submission.py:230-247`, `docs/WHAT_CHANGED.md:117-128`).

Lưu ý: `merged_hits` bây giờ chỉ là **alias** của `ranked_hits`
(`scripts/make_submission.py:259`) — tên cũ giữ lại để không vỡ giữa vòng thi.
**Nó không còn gộp danh sách nữa**; đừng suy ra hành vi từ cái tên.

### 6.4 Ngoại lệ quan trọng: **nới rộng** thì an toàn, **cộng điểm** thì không

Có đúng một cách dùng kênh phụ mà **an toàn về mặt toán học**: nối ứng viên mới
vào **CUỐI** danh sách, không bao giờ chèn lên đầu
(`scripts/vlm_rerank_run.py:135-139`).

Vì R@k là max trên k dòng đầu: một ứng viên **sai** thêm vào cuối chỉ tốn một chỗ
hạng rẻ và không mất gì thêm; còn một ứng viên **đúng** mà trước đó không hề có
trong danh sách thì đáng **cả một câu hỏi**. Chính sự bất đối xứng đó khiến nới
rộng an toàn ở chỗ mà cộng điểm thì không.

Phiên bản đầu tiên cho ứng viên lời thoại **tranh hạng 1** và nó đã **xáo hỏng bốn
câu vốn đã đúng** (dòng 309–313).

### 6.5 Đừng vứt bỏ kênh phụ chỉ vì phép đo âm

Đây là mặt trái của mục 6.2, và nó cũng quan trọng.

Phép đo âm chỉ nói về **60 câu ground truth**, mà cả 60 câu đó đều là mô tả cảnh
nhìn thấy. Vòng thi thật có câu **topical** — và chính lời thoại đã tìm ra hai
video mà hình ảnh bỏ sót **hoàn toàn**, còn OCR đã phát hiện một câu trong bài nộp
vòng 1 đang ở **SAI video**.

Nên kết luận đúng không phải *"kênh phụ vô dụng"* mà là:
**kênh phụ đi tới MẮT NGƯỜI, không tới bộ chấm điểm.**

---

## 7. Tóm tắt: bốn câu để nhớ

1. **Đúng video chưa đủ.** Luật đòi video **VÀ** frame trong cửa sổ. 60% phần điểm
   lấy lại được nằm ở vị trí frame.
2. **Độ chính xác cấp video là chỉ số đánh lừa.** Ba lần đo độc lập cho thấy nó
   tăng trong khi điểm thi giảm. Không bao giờ dùng nó để chỉnh tham số.
3. **Dòng thừa miễn phí, chỗ hạng thì không.** Luôn nộp đủ 100 dòng; nhưng 30 chỗ
   đầu là hữu hạn nên đừng để thứ gì chưa đo được chen vào đó.
4. **Bối cảnh chốt VIDEO, chi tiết thoáng qua chốt FRAME.** Dùng hai câu hỏi khác
   nhau cho hai việc khác nhau.

---

## Phụ lục — chạy lại mọi con số

```bash
# Điểm chính thức trên 60 mẫu (đáp án rút thăm không snap)
python scripts/evaluate_official.py
python scripts/evaluate_official.py --compare     # so với chiến lược cũ

# Điểm đang mất ở đâu (A: sai video / B: hạng thấp / C: lệch frame)
python scripts/loss_breakdown.py

# Cách chia 100 dòng. Cờ --snapped CHỈ để tái hiện cái bẫy, không phải để dùng.
python scripts/experiment_allocation.py

# Từng tín hiệu một
python scripts/experiment_retrieval.py         # ensemble / smoothing / chuẩn hoá
python scripts/experiment_merge.py             # cách kết hợp bản dịch tay
python scripts/experiment_objects_rerank.py    # đối tượng: per-frame vs per-video
python scripts/experiment_transcripts.py       # lời thoại cấp video
python scripts/experiment_frame_from_speech.py # lời thoại cấp frame
python scripts/experiment_vlm.py --limit 30    # VLM
python scripts/experiment_strategies.py        # xếp hạng theo TRƯỜNG HỢP XẤU NHẤT

# Test — 186 test trong tests/, các khẳng định về chấm điểm lấy thẳng từ
# ví dụ đã giải trong tài liệu luật, nên test đỏ nghĩa là ta đã lệch khỏi luật.
python -m pytest tests -q
```

**Ghi chú về số lượng test:** tôi đếm được **186** test khi collect `tests/`.
`docs/WHAT_CHANGED.md:93` ghi **388 test** cho toàn repo, và dòng 56 ghi CHRONOS
(TRAKE) có **213 test riêng** — nhiều khả năng con số 388 tính cả bộ test của
`src/task3_trake/` nằm ngoài thư mục `tests/`. Tôi **không** xác minh được điều này,
chỉ ghi lại để bạn biết vì sao hai con số khác nhau.
