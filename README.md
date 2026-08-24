# AI Challenge HCMC 2026 — truy xuất video đa phương thức

Hệ thống tìm khoảnh khắc trong video cho **Hội thi Thử thách Trí tuệ Nhân tạo TP.HCM 2026 (AIC 2026)**.
Kho dữ liệu: **873 video / 177.321 keyframe**, chỉ mục **SigLIP-2 SO400M-384** (ma trận `177321 × 1152`, float32 — kiểm bằng `numpy` trên `data/embeddings_siglip2_384.npy`).
Ba loại đề: **KIS** (tìm cảnh), **Q&A** (tìm cảnh + trả lời), **TRAKE** (tìm chuỗi khoảnh khắc có thứ tự).

**Đang ở đâu:** vừa nộp xong **vòng sơ tuyển đợt 1** (25 câu) — chưa có điểm. Trước đó, trên bộ đề
**luyện tập** (24 câu), điểm đi từ **5,8 lên 8,6 / 24** qua bốn mốc 5,8 → 7,2 → 7,8 → 8,6
(con số này lấy từ `docs/KIEN_TRUC_VA_HUONG_CAI_THIEN.md` dòng 3 — đó là nơi duy nhất trong repo ghi lại nó).
Hai bộ đề dễ lẫn vì **cả hai đều đánh số `p1-*`**:
* `round_p1/` — bộ **luyện tập** (24 câu). Đề có commit, dùng để chạy thử và để mọi phép đo có thứ thật mà đối chiếu.
* `round1/` — bộ **sơ tuyển đợt 1** (25 câu). **Không commit** (`.gitignore`): repo này công khai, mà thư mục đó chứa đề của BTC và đáp án của đội.

> Đọc tài liệu nào thấy ghi `p1-4`, hãy tự hỏi *vòng nào* trước khi tin. Toàn bộ `docs/*.md` viết
> trước ngày 21/08 đều nói về bộ **luyện tập**.

> Sắp vào giờ thi? Đừng đọc README. Mở thẳng **[docs/CONTEST_RUNBOOK.md](docs/CONTEST_RUNBOOK.md)** — quy trình 3 tiếng, in ra và làm theo.

---

## 1. Cài đặt (5 phút, chưa tính thời gian tải dữ liệu)

Máy đang chạy được là **Python 3.12.6** trên Windows. `requirements.txt` chỉ ghim *cận dưới*, nên bản mới hơn thường vẫn chạy.

```bash
git clone <repo> && cd ai-challenge-2026

# 1) môi trường ảo
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

# 2) thư viện
pip install -r requirements.txt
```

Máy **không có CUDA** thì cài `torch` bản CPU trước cho nhẹ (ghi trong `requirements.txt` dòng 4-6):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Khoá API

Tạo file `.env` **ở gốc repo**, đúng một dòng:

```
GEMINI_API_KEY=<khoá của bạn>
```

`.env` đã nằm trong `.gitignore` (dòng 28-30) nên không bao giờ lên git. Hai chỗ đọc nó:
`src/core/gemini_engine.py:24-26` (`load_dotenv()`) và `src/core/vlm.py:84-93` (`load_env()`).
Lưu ý `load_env` dùng `os.environ.setdefault`, tức **biến môi trường của shell thắng file `.env`** — nếu thấy dùng nhầm khoá thì kiểm shell trước khi nghi `.env`.

Không có khoá vẫn chạy được toàn bộ phần tìm kiếm; chỉ mất phần trả lời Q&A và phần chấm lại bằng VLM. Khi đó thêm cờ `--no-answer`.

### Tải dữ liệu

```bash
python scripts/download_data.py
```

Tải `metadata.json` (46 MB), `embeddings_siglip2_384.npy` (**780 MB**), `blank_frame_indices.json`, và ba gói phụ trợ của BTC — trong đó `objects-aic25-b1.zip` nặng **610 MB**. Tính chung khoảng **1,5 GB**, nên chuẩn bị mạng và ổ đĩa.

Script chạy lại được nhiều lần: file nào xong rồi thì bỏ qua, và mỗi file tải vào `.part` rồi mới đổi tên, nên Ctrl-C giữa chừng không để lại file cụt (`scripts/download_data.py:57-76`). Nếu cuối cùng in `CHƯA XONG — n mục tải lỗi` thì **chưa xong thật**, chạy lại.

### Kiểm tra máy đã sẵn sàng

```bash
python -m pytest src/task3_trake/tests tests -q
```

Phải xanh hết. Đếm ngày 24/08/2026: **186 test** trong `tests/` và **214 test** trong `src/task3_trake/tests/`.

---

## 2. Chạy thử — một lệnh ra file nộp

Repo có sẵn bộ đề diễn tập 12 câu ở `rehearsal/queries/` (đây là bộ duy nhất được commit; `.gitignore` chặn mọi thư mục con khác của `rehearsal/`).

```bash
python scripts/make_submission.py --queries rehearsal/queries --out rehearsal/check --no-answer
```

Kết quả mong đợi:

```
12 query files in rehearsal/queries
loading the SigLIP-2 index (this is the slow part; it happens once) ...
  ready in ...s
  query-1-kis.txt            kis    100 rows   ...
  ...
wrote rehearsal\check\submission.zip  (... KB)
format check passed: submission/ folder, every row parsed, <=100 rows, no BOM, no .mp4
```

Nó ghi ra:

* `rehearsal/check/csv/*.csv` — một CSV cho mỗi câu, **trùng tên file query**
* `rehearsal/check/submission.zip` — file nộp, bên trong có thư mục tên đúng là `submission/`

**Chưa từng thấy dòng `format check passed` thì coi như chưa cài xong.** Thấy `FORMAT PROBLEMS` thì đừng upload — sai định dạng vẫn tốn 1 trong 3 lượt nộp của vòng.

Bỏ `--no-answer` (khi đã có `GEMINI_API_KEY`) để hệ thống tự điền cột đáp án cho các câu Q&A. Câu Q&A bỏ trống ăn 0 điểm dù frame đúng.

Kiểm lại bất kỳ zip nào trước khi upload — kể cả zip do trang duyệt tự dựng trong trình duyệt — bằng lệnh dưới đây (không nạp chỉ mục, trả lời dưới 1 giây):

```bash
python scripts/verify_zip.py <đường-dẫn>/submission.zip --queries rehearsal/queries
```

Sửa CSV bằng tay xong thì **đóng gói lại rồi kiểm lại**, đừng zip tay:

```bash
python scripts/repackage.py --out rehearsal/check
```

---

## 3. Cấu trúc thư mục

```
src/
├── core/                  ★ mọi thứ đường nộp bài thật sự đi qua
│   ├── submission.py      ★ chấm điểm theo luật BTC, thang frame, chia 100 dòng, đóng gói + kiểm zip
│   ├── kis_engine.py      ★ truy xuất SigLIP-2 gọn nhất chạy được: dịch VI→EN, 4-prompt, cắt đoạn, top-N
│   ├── objects.py           điểm cộng từ nhãn vật thể của BTC — cộng ở tầng VIDEO, không phải tầng frame
│   ├── vlm.py             ★ VLMJudge: Gemini chấm lại ảnh, cache 2 tầng, xoay vòng model, xử lý hết quota
│   ├── transcripts.py       BM25 trên phụ đề, giữ mốc thời gian từng câu nói
│   ├── ocr.py / colours.py  chữ đọc được trên khung hình, và màu của chủ thể (đo trên hộp bao, không phải cả ảnh)
│   ├── evaluator.py         bản công thức R-Score/Final Score của nhóm (bản dùng thật nằm trong submission.py)
│   ├── gemini_engine.py     client Gemini cho Q&A (đọc GEMINI_API_KEY)
│   └── *index_builder.py    dựng lại chỉ mục embedding
├── task1_kis/               retriever gốc ~1150 dòng, nhiều nhánh thử nghiệm — dùng để thăm dò, không dùng để nộp
├── task2_vqa/               các engine VQA cũ; import mềm nên thiếu thư viện không làm sập package
└── task3_trake/           ★ CHRONOS — quy hoạch động căn chuỗi sự kiện, kèm bộ test riêng
    ├── alignment.py         lõi DP O(N·T), λ thích ứng, khoá anchor, 4 chế độ căn chỉnh
    ├── scoring.py           hợp nhất visual + OCR + ASR, z-score theo từng hàng sự kiện
    ├── decomposer.py        tách đề tiếng Việt thành danh sách sự kiện (số sự kiện = số cột CSV)
    ├── trake_engine.py      adapter nối vào chỉ mục repo này — đây là đường chạy thật khi nộp
    └── chronos_engine.py    engine đầy đủ theo spec; hiện chỉ có test gọi tới

scripts/                     40 script; ba nhóm:
├── make_submission.py     ★ công cụ ngày thi: thư mục query vào, submission.zip đã tự kiểm ra
├── verify_zip.py          ★ cửa chốt cuối trước khi upload
├── repackage.py             đóng gói lại + kiểm lại sau khi sửa CSV bằng tay
├── build_review_page.py     dựng trang HTML một file để soát bằng mắt (thumbnail, OCR, màu, lời thoại)
├── apply_picks.py           ghim lựa chọn của người soát đè lên kết quả máy (chạy SAU vlm_rerank_run.py)
├── vlm_rerank_run.py        chạy Gemini chấm lại cả một vòng rồi ghi lại CSV
├── answer_qa.py             điền đáp án Q&A; read_answer.py đọc chữ/số ở độ phân giải gốc
├── search_transcripts.py    tra tay theo lời nói; search_ocr.py tra theo chữ trên hình
├── experiment_*.py          các phép đo mà mọi giá trị mặc định trong repo được chọn từ đó
└── download_data.py         tải chỉ mục + dữ liệu phụ trợ

tests/                       186 test — phần lớn là bẫy đã từng sập vào một lần
docs/                        tài liệu chi tiết (bảng ở mục 4)
data/                        (không commit) chỉ mục, metadata, cache dịch / OCR / màu / VLM / thumbnail
frontend/                    index.html + app.js + style.css cho app.py
notebooks/index-siglip2.ipynb  dựng lại chỉ mục SigLIP-2 trên Colab
rehearsal/                   bộ đề diễn tập 12 câu + các lượt chạy thử
round_p1/                    vòng LUYỆN TẬP: đề (24 câu, có commit) + các lượt chạy (không commit)
round1/                      vòng SƠ TUYỂN 1: không commit thứ gì (xem .gitignore)
round1/                      cùng vòng đó, thư mục làm việc riêng — .gitignore chặn TOÀN BỘ (dòng 72)
app.py                       web app FastAPI để dò tay bằng trình duyệt
query_kis.py                 CLI cũ tra một câu qua TextualKISRetriever
conftest.py                  chèn gốc repo vào sys.path cho pytest
```

★ = file nên đọc trước khi sửa bất cứ thứ gì.

Web app:

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 4. Tài liệu chi tiết

| File | Nội dung |
|---|---|
| [docs/KIEN_TRUC.md](docs/KIEN_TRUC.md) | Bản đồ kiến trúc: một câu hỏi vào, một file zip ra — từng chặng, file nào chịu trách nhiệm. |
| [docs/PHUONG_PHAP.md](docs/PHUONG_PHAP.md) | Vì sao cách làm này ăn điểm: luật chấm thật sự chấm gì, ba kênh độc lập, và **những gì đã thử mà không ăn**. |
| [docs/QUY_TRINH_NOP.md](docs/QUY_TRINH_NOP.md) | Cẩm nang chạy trong ngày thi, tính bằng phút, kèm mục "khi hỏng thì làm gì". |
| [docs/PHAT_TRIEN.md](docs/PHAT_TRIEN.md) | Dành cho người sửa mã: chạy test, quy ước, cách thêm một kênh mới, các bẫy đã mất điểm. |
| [docs/VI_DU_LUAN_CHUNG.md](docs/VI_DU_LUAN_CHUNG.md) | Sáu ca thật: khi ba kênh nói ba điều khác nhau thì tin ai. Mã video và đáp án **đã che**. |
| [docs/CONTEST_RUNBOOK.md](docs/CONTEST_RUNBOOK.md) | Quy trình 3 tiếng thi, chia theo mốc phút. In ra, đừng đọc lần đầu lúc 19:30. |
| [docs/HUONG_DAN_KIEM_THU.md](docs/HUONG_DAN_KIEM_THU.md) | Chạy thử trọn một bộ đề từ đầu tới cuối, dùng chính `round_p1/queries`. |
| [docs/KIEN_TRUC_VA_HUONG_CAI_THIEN.md](docs/KIEN_TRUC_VA_HUONG_CAI_THIEN.md) | Sơ đồ đường đi của một câu hỏi, bảng **mọi tín hiệu đã đo** (cái nào dùng, cái nào loại), và chỗ còn dư địa. |
| [docs/WHAT_CHANGED.md](docs/WHAT_CHANGED.md) | Nhật ký đợt cải tiến 5,8 → 8,6: đã đổi gì, vì sao, và bằng chứng đo được. |
| [docs/DOC_NOI_DUNG_ANH.md](docs/DOC_NOI_DUNG_ANH.md) | Ba kênh mà embedding không biểu diễn được: chữ trên hình, màu chủ thể, lời nói. |
| [docs/CHAN_DOAN_TRAKE.md](docs/CHAN_DOAN_TRAKE.md) | Vì sao TRAKE yếu, và **trần lý thuyết** đo được của nó. |
| [docs/SUA_VONG_1.md](docs/SUA_VONG_1.md) | Các câu **vòng luyện tập** từng nộp sai, kèm bằng chứng. Tên file dễ gây nhầm. |

Hai lưu ý về tài liệu:

* Bảng trên liệt kê những gì có trong `docs/` lúc viết README này. Cứ `ls docs/` để chắc.
* **Repo này công khai.** Ba file trên máy phải coi ngang với đáp án và **không bao giờ** dán ra
  chat nhóm mở, slide hay tài liệu công khai: `round1/picks_verified.txt` (bản đồ đáp án kèm lý
  luận), `round1/sharp_questions.json` (mô tả rất chi tiết cảnh đúng của từng câu), và
  `round_p1/picks_verified.txt`. Cả ba đã bị `.gitignore` chặn — đừng dùng `git add -f` với chúng.
* Các tài liệu ghi `p1-4`, `p1-18`… mà viết **trước 21/08** đều nói về bộ **luyện tập**, không
  phải vòng sơ tuyển. `docs/SUA_VONG_1.md` là tên gây nhầm nhất: nó nói về vòng luyện tập.

---

## 5. Đọc gì trước — cho người mới vào

Theo thứ tự này, khoảng một buổi:

1. **Luật chấm.** Đọc `src/core/submission.py` **31 dòng đầu**. Ba công thức R-Score được chép nguyên văn ở đó, cùng hai hệ quả chi phối toàn bộ thiết kế phía dưới. Phần còn lại của file chỉ là hiện thực hoá hai hệ quả đó.
2. **Đường đi của một câu hỏi.** `docs/KIEN_TRUC_VA_HUONG_CAI_THIEN.md` mục 1 và mục 2. Mục 2 là bảng mọi tín hiệu đã đo — đọc nó trước khi nảy ra ý tưởng mới, phần lớn ý tưởng "nghe có lý" đã bị thử và đã đo ra âm.
3. **Chạy thử.** Mục 2 của README này. Tự tay thấy `format check passed` một lần.
4. **Ngày thi.** `docs/CONTEST_RUNBOOK.md`, đọc hết một lượt.
5. **Khi nào tin máy, khi nào tin mắt.** `docs/VI_DU_LUAN_CHUNG.md`.

### Ba điều dễ làm ngược nhất

Ghi ở đây vì cả ba đều đã thật sự bị làm ngược một lần trong dự án này.

**1. Đúng video vẫn có thể 0 điểm.** `frame_id` trong luật là **số nguyên bất kỳ** của video gốc, không bắt buộc phải là một keyframe đã trích. Mà keyframe trong kho này cách nhau xa hơn cửa sổ đáp án nhiều lần. Nộp thuần chỉ số keyframe là tự chặn trần điểm — nên hệ thống rải thêm một **thang frame nguyên** (±10, ±20, …) quanh mỗi keyframe. Xem `src/core/submission.py:161-192` (`frame_ladder`).

**2. Dòng thừa không bao giờ hại.** `R@k` là **max trên k dòng đầu**, không phải tổng, không phải trung bình. Về mặt toán học một dòng sai không thể kéo bất kỳ `R@k` nào xuống. Luôn nộp đủ 100 dòng. Khẳng định này có test: `tests/test_submission.py:82`.

**3. Chỉ 5 mốc hạng có giá trị.** `k ∈ {1, 5, 20, 50, 100}`. Đẩy một câu trúng từ hạng 15 lên hạng 10 được **0 điểm**; từ hạng 6 lên hạng 5 được **+0,2**. Hàm `score_bucket` (`src/core/submission.py:124-133`) tồn tại chỉ để trả lời câu "bỏ công vào đâu thì được trả".

### Ba cái bẫy đo lường

**Đừng dùng video R@1 làm thước đo.** Đây là cái bẫy lặp lại của cả dự án: đã có ít nhất hai lần một thay đổi làm **video R@1 tăng** trong khi **điểm thi giảm** — vì khung hình mà tín hiệu mới thích nhất không phải khung hình gần khoảnh khắc đáp án nhất. Chi tiết trong `src/core/objects.py` (docstring đầu file) và `scripts/vlm_rerank_run.py:14-17`.

**Đừng chấm điểm bằng `data/ground_truth.json` thô.** 93% frame đáp án trong file đó trùng khít một keyframe, vì người gán nhãn chọn ra từ chính chỉ mục này. Đo kiểu đó thì thang frame luôn đo ra vô dụng. Phải dùng bản **rút thăm lại** khoảnh khắc trong khe keyframe — `scripts/experiment_allocation.py` làm sẵn việc này.

**Giá trị mặc định trong dataclass không phải giá trị đang chạy.** `AllocationPlan.depth_cost` mặc định 0.75, còn cấu hình nộp thật là **0.5** (`DEFAULT_DEPTH_COST`, `scripts/make_submission.py:54`); `allocate_hybrid_rows` mặc định `n_flat=20` trong khi bản chạy thật dùng **30**. Viết `AllocationPlan()` trần trong một script thí nghiệm là đang đo một cấu hình khác với cấu hình đã nộp.

### Một quy tắc bất di bất dịch

Mọi công cụ phải xếp hạng bằng `ranked_hits` (`scripts/make_submission.py:227-256`), **không được gọi `engine.search()` trực tiếp** — `engine.search` bỏ qua hai bước xếp lại hạng phía sau và sẽ cho ra thứ tự khác với thứ tự đã ghi vào bài nộp. Đã từng có lúc `make_submission`, trang duyệt và `apply_picks` xếp hạng bằng ba hàm khác nhau, nên người soát duyệt một khung hình **không phải** khung hình ở dòng 1 của bài nộp. Có test đọc mã nguồn 7 script và fail nếu thấy lời gọi trực tiếp: `tests/test_review_workflow.py:95-128`.

---

## 6. Định dạng nộp bài

```
KIS    <video_id>,<frame_id>
Q&A    <video_id>,<frame_id>,<answer>
TRAKE  <video_id>,<frame_1>,...,<frame_n>
```

Tối đa **100 dòng** mỗi câu · một CSV mỗi câu, **trùng tên file query** · **không có dòng tiêu đề** · UTF-8 không BOM · kết dòng LF · `video_id` **không có đuôi `.mp4`** · tất cả CSV nằm trong một thư mục tên **đúng là `submission/`** bên trong file zip.

Với TRAKE: **số cột chính là số sự kiện**. Sai số cột thì cả câu ăn 0 điểm, và kiểu sai này vô hình vì mọi kiểm tra khác vẫn pass.

`src/core/submission.py::verify_submission_zip` kiểm từng điều trên — đọc **toàn bộ** file trong zip chứ không lấy mẫu — và `make_submission.py` gọi nó tự động. Sai định dạng vẫn tốn 1 trong 3 lượt nộp, nên việc này thuộc về code chứ không phải một checklist trên giấy.

Đừng mở CSV bằng Notepad hay Excel rồi Save: chúng ghi lại kết dòng CRLF, và ký tự `\r` trở thành một phần của trường cuối cùng. Đã có kiểm tra chặn việc này, nhưng cách an toàn là sửa xong thì chạy `scripts/repackage.py`.
