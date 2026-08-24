# Đóng góp

Tài liệu này nói về **quy trình**: cài môi trường nhẹ, tạo nhánh, chạy đúng cái mà CI
chạy, mở PR. Nó cố ý không dạy cách sửa mã — phần đó đã có ở
[`docs/PHAT_TRIEN.md`](docs/PHAT_TRIEN.md) và không nên viết lại ở hai chỗ.

| Bạn đang cần | Đọc file nào |
|---|---|
| Cài đặt để **chạy thật** đường truy xuất, tải dữ liệu, chạy thử một vòng | [`README.md`](README.md) mục 1-2 |
| Quy ước mã, docstring, hằng số, cách thêm một kênh mới, các bẫy đã mất điểm | [`docs/PHAT_TRIEN.md`](docs/PHAT_TRIEN.md) |
| Lộ khoá API, dữ liệu vòng thi, giới hạn của job `guard` | [`SECURITY.md`](SECURITY.md) |
| Quy trình trong ngày thi | [`docs/QUY_TRINH_NOP.md`](docs/QUY_TRINH_NOP.md) |

Repo này **công khai**. Trước khi dán bất cứ output nào vào issue hay PR, đọc mục
[Không bao giờ commit](#5-không-bao-giờ-commit).

---

## 1. Cài đặt cho người phát triển

Nếu bạn chỉ định đọc mã, sửa một hàm, thêm test và gửi PR thì **không cần torch,
không cần chỉ mục 780 MB**:

```bash
git clone https://github.com/khanhle1406/ai-challenge-2026.git
cd ai-challenge-2026
python -m venv .venv
```

```bash
source .venv/bin/activate         # macOS / Linux
```

```powershell
.venv\Scripts\activate            # Windows
```

```bash
pip install -r requirements-dev.txt
python -m pytest tests src/task3_trake/tests -q
```

Trông đợi: `400 passed`. `requirements-dev.txt` chỉ có **5 gói** — pytest, numpy,
scipy, pyyaml, pillow — và đầu file ghi rõ con số đó đo trên một venv sạch
(`400 passed in 18,5 s`, đo 24/08/2026), không phải phỏng đoán. Thiếu `pillow` thì
13 test đỏ.

Ba điều dễ vấp:

- **Phải gọi `pytest` từ gốc repo.** `conftest.py` chỉ có ba dòng và việc duy nhất
  nó làm là chèn gốc repo vào `sys.path`; gọi từ thư mục khác thì `import src...`
  chết ngay.
- **Cần Node.js 20** nếu muốn khớp với CI. 16 test trong
  `tests/test_js_allocator.py` chạy `scripts/review_export.js` qua `node` và
  **tự bỏ qua** khi không có node. Trên máy bạn `skipped` là bình thường; trên CI
  `skipped` là **đỏ** (xem mục 6).
- `requirements-dev.txt` **không** chạy được đường truy xuất thật. Muốn chạy
  `make_submission.py`, VLM, OCR thì cài `requirements.txt` theo README mục 1.

---

## 2. Chu trình làm việc

```bash
git switch -c feat/ten-viec-ngan          # 1) nhánh mới, không sửa thẳng trên main
# ... sửa mã, thêm test ...
python -m pytest tests src/task3_trake/tests -q   # 2) đúng lệnh CI chạy
git status                                 # 3) nhìn bằng mắt: không .env, không round*/
git add -p && git commit                   # 4) commit theo mục 3
git push -u origin feat/ten-viec-ngan      # (người ngoài đội: push lên fork của mình)
```

5) Mở pull request vào nhánh **`main` của `khanhle1406/ai-challenge-2026`**. GitHub
   tự điền mẫu ở [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md)
   — điền hết, và **dán nguyên văn output của lệnh test**, đừng gõ lại từ trí nhớ.

6) Đợi CI xanh. **Cả hai job** (`test` trên ubuntu + windows, và `guard`) phải xanh
   thì PR mới được xem xét. CI chỉ chạy khi push vào `main` hoặc khi **mở PR**
   (`.github/workflows/ci.yml:8-12`) — đẩy một nhánh lên fork mà chưa mở PR thì
   chưa có ai canh cả.

Vài điểm về remote, nói thẳng để khỏi nhầm:

- `origin` là repo chính `https://github.com/khanhle1406/ai-challenge-2026.git`,
  nhánh đích là **`main`**. Còn một nhánh `master` cũ nằm lại bên cạnh — nó đã lạc
  hậu, đừng nhắm vào đó.
- Người ngoài đội: fork rồi mở PR từ fork sang `origin/main`. Đó là cách PR đầu
  tiên của repo này đang đi.
- **Đừng upload file qua giao diện web GitHub.** Lịch sử `origin/main` còn nguyên
  mấy commit `Add files via upload`, `Update config.py` — chúng không có thông điệp
  nào giải thích được điều gì, và chúng vào thẳng `main` mà không qua PR nên không
  có lượt CI nào chạy trước.
- Repo **không có** pre-commit hook, không có linter, không có `pyproject.toml`.
  Không có gì chặn ở phía máy bạn — mắt bạn nhìn `git status` chính là hàng rào.

---

## 3. Đặt tên nhánh và viết commit

### Tên nhánh

Nói thật: repo mới có đúng **một** nhánh chủ đề từng tồn tại (`wip/my-work`), nên
chưa có tiền lệ để mô tả. Đề xuất dùng từ nay, cùng bộ từ khoá với commit:

```
feat/<viec-ngan>      fix/<loi-ngan>      docs/<viec-ngan>      test/<viec-ngan>
```

Chữ thường, tiếng Việt không dấu hoặc tiếng Anh, nối bằng `-`. Ví dụ:
`fix/verify-zip-cot-trake`, `docs/huong-dan-dong-gop`.

### Thông điệp commit

Đây thì **không phải đề xuất** — 16 commit gần nhất đã theo đúng một khuôn, cứ
`git log` mà xem. Tiêu đề:

```
<type>(<scope>): <câu tiếng Việt KHÔNG DẤU, chữ thường, không chấm cuối>
```

Ví dụ có thật trong repo:

```
feat(core): cham diem dung theo luat BTC va phan bo 100 dong theo chi phi
fix(task2_vqa): 'import src' khong duoc chet vi mot thu vien tuy chon thieu
test: 186 test, phan lon la bay da tung sap vao
chore: khong dua ban do dap an vong luyen tap len repo cong khai
docs: bo tai lieu tieng Viet day du cho ca doi
```

- `type` đã dùng: `feat`, `fix`, `docs`, `test`, `chore`, và một lần ghép `perf+ci`.
- `scope` không bắt buộc; khi có thì là tên module hoặc một cụm không dấu —
  đã dùng: `core`, `vlm`, `trake`, `review`, `kenh-phu`, `do-luong`, `vi-du`,
  `task2_vqa`.
- Độ dài tiêu đề thực tế: **44-80 ký tự**.
- **Tiêu đề là một khẳng định về hệ quả, không phải mô tả công việc.** So sánh
  `test: 186 test, phan lon la bay da tung sap vao` với "them test". Cùng tinh thần
  với quy ước "tên test là một câu khẳng định" (`docs/PHAT_TRIEN.md` mục 2.4).

Thân commit — phần này mới là chỗ giá trị nằm:

- Xuống dòng thủ công quanh 80 cột.
- **Vì sao trước, đổi gì sau.** Có số đo thì ghi số; có chỗ dựa trong mã thì ghi
  `file:dòng`.
- Ghi cả **cái đã thử mà không dùng**, kèm con số đo được. Repo này lên điểm chủ
  yếu nhờ gỡ bỏ ý tưởng nghe có lý, nên một kết quả âm được ghi lại là tài sản.

---

## 4. Test

**Mọi thay đổi hành vi phải đi kèm test.** Bộ test không cần chỉ mục 780 MB và chạy
xong trong khoảng 20 giây, nên không có cái cớ nào để bỏ qua.

- Tên test là **một câu khẳng định** về điều nó bảo vệ, không phải `test_case_3`:
  `test_extra_wrong_rows_can_never_reduce_the_score`,
  `test_per_minute_429_is_not_treated_as_the_daily_quota`.
- Docstring của test kể lại **sự cố có thật** đã sinh ra nó.
- Sửa `src/core/submission.py` (bộ phân bổ dòng) thì phải sửa **cả**
  `scripts/review_export.js` — đó là bản cài đặt thứ hai chạy trong trình duyệt —
  rồi chạy `python -m pytest tests/test_js_allocator.py -q`.
- Viết script mới có gọi bộ truy hồi thì thêm tên nó vào danh sách `must_agree`
  trong `tests/test_review_workflow.py:109-117`. Test đó **đọc mã nguồn** 7 script
  và đỏ nếu file nào gọi thẳng `engine.search(` thay vì `ranked_hits`.
- Sửa `src/task2_vqa/` thì chạy `python -m pytest src/task2_vqa -q` (15 test). CI
  cũng chạy nhóm này, nhưng chạy trước ở máy vẫn nhanh hơn chờ CI.

### Vì sao repo này đo bằng ground truth trước khi tin

Không phải nghi thức. Đây là bài học đã trả bằng điểm số:

`data/ground_truth.json` có 60 câu, và **56/60 frame đáp án trùng khít một keyframe
của chỉ mục** — vì file được tạo bằng cách chọn keyframe từ chính chỉ mục đó. Đáp án
thật của BTC thì rơi bất kỳ đâu trong khe giữa hai keyframe. Chấm bằng ground truth
thô, cấu hình "chỉ nộp keyframe" được **0.562** và trông như tốt nhất; rút thăm lại
khoảnh khắc 24 lần thì nó tụt xuống **0.257**, còn cấu hình đang dùng lên **0.333**.
Kết luận đảo ngược hoàn toàn. Đó chính là loại sai số đã tạo ra "Top-1 41,67%" trên
máy mà chỉ **5,8** trên bảng xếp hạng.

Nên một cải tiến chỉ được tin sau khi đi qua một script `scripts/experiment_*.py`
(hiện có 14 cái, đọc `experiment_allocation.py` — ngắn nhất và đủ mọi thành phần)
với **bốn quy tắc, không bỏ quy tắc nào**:

1. Dùng **công thức chấm chính thức** (`final_score`, `r_score_kis`), không dùng chỉ
   số thay thế. **`video R@1` không phải điểm thi** — ba lần riêng biệt trong dự án
   này, R@1 tăng trong khi điểm thi giảm.
2. **Rút thăm lại** khoảnh khắc đáp án, không dùng ground truth thô.
3. Báo cáo trên **cả dải độ rộng cửa sổ**, không phải một con số. Chiến lược tối ưu
   đảo chiều theo bề rộng đó.
4. Gọi `ranked_hits`, không gọi `engine.search`.

Và **in ra cả số âm**. Script chỉ in khi kết quả đẹp thì không phải thí nghiệm.
Chi tiết: `docs/PHAT_TRIEN.md` mục 4 và mục 5.

---

## 5. Không bao giờ commit

Repo này công khai, và mọi thứ đã commit đều nằm lại trong lịch sử git cùng mọi bản
fork — kể cả khi bạn xoá ở commit sau.

| Không commit | Vì sao |
|---|---|
| `.env`, `.env.*` | chứa `GEMINI_API_KEY`. Bị `.gitignore:28-30` chặn — đừng `git add -f` |
| `round1/`, `round2/`, `round3/`, `vongthi*/` | đề thật của BTC và 100 dòng đáp án mỗi câu (`.gitignore:72-75`) |
| `picks_verified.txt` (mọi vòng) | bản đồ đáp án kèm lý luận |
| `sharp_questions.json` | mô tả rất chi tiết cảnh đúng của từng câu — **coi ngang với đáp án** |
| `submission_out/`, `submission_*.csv`, mọi `*.zip` | một bản nộp, không phải mã nguồn |
| chuỗi bắt đầu bằng `AIza...` ở bất cứ đâu | dạng khoá Google API |

Ba file cuối bảng cũng **không được dán** ra chat nhóm mở, slide, issue hay bình
luận PR — không chỉ là chuyện git.

Được commit **có chủ ý**, dùng thoải mái làm ví dụ: `rehearsal/queries/` (12 câu
diễn tập) và `round_p1/queries/` (đề vòng luyện tập).

**Lỡ đẩy khoá lên rồi thì ĐỔI KHOÁ trước, sửa mã sau.** Xoá dòng không làm khoá an
toàn trở lại. Đầy đủ ở [`SECURITY.md`](SECURITY.md).

---

## 6. Chạy CI ở máy mình trước khi đẩy

CI có hai job và bạn chạy lại được cả hai trong khoảng nửa phút.

**Job `test`** — đúng một lệnh, chép nguyên văn từ `.github/workflows/ci.yml:56`:

```bash
python -m pytest tests src/task3_trake/tests -q
```

Rồi bước thứ hai, bước mà người ta hay quên: CI **chạy lại bộ test lần nữa chỉ để
bắt chữ `skipped`** và cho đỏ nếu thấy (`ci.yml:58-66`, thông điệp: *"Bỏ qua âm thầm
nghĩa là không ai kiểm nữa"*). Tự kiểm:

```bash
python -m pytest tests src/task3_trake/tests -q 2>&1 | tail -3 | grep -i skipped
```

Không ra gì là tốt. Ra một dòng nghĩa là CI sẽ đỏ — thường vì máy chưa cài Node 20.

**Job `guard`** — ba lệnh, chép từ `ci.yml:76-100`:

```bash
git grep -InE 'AIza[0-9A-Za-z_-]{30,}' -- . ':!*.lock'
git ls-files | grep -E '^(round[0-9]|vongthi)|picks_verified\.txt|sharp_questions\.json'
git ls-files --error-unmatch .env
```

Cả ba **không ra kết quả** mới là sạch. Lệnh cuối phải báo lỗi "did not match any
file" — nếu nó in ra `.env` thì file khoá đang bị git theo dõi.

Ba điều CI **không** làm, đừng coi nó là tấm khiên:

- CI chạy đủ cả **415 test** (`tests` + `src/task3_trake/tests` + `src/task2_vqa`),
  nhưng **không chạy** `scripts/experiment_*.py`. Các phép đo quyết định giá trị mặc
  định là việc của người sửa, không phải của CI.
- Job `guard` soi khoá API ở **cả cây hiện tại lẫn toàn bộ lịch sử** (`git log --all -p`),
  nên khoá từng commit rồi xoá ở commit sau vẫn bị bắt. Nhưng CI chỉ **phát hiện** —
  đã lỡ commit khoá thì việc đầu tiên là **đổi khoá**, xem `SECURITY.md`.
- Không có bước lint nào. Các dấu `# noqa: E402` trong mã là quy ước thủ công, giữ
  nguyên chứ đừng dọn.

Ma trận CI: `ubuntu-latest` **và** `windows-latest`, Python 3.12, Node 20. Chạy cả
hai hệ điều hành là có lý do ghi thẳng trong file: repo này đã ăn đủ hai lỗi chỉ
xuất hiện trên Windows.

---

## 7. Cảnh báo cho người làm trên Windows

**KHÔNG BAO GIỜ sửa file bằng PowerShell `Get-Content` / `Set-Content`.**

```powershell
# TUYỆT ĐỐI KHÔNG
(Get-Content f -Raw).Replace(...) | Set-Content f -Encoding utf8
```

Console của máy là UTF-8 nhưng ANSI code page là Windows-1252, và các cmdlet
`*-Content` đi theo ANSI chứ không theo console. `Get-Content` giải mã byte UTF-8
thành Windows-1252 nên dấu tiếng Việt thành mojibake; rồi `-Encoding utf8` ghi thêm
một BOM vào đầu file. **Việc này đã xảy ra hai lần trong chính dự án này** — một lần
trên `docs/`, một lần trên `scripts/experiment_objects_rerank.py`.

`Get-Content` để **đọc** cũng dính đúng cái bẫy đó: nó hiển thị một file UTF-8 lành
lặn thành mojibake. Trước khi kết luận file hỏng, kiểm byte bằng Python.

Cách đúng: dùng trình soạn thảo, hoặc một heredoc Python ghi tường minh
`encoding="utf-8", newline="\n"`.

Vì sao nghiêm trọng chứ không chỉ khó chịu: `write_query_csv` cố ý ghi UTF-8
**không BOM**, kết dòng LF, không header (`src/core/submission.py:434-447`), và
`tests/test_submission.py:258-264` khẳng định byte ghi ra đúng bằng
`b"L01_V001,505\nL01_V001,515\n"`. Một lần chạm PowerShell vào file đó là hỏng cả ba
điều kiện cùng lúc.

Liên quan: mọi script trong `scripts/` phải gọi `safe_console()` **trước mọi lệnh
in** — console Windows mặc định cp1252, in một chữ `ạ` là `UnicodeEncodeError`, và
cú crash đó từng rơi vào giữa lúc ghi CSV và lúc đóng gói zip, để lại một bài nộp
nửa vời. Khuôn mẫu đầy đủ ở `docs/PHAT_TRIEN.md` mục 2.5.

---

## 8. Trước khi bấm "Create pull request"

- [ ] `python -m pytest tests src/task3_trake/tests -q` xanh, **không dòng nào
      `skipped`**.
- [ ] Ba lệnh của job `guard` (mục 6) không ra kết quả nào.
- [ ] Thay đổi hành vi đã có test, tên test là một câu khẳng định.
- [ ] Có đụng thứ hạng: vẫn xếp qua `ranked_hits`.
- [ ] Có đụng bộ phân bổ: đã sửa **cả hai** bản Python và JS.
- [ ] Có đụng định dạng bài nộp: đã chạy
      `python scripts/verify_zip.py <đường-dẫn>/submission.zip --queries rehearsal/queries`
      và dán kết quả vào PR. Sai định dạng vẫn tốn 1 trong 3 lượt nộp.
- [ ] Tiêu đề commit theo khuôn ở mục 3, thân commit nói **vì sao** kèm số đo.
- [ ] Bảy câu tự hỏi ở `docs/PHAT_TRIEN.md` mục 8.
