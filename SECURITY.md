# Bảo mật

Repo này **công khai**. Mọi thứ commit vào đây, kể cả thứ đã xoá ở commit sau, đều
nằm trong lịch sử git và trong mọi bản fork/clone đã có. Tài liệu này nói ba việc:
báo sự cố cho ai, lộ khoá API thì làm gì, và vì sao dữ liệu vòng thi không được lên đây.

---

## 1. Báo một sự cố

**Đừng mở issue công khai, đừng viết trong bình luận PR.** Mô tả một lỗ hổng ở nơi
công khai là công bố nó cho tất cả mọi người trước khi đội kịp vá.

Thứ tự ưu tiên khi báo:

1. Tab **Security** của repo trên GitHub, mục *Report a vulnerability* — nếu thấy mục đó
   thì dùng, đây là kênh riêng tư do GitHub giữ.
2. Không thấy mục đó thì nhắn riêng cho chủ repo — tài khoản **`@khanhle1406`**
   (remote `origin` là `https://github.com/khanhle1406/ai-challenge-2026.git`), hoặc
   nhắn trong chat riêng của đội.

Trong tin nhắn nên có: file và số dòng, cách tái hiện, và mức ảnh hưởng bạn đánh giá.
**Không dán khoá API thật, không dán đáp án** vào tin nhắn báo lỗi — mô tả là đủ.

Đội đang trong mùa thi nên không cam kết thời gian trả lời tính bằng giờ. Riêng
trường hợp lộ khoá thì **đừng chờ trả lời**, làm mục 2 ngay.

---

## 2. Lỡ để khoá API ra ngoài

Đúng thứ tự, và thứ tự này quan trọng hơn từng bước:

**Bước 1 — ĐỔI KHOÁ trước, ngay lập tức.** Vào Google AI Studio, thu hồi khoá cũ và
tạo khoá mới. Làm việc này trước cả khi đụng vào mã nguồn.

**Bước 2 — rồi mới gỡ khỏi mã nguồn**, commit, và đẩy lên.

Lý do thứ tự này: gỡ khoá khỏi file *không* làm khoá an toàn trở lại. Chuỗi đó vẫn còn
trong lịch sử git, trong các bản fork đã clone, và trong giao diện PR của GitHub. Một
khoá đã lộ thì chỉ có đổi khoá mới cắt được thiệt hại; xoá dòng chỉ làm nó khó thấy hơn.

Bước *"Không có khoá API — soi cả cây hiện tại lẫn lịch sử"* trong CI soi **cả hai**:
`git grep` trên cây làm việc, và `git log --all -p` trên toàn bộ lịch sử (job checkout
với `fetch-depth: 0`). Nên một khoá từng commit rồi xoá ở commit sau **vẫn bị bắt**.

Nhưng CI chỉ **phát hiện**, không cứu được gì: khoá đã lên GitHub dù chỉ một phút thì
coi như đã lộ. Việc đầu tiên luôn là **đổi khoá**.

**Chỗ đúng của khoá** là file `.env` ở gốc repo, một dòng `GEMINI_API_KEY=...`.
`.gitignore:28-30` chặn `.env`, `.env.*`, `.env.local` — đừng thêm ngoại lệ, đừng
`git add -f`. Mã đọc khoá ở hai chỗ: `src/core/vlm.py:84-93` (`load_env`, dùng
`os.environ.setdefault` nên **biến môi trường của shell thắng file `.env`**) và
`src/core/gemini_engine.py:23-28` (`load_dotenv`).

Tự kiểm trước khi commit, chạy đúng cái mà CI chạy:

```bash
git grep -InE 'AIza[0-9A-Za-z_-]{30,}' -- . ':!*.lock'          # không ra gì là tốt
git log --all -p --no-color | grep -nE '^\+.*AIza[0-9A-Za-z_-]{30,}'   # cả lịch sử
git ls-files --error-unmatch .env                               # phải báo không khớp
```

Đừng viết `if ... | head; then` khi kiểm mấy lệnh này. Mã thoát của pipeline là của
lệnh **cuối**, mà `head` luôn trả 0 — viết thế thì phép kiểm luôn báo "có", kể cả khi
sạch. CI từng dính đúng bẫy đó.

---

## 3. Dữ liệu vòng thi

Đề của BTC và đáp án của đội **không được lên repo công khai**, vì hai lý do khác nhau:
đề là tài sản của ban tổ chức, còn đáp án nếu công khai thì mọi đội khác đọc được và
kỳ thi mất ý nghĩa. Phần *lý luận* thì đáng chia sẻ, phần *đáp án* thì không — nên bản
dùng để dạy đã gỡ đáp án nằm ở [`docs/VI_DU_LUAN_CHUNG.md`](docs/VI_DU_LUAN_CHUNG.md).

`.gitignore` đang chặn (số dòng đọc từ file thật):

| Dòng | Chặn | Vì sao |
|---|---|---|
| 28-30 | `.env`, `.env.*`, `.env.local` | khoá API |
| 72-75 | `round1/`, `round2/`, `round3/`, `vongthi*/` | vòng thi thật: đề BTC, 100 dòng đáp án mỗi câu, `picks_verified.txt`, `sharp_questions.json` |
| 78 | `round_*/picks_verified.txt` | bản đồ đáp án của vòng luyện tập |
| 61-64 | `round_*/*/`, `round_*/review*.html`, `round_*/*.zip` | các lượt chạy và bản dựng, chỉ chừa `!round_*/queries/` |
| 9-10 | `rehearsal/*/` | chỉ chừa `!rehearsal/queries/` |
| 8, 56 | `submission_out/`, `submission_*.csv` | không commit một bản nộp |

Có một cái bẫy được ghi thẳng trong `.gitignore:71`: mẫu `round_*` **không** khớp `round1`,
nên các dòng 72-75 phải viết riêng. Sửa `.gitignore` thì đừng gộp chúng lại.

Được commit **có chủ ý**: `rehearsal/queries/` và `round_p1/queries/` (26 file — đề vòng
luyện tập). Đó là đề luyện tập dùng làm ví dụ, không phải đề vòng thi thật.

Ba file trên máy phải coi ngang với đáp án và **không bao giờ** dán ra chat nhóm mở, slide
hay tài liệu công khai — kể cả khi chúng không nằm trong git:
`round1/picks_verified.txt`, `round1/sharp_questions.json`, `round_p1/picks_verified.txt`.

---

## 4. CI đang canh những gì

Job `guard` trong [`.github/workflows/ci.yml`](.github/workflows/ci.yml) (dòng 68-100) có
đúng ba bước, và cả ba đều cho đỏ chứ không chỉ cảnh báo:

1. **Khoá API** — bắt chuỗi khớp `AIza[0-9A-Za-z_-]{30,}`; thông điệp lỗi là
   *"Gỡ ra và ĐỔI KHOÁ ngay"*.
2. **Dữ liệu vòng thi** — chặn file được git theo dõi khớp
   `^(round[0-9]|vongthi)|picks_verified\.txt|sharp_questions\.json`. Mẫu này cố ý **không**
   khớp `round_p1/` vì đề luyện tập được commit có chủ ý.
3. **`.env` bị theo dõi** — `git ls-files --error-unmatch .env` mà thành công là đỏ.

Ba giới hạn phải biết, đừng coi `guard` là tấm khiên:

* Nó chạy **sau khi bạn đã đẩy lên GitHub**. Lúc CI đỏ thì dữ liệu đã nằm trên máy chủ công
  khai rồi. Hàng rào thật là mắt bạn nhìn `git status` trước khi commit.
* CI chỉ chạy khi push vào `main` hoặc khi **mở pull request** (`ci.yml`). Đẩy một nhánh
  lên fork mà chưa mở PR thì chưa có ai canh cả.
* Repo **không** có pre-commit hook, không có linter. Không có gì chặn ở phía máy bạn.

Danh sách kiểm trước khi commit nằm ở [`docs/PHAT_TRIEN.md`](docs/PHAT_TRIEN.md) mục 8.

---

## 5. Một lưu ý về web app

`app.py` là công cụ dò tay chạy **cục bộ**: không có xác thực, `allow_origins=["*"]`
(`app.py:28-35`). Route ảnh (`app.py:318-325`) chỉ trả về **chuyển hướng 302** sang CDN
của Hugging Face, không đọc file cục bộ nào; chỗ phục vụ file thật là
`StaticFiles(directory=frontend_dir)` ở cuối `app.py`, và nó chỉ mở thư mục `frontend/`.
Rủi ro ở đây không phải lộ file mà là **ai cũng gọi được API tìm kiếm**. Lệnh trong README dùng `--host 0.0.0.0`, nghĩa là mọi máy trong cùng mạng
đều gọi được. Đừng chạy nó trên mạng chung hay mạng công cộng; ở nhà thì dùng
`--host 127.0.0.1`.
