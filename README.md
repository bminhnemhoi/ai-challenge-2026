# AI Challenge 2026 (AIC 2026) — Multimodal Video Retrieval

Hệ thống truy xuất video đa phương thức cho **Hội thi Thử thách Trí tuệ Nhân tạo TP.HCM 2026**, vòng sơ tuyển. Dữ liệu: **873 video / 177.321 keyframe**, chỉ mục **SigLIP-2 SO400M-384** (1152 chiều).

**Vào thẳng việc:** [docs/CONTEST_RUNBOOK.md](docs/CONTEST_RUNBOOK.md) — quy trình 3 tiếng thi, in ra và làm theo.

---

## Điều quan trọng nhất cần biết

Cuộc thi chấm theo `R-Score = I(đúng video ∧ frame_id ∈ [s,e])`, và `Final Score = ⅕·Σ R@k` với **R@k là điểm cao nhất trong k dòng đầu**, k ∈ {1, 5, 20, 50, 100}.

Ba hệ quả không hiển nhiên, và cả ba đều từng bị làm ngược:

**1. Đúng video chưa đủ.** Keyframe cách nhau trung vị **55 frame**, còn đoạn đáp án theo ví dụ trong luật chỉ rộng **11 frame**. Nếu chỉ nộp index keyframe, xác suất rơi trúng cửa sổ hẹp chỉ **17.6%** — dù truy xuất hoàn hảo. Thêm một thang số nguyên ±10, ±20… quanh keyframe đó đưa con số lên **~90%**, và `frame_id` vốn là số nguyên bất kỳ nên việc này hoàn toàn hợp lệ.

**2. Dòng thừa không bao giờ hại.** R@k là **max** trên tiền tố, không phải tổng. Nộp nhiều frame của cùng một video chỉ có thể tăng điểm. Giới hạn 2 frame/video của phiên bản cũ là tối ưu cho một độ đo *khác*.

**3. Chỉ 5 mốc hạng có giá trị.** Đẩy câu trúng từ hạng 15 lên 10 được **0 điểm**; từ 6 lên 5 được **+0.2**.

---

## Kết quả đo được

Đo bằng `scripts/evaluate_official.py` trên 60 mẫu ground truth, theo **đúng công thức chính thức**. Độ rộng cửa sổ đáp án không được công bố nên báo cáo trên cả dải; khoảnh khắc thật được đặt ngẫu nhiên trong khe keyframe thay vì trùng khít keyframe (xem lý do trong docstring của script).

| Chiến lược nộp bài | W=10 | W=20 | W=50 | W=100 | W=200 | TB |
|---|---|---|---|---|---|---|
| **Mới** — hybrid flat-30 + thang frame | **0.205** | **0.263** | **0.395** | **0.467** | **0.489** | **0.364** |
| Cũ — chỉ keyframe, tối đa 2/video | 0.092 | 0.158 | 0.313 | 0.402 | 0.426 | 0.278 |

Cải thiện **+31% trung bình**, và **+123% ở cửa sổ hẹp** (W=10) — trường hợp mà ví dụ trong luật gợi ý là thực tế. Chiến lược mới **tốt hơn ở mọi độ rộng đã thử**, không phải đánh đổi.

Đối chiếu: số cũ trong README (Top-1 41.67%, Top-5 71.67%) là **độ chính xác cấp video** — [script đánh giá cũ](scripts/evaluate_official_pipeline.py) chưa bao giờ so `frame_idx`. Nó không phải thứ BTC chấm.

### Những gì đã thử và **không** hiệu quả

Ghi lại để không ai mất công làm lại (`scripts/experiment_retrieval.py`):

| Ý tưởng | Kết quả |
|---|---|
| Đổi trọng số ensemble 4 prompt | 0.60/0.40 hoà với 0.45/0.35/0.10/0.10 hiện tại — **không cải thiện** |
| Làm mượt điểm theo thời gian (±1, ±2 keyframe) | **Giảm** 0.405 → 0.382 |
| Chuẩn hoá điểm theo từng video | **Giảm mạnh** 0.405 → 0.276 |

Kết luận: khâu truy xuất của nhóm đã được tinh chỉnh tốt. **Toàn bộ dư địa nằm ở cách phân bổ 100 dòng nộp bài.**

---

## Chạy

```bash
pip install -r requirements.txt
python scripts/download_data.py            # ~830 MB metadata + chỉ mục
python -m pytest src/task3_trake/tests tests -q
```

### Tạo file nộp bài (dùng trong ngày thi)

```bash
python scripts/make_submission.py --queries round1/queries --out round1/a
```

Đọc thư mục query của BTC, nhận dạng loại từ tên file (`-kis` / `-qa` / `-trake`), sinh **100 dòng mỗi câu**, đóng gói `submission/` trong zip và **tự kiểm tra định dạng** trước khi kết thúc.

Muốn dùng bản dịch tiếng Anh tự viết (đáng giá ~8 điểm % video R@1): đặt `query-1-kis.en.txt` cạnh `query-1-kis.txt`.

### Đo điểm

```bash
python scripts/evaluate_official.py --compare    # điểm chính thức, mới vs cũ
python scripts/experiment_strategies.py          # quét toàn bộ chiến lược phân bổ
python scripts/experiment_retrieval.py           # các ý tưởng cải thiện truy xuất
```

### Web app (dò tay, kiểm tra bằng mắt)

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## Cấu trúc

```
src/
├── core/
│   ├── submission.py      ★ chấm điểm chính thức, thang frame, phân bổ dòng, đóng gói zip
│   ├── kis_engine.py      ★ truy xuất SigLIP-2 gọn nhẹ + dịch VI→EN nhiều lớp dự phòng
│   ├── evaluator.py         công thức R-Score/Final Score (bản của nhóm)
│   └── gemini_engine.py     VLM đa phương thức cho Task 2
├── task1_kis/               retriever gốc (nhiều nhánh thử nghiệm) + BM25 metadata
├── task2_vqa/               VQA; các engine phụ import mềm, thiếu thư viện không làm sập package
└── task3_trake/           ★ CHRONOS — quy hoạch động căn chỉnh sự kiện
    ├── alignment.py         lõi DP O(N·T), λ thích ứng, anchor lock, 4 chế độ
    ├── chronos_engine.py    engine độc lập (có bộ test riêng)
    └── trake_engine.py      adapter nối vào chỉ mục của repo này

scripts/
├── make_submission.py     ★ công cụ ngày thi
├── evaluate_official.py   ★ đo điểm CHÍNH THỨC
├── experiment_*.py          các phép đo mà mọi mặc định được chọn từ đó
└── download_data.py         tải dữ liệu

tests/                       34 test suy ra từ ví dụ trong luật
docs/CONTEST_RUNBOOK.md    ★ quy trình 3 tiếng thi
```

★ = viết mới hoặc viết lại trong đợt cải tiến này.

---

## Task 3 — TRAKE (CHRONOS)

Bản cũ lấy top-1 độc lập cho từng sự kiện rồi sắp xếp. Trên benchmark tổng hợp có ground truth cấy sẵn, cách đó đạt **20%** sequence accuracy; CHRONOS đạt **88%** — vì quy hoạch động khai thác được hai thứ mà top-1 độc lập bỏ qua: ràng buộc thứ tự, và việc các sự kiện thuộc **cùng một hành động** nên phải gần nhau.

Luật TRAKE định hình thiết kế: sai video là **0 điểm ngay**, đúng video thì tính tỉ lệ sự kiện khớp, và cửa sổ mỗi sự kiện *"thường dưới 10 frame"*. Nên mọi dòng đều dùng **một video duy nhất**, và ngân sách 100 dòng dành để thử các tổ hợp frame nhiễu dần quanh nghiệm căn chỉnh.

CHRONOS đi kèm **213 test riêng** và đã qua bốn vòng kiểm định đối kháng — chi tiết trong `src/task3_trake/` và `docs/` của module gốc.

---

## Định dạng nộp bài

```
KIS    <video_id>,<frame_id>
Q&A    <video_id>,<frame_id>,<answer>
TRAKE  <video_id>,<frame_1>,...,<frame_n>
```

Tối đa 100 dòng/câu · một CSV mỗi câu, **trùng tên file query** · **không có dòng tiêu đề** · UTF-8 · video id **không có `.mp4`** · tất cả CSV nằm trong thư mục tên đúng là `submission/` bên trong file zip.

`src/core/submission.py::verify_submission_zip` kiểm tra từng điều trên và `make_submission.py` gọi nó tự động — sai định dạng vẫn tốn 1 trong 3 lượt nộp, nên việc này thuộc về code chứ không phải checklist.
