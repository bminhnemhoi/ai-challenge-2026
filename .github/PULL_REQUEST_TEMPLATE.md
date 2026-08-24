<!-- Ngắn thôi. Bốn dòng có thật hơn bốn đoạn mô tả. -->

## Đổi gì và vì sao

<!-- Vì sao trước, đổi gì sau. Có số đo thì ghi số; có chỗ dựa trong mã thì ghi file:dòng. -->



## Tự kiểm

Dán nguyên văn output, đừng gõ lại từ trí nhớ:

```
$ python -m pytest tests src/task3_trake/tests src/task2_vqa -q


```

<!-- Đúng lệnh CI chạy (.github/workflows/ci.yml). Môi trường nhẹ: pip install -r requirements-dev.txt
     — 5 gói, không torch, khoảng 19 giây cho 415 test. Phải gọi pytest từ gốc repo (conftest.py). -->

## Checklist

- [ ] Đã chạy `python -m pytest tests src/task3_trake/tests src/task2_vqa -q` — xanh hết và **không dòng nào `skipped`** (CI cho đỏ nếu thấy chữ đó; thường là thiếu Node 20).
- [ ] `git status` sạch: không `.env`, không `round1/`, `picks_verified.txt`, `sharp_questions.json`, không chuỗi `AIza...` trong diff. Repo công khai — lỡ đẩy khoá lên thì **đổi khoá**, xoá commit không cứu được.
- [ ] Có đổi hành vi thì **đã thêm test**, tên test là một câu khẳng định về hệ quả.
- [ ] Có đụng định dạng bài nộp (`src/core/submission.py`, `scripts/make_submission.py`, `scripts/repackage.py`, `scripts/review_export.js`): đã chạy `python scripts/verify_zip.py <đường-dẫn>/submission.zip --queries rehearsal/queries` và dán kết quả ở trên. Sai định dạng vẫn tốn 1 trong 3 lượt nộp.
- [ ] Có đổi thứ hạng: vẫn xếp qua `ranked_hits`, không gọi `engine.search()` trực tiếp (`tests/test_review_workflow.py:109-117` canh chỗ này).
- [ ] Có sửa `src/task2_vqa/`: chạy `python -m pytest src/task2_vqa -q` (15 test). CI cũng chạy nhóm này, nhưng biết sớm vẫn hơn.

<!-- Sửa bộ phân bổ dòng thì phải sửa cả hai bản (Python + review_export.js) rồi chạy
     python -m pytest tests/test_js_allocator.py -q. Chi tiết: docs/PHAT_TRIEN.md mục 8. -->
