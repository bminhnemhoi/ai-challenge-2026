# AI Challenge 2026 - Local Video Retrieval System

Hệ thống tìm kiếm và truy vấn video local cho cuộc thi **Hội thi Thử thách Trí tuệ Nhân tạo TP.HCM 2026 (AIC 2026) - Vòng Sơ Tuyển**.

Hệ thống hỗ trợ 3 dạng bài thi:
1. **Textual KIS (Textual Known Item Search)**: Tìm kiếm khung hình video chính xác theo mô tả văn bản.
2. **Visual Q&A (VQA)**: Trích xuất thông tin và trả lời câu hỏi ngữ cảnh video.
3. **TRAKE (Temporal Alignment)**: Căn chỉnh chuỗi sự kiện theo thứ tự thời gian.

---

## 🚀 Tính Năng Nổi Bật (Features)

* **Tăng tốc GPU Local**: Trích xuất vector CLIP (ViT-B/32) cực nhanh trên chip Apple Silicon (`mps`).
* **Fast Cosine Search**: Tìm kiếm similarity qua ma trận Numpy chuẩn hóa L2 với tốc độ `< 0.05s` cho hàng ngàn khung hình.
* **Frame NMS (Non-Maximum Suppression)**: Loại bỏ các khung hình trùng lặp quá sát nhau trong cùng 1 video để đảm bảo tính đa dạng cho Top-100 kết quả nộp bài.
* **Bộ Chấm Điểm Chuẩn BTC (Evaluator)**: Đã lập trình thuật toán tính $R-Score$ và $Final\ Score$ ($R@1, R@5, R@20, R@50, R@100$) chuẩn quy định AIC 2026.
* **Unified Web Dashboard**: Giao diện Web App (FastAPI + HTML/CSS/JS) hiện đại, tích hợp xem trước keyframes local và xuất file CSV nộp bài 1-click.

---

## 🛠️ Cấu Trúc Dự Án (Project Structure)

```text
AI-Challenge-2026/
├── src/
│   ├── __init__.py
│   ├── index_builder.py     # Quét keyframes local & trích xuất vector CLIP
│   ├── retriever.py         # Engine tìm kiếm Cosine Similarity + NMS
│   └── evaluator.py         # Bộ tính điểm R-Score & Final Score
├── query_kis.py             # Công cụ tìm kiếm nhanh qua CLI
├── app.py                   # FastAPI Web Backend Server
├── frontend/
│   ├── index.html           # Web UI Dashboard (3 Tab)
│   ├── style.css            # Dark mode glassmorphism UI system
│   └── app.js               # Event handling & CSV submission exporter
├── data/                    # Thư mục chứa metadata.json và embeddings.npy (tự tạo)
├── keyframes/               # Tập keyframes mẫu từ BTC
├── README.md
└── .gitignore
```

---

## 📦 Hướng Dẫn Cài Đặt & Khởi Chạy

### 1. Cài đặt thư viện phụ thuộc
Đảm bảo máy đã cài đặt Python 3.10+ và PyTorch:

```bash
pip install torch transformers pillow numpy pandas fastapi uvicorn
```

### 2. Chuẩn bị dữ liệu & Tạo Index Vector
Đặt các thư mục keyframe vào `keyframes/` (ví dụ `keyframes/L24_V002/*.jpg`), sau đó chạy lệnh tạo vector index:

```bash
python3 src/index_builder.py
```
*Lưu ý: Quá trình tạo index chỉ cần thực hiện 1 lần duy nhất.*

---

## 🖥️ Hướng Dẫn Sử Dụng

### Lựa chọn 1: Sử dụng Giao diện Web App (Khuyên dùng)
Khởi chạy web server FastAPI:

```bash
python3 -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Mở trình duyệt truy cập: **`http://localhost:8000`**

1. Nhập câu tìm kiếm vào Tab **Textual KIS**.
2. Nhấn nút **Tìm Kiếm (KIS)**.
3. Xem kết quả xếp hạng và nhấn **Xuất File Nộp Bài (CSV)** để tải file submission.

### Lựa chọn 2: Chạy trực tiếp qua dòng lệnh CLI
```bash
python3 query_kis.py --query "a person speaking at a press conference" -k 10
```

---

## 📜 Định Dạng Nộp Bài (Submission Format)

* **Textual KIS**: `<video_id>, <frame_id>`
* **Visual Q&A**: `<video_id>, <frame_id>, <answer>`
* **TRAKE**: `<video_id>, <frame_id_1>, ..., <frame_id_n>`

---

## 📝 License
Phát triển cho cuộc thi **AIC 2026**. Tất cả mã nguồn được cấp phép sử dụng mở.
