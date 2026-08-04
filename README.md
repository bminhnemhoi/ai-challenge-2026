# AI Challenge 2026 - Local Video Retrieval System

Hệ thống tìm kiếm và truy vấn video local cho cuộc thi **Hội thi Thử thách Trí tuệ Nhân tạo TP.HCM 2026 (AIC 2026) - Vòng Sơ Tuyển**.

Hệ thống hỗ trợ 3 dạng bài thi:
1. **Textual KIS (Textual Known Item Search)**: Tìm kiếm khung hình video chính xác theo mô tả văn bản.
2. **Visual Q&A (VQA)**: Trích xuất thông tin và trả lời câu hỏi ngữ cảnh video.
3. **TRAKE (Temporal Alignment)**: Căn chỉnh chuỗi sự kiện theo thứ tự thời gian.

---

## 🚀 Tính Năng Nổi Bật (Features)

* **Tích hợp Kho Vector BTC Batch 1 (177,321 Keyframes)**: Tự động nạp toàn bộ 873 file vector CLIP (ViT-B/32) `.npy` chính thức từ BTC mà không tốn dung lượng đĩa tải hết ảnh gốc.
* **Ánh Xạ Chuẩn Khung Hình Video (`frame_idx`)**: Tự động liên kết `map-keyframes/*.csv` để xuất `frame_idx` chính xác theo từng giây video gốc.
* **Tăng tốc GPU Local**: Mã hóa câu mô tả văn bản cực nhanh trên Apple Silicon GPU (`mps`).
* **Fast Cosine Search**: Tìm kiếm similarity trên **177,321 khung hình** với tốc độ `< 0.1s`.
* **Frame NMS (Non-Maximum Suppression)**: Khống chế các khung hình trùng lặp quá sát nhau trong cùng 1 video để tối ưu Top-100 kết quả nộp bài.
* **Bộ Chấm Điểm Chuẩn BTC (Evaluator)**: Đã lập trình thuật toán tính $R-Score$ và $Final\ Score$ ($R@1, R@5, R@20, R@50, R@100$) chuẩn quy định AIC 2026.
* **Unified Web Dashboard**: Giao diện Web App (FastAPI + HTML/CSS/JS) hiện đại, hỗ trợ xuất file CSV nộp bài 1-click.

---

## 🛠️ Cấu Trúc Dự Án (Project Structure)

```text
AI-Challenge-2026/
├── src/
│   ├── __init__.py
│   ├── btc_index_builder.py # Quét & hợp nhất 873 file CLIP .npy & map-keyframes từ BTC
│   ├── index_builder.py     # Trích xuất CLIP embedding thủ công cho keyframe local
│   ├── retriever.py         # Engine tìm kiếm Cosine Similarity + NMS
│   └── evaluator.py         # Bộ tính điểm R-Score & Final Score
├── scripts/
│   └── download_data.py     # Script tự động tải & giải nén dữ liệu cốt lõi BTC
├── query_kis.py             # Công cụ tìm kiếm nhanh qua CLI
├── app.py                   # FastAPI Web Backend Server
├── frontend/
│   ├── index.html           # Web UI Dashboard (3 Tab)
│   ├── style.css            # Dark mode glassmorphism UI system
│   └── app.js               # Event handling & CSV submission exporter
├── data/                    # Thư mục chứa clip-features-32, map-keyframes, metadata.json, embeddings.npy
├── keyframes/               # Các thư mục keyframes xem trước
├── README.md
└── .gitignore
```

---

## 📦 Hướng Dẫn Cài Đặt & Khởi Chạy

### 1. Cài đặt thư viện phụ thuộc
```bash
pip install torch transformers pillow numpy pandas fastapi uvicorn
```

### 2. Tạo bộ chỉ mục dữ liệu chính thức BTC (177,321 Keyframes)
```bash
python3 src/btc_index_builder.py
```
*Thời gian thực thi chỉ mất 1.5 giây để hợp nhất toàn bộ 873 video vào bộ nhớ RAM!*

---

## 🖥️ Hướng Dẫn Sử Dụng

### Khởi chạy Giao diện Web Dashboard
```bash
python3 -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Mở trình duyệt truy cập: **`http://localhost:8000`**

1. Nhập câu tìm kiếm vào Tab **Textual KIS** (Ví dụ: *"a man riding a bicycle on a city street"* hoặc mô tả sự kiện tiếng Việt).
2. Nhấn nút **Tìm Kiếm (KIS)**.
3. Xem danh sách xếp hạng kèm theo `video_id`, `frame_idx` thực tế và nút **Xuất File Nộp Bài (CSV)**.

### Chạy trực tiếp qua dòng lệnh CLI
```bash
python3 query_kis.py --query "a man riding a bicycle on a city street" -k 10
```

---

## 📜 Định Dạng Nộp Bài (Submission Format)

* **Textual KIS**: `<video_id>, <frame_id>`
* **Visual Q&A**: `<video_id>, <frame_id>, <answer>`
* **TRAKE**: `<video_id>, <frame_id_1>, ..., <frame_id_n>`

---

## 📝 License
Phát triển cho cuộc thi **AIC 2026**.
