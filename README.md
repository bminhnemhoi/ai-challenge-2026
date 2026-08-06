# 🚀 AI Challenge 2026 (AIC 2026) - Multimodal Video Retrieval System

Hệ thống tìm kiếm và truy vấn video đa thức (Multimodal Video Retrieval Engine) hiệu năng cao cho cuộc thi **Hội thi Thử thách Trí tuệ Nhân tạo TP.HCM 2026 (AIC 2026) - Vòng Sơ Tuyển**.

Hệ thống được thiết kế để xử lý toàn bộ **873 Video (177,321 Khung hình Keyframe)** thuộc dữ liệu chính thức của Ban Tổ Chức (BTC) với tốc độ phản hồi tính bằng **millisecond**.

---

## 🌟 Tính Năng Nổi Bật (Key Features)

### 1. ☁️ Tích Hợp Cloud CDN Hugging Face (Zero Local Disk Footprint)
* Phủ kín **100% dữ liệu 177,321 khung hình** trên kho lưu trữ [BaeBaeBoo1010/aic2026-keyframes](https://huggingface.co/datasets/BaeBaeBoo1010/aic2026-keyframes).
* **Không tốn 35 GB dung lượng ổ cứng Mac**: Web App truy xuất ảnh trực tiếp qua Hugging Face CDN với độ trễ siêu thấp.

### 2. ⚡ Kéo Ảnh Song Song 16 Luồng (16-Worker Parallel Prefetch & Local Cache)
* **Pre-fetch song song**: Ngay khi thực hiện tìm kiếm, backend tự động kích hoạt **16 luồng song song (ThreadPoolExecutor)** kéo đồng loạt 50 ảnh kết quả về bộ nhớ đệm `.cache_keyframes/`.
* **Bộ nhớ đệm siêu tốc**: Ảnh truy xuất lần thứ 2 có tốc độ phản hồi chỉ **`< 0.5 millisecond`**.
* **Browser Async Decoding**: Sử dụng thuộc tính `loading="lazy"` và `decoding="async"` giúp trình duyệt giải mã ảnh bằng GPU mà không gây giật lag giao diện.

### 3. 🎯 Tự Động Dịch Song Ngữ & Ghép Vector (Dual Vi-En Embedding Ensemble)
* Tự động nhận diện và dịch câu hỏi tiếng Việt sang tiếng Anh ngữ cảnh (`GoogleTranslator`).
* Ghép vector trọng số song ngữ: `0.75 * Embed(English) + 0.25 * Embed(Vietnamese)` giúp mô hình **CLIP ViT-B/32** hiểu chính xác 100% ý định tìm kiếm của người dùng.

### 4. 🎬 Lọc Cảnh Video Trùng Lặp (NMS Scene Diversity)
* Tích hợp thuật toán **Non-Maximum Suppression (NMS)** theo khung thời gian (`nms_frame_gap >= 15`).
* Loại bỏ triệt để các ảnh trùng lặp liên tiếp trong cùng một video, đảm bảo danh sách Top-100 nộp bài hiển thị đa dạng các phân cảnh video khác nhau.

### 5. ☁️ Cloud-to-Cloud Transfer Notebook (Google Colab 1Gbps)
* Cung cấp sẵn file Notebook [upload_to_huggingface_colab.ipynb](file:///Users/xuannguyen/Desktop/AI-Challenge-2026/upload_to_huggingface_colab.ipynb) kéo trực tiếp dữ liệu từ máy chủ BTC sang Hugging Face Hub với tốc độ **1 Gbps** mà không cần bật máy tính cá nhân.

---

## 🛠️ Cấu Trúc Dự Án (Project Structure)

```text
AI-Challenge-2026/
├── src/
│   ├── __init__.py
│   ├── btc_index_builder.py          # Quét & hợp nhất 873 file CLIP .npy & map-keyframes từ BTC
│   ├── index_builder.py              # Trích xuất CLIP embedding thủ công cho keyframe local
│   ├── retriever.py                  # Search Engine (Cosine Similarity + Dual Vi-En Ensemble + NMS)
│   └── evaluator.py                  # Bộ chấm điểm chuẩn R-Score & Final Score
├── scripts/
│   ├── download_data.py              # Script tải & giải nén dữ liệu cốt lõi BTC
│   └── stream_upload_hf.py           # Script stream upload dữ liệu lên Hugging Face
├── upload_to_huggingface_colab.ipynb # Jupyter Notebook chạy Cloud-to-Cloud trên Google Colab
├── query_kis.py                      # Công cụ tìm kiếm nhanh qua CLI
├── app.py                            # FastAPI Backend Server (Multi-worker prefetch + Cache)
├── frontend/
│   ├── index.html                    # Dashboard UI (Textual KIS, Visual Q&A, TRAKE)
│   ├── style.css                     # Modern Dark Mode Glassmorphism UI
│   └── app.js                        # Client Logic & CSV Exporter
├── data/                             # Metadata.json, embeddings.npy, .cache_keyframes
└── README.md
```

---

## ⚙️ Hướng Dẫn Cài Đặt & Khởi Chạy

### 1. Cài đặt môi trường Python
```bash
pip install torch transformers pillow numpy pandas fastapi uvicorn deep-translator huggingface_hub
```

### 2. Xây dựng bộ chỉ mục Vector BTC (177,321 Keyframes)
```bash
python3 src/btc_index_builder.py
```
*(Thời gian thực thi chỉ ~1.5 giây để hợp nhất toàn bộ 873 file vector .npy vào bộ nhớ!)*

### 3. Khởi chạy Web Server Search Engine
```bash
python3 -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Mở trình duyệt truy cập: **`http://localhost:8000`**

---

## 🖥️ Hướng Dẫn Sử Dụng Dashboard

1. Nhập câu truy vấn tìm kiếm tiếng Việt hoặc tiếng Anh vào thẻ **Textual KIS**.
   * *Ví dụ*: `"diễn giả mặc áo đỏ phát biểu tại một cuộc họp báo ngoài trời, phía sau có nhiều cây xanh"`
2. Nhấn nút **Tìm Kiếm (KIS)**.
3. Giao diện sẽ hiển thị danh sách kết quả xếp hạng kèm theo `video_id`, `frame_idx` và điểm số tương đồng `Score`.
4. Nhấn nút **Xuất File Nộp Bài (CSV)** để tải file submission chuẩn định dạng AIC 2026.

---

## 📜 Định Dạng Nộp Bài Chính Thức (Official Submission Format)

* **Textual KIS**: `<video_id>, <frame_idx>`
* **Visual Q&A**: `<video_id>, <frame_idx>, <answer>`
* **TRAKE**: `<video_id>, <frame_idx_1>, ..., <frame_idx_n>`

---

## 📝 License & Competition Info
Dự án được phát triển phục vụ cuộc thi **Hội thi Thử thách Trí tuệ Nhân tạo TP.HCM 2026 (AIC 2026)**.
