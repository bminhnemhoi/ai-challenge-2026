# 🚀 AI Challenge 2026 (AIC 2026) - Multimodal Video Retrieval System

Hệ thống tìm kiếm và truy vấn video đa thức (Multimodal Video Retrieval Engine) hiệu năng cao cho cuộc thi **Hội thi Thử thách Trí tuệ Nhân tạo TP.HCM 2026 (AIC 2026) - Vòng Sơ Tuyển**.

Hệ thống được thiết kế để xử lý toàn bộ **873 Video (177,321 Khung hình Keyframe)** thuộc dữ liệu chính thức của Ban Tổ Chức (BTC) với tốc độ phản hồi tính bằng **millisecond** và độ chính xác **Top 10 đạt 93.3% (56 / 60 mẫu Ground Truth)**.

---

## 🌟 Tính Năng Nổi Bật (Key Features)

### 1. ☁️ Tích Hợp Cloud CDN Hugging Face (Zero Local Disk Footprint)
* Phủ kín **100% dữ liệu 177,321 khung hình** trên kho lưu trữ [BaeBaeBoo1010/aic2026-keyframes](https://huggingface.co/datasets/BaeBaeBoo1010/aic2026-keyframes).
* **Không tốn 35 GB dung lượng ổ cứng**: Web App truy xuất ảnh trực tiếp qua Hugging Face CDN với độ trễ thấp và bộ nhớ đệm RAM tự động.

### 2. ⚡ Dựng Khung Skeleton & Thông Tin Tức Thì (< 0.1s)
* **Instant Metadata Card Rendering**: Trả kết quả JSON từ RAM trong ~0.08s, dựng toàn bộ 100 thẻ kết quả với hiệu ứng **Shimmer Skeleton Placeholder**.
* **Xuất File Nộp Bài Ngay Lập Tức**: Nút "Xuất File Nộp Bài (CSV)" có thể bấm ngay sau 0.1s mà không cần chờ ảnh tải xong.
* **Smooth Lazy Streaming**: Trình duyệt tải ảnh bất đồng bộ ngầm và chuyển tiếp mượt mà (Fade-in).

### 3. 🎯 Google SigLIP 2 Multi-Prompt Ensemble
* Tự động dịch và kết hợp biểu diễn văn bản 4 nhánh: `[query_en, "a photo of " + query_en, "a high quality video scene of " + query_en, query_vi]`.
* Tối ưu hóa trên mô hình **Google SigLIP 2 (Base Patch16 224)** trên chip GPU Apple Silicon (MPS) / NVIDIA CUDA.

### 4. ⏱️ Tổng Hợp Thời Gian Đa Khung Hình (Temporal Multi-Frame Aggregation)
* Công thức tính điểm video: `Score = 0.85 * Top1_Frame + 0.15 * Mean(Top1, Top2)`.
* Tự động loại bỏ khung hình mào đầu tĩnh (n <= 2) và danh sách đen khung hình đen/trắng (solid blank frames).

---

## 📊 Kết Quả Benchmark Chính Thức (Task 1 KIS)

Kiểm thử trên toàn bộ **60 mẫu Ground Truth** chính thức:

| Chỉ số đánh giá | Số lượng mẫu chính xác | Tỷ lệ phần trăm |
| :--- | :---: | :---: |
| **Top 1 Accuracy** | **25 / 60** | **41.7%** |
| **Top 5 Accuracy** | **45 / 60** | **75.0%** |
| **Top 10 Accuracy** | **56 / 60** | **93.3%** |
| **Top 20 Accuracy** | **57 / 60** | **95.0%** |
| **Top 50 Accuracy** | **60 / 60** | **100.0%** |
| **Top 100 Accuracy** | **60 / 60** | **100.0%** |
| **Độ trễ trung bình** | **~111 ms / query** | **< 0.12s** |

---

## 🛠️ Cấu Trúc Dự Án (Monorepo Project Structure)

```text
AI-Challenge-2026/
├── src/
│   ├── __init__.py                   # Master Package Exporter
│   ├── core/                         # Thư mục dùng chung (Shared Core Modules)
│   │   ├── __init__.py
│   │   ├── gemini_engine.py          # Multimodal Image Fetcher & Session Pooling
│   │   └── evaluator.py              # Bộ tính điểm chuẩn R-Score & Final Score
│   ├── task1_kis/                    # 👤 Thành viên 1: Textual KIS
│   │   ├── __init__.py
│   │   ├── metadata_bm25.py          # Metadata BM25 Video Search
│   │   └── retriever.py              # Google SigLIP 2 Retriever & Multi-Prompt Fusion
│   ├── task2_vqa/                    # 👤 Thành viên 2 & 3: Visual Q&A (VLM + OCR)
│   └── task3_trake/                  # 👤 Thành viên 4: TRAKE (Temporal Alignment)
├── scripts/
│   ├── diagnose_60_detailed.py       # Script chạy Benchmark 60 mẫu Ground Truth
│   ├── build_objects_inverted_index.py # Builder cho 506 Object Classes
│   ├── build_siglip2_index_colab.py  # Script build SigLIP 2 index trên Colab
│   ├── download_data.py              # Script tải dữ liệu cốt lõi BTC
│   └── stream_upload_hf.py           # Script upload dữ liệu lên Hugging Face CDN
├── query_kis.py                      # Công cụ tìm kiếm nhanh qua CLI
├── app.py                            # FastAPI Backend Server daemon
├── frontend/                         # Giao diện Web App (Dashboard UI)
│   ├── index.html                    # HTML Dashboard
│   ├── style.css                     # Dark Mode Glassmorphism & Shimmer Skeletons
│   └── app.js                        # Client Search Logic & CSV Exporter
├── data/                             # Metadata.json, embeddings_siglip2.npy, caches
└── README.md
```

---

## ⚙️ Hướng Dẫn Cài Đặt & Chạy Nhanh Task 1

Dành cho bất kỳ ai clone repository về máy và muốn chạy ngay:

### 1. Clone Repository
```bash
git clone https://github.com/khanhle1406/ai-challenge-2026.git
cd ai-challenge-2026
```

### 2. Cài đặt các thư viện Python cần thiết
```bash
pip install torch torchvision transformers pillow numpy requests fastapi uvicorn deep-translator huggingface_hub
```

### 3. Tải file Vector Index SigLIP 2 (`data/`) (Tùy chọn nếu chưa có sẵn)
```bash
python3 scripts/download_data.py
```
*(Hoặc server sẽ tự động tải file embeddings khi khởi động lần đầu).*

---

## 🚀 Cách Chạy Task 1

### Cách 1: Chạy Giao Diện Web App (Khuyên dùng)
```bash
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
```
Mở trình duyệt truy cập: **`http://localhost:8000`**

1. Nhập câu truy vấn tiếng Việt hoặc tiếng Anh (ví dụ: *"người phụ nữ mặc váy đỏ"* hoặc *"xe đạp đua"*).
2. Bấm **Tìm Kiếm (KIS)**: Toàn bộ 100 kết quả và thông tin video sẽ hiện lên tức thì trong **< 0.1s**.
3. Bấm **Xuất File Nộp Bài (CSV)** để tải file submission chuẩn BTC.

### Cách 2: Chạy trực tiếp qua dòng lệnh (CLI Search)
```bash
# Tìm kiếm câu đơn
python3 query_kis.py --query "xe ô tô màu đỏ" --top_k 10

# Hoặc mở giao diện dòng lệnh tương tác (Interactive CLI)
python3 query_kis.py
```

### Cách 3: Chạy Benchmark kiểm tra độ chính xác (60 Mẫu Ground Truth)
```bash
python3 scripts/diagnose_60_detailed.py
```

---

## 📜 Định Dạng Nộp Bài Chính Thức (Official Submission Format)

* **Textual KIS**: `<video_id>, <frame_idx>`
* **Visual Q&A**: `<video_id>, <frame_idx>, <answer>`
* **TRAKE**: `<video_id>, <frame_idx_1>, ..., <frame_idx_n>`

---

## 📝 License & Competition Info
Dự án được phát triển phục vụ cuộc thi **Hội thi Thử thách Trí tuệ Nhân tạo TP.HCM 2026 (AIC 2026)**.

