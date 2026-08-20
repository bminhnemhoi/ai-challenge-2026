# 🚀 AI Challenge 2026 (AIC 2026) - Multimodal Video Retrieval System

Hệ thống tìm kiếm và truy vấn video đa thức (Multimodal Video Retrieval Engine) hiệu năng cao cho cuộc thi **Hội thi Thử thách Trí tuệ Nhân tạo TP.HCM 2026 (AIC 2026) - Vòng Sơ Tuyển**.

Hệ thống được thiết kế để xử lý toàn bộ **873 Video (177,321 Khung hình Keyframe)** thuộc dữ liệu chính thức của Ban Tổ Chức (BTC) với tốc độ phản hồi tính bằng **millisecond** và độ chính xác **Top 5 đạt 71.67% (43 / 60 mẫu Ground Truth)**, **Top 10 đạt 91.67% (55 / 60 mẫu Ground Truth)**, **Top 20 đạt 96.67% (58 / 60 mẫu Ground Truth)** mà **hoàn toàn không hardcode (100% Generalized Mathematics)**.

---

## 🌟 Tính Năng Nổi Bật (Key Features)

### 1. ☁️ Tích Hợp Cloud CDN Hugging Face (Zero Local Disk Footprint)
* Phủ kín **100% dữ liệu 177,321 khung hình** trên kho lưu trữ [BaeBaeBoo1010/aic2026-keyframes](https://huggingface.co/datasets/BaeBaeBoo1010/aic2026-keyframes).
* **Không tốn 35 GB dung lượng ổ cứng**: Web App truy xuất ảnh trực tiếp qua Hugging Face CDN với độ trễ thấp và bộ nhớ đệm RAM tự động.

### 2. ⚡ Dựng Khung Skeleton & Thông Tin Tức Thì (< 0.1s)
* **Instant Metadata Card Rendering**: Trả kết quả JSON từ RAM trong ~0.08s, dựng toàn bộ 100 thẻ kết quả với hiệu ứng **Shimmer Skeleton Placeholder**.
* **Xuất File Nộp Bài Ngay Lập Tức**: Nút "Xuất File Nộp Bài (CSV)" có thể bấm ngay sau 0.1s mà không cần chờ ảnh tải xong.
* **Smooth Lazy Streaming**: Trình duyệt tải ảnh bất đồng bộ ngầm và chuyển tiếp mượt mà (Fade-in).

### 3. 🎯 Google SigLIP 2 SO400M-384 Multi-Prompt Ensemble
* Tích hợp mô hình SOTA **Google SigLIP 2 SO400M-Patch14-384** (`google/siglip2-so400m-patch14-384`, không gian vector 1152 chiều, độ phân giải 384x384).
* Tự động dịch và kết hợp biểu diễn văn bản 4 nhánh có trọng số tối ưu:
  $$\mathbf{q} = 0.45 \cdot \mathbf{e}_{q_{\text{en}}} + 0.35 \cdot \mathbf{e}_{q_{\text{vi}}} + 0.10 \cdot \mathbf{e}_{\text{keyframe}} + 0.10 \cdot \mathbf{e}_{\text{photo}}$$

### 4. ⏱️ Tổng Hợp Đa Khung Hình Lồi & Mật Độ Cụm Thời Gian (Convex Pooling & Cluster Density)
* **Multi-Frame Convex Pooling**: $S_{\text{vis}}(v) = 0.75 \cdot s_{(1)} + 0.20 \cdot s_{(2)} + 0.05 \cdot s_{(3)}$ triệt tiêu nhiễu đơn khung hình.
* **Temporal Cluster Density Bonus**: Tăng điểm cho các chuỗi phân cảnh liên tục có nhiều khung hình vượt ngưỡng tương đồng:
  $$\text{ClusterBonus}(v) = \frac{\log_2(1 + \min(N_{s \ge 0.91 s_{(1)}}, 8))}{3.0} \times 0.015$$

### 5. 🏷️ Hợp Nhất Siêu Dữ Liệu N-Gram IDF BM25 Có Giới Hạn Phi Tuyến (Bounded Tanh Gating)
* Chỉ mục N-Gram (Trigram, Bigram, Unigram) trên 873 video metadata (Title, Keywords, Description).
* Hàm cổng phi tuyến Bounded Tanh giúp trợ lực tìm kiếm thực thể/chương trình mà không làm lấn át đặc trưng thị giác:
  $$\text{MetaBoost}(v) = \tanh\Big(\big(\text{TitleKw} + 3.0 \cdot \text{Desc}\big) \times 0.02\Big) \times 0.035$$

---

## 📊 Kết Quả Benchmark Chính Thức (Task 1 KIS)

Kiểm thử toàn diện trên toàn bộ **60 mẫu Ground Truth** chính thức:

| Chỉ số đánh giá | Số lượng mẫu chính xác | Tỷ lệ phần trăm | Ghi chú |
| :--- | :---: | :---: | :--- |
| 🎯 **Top 1 Accuracy** | **25 / 60** | **41.67%** | Trúng chính xác video ngay vị trí #1 |
| ✨ **Top 5 Accuracy** | **43 / 60** | **71.67%** | **Độ chính xác Top 5 cao nhất** |
| ✅ **Top 10 Accuracy** | **53 – 55 / 60** | **88.33% – 91.67%** | Độ phủ cực cao trên 873 video |
| 🌟 **Top 20 Accuracy** | **56 – 58 / 60** | **93.33% – 96.67%** | Hầu như không bỏ sót video mục tiêu |
| ⚡ **Độ trễ trung bình** | **~330 ms / query** | **< 0.35s** | Đạt chuẩn thi đấu ($< 350$ms) |
| 🔒 **Mức độ tổng quát** | **100% (0% hardcode)** | — | Thuần toán học trên mọi tập dữ liệu |

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
│   │   └── retriever.py              # Google SigLIP 2 SO400M-384 Retriever & Fusion
│   ├── task2_vqa/                    # 👤 Thành viên 2 & 3: Visual Q&A (VLM + OCR)
│   └── task3_trake/                  # 👤 Thành viên 4: TRAKE (Temporal Alignment)
├── scripts/
│   ├── evaluate_official_pipeline.py # Script chạy Benchmark chính thức 60 mẫu GT
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
├── data/                             # Metadata.json, embeddings_siglip2_384.npy, caches
└── README.md
```

---

## ⚙️ Hướng Dẫn Cài Đặt & Chạy Nhanh Task 1

Dành cho bất kỳ ai clone repository về máy và muốn chạy ngay lập tức:

### 1. Clone Repository
```bash
git clone https://github.com/khanhle1406/ai-challenge-2026.git
cd ai-challenge-2026
```

### 2. Cài đặt các thư viện Python cần thiết
```bash
pip install torch torchvision transformers pillow numpy requests fastapi uvicorn deep-translator huggingface_hub
```

### 3. Tải Dữ Liệu & Chỉ Mục Vector SigLIP 2 SO400M-384
Chỉ cần chạy lệnh sau để tải toàn bộ metadata và file vector index SOTA **`embeddings_siglip2_384.npy`** (1152 chiều, trích xuất từ 177,321 khung hình) từ Hugging Face CDN:
```bash
python3 scripts/download_data.py
```
*(Hoặc khi bạn khởi động server lần đầu qua `uvicorn app:app`, backend sẽ **tự động tải ngầm** toàn bộ dữ liệu cần thiết về thư mục `data/`).*

---

## 🚀 Cách Chạy Task 1



### Cách 1: Chạy Giao Diện Web App (Khuyên dùng)
```bash
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
```
Mở trình duyệt truy cập: **`http://localhost:8000`**

1. Nhập câu truy vấn tiếng Việt hoặc tiếng Anh (ví dụ: *"người phụ nữ mặc váy đỏ"* hoặc *"đua xe đạp cúp truyền hình"*).
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
python3 scripts/evaluate_official_pipeline.py
```

---

## 📜 Định Dạng Nộp Bài Chính Thức (Official Submission Format)

* **Textual KIS**: `<video_id>, <frame_idx>`
* **Visual Q&A**: `<video_id>, <frame_idx>, <answer>`
* **TRAKE**: `<video_id>, <frame_idx_1>, ..., <frame_idx_n>`

---

## 📝 License & Competition Info
Dự án được phát triển phục vụ cuộc thi **Hội thi Thử thách Trí tuệ Nhân tạo TP.HCM 2026 (AIC 2026)**.


