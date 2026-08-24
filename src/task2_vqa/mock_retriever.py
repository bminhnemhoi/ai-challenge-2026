"""
Mock retriever để TEST luồng Task 2 (VQA) độc lập, KHÔNG cần Task 1 (src/task1_kis)
đã sẵn sàng.

Cách dùng:
    - Đặt vài video mẫu vào AIC_VIDEO_DIR (ví dụ ./data/videos/L21_V001.mp4)
    - Sửa CANDIDATE bên dưới cho khớp video_id/frame_idx bạn có
    - Trong app.py, khởi tạo engine với retriever=MockRetriever() thay vì để None

Khi có Task 1 thật (src/task1_kis.TextualKISRetriever), chỉ cần XÓA việc
truyền retriever=MockRetriever() -- code sẽ tự import retriever thật.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class MockRetriever:
    """Giả lập TextualKISRetriever: luôn trả về 1 candidate cố định."""

    _is_loaded = True  # đã "loaded" sẵn, khỏi cần gọi load_index_and_model thật

    def __init__(self, candidates: Optional[List[Dict[str, Any]]] = None):
        # Sửa video_id/frame_idx này cho khớp file .mp4 bạn có trong AIC_VIDEO_DIR
        self.candidates = candidates or [
            {
                "video_id": "L21_V001",
                "frame_idx": 10,
                "frame_filename": "006.jpg",
                "score": 0.85,
                "rel_path": "L21_V001/006.jpg",
            },
        ]

    def load_index_and_model(self) -> None:
        # Không có index thật để load -- no-op.
        return None

    def search(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        return self.candidates[:top_k]
