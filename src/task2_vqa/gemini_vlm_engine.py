"""Gemini API wrapper -- thay thế Qwen3-VL cho Task 2 VQA.

Dùng Google AI Studio API (google-generativeai) thay vì load model local.
Giữ nguyên interface: answer(frames, question, ...) -> str
để vqa_engine_VLM.py không cần thay đổi logic pipeline.

Cài thư viện:
    pip install google-generativeai pillow

Cần set biến môi trường:
    export GOOGLE_API_KEY="your_api_key_here" (Linux/macOS)
    $env:GOOGLE_API_KEY="your_api_key_here" (Window)
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import List, Optional, Tuple, Union

from PIL import Image

if __package__:
    from .config import MAX_NEW_TOKENS, MODEL_ID
    from .frame_utils import resize_cap
else:  # cho phép chạy file trực tiếp để test local
    from config import MAX_NEW_TOKENS, MODEL_ID
    from frame_utils import resize_cap

import google.generativeai as genai


def load_gemini_model(model_id: str = MODEL_ID) -> Tuple[genai.GenerativeModel, None]:
    """Khởi tạo Gemini API client.

    Returns:
        (model, None) -- trả về None thay cho processor để giữ nguyên
        interface với vqa_engine_VLM.py (vốn dùng model, processor = load_qwen_model()).
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Chưa set GOOGLE_API_KEY.\n"
            "Lấy API key miễn phí tại: https://aistudio.google.com/apikey\n"
            "Sau đó chạy: export GOOGLE_API_KEY='your_key_here'"
        )
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_id)
    return model, None  # processor=None vì Gemini API tự xử lý ảnh


class GeminiVLMEngine:
    """Prompt engineering + inference wrapper dùng Gemini API.

    Interface giống QwenVLMEngine để vqa_engine_VLM.py dùng được
    mà không cần sửa thêm logic.
    """

    def __init__(self, model: genai.GenerativeModel, processor=None, enable_cache: bool = True):
        self.model = model
        self.enable_cache = enable_cache
        self._cache: dict = {}

    def _build_prompt(self, question: str, visual_context: Optional[str] = None) -> str:
        few_shot = (
            "Ví dụ:\n"
            "Câu hỏi: Có bao nhiêu người trong ảnh? -> Trả lời: 3\n"
            "Câu hỏi: Người đàn ông có đang đội mũ không? -> Trả lời: Có\n"
            "Câu hỏi: Chiếc áo màu gì? -> Trả lời: Màu xanh\n\n"
        )
        context_block = ""
        if visual_context:
            context_block = (
                "Thông tin bổ sung phát hiện được trong ảnh (từ OCR và object detection, "
                "có thể dùng để đối chiếu, KHÔNG bắt buộc đúng 100%):\n"
                f"{visual_context}\n\n"
            )
        return (
            "Bạn là hệ thống phân tích video cho một cuộc thi truy vấn sự kiện. "
            "Nhìn kỹ (các) ảnh được cung cấp — chúng là các khung hình liên tiếp trích từ cùng một video, "
            "thể hiện một khoảnh khắc/sự kiện diễn ra theo thời gian.\n"
            "Trả lời câu hỏi CHỈ bằng một từ hoặc cụm từ NGẮN GỌN nhất có thể — "
            "KHÔNG giải thích, KHÔNG lặp lại câu hỏi, KHÔNG thêm câu dẫn.\n"
            "Nếu câu hỏi bằng tiếng Việt, trả lời bằng tiếng Việt. Nếu bằng tiếng Anh, trả lời bằng tiếng Anh.\n\n"
            f"{few_shot}"
            f"{context_block}"
            f"Câu hỏi: {question}\n"
            "Trả lời:"
        )

    def _parse_answer(self, raw: str) -> str:
        ans = raw.strip()
        ans = re.sub(r'^(Trả lời|Answer|Câu trả lời)\s*[:\-]\s*', '', ans, flags=re.IGNORECASE)
        ans = ans.split("\n")[0].strip()
        return ans.rstrip(".")

    def _cache_key(self, video_id: Optional[str], frame_idx: Optional[int], question: str) -> str:
        raw = f"{video_id}|{frame_idx}|{question}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def answer(
        self,
        frames: Union[Image.Image, List[Image.Image]],
        question: str,
        max_new_tokens: int = MAX_NEW_TOKENS,  # giữ param để không phá interface cũ
        video_id: Optional[str] = None,
        frame_idx: Optional[int] = None,
        visual_context: Optional[str] = None,
    ) -> str:
        """Gọi Gemini API để trả lời câu hỏi dựa trên frame(s) video.

        Args:
            frames: 1 hoặc nhiều PIL Image (khung hình liên tiếp từ video).
            question: Câu hỏi tiếng Việt hoặc tiếng Anh.
            max_new_tokens: Không dùng với Gemini API (giữ để tương thích interface).
            video_id: Dùng để tạo cache key.
            frame_idx: Dùng để tạo cache key.
            visual_context: Context từ OCR + object detection (Member 3).

        Returns:
            Câu trả lời ngắn gọn đã được chuẩn hóa.
        """
        if isinstance(frames, Image.Image):
            frames = [frames]
        frames = [resize_cap(f) for f in frames]  # giới hạn kích thước ảnh, chống timeout

        # Kiểm tra cache
        cache_key = None
        if self.enable_cache and video_id is not None and frame_idx is not None:
            cache_key = self._cache_key(video_id, frame_idx, question)
            if cache_key in self._cache:
                return self._cache[cache_key]

        prompt = self._build_prompt(question, visual_context)

        # Gemini API nhận list [image1, image2, ..., text_prompt]
        content: list = frames + [prompt]

        response = self.model.generate_content(content)
        raw_answer = response.text
        result = self._parse_answer(raw_answer)

        if cache_key:
            self._cache[cache_key] = result

        # Không cần giải phóng GPU memory như Qwen vì Gemini chạy trên cloud
        return result
