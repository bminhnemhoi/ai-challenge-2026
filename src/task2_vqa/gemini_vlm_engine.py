"""Gemini API wrapper -- thay thế Qwen3-VL cho Task 2 VQA.

Dùng SDK MỚI `google-genai` (package `google.genai`) thay cho SDK cũ
`google-generativeai` đã bị Google ngừng hỗ trợ hoàn toàn (xem
https://github.com/google-gemini/deprecated-generative-ai-python).

Giữ nguyên interface: answer(frames, question, ...) -> str
để vqa_engine_VLM.py không cần thay đổi logic pipeline.

Cài thư viện:
    pip uninstall google-generativeai -y   # gỡ SDK cũ, tránh xung đột
    pip install google-genai pillow

Cần set biến môi trường:
    export GOOGLE_API_KEY="your_api_key_here"
    (tuỳ chọn) export AIC_VLM_MODEL_ID="gemini-2.5-flash"
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import List, Optional, Tuple, Union

from PIL import Image

try:
    from .config import MAX_NEW_TOKENS, MODEL_ID
    from .frame_utils import resize_cap
except ImportError:  # cho phép chạy file trực tiếp để test local
    from config import MAX_NEW_TOKENS, MODEL_ID
    from frame_utils import resize_cap

from google import genai


# Các model KHÔNG hỗ trợ nhận ảnh đầu vào (image input) dù tên có chữ "gemini" --
# nếu để lọt vào candidate_models sẽ luôn báo lỗi
# "Image input modality is not enabled" và làm mất thời gian thử vô ích.
_NON_VISION_HINTS = (
    "embedding", "aqa", "tts", "learnlm", "gemma",
    "image-generation", "imagen", "veo", "native-audio",
)


def _discover_models(client: "genai.Client") -> List[str]:
    """Tự động truy vấn danh sách model đang hoạt động từ Google API,
    loại bỏ các model rõ ràng không hỗ trợ ảnh đầu vào."""
    try:
        models = []
        for m in client.models.list():
            name = getattr(m, "name", "")
            m_id = name.replace("models/", "").strip()
            low = m_id.lower()
            if not m_id or "gemini" not in low:
                continue
            if any(hint in low for hint in _NON_VISION_HINTS):
                continue
            models.append(m_id)
        if models:
            print(f"[GeminiVLMEngine] Danh sách model AI khả dụng từ tài khoản của bạn: {models}")
            return models
    except Exception as e:
        print(f"[GeminiVLMEngine] Không thể lấy danh sách model tự động ({e})")
    return []


def load_gemini_model(model_id: str = MODEL_ID) -> Tuple[Optional["genai.Client"], str]:
    """Khởi tạo Gemini API client (SDK mới google-genai).

    Returns:
        (client, model_id) -- Nếu chưa có GOOGLE_API_KEY, client=None và engine sẽ chạy
        chế độ Mock VLM để không làm sập server, cho phép test các endpoint và UI.
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        print(
            "\n" + "=" * 60 + "\n"
            "[CẢNH BÁO TASK 2 VQA] Chưa cấu hình GOOGLE_API_KEY!\n"
            "-> Hệ thống sẽ chạy ở chế độ MOCK VLM để không bị crash server.\n"
            "-> Để trả lời bằng mô hình AI Gemini thật, hãy điền GOOGLE_API_KEY vào file .env\n"
            "   hoặc chạy: export GOOGLE_API_KEY='your_api_key_here'\n"
            "   (Lấy API key miễn phí tại: https://aistudio.google.com/apikey)\n"
            + "=" * 60 + "\n"
        )
        return None, model_id

    client = genai.Client(api_key=api_key)
    discovered = _discover_models(client)
    selected_model = model_id
    if discovered:
        # Ưu tiên các model flash/pro thế hệ mới nhất
        for pref in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-pro", "gemini-1.5-pro", "gemini-pro"]:
            for av in discovered:
                if pref == av or pref in av:
                    selected_model = av
                    break
            if selected_model != model_id:
                break
        if selected_model == model_id and discovered:
            selected_model = discovered[0]

    print(f"[GeminiVLMEngine] Đã chọn model: {selected_model}")
    return client, selected_model




class GeminiVLMEngine:
    """Prompt engineering + inference wrapper dùng Gemini API (SDK mới).

    Interface giống bản cũ để vqa_engine_VLM.py dùng được mà không cần
    sửa thêm logic: answer(frames, question, video_id=..., frame_idx=..., visual_context=...)
    """

    def __init__(self, model, processor: str = MODEL_ID, enable_cache: bool = True):
        # Lưu ý: ở đây "model" thực ra là genai.Client (SDK mới không còn
        # đối tượng GenerativeModel riêng), "processor" được tái dùng để
        # chứa model_id (string) -- xem load_gemini_model() ở trên.
        self.client = model
        self.model_id = processor or MODEL_ID
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
        ans = (raw or "").strip()
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
        """Gọi Gemini API (SDK mới) để trả lời câu hỏi dựa trên frame(s) video.

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

        if self.client is None:
            # Chế độ Mock VLM khi chưa cấu hình GOOGLE_API_KEY
            if visual_context and "Visible text:" in visual_context:
                # Trích xuất text từ OCR nếu có
                text_match = re.search(r"Visible text:\s*([^\n]+)", visual_context)
                if text_match and text_match.group(1).strip() and "None" not in text_match.group(1):
                    return text_match.group(1).split(",")[0].strip()
            return f"Mock answer for '{question}' (Chưa cấu hình GOOGLE_API_KEY)"

        prompt = self._build_prompt(question, visual_context)

        # Chuyển đổi frame ảnh sang định dạng Part chuẩn của google-genai
        from google.genai import types
        import io

        parts = []
        for f in frames:
            buf = io.BytesIO()
            f.save(buf, format="JPEG", quality=90)
            parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
        parts.append(prompt)

        # Model vision đã xác nhận hoạt động ổn định -- thử các model NÀY trước
        # (nhanh, tránh lãng phí lượt gọi vào các model tự dò được nhưng có thể
        # không hỗ trợ ảnh / bị giới hạn quyền truy cập trên tài khoản).
        known_good_vision = [
            self.model_id,  # model đã chọn lúc load_gemini_model(), thường là lựa chọn tốt nhất
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-001",
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro",
            "gemini-1.5-pro-latest",
        ]

        candidate_models: List[str] = []
        for m in known_good_vision:
            if m and m not in candidate_models:
                candidate_models.append(m)

        raw_answer = None
        last_exc = None
        tried_discovery_fallback = False

        def _try_models(model_list: List[str]) -> Optional[str]:
            nonlocal last_exc
            for m_name in model_list:
                try:
                    response = self.client.models.generate_content(
                        model=m_name,
                        contents=parts,
                    )
                    if response and response.text:
                        self.model_id = m_name  # Lưu lại model hoạt động tốt, ưu tiên dùng lại lần sau
                        print(f"[GeminiVLMEngine] Đã gọi thành công model Vision: {m_name}")
                        return response.text
                except Exception as exc:
                    last_exc = exc
                    err_str = str(exc)
                    print(
                        f"[GeminiVLMEngine] Model '{m_name}' không chạy được "
                        f"({type(exc).__name__}: {err_str[:80]}), đang thử model tiếp theo..."
                    )
                    continue
            return None

        raw_answer = _try_models(candidate_models)

        # Chỉ dò thêm danh sách model động (đã lọc bỏ model không hỗ trợ ảnh)
        # nếu TOÀN BỘ danh sách known-good ở trên đều thất bại.
        if raw_answer is None:
            tried_discovery_fallback = True
            discovered_dynamic = [m for m in _discover_models(self.client) if m not in candidate_models]
            if discovered_dynamic:
                raw_answer = _try_models(discovered_dynamic)

        if raw_answer is not None:
            result = self._parse_answer(raw_answer)
        else:
            err_msg = str(last_exc).strip() if last_exc else "Unknown error"
            print(
                f"[GeminiVLMEngine] TẤT CẢ model đều lỗi khi gọi Gemini API "
                f"({type(last_exc).__name__ if last_exc else '?'}: {err_msg}). "
                f"(đã thử cả model dò tự động: {tried_discovery_fallback}). "
                "Kiểm tra GOOGLE_API_KEY / quyền vision trên tài khoản. Dùng fallback an toàn cho answer."
            )
            # QUAN TRỌNG: không bao giờ nhét raw error message vào `answer` --
            # BTC chấm theo format <video_id>, <frame_idx>, <answer>, một answer
            # chứa "Fallback (ClientError: ...)" sẽ luôn bị chấm sai (0 điểm) và
            # còn làm lộ chi tiết lỗi hệ thống ra output nộp bài.
            result = None
            if visual_context and "Visible text:" in visual_context:
                text_match = re.search(r"Visible text:\s*([^\n]+)", visual_context)
                if text_match and text_match.group(1).strip() and "None" not in text_match.group(1):
                    result = text_match.group(1).split(",")[0].strip()
            if not result:
                # Không có OCR context để bám vào -- trả một answer hợp lệ về mặt
                # format (ngắn gọn, không phải câu lỗi) thay vì raise/crash pipeline.
                result = "unknown"

        if cache_key:
            self._cache[cache_key] = result

        return result