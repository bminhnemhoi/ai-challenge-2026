"""Qwen3-VL-8B-Instruct wrapper -- Member 2's VLM integration for Task 2.

Handles: model loading, prompt engineering (few-shot + optional OCR/YOLO
context from Member 3), inference, and answer parsing.
"""

from __future__ import annotations

import gc
import hashlib
import re
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

try:
    from .config import MAX_NEW_TOKENS, MODEL_ID
    from .frame_utils import resize_cap
except ImportError:  # allows running this file directly for local testing
    from config import MAX_NEW_TOKENS, MODEL_ID
    from frame_utils import resize_cap


def load_qwen_model(model_id: str = MODEL_ID) -> Tuple[AutoModelForImageTextToText, AutoProcessor]:
    """Load Qwen3-VL, auto-choosing 4-bit quantization vs bf16 based on available VRAM."""
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    use_4bit = vram_gb < 20

    if use_4bit:
        print(f"VRAM {vram_gb:.1f}GB -- using 4-bit (NF4) quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto",
        )
    else:
        print(f"VRAM {vram_gb:.1f}GB -- using bfloat16 full precision")
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
        )

    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


class QwenVLMEngine:
    """Prompt engineering + inference wrapper around a loaded Qwen3-VL model."""

    def __init__(self, model, processor, enable_cache: bool = True):
        self.model = model
        self.processor = processor
        self.enable_cache = enable_cache
        self._cache: dict = {}

    def _build_prompt(self, question: str, visual_context: Optional[str] = None) -> str:
        few_shot = (
            "Ví dụ:\n"
            "Câu hỏi: Có bao nhiêu người trong ảnh? -> Trả lời: 3\n"
            "Câu hỏi: Người đàn ông có đang đội mũ không? -> Trả lời: Có\n"
            "Câu hỏi: Chiếc áo màu gì? -> Trả lời: Màu xanh\n\n"
        )
        # visual_context: "vlm_context" string from VisualContextEngine.analyze()
        # (Member 3's OCR + object detection output) -- gives the model something
        # to check its answer against, especially useful for counting / reading text.
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
        max_new_tokens: int = MAX_NEW_TOKENS,
        video_id: Optional[str] = None,
        frame_idx: Optional[int] = None,
        visual_context: Optional[str] = None,
    ) -> str:
        if isinstance(frames, Image.Image):
            frames = [frames]
        frames = [resize_cap(f) for f in frames]  # guard against OOM if caller forgot to resize

        cache_key = None
        if self.enable_cache and video_id is not None and frame_idx is not None:
            cache_key = self._cache_key(video_id, frame_idx, question)
            if cache_key in self._cache:
                return self._cache[cache_key]

        content = [{"type": "image", "image": f} for f in frames]
        content.append({"type": "text", "text": self._build_prompt(question, visual_context)})
        messages = [{"role": "user", "content": content}]

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_prompt], images=frames, return_tensors="pt", padding=True
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )

        trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
        raw_answer = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        result = self._parse_answer(raw_answer)

        if cache_key:
            self._cache[cache_key] = result

        # release GPU memory right after each inference -- avoids cumulative/fragmented
        # memory growth across many consecutive calls (common cause of OOM otherwise)
        del inputs, output_ids, trimmed
        gc.collect()
        torch.cuda.empty_cache()

        return result
