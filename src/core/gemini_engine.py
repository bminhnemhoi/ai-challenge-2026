"""
Google Gemini Multimodal AI Engine for AIC 2026.
Uses Google GenAI SDK with gemini-3.5-flash-lite.

Features:
- Multi-Hypothesis Dual-Prompt Visual Translation (Focal Subject + Scene Environment).
- Multi-Threaded Parallel VLM Candidate Re-Ranking for Top-20 Candidates.
- On-the-fly streaming from Hugging Face CDN.
- Ultra-low token output (< 20 tokens) with 0-temperature greedy decoding.
"""

import os
import io
import re
import urllib.request
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

# python-dotenv is a convenience, not a requirement. Importing it
# unconditionally used to abort `import src` on any machine that lacked it,
# which took Task 1 and Task 3 down with it even though neither uses Gemini.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - env-dependent
    pass

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

HF_KEYFRAMES_CDN_BASE = "https://huggingface.co/datasets/BaeBaeBoo1010/aic2026-keyframes/resolve/main"

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_HTTP_SESSION = None
_IMG_CACHE = {}

def _get_http_session():
    global _HTTP_SESSION
    if _HTTP_SESSION is None:
        _HTTP_SESSION = requests.Session()
        adapter = HTTPAdapter(pool_connections=25, pool_maxsize=25, max_retries=Retry(total=2, backoff_factor=0.05))
        _HTTP_SESSION.mount("https://", adapter)
        _HTTP_SESSION.mount("http://", adapter)
        _HTTP_SESSION.headers.update({"User-Agent": "AIC2026-Engine/1.0", "Accept-Encoding": "gzip, deflate"})
    return _HTTP_SESSION

def fetch_single_image(item: Dict[str, Any], max_dim: int = 320) -> Optional[Image.Image]:
    """
    High-Speed Parallel Image Fetcher:
    - In-memory RAM cache for repeated frames.
    - Persistent HTTP keep-alive connection pool (reusing TCP/TLS sockets).
    - Auto-thumbnailing to max_dim (320px) for minimal RAM & ultra-fast MPS inference.
    """
    v_id = item.get("video_id", "")
    f_name = item.get("frame_filename")
    if not f_name and "n" in item:
        f_name = f"{item['n']:03d}.jpg"

    cache_key = f"{v_id}/{f_name}"
    if cache_key in _IMG_CACHE:
        return _IMG_CACHE[cache_key]

    local_path = item.get("path") or item.get("image_path")
    if local_path and os.path.exists(local_path):
        try:
            im = Image.open(local_path).convert("RGB")
            im.thumbnail((max_dim, max_dim))
            if len(_IMG_CACHE) < 1000:
                _IMG_CACHE[cache_key] = im
            return im
        except Exception:
            pass

    if v_id and f_name:
        cdn_url = f"{HF_KEYFRAMES_CDN_BASE}/{v_id}/{f_name}"
        try:
            session = _get_http_session()
            resp = session.get(cdn_url, timeout=2.5)
            if resp.status_code == 200:
                im = Image.open(io.BytesIO(resp.content)).convert("RGB")
                im.thumbnail((max_dim, max_dim))
                if len(_IMG_CACHE) < 1000:
                    _IMG_CACHE[cache_key] = im
                return im
        except Exception:
            return None
    return None


class GeminiAIOptimizer:
    """
    Ultra-Fast Gemini Multimodal Engine for AIC 2026.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self._client = None
        self.is_ready = False
        self._query_cache: Dict[str, Dict[str, str]] = {}

        if GENAI_AVAILABLE and self.api_key:
            try:
                self._client = genai.Client(api_key=self.api_key)
                self.is_ready = True
                print(f"⚡ Google Gemini Multimodal Engine initialized ('{self.model_name}')!")
            except Exception as e:
                print(f"⚠️ Warning: Gemini Client initialization error: {e}")

    def optimize_query(self, vietnamese_query: str) -> dict:
        """
        Translates Vietnamese query to compact natural English visual description.
        """
        q_clean = vietnamese_query.strip()
        if not self.is_ready or not self._client or not q_clean:
            return {
                "clean_en": q_clean,
                "visual_prompt": f"a photo of {q_clean}"
            }

        if q_clean in self._query_cache:
            return self._query_cache[q_clean]

        config = types.GenerateContentConfig(
            max_output_tokens=35,
            temperature=0.0
        )

        try:
            prompt = f"Translate this Vietnamese description into concise natural English for video search (output only the English sentence): {q_clean}"
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            raw = response.text.strip().strip('"').strip("'")
            clean_en = raw.split("\n")[0].strip().replace("*", "")
            if not clean_en:
                clean_en = q_clean

            result = {
                "clean_en": clean_en,
                "visual_prompt": f"a photo of {clean_en}"
            }
            self._query_cache[q_clean] = result
            return result
        except Exception:
            return {
                "clean_en": q_clean,
                "visual_prompt": f"a photo of {q_clean}"
            }

    def rerank_candidates(self, query: str, candidate_items: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Ultra-Fast Single-Shot Parallel VLM Re-Ranking for top_n candidates (default 10).
        Fetches images concurrently with fast 1.5s timeout and scores all in a single API call (< 15 tokens output).
        """
        if not self.is_ready or not self._client or not candidate_items:
            return candidate_items

        top_pool = candidate_items[:top_n]
        remaining = candidate_items[top_n:]

        # 1. Fetch images in parallel with strict timeout
        with ThreadPoolExecutor(max_workers=min(top_n, 10)) as executor:
            downloaded = list(executor.map(lambda it: fetch_single_image(it, max_dim=256), top_pool))

        images = []
        valid_indices = []
        for orig_idx, img in enumerate(downloaded):
            if img is not None:
                images.append(img)
                valid_indices.append(orig_idx)

        if not images:
            return candidate_items

        # 2. Single-shot scoring across all valid candidates (< 1.0s latency)
        config = types.GenerateContentConfig(
            system_instruction=f"Score each of the {len(images)} images from 0 to 100 on how accurately it matches the query. Output ONLY a comma-separated list of integers in order (e.g. 95, 80, 20). No other text.",
            max_output_tokens=max(20, len(images) * 4),
            temperature=0.0
        )

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=[*images, f"Query: {query}"],
                config=config
            )
            raw_text = response.text.strip()
            num_tokens = re.findall(r"\b\d+\b", raw_text)
            
            for j, num_str in enumerate(num_tokens[:len(valid_indices)]):
                orig_i = valid_indices[j]
                vlm_sc = float(num_str) / 100.0
                # Blend: 35% Base Vector + 65% Gemini Direct Visual Scoring
                top_pool[orig_i]["score"] = 0.35 * top_pool[orig_i].get("score", 0.0) + 0.65 * vlm_sc
                top_pool[orig_i]["vlm_score"] = int(num_str)

            top_pool.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return top_pool + remaining
        except Exception:
            return candidate_items

    def answer_vqa(self, context: str, question: str, candidate_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Task 2 (Visual Q&A) streaming images from Hugging Face CDN / Local Disk.
        """
        if not self.is_ready or not self._client:
            return {"answer": "Không xác định", "best_frame": None}

        with ThreadPoolExecutor(max_workers=4) as executor:
            downloaded = list(executor.map(fetch_single_image, candidate_items[:4]))

        images = [im for im in downloaded if im is not None]
        if not images:
            return {"best_image_index": 0, "answer": "Không tìm thấy ảnh keyframe"}

        config = types.GenerateContentConfig(
            system_instruction="Answer the question based on the keyframe images. Output ONLY in the format: 'Frame: <index>, Answer: <concise answer under 100 chars>'. No other text.",
            max_output_tokens=35,
            temperature=0.0
        )
        try:
            prompt = f"Event Context: {context}\nQuestion: {question}"
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=[*images, prompt],
                config=config
            )
            raw = response.text.strip()
            ans_match = re.search(r"Answer:\s*(.+)", raw, re.IGNORECASE)
            answer = ans_match.group(1).strip() if ans_match else raw
            
            frame_match = re.search(r"Frame:\s*(\d+)", raw, re.IGNORECASE)
            best_idx = int(frame_match.group(1)) if frame_match else 0
            
            return {
                "best_image_index": min(best_idx, len(candidate_items) - 1),
                "answer": answer,
                "raw_response": raw
            }
        except Exception as e:
            return {"best_image_index": 0, "answer": f"Lỗi xử lý VQA: {e}"}

    def answer_single_frame(self, image: Image.Image, question: str, context: Optional[str] = None) -> str:
        """
        Answers a question for a single image with Gemini.
        Returns a concise answer (e.g. 'Màu xanh', '5', 'Áo đỏ').
        """
        if not self.is_ready or not self._client or image is None:
            return ""

        config = types.GenerateContentConfig(
            system_instruction="You are an AI assistant for a video question answering competition. Answer the question about the provided image concisely (under 10 words). Output ONLY the direct short answer in the same language as the question. No explanations, no filler words.",
            max_output_tokens=25,
            temperature=0.0
        )
        try:
            prompt = f"Context: {context}\nQuestion: {question}" if context else f"Question: {question}"
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=[image, prompt],
                config=config
            )
            raw = response.text.strip().strip('"').strip("'").rstrip(".")
            cleaned = re.sub(r'^(Trả lời|Answer|Câu trả lời)\s*[:\-]\s*', '', raw, flags=re.IGNORECASE)
            return cleaned.strip()
        except Exception as e:
            return ""

