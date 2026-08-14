import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from PIL import Image

class VLMReranker:
    """
    Level 3 Google SOTA Two-Stage VLM Re-Ranker.
    Re-evaluates Stage 1 candidate keyframes using cross-modal Vision-Language Model
    alignment to eliminate false positives and boost Top-1 Precision.
    """

    def __init__(
        self,
        model_id: str = "google/siglip2-base-patch16-224",
        device: Optional[str] = None,
        model: Optional[Any] = None,
        processor: Optional[Any] = None
    ):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model_id = model_id
        self.model = model
        self.processor = processor
        self.is_ready = (model is not None and processor is not None)

    def load_model(self):
        """Lazy load VLM re-ranker components if not already passed."""
        if self.is_ready:
            return

        try:
            print(f"🔥 Loading Level 3 VLM Re-Ranker '{self.model_id}' on '{self.device}'...")
            from transformers import AutoProcessor, AutoModel
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
            self.model.eval()
            self.is_ready = True
            print("✅ Level 3 VLM Re-Ranker initialized successfully!")
        except Exception as e:
            print(f"⚠️ VLM Re-Ranker load warning: {e}. Falling back to Stage 1 scores.")
            self.is_ready = False

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        image_base_dir: Optional[str] = None,
        alpha: float = 0.40
    ) -> List[Dict[str, Any]]:
        """
        Re-rank Stage 1 candidate keyframes.
        
        Args:
            query: User search text query.
            candidates: List of candidate dicts from Stage 1.
            image_base_dir: Optional base directory where keyframe images are stored.
            alpha: Weight for Stage 1 score (1 - alpha weight for VLM score).
            
        Returns:
            Re-ranked candidates list with updated scores.
        """
        if not candidates:
            return candidates

        self.load_model()
        if not self.is_ready or self.model is None or self.processor is None:
            return candidates

        try:
            from concurrent.futures import ThreadPoolExecutor

            import urllib.request
            import io

            def load_single_image(args):
                idx, item = args
                v_id = item.get("video_id", "")
                f_name = item.get("frame_filename", "")
                img_path = item.get("abs_path") or item.get("image_path")
                
                if not img_path or not os.path.exists(img_path):
                    if image_base_dir and v_id and f_name:
                        img_path = os.path.join(image_base_dir, v_id, f_name)

                # Check cache directory if main keyframe path not present
                if (not img_path or not os.path.exists(img_path)) and v_id and f_name:
                    parent_dir = os.path.dirname(image_base_dir) if image_base_dir else "."
                    cache_path = os.path.join(parent_dir, "cache", v_id, f_name)
                    if os.path.exists(cache_path):
                        img_path = cache_path
                    else:
                        # Fetch from public Hugging Face dataset CDN on-the-fly
                        try:
                            hf_url = f"https://huggingface.co/datasets/BaeBaeBoo1010/aic2026-keyframes/resolve/main/{v_id}/{f_name}"
                            req = urllib.request.Request(hf_url, headers={"User-Agent": "Mozilla/5.0"})
                            with urllib.request.urlopen(req, timeout=5) as resp:
                                img_data = resp.read()
                                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                                with open(cache_path, "wb") as f:
                                    f.write(img_data)
                                img_path = cache_path
                        except Exception:
                            pass

                if img_path and os.path.exists(img_path):
                    try:
                        img = Image.open(img_path).convert("RGB")
                        # Quality Inspection: Filter out motion blur & solid/black/blank frames
                        gray = np.array(img.convert("L"), dtype=np.float32)
                        std_dev = float(np.std(gray))
                        if std_dev < 12.0:  # Solid color / Black screen / Fade transition
                            return idx, None
                        
                        lap = np.abs(gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1])
                        blur_var = float(np.var(lap))
                        if blur_var < 20.0: # Heavy motion blur
                            return idx, None

                        return idx, img
                    except Exception:
                        pass
                return idx, None

            # Parallel multi-threaded keyframe disk loading
            with ThreadPoolExecutor(max_workers=8) as executor:
                loaded_results = list(executor.map(load_single_image, enumerate(candidates)))

            images = []
            valid_indices = []
            for idx, img in loaded_results:
                if img is not None:
                    images.append(img)
                    valid_indices.append(idx)

            if not images:
                return candidates

            text_prompt = f"a photo of {query}"
            inputs_txt = self.processor(text=[text_prompt], return_tensors="pt", padding="max_length", max_length=64).to(self.device)
            inputs_img = self.processor(images=images, return_tensors="pt").to(self.device)

            with torch.inference_mode():
                vision_out = self.model.vision_model(**inputs_img)
                patch_embeds = vision_out.last_hidden_state  # (B_images, 196, 768)

                text_out = self.model.text_model(**inputs_txt)
                text_tokens = text_out.last_hidden_state     # (1, 64, 768)

                mask = (inputs_txt["input_ids"] > 0).float()

                raw_scores = []
                for i in range(len(images)):
                    p_norm = patch_embeds[i] / patch_embeds[i].norm(dim=-1, keepdim=True)
                    t_norm = text_tokens / text_tokens.norm(dim=-1, keepdim=True)
                    sim_mat = torch.matmul(t_norm, p_norm.T)
                    maxsim_per_tok = sim_mat.max(dim=-1).values
                    raw_maxsim = float(((maxsim_per_tok * mask).sum(dim=-1) / mask.sum(dim=-1)).item())
                    raw_scores.append(raw_maxsim)

                # Min-Max Normalization across valid batch candidates for sharp contrast
                if len(raw_scores) > 1:
                    min_s, max_s = min(raw_scores), max(raw_scores)
                    range_s = max_s - min_s if max_s > min_s else 1.0
                    vlm_probs = [(s - min_s) / range_s for s in raw_scores]
                else:
                    vlm_probs = [1.0] if raw_scores else []

            reranked = [dict(item) for item in candidates]
            valid_set = set(valid_indices)

            # Preserve Stage 1 scores for un-cached images, only apply VLM score to valid images
            for valid_idx, vlm_score in zip(valid_indices, vlm_probs):
                stage1_score = float(reranked[valid_idx].get("score", 0.5))
                vlm_score_val = float(vlm_score)
                fused_score = alpha * stage1_score + (1.0 - alpha) * vlm_score_val
                reranked[valid_idx]["score"] = round(fused_score, 4)
                reranked[valid_idx]["vlm_score"] = round(vlm_score_val, 4)

            # Re-sort after quality demotion & VLM MaxSim Re-Ranking
            reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)
            return reranked

            reranked.sort(key=lambda x: x["score"], reverse=True)
            return reranked

        except Exception as e:
            print(f"⚠️ VLM Re-Ranking error: {e}. Returning Stage 1 results.")
            return candidates
