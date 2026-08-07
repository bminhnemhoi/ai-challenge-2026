import os
import json
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Any, Optional

class TextualKISRetriever:
    """
    Search Engine for Textual Known Item Search (Task 1).
    Performs CLIP text-to-image similarity search over indexed keyframes.
    """
    def __init__(
        self,
        data_dir: str,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None
    ):
        self.data_dir = data_dir
        self.model_name = model_name
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        
        self.metadata_path = os.path.join(data_dir, "metadata.json")
        self.embeddings_path = os.path.join(data_dir, "embeddings.npy")
        
        self.metadata: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        
        self._is_loaded = False

    def load_index_and_model(self):
        """Loads metadata, vector embeddings, and initialized CLIP model."""
        if self._is_loaded:
            return

        if not os.path.exists(self.metadata_path) or not os.path.exists(self.embeddings_path):
            raise FileNotFoundError(
                f"Index files not found in '{self.data_dir}'. Please run `src/index_builder.py` first!"
            )

        print(f"Loading metadata from {self.metadata_path}...")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print(f"Loading embeddings from {self.embeddings_path}...")
        self.embeddings = np.load(self.embeddings_path) # Shape: (N, D)

        print(f"Loading CLIP model '{self.model_name}' on '{self.device}'...")
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.eval()

        self._is_loaded = True
        print(f"Retriever initialized with {len(self.metadata)} indexed keyframes.")

    def search(
        self,
        query: str,
        top_k: int = 100,
        nms_frame_gap: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Executes Textual KIS search for a natural language text query.
        
        Args:
            query: Natural language text query in English or Vietnamese.
            top_k: Number of predictions to return (max 100 for AIC competition).
            nms_frame_gap: Frame index gap threshold to suppress consecutive duplicate frames.
            
        Returns:
            List of dicts containing video_id, frame_idx, score, rel_path.
        """
        if not self._is_loaded:
            self.load_index_and_model()

        # Simple translation check / query text processing
        clean_query = query.strip()
        
        # Check if query has non-ASCII (Vietnamese) characters for translation
        en_query = None
        has_non_ascii = any(ord(char) > 127 for char in clean_query)
        if has_non_ascii:
            try:
                from deep_translator import GoogleTranslator
                en_query = GoogleTranslator(source="auto", target="en").translate(clean_query)
            except Exception:
                en_query = None

        def encode_text(t_str: str) -> np.ndarray:
            inputs = self.processor(
                text=[t_str],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
                if hasattr(outputs, "text_embeds"):
                    tf = outputs.text_embeds
                elif hasattr(outputs, "pooler_output"):
                    tf = outputs.pooler_output
                elif isinstance(outputs, torch.Tensor):
                    tf = outputs
                else:
                    tf = outputs[0]
                tf = tf / tf.norm(dim=-1, keepdim=True)
                return tf.cpu().numpy().squeeze(0)

        vi_vec = encode_text(clean_query)
        if en_query and en_query.strip() != clean_query:
            en_vec = encode_text(en_query)
            text_vec = 0.75 * en_vec + 0.25 * vi_vec
            text_vec = text_vec / np.linalg.norm(text_vec)
        else:
            text_vec = vi_vec

        # Cosine similarity via dot product (since both vectors are L2 normalized)
        similarities = np.dot(self.embeddings, text_vec) # Shape: (N,)

        # Sort indices by descending score
        sorted_indices = np.argsort(-similarities)

        results = []
        visited_frames = {} # video_id -> list of selected frame_indices for NMS

        effective_nms_gap = max(nms_frame_gap, 15) # Ensure at least 15 frames gap for scene diversity

        for idx in sorted_indices:
            score = float(similarities[idx])
            item = self.metadata[idx]
            v_id = item["video_id"]
            f_idx = item["frame_idx"]

            # Frame Non-Maximum Suppression (NMS) to avoid filling top_k with adjacent identical frames
            if effective_nms_gap > 0 and v_id in visited_frames:
                too_close = any(abs(f_idx - prev_f) < effective_nms_gap for prev_f in visited_frames[v_id])
                if too_close:
                    continue

            if v_id not in visited_frames:
                visited_frames[v_id] = []
            visited_frames[v_id].append(f_idx)

            results.append({
                "video_id": v_id,
                "frame_idx": f_idx,
                "frame_filename": item["frame_filename"],
                "score": round(score, 4),
                "rel_path": item["rel_path"],
                "abs_path": item["abs_path"]
            })

            if len(results) >= top_k:
                break

        return results
