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
        
        # Encode text query with truncation (CLIP text encoder max_length = 77)
        inputs = self.processor(
            text=[clean_query],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            if hasattr(outputs, "pooler_output"):
                text_features = outputs.pooler_output
            elif hasattr(outputs, "text_embeds"):
                text_features = outputs.text_embeds
            elif isinstance(outputs, torch.Tensor):
                text_features = outputs
            else:
                text_features = outputs[0]
            
            # L2 normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_vec = text_features.cpu().numpy().squeeze(0) # Shape: (D,)

        # Cosine similarity via dot product (since both vectors are L2 normalized)
        similarities = np.dot(self.embeddings, text_vec) # Shape: (N,)

        # Sort indices by descending score
        sorted_indices = np.argsort(-similarities)

        results = []
        visited_frames = {} # video_id -> list of selected frame_indices for NMS

        for idx in sorted_indices:
            score = float(similarities[idx])
            item = self.metadata[idx]
            v_id = item["video_id"]
            f_idx = item["frame_idx"]

            # Frame Non-Maximum Suppression (NMS) to avoid filling top_k with adjacent identical frames
            if nms_frame_gap > 0 and v_id in visited_frames:
                too_close = any(abs(f_idx - prev_f) < nms_frame_gap for prev_f in visited_frames[v_id])
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
